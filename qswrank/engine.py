"""Quantum stochastic walk engine: exact Lindblad evolution in O(N^2) memory (Section 6.2 of the paper).

d rho/dt = -i(1-w)[H, rho] + w( diag(J p) - 1/2 (Lambda rho + rho Lambda) ),  p = diag(rho)
build_operators() makes H, J and the exit rates; qsw_numpy() integrates one rate on the CPU (complex128);
qsw_batch() integrates several rates at once with PyTorch (complex64) when a GPU is available.
"""
from .config import *  # noqa: F401,F403


def build_operators(n, src, dst, wgt):
    """H = normalised symmetric adjacency (coherent part); G[dst,src] = w/s_out(src) (jump rates); d = exit rates"""
    W = sp.csr_matrix((np.asarray(wgt, float), (np.asarray(src), np.asarray(dst))), shape=(n, n))
    W.setdiag(0.0); W.eliminate_zeros()
    s = np.asarray(W.sum(axis=1)).ravel(); inv = np.zeros(n); inv[s > 0] = 1 / s[s > 0]
    G = (sp.diags(inv) @ W).T.tocsr(); d = np.asarray(G.sum(axis=0)).ravel()
    A = (W + W.T).tocsr(); deg = np.asarray(A.sum(axis=1)).ravel()
    di = np.zeros(n); di[deg > 0] = 1 / np.sqrt(deg[deg > 0])
    return (sp.diags(di) @ A @ sp.diags(di)).tocsr(), G, d


def _rhs(rho, H, G, d, w):
    out = np.zeros_like(rho)
    if w < 1:
        Hr = H @ rho; out += (-1j * (1 - w)) * (Hr - Hr.conj().T)
    if w > 0:
        p = rho.real.diagonal()
        out -= (0.5 * w) * (d[:, None] * rho + rho * d[None, :])
        out[np.diag_indices_from(out)] += w * (G @ p)
    return out


def qsw_numpy(H, G, d, rho0, w, T):
    """RK4, complex128; returns the time-averaged (Cesaro) populations over [0, T]"""
    rho = rho0.astype(complex).copy(); acc = np.zeros(rho.shape[0])
    for _ in range(int(round(T / DT))):
        k1 = _rhs(rho, H, G, d, w); s2 = rho + 0.5 * DT * k1
        k2 = _rhs(s2, H, G, d, w); s3 = rho + 0.5 * DT * k2
        k3 = _rhs(s3, H, G, d, w); s4 = rho + DT * k3
        k4 = _rhs(s4, H, G, d, w)
        acc += (DT / 6) * (rho.real.diagonal() + 2 * s2.real.diagonal() + 2 * s3.real.diagonal() + s4.real.diagonal())
        rho = rho + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4); rho = 0.5 * (rho + rho.conj().T)
    return acc / T


def _tsparse(M):
    c = M.tocoo()
    idx = torch.tensor(np.vstack([c.row, c.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.tensor(c.data.astype(np.float32)), c.shape).coalesce().to(DEV)


def qsw_batch(H, G, d, rho0, omegas, T):
    """all decoherence rates at once; torch complex64 on GPU, numpy fallback otherwise -> (len(omegas), n)"""
    if not USE_TORCH:
        return np.stack([qsw_numpy(H, G, d, rho0, w, T) for w in omegas])
    n = H.shape[0]; B = len(omegas); c64 = torch.complex64
    Hs, Gs = _tsparse(H), _tsparse(G)
    dd = torch.tensor(d, dtype=torch.float32, device=DEV); w = torch.tensor(omegas, dtype=torch.float32, device=DEV)
    rho = torch.tensor(rho0, dtype=c64, device=DEV).unsqueeze(0).repeat(B, 1, 1)
    coh = (-1j * (1 - w)).to(c64)[:, None, None]; dis = (0.5 * w)[:, None, None]

    def hmul(r):
        X = torch.cat([r.real, r.imag], 0).permute(1, 0, 2).reshape(n, -1)
        Y = torch.sparse.mm(Hs, X).reshape(n, 2 * B, n).permute(1, 0, 2)
        return torch.complex(Y[:B].contiguous(), Y[B:].contiguous())

    def rhs(r):
        Hr = hmul(r)
        o = coh * (Hr - Hr.mH) - dis * (dd[None, :, None] * r + r * dd[None, None, :])
        p = r.diagonal(dim1=-2, dim2=-1).real
        o.diagonal(dim1=-2, dim2=-1).add_((w[:, None] * torch.sparse.mm(Gs, p.T.contiguous()).T).to(c64))
        return o

    dg = lambda r: r.diagonal(dim1=-2, dim2=-1).real
    acc = torch.zeros(B, n, device=DEV)
    for _ in range(int(round(T / DT))):
        k1 = rhs(rho); s2 = rho + 0.5 * DT * k1
        k2 = rhs(s2); s3 = rho + 0.5 * DT * k2
        k3 = rhs(s3); s4 = rho + DT * k3
        k4 = rhs(s4)
        acc += (DT / 6) * (dg(rho) + 2 * dg(s2) + 2 * dg(s3) + dg(s4))
        rho = rho + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4); rho = 0.5 * (rho + rho.mH)
    out = (acc / T).cpu().numpy().astype(float)
    del rho, k1, k2, k3, k4, s2, s3, s4
    if GPU: torch.cuda.empty_cache()
    return out


def engine_self_test():
    rng = np.random.default_rng(0); worst = dict(rhs=0.0, rk4=0.0, markov=0.0, batch=0.0)
    for _ in range(6):
        n = int(rng.integers(4, 9)); m = 3 * n; s = rng.integers(0, n, m); t = rng.integers(0, n, m); k = s != t
        s, t, x = s[k], t[k], rng.uniform(0.1, 1.0, k.sum())
        H, G, d = build_operators(n, s, t, x); Hd, Gd = H.toarray(), G.toarray()
        rho0 = np.zeros((n, n), complex); rho0[0, 0] = 0.6; rho0[1, 1] = 0.4
        for w in (0.0, 0.5, 1.0):
            I = np.eye(n); L = -1j * (1 - w) * (np.kron(I, Hd) - np.kron(Hd.T, I))
            for i in range(n):
                for j in range(n):
                    if Gd[i, j] > 0:
                        Lk = np.zeros((n, n)); Lk[i, j] = np.sqrt(Gd[i, j]); LdL = Lk.T @ Lk
                        L += w * (np.kron(Lk, Lk) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I))
            X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); R = X @ X.conj().T; R /= np.trace(R)
            worst["rhs"] = max(worst["rhs"], np.abs(_rhs(R, H, G, d, w) - (L @ R.reshape(-1, order="F")).reshape(n, n, order="F")).max())
            ts = np.linspace(0, 3.0, 3001); E = sla.expm(L * (ts[1] - ts[0])); v = rho0.reshape(-1, order="F"); ps = []
            for _t in ts:
                ps.append(np.real(np.diag(v.reshape(n, n, order="F")))); v = E @ v
            exact = np.trapezoid(np.array(ps), ts, axis=0) / 3.0
            worst["rk4"] = max(worst["rk4"], np.abs(qsw_numpy(H, G, d, rho0, w, 3.0) - exact).max())
        p = sla.expm((Gd - np.diag(d)) * 0.01)
        cls = np.zeros(n); pt = np.real(np.diag(rho0)).copy()
        for _t in range(300):
            cls += 0.01 * 0.5 * pt; pt = p @ pt; cls += 0.01 * 0.5 * pt
        worst["markov"] = max(worst["markov"], np.abs(qsw_numpy(H, G, d, rho0, 1.0, 3.0) - cls / 3.0).max())
        if USE_TORCH:
            bt = qsw_batch(H, G, d, rho0, [0.0, 0.5, 1.0], 3.0)
            for i, w in enumerate((0.0, 0.5, 1.0)):
                worst["batch"] = max(worst["batch"], np.abs(bt[i] - qsw_numpy(H, G, d, rho0, w, 3.0)).max())
    say(f"[self-test] O(N^2) Lindblad RHS vs full N^2xN^2 superoperator: {worst['rhs']:.1e} (must be < 1e-10)")
    say(f"[self-test] RK4 Cesaro populations vs exact expm:             {worst['rk4']:.1e} (must be < 1e-4)")
    say(f"[self-test] omega=1 vs classical master equation:              {worst['markov']:.1e} (must be < 1e-3)")
    if USE_TORCH:
        say(f"[self-test] torch complex64 batch vs numpy complex128:          {worst['batch']:.1e} (must be < 1e-4)")
    assert worst["rhs"] < 1e-10 and worst["rk4"] < 1e-4 and worst["markov"] < 1e-3 and worst["batch"] < 1e-4
    say("[self-test] PASSED")


__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]
