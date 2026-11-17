"""Ground truth: Monte-Carlo budget-limited attackers over MulVAL AND/OR semantics (20,000 runs per network)."""
from .config import *  # noqa: F401,F403


def mc_attacker(g, S, seed, budget=None, chunk=2000, probs=None):
    """budget=None: every enabled exploit is attempted once (least fixpoint, = unlimited attacker).
    budget=B: the attacker makes B attempts, each time picking uniformly among currently enabled,
    not-yet-tried rules. Exploit outcomes ~ Bernoulli(rule probability x CVSS exploit probability).
    Returns {host: P(code execution on host)}."""
    and_p, leaf_p = probs if probs is not None else (g["and_p"], g["leaf_p"])
    n, typ = g["n"], g["type"]; rng = np.random.default_rng(seed)
    C = sp.csr_matrix((np.ones(len(g["par"]), np.float32), (g["chi"], g["par"])), shape=(n, n))
    nchild = np.bincount(g["par"], minlength=n)
    A = np.flatnonzero(typ == "AND"); O = np.flatnonzero(typ == "OR"); L = np.flatnonzero(typ == "LEAF")
    hosts = list(g["host_nodes"]); hits = np.zeros(len(hosts)); done = 0
    cnt_of = lambda X: np.asarray(X.astype(np.float32) @ C)
    while done < S:
        s = min(chunk, S - done)
        X = np.zeros((s, n), bool); X[:, L] = rng.random((s, len(L))) < leaf_p[L]
        fire = rng.random((s, len(A))) < and_p[A]
        if budget is None:
            for _ in range(2 * n + 2):
                cnt = cnt_of(X)
                newA = (cnt[:, A] == nchild[A]) & fire & ~X[:, A]; newO = (cnt[:, O] > 0) & ~X[:, O]
                if not newA.any() and not newO.any(): break
                X[:, A] |= newA; X[:, O] |= newO
        else:
            tried = np.zeros((s, len(A)), bool)
            for _ in range(n):
                newO = (cnt_of(X)[:, O] > 0) & ~X[:, O]
                if not newO.any(): break
                X[:, O] |= newO
            for _ in range(budget):
                avail = (cnt_of(X)[:, A] == nchild[A]) & ~tried & ~X[:, A]
                r = np.flatnonzero(avail.any(1))
                if len(r) == 0: break
                ch = np.where(avail[r], rng.random((len(r), len(A))), -1.0).argmax(1)
                tried[r, ch] = True; ok = fire[r, ch]
                X[r[ok], A[ch[ok]]] = True
                for _ in range(n):
                    newO = (cnt_of(X)[:, O] > 0) & ~X[:, O]
                    if not newO.any(): break
                    X[:, O] |= newO
        for j, h in enumerate(hosts):
            hits[j] += X[:, g["host_nodes"][h]].any(1).sum()
        done += s
    return {h: hits[j] / S for j, h in enumerate(hosts)}


__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]
