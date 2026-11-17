"""Walk graph, QSW-Rank scoring helpers, baselines and ranking metrics.

Every method receives the same input: the attack graph, the step success probabilities and the attacker start.
Baselines: BFS, DFS, A*, AO* (Knuth), PageRank, Markov walk (QSW at w=1), pure CTQW (QSW at w=0) and a pure DTQW
(weighted Grover coin, flip-flop shift). The qagc_* functions are our own port of the walks described in the
QAGC paper; they were used in exploratory runs and are not reported in the QSW-Rank paper.
"""
from .config import *  # noqa: F401,F403
from .engine import *  # noqa: F401,F403
from .ground_truth import *  # noqa: F401,F403


def walk_graph(g):
    """attack-flow graph for walks/search: child -> parent, LEAF nodes dropped except the attacker start;
    an edge into an AND step carries its success probability (rule p x CVSS p of its vulnerability leaves)"""
    typ, n = g["type"], g["n"]; q = g["and_p"].copy()
    for p, c in zip(g["par"], g["chi"]):
        if typ[p] == "AND" and typ[c] == "LEAF": q[p] *= g["leaf_p"][c]
    keep = np.flatnonzero((typ != "LEAF") | np.isin(np.arange(n), g["start"]))
    new = -np.ones(n, int); new[keep] = np.arange(len(keep))
    m = (new[g["par"]] >= 0) & (new[g["chi"]] >= 0)
    par, chi = g["par"][m], g["chi"][m]
    w = np.where(typ[par] == "AND", q[par], 1.0)
    return dict(n=len(keep), src=new[chi], dst=new[par], w=w.astype(float), start=[int(new[s]) for s in g["start"]],
                q=np.full(len(g["start"]), 1 / len(g["start"])),
                hosts={h: [int(new[i]) for i in idx if new[i] >= 0] for h, idx in g["host_nodes"].items()})


def host_scores(node_score, wg):
    return {h: float(max(node_score[i] for i in idx)) if idx else NEG for h, idx in wg["hosts"].items()}


def t_rule_dir(wg):
    """label-free time scale: mean directed hop distance from the attacker over reachable nodes"""
    adj = [[] for _ in range(wg["n"])]
    for a, b in zip(wg["src"], wg["dst"]): adj[a].append(b)
    dist = np.full(wg["n"], -1); q = deque(wg["start"])
    for s in wg["start"]: dist[s] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0: dist[v] = dist[u] + 1; q.append(v)
    r = dist[dist > 0]
    return max(DT, round(max(float(r.mean()) if len(r) else 1.0, 1.0) / DT) * DT)


def order_scores(wg, dfs, undirected=False):
    adj = [set() for _ in range(wg["n"])]
    for a, b in zip(wg["src"], wg["dst"]):
        adj[a].add(b)
        if undirected: adj[b].add(a)
    adj = [sorted(x) for x in adj]
    starts = [wg["start"][i] for i in np.argsort(-np.asarray(wg["q"]), kind="stable")]
    sc = np.full(wg["n"], NEG); seen = np.zeros(wg["n"], bool); k = 0
    if dfs:
        for s0 in starts:
            st = [s0]
            while st:
                u = st.pop()
                if seen[u]: continue
                seen[u] = True; sc[u] = -k; k += 1
                st += [v for v in reversed(adj[u]) if not seen[v]]
    else:
        dq = deque()
        for s0 in starts: seen[s0] = True; dq.append(s0)
        while dq:
            u = dq.popleft(); sc[u] = -k; k += 1
            for v in adj[u]:
                if not seen[v]: seen[v] = True; dq.append(v)
    return sc


def astar_scores(wg, undirected=False, normalise=False):
    """A* with no goal set (h = 0): most probable path from the start(s); score = -best path cost.
    cost = -log(w) (attack graphs) or -log(w / strength) (flow graphs); start cost -log q"""
    nb = [dict() for _ in range(wg["n"])]
    for a, b, x in zip(wg["src"], wg["dst"], wg["w"]):
        nb[a][b] = nb[a].get(b, 0) + x
        if undirected: nb[b][a] = nb[b].get(a, 0) + x
    strength = np.array([sum(d.values()) for d in nb])
    dist = np.full(wg["n"], np.inf); h = []
    for s0, q0 in zip(wg["start"], wg["q"]):
        c0 = -np.log(max(q0, 1e-300)); dist[s0] = min(dist[s0], c0); heapq.heappush(h, (dist[s0], s0))
    while h:
        c, u = heapq.heappop(h)
        if c > dist[u]: continue
        for v, x in nb[u].items():
            nc = c - np.log(max(x / strength[u] if normalise else x, 1e-300))
            if nc < dist[v] - 1e-15: dist[v] = nc; heapq.heappush(h, (nc, v))
    return np.where(np.isfinite(dist), -dist, NEG)


def aostar_scores(g):
    """exact minimal-cost AND/OR solution graph for every node (Knuth's generalised Dijkstra = all-nodes AO*):
    OR = min over children, AND = -log p_rule + sum of children, vulnerability leaf = -log p_cvss"""
    n, typ = g["n"], g["type"]
    kids = [[] for _ in range(n)]; parents = [[] for _ in range(n)]
    for p, c in zip(g["par"], g["chi"]): kids[p].append(c); parents[c].append(p)
    cost = np.full(n, np.inf); done = np.zeros(n, bool); rem = np.array([len(k) for k in kids]); acc = np.zeros(n); h = []
    for i in range(n):
        if typ[i] == "LEAF": cost[i] = -np.log(max(g["leaf_p"][i], 1e-300)); heapq.heappush(h, (cost[i], i))
    while h:
        c, u = heapq.heappop(h)
        if done[u] or c > cost[u]: continue
        done[u] = True
        for p in parents[u]:
            if done[p]: continue
            if typ[p] == "AND":
                rem[p] -= 1; acc[p] += c
                if rem[p] == 0:
                    cost[p] = acc[p] - np.log(max(g["and_p"][p], 1e-300)); heapq.heappush(h, (cost[p], p))
            elif c < cost[p]:
                cost[p] = c; heapq.heappush(h, (c, p))
    sc = np.where(np.isfinite(cost), -cost, NEG)
    return {hh: float(max(sc[i] for i in idx)) for hh, idx in g["host_nodes"].items()}


def pagerank_scores(wg, alpha=0.85):
    n = wg["n"]; Wm = sp.csr_matrix((wg["w"], (wg["src"], wg["dst"])), shape=(n, n))
    s = np.asarray(Wm.sum(axis=1)).ravel(); inv = np.zeros(n); inv[s > 0] = 1 / s[s > 0]
    PT = (sp.diags(inv) @ Wm).T.tocsr(); v = np.zeros(n); v[wg["start"]] = wg["q"]; x = v.copy(); dang = s == 0
    for _ in range(1000):
        xn = alpha * (PT @ x) + (alpha * x[dang].sum() + 1 - alpha) * v
        if np.abs(xn - x).sum() < 1e-12: x = xn; break
        x = xn
    return x


def assetrank_style_scores(g, d=0.85):
    """dependency PageRank in the spirit of AssetRank (Sawilla & Ou, 2008), implemented from its description:
    rank flows from attacker goals to preconditions; AND passes full rank to each precondition, OR splits it"""
    n, typ = g["n"], g["type"]; outdeg = np.bincount(g["par"], minlength=n)
    wt = np.where(typ[g["par"]] == "AND", 1.0, 1.0 / np.maximum(outdeg[g["par"]], 1))
    M = sp.csr_matrix((wt, (g["chi"], g["par"])), shape=(n, n))
    e = np.zeros(n); goals = [i for idx in g["host_nodes"].values() for i in idx]; e[goals] = 1 / max(len(goals), 1)
    x = e.copy()
    for _ in range(500):
        xn = d * (M @ x) + (1 - d) * e; xn /= xn.sum()
        if np.abs(xn - x).sum() < 1e-12: x = xn; break
        x = xn
    return {h: float(max(x[i] for i in idx)) for h, idx in g["host_nodes"].items()}


def qagc_struct(wg):
    """arc space + spectrum shared by the QAGC walks (paper: weighted Grover coin, flip-flop shift, H = gamma*A)"""
    n = wg["n"]
    A_ = sp.csr_matrix((np.r_[wg["w"], wg["w"]], (np.r_[wg["src"], wg["dst"]], np.r_[wg["dst"], wg["src"]])), shape=(n, n))
    A_.sum_duplicates(); Ad = A_.toarray(); A_ = A_.tocoo(); au, av, aw = A_.row, A_.col, A_.data
    s = np.bincount(au, weights=aw, minlength=n); q = np.sqrt(aw / s[au])
    key = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(au, av))}
    rev = np.array([key[(int(v), int(u))] for u, v in zip(au, av)], int)
    lam, V = np.linalg.eigh(Ad)
    psi0 = np.zeros(n, complex); psi0[wg["start"]] = np.sqrt(wg["q"])
    return dict(n=n, au=au, q=q, rev=rev, has=s > 0, lam=lam, V=V, psi0=psi0)


def _dtqw(S, psi, m):
    n, au, q = S["n"], S["au"], S["q"]; a = psi[au] * q
    for _ in range(m):
        pr = np.bincount(au, weights=(q * a).real, minlength=n) + 1j * np.bincount(au, weights=(q * a).imag, minlength=n)
        a = (2 * q * pr[au] - a)[S["rev"]]
    P = np.bincount(au, weights=np.abs(a) ** 2, minlength=n)
    pr = np.bincount(au, weights=(q * a).real, minlength=n) + 1j * np.bincount(au, weights=(q * a).imag, minlength=n)
    psi = np.where(S["has"], np.sqrt(P) * np.exp(1j * np.angle(pr)), psi)
    return psi / np.linalg.norm(psi)


def _ctqw(S, psi, gamma, t):
    psi = S["V"] @ (np.exp(-1j * gamma * S["lam"] * t) * (S["V"].conj().T @ psi))
    return psi / np.linalg.norm(psi)


def qagc_hqw_scores(S, gamma=0.05, t0=1.0, m0=1, K=5):
    """QAGC Algorithm 3 (hybrid DTQW/CTQW), our reading of the paper: K = 5 cycles, m_k = m0*k, t_k = t0*k"""
    psi = S["psi0"].copy()
    for k in range(1, K + 1):
        psi = _ctqw(S, _dtqw(S, psi, m0 * k), gamma, t0 * k)
    return np.abs(psi) ** 2


def qagc_dtqw_scores(S, m=15):
    """QAGC DTQW component alone, with the hybrid's total number of coin steps (1+2+3+4+5 = 15)"""
    return np.abs(_dtqw(S, S["psi0"].copy(), m)) ** 2


def qagc_ctqw_scores(S, gamma=0.05, t=15.0):
    """QAGC CTQW component alone (H = gamma*A, unnormalised), with the hybrid's total evolution time"""
    return np.abs(_ctqw(S, S["psi0"].copy(), gamma, t)) ** 2


def qagc_repo_scores(wg, walk_steps=100):
    """port of the AQAGC repository's qagc_baseline.py (start at node index 0, 1/sqrt(out-degree), renormalise)"""
    n = wg["n"]; succ = [set() for _ in range(n)]
    for u, v in zip(wg["src"], wg["dst"]): succ[u].add(v)
    rows, cols, vals = [], [], []
    for u in range(n):
        if not succ[u]: rows.append(u); cols.append(u); vals.append(1.0); continue
        for v in succ[u]: rows.append(v); cols.append(u); vals.append(1 / np.sqrt(len(succ[u])))
    op = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)); st = np.zeros(n); st[0] = 1.0
    for _ in range(walk_steps):
        st = op @ st; nr = np.linalg.norm(st)
        if nr: st /= nr
    return st ** 2


# ---- QAGC reference-protocol classical baselines (ports of the AQAGC repository, stage06_benchmarking):
#      no attacker location; every node is a search source, as in the repository code
def _unweighted(wg):
    A_ = sp.csr_matrix((np.ones(len(wg["src"])), (wg["src"], wg["dst"])), shape=(wg["n"], wg["n"]))
    A_.sum_duplicates(); A_.data[:] = 1.0; return A_


def ref_reach_scores(wg):
    """bfs_baseline.py / astar_baseline.py: a search from EVERY node, +1 to each node it visits
    (A* there has no goal and a zero heuristic, so it visits the same nodes) = number of nodes that can reach v"""
    from scipy.sparse.csgraph import shortest_path
    D = shortest_path(_unweighted(wg), directed=True, unweighted=True)
    return np.isfinite(D).sum(axis=0).astype(float)


def ref_dfs_scores(wg, L=6):
    """dfs_baseline.py counts every simple path ending at v from every start node (exponential; it does not finish
    on our graphs). Depth-limited relaxation used here: number of directed walks of length <= L ending at v"""
    A_ = _unweighted(wg).T.tocsr(); x = np.ones(wg["n"]); tot = np.ones(wg["n"])
    for _ in range(L):
        x = A_ @ x; tot += x
    return np.log1p(tot)


def ref_markov_scores(wg, max_iter=1000, tol=1e-8):
    """markov_baseline.py: row-normalised weighted transitions, self-loop on dead ends, power iteration from uniform"""
    n = wg["n"]; W = sp.csr_matrix((wg["w"], (wg["src"], wg["dst"])), shape=(n, n)); W.sum_duplicates()
    s = np.asarray(W.sum(axis=1)).ravel(); dead = s == 0
    W = W + sp.csr_matrix((np.ones(dead.sum()), (np.flatnonzero(dead), np.flatnonzero(dead))), shape=(n, n))
    s = np.asarray(W.sum(axis=1)).ravel(); P = (sp.diags(1 / s) @ W).T.tocsr(); p = np.ones(n) / n
    for _ in range(max_iter):
        pn = P @ p
        if np.linalg.norm(pn - p) < tol: p = pn; break
        p = pn
    return p


def graded_metrics(sc, gt, seed):
    hs = sorted(gt); g = np.array([gt[h] for h in hs]); s = np.array([sc.get(h, NEG) for h in hs])
    tie = np.random.default_rng(seed).permutation(len(hs)); order = np.lexsort((tie, -s)); out = {}
    for k in (5, 10):
        kk = min(k, len(hs)); disc = 1 / np.log2(np.arange(2, kk + 2))
        out[f"NDCG@{k}"] = float((g[order[:kk]] * disc).sum() / max((np.sort(g)[::-1][:kk] * disc).sum(), 1e-12))
    kk = min(5, len(hs)); topg = set(np.lexsort((np.arange(len(hs)), -g))[:kk]); out["Top5"] = len(topg & set(order[:kk])) / kk
    fin = s > NEG / 2; sr = np.where(fin, s, (s[fin].min() - 1) if fin.any() else 0)
    rho = stats.spearmanr(sr, g).correlation; tau = stats.kendalltau(sr, g).correlation
    out["Spearman"] = 0.0 if np.isnan(rho) else float(rho); out["Kendall"] = 0.0 if np.isnan(tau) else float(tau)
    return out


SUMMARY = []


def headline(df, metric, groups, key_cols, setting):
    """QSW against the strongest member (by mean) of each baseline family; paired Wilcoxon over instances"""
    means = df.groupby("method")[metric].mean(); qm = means["QSW (w=0.5)"]
    q = df[df.method == "QSW (w=0.5)"].set_index(key_cols)[metric]; rows = []
    for gname, ms in groups.items():
        ms = [m for m in ms if m in means.index]
        if not ms: continue
        best = means[ms].idxmax(); b = df[df.method == best].set_index(key_cols)[metric].reindex(q.index)
        diff = (q - b).to_numpy(); diff = diff[~np.isnan(diff)]
        p = 1.0 if len(diff) == 0 or np.allclose(diff, 0) else stats.wilcoxon(diff, zero_method="zsplit").pvalue
        rows.append(dict(family=gname, strongest=best, family_best=means[best], QSW=qm, QSW_minus_best=qm - means[best],
                         wins=int((diff > 1e-12).sum()), losses=int((diff < -1e-12).sum()), p=p))
        SUMMARY.append(dict(setting=setting, metric=metric, family=gname, strongest=best, QSW=qm, best=means[best],
                            diff=qm - means[best], wins=int((diff > 1e-12).sum()), losses=int((diff < -1e-12).sum()), p=p))
    rank = int((means.drop([m for m in means.index if m.startswith("QW@")], errors="ignore") > qm + 1e-12).sum()) + 1
    SUMMARY.append(dict(setting=setting, metric=metric, family="(QSW rank among all methods)", strongest="", QSW=qm,
                        best=means.drop([m for m in means.index if m.startswith("QW@")], errors="ignore").max(), diff=np.nan,
                        wins=rank, losses=len([m for m in means.index if not m.startswith("QW@")]), p=np.nan))
    return pd.DataFrame(rows).set_index("family")


def subgraph(g, keep):
    keep_idx = np.flatnonzero(keep); new = -np.ones(g["n"], int); new[keep_idx] = np.arange(len(keep_idx))
    m = keep[g["par"]] & keep[g["chi"]]; g2 = dict(g)
    g2.update(n=len(keep_idx), label=[g["label"][i] for i in keep_idx], type=g["type"][keep_idx],
              par=new[g["par"][m]], chi=new[g["chi"][m]], and_p=g["and_p"][keep_idx], leaf_p=g["leaf_p"][keep_idx],
              vuln=g["vuln"][keep_idx], start=[int(new[s]) for s in g["start"]],
              host_nodes={h: [int(new[i]) for i in idx if keep[i]] for h, idx in g["host_nodes"].items()})
    g2["host_nodes"] = {h: v for h, v in g2["host_nodes"].items() if v}
    return g2


def view_noise(g, sigma, seed):
    """the defender's CVSS-derived exploit probabilities are uncertain: multiplicative log-normal noise"""
    rng = np.random.default_rng(seed); g2 = dict(g); lp = g["leaf_p"].copy()
    lp[g["vuln"]] = np.clip(lp[g["vuln"]] * np.exp(rng.normal(0, sigma, g["vuln"].sum())), 1e-4, 1.0)
    g2["leaf_p"] = lp; return g2


def view_missing(g, frac, seed):
    """the defender's scan misses a fraction of the vulnerabilities: those leaves and the steps using them vanish"""
    rng = np.random.default_rng(seed); vi = np.flatnonzero(g["vuln"])
    drop = rng.choice(vi, int(round(frac * len(vi))), replace=False) if len(vi) else np.array([], int)
    keep = np.ones(g["n"], bool); keep[drop] = False
    dset = np.zeros(g["n"], bool); dset[drop] = True
    keep[g["par"][dset[g["chi"]] & (g["type"][g["par"]] == "AND")]] = False
    return subgraph(g, keep)


Q_NAMES = {0.5: "QSW (w=0.5)", 0.0: "CTQW (w=0)", 1.0: "Markov (w=1)"}


def methods_on(gv, omegas, seed):
    wg = walk_graph(gv); T = t_rule_dir(wg)
    H, G, d = build_operators(wg["n"], wg["src"], wg["dst"], wg["w"])
    rho0 = np.zeros((wg["n"], wg["n"])); rho0[wg["start"], wg["start"]] = 1 / len(wg["start"])
    t0 = time.perf_counter(); Q = qsw_batch(H, G, d, rho0, omegas, T); tq = time.perf_counter() - t0
    sc = {f"QW@{w}": host_scores(Q[i], wg) for i, w in enumerate(omegas)}
    for w, nm in Q_NAMES.items():
        if w in omegas: sc[nm] = sc[f"QW@{w}"]
    tm = {"quantum batch": tq / len(omegas)}
    classical = {
        "BFS": lambda: host_scores(order_scores(wg, False), wg),
        "DFS": lambda: host_scores(order_scores(wg, True), wg),
        "A*": lambda: host_scores(astar_scores(wg), wg),
        "AO*": lambda: aostar_scores(gv),
        "PageRank": lambda: host_scores(pagerank_scores(wg), wg),
        "AssetRank-style": lambda: assetrank_style_scores(gv),
        **{f"MC-{S}": (lambda S=S: mc_attacker(gv, S, seed=seed * 31 + S)) for S in CFG["MC_BASE"]},
        "BFS-ref": lambda: host_scores(ref_reach_scores(wg), wg),
        "DFS-ref": lambda: host_scores(ref_dfs_scores(wg), wg),
        "A*-ref": lambda: host_scores(ref_reach_scores(wg), wg),
        "Markov-ref": lambda: host_scores(ref_markov_scores(wg), wg),
        "QAGC-repo (ref. impl.)": lambda: host_scores(qagc_repo_scores(wg), wg),
    }
    t0 = time.perf_counter(); S_ = qagc_struct(wg); t_struct = time.perf_counter() - t0
    for nm, fn in {"QAGC-DTQW": lambda: qagc_dtqw_scores(S_), "QAGC-CTQW": lambda: qagc_ctqw_scores(S_),
                   "QAGC-HQW (Alg.3)": lambda: qagc_hqw_scores(S_)}.items():
        t0 = time.perf_counter(); sc[nm] = host_scores(fn(), wg); tm[nm] = time.perf_counter() - t0 + t_struct
    for nm, fn in classical.items():
        t0 = time.perf_counter(); sc[nm] = fn(); tm[nm] = time.perf_counter() - t0
    return sc, tm, T, wg["n"], (H, G, d, rho0)


__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]
