"""Experiment 4 - variants and the held-out networks (Table 5 of the paper; log: results/logs/04_heldout_networks.log)

E1 QSW vs classical rate analogue; E2 unnormalised jumps; E3 AND-aware weights; E4 QSW + AO* ensemble;
E5 the label-free omega average (QSW-Rank); E6 does QSW minus path search grow with network size?
For the held-out networks of the paper, point QSAW_PREV_ROOTS at the output of experiment 1 run with
QSAW_SEED0=300 QSAW_N_GRAPHS=5. The held-out log came from a lightly edited copy of this script whose
messages start with "[fresh]" instead of "[tiny]"; that copy was not kept.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qswrank import *  # noqa: F401,F403

# %% 11. TINY EXPERIMENTS: can we fix the paper's limitations? (MulVAL only, all settings fixed before the run)
#   E1  source of gain : QSW vs classical rate analogue (same couplings, H_ij^2), omega 0.1 and 0.5, Wilcoxon
#   E2  limitation 5   : unnormalised jumps (rate i->j = exploit probability, not split per node)
#   E3  limitation 1   : AND-aware weights (arc into an AND step x best-derivation probability of its other preconditions)
#   E4  limitation 1   : rank ensemble QSW + AO* (equal weights)
#   E5  limitation 3   : label-free omega average (mean of the QSW scores over all 11 omegas)
#   E6  limitation 2   : does QSW - A* grow with network size? (regression over the 40 graphs, no new runs)
TY = f"{OUT}/tiny"; os.makedirs(TY, exist_ok=True); TC = f"{TY}/cache"; os.makedirs(TC, exist_ok=True)
set_report(f"{TY}/TINY.md", "# Tiny experiments\n\n"); t_start = time.time()
print(f"[env] device={DEV} | cpu cores={NCPU}")
roots = [r for r in os.environ.get("QSAW_PREV_ROOTS", "/kaggle/input:/kaggle/working").split(":") if os.path.isdir(r)]
hits = [p for r in roots for p in glob.glob(f"{r}/**/part1_all_metrics.csv", recursive=True) if not p.startswith(OUT)]
assert hits, "previous output not found: Add Input -> Notebooks -> Your Work -> shiva"
PREV = os.path.dirname(sorted(hits)[0]); PCACHE = [c for c in glob.glob(f"{PREV}/cache_*") if glob.glob(f"{c}/graph_H*_s*.pkl")][0]
D1 = pd.read_csv(f"{PREV}/part1_all_metrics.csv"); OMS = CFG["OMEGAS"]
print(f"[tiny] previous run: {PREV}")


def classical_mix_sq(H, G, d, p0, omegas, T):
    """classical rate analogue: dp/dt = (1-w)(S - diag S^T 1) p + w (G - diag d) p with S = H o H"""
    S = H.multiply(H).tocsr(); hs = np.asarray(S.sum(axis=0)).ravel(); w = np.asarray(omegas, float)[None, :]
    P = np.repeat(np.asarray(p0, float)[:, None], len(omegas), 1); acc = np.zeros_like(P)
    f = lambda X: (1 - w) * (S @ X - hs[:, None] * X) + w * (G @ X - d[:, None] * X)
    for _ in range(int(round(T / DT))):
        k1 = f(P); k2 = f(P + 0.5 * DT * k1); k3 = f(P + 0.5 * DT * k2); k4 = f(P + DT * k3)
        acc += (DT / 6) * (P + 2 * (P + 0.5 * DT * k1) + 2 * (P + 0.5 * DT * k2) + (P + DT * k3)); P = P + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return (acc / T).T


def unnormalised_ops(n, src, dst, wgt):
    """same H; jump rate i->j = w_ij itself (a failing exploit keeps the walker where it is)"""
    H, _, _ = build_operators(n, src, dst, wgt)
    W = sp.csr_matrix((np.asarray(wgt, float), (np.asarray(src), np.asarray(dst))), shape=(n, n)); W.setdiag(0.0); W.eliminate_zeros()
    G = W.T.tocsr(); return H, G, np.asarray(G.sum(axis=0)).ravel()


def aostar_cost(g):
    """node-level cost of the cheapest AND/OR derivation (-log probability), same algorithm as aostar_scores"""
    n, typ = g["n"], g["type"]; kids = [[] for _ in range(n)]; parents = [[] for _ in range(n)]
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
                if rem[p] == 0: cost[p] = acc[p] - np.log(max(g["and_p"][p], 1e-300)); heapq.heappush(h, (cost[p], p))
            elif c < cost[p]: cost[p] = c; heapq.heappush(h, (c, p))
    return cost


def and_gated_weights(g, wg):
    """weight of arc c->r into AND step r, times P(best derivation) of r's OTHER kept preconditions (floor 1e-9)"""
    typ, n = g["type"], g["n"]; cost = aostar_cost(g)
    keep = np.flatnonzero((typ != "LEAF") | np.isin(np.arange(n), g["start"])); kept = np.zeros(n, bool); kept[keep] = True
    m = kept[g["par"]] & kept[g["chi"]]; par, chi = g["par"][m], g["chi"][m]
    assert len(par) == len(wg["w"]), "arc order differs from walk_graph"
    ccost = np.where(np.isfinite(cost[chi]), cost[chi], 700.0); tot = np.zeros(n); np.add.at(tot, par, np.where(typ[par] == "AND", ccost, 0.0))
    mult = np.where(typ[par] == "AND", np.exp(-np.clip(tot[par] - ccost, 0, 700)), 1.0)
    return np.maximum(wg["w"] * mult, 1e-9)


def rank_avg(a, b):
    hs = sorted(a); ra = stats.rankdata([a[h] for h in hs]); rb = stats.rankdata([b.get(h, NEG) for h in hs]); return dict(zip(hs, (ra + rb) / 2))


def paired(df, ref, other, metric):
    q = df[df.method == ref].set_index(["H", "seed"])[metric]; b = df[df.method == other].set_index(["H", "seed"])[metric].reindex(q.index)
    dl = (q - b).dropna().to_numpy(); p = 1.0 if np.allclose(dl, 0) else stats.wilcoxon(dl, zero_method="zsplit").pvalue
    return dict(a=ref, b=other, mean_a=q.mean(), mean_b=b.mean(), diff=dl.mean(), wins=int((dl > 1e-12).sum()), losses=int((dl < -1e-12).sum()), p=p)


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r_, j in enumerate(o): run = max(run, (len(ps) - r_) * ps[j]); adj[j] = min(1, run)
    return adj


# self-tests: gated weights align with walk_graph arcs; unnormalised ops conserve probability at w=1 on a toy graph
_r = np.random.default_rng(5); _n = 8; _s = _r.integers(0, _n, 24); _t = _r.integers(0, _n, 24); _k = _s != _t
_H, _G, _d = unnormalised_ops(_n, _s[_k], _t[_k], _r.uniform(.1, 1, _k.sum())); _r0 = np.zeros((_n, _n)); _r0[0, 0] = 1
_tr = qsw_numpy(_H, _G, _d, _r0, 1.0, 2.0); _H2, _G2, _d2 = build_operators(_n, _s[_k], _t[_k], np.ones(_k.sum()))
_e = np.abs(classical_mix_sq(_H2, _G2, _d2, np.diag(_r0), [1.0], 2.0)[0] - qsw_numpy(_H2, _G2, _d2, _r0, 1.0, 2.0)).max()
assert np.isfinite(_tr).all() and _e < 1e-10, _e
print(f"[self-test] classical analogue (w=1) = QSW (w=1): {_e:.1e}; unnormalised engine finite -> OK")

seeds = D1[["H", "seed"]].drop_duplicates().sort_values(["H", "seed"]).to_numpy(); ROWS = []; repro = []
for i, (Hh, sd) in enumerate(seeds):
    cf = f"{TC}/H{Hh}_s{sd}.pkl"
    if os.path.exists(cf):
        rows = pickle.load(open(cf, "rb")); ROWS += rows; print(f"  H{Hh}_s{sd} cached"); continue
    g = pickle.load(open(f"{PCACHE}/graph_H{Hh}_s{sd}.pkl", "rb"))["g"]; gt = pickle.load(open(f"{PCACHE}/gt_H{Hh}_s{sd}.pkl", "rb"))
    wg = walk_graph(g); T = t_rule_dir(wg); n = wg["n"]; p0 = np.zeros(n); p0[wg["start"]] = 1 / len(wg["start"]); rho0 = np.diag(p0)
    H_, G_, d_ = build_operators(n, wg["src"], wg["dst"], wg["w"]); Hn, Gn, dn = unnormalised_ops(n, wg["src"], wg["dst"], wg["w"])
    wgate = and_gated_weights(g, wg); Hg, Gg, dg_ = build_operators(n, wg["src"], wg["dst"], wgate); Hgn, Ggn, dgn = unnormalised_ops(n, wg["src"], wg["dst"], wgate)
    Q = qsw_batch(H_, G_, d_, rho0, OMS, T); sc = {}
    sc["QSW w=0.5"] = host_scores(Q[OMS.index(0.5)], wg); sc["QSW w=0.1"] = host_scores(Q[OMS.index(0.1)], wg)
    sc["QSW omega-average"] = host_scores(Q.mean(0), wg)
    C = classical_mix_sq(H_, G_, d_, p0, [0.1, 0.5], T)
    sc["Classical analogue w=0.1"] = host_scores(C[0], wg); sc["Classical analogue w=0.5"] = host_scores(C[1], wg)
    sc["QSW unnormalised jumps"] = host_scores(qsw_batch(Hn, Gn, dn, rho0, [0.5], T)[0], wg)
    sc["QSW AND-aware"] = host_scores(qsw_batch(Hg, Gg, dg_, rho0, [0.5], T)[0], wg)
    sc["QSW AND-aware + unnormalised"] = host_scores(qsw_batch(Hgn, Ggn, dgn, rho0, [0.5], T)[0], wg)
    ao = aostar_scores(g); sc["Ensemble QSW + AO*"] = rank_avg(sc["QSW w=0.5"], ao)
    rows = []
    for k in GT_KEYS:
        for m_, s_ in sc.items():
            rows.append(dict(H=int(Hh), seed=int(sd), attacker=k, method=m_, **graded_metrics(s_, gt[k], int(sd))))
        old = D1[(D1.H == Hh) & (D1.seed == sd) & (D1.view == "clean") & (D1["attacker"] == k) & (D1.method == "QSW (w=0.5)")]["NDCG@10"]
        if len(old): repro.append(abs(rows[-len(sc)]["NDCG@10"] - float(old.iloc[0])))
    pickle.dump(rows, open(cf + ".tmp", "wb")); os.replace(cf + ".tmp", cf); ROWS += rows
    if GPU: torch.cuda.empty_cache()
    print(f"  H{Hh}_s{sd} ({i+1}/{len(seeds)}) | {time.time()-t_start:.0f}s")
if repro: say(f"[reproducibility] QSW w=0.5 vs previous run: max |NDCG@10 diff| {max(repro):.1e} over {len(repro)} checks")
R = pd.DataFrame(ROWS)
prev = D1[(D1.view == "clean") & D1.method.isin(["A*", "AO*", "PageRank", "Markov (w=1)"])][["H", "seed", "attacker", "method", "NDCG@10", "Spearman"]]
ALL = pd.concat([R[["H", "seed", "attacker", "method", "NDCG@10", "Spearman"]], prev]); ALL.to_csv(f"{TY}/tiny_all.csv", index=False)

table(ALL.groupby(["method", "attacker"])[["NDCG@10", "Spearman"]].mean().unstack("attacker").round(4), "MEAN OVER 40 GRAPHS (A*, AO*, PageRank, Markov from the previous run)")

# E1: source of gain
rows = []
for k in GT_KEYS:
    for w_ in ("0.1", "0.5"):
        for mt in ("NDCG@10", "Spearman"):
            rows.append(dict(attacker=k, metric=mt, **paired(ALL[ALL.attacker == k], f"QSW w={w_}", f"Classical analogue w={w_}", mt)))
E1 = pd.DataFrame(rows); E1["p_holm"] = holm(E1.p); table(E1.set_index(["attacker", "metric", "a"]).drop(columns="b"), "E1  QSW minus classical rate analogue (same couplings), Holm over all 12 tests")

# E2-E5: each variant against QSW w=0.5, A* and AO*
VAR = ["QSW unnormalised jumps", "QSW AND-aware", "QSW AND-aware + unnormalised", "Ensemble QSW + AO*", "QSW omega-average"]
rows = []
for k in GT_KEYS:
    for v in VAR:
        for o in ("QSW w=0.5", "A*", "AO*"):
            for mt in ("NDCG@10", "Spearman"):
                rows.append(dict(attacker=k, metric=mt, **paired(ALL[ALL.attacker == k], v, o, mt)))
E2 = pd.DataFrame(rows); E2["p_holm"] = holm(E2.p)
for k in GT_KEYS:
    table(E2[E2.attacker == k].set_index(["a", "b", "metric"]).drop(columns="attacker"), f"E2-E5 [{k}] variant minus comparator (Holm over all {len(E2)} tests)")

# E6: size trend of QSW - A* on the previous run (no new compute)
rows = []
for k in GT_KEYS:
    for o in ("A*", "AO*"):
        for mt in ("NDCG@10", "Spearman"):
            c = D1[(D1.view == "clean") & (D1["attacker"] == k)]; a = c[c.method == "QSW (w=0.5)"].set_index(["H", "seed"])[mt]
            b = c[c.method == o].set_index(["H", "seed"])[mt].reindex(a.index); dl = (a - b).dropna(); h_ = dl.index.get_level_values("H").to_numpy()
            lr = stats.linregress(h_, dl.to_numpy()); rs = stats.spearmanr(h_, dl.to_numpy())
            rows.append(dict(attacker=k, vs=o, metric=mt, slope_per_25_hosts=25 * lr.slope, p_slope=lr.pvalue, spearman_size=rs.correlation, p_spearman=rs.pvalue,
                             **{f"diff_H{s}": dl[h_ == s].mean() for s in sorted(set(h_))}))
table(pd.DataFrame(rows).set_index(["attacker", "vs", "metric"]), "E6  does QSW minus exact search grow with network size? (40 graphs, previous run)")
say(f"\n[tiny] done in {time.time()-t_start:.0f}s. Outputs in {TY}")
