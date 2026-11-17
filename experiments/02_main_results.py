"""Experiment 2 - main results of the paper (Tables 2 and 3, unlimited-attacker column of Table 5, cost figures).

Re-scores the 40 networks of experiment 1 with the final method set:
  QSW-Rank (time-averaged populations averaged over the 11 omegas), QSW at the fixed rate omega = 0.5,
  pure CTQW (omega = 0), Markov walk (omega = 1), pure DTQW (best step count from {1, 2, 5, 10, 15, 20},
  chosen on the ground truth, which favours this baseline), A*, AO*, BFS, DFS and PageRank.
Metrics: NDCG@10, NDCG@20 and Spearman's r_s, for the attackers B10, B30 and Binf; paired Wilcoxon tests with
Holm's correction per attacker x metric.

Needs the output of experiment 1: set QSAW_PREV_ROOTS to the folder that holds part1_all_metrics.csv and the
cache_* folder, and use a different QSAW_OUT. On Kaggle, attach the experiment-1 notebook output instead.

This script was rebuilt from the cell that produced results/logs/02_main_results.log. That cell also scored our
own port of the QAGC walks (CTQW and hybrid HQW on the grid of the QAGC paper, plus earlier default settings).
The paper does not report them, but they were part of each Holm family, which makes the paper's tests more
conservative. With QSAW_FULL_FAMILY=1 (default) they are computed again and kept in the families only, so the
p-values can be compared with the log; QSAW_FULL_FAMILY=0 drops them.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qswrank import *  # noqa: F401,F403

MR = f"{OUT}/main_results"; os.makedirs(MR, exist_ok=True); MC = f"{MR}/cache"; os.makedirs(MC, exist_ok=True)
set_report(f"{MR}/MAIN_RESULTS.md", "# QSW-Rank main results (40 MulVAL networks)\n\n"); t_start = time.time()
OMS = CFG["OMEGAS"]; DTQW_STEPS = [1, 2, 5, 10, 15, 20]
FULL_FAMILY = os.environ.get("QSAW_FULL_FAMILY", "1") == "1"
GAMMAS = [0.01, 0.05, 0.1]; TIMES = [0.5, 1.0, 2.0, 3.0]          # grid given in the QAGC paper
print(f"[env] device={DEV} | cpu cores={NCPU}")

roots = [r for r in os.environ.get("QSAW_PREV_ROOTS", "/kaggle/input").split(":") if os.path.isdir(r)]
hits = [p for r in roots for p in glob.glob(f"{r}/**/part1_all_metrics.csv", recursive=True) if not p.startswith(OUT)]
assert hits, "experiment-1 output not found: set QSAW_PREV_ROOTS (or attach the experiment-1 notebook output)"
PREV = os.path.dirname(sorted(hits)[0])
PCACHE = [c for c in glob.glob(f"{PREV}/cache_*") if glob.glob(f"{c}/graph_H*_s*.pkl")][0]
SEEDS = pd.read_csv(f"{PREV}/part1_all_metrics.csv")[["H", "seed"]].drop_duplicates()
say(f"[main] experiment-1 output: {PREV} | {len(SEEDS)} networks")


def hqw_grid(S, gamma, K=5):
    """our port of QAGC Algorithm 3 on the paper's grid: K = 5 cycles, t_k from 0.5 to 3.0, m_k = k"""
    psi = S["psi0"].copy()
    for k, tk in enumerate(np.linspace(0.5, 3.0, K), start=1):
        psi = _ctqw(S, _dtqw(S, psi, k), gamma, tk)
    return np.abs(psi) ** 2


def ndcg_k(sc, gt, seed, k):
    hs = sorted(gt); g = np.array([gt[h] for h in hs]); s = np.array([sc.get(h, NEG) for h in hs])
    tie = np.random.default_rng(seed).permutation(len(hs)); order = np.lexsort((tie, -s)); kk = min(k, len(hs))
    disc = 1 / np.log2(np.arange(2, kk + 2))
    return float((g[order[:kk]] * disc).sum() / max((np.sort(g)[::-1][:kk] * disc).sum(), 1e-12))


ROWS = []
for i, (Hh, sd) in enumerate(SEEDS.sort_values(["H", "seed"]).to_numpy()):
    cf = f"{MC}/H{Hh}_s{sd}.pkl"
    if os.path.exists(cf): ROWS += pickle.load(open(cf, "rb")); continue
    g = pickle.load(open(f"{PCACHE}/graph_H{Hh}_s{sd}.pkl", "rb"))["g"]; gt = pickle.load(open(f"{PCACHE}/gt_H{Hh}_s{sd}.pkl", "rb"))
    wg = walk_graph(g); T = t_rule_dir(wg); n = wg["n"]
    H_, G_, d_ = build_operators(n, wg["src"], wg["dst"], wg["w"])
    rho0 = np.zeros((n, n)); rho0[wg["start"], wg["start"]] = 1 / len(wg["start"])
    t0 = time.perf_counter(); Q = qsw_batch(H_, G_, d_, rho0, OMS, T); tq = time.perf_counter() - t0
    sc = {"QSW-Rank (omega average)": host_scores(Q.mean(0), wg), "QSW (w=0.5)": host_scores(Q[OMS.index(0.5)], wg),
          "Pure CTQW (w=0)": host_scores(Q[OMS.index(0.0)], wg), "Markov walk (w=1)": host_scores(Q[OMS.index(1.0)], wg)}
    sc["A*"] = host_scores(astar_scores(wg), wg); sc["AO*"] = aostar_scores(g)
    sc["BFS"] = host_scores(order_scores(wg, False), wg); sc["DFS"] = host_scores(order_scores(wg, True), wg)
    sc["PageRank"] = host_scores(pagerank_scores(wg), wg)
    S_ = qagc_struct(wg)                                   # arc space for the weighted-Grover-coin DTQW
    for m_ in DTQW_STEPS:
        sc[f"Pure DTQW m={m_}"] = host_scores(np.abs(_dtqw(S_, S_["psi0"].copy(), m_)) ** 2, wg)
    if FULL_FAMILY:                                        # comparators kept only for the logged Holm families
        sc["QAGC-CTQW port (default setting)"] = host_scores(qagc_ctqw_scores(S_), wg)
        sc["QAGC-HQW port (default setting)"] = host_scores(qagc_hqw_scores(S_), wg)
        sc["QAGC-DTQW port (default setting)"] = host_scores(qagc_dtqw_scores(S_), wg)
        for gm in GAMMAS:
            for t_ in TIMES: sc[f"QAGC-CTQW port g={gm} t={t_}"] = host_scores(np.abs(_ctqw(S_, S_["psi0"].copy(), gm, t_)) ** 2, wg)
            sc[f"QAGC-HQW port g={gm}"] = host_scores(hqw_grid(S_, gm), wg)
    rows = []
    for k in GT_KEYS:
        for m_, s_ in sc.items():
            gm_ = graded_metrics(s_, gt[k], int(sd))
            rows.append(dict(H=int(Hh), seed=int(sd), n=n, attacker=k, method=m_, **{"NDCG@10": gm_["NDCG@10"],
                        "NDCG@20": ndcg_k(s_, gt[k], int(sd), 20), "Spearman": gm_["Spearman"]}))
    rows.append(dict(H=int(Hh), seed=int(sd), n=n, attacker="time", method="QSW s per omega", **{"NDCG@10": tq / len(OMS)}))
    save(rows, cf); ROWS += rows
    if GPU: torch.cuda.empty_cache()
    print(f"  H{Hh}_s{sd}: {n} walk nodes | QSW {tq/len(OMS):.1f}s per omega | {time.time()-t_start:.0f}s")
R = pd.DataFrame(ROWS); R.to_csv(f"{MR}/main_all.csv", index=False); R = R[R.attacker != "time"]

# grid baselines: best setting per attacker and metric, chosen on the ground truth (favours the baseline)
GRIDS = [("Pure DTQW (best steps)", "Pure DTQW m=")]
if FULL_FAMILY: GRIDS += [("QAGC-CTQW port (paper grid, best)", "QAGC-CTQW port g="), ("QAGC-HQW port (paper grid, best)", "QAGC-HQW port g=")]
add, chosen = [], []
for fam, pat in GRIDS:
    for k, c in R[R.method.str.startswith(pat)].groupby("attacker"):
        for mt in ("NDCG@10", "NDCG@20", "Spearman"):
            best = c.groupby("method")[mt].mean().idxmax(); b = c[c.method == best].copy()
            b["method"] = fam; b[[x for x in ("NDCG@10", "NDCG@20", "Spearman") if x != mt]] = np.nan
            add.append(b); chosen.append(dict(family=fam, attacker=k, metric=mt, chosen=best))
R = pd.concat([R] + add)
table(pd.DataFrame(chosen).set_index(["family", "attacker", "metric"]), "Grid baselines: setting chosen per attacker and metric")

MAIN = ["QSW-Rank (omega average)", "Pure CTQW (w=0)", "Pure DTQW (best steps)", "Markov walk (w=1)", "PageRank",
        "A*", "AO*", "BFS", "DFS", "QSW (w=0.5)"]
EXTRA = ["QAGC-DTQW port (default setting)", "QAGC-CTQW port (paper grid, best)", "QAGC-HQW port (paper grid, best)",
         "QAGC-CTQW port (default setting)", "QAGC-HQW port (default setting)"] if FULL_FAMILY else []
FAMILY = MAIN + EXTRA                                      # Holm family; EXTRA is never printed in the main tables
c = R[R.method.isin(FAMILY)]
table((100 * c.groupby(["method", "attacker"])[["NDCG@10", "NDCG@20", "Spearman"]].mean()).unstack("attacker").reindex(MAIN).round(1),
      "MEAN (%) over the 40 networks")
table((100 * c[c.attacker == "B10"].groupby(["method", "H"])["NDCG@10"].mean()).unstack("H").reindex(MAIN).round(1),
      "B10 NDCG@10 (%) by number of hosts")


def paired(ref):
    out = []
    for k in GT_KEYS:
        for mt in ("NDCG@10", "NDCG@20", "Spearman"):
            a = c[(c.attacker == k) & (c.method == ref)].dropna(subset=[mt]).set_index(["H", "seed"])[mt]
            for o in [m for m in FAMILY if m != ref]:
                b = c[(c.attacker == k) & (c.method == o)].dropna(subset=[mt]).set_index(["H", "seed"])[mt].reindex(a.index)
                dl = (a - b).dropna().to_numpy()
                p = 1.0 if len(dl) < 2 or np.allclose(dl, 0) else stats.wilcoxon(dl, zero_method="zsplit").pvalue
                out.append(dict(attacker=k, metric=mt, vs=o, diff_pts=100 * dl.mean() if len(dl) else np.nan,
                                wins=int((dl > 1e-12).sum()), losses=int((dl < -1e-12).sum()), p=p))
    out = pd.DataFrame(out)
    for (k, mt), idx in out.groupby(["attacker", "metric"]).groups.items():
        ps = out.loc[idx, "p"].to_numpy(); o_ = np.argsort(ps); adj = np.empty(len(ps)); run = 0
        for r_, j in enumerate(o_): run = max(run, (len(ps) - r_) * ps[j]); adj[j] = min(1, run)
        out.loc[idx, "p_holm"] = adj
    return out


PT = paired("QSW-Rank (omega average)")
table(PT[PT.vs.isin(MAIN)].set_index(["attacker", "metric", "vs"]).round(4),
      f"QSW-Rank minus each method (points; Wilcoxon + Holm per attacker x metric over {len(FAMILY) - 1} comparisons)")
tt = pd.DataFrame(ROWS); tt = tt[tt.attacker == "time"]
table(tt.groupby("H").agg(walk_nodes=("n", "mean"), qsw_s_per_omega=("NDCG@10", "mean")).round(2), "QSW cost per network")
say(f"\n[main] done in {time.time()-t_start:.0f}s. Outputs in {MR}")
