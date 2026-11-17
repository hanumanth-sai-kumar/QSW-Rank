"""Experiment 3 - is the gain quantum? (Table 4 and Fig. 3 of the paper; log: results/logs/03_classical_hopping.log)

(a) classical control: the same graph, start and horizon, with the coherent part replaced by classical hopping on
    the same symmetric links, mixed with the directed jumps at every omega.
(b), (c) omega chosen without hindsight / transferred between benchmarks (parts (b) and (c) also use the flow data).
Needs the output of experiment 1 (QSAW_PREV_ROOTS) and, for (b)/(c), the flow dataset.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qswrank import *  # noqa: F401,F403

# %% 8. SUPPLEMENT: is the gain really quantum?
# (a) classical control: the SAME graph, start and time, but the coherent part replaced by classical hopping on the
#     same symmetric edges (two rate laws), mixed with the directed jumps at every omega -> "classical mixture" walk
# (b) interior-optimum test without hindsight: omega chosen on OTHER sizes (MulVAL) / OTHER days (flows)
# (c) zero-label transfer: omega chosen on one benchmark, frozen, applied to another
# Needs the previous run's output attached (Add Input -> Your Work -> the Q-SAW notebook) + the flow dataset. GPU on.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a85", "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SUP = f"{OUT}/supplement"; os.makedirs(SUP, exist_ok=True)
set_report(f"{SUP}/SUPPLEMENT.md", "# Q-SAW supplement: is the gain quantum?\n\n")
OMS = CFG["OMEGAS"]; t_start = time.time()

# ---------- locate the previous run
roots = [r for r in os.environ.get("QSAW_PREV_ROOTS", "/kaggle/input:/kaggle/working").split(":") if os.path.isdir(r)]
hits = [p for r in roots for p in glob.glob(f"{r}/**/part1_all_metrics.csv", recursive=True) if not p.startswith(OUT)]
assert hits, "previous output not found: attach the Q-SAW notebook output (Add Input -> Your Work -> Notebooks)"
PREV = os.path.dirname(sorted(hits)[0])
cdirs = [c for c in glob.glob(f"{PREV}/cache_*") if glob.glob(f"{c}/graph_H*_s*.pkl")]
assert cdirs, f"no cached MulVAL graphs under {PREV}"
PCACHE = cdirs[0]
say(f"[supplement] previous run: {PREV} | cache: {PCACHE}")
D1 = pd.read_csv(f"{PREV}/part1_all_metrics.csv")
D2 = {n: pd.read_csv(f"{PREV}/part2_{n}_metrics.csv") for n in CFG["FLOW_DATASETS"] if os.path.exists(f"{PREV}/part2_{n}_metrics.csv")}
NON_BASE = lambda m: m.startswith("QW@") or m in ("QSW (w=0.5)", "CTQW (w=0)", "Markov (w=1)", "Random")


# ---------- classical mixture walk (exactly the classical analogue of every ingredient of the QSW)
def classical_mix(H, G, d, p0, omegas, T, law):
    """dp/dt = (1-w) K_sym p + w (G p - d p). K_sym = classical hopping on the edges of H with rate H_ij ('lin')
    or H_ij^2 ('sq', the strong-dephasing / Haken-Strobl rate law). w=1 is identical to the QSW at w=1."""
    S = H.copy() if law == "lin" else H.multiply(H).tocsr()
    hs = np.asarray(S.sum(axis=0)).ravel(); w = np.asarray(omegas, float)[None, :]
    P = np.repeat(np.asarray(p0, float)[:, None], len(omegas), 1); acc = np.zeros_like(P)
    f = lambda X: (1 - w) * (S @ X - hs[:, None] * X) + w * (G @ X - d[:, None] * X)
    for _ in range(int(round(T / DT))):
        k1 = f(P); k2 = f(P + 0.5 * DT * k1); k3 = f(P + 0.5 * DT * k2); k4 = f(P + DT * k3)
        acc += (DT / 6) * (P + 2 * (P + 0.5 * DT * k1) + 2 * (P + 0.5 * DT * k2) + (P + DT * k3))
        P = P + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return (acc / T).T


# self-test: classical mixture at w=1 must equal the exact QSW at w=1; at w=0 it must conserve probability
_rng = np.random.default_rng(3); _n = 9; _s = _rng.integers(0, _n, 30); _t = _rng.integers(0, _n, 30); _k = _s != _t
_H, _G, _d = build_operators(_n, _s[_k], _t[_k], _rng.uniform(0.1, 1, _k.sum()))
_r0 = np.zeros((_n, _n)); _r0[0, 0] = 1.0
_e = np.abs(classical_mix(_H, _G, _d, np.diag(_r0), [1.0], 3.0, "lin")[0] - qsw_numpy(_H, _G, _d, _r0, 1.0, 3.0)).max()
_m = classical_mix(_H, _G, _d, np.diag(_r0), [0.0, 0.5], 3.0, "sq").sum(1)
assert _e < 1e-10 and np.abs(_m - 1).max() < 1e-9, (_e, _m)
say(f"[self-test] classical mixture (w=1) vs exact QSW (w=1): {_e:.1e}; probability conserved -> OK")

ROWS, BASE = [], []      # ROWS: our walk family per unit/config; BASE: every other method from the previous run


def add_rows(bench, setting, unit, fold, fam, scores_by_w, scorer):
    for w, s_ in scores_by_w.items():
        ROWS.append(dict(bench=bench, setting=setting, unit=unit, fold=fold, family=fam, omega=float(w), **scorer(s_)))


# ---------- Part 1: MulVAL (clean + 20% vulnerabilities unseen), every omega, QSW recomputed on the missing view
seeds = D1[["H", "seed"]].drop_duplicates().sort_values(["H", "seed"]).to_numpy()
mv_name = f"missing{int(CFG['MISSING'] * 100)}"; repro = []
for i, (Hh, sd) in enumerate(seeds):
    rec = pickle.load(open(f"{PCACHE}/graph_H{Hh}_s{sd}.pkl", "rb")); gt = pickle.load(open(f"{PCACHE}/gt_H{Hh}_s{sd}.pkl", "rb"))
    g = rec["g"]; unit = f"H{Hh}_s{sd}"
    for vname, gv in (("clean", g), (mv_name, view_missing(g, CFG["MISSING"], int(sd)))):
        wg = walk_graph(gv); T = t_rule_dir(wg); H_, G_, d_ = build_operators(wg["n"], wg["src"], wg["dst"], wg["w"])
        p0 = np.zeros(wg["n"]); p0[wg["start"]] = 1 / len(wg["start"])
        fams = {f"Classical mixture ({law})": classical_mix(H_, G_, d_, p0, OMS, T, law) for law in ("lin", "sq")}
        if vname != "clean":
            fams["QSW"] = qsw_batch(H_, G_, d_, np.diag(p0), OMS, T)
        elif i % max(1, len(seeds) // 4) == 0:      # reproducibility check against the previous run
            Qc = qsw_batch(H_, G_, d_, np.diag(p0), OMS, T)
        for k in GT_KEYS:
            setting = f"{k} {vname}"; sc_ = lambda s_: graded_metrics(host_scores(s_, wg), gt[k], int(sd))
            for fam, Q in fams.items():
                add_rows("MulVAL", setting, unit, int(Hh), fam, dict(zip(OMS, Q)), sc_)
            prev = D1[(D1.H == Hh) & (D1.seed == sd) & (D1.view == vname) & (D1.attacker == k)]
            if vname == "clean":
                for _, r in prev[prev.method.str.startswith("QW@")].iterrows():
                    ROWS.append(dict(bench="MulVAL", setting=setting, unit=unit, fold=int(Hh), family="QSW", omega=float(r.method[3:]),
                                     **{m: r[m] for m in ("NDCG@5", "NDCG@10", "Top5", "Spearman", "Kendall")}))
                if i % max(1, len(seeds) // 4) == 0:
                    for j, w in enumerate(OMS):
                        old = prev[prev.method == f"QW@{w}"]["NDCG@10"]
                        if len(old): repro.append(abs(sc_(Qc[j])["NDCG@10"] - float(old.iloc[0])))
            else:
                old = prev.set_index("method")["NDCG@10"]
                for w, nm in Q_NAMES.items():
                    if nm in old.index: repro.append(abs(sc_(fams["QSW"][OMS.index(w)])["NDCG@10"] - old[nm]))
            for _, r in prev[[not NON_BASE(m) for m in prev.method]].iterrows():
                BASE.append(dict(bench="MulVAL", setting=setting, unit=unit, method=r.method,
                                 **{m: r[m] for m in ("NDCG@10", "Spearman")}))
    print(f"  MulVAL {unit} ({i+1}/{len(seeds)}) | {time.time()-t_start:.0f}s")
say(f"[reproducibility] |NDCG@10 now - previous run| over {len(repro)} checks: max {max(repro) if repro else float('nan'):.2e}")

# ---------- Part 2: flows, every omega for the classical mixtures; QSW values from the previous run
for name, Dp in D2.items():
    if not os.path.isdir(os.path.join(CFG["FLOW_ROOT"], FLOW_SPEC[name]["subdir"])):
        say(f"[flow] {name}: data folder not found, skipped"); continue
    insts, _ = flow_dataset(name); have = set(Dp.uid)
    insts = [x for x in insts if x["uid"] in have]
    for j, inst in enumerate(insts):
        H_, G_, d_ = build_operators(inst["n"], inst["src"], inst["dst"], inst["w"]); T = flow_t_rule(inst)
        p0 = np.zeros(inst["n"]); p0[inst["start"]] = inst["q"]; setting = f"N={inst['N']}"
        sc_ = lambda s_: rank_metrics(s_, inst["cand"], inst["gt"], inst["tieperm"])
        for law in ("lin", "sq"):
            add_rows(name, setting, inst["uid"], inst["day"], f"Classical mixture ({law})", dict(zip(OMS, classical_mix(H_, G_, d_, p0, OMS, T, law))), sc_)
        prev = Dp[Dp.uid == inst["uid"]]
        for _, r in prev.iterrows():
            if r.method.startswith("QW@"):
                ROWS.append(dict(bench=name, setting=setting, unit=inst["uid"], fold=inst["day"], family="QSW", omega=float(r.method[3:]),
                                 **{m: r[m] for m in FLOW_M}))
            elif not NON_BASE(r.method):
                BASE.append(dict(bench=name, setting=setting, unit=inst["uid"], method=r.method, MAP=r["MAP"], MRR=r["MRR"]))
    print(f"  [{name}] classical mixtures on {len(insts)} instances | {time.time()-t_start:.0f}s")

R = pd.DataFrame(ROWS); B = pd.DataFrame(BASE)
R.to_csv(f"{SUP}/supp_walk_family.csv", index=False); B.to_csv(f"{SUP}/supp_baselines.csv", index=False)
METRIC = {"MulVAL": "NDCG@10", "CIC2017": "MAP", "CSE2018": "MAP"}
SETTINGS = [(b, s) for b in ["MulVAL"] + list(D2) for s in sorted(R[R.bench == b].setting.unique())]


def paired(a, b):
    """a, b: Series on the same units -> mean diff, bootstrap CI, wins/losses, Wilcoxon p"""
    j = a.index.intersection(b.index); dlt = (a[j] - b[j]).to_numpy(); dlt = dlt[~np.isnan(dlt)]
    p = 1.0 if len(dlt) == 0 or np.allclose(dlt, 0) else stats.wilcoxon(dlt, zero_method="zsplit").pvalue
    bs = np.random.default_rng(1).choice(dlt, (2000, len(dlt))).mean(1) if len(dlt) else np.zeros(1)
    return dict(diff=dlt.mean() if len(dlt) else np.nan, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                wins=int((dlt > 1e-12).sum()), losses=int((dlt < -1e-12).sum()), p=p)


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r_, i in enumerate(o): run = max(run, (len(ps) - r_) * ps[i]); adj[i] = min(1, run)
    return adj


def loo_select(sub, fams, metric):
    """held-out score per unit: config (family, omega) chosen on the other folds only"""
    sub = sub[sub.family.isin(fams)].copy(); sub["cfg"] = sub.family + "@" + sub.omega.round(2).astype(str)
    piv = sub.pivot_table(index=["unit", "fold"], columns="cfg", values=metric); folds = piv.index.get_level_values("fold")
    out, chosen = {}, {}
    for f in pd.unique(folds):
        tr = piv[folds != f]
        if len(tr) == 0: continue
        best = tr.mean().idxmax(); chosen[f] = best
        for (u, _), v in piv[folds == f][best].items(): out[u] = v
    return pd.Series(out), chosen


# ---------- (A) curves: QSW vs classical mixtures at every omega
curves = R.groupby(["bench", "setting", "family", "omega"])[[c_ for c_ in ("NDCG@10", "Spearman", "MAP") if c_ in R.columns]].mean().reset_index()
# (MAP exists only when the flow dataset is attached; parts (b) and (c) then also cover the flow benchmarks)
curves.to_csv(f"{SUP}/supp_curves.csv", index=False)
for b, s in SETTINGS:
    mt = METRIC[b]; c = curves[(curves.bench == b) & (curves.setting == s)].pivot_table(index="family", columns="omega", values=mt)
    table(c, f"(A) {b} {s}: mean {mt} vs omega (0 = coherent / symmetric end, 1 = classical directed walk, identical for all rows)", 4)
    if b == "MulVAL":
        c2 = curves[(curves.bench == b) & (curves.setting == s)].pivot_table(index="family", columns="omega", values="Spearman")
        table(c2, f"(A) {b} {s}: mean Spearman vs omega", 4)

# ---------- (B) the decisive table: omega chosen without hindsight (other sizes / other days), held-out scores
CL = ["Classical mixture (lin)", "Classical mixture (sq)"]; rowsB, chosen_log = [], []
for b, s in SETTINGS:
    for mt in ([METRIC[b], "Spearman"] if b == "MulVAL" else [METRIC[b]]):
        sub = R[(R.bench == b) & (R.setting == s)]
        q_loo, q_ch = loo_select(sub, ["QSW"], mt); c_loo, c_ch = loo_select(sub, CL, mt)
        chosen_log.append(dict(bench=b, setting=s, metric=mt, QSW_choice=sorted(set(q_ch.values())), classical_choice=sorted(set(c_ch.values()))))
        ser = lambda fam, w: sub[(sub.family == fam) & (np.isclose(sub.omega, w))].set_index("unit")[mt]
        bb = B[(B.bench == b) & (B.setting == s)]
        best_base = bb.groupby("method")[mt].mean().idxmax() if len(bb) else None
        comps = {"pure coherent QSW (w=0)": ser("QSW", 0.0), "classical walk (w=1)": ser("QSW", 1.0),
                 "QSW a-priori (w=0.5)": ser("QSW", 0.5), "best classical mixture (same protocol)": c_loo}
        if best_base: comps[f"best other method: {best_base}"] = bb[bb.method == best_base].set_index("unit")[mt]
        res = [dict(bench=b, setting=s, metric=mt, QSW_heldout=q_loo.mean(), vs=k, other=v.reindex(q_loo.index).mean(), **paired(q_loo, v))
               for k, v in comps.items()]
        for r_, ph in zip(res, holm([r_["p"] for r_ in res])): r_["p_holm"] = ph
        rowsB += res
TB = pd.DataFrame(rowsB); TB.to_csv(f"{SUP}/supp_heldout_omega.csv", index=False)
table(TB.set_index(["bench", "setting", "metric", "vs"])[["QSW_heldout", "other", "diff", "ci_lo", "ci_hi", "wins", "losses", "p", "p_holm"]],
      "(B) QSW with omega chosen on OTHER sizes/days (no hindsight) minus each alternative; positive = QSW better")
table(pd.DataFrame(chosen_log).set_index(["bench", "setting", "metric"]), "(B) configurations chosen per held-out fold")

# ---------- (C) zero-label transfer: omega fixed on a source benchmark, applied unchanged to a target
def best_cfg(mask, fam_list, mt_of):
    sub = R[mask & R.family.isin(fam_list)].copy(); sub["mt"] = [r[mt_of[b]] for b, r in zip(sub.bench, sub.to_dict("records"))]
    m = sub.groupby(["family", "omega"])["mt"].mean(); return m.idxmax()

rowsC = []
sources = {"MulVAL (all attackers, clean)": (R.bench == "MulVAL") & R.setting.str.endswith("clean")}
for n in D2: sources[n] = R.bench == n
for src, mask in sources.items():
    qf, qw = best_cfg(mask, ["QSW"], METRIC); cf, cw = best_cfg(mask, CL, METRIC)
    for b, s in SETTINGS:
        if b == "MulVAL" or src == b or (src.startswith("MulVAL") and b == "MulVAL"): continue
        mt = METRIC[b]; sub = R[(R.bench == b) & (R.setting == s)]
        q = sub[(sub.family == "QSW") & np.isclose(sub.omega, qw)].set_index("unit")[mt]
        c = sub[(sub.family == cf) & np.isclose(sub.omega, cw)].set_index("unit")[mt]
        bb = B[(B.bench == b) & (B.setting == s)]
        for k, v in {"classical mixture (same transfer)": c, "classical walk (w=1)": sub[(sub.family == "QSW") & np.isclose(sub.omega, 1.0)].set_index("unit")[mt],
                     **{f"{m}": bb[bb.method == m].set_index("unit")[mt] for m in ("A*", "PageRank", "QAGC-DTQW", "QAGC-CTQW") if m in set(bb.method)}}.items():
            rowsC.append(dict(source=src, omega_from_source=qw, target=f"{b} {s}", QSW=q.mean(), vs=k, other=v.reindex(q.index).mean(), **paired(q, v)))
TC = pd.DataFrame(rowsC); TC.to_csv(f"{SUP}/supp_transfer.csv", index=False)
if len(TC):
    table(TC.set_index(["source", "omega_from_source", "target", "vs"])[["QSW", "other", "diff", "ci_lo", "ci_hi", "wins", "losses", "p"]],
          "(C) zero-label transfer: omega fixed on the source benchmark, applied to the target (positive = QSW better)")

# ---------- (D) one-line verdicts for the paper
say("\n## VERDICT (held-out omega, primary metric)")
for b, s in SETTINGS:
    t = TB[(TB.bench == b) & (TB.setting == s) & (TB.metric == METRIC[b])].set_index("vs")
    cmix = t.loc["best classical mixture (same protocol)"]; c1 = t.loc["classical walk (w=1)"]; c0 = t.loc["pure coherent QSW (w=0)"]
    tag = lambda r: "WIN" if r["diff"] > 0 and r["p_holm"] < 0.05 else ("LOSS" if r["diff"] < 0 and r["p_holm"] < 0.05 else "tie")
    say(f"  {b:8s} {s:16s} QSW(held-out)={t.QSW_heldout.iloc[0]:.4f} | vs classical mixture {cmix['diff']:+.4f} {tag(cmix)} | "
        f"vs classical walk {c1['diff']:+.4f} {tag(c1)} | vs pure coherent {c0['diff']:+.4f} {tag(c0)}")

# ---------- figure: decoherence curve of the QSW vs the classical mixtures, best other method dashed
panels = [(b, s) for b, s in SETTINGS if not s.startswith("noise")]
nc = 4; nr = int(np.ceil(len(panels) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.5 * nr), squeeze=False)
for a, (b, s) in zip(axes.ravel(), panels):
    mt = METRIC[b]; c = curves[(curves.bench == b) & (curves.setting == s)]
    for i, fam in enumerate(["QSW", "Classical mixture (lin)", "Classical mixture (sq)"]):
        y = c[c.family == fam].set_index("omega")[mt]
        a.plot(y.index, y.values, marker="os^"[i], color=SERIES[i], lw=2 if fam == "QSW" else 1.5, ms=3.5, label=fam)
    bb = B[(B.bench == b) & (B.setting == s)]
    if len(bb):
        m_ = bb.groupby("method")[mt].mean(); a.axhline(m_.max(), color="#8a8a85", ls="--", lw=1)
        a.text(1.0, m_.max(), f" {m_.idxmax()}", fontsize=6, color="#5f5f5a", va="bottom", ha="right")
    a.set_title(f"{b} {s.replace('Binf', 'unlimited').replace(mv_name, '20% vulns unseen')}", fontsize=8, loc="left")
    a.set_xlabel("omega", fontsize=7); a.set_ylabel(f"mean {mt}", fontsize=7); a.tick_params(labelsize=7)
for a in axes.ravel()[len(panels):]: a.axis("off")
h_, l_ = axes[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(); fig.savefig(f"{SUP}/fig8_quantum_vs_classical_mixture.png"); fig.savefig(f"{SUP}/fig8_quantum_vs_classical_mixture.pdf"); plt.close(fig)
say(f"[figure] {SUP}/fig8_quantum_vs_classical_mixture.png")
say(f"\n[supplement] done in {time.time()-t_start:.0f}s. Outputs in {SUP}: SUPPLEMENT.md, supp_*.csv, fig8_*.png|pdf")
