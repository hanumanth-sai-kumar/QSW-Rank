"""Experiment 1 - main MulVAL benchmark (Part 1 of the Kaggle notebook).

Generates 4 sizes x 10 networks (seeds from 200), builds their MulVAL attack graphs, computes the ground truth
for three attackers (B=10, B=30, unlimited; 20,000 runs each) and runs every method. Writes
part1_all_metrics.csv, the per-network caches (graph_H*_s*.pkl, gt_H*_s*.pkl) and RESULTS.md into $QSAW_OUT.
Held-out networks of the paper: run again with QSAW_SEED0=300 QSAW_N_GRAPHS=5 and a different QSAW_OUT.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qswrank import *  # noqa: F401,F403

set_report(f"{OUT}/RESULTS.md", "# Q-SAW results\n\n")
engine_self_test()


def bkey(b): return "Binf" if b is None else f"B{b}"


GT_KEYS = [bkey(b) for b in CFG["BUDGETS"]]
GROUPS = {"QAGC quantum walks": ["QAGC-DTQW", "QAGC-CTQW", "QAGC-HQW (Alg.3)", "QAGC-repo (ref. impl.)"],
          "QAGC reference classical baselines": ["BFS-ref", "DFS-ref", "A*-ref", "Markov-ref"],
          "Information-matched classical baselines": ["BFS", "DFS", "A*", "AO*", "PageRank", "AssetRank-style"]
                                                     + [f"MC-{S}" for S in CFG["MC_BASE"]],
          "Ablation (same model, quantum part switched)": ["CTQW (w=0)", "Markov (w=1)"]}
MAIN_METHODS = ["QSW (w=0.5)", "CTQW (w=0)", "Markov (w=1)"] + GROUPS["QAGC quantum walks"] + \
               GROUPS["QAGC reference classical baselines"] + GROUPS["Information-matched classical baselines"] + ["Random"]
P1 = None

if CFG["RUN_MULVAL"]:
    say("\n## Part 1: MulVAL multi-stage attack graphs")
    if "run_mulval_override" not in globals():
        say("[mulval]", ensure_mulval())
        XSB_BIN = os.path.dirname(glob.glob(f"{W_}/XSB/**/bin/xsb", recursive=True)[0])
        ENV = dict(os.environ, MULVALROOT=f"{W_}/mulval", PATH=f"{XSB_BIN}:{W_}/mulval/bin:{W_}/mulval/utils:" + os.environ["PATH"])
        jc = sh("readlink -f $(which javac)").stdout.strip()
        if jc: ENV["JAVA_HOME"] = os.path.dirname(os.path.dirname(jc))
        RULES = rule_probs(f"{W_}/mulval/kb/interaction_rules.P")
        import qswrank.mulval as _mulval
        _mulval.ENV, _mulval.RULES = ENV, RULES          # read inside run_mulval() and build_graph()
    else:
        run_mulval = run_mulval_override; gen_network = gen_network_override
    say(f"[plan] sizes {CFG['SIZES']} x first {CFG['N_GRAPHS']} valid seeds from {CFG['SEED0']} | ground truth "
        f"{CFG['GT_SAMPLES']} runs x attacker budgets {GT_KEYS} | QSW omega={CFG['MAIN_OMEGA']} fixed a priori")

    # ---- phase A: generate graphs (MulVAL runs in parallel on all CPU cores)
    def gen_one(H, seed):
        cf = f"{CACHE}/graph_H{H}_s{seed}.pkl"
        if os.path.exists(cf): return pickle.load(open(cf, "rb"))
        facts, meta = gen_network(H, seed)
        try:
            V, E, t = run_mulval(facts, f"{OUT}/mulval_runs/H{H}_s{seed}", CFG["MULVAL_TIMEOUT"])
            g = build_graph(V, E, meta); ok = len(g["host_nodes"]) >= 3
            rec = dict(H=H, seed=seed, ok=ok, g=g if ok else None, t_mulval=t,
                       why="" if ok else "fewer than 3 reachable hosts", steps=depth_stats(g) if ok else None)
        except Exception as e:
            rec = dict(H=H, seed=seed, ok=False, g=None, t_mulval=np.nan, why=str(e)[:160], steps=None)
        save(rec, cf); return rec

    valid = {H: [] for H in CFG["SIZES"]}; nxt = {H: CFG["SEED0"] for H in CFG["SIZES"]}; t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, NCPU)) as ex:
        first = True
        while True:
            jobs = []
            for H in sorted(CFG["SIZES"], reverse=True):
                need = CFG["N_GRAPHS"] - len(valid[H])
                for _ in range(max(0, need + (2 if first and need > 0 else 0))):
                    if nxt[H] < CFG["SEED0"] + CFG["MAX_TRIES"]: jobs.append((H, nxt[H])); nxt[H] += 1
            first = False
            if not jobs: break
            for rec in ex.map(lambda a: gen_one(*a), jobs):
                tag = "ok" if rec["ok"] else f"skip ({rec['why']})"
                print(f"  graph H={rec['H']} seed={rec['seed']}: {tag} | MulVAL {rec['t_mulval']:.0f}s | {time.time()-t0:.0f}s")
                if rec["ok"] and len(valid[rec["H"]]) < CFG["N_GRAPHS"]: valid[rec["H"]].append(rec)
            for H in valid: valid[H].sort(key=lambda r: r["seed"])
    RECS = [r for H in CFG["SIZES"] for r in valid[H]]
    gstat = pd.DataFrame([dict(hosts=r["H"], seed=r["seed"], nodes=r["g"]["n"], AND=int((r["g"]["type"] == "AND").sum()),
                               reachable_hosts=len(r["g"]["host_nodes"]), steps_median=r["steps"][0], steps_max=r["steps"][1],
                               mulval_s=r["t_mulval"]) for r in RECS])
    table(gstat.groupby("hosts").agg(graphs=("seed", "size"), nodes=("nodes", "mean"), AND_steps=("AND", "mean"),
                                     reachable_hosts=("reachable_hosts", "mean"), steps_median=("steps_median", "median"),
                                     steps_max=("steps_max", "max"), mulval_s=("mulval_s", "mean")), "Attack graphs generated", 1)

    # ---- phase B: ground truth for every attacker model (parallel over graphs)
    def gt_one(rec):
        cf = f"{CACHE}/gt_H{rec['H']}_s{rec['seed']}.pkl"
        if os.path.exists(cf): return pickle.load(open(cf, "rb"))
        out = {bkey(b): mc_attacker(rec["g"], CFG["GT_SAMPLES"], seed=CFG["GT_SEED"] + 1000 * rec["seed"] + i, budget=b)
               for i, b in enumerate(CFG["BUDGETS"])}
        save(out, cf); return out

    t0 = time.time()
    from joblib import Parallel, delayed
    GTS = Parallel(n_jobs=max(1, NCPU), backend="loky", inner_max_num_threads=1)(delayed(gt_one)(r) for r in RECS)
    say(f"[ground truth] {len(GTS)} graphs x {len(GT_KEYS)} attacker models in {time.time()-t0:.0f}s")
    gts = pd.DataFrame([dict(hosts=r["H"], model=k, mean_p=np.mean(list(gt[k].values())), max_p=max(gt[k].values()),
                             zero_share=np.mean(np.array(list(gt[k].values())) == 0)) for r, gt in zip(RECS, GTS) for k in gt])
    table(gts.groupby(["model", "hosts"]).mean(), "Ground-truth compromise probabilities (mean / max / share of hosts never compromised)", 3)

    # ---- phase C: every method on every graph and view (cached per graph)
    def run_one(rec, gt):
        cf = f"{CACHE}/res_H{rec['H']}_s{rec['seed']}.pkl"
        if os.path.exists(cf): return pickle.load(open(cf, "rb"))
        g, seed = rec["g"], rec["seed"]
        views = {"clean": (g, CFG["OMEGAS"])}
        for s_ in CFG["NOISE"]: views[f"noise{s_}"] = (view_noise(g, s_, seed), [0.0, 0.5, 1.0])
        views[f"missing{int(CFG['MISSING'] * 100)}"] = (view_missing(g, CFG["MISSING"], seed), [0.0, 0.5, 1.0])
        rows, times, extra = [], [], {}
        for vname, (gv, oms) in views.items():
            sc, tm, T, nw, _ = methods_on(gv, oms, seed)
            if vname == "clean": extra = dict(T=T, walk_nodes=nw)
            times.append(dict(view=vname, **tm))
            for k in GT_KEYS:
                for m, s_ in sc.items():
                    rows.append(dict(H=rec["H"], seed=seed, view=vname, attacker=k, method=m, **graded_metrics(s_, gt[k], seed)))
                rng = np.random.default_rng(seed); rr = [graded_metrics({h: rng.random() for h in gt[k]}, gt[k], seed + i) for i in range(20)]
                rows.append(dict(H=rec["H"], seed=seed, view=vname, attacker=k, method="Random", **{a: np.mean([r[a] for r in rr]) for a in rr[0]}))
        out = dict(rows=rows, times=times, **extra); save(out, cf); return out

    t0 = time.time(); RES = []
    for i, (rec, gt) in enumerate(zip(RECS, GTS)):
        RES.append(run_one(rec, gt))
        r = pd.DataFrame(RES[-1]["rows"]); r = r[(r.view == "clean") & (r["attacker"] == "Binf")].set_index("method")["NDCG@10"]
        print(f"  methods H={rec['H']} seed={rec['seed']} ({i+1}/{len(RECS)}): NDCG@10 (unlimited attacker) QSW {r['QSW (w=0.5)']:.3f} "
              f"Markov {r['Markov (w=1)']:.3f} A* {r['A*']:.3f} AO* {r['AO*']:.3f} MC-100 {r['MC-100']:.3f} | {time.time()-t0:.0f}s")
    D = pd.DataFrame([x for r in RES for x in r["rows"]]); D.to_csv(f"{OUT}/part1_all_metrics.csv", index=False)
    TM = pd.DataFrame([dict(H=rec["H"], seed=rec["seed"], **t) for rec, r in zip(RECS, RES) for t in r["times"]])

    # ---- phase D: fair CPU timing + peak memory (first graph of every size), and GPU-vs-CPU agreement
    cpu_rows = []
    for H in CFG["SIZES"]:
        if not valid[H]: continue
        rec = valid[H][0]; g = rec["g"]; wg = walk_graph(g); T = t_rule_dir(wg)
        H_, G_, d_ = build_operators(wg["n"], wg["src"], wg["dst"], wg["w"])
        rho0 = np.zeros((wg["n"], wg["n"])); rho0[wg["start"], wg["start"]] = 1.0 / len(wg["start"])
        t0 = time.perf_counter(); qn = qsw_numpy(H_, G_, d_, rho0, CFG["MAIN_OMEGA"], T); tcpu = time.perf_counter() - t0
        tracemalloc.start(); qsw_numpy(H_, G_, d_, rho0, CFG["MAIN_OMEGA"], T); mq = tracemalloc.get_traced_memory()[1] / 2**20; tracemalloc.stop()
        qb = qsw_batch(H_, G_, d_, rho0, [CFG["MAIN_OMEGA"]], T)[0]
        row = dict(hosts=H, walk_nodes=wg["n"], **{"QSW cpu_s": tcpu, "QSW peak_MB": mq, "GPU-vs-CPU max diff": np.abs(qb - qn).max()})
        for nm, fn in {"A*": lambda: astar_scores(wg), "AO*": lambda: aostar_scores(g), "PageRank": lambda: pagerank_scores(wg),
                       "MC-1000": lambda: mc_attacker(g, 1000, seed=1)}.items():
            tracemalloc.start(); t0 = time.perf_counter(); fn(); el = time.perf_counter() - t0
            row[f"{nm} cpu_s"] = el; row[f"{nm} peak_MB"] = tracemalloc.get_traced_memory()[1] / 2**20; tracemalloc.stop()
        cpu_rows.append(row)
    CPU1 = pd.DataFrame(cpu_rows).set_index("hosts")

    # ---- phase E: report
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    Mx = ["NDCG@5", "NDCG@10", "Top5", "Spearman", "Kendall"]

    def stats_vs(df, ref="QSW (w=0.5)", against=None, metrics_=("NDCG@10", "Spearman")):
        q = df[df.method == ref].set_index(["H", "seed"]); out = []
        for b in (against or [m for m in MAIN_METHODS if m != ref]):
            bb = df[df.method == b].set_index(["H", "seed"]).reindex(q.index)
            for mt in metrics_:
                diff = (q[mt] - bb[mt]).to_numpy(); diff = diff[~np.isnan(diff)]
                p = 1.0 if len(diff) == 0 or np.allclose(diff, 0) else stats.wilcoxon(diff, zero_method="zsplit").pvalue
                bs = np.random.default_rng(1).choice(diff, (2000, len(diff))).mean(1) if len(diff) else np.zeros(1)
                out.append(dict(vs=b, metric=mt, mean_diff=diff.mean() if len(diff) else 0, ci_lo=np.percentile(bs, 2.5),
                                ci_hi=np.percentile(bs, 97.5), wins=int((diff > 1e-12).sum()), ties=int((abs(diff) <= 1e-12).sum()),
                                losses=int((diff < -1e-12).sum()), p=p))
        out = pd.DataFrame(out)
        for mt in metrics_:
            k_ = out.metric == mt; ps = out.loc[k_, "p"].to_numpy(); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
            for r_, i in enumerate(o): run = max(run, (len(ps) - r_) * ps[i]); adj[i] = min(1, run)
            out.loc[k_, "p_holm"] = adj
        return out

    for k in GT_KEYS:
        c = D[(D.view == "clean") & (D["attacker"] == k)]
        table(c.groupby("method")[Mx].mean().reindex(MAIN_METHODS), f"[{k}] clean graphs: mean over all {c.seed.nunique()} graphs")
        table(c.pivot_table(index="method", columns="H", values="NDCG@10").reindex(MAIN_METHODS), f"[{k}] NDCG@10 by network size (hosts)")
        table(headline(c, "NDCG@10", GROUPS, ["H", "seed"], f"MulVAL {k} clean"), f"[{k}] HEADLINE: QSW vs the strongest method of each family (NDCG@10)")
        table(headline(c, "Spearman", GROUPS, ["H", "seed"], f"MulVAL {k} clean"), f"[{k}] HEADLINE: QSW vs the strongest method of each family (Spearman)")
        st = stats_vs(c); st.to_csv(f"{OUT}/part1_stats_{k}.csv", index=False)
        table(st.set_index(["vs", "metric"]), f"[{k}] QSW minus each method (paired over graphs, Wilcoxon + Holm, bootstrap 95% CI)")
    # quantum contribution: does coherence change the ranking quality, all attacker models and views?
    qc = []
    for k in GT_KEYS:
        for v in D.view.unique():
            s_ = stats_vs(D[(D.view == v) & (D["attacker"] == k)], ref="QSW (w=0.5)", against=["Markov (w=1)", "CTQW (w=0)"], metrics_=("NDCG@10",))
            s2 = stats_vs(D[(D.view == v) & (D["attacker"] == k)], ref="CTQW (w=0)", against=["Markov (w=1)"], metrics_=("NDCG@10",))
            s_["comparison"] = "QSW - " + s_["vs"]; s2["comparison"] = "CTQW - Markov"
            for x in (s_, s2):
                x["attacker"] = k; x["view"] = v; qc.append(x)
    QC = pd.concat(qc); QC.to_csv(f"{OUT}/part1_quantum_contribution.csv", index=False)
    table(QC.set_index(["attacker", "view", "comparison"])[["mean_diff", "ci_lo", "ci_hi", "wins", "losses", "p"]],
          "Quantum contribution: QSW / CTQW minus the classical walk (Markov, omega=1) and QSW minus CTQW, NDCG@10")
    OM = D[(D.view == "clean") & (D.method.str.startswith("QW@"))].copy(); OM["omega"] = OM.method.str[3:].astype(float)
    table(OM.pivot_table(index="omega", columns="attacker", values="NDCG@10").T, "Decoherence curve (clean, mean NDCG@10; omega=0 pure quantum ... 1 classical)")
    table(OM.pivot_table(index="omega", columns="attacker", values="Spearman").T, "Decoherence curve (clean, mean Spearman)")
    rob = D[D.method.isin(MAIN_METHODS)].pivot_table(index=["attacker", "method"], columns="view", values="NDCG@10")
    table(rob.reindex(pd.MultiIndex.from_product([GT_KEYS, MAIN_METHODS])), "Robustness: mean NDCG@10 per defender view (noise = CVSS uncertainty, missing = 20% of vulnerabilities unseen)")
    for k in GT_KEYS:
        for v in [x for x in D.view.unique() if x != "clean"]:
            headline(D[(D.view == v) & (D["attacker"] == k)], "NDCG@10", GROUPS, ["H", "seed"], f"MulVAL {k} {v}")
    rank_rows = []
    for (k, v), grp in D[D.method.isin(MAIN_METHODS)].groupby(["attacker", "view"]):
        mm = grp.groupby("method")["NDCG@10"].mean().sort_values(ascending=False)
        rank_rows.append(dict(attacker=k, view=v, QSW_rank=int(list(mm.index).index("QSW (w=0.5)") + 1), of=len(mm), best=mm.index[0],
                              best_NDCG10=mm.iloc[0], QSW_NDCG10=mm["QSW (w=0.5)"]))
    table(pd.DataFrame(rank_rows).set_index(["attacker", "view"]), "Where QSW ranks among all methods (mean NDCG@10)")
    table(TM.groupby(["H"]).mean(numeric_only=True).drop(columns=["seed"], errors="ignore").T, "Run time per graph in seconds (quantum batch = GPU time per omega)")
    table(CPU1.T, "Fair CPU timing, peak memory (first graph of each size) and GPU-vs-CPU agreement")
    P1 = dict(D=D, TM=TM, CPU=CPU1, QC=QC, OM=OM, gstat=gstat)
