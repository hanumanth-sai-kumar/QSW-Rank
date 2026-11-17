"""Exploratory - Part 2 of the Kaggle notebook: flow datasets (CIC-IDS2017 / CSE-CIC-IDS2018).

Not used in the QSW-Rank paper. Needs the improved CIC-IDS2017/CSE-CIC-IDS2018 dataset (QSAW_FLOW_ROOT).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from qswrank import *  # noqa: F401,F403

set_report(f"{OUT}/RESULTS_flows.md", "# Q-SAW flow-dataset results\n\n")
P2 = {}
if CFG["RUN_FLOW"]:
    say("\n## Part 2: flow datasets (strictly unsupervised protocol v4.1)")
    for name in CFG["FLOW_DATASETS"]:
        if not os.path.isdir(os.path.join(CFG["FLOW_ROOT"], FLOW_SPEC[name]["subdir"])):
            say(f"[flow] {name}: data folder not found, skipped"); continue
        cf = f"{CACHE}/flow_{name}.pkl"
        if os.path.exists(cf):
            F = pickle.load(open(cf, "rb"))
        else:
            insts, log = flow_dataset(name); rows, times = [], []
            t0 = time.time()
            for j, inst in enumerate(insts):
                sc, tm, T = flow_methods(inst); times.append(dict(N=inst["N"], **tm))
                prng = np.random.default_rng(inst["seed"] + 99); gp = np.zeros(inst["n"], bool)
                gp[prng.choice(np.flatnonzero(inst["cand"]), inst["n_gt"], replace=False)] = True
                for m, s_ in sc.items():
                    r = rank_metrics(s_, inst["cand"], inst["gt"], inst["tieperm"]); rp = rank_metrics(s_, inst["cand"], gp, inst["tieperm"])
                    rows.append(dict(uid=inst["uid"], day=inst["day"], N=inst["N"], method=m, perm_MAP=rp["MAP"], **r))
                rr = [rank_metrics(np.random.default_rng(inst["seed"] + i).random(inst["n"]), inst["cand"], inst["gt"], inst["tieperm"]) for i in range(30)]
                rows.append(dict(uid=inst["uid"], day=inst["day"], N=inst["N"], method="Random", perm_MAP=np.mean([x["MAP"] for x in rr]),
                                 **{k: np.mean([x[k] for x in rr]) for k in rr[0]}))
                if (j + 1) % 50 == 0: print(f"  [{name}] {j+1}/{len(insts)} instances ({time.time()-t0:.0f}s)")
            F = dict(D=pd.DataFrame(rows), TM=pd.DataFrame(times), log=log,
                     meta=pd.DataFrame([{k: i[k] for k in ("uid", "day", "N", "n_edges", "n_entries", "n_gt", "n_cand")} for i in insts]))
            save(F, cf)
        F["D"].to_csv(f"{OUT}/part2_{name}_metrics.csv", index=False); P2[name] = F
        Df = F["D"]
        table(F["log"].set_index(["day", "N"]).drop(columns=["dataset"]), f"{name}: attack windows -> usable instances (skip reasons)", 0)
        table(F["meta"].groupby("N")[["n_cand", "n_gt", "n_entries"]].mean(), f"{name}: mean internal candidates / attacked hosts / entry nodes per graph", 1)
        for N in sorted(Df.N.unique()):
            t = Df[Df.N == N]
            tab = t.groupby("method")[FLOW_M].mean().reindex(FLOW_METHODS)
            tab["time_ms"] = F["TM"][F["TM"].N == N].mean(numeric_only=True).reindex(["quantum batch" if m in Q_NAMES.values() else m for m in FLOW_METHODS]).to_numpy() * 1e3
            table(tab, f"{name} N={N}: mean over {t.uid.nunique()} windows (time_ms: quantum = GPU per omega)")
            table(t.pivot_table(index="method", columns="day", values="MAP").reindex(FLOW_METHODS), f"{name} N={N}: MAP per day", 3)
            table(t.groupby("method")[["perm_MAP"]].mean().reindex(FLOW_METHODS).T, f"{name} N={N}: label-permutation control (all should be near Random)", 3)
            table(headline(t, "MAP", FLOW_GROUPS, ["uid"], f"{name} N={N}"), f"{name} N={N} HEADLINE: QSW vs the strongest method of each family (MAP)")
            q = t[t.method == "QSW (w=0.5)"].set_index("uid"); st = []
            for b in FLOW_METHODS[1:]:
                bb = t[t.method == b].set_index("uid").reindex(q.index)
                for mt in ("MAP", "MRR"):
                    diff = (q[mt] - bb[mt]).to_numpy()
                    p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(diff, zero_method="zsplit").pvalue
                    bs = np.random.default_rng(1).choice(diff, (2000, len(diff))).mean(1)
                    st.append(dict(vs=b, metric=mt, mean_diff=diff.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                                   wins=int((diff > 1e-12).sum()), losses=int((diff < -1e-12).sum()), p=p))
            st = pd.DataFrame(st)
            for mt in ("MAP", "MRR"):
                k_ = st.metric == mt; ps = st.loc[k_, "p"].to_numpy(); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
                for r_, i in enumerate(o): run = max(run, (len(ps) - r_) * ps[i]); adj[i] = min(1, run)
                st.loc[k_, "p_holm"] = adj
            table(st.set_index(["vs", "metric"]), f"{name} N={N}: QSW minus each method (Wilcoxon + Holm, bootstrap 95% CI)")
        om = Df[Df.method.str.startswith("QW@")].copy(); om["omega"] = om.method.str[3:].astype(float)
        table(om.pivot_table(index="N", columns="omega", values="MAP"), f"{name}: decoherence curve (mean MAP; omega=0 quantum ... 1 classical)", 3)
