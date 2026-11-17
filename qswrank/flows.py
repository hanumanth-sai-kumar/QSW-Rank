"""Flow datasets (improved CIC-IDS2017 / CSE-CIC-IDS2018): instance building and metrics.

Exploratory only: these traffic-flow experiments are not part of the QSW-Rank paper, which uses MulVAL graphs.
"""
from .config import *  # noqa: F401,F403
from .engine import *  # noqa: F401,F403
from .methods import *  # noqa: F401,F403


COLS = ["Src IP", "Dst IP", "Timestamp", "Label", "Attempted Category"]
FLOW_SPEC = {"CIC2017": dict(subdir="CICIDS2017_improved", internal=["192.168.10.0/24"],
                             files=["monday", "tuesday", "wednesday", "thursday", "friday"]),
             "CSE2018": dict(subdir="CSECICIDS2018_improved", internal=["172.31.0.0/16"], files=None)}
KS = (1, 3, 5, 10); FLOW_M = [f"R@{k}" for k in KS] + ["MRR", "MAP", "NDCG@10"]
FLOW_METHODS = ["QSW (w=0.5)", "CTQW (w=0)", "Markov (w=1)", "QAGC-DTQW", "QAGC-CTQW", "QAGC-HQW (Alg.3)",
                "QAGC-repo (ref. impl.)", "BFS-ref", "DFS-ref", "A*-ref", "Markov-ref", "BFS", "DFS", "A*", "PageRank",
                "WDegree", "Random"]
FLOW_GROUPS = {"QAGC quantum walks": ["QAGC-DTQW", "QAGC-CTQW", "QAGC-HQW (Alg.3)", "QAGC-repo (ref. impl.)"],
               "QAGC reference classical baselines": ["BFS-ref", "DFS-ref", "A*-ref", "Markov-ref"],
               "Information-matched classical baselines": ["BFS", "DFS", "A*", "PageRank", "WDegree"],
               "Ablation (same model, quantum part switched)": ["CTQW (w=0)", "Markov (w=1)"]}


def flow_load_day(path):
    try:
        df = pd.read_csv(path, usecols=COLS, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path, usecols=COLS, low_memory=False)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="ISO8601", errors="coerce")
    df = df.dropna(subset=["Timestamp", "Src IP", "Dst IP"])
    df["Src IP"] = df["Src IP"].astype(str).str.strip(); df["Dst IP"] = df["Dst IP"].astype(str).str.strip()
    lab = df["Label"].astype(str).str.strip().str.upper(); df["mal"] = lab.ne("BENIGN")
    df["succ"] = df["mal"] & (pd.to_numeric(df["Attempted Category"], errors="coerce") == -1)
    return df[["Src IP", "Dst IP", "Timestamp", "mal", "succ"]]


def flow_internal_map(ips, nets):
    nets = [ipaddress.ip_network(x) for x in nets]; out = {}
    for ip in ips:
        try: a = ipaddress.ip_address(ip); out[ip] = any(a in nt for nt in nets)
        except ValueError: out[ip] = False
    return out


def flow_instance(w, N, is_int, ext_mal, meta):
    """label-free graph: ALL active internal hosts, then external initiators (entries), then busiest others, up to N"""
    act = pd.concat([w["Src IP"], w["Dst IP"]]).value_counts()
    if len(act) < N: return None, "too_small"
    act = act.rename_axis("ip").reset_index(name="c"); act["internal"] = act["ip"].map(is_int).fillna(False).astype(bool)
    n_int = int(act["internal"].sum())
    if n_int >= N: return None, "internal_hosts_exceed_N"
    si = w["Src IP"].map(is_int).fillna(False).astype(bool); di = w["Dst IP"].map(is_int).fillna(False).astype(bool)
    initiators = set(w.loc[~si & di, "Src IP"])
    act["prio"] = np.where(act["internal"], 2, np.where(act["ip"].isin(initiators), 1, 0))
    act = act.sort_values(["prio", "c", "ip"], ascending=[False, False, True]).head(N)
    names = act["ip"].to_numpy(); pos = {ip: i for i, ip in enumerate(names)}
    sel = w[w["Src IP"].isin(pos) & w["Dst IP"].isin(pos) & (w["Src IP"] != w["Dst IP"])]
    e = sel.groupby(["Src IP", "Dst IP"]).size().reset_index(name="c")
    if len(e) == 0: return None, "no_edges"
    src = e["Src IP"].map(pos).to_numpy(int); dst = e["Dst IP"].map(pos).to_numpy(int); wgt = np.log1p(e["c"].to_numpy(float))
    internal = np.array([bool(is_int.get(ip, False)) for ip in names])
    entries = sorted({int(u) for u, v in zip(src, dst) if not internal[u] and internal[v]})
    if not entries: return None, "no_entry"
    succ = w[w["succ"]]
    gt_ips = {ip for ip in pd.unique(pd.concat([succ["Src IP"], succ["Dst IP"]])) if is_int.get(ip, False)}
    cand = internal.copy(); gt = np.array([ip in gt_ips for ip in names]) & cand
    if gt.sum() == 0: return None, "gt_not_in_graph"
    if cand.sum() < 2: return None, "few_candidates"
    s_out = np.bincount(src, weights=wgt, minlength=N)[entries]; q = s_out / s_out.sum()
    seed = zlib.crc32(f"{meta['dataset']}|{meta['day']}|{meta['window']}|{N}".encode())
    return dict(meta, N=N, n=N, names=names, src=src, dst=dst, w=wgt, start=entries, q=q, internal=internal, cand=cand, gt=gt,
                seed=seed, tieperm=np.random.default_rng(seed).permutation(N), n_edges=len(e), n_entries=len(entries),
                n_gt=int(gt.sum()), n_cand=int(cand.sum())), "ok"


def flow_t_rule(inst):
    adj = [[] for _ in range(inst["n"])]
    for a, b in zip(inst["src"], inst["dst"]): adj[a].append(b); adj[b].append(a)
    dist = np.full(inst["n"], -1); q = deque(inst["start"])
    for s in inst["start"]: dist[s] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0: dist[v] = dist[u] + 1; q.append(v)
    r = dist[dist > 0]
    return max(DT, round(max(float(r.mean()) if len(r) else 1.0, 1.0) / DT) * DT)


def rank_metrics(score, cand, gt, tieperm):
    s = np.where(np.isfinite(score), score, NEG); idx = np.flatnonzero(cand)
    order = idx[np.lexsort((tieperm[idx], -s[idx]))]; hit = gt[order]; ng = int(gt.sum()); ranks = np.flatnonzero(hit) + 1
    out = {f"R@{k}": hit[:k].sum() / ng for k in KS}
    out["MRR"] = 1.0 / ranks[0] if len(ranks) else 0.0
    out["MAP"] = float(np.mean(np.arange(1, len(ranks) + 1) / ranks)) if len(ranks) else 0.0
    out["NDCG@10"] = np.sum(1 / np.log2(ranks[ranks <= 10] + 1)) / np.sum(1 / np.log2(np.arange(1, min(ng, 10) + 1) + 1))
    return out


def flow_dataset(name):
    spec = FLOW_SPEC[name]; base = os.path.join(CFG["FLOW_ROOT"], spec["subdir"])
    files = spec["files"] or sorted(f[:-4] for f in os.listdir(base) if f.endswith(".csv"))
    insts, log = [], []
    for day in files:
        t0 = time.time(); df = flow_load_day(os.path.join(base, f"{day}.csv"))
        ips = pd.unique(pd.concat([df["Src IP"], df["Dst IP"]])); is_int = flow_internal_map(ips, spec["internal"])
        mal = df[df["mal"]]; ext_mal = {ip for ip in pd.unique(pd.concat([mal["Src IP"], mal["Dst IP"]])) if not is_int[ip]}
        nd = 0
        for N, freq in CFG["FLOW_SIZES"].items():
            df["win"] = df["Timestamp"].dt.floor(freq); att = np.sort(df.loc[df["succ"], "win"].unique())
            sub = df[df["win"].isin(att)]; ok, why = [], {}
            for wk, w in sub.groupby("win", sort=True):
                inst, r = flow_instance(w, N, is_int, ext_mal, dict(dataset=name, day=day, window=str(wk)))
                why[r] = why.get(r, 0) + 1
                if inst is not None: ok.append(inst)
            if len(ok) > CFG["FLOW_MAXWIN"]:
                ok = [ok[i] for i in np.unique(np.linspace(0, len(ok) - 1, CFG["FLOW_MAXWIN"]).round().astype(int))]
            insts += ok; nd += len(ok)
            log.append(dict(dataset=name, day=day, N=N, attack_windows=len(att), used=len(ok), **{f"skip_{k}": v for k, v in why.items() if k != "ok"}))
        print(f"  [{name}/{day}] rows={len(df):,} instances={nd} ({time.time()-t0:.0f}s)")
        del df, sub; gc.collect()
    for i in insts: i["uid"] = f"{name}|{i['day']}|{i['window']}|{i['N']}"
    return insts, pd.DataFrame(log).fillna(0)


def flow_methods(inst):
    wg = dict(n=inst["n"], src=inst["src"], dst=inst["dst"], w=inst["w"], start=inst["start"], q=inst["q"])
    H, G, d = build_operators(inst["n"], inst["src"], inst["dst"], inst["w"]); T = flow_t_rule(inst)
    rho0 = np.zeros((inst["n"], inst["n"])); rho0[inst["start"], inst["start"]] = inst["q"]
    t0 = time.perf_counter(); Q = qsw_batch(H, G, d, rho0, CFG["OMEGAS"], T); tq = (time.perf_counter() - t0) / len(CFG["OMEGAS"])
    sc = {f"QW@{w}": Q[i] for i, w in enumerate(CFG["OMEGAS"])}
    for w, nm in Q_NAMES.items(): sc[nm] = sc[f"QW@{w}"]
    tm = {"quantum batch": tq}
    s_in = np.bincount(inst["dst"], weights=inst["w"], minlength=inst["n"]); s_out = np.bincount(inst["src"], weights=inst["w"], minlength=inst["n"])
    for nm, fn in {"BFS": lambda: order_scores(wg, False, undirected=True), "DFS": lambda: order_scores(wg, True, undirected=True),
                   "A*": lambda: astar_scores(wg, undirected=True, normalise=True), "PageRank": lambda: pagerank_scores(wg),
                   "WDegree": lambda: s_in + s_out, "QAGC-repo (ref. impl.)": lambda: qagc_repo_scores(wg),
                   "BFS-ref": lambda: ref_reach_scores(wg), "DFS-ref": lambda: ref_dfs_scores(wg),
                   "A*-ref": lambda: ref_reach_scores(wg), "Markov-ref": lambda: ref_markov_scores(wg)}.items():
        t0 = time.perf_counter(); sc[nm] = fn(); tm[nm] = time.perf_counter() - t0
    t0 = time.perf_counter(); S_ = qagc_struct(wg); t_struct = time.perf_counter() - t0
    for nm, fn in {"QAGC-DTQW": lambda: qagc_dtqw_scores(S_), "QAGC-CTQW": lambda: qagc_ctqw_scores(S_),
                   "QAGC-HQW (Alg.3)": lambda: qagc_hqw_scores(S_)}.items():
        t0 = time.perf_counter(); sc[nm] = fn(); tm[nm] = time.perf_counter() - t0 + t_struct
    return sc, tm, T



__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]
