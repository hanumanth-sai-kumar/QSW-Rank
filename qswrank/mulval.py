"""MulVAL: build it once (Kaggle), generate enterprise networks, run MulVAL, parse the attack graph."""
from .config import *  # noqa: F401,F403


def sh(cmd, timeout=None):
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)


W_ = "/kaggle/working"


def ensure_mulval():
    if os.path.isfile(f"{W_}/mulval/bin/attack_graph") and glob.glob(f"{W_}/XSB/bin/xsb"):
        return "already built"
    if not glob.glob(f"{W_}/XSB/build/configure"):
        src = glob.glob("/kaggle/input/**/XSB*", recursive=True)
        if not src:
            raise SystemExit("XSB archive not found: attach the dataset containing XSB-5.1.0-source.bin")
        shutil.rmtree(f"{W_}/xsbx", ignore_errors=True); os.makedirs(f"{W_}/xsbx")
        (zipfile.ZipFile(src[0]) if zipfile.is_zipfile(src[0]) else tarfile.open(src[0])).extractall(f"{W_}/xsbx")
        cfg = glob.glob(f"{W_}/xsbx/**/build/configure", recursive=True)
        shutil.rmtree(f"{W_}/XSB", ignore_errors=True); shutil.move(os.path.dirname(os.path.dirname(cfg[0])), f"{W_}/XSB")
    script = r'''
    cd /kaggle/working
    apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y build-essential flex bison default-jdk graphviz > /dev/null 2>&1
    export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac)))); cp $JAVA_HOME/include/linux/jni_md.h $JAVA_HOME/include/ 2>/dev/null
    [ -d mulval ] || git clone -q https://github.com/risksense/mulval.git
    if [ -z "$(find XSB -path '*bin/xsb' 2>/dev/null)" ]; then chmod -R u+x XSB/build; (cd XSB/build && CFLAGS="-fcommon -O2" ./configure > /tmp/xc.log 2>&1 && ./makexsb > /tmp/xm.log 2>&1) || { tail -20 /tmp/xm.log; exit 1; }; fi
    export PATH=$(dirname $(find /kaggle/working/XSB -path "*bin/xsb" | head -1)):$PATH
    cd mulval/src/attack_graph
    sed -i 's/^\tg++ -g -DLINUX/\tg++ -std=gnu++11 -fpermissive -fcommon -Wl,--allow-multiple-definition -g -DLINUX/' Makefile
    sed -i 's/^\tgcc -g -c lex.yy.c/\tgcc -fcommon -g -c lex.yy.c/' Makefile
    sed -i 's/^\tmv graphit.tab.h y.tab.cpp.h/\tcp graphit.tab.h y.tab.cpp.h/' Makefile
    sed -i 's/fa >0;/fa != 0;/g' attack_graph.cpp
    cd /kaggle/working; mkdir -p mulval/bin/adapter mulval/bin/metrics; [ -d mulval/bin/attack_graph ] && rmdir mulval/bin/attack_graph
    export MULVALROOT=/kaggle/working/mulval; (cd mulval && make > /tmp/mm.log 2>&1) || { grep -i -A3 " error" /tmp/mm.log | tail -20; exit 1; }
    '''
    r = sh(script)
    if r.returncode != 0:
        raise SystemExit("MulVAL build failed:\n" + r.stdout[-2000:] + r.stderr[-1000:])
    return "built"


def rule_probs(path):
    txt = re.sub(r"/\*.*?\*/", "", open(path).read(), flags=re.S)
    txt = "\n".join(l for l in txt.splitlines() if not l.strip().startswith("%"))
    out = []
    for p in txt.split("interaction_rule(")[1:]:
        m = re.search(r"rule_desc\(\s*'(.*?)'\s*,\s*([0-9.]+)\s*\)", p, flags=re.S)
        out.append((m.group(1), float(m.group(2))) if m else ("", 1.0))
    return out


KNOBS = dict(p_vuln=0.30, p_local=0.30, p_client=0.40, p_incompetent=0.30, p_reuse=0.20, p_admin_cache=0.15, p_leak=0.02)
CVSS_W = dict(AV={"N": 0.85, "L": 0.55}, AC={"L": 0.77, "H": 0.44}, PR={"N": 0.85, "L": 0.62}, UI={"N": 0.85, "R": 0.62})


def exploit_prob(rng, av, ui):
    """CVSS v3.1 exploitability 8.22*AV*AC*PR*UI normalised by its maximum 3.887; AC, PR sampled from CVSS categories"""
    ac = rng.choice(["L", "H"], p=[0.8, 0.2]); pr = rng.choice(["N", "L"], p=[0.6, 0.4])
    return round(min(1.0, 8.22 * CVSS_W["AV"][av] * CVSS_W["AC"][ac] * CVSS_W["PR"][pr] * CVSS_W["UI"][ui] / 3.887), 4)


def gen_network(H, seed, k=KNOBS):
    rng = np.random.default_rng(seed)
    nd = max(2, round(0.06 * H)); ns = max(2, round(0.12 * H)); nw = max(2, H - nd - ns - 2)
    dmz = [f"{['web', 'mail', 'vpn'][i % 3]}{i}" for i in range(nd)]
    srv = [f"{['db', 'file', 'app'][i % 3]}{i}" for i in range(ns)]
    ndep = max(1, round(nw / 15)); ws = [f"ws_d{i % ndep}_{i}" for i in range(nw)]
    dept = {w: int(w.split("_")[1][1:]) for w in ws}; admin = ["dc0", "adminws0"]; hosts = dmz + srv + ws + admin
    zone = {**{h: "dmz" for h in dmz}, **{h: "srv" for h in srv}, **{h: "lan" for h in ws}, **{h: "adm" for h in admin}}
    F, P, svc = ["attackerLocated(internet).", "hacl(H,H,_,_)."], {}, []
    F += [f"attackGoal(execCode({h},_))." for h in hosts]

    def add_svc(h, prog, port, perm):
        svc.append((h, prog, port, perm)); F.append(f"networkServiceInfo({h},{prog},tcp,{port},{perm}).")

    for h in dmz:
        t = h.rstrip("0123456789")
        if t == "web": add_svc(h, "httpd", 80, "user"); F.append(f"isWebServer({h}).")
        if t == "mail": add_svc(h, "smtpd", 25, "user")
        if t == "vpn": add_svc(h, "vpnService", 443, "root")
    for h in srv:
        t = h.rstrip("0123456789")
        add_svc(h, {"db": "mysqld", "file": "smbd", "app": "tomcat"}[t], {"db": 3306, "file": 445, "app": 8080}[t],
                "root" if t == "file" else "user")
        add_svc(h, "sshd", 22, "root")
    for h in ws:
        add_svc(h, "smbd", 445, "root")
        if rng.random() < 0.3: add_svc(h, "sshd", 22, "user")
    add_svc("dc0", "ldapd", 389, "root"); add_svc("dc0", "sshd", 22, "root"); add_svc("adminws0", "sshd", 22, "root")
    for (h, prog, port, perm) in svc:
        if rng.random() < k["p_vuln"]:
            v = f"v_{h}_{prog}"; P[v] = exploit_prob(rng, "N", "N")
            F += [f"vulExists({h},{v},{prog}).", f"vulProperty({v},remoteExploit,privEscalation)."]
    for h in hosts:
        if rng.random() < k["p_local"]:
            v = f"l_{h}"; P[v] = exploit_prob(rng, "L", "N")
            F += [f"vulExists({h},{v},kernel).", f"vulProperty({v},localExploit,privEscalation)."]
    for h in ws:
        u = f"u_{h}"; F.append(f"hasAccount({u},{h},user).")
        if rng.random() < k["p_incompetent"]: F.append(f"inCompetent({u}).")
        if rng.random() < k["p_client"]:
            v = f"c_{h}"; P[v] = exploit_prob(rng, "N", "R")
            F += [f"vulExists({h},{v},browser).", f"vulProperty({v},remoteClient,privEscalation)."]
        same = [x for x in ws if dept[x] == dept[h] and x != h]
        if same and rng.random() < k["p_reuse"]: F.append(f"hasAccount({u},{same[rng.integers(len(same))]},user).")
        if rng.random() < k["p_admin_cache"]: F.append(f"hasAccount(itadmin,{h},root).")
    F += [f"hasAccount(itadmin,{h},root)." for h in srv + admin]
    A = set()

    def allow(a, b, port): A.add((a, b, "tcp", port))

    for h in dmz:
        allow("internet", h, {"web": 80, "mail": 25, "vpn": 443}[h.rstrip("0123456789")])
    for h in srv:
        t = h.rstrip("0123456789")
        if t == "db":
            for x in dmz + srv:
                if x.startswith("web") or x.startswith("app"): allow(x, h, 3306)
        if t == "app":
            for x in dmz:
                if x.startswith("web"): allow(x, h, 8080)
            for x in ws: allow(x, h, 8080)
        if t == "file":
            for x in ws: allow(x, h, 445)
        allow(h, "dc0", 389); allow("adminws0", h, 22)
    for x in dmz:
        if x.startswith("vpn"):
            for w in ws: allow(x, w, 445); allow(x, w, 22)
    for w in ws:
        allow(w, "dc0", 389); allow("adminws0", w, 22); allow("adminws0", w, 445)
        for y in ws:
            if y != w and dept[y] == dept[w]: allow(w, y, 445); allow(w, y, 22)
    allow("adminws0", "dc0", 22); allow("adminws0", "dc0", 389)
    ports = {"dmz": [80, 25, 443, 22], "srv": [3306, 445, 8080, 22], "lan": [445, 22], "adm": [22, 389]}
    for a in hosts:
        for b in hosts:
            if a != b and zone[a] != zone[b] and rng.random() < k["p_leak"] / max(1, len(hosts) / 50):
                allow(a, b, int(rng.choice(ports[zone[b]])))
    F += [f"hacl({a},{b},{p},{q})." for (a, b, p, q) in sorted(A)]
    for w in ws:
        F.append(f"hacl({w},internet,httpProtocol,httpPort).")
        for x in dmz:
            if x.startswith("web"): F.append(f"hacl({w},{x},httpProtocol,httpPort).")
    return "\n".join(F) + "\n", dict(hosts=hosts, zone=zone, vuln_p=P)


def run_mulval(facts, workdir, timeout):
    shutil.rmtree(workdir, ignore_errors=True); os.makedirs(workdir)
    open(f"{workdir}/input.P", "w").write(facts)
    t0 = time.time()
    subprocess.run(["bash", "-c", f"cd {workdir} && graph_gen.sh input.P -v > run.log 2>&1"],
                   env=ENV, capture_output=True, text=True, timeout=timeout)
    el = time.time() - t0
    if not os.path.exists(f"{workdir}/VERTICES.CSV"):
        raise RuntimeError("no graph: " + open(f"{workdir}/run.log").read()[-300:].replace("\n", " "))
    V = pd.read_csv(f"{workdir}/VERTICES.CSV", header=None, names=["id", "label", "type", "p0"])
    E = pd.read_csv(f"{workdir}/ARCS.CSV", header=None, names=["parent", "child", "w"])
    return V, E, el


def build_graph(V, E, meta):
    """MulVAL arcs point from a node to its preconditions; attack flow = child -> parent"""
    ids = V["id"].to_numpy(); ix = {v: i for i, v in enumerate(ids)}
    lab = V["label"].tolist(); typ = np.array(V["type"].tolist()); n = len(V)
    par = E["parent"].map(ix).to_numpy(); chi = E["child"].map(ix).to_numpy()
    and_p = np.ones(n); leaf_p = np.ones(n); vuln = np.zeros(n, bool)
    for i in range(n):
        if typ[i] == "AND":
            m = re.match(r"RULE (\d+)", lab[i]); k = int(m.group(1)) if m else -1
            and_p[i] = RULES[k][1] if 0 <= k < len(RULES) else 1.0
        if typ[i] == "LEAF" and lab[i].startswith("vulExists("):
            vuln[i] = True; leaf_p[i] = meta["vuln_p"].get(lab[i][len("vulExists("):-1].split(",")[1].strip("'"), 1.0)
    host_nodes = {}
    for i in range(n):
        m = re.match(r"execCode\(([^,]+),", lab[i])
        if m and typ[i] == "OR": host_nodes.setdefault(m.group(1), []).append(i)
    start = [i for i in range(n) if lab[i].startswith("attackerLocated(")]
    return dict(n=n, label=lab, type=typ, par=par, chi=chi, and_p=and_p, leaf_p=leaf_p, vuln=vuln,
                host_nodes=host_nodes, start=start, hosts=meta["hosts"], zone=meta["zone"])


def depth_stats(g):
    adj = [[] for _ in range(g["n"])]
    for p, c in zip(g["par"], g["chi"]): adj[c].append(p)
    d = np.full(g["n"], -1); q = deque(g["start"])
    for s in g["start"]: d[s] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] < 0: d[v] = d[u] + 1; q.append(v)
    st = [min(d[i] for i in idx) // 2 for idx in g["host_nodes"].values() if min(d[i] for i in idx) >= 0]
    return (int(np.median(st)), int(max(st))) if st else (0, 0)


__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]
