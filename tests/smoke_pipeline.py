"""End-to-end smoke test of experiments 01-04 on small synthetic attack graphs (no MulVAL, CPU only, ~2 minutes).

    python tests/smoke_pipeline.py [work_dir]

It replaces MulVAL with a tiny generator of MulVAL-format attack graphs (through the run_mulval_override and
gen_network_override hooks of experiment 1), then runs experiments 2-4 on that output. It checks that the code
runs and writes its reports; the numbers mean nothing.
"""
import os, sys, json, subprocess, tempfile
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WORK = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "qswrank_smoke"))
P1 = os.path.join(WORK, "exp1")
os.environ.update(QSAW_OUT=P1, QSAW_RUN_FLOW="0", QSAW_SIZES=json.dumps([5, 6]), QSAW_N_GRAPHS="4",
                  QSAW_GT_SAMPLES="2000")
sys.path.insert(0, ROOT)
import qswrank.mulval as mv  # noqa: E402

mv.RULES = [(f"rule{i}", p) for i, p in enumerate([1.0, 0.8, 1.0, 0.9, 1.0, 0.5, 1.0, 0.6, 1.0, 0.5])]


def gen_network_override(H, seed):
    rng = np.random.default_rng(1000 + seed)
    vuln_p = {f"CVE-{i}": round(float(rng.uniform(0.3, 0.95)), 2) for i in range(H)}
    return dict(H=H, seed=seed), dict(vuln_p=vuln_p, hosts=[f"h{i}" for i in range(H)], zone={})


def run_mulval_override(facts, workdir, timeout):
    """chain of H hosts: h0, h1 reachable from the internet, h(i) from h(i-1) by a multi-hop step, plus random
    password-reuse shortcuts; labels and CSV layout follow MulVAL's VERTICES.CSV / ARCS.CSV"""
    H, rng = facts["H"], np.random.default_rng(facts["seed"])
    V, E = [], []
    def node(label, typ): V.append((len(V) + 1, label, typ, 0)); return len(V)
    att = node("attackerLocated(internet)", "LEAF"); code = []
    for i in range(H):
        h = f"h{i}"
        src = att if i < 2 else code[i - 1]
        hacl = node(f"hacl(internet,{h},tcp,80)" if i < 2 else f"hacl(h{i-1},{h},tcp,80)", "LEAF")
        step = node("RULE 6 (direct network access)" if i < 2 else "RULE 5 (multi-hop access)", "AND")
        net = node(f"netAccess({h},tcp,80)", "OR")
        vul = node(f"vulExists({h},'CVE-{i}',httpd,remoteExploit,privEscalation)", "LEAF")
        svc = node(f"networkServiceInfo({h},httpd,tcp,80,user)", "LEAF")
        ex = node("RULE 2 (remote exploit of a server program)", "AND")
        cx = node(f"execCode({h},user)", "OR")
        E += [(step, src, 1), (step, hacl, 1), (net, step, 1), (ex, net, 1), (ex, vul, 1), (ex, svc, 1), (cx, ex, 1)]
        code.append(cx)
        if i >= 2 and rng.random() < 0.5:
            j = int(rng.integers(0, i - 1)); pw = node(f"sharedPassword(h{j},{h})", "LEAF")
            lg = node("RULE 9 (log in with a reused password)", "AND"); E += [(lg, code[j], 1), (lg, pw, 1), (cx, lg, 1)]
    return (pd.DataFrame(V, columns=["id", "label", "type", "p0"]), pd.DataFrame(E, columns=["parent", "child", "w"]), 0.0)


def run_script(path, out, env_extra=None):
    env = dict(os.environ, QSAW_OUT=out, QSAW_PREV_ROOTS=P1, **(env_extra or {}))
    r = subprocess.run([sys.executable, path], cwd=ROOT, env=env, capture_output=True, text=True)
    print(f"[smoke] {os.path.basename(path)}: exit {r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(1)
    return r.stdout


if __name__ == "__main__":
    exp1 = os.path.join(ROOT, "experiments", "01_mulval_benchmark.py")
    ns = dict(__name__="__main__", __file__=exp1, run_mulval_override=run_mulval_override,
              gen_network_override=gen_network_override)
    exec(compile(open(exp1).read(), exp1, "exec"), ns)
    print("[smoke] 01_mulval_benchmark.py: done")
    run_script("experiments/02_main_results.py", os.path.join(WORK, "exp2"))
    run_script("experiments/04_variants_heldout.py", os.path.join(WORK, "exp4"))
    run_script("experiments/03_classical_hopping.py", os.path.join(WORK, "exp3"))
    print("[smoke] all experiments ran")
