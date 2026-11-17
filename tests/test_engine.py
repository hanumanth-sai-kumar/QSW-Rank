"""Checks of the walk engine. Run from the repository root:  python -m pytest -q   (or  python tests/test_engine.py)

1. engine_self_test(): the O(N^2) equation against the full N^2 x N^2 superoperator, RK4 time averages against an
   exact matrix exponential, and omega = 1 against the classical master equation.
2. The two-host example of Fig. 1 of the paper: QSW-Rank scores of web0 and db0.
"""
import os, sys, tempfile
os.environ.setdefault("QSAW_OUT", os.path.join(tempfile.gettempdir(), "qswrank_test_out"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from qswrank import build_operators, engine_self_test, qsw_numpy, t_rule_dir

EXPLOIT = lambda AV, AC, PR, UI: 8.22 * AV * AC * PR * UI / 3.887      # CVSS v3.1 exploitability scaled to [0, 1]


def two_host_example():
    """walk graph of Fig. 1: 0 attacker, 1 direct access, 2 net web0, 3 exploit Apache, 4 code web0, 5 multi-hop,
    6 net db0, 7 exploit MySQL, 8 code db0, 9 reused password"""
    p_apache, p_mysql = EXPLOIT(0.85, 0.44, 0.85, 0.85), EXPLOIT(0.85, 0.44, 0.62, 0.85)
    src = np.array([0, 1, 2, 3, 4, 5, 6, 7, 4, 9]); dst = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 8])
    w = np.array([1.0, 1, p_apache, 1, 0.5, 1, p_mysql, 1, 0.5, 1])
    return 10, src, dst, w


def test_engine_self_test():
    engine_self_test()          # raises AssertionError if any check fails


def test_two_host_example():
    n, src, dst, w = two_host_example()
    H, G, d = build_operators(n, src, dst, w)
    T = t_rule_dir(dict(n=n, src=src, dst=dst, start=[0]))
    rho0 = np.zeros((n, n), complex); rho0[0, 0] = 1
    S = np.mean([qsw_numpy(H, G, d, rho0, om, T) for om in [round(0.1 * i, 1) for i in range(11)]], axis=0)
    assert abs(T - 4.35) < 1e-9
    assert abs(S[4] - 0.0545) < 5e-4 and abs(S[8] - 0.0098) < 5e-4      # web0 above db0, as in the paper
    assert abs(np.real(np.trace(rho0)) - 1) < 1e-12


if __name__ == "__main__":
    test_engine_self_test(); test_two_host_example(); print("all tests passed")
