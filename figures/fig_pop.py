"""Fig. 4: node populations over time on the two-host graph of Fig. 1 at omega = 0, 0.5, 1.

Operators, horizon T and the time-averaged check values come from the qswrank package.
Populations are integrated here with the same reduced equation (Eq. 8) and RK4, dt = 0.05.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np
from matplotlib.colors import PowerNorm

import tempfile
os.environ.setdefault("QSAW_OUT", os.path.join(tempfile.gettempdir(), "qswrank_fig_out"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qswrank
E = vars(qswrank)                      # build_operators, t_rule_dir, qsw_numpy from the package

cv = lambda AV, AC, PR, UI: 8.22 * AV * AC * PR * UI / 3.887
pX, pZ = cv(0.85, 0.44, 0.85, 0.85), cv(0.85, 0.44, 0.62, 0.85)
# 0 s, 1 direct access, 2 net web0, 3 exploit Apache, 4 code web0, 5 multi-hop, 6 net db0, 7 exploit MySQL, 8 code db0, 9 reused pwd
src = np.array([0, 1, 2, 3, 4, 5, 6, 7, 4, 9]); dst = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 8])
w = np.array([1.0, 1, pX, 1, 0.5, 1, pZ, 1, 0.5, 1])
n = 10
H, Gm, d = E["build_operators"](n, src, dst, w)
dense = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
H0, G0, d0 = H, Gm, d
H, Gm, d = dense(H).astype(complex), dense(Gm).astype(float), dense(d).astype(float).ravel()
T = E["t_rule_dir"](dict(n=n, src=src, dst=dst, start=[0]))
dt = 0.05; steps = int(round(T / dt))

def F(r, om):
    pi = np.real(np.diag(r))
    L = np.diag(d)
    return -1j * (1 - om) * (H @ r - r @ H) + om * (np.diag(Gm @ pi) - 0.5 * (L @ r + r @ L))

def run(om):
    r = np.zeros((n, n), complex); r[0, 0] = 1
    P = [np.real(np.diag(r))]
    for _ in range(steps):
        k1 = F(r, om); k2 = F(r + dt / 2 * k1, om); k3 = F(r + dt / 2 * k2, om); k4 = F(r + dt * k3, om)
        r = r + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        P.append(np.real(np.diag(r)))
    return np.array(P)

rho0 = np.zeros((n, n), complex); rho0[0, 0] = 1
order = [0, 1, 2, 3, 4, 5, 9, 6, 7, 8]
labels = ["start $s$", "Direct access", "Net. access web0", "Exploit Apache", "Code exec. web0",
          "Multi-hop access", "Reused password", "Net. access db0", "Exploit MySQL", "Code exec. db0"]
db_side = [5, 6, 7, 8, 9]
print("T =", T, "steps", steps)
runs = {}
for om in (0.0, 0.5, 1.0):
    P = run(om); runs[om] = P
    avg = np.asarray(E["qsw_numpy"](H0, G0, d0, rho0, om, T)).real
    print(f"w={om}: db-side mass at T {P[-1][db_side].sum():.3f}; engine time-avg web0 {avg[4]:.4f} db0 {avg[8]:.4f}")

setup(7)
fig, axes = plt.subplots(1, 3, figsize=(6.22, 1.95), sharey=True)
norm = PowerNorm(0.5, vmin=0, vmax=1)
titles = [r"(a) $\omega=0$: pure CTQW", r"(b) $\omega=0.5$", r"(c) $\omega=1$: Markov walk"]
tt = np.arange(steps + 1) * dt
for ax, om, ti in zip(axes, (0.0, 0.5, 1.0), titles):
    P = runs[om][:, order].T
    im = ax.imshow(P, aspect="auto", cmap="Blues", norm=norm, interpolation="nearest",
                   extent=(0, tt[-1], n - 0.5, -0.5))
    ax.axhline(4.5, color=ORANGE, lw=0.7, ls=(0, (3, 2)))
    ax.set_title(ti, loc="left", pad=3)
    ax.set_xlabel("Time $t$", labelpad=1.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.tick_params(length=2, pad=1.5)
    for s in ax.spines.values(): s.set_linewidth(0.4)
axes[0].set_yticks(range(n)); axes[0].set_yticklabels(labels, fontsize=6.2)
fig.subplots_adjust(left=0.135, right=0.885, bottom=0.2, top=0.88, wspace=0.08)
cax = fig.add_axes([0.9, 0.2, 0.012, 0.68])
cb = fig.colorbar(im, cax=cax, ticks=[0, 0.05, 0.2, 0.5, 1])
cb.set_label(r"Population $\langle v|\rho(t)|v\rangle$", fontsize=6.5)
cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_pop.pdf")); fig.savefig(os.path.join(out, "fig_pop.png"), dpi=250)
print("fig_pop done")
