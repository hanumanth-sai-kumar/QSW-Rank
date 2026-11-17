"""Fig. 2: four-layer architecture of QSW-Rank (column width)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

setup(6.6)
fig, ax = plt.subplots(figsize=(2.95, 3.55))
ax.set_xlim(0, 10); ax.set_ylim(0, 14.2); ax.axis("off")

def rbox(x, y, w, h, fc, ec, lw=0.7, r=0.12, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))

def down(y0, y1, label=None):
    ax.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>", mutation_scale=6,
                                 color=MUTED, lw=0.8, shrinkA=0, shrinkB=0, zorder=3))
    if label:
        ax.text(5.25, (y0 + y1) / 2, label, ha="left", va="center", fontsize=5.9,
                style="italic", color=MUTED)

# input
rbox(1.2, 13.15, 7.6, 0.95, "white", "#8a8a8a")
ax.text(5, 13.62, "Network specification\nhosts, zones, services, firewall rules, CVSS v3.1",
        ha="center", va="center", fontsize=6.2, linespacing=1.2)

layers = [
    ("(i) Attack-graph generation", ["Network\ngenerator", "Vulnerability\n& CVSS sampler", "MulVAL\nreasoning"]),
    ("(ii) Walk-graph compiler", ["Fact folding\n$w_r$, Eq. (4)", "Jump rates\n$J$, Eq. (5)", "Hamiltonian\n$H$, Eq. (6)"]),
    ("(iii) Quantum stochastic walk", ["Coherent\n$-i(1-\\omega)[H,\\rho]$", "Jumps\n$\\omega\\,\\mathcal{D}[\\rho]$", "RK4 over\n$\\omega\\in\\Omega$"]),
    ("(iv) Scoring and ranking", ["Time average\n$S_\\omega(v)$", "Rate average\n$S(v)$", "Host score\n$f(h)$"]),
]
passed = ["graph $G$ and start $s$", "$W, J, d, H$, horizon $T$", "populations $\\langle v|\\rho_\\omega(t)|v\\rangle$"]
fills = ["#eef3fb", "#f6efe8", "#eef3fb", "#eef6f0"]
edges = [BLUE, ORANGE, BLUE, GREEN]
y = 12.25
tops = []
for k, (title, items) in enumerate(layers):
    h = 1.95
    y0 = y - h
    rbox(0.15, y0, 9.7, h, fills[k], edges[k], lw=0.8, r=0.2, z=1)
    ax.text(0.4, y - 0.12, title, ha="left", va="top", fontsize=6.6, fontweight="bold")
    bw, gap = 2.95, 0.2
    for j, it in enumerate(items):
        bx = 0.4 + j * (bw + gap)
        rbox(bx, y0 + 0.17, bw, 1.18, "white", "#9a9a9a", lw=0.6, r=0.1)
        ax.text(bx + bw / 2, y0 + 0.76, it, ha="center", va="center", fontsize=6.0, linespacing=1.25)
        if j < 2:
            ax.add_patch(FancyArrowPatch((bx + bw + 0.01, y0 + 0.76), (bx + bw + gap - 0.01, y0 + 0.76),
                                         arrowstyle="-|>", mutation_scale=4, color=MUTED, lw=0.6,
                                         shrinkA=0, shrinkB=0, zorder=3))
    tops.append((y, y0))
    y = y0 - 0.7

down(13.15, tops[0][0])
for k in range(3):
    down(tops[k][1], tops[k + 1][0], passed[k])
# output
oy = tops[3][1] - 1.35
rbox(1.2, oy - 0.05, 7.6, 0.8, "white", "#8a8a8a")
ax.text(5, oy + 0.35, "Ranked host list for remediation", ha="center", va="center", fontsize=6.2)
down(tops[3][1], oy + 0.75)

fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
ax.set_ylim(oy - 0.15, 14.2)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_arch.pdf")); fig.savefig(os.path.join(out, "fig_arch.png"), dpi=250)
print("fig_arch done")
