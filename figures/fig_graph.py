"""Fig. 1: illustrative two-host MulVAL attack graph and the QSW-Rank walk on it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

setup(7.2)
fig, ax = plt.subplots(figsize=(6.6, 2.55))
ax.set_xlim(-1.2, 25.7); ax.set_ylim(-5.3, 2.9); ax.axis("off")

def box(x, y, w, h, text, kind):
    style = {"start": dict(fc="#f8e3e0", ec=RED, lw=1.1, ls="-", r=0.15, bold=False),
             "step": dict(fc=STEPFILL, ec="#555555", lw=0.8, ls="-", r=0.06, bold=True),
             "priv": dict(fc=PRIVFILL, ec=GREEN, lw=1.1, ls="-", r=0.45, bold=False),
             "fact": dict(fc="white", ec="#8a8a8a", lw=0.8, ls=(0, (3, 2)), r=0.06, bold=False)}[kind]
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={style['r']}",
                       fc=style["fc"], ec=style["ec"], lw=style["lw"], ls=style["ls"], zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=6.2, zorder=4,
            fontweight="bold" if style["bold"] else "normal", linespacing=1.15)
    return (x, y, w, h)

# host zones
for x0, x1, lab in [(1.6, 13.45, "Host web0 (DMZ)"), (13.6, 25.5, "Host db0 (server zone)")]:
    ax.add_patch(FancyBboxPatch((x0, -3.95), x1 - x0, 6.65, boxstyle="round,pad=0,rounding_size=0.25",
                                fc=ZONEFILL, ec="#d4d4d0", lw=0.6, zorder=0))
    ax.text(x0 + 0.2, 2.45, lab, ha="left", va="top", fontsize=7, fontweight="bold", zorder=1)

W, Hh = 2.55, 1.05
N = {
    "s": box(0, 0, 1.95, Hh, "Attacker on\nthe internet", "start"),
    "r1": box(3.0, 0, W, Hh, "Direct network\naccess", "step"),
    "p1": box(6.0, 0, W, Hh, "Network access\nto web0:80", "priv"),
    "r2": box(9.0, 0, W, Hh, "Exploit Apache\n(CVE-X)", "step"),
    "p2": box(12.0, 0, W, Hh, "Code execution\non web0", "priv"),
    "r3": box(15.0, 0, W, Hh, "Multi-hop\naccess", "step"),
    "p3": box(18.0, 0, W, Hh, "Network access\nto db0:3306", "priv"),
    "r4": box(21.0, 0, W, Hh, "Exploit MySQL\n(CVE-Z)", "step"),
    "p4": box(24.0, 0, W, Hh, "Code execution\non db0", "priv"),
    "r5": box(18.0, -1.95, 2.8, Hh, "Log in with\nreused password", "step"),
}
F = {
    "f1": box(3.0, 1.5, 2.75, 0.78, "Firewall: internet\n→ web0:80 open", "fact"),
    "f2": box(9.0, 1.5, 2.75, 0.78, "web0 runs Apache\nwith CVE-X", "fact"),
    "f3": box(15.0, 1.5, 2.75, 0.78, "Firewall: web0\n→ db0:3306 open", "fact"),
    "f4": box(21.0, 1.5, 2.75, 0.78, "db0 runs MySQL\nwith CVE-Z", "fact"),
    "f5": box(18.0, -3.3, 2.95, 0.78, "Same password\non web0 and db0", "fact"),
}
def arrow(p, q, color, ls="-", lw=1.0, style="-|>", ms=6, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))
# facts feed their steps (dotted)
for f, r in [("f1", "r1"), ("f2", "r2"), ("f3", "r3"), ("f4", "r4")]:
    (fx, fy, fw, fh), (rx, ry, rw, rh) = F[f], N[r]
    arrow((fx, fy - fh / 2), (rx, ry + rh / 2), "#8a8a8a", ls=(0, (1, 1.5)), lw=0.8, ms=5)
(fx, fy, fw, fh), (rx, ry, rw, rh) = F["f5"], N["r5"]
arrow((fx, fy + fh / 2), (rx, ry - rh / 2), "#8a8a8a", ls=(0, (1, 1.5)), lw=0.8, ms=5)
# walk edges: jump (solid orange, above) and coherent hopping (dashed blue, both ways, below)
chain = ["s", "r1", "p1", "r2", "p2", "r3", "p3", "r4", "p4"]
for a, b in zip(chain, chain[1:]):
    (ax_, ay, aw, ah), (bx, by, bw, bh) = N[a], N[b]
    x1, x2 = ax_ + aw / 2, bx - bw / 2
    arrow((x1, 0.17), (x2, 0.17), ORANGE, lw=1.1)
    arrow((x1, -0.2), (x2, -0.2), BLUE, ls=(0, (3, 2)), lw=0.8, style="<|-|>", ms=5)
# elbows p2 -> r5 -> p4
p2x, r5x, r5y, p4x = 12.0, 18.0, -1.95, 24.0
ax.plot([p2x + 0.15, p2x + 0.15, r5x - 1.4], [-0.53, r5y + 0.17, r5y + 0.17], color=ORANGE, lw=1.1, zorder=2)
arrow((r5x - 1.7, r5y + 0.17), (r5x - 1.4, r5y + 0.17), ORANGE, lw=1.1)
ax.plot([p2x - 0.2, p2x - 0.2, r5x - 1.4], [-0.53, r5y - 0.2, r5y - 0.2], color=BLUE, lw=0.8, ls=(0, (3, 2)), zorder=2)
ax.plot([r5x + 1.4, p4x - 0.15, p4x - 0.15], [r5y + 0.17, r5y + 0.17, -0.53], color=ORANGE, lw=1.1, zorder=2)
arrow((p4x - 0.15, -0.8), (p4x - 0.15, -0.53), ORANGE, lw=1.1)
ax.plot([r5x + 1.4, p4x + 0.2, p4x + 0.2], [r5y - 0.2, r5y - 0.2, -0.53], color=BLUE, lw=0.8, ls=(0, (3, 2)), zorder=2)
# step success probabilities on the edges into steps
for x, y, t in [(7.5, 0.45, "0.57"), (13.5, 0.45, "0.5"), (19.5, 0.45, "0.42"), (15.0, -1.55, "0.5")]:
    ax.text(x, y, t, ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=ORANGE, zorder=5)
# legend
ly = -4.75
box(0.4, ly, 1.2, 0.6, "fact", "fact"); ax.text(1.15, ly, "holds before the attack", va="center", fontsize=6.6)
box(6.6, ly, 1.2, 0.6, "step", "step"); ax.text(7.35, ly, "needs all its inputs", va="center", fontsize=6.6)
box(12.2, ly, 1.7, 0.6, "privilege", "priv"); ax.text(13.2, ly, "needs any one step", va="center", fontsize=6.6)
arrow((17.2, ly), (18.2, ly), ORANGE, lw=1.1); ax.text(18.35, ly, "directed jump", va="center", fontsize=6.6)
arrow((21.2, ly), (22.2, ly), BLUE, ls=(0, (3, 2)), lw=0.8, style="<|-|>", ms=5); ax.text(22.35, ly, "coherent hopping", va="center", fontsize=6.6)
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_graph.pdf")); fig.savefig(os.path.join(out, "fig_graph.png"), dpi=220)
print("fig_graph done")
