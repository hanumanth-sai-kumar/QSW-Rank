"""Fig. 3: ranking quality vs mixing rate (stealthy attacker, 40 networks), column width.

Values are the means printed in the Kaggle logs:
  QSW curve          -> aba74914 ("Decoherence curve", B10)
  classical hopping  -> ff28785a ("Classical mixture (sq)", B10)
  QSW-Rank average   -> 1bd9051b (98.1 / 80.0)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from matplotlib.lines import Line2D

om = [round(0.1 * i, 1) for i in range(11)]
qsw_n = [96.97, 98.31, 98.05, 98.22, 96.97, 95.82, 95.27, 95.08, 95.11, 94.45, 92.67]
qsw_r = [70.94, 75.19, 80.10, 80.27, 80.24, 80.39, 80.05, 79.82, 79.70, 79.34, 78.99]
cls_n = [97.67, 97.01, 97.33, 96.63, 95.88, 95.43, 95.25, 95.44, 95.40, 94.65, 92.67]
cls_r = [71.19, 79.60, 79.67, 79.64, 79.78, 79.99, 80.25, 79.62, 79.49, 79.10, 78.83]
avg_n, avg_r = 98.1, 80.0

setup(7)
fig, axes = plt.subplots(1, 2, figsize=(2.95, 1.72))
panels = [(axes[0], qsw_n, cls_n, avg_n, "(a) NDCG@10 (%)", (91.5, 99.2)),
          (axes[1], qsw_r, cls_r, avg_r, r"(b) $r_s$ (%)", (69.0, 82.5))]
for ax, q, c, a, title, ylim in panels:
    ax.axhline(a, color=MUTED, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
    ax.plot(om, c, color=ORANGE, lw=1.1, ls=(0, (4, 2)), marker="s", ms=2.3, zorder=2)
    ax.plot(om, q, color=BLUE, lw=1.3, marker="o", ms=2.5, zorder=3)
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(*ylim)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel(r"Mixing rate $\omega$", labelpad=1.5)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2, pad=1.5)

handles = [Line2D([], [], color=BLUE, lw=1.3, marker="o", ms=2.5, label="QSW at rate $\\omega$"),
           Line2D([], [], color=ORANGE, lw=1.1, ls=(0, (4, 2)), marker="s", ms=2.3, label="Classical hopping"),
           Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 1.5)), label="QSW-Rank (average)")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01),
           handlelength=2.0, columnspacing=0.9, handletextpad=0.4, fontsize=6.3)
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.19, top=0.72, wspace=0.34)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_omega.pdf")); fig.savefig(os.path.join(out, "fig_omega.png"), dpi=250)
print("fig_omega done")
