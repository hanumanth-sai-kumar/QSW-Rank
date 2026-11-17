"""Shared style for all QSW-Rank paper figures (matplotlib only)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"      # coherent / quantum
ORANGE = "#d35400"    # directed jumps / classical
GREEN = "#1e8449"     # privileges
RED = "#c0392b"       # attacker
INK = "#1a1a1a"
MUTED = "#55534f"
GRID = "#e4e3df"
STEPFILL = "#e6e6e6"
PRIVFILL = "#e3f1e8"
ZONEFILL = "#f4f4f2"

def setup(size=7.5):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": size, "axes.labelsize": size,
        "axes.titlesize": size, "xtick.labelsize": size - 0.5, "ytick.labelsize": size - 0.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "text.color": INK,
        "axes.labelcolor": INK, "pdf.fonttype": 42, "legend.frameon": False,
    })
