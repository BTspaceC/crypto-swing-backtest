"""Shared chart style: one fixed colour per coin, recessive axes, CJK-capable fonts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYMS = ["BTC", "ETH", "BNB"]
COLOR = {"BTC": "#2a78d6", "ETH": "#eb6834", "BNB": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#6f6e6a", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
    "font.family": "sans-serif", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
