"""One plotting style for all project notebooks (matplotlib).

Colours follow the entity, never its rank: a model keeps its colour in every
chart of every notebook. The palette is a validated categorical palette
(colour-vision-deficiency safe for neighbouring slots); the baseline is a
neutral grey so that every other model reads as "compared with it". Two
colours are lighter than 3:1 against the background, so every chart is
shown next to the table it is drawn from.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
BASELINE_GREY = "#8a8984"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]  # blue, orange, aqua, yellow, violet

MODEL_COLOURS = {
    "popularity (baseline)": BASELINE_GREY,
    "recent popularity": SERIES[4],
    "fixed blend": SERIES[0],
    "time-weighted blend": SERIES[2],
    "ranker": SERIES[1],
    "newest-popular": SERIES[3],
}

PERIOD_COLOURS = {"burn-in": "#efeeea", "Dev A": "#dfe9f7", "Dev B": "#dcf1e8", "Test": "#fbe3d8"}


def colour(name: str, fallback_slot: int = 0) -> str:
    """The colour of a model by name; any "ranker ..." name gets the ranker colour; unknown names get a series slot."""
    if name.startswith("ranker"):
        return MODEL_COLOURS["ranker"]
    return MODEL_COLOURS.get(name, SERIES[fallback_slot % len(SERIES)])


def style():
    """Set quiet defaults: light surface, thin recessive grid, no top/right spines."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "lines.markersize": 5,
        "figure.dpi": 110,
    })  # fmt: skip


def shade_periods(ax, periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]], label_y: float = 1.0):
    """Shade the calendar periods behind a time axis and name each at the top."""
    for name, (start, end) in periods.items():
        ax.axvspan(start, end, color=PERIOD_COLOURS.get(name, GRID), zorder=0, linewidth=0)
        ax.text(start + (end - start) / 2, label_y, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)  # fmt: skip


def per_point_lines(ax, tables: dict[str, pd.DataFrame], metric: str = "recall"):
    """One line per model over evaluation points, with markers, the baseline dashed, and a legend."""
    for slot, (name, table) in enumerate(tables.items()):
        dashed = "baseline" in name
        ax.plot(table["point"], table[metric], marker="o", color=colour(name, slot),
                linestyle="--" if dashed else "-", label=name, zorder=3)  # fmt: skip
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", ncols=min(len(tables), 4))
