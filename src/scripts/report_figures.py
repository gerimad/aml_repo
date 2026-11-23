"""The figures of the final report, drawn from the saved result tables (nothing is recomputed).

Usage, from the repository root:

    uv run python src/scripts/report_figures.py <output folder>

Each figure is written as a vector PDF. The numbers come from results/,
which the notebooks 10 to 15 wrote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen: write files only
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import plots
import protocol as pc

RESULTS = pc.RESULTS_DIR


def read(path: Path) -> pd.DataFrame:
    """A saved result table, with its evaluation points as timestamps."""
    table = pd.read_csv(path)
    if "point" in table.columns:
        table["point"] = pd.to_datetime(table["point"])
    return table


def short_dates(ax) -> None:
    """Date ticks as "13 Aug", one per week, so that they do not overlap."""
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def test_per_point(out: Path) -> None:
    """Recall@10 at each test point: the baseline, the two fixed blends and the primary model."""
    per_point = read(RESULTS / "test" / "per_point.csv")
    names = {
        "popularity (baseline)": "popularity (baseline)",
        "fixed blend": "fixed blend",
        "time-weighted blend": "time-weighted blend",
        "ranker: lambdamart": "ranker (LambdaMART, primary)",
    }
    lines = {label: per_point[per_point["model"] == model] for model, label in names.items()}
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    plots.per_point_lines(ax, lines)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("recall@10")
    short_dates(ax)
    ax.legend(loc="lower center", ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "test_per_point.pdf")
    plt.close(fig)


def neighbour_grid(out: Path) -> None:
    """Mean Dev A recall over k and shrinkage, at the best popularity weight of each pair."""
    parts = [read(RESULTS / "wp3" / "grid_dev_a.csv")]
    parts += [read(f) for f in sorted((RESULTS / "wp3" / "grid_dev_a_extension").glob("*.csv"))]
    grid = pd.concat(parts, ignore_index=True)
    best = grid.groupby(["k", "shrink", "weight"])["recall"].mean().groupby(["k", "shrink"]).max().unstack("shrink")
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    image = ax.imshow(best.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(best.shape[1]), [f"{s:g}" for s in best.columns])
    ax.set_yticks(range(best.shape[0]), [str(k) for k in best.index])
    ax.set_xlabel("shrinkage λ")
    ax.set_ylabel("neighbours k")
    ax.grid(False)
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            value = best.iat[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if value > best.to_numpy().mean() else plots.INK)  # fmt: skip
    fig.colorbar(image, ax=ax, label="recall@10, Dev A")
    fig.tight_layout()
    fig.savefig(out / "neighbour_grid.pdf")
    plt.close(fig)


def half_life(out: Path) -> None:
    """Mean Dev A recall over the half-life, one line per history length, at the best weight."""
    grid = pd.concat([read(f) for f in sorted((RESULTS / "wp4" / "grid_dev_a").glob("*.csv"))], ignore_index=True)
    mean = grid.groupby(["half_life", "history_days", "weight"])["recall"].mean()
    best = mean.groupby(["half_life", "history_days"]).max().unstack("history_days")
    labels = {7.0: "7 days", 14.0: "14 days", 28.0: "28 days", float("inf"): "all since 18 June"}
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = [("∞" if h == float("inf") else f"{h:g}") for h in best.index]
    for slot, length in enumerate(best.columns):
        ax.plot(x, best[length], marker="o", color=plots.SERIES[slot], label=f"history: {labels[length]}")
    ax.set_xlabel("half-life h, days")
    ax.set_ylabel("recall@10, Dev A")
    ax.legend(fontsize=7, loc="lower center")
    fig.tight_layout()
    fig.savefig(out / "half_life.pdf")
    plt.close(fig)


def refresh(out: Path) -> None:
    """Dev B recall and share of cold pairs reached, over how often the lists are rebuilt."""
    table = read(RESULTS / "wp5" / "refreshed_dev_b.csv")
    shown = {
        "baseline, refreshed": ("baseline, refreshed", plots.BASELINE_GREY, "--"),
        "time-weighted blend, refreshed": ("time-weighted blend", plots.colour("time-weighted blend"), "-"),
        "mixed list": ("blend + 1 new-token slot", plots.SERIES[4], "-"),
        "newest-popular": ("newest-popular", plots.colour("newest-popular"), "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    for ax, metric, title in [(axes[0], "recall", "recall@10"), (axes[1], "cold_reached", "share of cold pairs reached")]:
        for model, (label, colour, style) in shown.items():
            curve = table[table["model"] == model].groupby("refresh_hours")[metric].mean().sort_index(ascending=False)
            ax.plot([str(h) for h in curve.index], curve.to_numpy(), marker="o", color=colour, linestyle=style, label=label)
        ax.set_xlabel("lists rebuilt every … hours")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "refresh.pdf")
    plt.close(fig)


def outcomes_by_age(out: Path) -> None:
    """What happened to the positive test pairs of the primary model, by the token's age at t."""
    table = pd.read_csv(RESULTS / "wp7" / "test_outcome_by_token_age.csv", index_col=0)
    outcomes = {
        "found": plots.SERIES[2],
        "candidate, ranked below 10": plots.SERIES[3],
        "not a candidate": plots.SERIES[1],
        "did not exist yet": plots.BASELINE_GREY,
    }
    labels = [f"{age} ({int(n):,} pairs)" for age, n in zip(table.index, table["pairs"])]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    left = pd.Series(0.0, index=table.index)
    for outcome, colour in outcomes.items():
        values = table[outcome].fillna(0)
        ax.barh(labels, values, left=left, color=colour, label=outcome)
        left += values
    ax.invert_yaxis()
    ax.set_xlabel("share of the positive pairs")
    ax.grid(axis="x")
    ax.legend(loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.4, -0.42))
    fig.tight_layout()
    fig.savefig(out / "outcomes_by_age.pdf")
    plt.close(fig)


def returning(out: Path) -> None:
    """Share of returning pairs at every evaluation point (open point 11.2)."""
    table = read(RESULTS / "wp7" / "returning_share.csv")
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.plot(table["point"], table["returning_share"], marker="o", color=plots.SERIES[0])
    ax.set_ylabel("returning share")
    ax.set_ylim(bottom=0)
    short_dates(ax)
    fig.tight_layout()
    fig.savefig(out / "returning.pdf")
    plt.close(fig)


FIGURES = [test_per_point, neighbour_grid, half_life, refresh, outcomes_by_age, returning]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plots.style()
    for draw in FIGURES:
        draw(out)
        print(f"  {draw.__name__}: done")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
