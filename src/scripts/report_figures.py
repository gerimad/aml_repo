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


def split_line(ax, label_y: float) -> None:
    """A thin vertical line where Dev B ends and the test starts, with both parts named."""
    start = pc.POINTS["test"][0]
    ax.axvline(start, color=plots.INK_SECONDARY, linewidth=0.8, linestyle=":")
    ax.text(start, label_y, "Dev B  ", ha="right", va="top", fontsize=8, color=plots.INK_SECONDARY)
    ax.text(start, label_y, "  test", ha="left", va="top", fontsize=8, color=plots.INK_SECONDARY)


def audience(out: Path) -> None:
    """WP15: share of a token's new buyers among the first 1,000 accounts, per point (mean over tokens)."""
    tokens = pd.concat([pd.read_parquet(RESULTS / "wp15" / f"tokens_{part}.parquet") for part in ("dev_b", "test")])
    per_point = tokens.groupby(["method", "point"])["recall_1000"].mean().reset_index()
    methods = {  # method in the table -> label, colour, dashed
        "raw neighbours": ("neighbour score, unscaled (chosen on Dev A)", plots.SERIES[0], False),
        "activity": ("most active accounts (baseline)", plots.SERIES[1], True),
        "model": ("blend as planned (scaled per account)", plots.SERIES[2], False),
        "random": ("random accounts", plots.INK_SECONDARY, True),
    }
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for method, (label, colour, dashed) in methods.items():
        line = per_point[per_point["method"] == method]
        ax.plot(line["point"], line["recall_1000"], marker="o", markersize=4, color=colour, linestyle="--" if dashed else "-", label=label)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("recall@1000 (buyers found)")
    split_line(ax, 0.59)
    short_dates(ax)
    ax.legend(loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "app_audience.pdf")
    plt.close(fig)


def safer(out: Path) -> None:
    """WP16: recall@10 against the share of listed tokens that collapse, over the re-ranking weight λ."""
    table = pd.read_csv(RESULTS / "wp16" / "tradeoff.csv")
    curve = table.groupby(["part", "lam"])[["recall", "collapsed_share"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    for slot, (part, label) in enumerate([("dev_b", "Dev B"), ("test", "test")]):
        line = curve[curve["part"] == part]
        ax.plot(100 * line["collapsed_share"], line["recall"], marker="o", markersize=4, color=plots.SERIES[slot], label=label)
        for _, row in line.iterrows():
            if row["lam"] in (0, 0.5, 2):
                ax.annotate(f"λ={row['lam']:g}", (100 * row["collapsed_share"], row["recall"]), textcoords="offset points",
                            xytext=(0, 6 if part == "test" else -11), ha="center", fontsize=7, color=plots.INK_SECONDARY)  # fmt: skip
    ax.set_xlabel("listed tokens that collapse (%)")
    ax.set_ylabel("recall@10")
    ax.set_xlim(left=0)
    ax.grid(True, axis="both")
    ax.legend(loc="center left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "app_safer.pdf")
    plt.close(fig)


def scam_roc(out: Path) -> None:
    """WP13: ROC for collapse on the test, baseline against baseline plus the recommender's features (fitted on Dev A)."""
    from sklearn.metrics import roc_auc_score, roc_curve

    import scam_push as sp

    windows = pd.read_parquet(RESULTS / "wp13" / "tokens.parquet")
    dev_a, test = windows[windows["part"] == "dev_a"], windows[windows["part"] == "test"]
    models = {
        "public baseline": (sp.BASELINE_FEATURES, plots.SERIES[1]),
        "baseline + recommender": (sp.BASELINE_FEATURES + sp.RECOMMENDER_FEATURES, plots.SERIES[0]),
        "recommender alone": (sp.RECOMMENDER_FEATURES, plots.SERIES[2]),
    }
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    for label, (features, colour) in models.items():
        score = sp.fit_score(dev_a, test, features)
        fpr, tpr, _ = roc_curve(test["collapsed"], score)
        ax.plot(fpr, tpr, color=colour, linewidth=1.5, label=f"{label} ({roc_auc_score(test['collapsed'], score):.2f})")
    ax.plot([0, 1], [0, 1], color=plots.GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.grid(True, axis="both")
    ax.legend(loc="lower right", fontsize=7, title="model (AUC)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "app_scam_roc.pdf")
    plt.close(fig)


def buyer_forecast(out: Path) -> None:
    """WP17: mean absolute error of the new-buyer forecast per point, without and with the list count."""
    import demand_forecast as dfc

    tokens = dfc.add_logs(pd.read_parquet(RESULTS / "wp17" / "tokens.parquet"))
    dev_a, later = tokens[tokens["part"] == "dev_a"], tokens[tokens["part"].isin(["dev_b", "test"])]
    models = {
        "recent buyers, activity, age (baseline)": (dfc.BASELINE_FEATURES, plots.SERIES[1], "--"),
        "... plus the number of lists the token is on": (dfc.WITH_LISTS, plots.SERIES[0], "-"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for label, (features, colour, style) in models.items():
        errors = dfc.fit_and_score(dev_a, later, features)
        ax.plot(errors["point"], errors["mae"], marker="o", markersize=4, color=colour, linestyle=style, label=label)
    ax.set_ylim(0.15, 0.3)
    ax.set_ylabel("mean abs. error, log(1 + N)")
    split_line(ax, 0.298)
    short_dates(ax)
    ax.legend(loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "app_buyer_forecast.pdf")
    plt.close(fig)


FIGURES = [test_per_point, neighbour_grid, half_life, refresh, outcomes_by_age, returning, audience, safer, scam_roc, buyer_forecast]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plots.style()
    for draw in FIGURES:
        draw(out)
        print(f"  {draw.__name__}: done")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
