"""Tests for src/scripts/plots.py."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import plots


def test_a_model_keeps_its_colour_and_unknown_models_get_a_fixed_slot():
    assert plots.colour("popularity (baseline)") == plots.BASELINE_GREY
    assert plots.colour("something new", fallback_slot=1) == plots.SERIES[1]
    assert plots.colour("something new", fallback_slot=1) == plots.colour("other new", fallback_slot=1)


def test_style_and_drawing_helpers_run():
    plots.style()
    fig, ax = plt.subplots()
    t = pd.Timestamp("2026-07-02", tz="UTC")
    plots.shade_periods(ax, {"Dev A": (t, t + pd.Timedelta(days=4))})
    table = pd.DataFrame({"point": [t, t + pd.Timedelta(days=2)], "recall": [0.1, 0.2]})
    plots.per_point_lines(ax, {"popularity (baseline)": table, "fixed blend": table})
    assert len(ax.lines) == 2 and ax.lines[0].get_linestyle() == "--"
    plt.close(fig)
