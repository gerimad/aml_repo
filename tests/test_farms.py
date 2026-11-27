"""Tests for src/scripts/farms.py (WP10), on small hand-made markets."""

import itertools

import numpy as np
import pandas as pd

import farms as fm
import protocol as pc

T = pd.Timestamp("2026-07-02", tz="UTC")
MIN = pd.Timedelta(minutes=1)
DAY = pd.Timedelta(days=1)


def legs(rows):
    """Legs from (time, user, token) triples."""
    return pd.DataFrame([{"timestamp": when, "user": u, "token": i} for when, u, i in rows])


def first_of(rows):
    """First trades in [T, T + 30 days) of a market given as triples."""
    return fm.first_trades(legs(rows), T, T + 30 * DAY)


# Accounts 1 and 2 first-trade tokens 10, 11 and 12, each time 2 minutes apart, hours apart between tokens.
TWINS = [(T + h * 60 * MIN, 1, tok) for h, tok in enumerate([10, 11, 12])]
TWINS += [(T + h * 60 * MIN + 2 * MIN, 2, tok) for h, tok in enumerate([10, 11, 12])]


def test_three_shared_trades_two_minutes_apart_link_for_delta_5_not_for_delta_1():
    first = first_of(TWINS)
    linked = fm.coaction_edges(first, delta_minutes=5, min_shared=3)
    assert linked[["user_a", "user_b", "weight"]].values.tolist() == [[1, 2, 3]]
    assert fm.coaction_edges(first, delta_minutes=1, min_shared=3).empty


def test_a_single_shared_trade_never_links():
    first = first_of([(T, 1, 10), (T + MIN, 2, 10)])
    assert fm.coaction_edges(first, delta_minutes=5, min_shared=2).empty


def test_only_the_first_trade_of_a_token_counts():
    # Account 1 traded token 10 before the window; its later trade close to account 2's is not a first trade.
    rows = [(T - DAY, 1, 10), (T + MIN, 1, 10), (T + MIN, 2, 10)]
    first = fm.first_trades(legs(rows), T, T + DAY)
    assert first["user"].tolist() == [2]


def test_the_sweep_gives_the_same_edges_as_a_pair_loop():
    rng = np.random.default_rng(1)
    rows = [(T + int(rng.integers(0, 600)) * MIN, int(u), int(tok)) for u, tok in zip(rng.integers(0, 15, 300), rng.integers(0, 8, 300))]
    first = first_of(rows)
    for delta, m in [(5, 1), (30, 2), (120, 3)]:
        brute = {}
        for _, token_rows in first.groupby("token"):
            for (_, x), (_, y) in itertools.combinations(token_rows.iterrows(), 2):
                if abs((x["timestamp"] - y["timestamp"]).total_seconds()) <= delta * 60:
                    pair = (min(x["user"], y["user"]), max(x["user"], y["user"]))
                    brute[pair] = brute.get(pair, 0) + 1
        expected = sorted((a, b, w) for (a, b), w in brute.items() if w >= m)
        got = sorted(map(tuple, fm.coaction_edges(first, delta, m)[["user_a", "user_b", "weight"]].values.tolist()))
        assert got == expected, f"delta {delta}, m {m}"


def test_groups_are_components_and_large_ones_are_split():
    # Two cliques of 6 joined by one weak edge: one component; split when the size limit is 6.
    clique = lambda users: [(a, b, 5) for a, b in itertools.combinations(users, 2)]
    edges = pd.DataFrame(clique(range(0, 6)) + clique(range(10, 16)) + [(5, 10, 1)], columns=["user_a", "user_b", "weight"])
    whole = fm.groups(edges, max_component=100)
    assert whole.nunique() == 1
    split = fm.groups(edges, max_component=6)
    assert split.nunique() == 2
    assert split.loc[0] == split.loc[3] and split.loc[10] == split.loc[14] and split.loc[0] != split.loc[10]


def test_the_coaction_rate_counts_linked_pairs_of_active_members():
    group_of = pd.Series({1: 0, 2: 0, 3: 0, 4: 0}, name="group")
    # Later: 1 and 2 first-trade token 20 a minute apart; 3 trades token 21 alone; 4 is inactive.
    later = fm.first_trades(legs([(T, 1, 20), (T + MIN, 2, 20), (T + 5 * 60 * MIN, 3, 21)]), T, T + DAY)
    rate = fm.coaction_rate(group_of, later, delta_minutes=5).set_index("group").loc[0]
    assert (rate["active"], rate["pairs"], rate["linked"]) == (3, 3, 1)
    assert rate["rate"] == 1 / 3


def test_matched_pairs_have_the_members_trade_counts_and_days_and_are_not_members():
    rng = np.random.default_rng(0)
    build_rows = []
    for u in range(100):  # account u first-trades (u % 4) + 1 tokens on day u % 3
        build_rows += [(T + (u % 3) * DAY + k * MIN, u, 1000 + u * 10 + k) for k in range(u % 4 + 1)]
    build = fm.first_trades(legs(build_rows), T, T + 10 * DAY)
    later = fm.first_trades(legs([(T + 20 * DAY + int(rng.integers(0, 50)) * MIN, u, 5) for u in range(100)]), T + 15 * DAY, T + 30 * DAY)
    group_of = pd.Series({u: 0 for u in range(0, 24, 2)}, name="group")
    matched = fm.matched_pairs(group_of, build, later, pairs_per_group=50)
    stratum = fm.strata(build)
    assert len(matched) > 0
    for side in ("a", "b"):
        assert (stratum.loc[matched[f"user_{side}"]].to_numpy() == stratum.loc[matched[f"member_{side}"]].to_numpy()).all()
        assert not matched[f"user_{side}"].isin(group_of.index).any()


def test_identical_set_members_label_users_with_the_same_window():
    rows = [(T - DAY, u, tok) for u in range(25) for tok in (10, 11)] + [(T - DAY, 99, 10)]
    rows += [(T + DAY, u, 12) for u in list(range(25)) + [99]]
    members = fm.identical_set_members(pc.make_task(legs(rows), T), min_size=20)
    assert sorted(members.index) == list(range(25)) and members.nunique() == 1


def test_the_choice_takes_the_passing_setting_with_most_accounts():
    grid = pd.DataFrame({"delta": [5, 30, 120, 5], "min_shared": [5, 5, 2, 3],
                         "accounts": [700, 900, 9000, 900], "success": [True, True, False, True]})  # fmt: skip
    chosen = fm.choose_setting(grid)
    assert (chosen["delta"], chosen["min_shared"]) == (30, 5), "900 accounts twice: the larger m wins"
    assert fm.choose_setting(grid.assign(success=False)) is None
