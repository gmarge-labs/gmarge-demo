"""Every metric must land on the truth the generator planted.

Tolerances used here, and why:

* **Reconciliation and reported ROAS: 1e-6 relative against the tables.**
  These are pure aggregation -- a groupby and a division -- so they must
  reproduce the source tables to floating-point error. Anything looser would
  hide a real bug, such as summing gross where net was meant.

* **Against ``truth.json``: that file's own rounding.** It stores ROAS to four
  decimals and money to the cent, so comparisons to it use ``abs=5e-5`` and
  ``abs=0.01`` respectively. That is the tightest honest tolerance: the
  remaining difference is the truth file's rounding, not the metric's error.

* **Holdout lift and incremental ROAS: 15% relative.** These are *estimates*.
  A difference-in-differences reads the true effect off noisy data, so exact
  agreement is not available at any sample size. Measured error across the five
  holdouts runs 0.1-5.6%, so 15% leaves roughly 2.7x headroom against the worst
  case -- loose enough not to be fragile to the seed, tight enough that a
  genuinely broken estimator fails. Dropping the pre-period correction, or
  swapping test and control regions, moves these numbers far more than 15%.

* **Confidence intervals: coverage, not width.** A 90% interval has to contain
  the planted truth. That is checked on two independently generated datasets,
  so it is not an artifact of one lucky seed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gmarge import generate as gen
from gmarge import metrics as mx

LIFT_TOLERANCE = 0.15
EXACT = 1e-6          # metric vs source table
TRUTH_ROAS = 5e-5     # truth.json stores ROAS to 4 decimal places
TRUTH_MONEY = 0.01    # truth.json stores money to the cent


@pytest.fixture(scope="session")
def tables(dataset):
    return mx.load_tables(dataset.path)


@pytest.fixture(scope="session")
def metrics(tables):
    return mx.all_metrics(tables)


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------


def test_daily_reconciliation_covers_every_day(dataset, tables):
    daily = mx.daily_reconciliation(tables.orders, tables.ad_spend)

    assert len(daily) == gen.N_DAYS
    assert set(daily["date"]) == set(dataset.dates)
    assert daily["date"].is_monotonic_increasing

    # The platforms over-claim every settled day. The last two are still
    # filling in, so they understate -- the reconciliation must show that
    # rather than hide it.
    settled = daily.iloc[:-2]
    assert (settled["over_claim_ratio"] > 1.0).all(), "platforms should over-claim every settled day"
    assert daily["over_claim_ratio"].iloc[-1] < settled["over_claim_ratio"].min()


def test_daily_reconciliation_totals_match_the_tables(tables):
    daily = mx.daily_reconciliation(tables.orders, tables.ad_spend)

    assert daily["shopify_revenue"].sum() == pytest.approx(tables.orders["revenue"].sum(), rel=EXACT)
    assert daily["shopify_net_revenue"].sum() == pytest.approx(
        tables.orders["net_revenue"].sum(), rel=EXACT
    )
    assert daily["platform_attributed_revenue"].sum() == pytest.approx(
        tables.ad_spend["platform_attributed_revenue"].sum(), rel=EXACT
    )
    assert daily["ad_spend"].sum() == pytest.approx(tables.ad_spend["spend"].sum(), rel=EXACT)


def test_weekly_reconciliation_agrees_with_daily(tables):
    daily = mx.daily_reconciliation(tables.orders, tables.ad_spend)
    weekly = mx.weekly_reconciliation(tables.orders, tables.ad_spend)

    assert len(weekly) == gen.N_WEEKS
    assert (weekly["days"] == 7).all()
    assert weekly["week"].tolist() == list(range(1, gen.N_WEEKS + 1))
    for column in ("shopify_revenue", "platform_attributed_revenue", "ad_spend"):
        assert weekly[column].sum() == pytest.approx(daily[column].sum(), rel=EXACT)


def test_over_claim_ratio_matches_truth(tables, metrics):
    truth = tables.truth["attribution_gap"]

    assert metrics["headline"]["over_claim_ratio"] == pytest.approx(truth["ratio"], abs=1e-4)
    assert metrics["headline"]["over_claim_ratio"] == pytest.approx(truth["designed_ratio"], abs=1e-3)
    assert metrics["headline"]["shopify_revenue"] == pytest.approx(
        truth["shopify_revenue"], abs=TRUTH_MONEY
    )
    assert metrics["headline"]["platform_attributed_revenue"] == pytest.approx(
        truth["platform_attributed_revenue"], abs=TRUTH_MONEY
    )
    assert metrics["headline"]["over_claimed_revenue"] == pytest.approx(
        truth["platform_attributed_revenue"] - truth["shopify_revenue"], abs=2 * TRUTH_MONEY
    )


def test_the_over_claim_is_not_just_a_period_total(tables):
    """It should show up week after week, not net out to 1.4x by accident."""
    weekly = mx.weekly_reconciliation(tables.orders, tables.ad_spend)
    # The last week is short on platform data (reporting lag), so exclude it.
    settled = weekly.iloc[:-1]
    assert (settled["over_claim_ratio"] > 1.2).all()
    assert settled["over_claim_ratio"].median() == pytest.approx(1.4, abs=0.1)


# --------------------------------------------------------------------------
# Reported ROAS by channel and week
# --------------------------------------------------------------------------


def test_reported_roas_grid_is_complete(tables):
    grid = mx.reported_roas_by_channel_week(tables.ad_spend)

    assert len(grid) == gen.N_WEEKS * len(gen.CHANNELS)
    assert set(grid["channel"]) == set(gen.CHANNELS)
    assert grid["reported_roas"].notna().all()
    assert (grid["spend"] > 0).all()


def test_weekly_roas_rolls_up_to_the_channel_truth(tables):
    grid = mx.reported_roas_by_channel_week(tables.ad_spend)
    rolled = grid.groupby("channel")[["spend", "platform_attributed_revenue"]].sum()
    rolled["reported_roas"] = rolled["platform_attributed_revenue"] / rolled["spend"]

    for channel in tables.truth["channel_truth"]:
        assert rolled.loc[channel["channel"], "reported_roas"] == pytest.approx(
            channel["reported_roas"], abs=TRUTH_ROAS
        )
        assert rolled.loc[channel["channel"], "spend"] == pytest.approx(
            channel["spend"], abs=TRUTH_MONEY
        )


def test_channel_summary_matches_truth(tables):
    summary = mx.channel_summary(tables.ad_spend).set_index("channel")

    assert summary["share_of_spend"].sum() == pytest.approx(1.0, rel=EXACT)
    for channel in tables.truth["channel_truth"]:
        assert summary.loc[channel["channel"], "reported_roas"] == pytest.approx(
            channel["reported_roas"], abs=TRUTH_ROAS
        )


def test_branded_search_looks_best_on_reported_roas(tables):
    """The trap the dataset is built around, stated as a metric."""
    summary = mx.channel_summary(tables.ad_spend).set_index("channel")
    best = summary["reported_roas"].idxmax()

    assert best == "Google branded search"
    assert summary.loc[best, "reported_roas"] > summary.loc["Meta prospecting", "reported_roas"]


# --------------------------------------------------------------------------
# Geo holdouts
# --------------------------------------------------------------------------


def test_holdout_plan_hides_the_answer(tables):
    plan = mx.holdout_plan(tables.truth)

    assert len(plan) == 5
    for holdout in plan:
        assert set(holdout) == {
            "channel", "start_date", "end_date", "test_regions", "control_regions",
        }
        assert "true_incremental_revenue_lost" not in holdout
        assert "implied_true_iroas" not in holdout


def _truth_by_channel(truth):
    return {h["channel"]: h for h in truth["geo_holdouts"]}


def test_holdout_lift_matches_the_planted_truth(tables, metrics):
    expected = _truth_by_channel(tables.truth)

    for row in metrics["holdouts"].itertuples():
        true_lift = expected[row.channel]["true_incremental_revenue_lost"]
        assert row.incremental_revenue == pytest.approx(true_lift, rel=LIFT_TOLERANCE), (
            f"{row.channel}: measured {row.incremental_revenue:,.0f} vs true {true_lift:,.0f}"
        )


def test_holdout_incremental_roas_matches_the_planted_truth(tables, metrics):
    expected = _truth_by_channel(tables.truth)

    for row in metrics["holdouts"].itertuples():
        true_iroas = expected[row.channel]["implied_true_iroas"]
        assert row.incremental_roas == pytest.approx(true_iroas, rel=LIFT_TOLERANCE), (
            f"{row.channel}: measured {row.incremental_roas:.2f} vs true {true_iroas:.2f}"
        )
        designed = gen.CHANNELS[row.channel]["true_iroas"]
        assert row.incremental_roas == pytest.approx(designed, rel=LIFT_TOLERANCE)


def test_estimated_paused_spend_matches_what_was_actually_paused(tables, metrics):
    """The paused regions spent nothing, so the counterfactual spend is itself
    an estimate -- from the control regions. It should land on the truth."""
    expected = _truth_by_channel(tables.truth)

    for row in metrics["holdouts"].itertuples():
        true_spend = expected[row.channel]["paused_spend_counterfactual"]
        assert row.paused_spend_estimate == pytest.approx(true_spend, rel=LIFT_TOLERANCE)


def test_confidence_intervals_contain_the_planted_truth(tables, metrics):
    expected = _truth_by_channel(tables.truth)

    for row in metrics["holdouts"].itertuples():
        truth = expected[row.channel]
        assert row.confidence == 0.90
        assert (
            row.incremental_revenue_ci_low
            <= truth["true_incremental_revenue_lost"]
            <= row.incremental_revenue_ci_high
        ), f"{row.channel}: true lift outside its 90% interval"
        assert (
            row.incremental_roas_ci_low
            <= truth["implied_true_iroas"]
            <= row.incremental_roas_ci_high
        ), f"{row.channel}: true iROAS outside its 90% interval"


def test_intervals_bracket_their_point_estimate(metrics):
    for row in metrics["holdouts"].itertuples():
        assert row.incremental_revenue_ci_low < row.incremental_revenue < row.incremental_revenue_ci_high
        assert row.incremental_roas_ci_low < row.incremental_roas < row.incremental_roas_ci_high
        assert row.lift_pct_ci_low < row.lift_pct < row.lift_pct_ci_high
        # Wide enough to be honest about five markets, not so wide it says nothing.
        width = row.incremental_revenue_ci_high - row.incremental_revenue_ci_low
        assert 0 < width < row.incremental_revenue


def test_confidence_intervals_cover_truth_on_a_second_dataset(tmp_path):
    """Coverage should not be an artifact of one generator seed."""
    gen.generate(tmp_path, seed=gen.SEED + 1)
    tables = mx.load_tables(tmp_path)
    holdouts = mx.all_metrics(tables)["holdouts"]
    expected = _truth_by_channel(tables.truth)

    for row in holdouts.itertuples():
        truth = expected[row.channel]
        assert row.incremental_revenue == pytest.approx(
            truth["true_incremental_revenue_lost"], rel=LIFT_TOLERANCE
        )
        assert (
            row.incremental_revenue_ci_low
            <= truth["true_incremental_revenue_lost"]
            <= row.incremental_revenue_ci_high
        ), f"{row.channel}: true lift outside its interval on seed {gen.SEED + 1}"


def test_bootstrap_is_deterministic(tables):
    plan = mx.holdout_plan(tables.truth)
    first = mx.holdout_results(tables.orders, tables.ad_spend, plan)
    second = mx.holdout_results(tables.orders, tables.ad_spend, plan)

    pd.testing.assert_frame_equal(first, second)


def test_holdouts_recover_the_incrementality_ranking(metrics, tables):
    """The whole point: reported ROAS ranks the channels wrongly."""
    holdouts = metrics["holdouts"].set_index("channel")
    over_claim = holdouts["over_claim_multiple"]

    assert set(over_claim.nlargest(2).index) == {"Google branded search", "Meta retargeting"}
    assert over_claim.idxmin() == "Meta prospecting"
    assert (
        set(over_claim.nlargest(2).index)
        == set(tables.truth["incrementality_ranking"]["most_overstated"])
    )
    assert over_claim.idxmin() == tables.truth["incrementality_ranking"]["closest_to_reported"]


def test_every_channel_is_worth_less_than_the_platform_says(metrics):
    for row in metrics["holdouts"].itertuples():
        assert row.incremental_roas < row.reported_roas_in_window
        assert row.over_claim_multiple > 1.0


# --------------------------------------------------------------------------
# The one function that returns everything
# --------------------------------------------------------------------------


def test_all_metrics_returns_the_full_set(metrics):
    assert set(metrics) == {
        "brand", "disclaimer", "window", "headline",
        "daily_reconciliation", "weekly_reconciliation",
        "reported_roas_by_channel_week", "channel_summary", "holdouts",
    }
    for key in (
        "daily_reconciliation", "weekly_reconciliation",
        "reported_roas_by_channel_week", "channel_summary", "holdouts",
    ):
        assert isinstance(metrics[key], pd.DataFrame) and not metrics[key].empty

    assert metrics["window"] == {
        "start": "2025-01-06", "end": "2025-07-06", "n_days": 182, "n_weeks": 26,
    }
    assert "fictional" in metrics["disclaimer"].lower()


def test_headline_is_internally_consistent(metrics):
    head = metrics["headline"]

    assert head["over_claim_ratio"] == pytest.approx(
        head["platform_attributed_revenue"] / head["shopify_revenue"], rel=EXACT
    )
    assert head["blended_reported_roas"] == pytest.approx(
        head["platform_attributed_revenue"] / head["ad_spend"], rel=EXACT
    )
    assert head["average_order_value"] == pytest.approx(
        head["shopify_revenue"] / head["n_orders"], rel=EXACT
    )
    assert head["shopify_net_revenue"] < head["shopify_revenue"]
    assert head["measured_incremental_roas"] < head["blended_reported_roas"]


def test_all_metrics_accepts_a_directory_or_loaded_tables(dataset, tables):
    from_dir = mx.all_metrics(dataset.path)
    from_tables = mx.all_metrics(tables)
    pd.testing.assert_frame_equal(from_dir["holdouts"], from_tables["holdouts"])
