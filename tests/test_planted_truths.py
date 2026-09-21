"""Every truth recorded in truth.json must actually be present in the data.

These tests are written the way an analyst would look for each effect, not by
reading the generator's internals back out -- the point is that the planted
truth is *recoverable* from the tables alone.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gmarge import generate as gen

# --------------------------------------------------------------------------
# 1. The attribution gap
# --------------------------------------------------------------------------


def test_platform_attributed_revenue_is_about_1_4x_shopify(dataset):
    shopify = dataset.orders["revenue"].sum()
    attributed = dataset.ad_spend["platform_attributed_revenue"].sum()
    ratio = attributed / shopify

    assert 1.35 < ratio < 1.45, f"attribution ratio {ratio:.3f} is not ~1.4x"
    assert ratio == pytest.approx(dataset.truth["attribution_gap"]["ratio"], abs=1e-3)
    assert dataset.truth["attribution_gap"]["designed_ratio"] == 1.40


# --------------------------------------------------------------------------
# 2. True incremental ROAS vs reported ROAS
# --------------------------------------------------------------------------


def test_every_channel_overstates_its_roas(dataset):
    for c in dataset.truth["channel_truth"]:
        assert c["true_incremental_roas"] < c["reported_roas"], c["channel"]


def test_branded_search_and_retargeting_are_the_most_overstated(dataset):
    by_channel = {c["channel"]: c["true_over_reported"] for c in dataset.truth["channel_truth"]}

    for channel in ("Google branded search", "Meta retargeting"):
        assert by_channel[channel] < 0.25, f"{channel} should be far below reported"

    worst = sorted(by_channel, key=by_channel.get)[:2]
    assert set(worst) == {"Google branded search", "Meta retargeting"}
    assert set(dataset.truth["incrementality_ranking"]["most_overstated"]) == set(worst)


def test_meta_prospecting_is_closest_to_reported(dataset):
    by_channel = {c["channel"]: c["true_over_reported"] for c in dataset.truth["channel_truth"]}
    best = max(by_channel, key=by_channel.get)

    assert best == "Meta prospecting"
    assert by_channel[best] > 0.60
    assert dataset.truth["incrementality_ranking"]["closest_to_reported"] == "Meta prospecting"


def test_reported_roas_in_truth_matches_the_ad_spend_table(dataset):
    a = dataset.ad_spend
    for c in dataset.truth["channel_truth"]:
        sub = a[a["channel"] == c["channel"]]
        computed = sub["platform_attributed_revenue"].sum() / sub["spend"].sum()
        assert computed == pytest.approx(c["reported_roas"], abs=1e-3)
        assert computed == pytest.approx(c["designed_reported_roas"], rel=0.05)


# --------------------------------------------------------------------------
# 3. Geo holdouts
# --------------------------------------------------------------------------


def _matched_market_lift(orders: pd.DataFrame, holdout: dict) -> float:
    """Recover the revenue the holdout cost, by matched-market difference-in-
    differences: each test region is predicted from its paired control region
    using their ratio over the four weeks before the test."""
    rev = orders.groupby(["date", "region"])["revenue"].sum().unstack()
    start, end = pd.Timestamp(holdout["start_date"]), pd.Timestamp(holdout["end_date"])
    pre_start, pre_end = start - pd.Timedelta(days=28), start - pd.Timedelta(days=1)

    lift = 0.0
    for test, control in zip(holdout["test_regions"], holdout["control_regions"]):
        ratio = rev.loc[pre_start:pre_end, test].sum() / rev.loc[pre_start:pre_end, control].sum()
        counterfactual = rev.loc[start:end, control].sum() * ratio
        lift += counterfactual - rev.loc[start:end, test].sum()
    return lift


def test_there_is_one_holdout_per_paid_channel(dataset):
    holdouts = dataset.truth["geo_holdouts"]
    assert len(holdouts) == 5
    assert {h["channel"] for h in holdouts} == set(gen.CHANNELS)


def test_holdouts_are_four_weeks_and_never_overlap(dataset):
    windows = []
    for h in dataset.truth["geo_holdouts"]:
        start, end = pd.Timestamp(h["start_date"]), pd.Timestamp(h["end_date"])
        assert (end - start).days + 1 == 28, h["channel"]
        assert len(h["test_regions"]) == 5 and len(h["control_regions"]) == 5
        assert not set(h["test_regions"]) & set(h["control_regions"])
        windows.append((start, end))

    windows.sort()
    for (_, earlier_end), (later_start, _) in zip(windows, windows[1:]):
        assert earlier_end < later_start, "holdouts must run at different times"


def test_spend_is_actually_paused_in_the_test_regions(dataset):
    a = dataset.ad_spend
    for h in dataset.truth["geo_holdouts"]:
        window = a["date"].between(pd.Timestamp(h["start_date"]), pd.Timestamp(h["end_date"]))
        channel = a["channel"] == h["channel"]

        paused = a[channel & window & a["region"].isin(h["test_regions"])]
        control = a[channel & window & a["region"].isin(h["control_regions"])]
        outside = a[channel & ~window & a["region"].isin(h["test_regions"])]

        assert paused["spend"].sum() == 0, f"{h['channel']} test regions still spent"
        assert control["spend"].sum() > 0, f"{h['channel']} control regions went dark too"
        assert outside["spend"].sum() > 0, f"{h['channel']} test regions never resumed"

        # Only the channel under test goes dark.
        others = a[~channel & window & a["region"].isin(h["test_regions"])]
        assert others["spend"].sum() > 0


def test_holdouts_recover_their_known_true_lift(dataset):
    for h in dataset.truth["geo_holdouts"]:
        measured = _matched_market_lift(dataset.orders, h)
        true = h["true_incremental_revenue_lost"]

        assert true > 0
        assert measured == pytest.approx(true, rel=0.15), (
            f"{h['channel']}: matched-market lift {measured:,.0f} vs true {true:,.0f}"
        )


def test_holdout_lift_implies_the_channels_true_iroas(dataset):
    designed = {name: cfg["true_iroas"] for name, cfg in gen.CHANNELS.items()}
    for h in dataset.truth["geo_holdouts"]:
        assert h["implied_true_iroas"] == pytest.approx(designed[h["channel"]], rel=0.02)


# --------------------------------------------------------------------------
# 4. Week 20: creative fatigue on one Meta prospecting ad set
# --------------------------------------------------------------------------


def test_one_ad_set_doubles_its_frequency_in_week_20(dataset):
    fatigue = dataset.truth["anomalies"]["creative_fatigue"]
    assert fatigue["week"] == 20

    a = dataset.ad_spend
    ad_set = a[a["ad_set"] == fatigue["ad_set"]]
    this_week = ad_set[ad_set["date"].between(*dataset.week(20))]["frequency"].mean()
    prior_week = ad_set[ad_set["date"].between(*dataset.week(19))]["frequency"].mean()

    assert this_week / prior_week > 1.8, "frequency should roughly double"
    assert fatigue["frequency_multiple"] == pytest.approx(this_week / prior_week, rel=0.01)


def test_that_ad_sets_roas_drops_sharply_in_week_20(dataset):
    fatigue = dataset.truth["anomalies"]["creative_fatigue"]
    a = dataset.ad_spend
    ad_set = a[a["ad_set"] == fatigue["ad_set"]]

    def roas(frame):
        return frame["platform_attributed_revenue"].sum() / frame["spend"].sum()

    this_week = roas(ad_set[ad_set["date"].between(*dataset.week(20))])
    prior_week = roas(ad_set[ad_set["date"].between(*dataset.week(19))])

    assert this_week < 0.6 * prior_week, "ROAS should fall sharply"
    assert fatigue["reported_roas_fatigue_week"] == pytest.approx(this_week, rel=0.01)


def test_the_other_meta_prospecting_ad_sets_are_unaffected(dataset):
    fatigue = dataset.truth["anomalies"]["creative_fatigue"]
    a = dataset.ad_spend
    others = a[(a["channel"] == "Meta prospecting") & (a["ad_set"] != fatigue["ad_set"])]

    this_week = others[others["date"].between(*dataset.week(20))]["frequency"].mean()
    prior_week = others[others["date"].between(*dataset.week(19))]["frequency"].mean()
    assert this_week == pytest.approx(prior_week, rel=0.10)


# --------------------------------------------------------------------------
# 5. Week 12: the Meta pixel double-counts purchases for three days
# --------------------------------------------------------------------------


def test_meta_attributed_revenue_doubles_for_three_days_in_week_12(dataset):
    bug = dataset.truth["anomalies"]["meta_pixel_double_count"]
    assert bug["week"] == 12
    assert len(bug["dates"]) == 3

    affected = [pd.Timestamp(d) for d in bug["dates"]]
    week_start, week_end = dataset.week(12)
    assert all(week_start <= d <= week_end for d in affected)

    a = dataset.ad_spend
    meta = a[a["channel"].isin(bug["channels"])]
    daily_roas = meta.groupby("date").apply(
        lambda d: d["platform_attributed_revenue"].sum() / d["spend"].sum(), include_groups=False
    )

    normal = daily_roas.drop(affected).median()
    for day in affected:
        assert daily_roas[day] > 1.8 * normal, f"{day.date()} does not look double-counted"


def test_the_double_count_touches_nothing_else(dataset):
    bug = dataset.truth["anomalies"]["meta_pixel_double_count"]
    affected = [pd.Timestamp(d) for d in bug["dates"]]
    a = dataset.ad_spend

    # Non-Meta channels are clean.
    other = a[~a["channel"].isin(bug["channels"])]
    daily = other.groupby("date").apply(
        lambda d: d["platform_attributed_revenue"].sum() / d["spend"].sum(), include_groups=False
    )
    for day in affected:
        assert daily[day] == pytest.approx(daily.drop(affected).median(), rel=0.10)

    # Meta spend and clicks are unaffected -- only the revenue column is wrong.
    meta = a[a["channel"].isin(bug["channels"])]
    spend = meta.groupby("date")["spend"].sum()
    for day in affected:
        assert spend[day] == pytest.approx(spend.drop(affected).median(), rel=0.15)

    # And the store's own revenue never saw it.
    revenue = dataset.orders.groupby("date")["revenue"].sum()
    for day in affected:
        assert revenue[day] == pytest.approx(revenue.drop(affected).median(), rel=0.20)


# --------------------------------------------------------------------------
# 6. Week 8: two days missing from GA4
# --------------------------------------------------------------------------


def test_two_days_of_week_8_are_missing_from_ga4(dataset):
    missing = dataset.truth["anomalies"]["ga4_missing_days"]
    assert missing["week"] == 8
    assert len(missing["dates"]) == 2

    gone = [pd.Timestamp(d) for d in missing["dates"]]
    week_start, week_end = dataset.week(8)
    assert all(week_start <= d <= week_end for d in gone)

    present = set(dataset.ga4["date"])
    assert not (set(gone) & present), "the missing days should have no GA4 rows at all"
    assert len(present) == gen.N_DAYS - 2


def test_the_missing_days_still_exist_everywhere_else(dataset):
    gone = [pd.Timestamp(d) for d in dataset.truth["anomalies"]["ga4_missing_days"]["dates"]]

    assert set(gone) <= set(dataset.orders["date"])
    assert set(gone) <= set(dataset.ad_spend["date"])

    revenue = dataset.orders.groupby("date")["revenue"].sum()
    for day in gone:
        assert revenue[day] == pytest.approx(revenue.median(), rel=0.20)


# --------------------------------------------------------------------------
# 7. The last two days are still filling in
# --------------------------------------------------------------------------


def test_platform_numbers_lag_on_the_final_two_days(dataset):
    lag = dataset.truth["anomalies"]["reporting_lag"]
    last_two = [pd.Timestamp(d) for d in lag["dates"]]
    assert last_two == sorted(last_two)
    assert last_two[-1] == dataset.dates[-1]

    spend = dataset.ad_spend.groupby("date")["spend"].sum()
    settled = spend.iloc[-30:-2].median()

    assert spend[last_two[0]] < 0.95 * settled
    assert spend[last_two[1]] < 0.65 * settled
    assert spend[last_two[1]] < spend[last_two[0]], "the final day should be the least complete"
    assert spend.iloc[-3] == pytest.approx(settled, rel=0.15), "the lag stops at two days"


def test_ga4_lags_too_but_shopify_does_not(dataset):
    lag = dataset.truth["anomalies"]["reporting_lag"]
    last_two = [pd.Timestamp(d) for d in lag["dates"]]

    sessions = dataset.ga4.groupby("date")["sessions"].sum()
    settled_sessions = sessions.iloc[-30:-2].median()
    assert sessions[last_two[1]] < 0.75 * settled_sessions

    # Shopify is the brand's own system of record -- it is complete.
    revenue = dataset.orders.groupby("date")["revenue"].sum()
    settled_revenue = revenue.iloc[-30:-2].median()
    for day in last_two:
        assert revenue[day] == pytest.approx(settled_revenue, rel=0.20)
