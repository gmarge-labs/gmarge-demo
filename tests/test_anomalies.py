"""The weekly anomaly scan must find the planted faults and stay quiet otherwise.

Two planted faults live in ``ad_spend`` and belong to this module: the Meta
pixel double count in week 12 and the creative fatigue on one Meta prospecting
ad set in week 20. Both have to be found, on the right channel, in the right
week, and attributed to the right ad set.

The rest of these tests are about silence. Over 26 weeks the data also contains
five geo holdouts that take a channel dark in five of twenty regions, a promo
week, and two days at the end that are still filling in. None of those is a
fault, and a scan that called them one would be ignored within a week. The
budget is two false alarms across the whole window; the scan uses none of it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gmarge import anomalies as an

MAX_FALSE_ALARMS = 2


@pytest.fixture(scope="session")
def flags(dataset):
    return an.detect_anomalies(dataset.ad_spend)


def _planted(truth) -> set[tuple[str, int]]:
    """The channel-weeks a fault was actually planted in."""
    pixel = truth["anomalies"]["meta_pixel_double_count"]
    fatigue = truth["anomalies"]["creative_fatigue"]
    return {(c, pixel["week"]) for c in pixel["channels"]} | {(fatigue["channel"], fatigue["week"])}


def _week_of(dataset, date) -> int:
    return int((pd.Timestamp(date) - dataset.dates[0]).days // 7) + 1


def _one(flags, channel, week, metric):
    found = [f for f in flags if f.channel == channel and f.week == week and f.metric == metric]
    assert len(found) == 1, f"expected one {metric} flag for {channel} in week {week}, got {len(found)}"
    return found[0]


# --------------------------------------------------------------------------
# Week 20: creative fatigue on one ad set
# --------------------------------------------------------------------------


def test_the_week_20_frequency_spike_is_found(dataset, flags):
    planted = dataset.truth["anomalies"]["creative_fatigue"]
    flag = _one(flags, planted["channel"], planted["week"], "frequency")

    assert flag.direction == "up"
    assert flag.severity == "high"
    assert flag.week_start == planted["start_date"] and flag.week_end == planted["end_date"]
    assert flag.robust_score > an.SCORE_THRESHOLD
    assert flag.pct_change > 0.25


def test_the_frequency_spike_is_pinned_on_the_ad_set_that_caused_it(dataset, flags):
    planted = dataset.truth["anomalies"]["creative_fatigue"]
    flag = _one(flags, planted["channel"], planted["week"], "frequency")

    assert flag.ad_set == planted["ad_set"]
    assert flag.campaign == "MP - Core Prospecting"
    assert flag.share_of_move > 0.80, "the fatigued ad set is nearly the whole move"

    top = flag.numbers["ad_set_shares"][0]
    assert top["value"] == pytest.approx(planted["frequency_fatigue_week"], rel=0.05)
    assert top["baseline"] == pytest.approx(planted["frequency_prior_week"], rel=0.10)


def test_the_week_20_roas_collapse_is_found_and_attributed(dataset, flags):
    planted = dataset.truth["anomalies"]["creative_fatigue"]
    flag = _one(flags, planted["channel"], planted["week"], "reported_roas")

    assert flag.direction == "down"
    assert flag.pct_change < -an.MIN_RELATIVE_MOVE
    assert flag.ad_set == planted["ad_set"]
    assert flag.share_of_move > 0.80

    top = flag.numbers["ad_set_shares"][0]
    assert top["value"] == pytest.approx(planted["reported_roas_fatigue_week"], rel=0.02)
    assert top["baseline"] == pytest.approx(planted["reported_roas_prior_week"], rel=0.10)


def test_the_other_meta_prospecting_ad_sets_are_not_blamed(dataset, flags):
    planted = dataset.truth["anomalies"]["creative_fatigue"]
    flag = _one(flags, planted["channel"], planted["week"], "frequency")

    others = [s for s in flag.numbers["ad_set_shares"] if s["ad_set"] != planted["ad_set"]]
    assert others, "the channel has more than one ad set"
    assert all(abs(s["share_of_move"]) < 0.10 for s in others), others


# --------------------------------------------------------------------------
# Week 12: the Meta pixel double count
# --------------------------------------------------------------------------


def test_the_week_12_double_count_is_found_on_both_meta_channels(dataset, flags):
    planted = dataset.truth["anomalies"]["meta_pixel_double_count"]

    for channel in planted["channels"]:
        flag = _one(flags, channel, planted["week"], "reported_roas")
        assert flag.direction == "up"
        assert flag.severity == "high"
        assert flag.pct_change > 0.30, "three days doubled in a seven-day week"
        assert flag.robust_score > 10


def test_the_double_count_spares_the_other_channels(dataset, flags):
    planted = dataset.truth["anomalies"]["meta_pixel_double_count"]
    others = {"Google branded search", "Google shopping", "TikTok"}

    assert not [f for f in flags if f.week == planted["week"] and f.channel in others]


def test_the_double_count_is_spread_across_the_ad_sets(dataset, flags):
    """A pixel fires for the whole platform, so no one ad set is to blame.

    The shares should track each ad set's size rather than singling one out,
    which is what the attribution has to say when nothing is singular.
    """
    planted = dataset.truth["anomalies"]["meta_pixel_double_count"]
    flag = _one(flags, "Meta prospecting", planted["week"], "reported_roas")

    shares = {s["ad_set"]: s["share_of_move"] for s in flag.numbers["ad_set_shares"]}
    assert len(shares) == 4
    assert max(shares.values()) < 0.50, shares

    week = an.weekly_metrics(dataset.ad_spend, by=["channel", "ad_set"])
    week = week[(week["channel"] == "Meta prospecting") & (week["week"] == planted["week"])]
    spend_share = (week.set_index("ad_set")["spend"] / week["spend"].sum()).to_dict()
    for ad_set, share in shares.items():
        assert share == pytest.approx(spend_share[ad_set], abs=0.05)


# --------------------------------------------------------------------------
# Silence
# --------------------------------------------------------------------------


def test_no_more_than_two_false_alarms_across_the_window(dataset, flags):
    planted = _planted(dataset.truth)
    false_alarms = [f for f in flags if (f.channel, f.week) not in planted]

    assert len(false_alarms) <= MAX_FALSE_ALARMS, [f.description for f in false_alarms]
    assert not false_alarms, "no budget spent at all on this seed"


def test_geo_holdouts_do_not_raise_alarms(dataset, flags):
    """A channel going dark in five of twenty regions is a planned test.

    It moves that channel's spend by a fifth to a third, and its recovery
    moves it back. Scoring rates rather than levels is what keeps both out of
    the alert list.
    """
    planted = _planted(dataset.truth)
    for holdout in dataset.truth["geo_holdouts"]:
        start = _week_of(dataset, holdout["start_date"])
        end = _week_of(dataset, holdout["end_date"])
        weeks = set(range(start, end + 2))  # plus the week spend comes back
        raised = [
            f
            for f in flags
            if f.channel == holdout["channel"]
            and f.week in weeks
            and (f.channel, f.week) not in planted  # week 12 really is faulty
        ]
        assert not raised, [f.description for f in raised]


def test_the_promo_week_does_not_raise_an_alarm(dataset, flags):
    promo = dataset.truth["other_features"]["promo_window"]
    assert not [f for f in flags if f.week == promo["week"]]


def test_the_reporting_lag_does_not_raise_an_alarm(dataset, flags):
    """The lag scales a rate's numerator and denominator together."""
    last_week = _week_of(dataset, dataset.truth["anomalies"]["reporting_lag"]["dates"][-1])
    assert last_week == 26
    assert not [f for f in flags if f.week == last_week]


def test_neither_guard_is_carrying_the_result_alone(dataset, monkeypatch):
    """Both the MAD floor and the 10% move floor suppress the noise by themselves.

    The trailing weeks are so tight that a 2% wobble can score past 3.5, which
    is the case for having two conditions rather than one. This is the
    measurement behind that claim, so it is worth a test rather than a comment.
    """
    planted = _planted(dataset.truth)
    scores = an.robust_scores(dataset.ad_spend)
    clean = scores[[(c, w) not in planted for c, w in zip(scores["channel"], scores["week"])]]

    assert len(clean) > 350
    assert clean["pct_change"].abs().max() < an.MIN_RELATIVE_MOVE
    assert clean["robust_score"].abs().max() < an.SCORE_THRESHOLD

    monkeypatch.setattr(an, "MAD_FLOOR_SHARE", 0.0)
    unfloored = an.robust_scores(dataset.ad_spend)
    unfloored = unfloored[[(c, w) not in planted for c, w in zip(unfloored["channel"], unfloored["week"])]]
    loud = unfloored[unfloored["robust_score"].abs() >= an.SCORE_THRESHOLD]

    assert len(loud) > 10, "without the floor the score alone lets noise through"
    assert loud["pct_change"].abs().max() < an.MIN_RELATIVE_MOVE, "and the move floor still stops it"


# --------------------------------------------------------------------------
# The scan itself
# --------------------------------------------------------------------------


def test_every_channel_week_with_history_is_scored(dataset):
    scores = an.robust_scores(dataset.ad_spend)
    scored_weeks = sorted(scores["week"].unique())

    assert scored_weeks == list(range(an.TRAILING_WEEKS + 1, 27))
    assert sorted(scores["channel"].unique()) == sorted(dataset.ad_spend["channel"].unique())
    assert (scores["trailing_weeks"] == an.TRAILING_WEEKS).all()


def test_search_channels_are_not_scored_on_frequency(dataset):
    scores = an.robust_scores(dataset.ad_spend)
    frequency = scores[scores["metric"] == "frequency"]

    assert set(frequency["channel"]) == {"Meta prospecting", "Meta retargeting", "TikTok"}
    assert frequency["value"].notna().all()


def test_weekly_metrics_reproduce_the_table(dataset):
    weekly = an.weekly_metrics(dataset.ad_spend)
    a = dataset.ad_spend

    assert weekly["spend"].sum() == pytest.approx(a["spend"].sum(), rel=1e-9)
    assert weekly["clicks"].sum() == a["clicks"].sum()
    assert (weekly["days"] == 7).all()

    row = weekly[(weekly["channel"] == "TikTok") & (weekly["week"] == 3)].iloc[0]
    sub = a[(a["channel"] == "TikTok") & a["date"].between(*dataset.week(3))]
    assert row["reported_roas"] == pytest.approx(
        sub["platform_attributed_revenue"].sum() / sub["spend"].sum(), rel=1e-9
    )
    assert row["frequency"] == pytest.approx(
        (sub["frequency"] * sub["impressions"]).sum() / sub["impressions"].sum(), rel=1e-9
    )


def test_contributions_account_for_the_whole_move(dataset, flags):
    """The shares are shares of something: they sum to the channel's move."""
    for flag in flags:
        shares = [s["share_of_move"] for s in flag.numbers["ad_set_shares"]]
        assert sum(shares) == pytest.approx(1.0, abs=1e-6)
        assert flag.share_of_move == pytest.approx(max(shares), abs=1e-6)


def test_the_top_contributor_is_measured_against_its_own_history(dataset, flags):
    planted = dataset.truth["anomalies"]["creative_fatigue"]
    flag = _one(flags, planted["channel"], planted["week"], "reported_roas")

    weekly = an.weekly_metrics(dataset.ad_spend, by=["channel", "ad_set"])
    history = weekly[
        (weekly["ad_set"] == flag.ad_set)
        & (weekly["week"] >= flag.week - an.TRAILING_WEEKS)
        & (weekly["week"] < flag.week)
    ]["reported_roas"]

    assert flag.numbers["ad_set_shares"][0]["baseline"] == pytest.approx(history.median(), abs=1e-6)


def test_flags_are_sorted_worst_first_and_carry_their_evidence(flags):
    ranks = [(an.SEVERITY_ORDER[f.severity], f.week) for f in flags]
    assert ranks == sorted(ranks)

    for flag in flags:
        assert flag.metric in an.METRICS_BY_KEY
        assert flag.direction in ("up", "down")
        assert flag.week_start <= flag.week_end
        assert len(flag.description) > 80 and flag.description.endswith(".")
        assert flag.numbers["trailing_weeks"] == an.TRAILING_WEEKS
        assert flag.numbers["contribution_unit"] in an.BASE_COLUMNS
        assert flag.description.count(flag.ad_set) >= 1


def test_anomalies_frame_is_one_row_per_flag(flags):
    frame = an.anomalies_frame(flags)
    assert len(frame) == len(flags)
    assert list(frame["channel"]) == [f.channel for f in flags]
    assert an.anomalies_frame([]).empty


def test_detect_takes_a_directory_a_frame_or_loaded_tables(dataset, flags):
    from_path = an.detect_anomalies(dataset.path)
    assert [f.to_dict() for f in from_path] == [f.to_dict() for f in flags]


# --------------------------------------------------------------------------
# A different seed
# --------------------------------------------------------------------------


def test_the_same_faults_are_found_on_a_second_seed(other_dataset):
    flags = an.detect_anomalies(other_dataset.ad_spend)
    planted = _planted(other_dataset.truth)
    fatigue = other_dataset.truth["anomalies"]["creative_fatigue"]

    assert {(f.channel, f.week) for f in flags} == planted
    assert len([f for f in flags if (f.channel, f.week) not in planted]) <= MAX_FALSE_ALARMS

    spike = _one(flags, fatigue["channel"], fatigue["week"], "frequency")
    assert spike.ad_set == fatigue["ad_set"]
    assert spike.share_of_move > 0.80
