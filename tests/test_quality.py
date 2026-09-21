"""The data-quality checks must find every planted artifact, and little else.

The dataset has four known defects. Three of them are this module's job -- the
week-8 GA4 gap, the week-12 Meta pixel double count and the reporting lag on
the last two days -- and they are checked here against the figures in
``truth.json``, which the checks themselves never see.

The fourth thing checked here is what the module does *not* say. This table
contains 163 pairs of orders that are identical line for line by coincidence:
same day, same region, same price, same basket. A duplicate check that called
those duplicates would be worse than no check at all, so a clean table has to
come back clean, and the checks that do fire are proven on tables with a
defect deliberately put into them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gmarge import metrics as mx
from gmarge import quality as qc

LAG_SHARE_TOLERANCE = 0.06  # the share of a settled day is an estimate, not a reading


@pytest.fixture(scope="session")
def tables(dataset):
    return mx.load_tables(dataset.path)


@pytest.fixture(scope="session")
def findings(tables):
    return qc.run_checks(tables)


def _of(findings, check, source=None):
    return [f for f in findings if f.check == check and (source is None or f.source == source)]


# --------------------------------------------------------------------------
# 1. Week 8: two days missing from GA4
# --------------------------------------------------------------------------


def test_the_ga4_gap_is_found(dataset, findings):
    planted = dataset.truth["anomalies"]["ga4_missing_days"]
    found = _of(findings, "missing_days", "ga4_sessions")

    assert len(found) == 1, "two consecutive dead days are one outage, not two findings"
    finding = found[0]
    assert list(finding.dates) == planted["dates"]
    assert finding.severity == "high"
    assert finding.numbers["weeks"] == [planted["week"]]
    assert finding.numbers["present_in"] == ["ad_spend", "shopify_orders"]
    assert finding.numbers["days_present"] == finding.numbers["days_expected"] - 2


def test_the_ga4_gap_is_quantified(dataset, findings):
    finding = _of(findings, "missing_days", "ga4_sessions")[0]
    daily = dataset.ga4.groupby("date")["sessions"].sum()

    assert finding.numbers["typical_daily_sessions"] == pytest.approx(daily.median(), rel=1e-6)
    assert finding.numbers["estimated_missing_sessions"] == pytest.approx(daily.median() * 2, rel=1e-6)
    assert "week 8" in finding.description
    assert "5 days of data, not 7" in finding.description


def test_the_other_tables_are_not_reported_missing(findings):
    assert not _of(findings, "missing_days", "shopify_orders")
    assert not _of(findings, "missing_days", "ad_spend")


def test_a_single_channel_going_dark_for_a_day_is_found(dataset, tables):
    """A gap that leaves the day looking complete unless you split by channel."""
    day = pd.Timestamp("2025-04-10")
    trimmed = tables.ad_spend[~((tables.ad_spend["date"] == day) & (tables.ad_spend["channel"] == "TikTok"))]
    injected = mx.Tables(orders=tables.orders, ad_spend=trimmed, ga4=tables.ga4, truth=tables.truth)

    found = _of(qc.check_missing_days(injected), "missing_days", "ad_spend")
    assert len(found) == 1
    assert found[0].dates == ("2025-04-10",)
    assert found[0].numbers["channel"] == "TikTok"
    assert found[0].severity == "medium"
    assert found[0].numbers["estimated_missing_spend"] > 0


def test_a_geo_holdout_is_not_reported_as_a_gap(dataset, findings):
    """Five regions spend zero for four weeks, but the rows are still there.

    Zero is not missing, and the check reads them differently: a paused region
    still reports, it just reports nothing spent.
    """
    holdout = dataset.truth["geo_holdouts"][0]
    a = dataset.ad_spend
    paused = a[
        a["date"].between(holdout["start_date"], holdout["end_date"])
        & a["region"].isin(holdout["test_regions"])
        & (a["channel"] == holdout["channel"])
    ]
    assert len(paused) > 0 and paused["spend"].sum() == 0

    assert not _of(findings, "missing_days", "ad_spend")


# --------------------------------------------------------------------------
# 2. Duplicate orders
# --------------------------------------------------------------------------


def test_a_clean_order_table_raises_nothing(dataset, findings):
    """163 pairs of coincidentally identical orders must not become 163 alarms."""
    fingerprinted = dataset.orders.duplicated(qc.ORDER_FINGERPRINT, keep=False)
    assert fingerprinted.sum() > 100, "the coincidences this check has to ignore are gone"
    assert not _of(findings, "duplicate_orders")


def test_a_repeated_order_id_is_found(dataset):
    repeated = dataset.orders.iloc[[0, 5]]
    injected = pd.concat([dataset.orders, repeated], ignore_index=True)

    found = qc.check_duplicate_orders(injected)
    assert len(found) == 1
    assert found[0].severity == "high"
    assert found[0].numbers["n_duplicate_ids"] == 2
    assert found[0].numbers["n_extra_rows"] == 2
    assert found[0].numbers["overstated_revenue"] == pytest.approx(repeated["revenue"].sum(), abs=0.01)


def test_a_batch_loaded_twice_is_found(dataset):
    """Fresh order ids, identical contents -- the shape of a re-ingested batch."""
    day = pd.Timestamp("2025-04-10")
    batch = dataset.orders[dataset.orders["date"] == day].head(200).copy()
    batch["order_id"] = [f"RELOAD-{i:06d}" for i in range(len(batch))]
    injected = pd.concat([dataset.orders, batch], ignore_index=True)

    found = qc.check_duplicate_orders(injected)
    assert len(found) == 1
    assert found[0].dates == ("2025-04-10",)
    # 200 injected, plus whatever the day already held by coincidence.
    assert 200 <= found[0].numbers["n_extra_rows"] <= 210
    assert found[0].numbers["overstated_revenue"] == pytest.approx(batch["revenue"].sum(), rel=0.02)
    assert found[0].numbers["duplicate_share"] > qc.DUP_DATE_SHARE
    assert found[0].numbers["table_baseline_share"] < 0.01


# --------------------------------------------------------------------------
# 3. Week 12: the Meta pixel double-counts purchases
# --------------------------------------------------------------------------


def test_the_pixel_double_count_is_found(dataset, findings):
    planted = dataset.truth["anomalies"]["meta_pixel_double_count"]
    found = _of(findings, "pixel_double_counting")

    assert len(found) == 1
    finding = found[0]
    assert list(finding.dates) == planted["dates"]
    assert finding.severity == "high"
    assert finding.numbers["platform"] == "Meta"
    assert finding.numbers["channels"] == sorted(planted["channels"])


def test_the_measured_multiple_is_the_planted_one(dataset, findings):
    planted = dataset.truth["anomalies"]["meta_pixel_double_count"]
    finding = _of(findings, "pixel_double_counting")[0]

    # Measured against a trailing median rather than against the untouched
    # figure, so it lands near the planted 2.0x rather than exactly on it.
    assert finding.numbers["implied_multiple"] == pytest.approx(planted["multiplier"], rel=0.15)
    for day in planted["dates"]:
        assert finding.numbers["daily_multiple"][day] > 1.8

    meta = dataset.ad_spend[dataset.ad_spend["channel"].isin(planted["channels"])]
    claimed = meta[meta["date"].isin([pd.Timestamp(d) for d in planted["dates"]])]
    assert finding.numbers["attributed_revenue"] == pytest.approx(
        claimed["platform_attributed_revenue"].sum(), abs=0.01
    )
    assert finding.numbers["excess_attributed_revenue"] > 0


def test_the_double_count_is_reported_as_a_counting_bug_not_a_sales_day(findings):
    """Spend and the store both stood still -- that is what makes it a bug."""
    numbers = _of(findings, "pixel_double_counting")[0].numbers
    assert numbers["max_spend_move"] < 0.10
    assert numbers["max_store_revenue_move"] < qc.STORE_TOLERANCE
    assert numbers["max_store_order_move"] < qc.STORE_TOLERANCE


def test_a_real_sales_spike_is_not_called_a_double_count(dataset, tables):
    """Double the store's revenue as well, and the check goes quiet."""
    planted = dataset.truth["anomalies"]["meta_pixel_double_count"]
    days = [pd.Timestamp(d) for d in planted["dates"]]
    orders = tables.orders.copy()
    same_days = orders[orders["date"].isin(days)]
    orders = pd.concat([orders, same_days.assign(order_id=same_days["order_id"] + "-B")], ignore_index=True)

    assert not qc.check_pixel_double_counting(orders, tables.ad_spend)


def test_only_meta_is_implicated(dataset, findings):
    """Google and TikTok were untouched, and must not be swept in."""
    for finding in _of(findings, "pixel_double_counting"):
        assert finding.numbers["platform"] == "Meta"


# --------------------------------------------------------------------------
# 4. The last two days are still filling in
# --------------------------------------------------------------------------


def test_the_reporting_lag_is_found_in_both_platform_tables(dataset, findings):
    planted = dataset.truth["anomalies"]["reporting_lag"]
    found = {f.source: f for f in _of(findings, "reporting_lag")}

    assert sorted(found) == sorted(planted["tables"])
    for source in planted["tables"]:
        assert list(found[source].dates) == planted["dates"]
        assert found[source].severity == "medium"
        assert found[source].numbers["complete_tables"] == ["shopify_orders"]


def test_the_measured_completeness_matches_the_planted_lag(dataset, findings):
    planted = dataset.truth["anomalies"]["reporting_lag"]
    found = {f.source: f for f in _of(findings, "reporting_lag")}
    expected = {
        "ad_spend": planted["ad_spend_share_of_final"],
        "ga4_sessions": planted["ga4_share_of_final"],
    }

    for source, shares in expected.items():
        measured = found[source].numbers["share_of_expected"]
        assert list(measured) == planted["dates"], "dates must stay in order"
        for date, share in zip(planted["dates"], shares):
            assert measured[date] == pytest.approx(share, abs=LAG_SHARE_TOLERANCE)
        assert measured[planted["dates"][1]] < measured[planted["dates"][0]]


def test_shopify_is_complete_and_is_not_flagged(findings):
    """The store's own system of record has no lag, and must not be given one."""
    assert not _of(findings, "reporting_lag", "shopify_orders")


def test_the_lag_stops_at_two_days(dataset, findings):
    planted = dataset.truth["anomalies"]["reporting_lag"]
    for finding in _of(findings, "reporting_lag"):
        assert len(finding.dates) == 2
        assert finding.dates[-1] == dataset.dates[-1].date().isoformat()
        assert finding.numbers["estimated_shortfall_" + qc.TABLE_VOLUME[finding.source]] > 0
    assert len(planted["dates"]) == 2


# --------------------------------------------------------------------------
# Every finding, and nothing but
# --------------------------------------------------------------------------


def test_every_planted_artifact_is_found(dataset, findings):
    checks = {f.check for f in findings}
    assert {"missing_days", "pixel_double_counting", "reporting_lag"} <= checks


def test_there_are_no_other_findings(dataset, findings):
    """Three artifacts, four findings -- the lag hits two tables."""
    planted_dates = set()
    for key in ("ga4_missing_days", "meta_pixel_double_count", "reporting_lag"):
        planted_dates |= set(dataset.truth["anomalies"][key]["dates"])

    unexplained = [f for f in findings if not set(f.dates) <= planted_dates]
    assert not unexplained, [f.description for f in unexplained]
    assert len(findings) == 4


def test_findings_are_sorted_worst_first(findings):
    ranks = [qc.SEVERITY_ORDER[f.severity] for f in findings]
    assert ranks == sorted(ranks)


def test_every_finding_carries_its_evidence(findings):
    for finding in findings:
        assert finding.severity in qc.SEVERITY_ORDER
        assert finding.dates and list(finding.dates) == sorted(finding.dates)
        assert finding.source in ("shopify_orders", "ad_spend", "ga4_sessions")
        assert len(finding.description) > 80 and finding.description.endswith(".")
        assert finding.numbers, "a finding must carry the numbers behind it"
        assert finding.to_dict()["n_days"] == len(finding.dates)


def test_findings_frame_is_one_row_per_finding(findings):
    frame = qc.findings_frame(findings)
    assert len(frame) == len(findings)
    assert list(frame["severity"]) == [f.severity for f in findings]
    assert qc.findings_frame([]).empty


def test_run_checks_takes_a_directory_or_loaded_tables(dataset, findings):
    from_path = qc.run_checks(dataset.path)
    assert [f.to_dict() for f in from_path] == [f.to_dict() for f in findings]


# --------------------------------------------------------------------------
# A different seed
# --------------------------------------------------------------------------


def test_the_same_artifacts_are_found_on_a_second_seed(other_dataset):
    """The thresholds are set from the spread of the data, not from one seed."""
    findings = qc.run_checks(other_dataset.path)
    planted = other_dataset.truth["anomalies"]
    by_check = {}
    for finding in findings:
        by_check.setdefault(finding.check, []).append(finding)

    assert list(by_check["missing_days"][0].dates) == planted["ga4_missing_days"]["dates"]
    assert list(by_check["pixel_double_counting"][0].dates) == planted["meta_pixel_double_count"]["dates"]

    lagging = {f.source for f in by_check["reporting_lag"]}
    assert lagging == set(planted["reporting_lag"]["tables"])
    for finding in by_check["reporting_lag"]:
        # The day-two shortfall can fall inside the noise on some seeds; the
        # final day never does.
        assert finding.dates[-1] == planted["reporting_lag"]["dates"][-1]

    assert not [f for f in findings if f.check == "duplicate_orders"]
