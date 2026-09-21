"""Shape, columns and internal consistency of the three tables."""

from __future__ import annotations

import pandas as pd

from gmarge import generate as gen

SEARCH_CHANNELS = ["Google branded search", "Google shopping"]


def test_date_coverage(dataset):
    expected = set(dataset.dates)
    assert set(dataset.orders["date"]) == expected
    assert set(dataset.ad_spend["date"]) == expected
    assert len(expected) == 182 == gen.N_WEEKS * 7


def test_regions_are_region_01_to_20(dataset):
    expected = [f"Region {i:02d}" for i in range(1, 21)]
    assert sorted(dataset.orders["region"].unique()) == expected
    assert sorted(dataset.ad_spend["region"].unique()) == expected
    assert sorted(dataset.ga4["region"].unique()) == expected
    assert dataset.truth["generator"]["regions"] == expected


def test_order_columns(dataset):
    o = dataset.orders
    assert list(o.columns) == [
        "order_id", "date", "region", "revenue", "discount", "refund",
        "net_revenue", "items", "customer_type",
    ]
    assert o["order_id"].is_unique
    assert not o.isna().any().any()
    assert (o["revenue"] > 0).all()
    assert (o["discount"] >= 0).all() and (o["discount"] <= o["revenue"]).all()
    assert (o["refund"] >= 0).all() and (o["refund"] <= o["revenue"]).all()
    assert ((o["net_revenue"] - (o["revenue"] - o["discount"] - o["refund"])).abs() < 0.005).all()
    assert set(o["customer_type"]) == {"new", "returning"}


def test_ad_spend_columns(dataset):
    a = dataset.ad_spend
    assert list(a.columns) == [
        "date", "region", "channel", "campaign", "ad_set", "spend",
        "impressions", "clicks", "frequency", "platform_attributed_revenue",
    ]
    assert sorted(a["channel"].unique()) == sorted(gen.CHANNELS)
    assert (a["spend"] >= 0).all()
    assert (a["impressions"] >= 0).all() and (a["clicks"] >= 0).all()
    assert (a["platform_attributed_revenue"] >= 0).all()
    assert a.drop(columns="frequency").notna().all().all()
    # Every ad set belongs to exactly one campaign and one channel.
    assert (a.groupby("ad_set")[["channel", "campaign"]].nunique() == 1).all().all()


def test_only_paid_channels_have_spend_rows(dataset):
    """Email and organic drive revenue but never appear in the ad platform."""
    assert not set(dataset.ad_spend["channel"]) & {"Email", "Organic"}
    assert dataset.truth["generator"]["unpaid_channels"] == ["Email", "Organic"]
    sources = set(dataset.ga4["source_medium"])
    assert {"google / organic", "klaviyo / email", "(direct) / (none)"} <= sources


def test_frequency_only_reported_for_social(dataset):
    a = dataset.ad_spend
    assert a.loc[a["channel"].isin(SEARCH_CHANNELS), "frequency"].isna().all()
    social = a.loc[~a["channel"].isin(SEARCH_CHANNELS), "frequency"]
    assert social.notna().all()
    assert (social >= 1.0).all()


def test_ga4_columns(dataset):
    g = dataset.ga4
    assert list(g.columns) == ["date", "region", "source_medium", "landing_page", "sessions"]
    assert not g.isna().any().any()
    assert (g["sessions"] > 0).all()
    assert g["landing_page"].str.startswith("/").all()
    assert g["source_medium"].str.contains(" / ").all()


def test_truth_totals_match_the_tables(dataset):
    t = dataset.truth
    assert round(dataset.orders["revenue"].sum(), 2) == t["attribution_gap"]["shopify_revenue"]
    assert (
        round(dataset.ad_spend["platform_attributed_revenue"].sum(), 2)
        == t["attribution_gap"]["platform_attributed_revenue"]
    )
    by_channel = dataset.ad_spend.groupby("channel")["spend"].sum()
    for c in t["channel_truth"]:
        assert round(float(by_channel[c["channel"]]), 2) == c["spend"]


def test_brand_is_labelled_fictional(dataset):
    assert dataset.truth["brand"] == "Northfield Goods (sample brand)"
    assert "fictional" in dataset.truth["disclaimer"].lower()
