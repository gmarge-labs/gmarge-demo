"""Seeded, reproducible generator of sample marketing data.

Builds 26 weeks of daily data for a FICTIONAL direct-to-consumer skincare brand,
"Northfield Goods (sample brand)". Nothing here describes a real company.

The generator plants a set of known "truths" in the data -- an attribution gap,
true incremental ROAS per channel, five geo holdouts, and four data-quality
artifacts -- and records every one of them in ``data/truth.json`` so tests (and
analyses) can be checked against ground truth.

Run with::

    python -m gmarge.generate --out data

Determinism: every random draw comes from a named ``numpy`` bit-stream derived
from the seed, so the same seed always produces the same figures regardless of
call order elsewhere in the module.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

BRAND = "Northfield Goods (sample brand)"
DISCLAIMER = (
    "Synthetic sample data for a fictional brand. Not a real company, "
    "not real revenue, not any real advertiser's performance."
)

SEED = 20260921
START_DATE = date(2025, 1, 6)  # a Monday
N_WEEKS = 26
N_DAYS = N_WEEKS * 7  # 182
N_REGIONS = 20

# The attribution gap: platform-reported revenue over Shopify revenue.
ATTRIBUTION_RATIO = 1.40

# Region scale. Regions are built in matched pairs of near-identical size so
# that geo holdouts have a credible control group.
PAIR_WEIGHTS = [1.00, 0.92, 0.85, 0.78, 0.72, 0.66, 0.60, 0.55, 0.50, 0.45]

# Demand shape.
DOW_DEMAND = [1.04, 1.07, 1.01, 0.98, 0.94, 0.86, 0.94]  # Mon..Sun
DOW_SPEND = [1.02, 1.03, 1.01, 1.00, 1.00, 0.96, 0.98]
DEMAND_TREND = (1.00, 1.18)  # start -> end, linear
SPEND_TREND = (1.00, 1.25)
# Noise is split into a day-level shock shared by every region (a budget
# change, a press mention, a weather week) and a smaller region-day wobble.
# Keeping most of the variance at day level is both realistic and what makes
# matched-market comparisons work: a shared shock cancels between test and
# control, an independent one does not.
DEMAND_DAY_SIGMA = 0.05
DEMAND_REGION_SIGMA = 0.025
SPEND_DAY_SIGMA = 0.09
SPEND_REGION_SIGMA = 0.03

# A single promo window, to give discounts something to do.
PROMO_WEEK = 15
PROMO_DAYS = 4
PROMO_DEMAND_LIFT = 1.35

# Baseline (unpaid) demand split, used for GA4 session volumes.
BASELINE_SPLIT = {"google / organic": 0.40, "(direct) / (none)": 0.35, "klaviyo / email": 0.25}

AOV_BASE = 68.0
REVENUE_PER_SESSION = 2.10

# --------------------------------------------------------------------------
# Channels
#
# ``true_iroas``   -- the real incremental revenue per $1 of spend.
# ``reported_roas``-- what the ad platform claims, via its own attribution.
# The gap between them is the whole point of the dataset.
# --------------------------------------------------------------------------

CHANNELS: dict[str, dict] = {
    "Meta prospecting": {
        "daily_spend": 9000.0,
        "true_iroas": 1.80,
        "reported_roas": 2.42,
        "cpm": 11.5,
        "cpc": 1.30,
        "base_frequency": 1.9,
        "campaigns": {
            "MP - Core Prospecting": {"Core | Broad 25-44": 0.34, "Core | Interest Skincare": 0.22},
            "MP - Lookalike": {"LAL | 1% Purchasers": 0.26, "LAL | 3% Site Visitors": 0.18},
        },
    },
    "Meta retargeting": {
        "daily_spend": 2500.0,
        "true_iroas": 0.60,
        "reported_roas": 3.60,
        "cpm": 16.0,
        "cpc": 1.05,
        "base_frequency": 3.4,
        "campaigns": {
            "MR - Site Retargeting": {"RET | 7d Add-to-Cart": 0.58, "RET | 30d Viewers": 0.42},
        },
    },
    "Google branded search": {
        "daily_spend": 1800.0,
        "true_iroas": 0.45,
        "reported_roas": 4.05,
        "cpm": 38.0,
        "cpc": 0.95,
        "base_frequency": None,  # search does not report frequency
        "campaigns": {
            "GBS - Brand": {"Brand | Exact": 0.62, "Brand | Phrase": 0.38},
        },
    },
    "Google shopping": {
        "daily_spend": 4200.0,
        "true_iroas": 1.30,
        "reported_roas": 2.70,
        "cpm": 22.0,
        "cpc": 0.72,
        "base_frequency": None,
        "campaigns": {
            "GS - Shopping": {"Shopping | Core Catalog": 0.63, "Shopping | Bestsellers": 0.37},
        },
    },
    "TikTok": {
        "daily_spend": 2800.0,
        "true_iroas": 1.00,
        "reported_roas": 2.28,
        "cpm": 7.5,
        "cpc": 1.55,
        "base_frequency": 2.3,
        "campaigns": {
            "TT - Prospecting": {"TT | Broad": 0.55, "TT | Spark Ads": 0.45},
        },
    },
}

UNPAID_CHANNELS = ["Email", "Organic"]

META_CHANNELS = ["Meta prospecting", "Meta retargeting"]

# GA4 source / medium for each paid channel's clicks.
PAID_SOURCE_MEDIUM = {
    "Meta prospecting": "facebook / cpc",
    "Meta retargeting": "facebook / cpc",
    "Google branded search": "google / cpc",
    "Google shopping": "google / cpc",
    "TikTok": "tiktok / cpc",
}

LANDING_PAGES = {
    "facebook / cpc": {"/products/barrier-repair-cream": 0.45, "/pages/skin-quiz": 0.30, "/": 0.25},
    "google / cpc": {"/collections/best-sellers": 0.40, "/products/hydrating-serum": 0.35, "/": 0.25},
    "tiktok / cpc": {"/products/hydrating-serum": 0.50, "/pages/skin-quiz": 0.30, "/": 0.20},
    "google / organic": {"/": 0.45, "/collections/best-sellers": 0.30, "/products/barrier-repair-cream": 0.25},
    "(direct) / (none)": {"/": 0.70, "/collections/new-arrivals": 0.20, "/account/login": 0.10},
    "klaviyo / email": {"/collections/new-arrivals": 0.50, "/": 0.30, "/collections/best-sellers": 0.20},
}

# --------------------------------------------------------------------------
# Planted anomalies (1-indexed weeks, 0-indexed day offsets within the week)
# --------------------------------------------------------------------------

# Creative fatigue: one Meta prospecting ad set burns out in week 20.
FATIGUE_WEEK = 20
FATIGUE_AD_SET = "Core | Broad 25-44"
FATIGUE_FREQ_MULT = (1.70, 2.45)  # ramps across the week, mean ~2.05x
FATIGUE_TRUE_MULT = (0.55, 0.26)
FATIGUE_REPORTED_MULT = (0.58, 0.28)

# Meta pixel double-counts purchases for three days in week 12.
PIXEL_WEEK = 12
PIXEL_DAY_OFFSETS = (1, 2, 3)
PIXEL_MULT = 2.0

# GA4 loses two days of data in week 8.
GA4_MISSING_WEEK = 8
GA4_MISSING_DAY_OFFSETS = (2, 3)

# Platform reporting lag on the final two days: numbers are still filling in.
LAG_FACTORS = {1: 0.82, 0: 0.45}  # days-from-end -> share of final value
GA4_LAG_FACTORS = {1: 0.88, 0: 0.60}

# Each paid channel gets one 4-week geo holdout, at a different time.
HOLDOUT_START_WEEKS = {
    "Meta prospecting": 5,
    "Meta retargeting": 9,
    "Google branded search": 13,
    "Google shopping": 17,
    "TikTok": 21,
}
HOLDOUT_WEEKS = 4

# --------------------------------------------------------------------------
# Deterministic random streams
# --------------------------------------------------------------------------

_STREAMS = [
    "spend",
    "ad_set_split",
    "impressions",
    "clicks",
    "frequency",
    "attributed",
    "demand_noise",
    "order_value",
    "order_discount",
    "order_refund",
    "order_customer",
    "ga4_paid",
    "ga4_unpaid",
    "ga4_landing",
]


def _rng(seed: int, stream: str, extra: int = 0) -> np.random.Generator:
    """A named, reproducible bit-stream. Independent of call order."""
    return np.random.default_rng([seed, _STREAMS.index(stream), extra])


def _lognormal(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    """Multiplicative noise with mean 1."""
    return rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=shape)


# --------------------------------------------------------------------------
# Calendar and regions
# --------------------------------------------------------------------------


def _week_days(week: int) -> list[int]:
    """Day indices (0-based) for a 1-indexed week number."""
    return list(range((week - 1) * 7, week * 7))


def _dates() -> pd.DatetimeIndex:
    return pd.date_range(START_DATE, periods=N_DAYS, freq="D")


def _regions() -> list[str]:
    return [f"Region {i:02d}" for i in range(1, N_REGIONS + 1)]


def _region_weights() -> np.ndarray:
    """Region size weights, arranged so regions 1&2, 3&4, ... are matched pairs."""
    weights = []
    for w in PAIR_WEIGHTS:
        weights.extend([w * 1.01, w * 0.99])
    arr = np.array(weights, dtype=float)
    return arr / arr.sum()


def _region_pairs() -> list[tuple[str, str]]:
    regions = _regions()
    return [(regions[2 * p], regions[2 * p + 1]) for p in range(len(PAIR_WEIGHTS))]


def _holdout_specs() -> list[dict]:
    """One 4-week geo holdout per paid channel, each at a different time.

    Consecutive holdouts use the two disjoint halves of the matched pairs in
    turn. The four weeks before any holdout are therefore free of *that*
    holdout's regions, even though a different channel is dark at the time --
    which keeps each test's pre-period usable as a baseline.
    """
    pairs = _region_pairs()
    dates = _dates()
    specs = []
    for ci, (channel, start_week) in enumerate(HOLDOUT_START_WEEKS.items()):
        half = 0 if ci % 2 == 0 else 5
        chosen = [pairs[half + j] for j in range(5)]
        day_idx = [d for w in range(start_week, start_week + HOLDOUT_WEEKS) for d in _week_days(w)]
        specs.append(
            {
                "channel": channel,
                "start_week": start_week,
                "end_week": start_week + HOLDOUT_WEEKS - 1,
                "day_idx": day_idx,
                "start_date": dates[day_idx[0]].date().isoformat(),
                "end_date": dates[day_idx[-1]].date().isoformat(),
                "test_regions": [t for t, _ in chosen],
                "control_regions": [c for _, c in chosen],
            }
        )
    return specs


def _trend(lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, N_DAYS)


def _demand_shape() -> np.ndarray:
    """Per-day multiplier for unpaid baseline demand (before region weights)."""
    dow = np.array([DOW_DEMAND[(START_DATE + timedelta(days=d)).weekday()] for d in range(N_DAYS)])
    shape = dow * _trend(*DEMAND_TREND)
    promo = _week_days(PROMO_WEEK)[:PROMO_DAYS]
    shape[promo] *= PROMO_DEMAND_LIFT
    return shape


def _promo_day_idx() -> list[int]:
    return _week_days(PROMO_WEEK)[:PROMO_DAYS]


# --------------------------------------------------------------------------
# Ad spend
# --------------------------------------------------------------------------


def _ramp(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, n)


def build_ad_spend(seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Build the ad-set level spend table.

    Returns ``(ad_spend, true_inc, counterfactual_inc, meta)`` where the two
    arrays are shaped ``(n_channels, n_days, n_regions)`` and hold *true*
    incremental revenue -- what the spend actually caused, before any platform
    attribution -- with and without the geo holdouts applied.
    """
    dates = _dates()
    regions = _regions()
    region_share = _region_weights()
    spend_dow = np.array([DOW_SPEND[(START_DATE + timedelta(days=d)).weekday()] for d in range(N_DAYS)])
    spend_shape = (spend_dow * _trend(*SPEND_TREND))[:, None]

    holdouts = {h["channel"]: h for h in _holdout_specs()}
    channel_names = list(CHANNELS)
    true_inc = np.zeros((len(channel_names), N_DAYS, N_REGIONS))
    cf_inc = np.zeros_like(true_inc)

    fatigue_days = _week_days(FATIGUE_WEEK)
    pixel_days = [_week_days(PIXEL_WEEK)[o] for o in PIXEL_DAY_OFFSETS]

    lag_mult = np.ones(N_DAYS)
    for back, factor in LAG_FACTORS.items():
        lag_mult[N_DAYS - 1 - back] = factor

    frames = []
    for ci, channel in enumerate(channel_names):
        cfg = CHANNELS[channel]

        # Counterfactual spend: what would have been spent with no holdout.
        rng_spend = _rng(seed, "spend", ci)
        noise = _lognormal(rng_spend, (N_DAYS, 1), SPEND_DAY_SIGMA) * _lognormal(
            rng_spend, (N_DAYS, N_REGIONS), SPEND_REGION_SIGMA
        )
        cf_spend = cfg["daily_spend"] * region_share[None, :] * spend_shape * noise

        # Apply the geo holdout: spend goes to exactly zero in the test regions.
        spend = cf_spend.copy()
        hold = holdouts[channel]
        test_idx = [regions.index(r) for r in hold["test_regions"]]
        spend[np.ix_(hold["day_idx"], test_idx)] = 0.0

        # Split each region-day across ad sets, renormalised so the ad-set
        # rows always sum back to the channel's region-day spend.
        ad_sets = [(camp, ad_set, w) for camp, sets in cfg["campaigns"].items() for ad_set, w in sets.items()]
        split_noise = _lognormal(_rng(seed, "ad_set_split", ci), (len(ad_sets), N_DAYS, N_REGIONS), 0.08)
        weights = np.array([w for _, _, w in ad_sets])[:, None, None] * split_noise
        weights /= weights.sum(axis=0, keepdims=True)

        rng_imp = _rng(seed, "impressions", ci)
        rng_clk = _rng(seed, "clicks", ci)
        rng_frq = _rng(seed, "frequency", ci)
        rng_att = _rng(seed, "attributed", ci)

        for ai, (campaign, ad_set, _) in enumerate(ad_sets):
            a_spend = spend * weights[ai]
            a_cf_spend = cf_spend * weights[ai]

            true_mult = np.ones((N_DAYS, 1))
            reported_mult = np.ones((N_DAYS, 1))
            freq_mult = np.ones((N_DAYS, 1))
            if channel == "Meta prospecting" and ad_set == FATIGUE_AD_SET:
                true_mult[fatigue_days, 0] = _ramp(*FATIGUE_TRUE_MULT, len(fatigue_days))
                reported_mult[fatigue_days, 0] = _ramp(*FATIGUE_REPORTED_MULT, len(fatigue_days))
                freq_mult[fatigue_days, 0] = _ramp(*FATIGUE_FREQ_MULT, len(fatigue_days))

            true_inc[ci] += a_spend * cfg["true_iroas"] * true_mult
            cf_inc[ci] += a_cf_spend * cfg["true_iroas"] * true_mult

            # Reported figures. The platform sees its own attributed revenue,
            # which is inflated by the channel's over-attribution.
            attributed = (
                a_spend
                * cfg["reported_roas"]
                * reported_mult
                * _lognormal(rng_att, (N_DAYS, N_REGIONS), 0.12)
            )
            if channel in META_CHANNELS:
                attributed[pixel_days, :] *= PIXEL_MULT

            impressions = a_spend / cfg["cpm"] * 1000.0 * _lognormal(rng_imp, (N_DAYS, N_REGIONS), 0.07)
            clicks = a_spend / cfg["cpc"] * _lognormal(rng_clk, (N_DAYS, N_REGIONS), 0.09)

            if cfg["base_frequency"] is None:
                frequency = np.full((N_DAYS, N_REGIONS), np.nan)
            else:
                drift = _trend(1.0, 1.15)[:, None]
                frequency = (
                    cfg["base_frequency"] * drift * freq_mult * _lognormal(rng_frq, (N_DAYS, N_REGIONS), 0.06)
                )

            # Reporting lag applies to everything the platform reports.
            reported_spend = a_spend * lag_mult[:, None]
            frames.append(
                pd.DataFrame(
                    {
                        "date": np.repeat(dates.values, N_REGIONS),
                        "region": np.tile(regions, N_DAYS),
                        "channel": channel,
                        "campaign": campaign,
                        "ad_set": ad_set,
                        "spend": reported_spend.ravel().round(2),
                        "impressions": np.rint(impressions * lag_mult[:, None]).ravel().astype("int64"),
                        "clicks": np.rint(clicks * lag_mult[:, None]).ravel().astype("int64"),
                        "frequency": frequency.ravel().round(3),
                        "platform_attributed_revenue": (attributed * lag_mult[:, None]).ravel().round(2),
                    }
                )
            )

    ad_spend = pd.concat(frames, ignore_index=True)
    ad_spend = ad_spend.sort_values(["date", "channel", "campaign", "ad_set", "region"]).reset_index(drop=True)

    meta = {
        "channel_names": channel_names,
        "pixel_days": pixel_days,
        "fatigue_days": fatigue_days,
        "lag_mult": lag_mult,
    }
    return ad_spend, true_inc, cf_inc, meta


# --------------------------------------------------------------------------
# Revenue and orders
# --------------------------------------------------------------------------


def build_revenue(seed: int, true_inc: np.ndarray, attributed_total: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Region-day Shopify revenue, calibrated to the planted attribution ratio.

    Revenue is unpaid baseline demand plus the true incremental revenue from
    paid media, with one multiplicative demand shock per region-day. The
    baseline is scaled so total Shopify revenue lands exactly at
    ``attributed_total / ATTRIBUTION_RATIO``.
    """
    rng_demand = _rng(seed, "demand_noise")
    demand_noise = _lognormal(rng_demand, (N_DAYS, 1), DEMAND_DAY_SIGMA) * _lognormal(
        rng_demand, (N_DAYS, N_REGIONS), DEMAND_REGION_SIGMA
    )
    shape = _demand_shape()[:, None] * _region_weights()[None, :]

    inc = true_inc.sum(axis=0)
    target_total = attributed_total / ATTRIBUTION_RATIO
    realised_inc = float((inc * demand_noise).sum())
    k = (target_total - realised_inc) / float((shape * demand_noise).sum())
    if k <= 0:
        raise ValueError("Baseline demand solved negative -- lower spend or true iROAS.")

    baseline = shape * k
    revenue = (baseline + inc) * demand_noise
    return revenue, baseline * demand_noise, demand_noise


def build_orders(seed: int, revenue: np.ndarray, true_inc: np.ndarray) -> pd.DataFrame:
    """Explode region-day revenue into order-level rows summing to it exactly."""
    dates = _dates()
    regions = _regions()
    rng_val = _rng(seed, "order_value")
    rng_disc = _rng(seed, "order_discount")
    rng_ref = _rng(seed, "order_refund")
    rng_cust = _rng(seed, "order_customer")

    # New-customer share tracks how much of the day's revenue came from
    # prospecting-style channels.
    channel_names = list(CHANNELS)
    prospecting = [channel_names.index(c) for c in ("Meta prospecting", "Google shopping", "TikTok")]
    total_inc = true_inc.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        prospect_share = np.where(revenue > 0, true_inc[prospecting].sum(axis=0) / revenue, 0.0)
    new_share = np.clip(0.30 + 0.45 * prospect_share, 0.22, 0.72)

    promo = set(_promo_day_idx())
    aov = AOV_BASE * _trend(1.00, 1.06)

    order_no = 0
    chunks = []
    for d in range(N_DAYS):
        for r in range(N_REGIONS):
            total = revenue[d, r]
            n = max(1, int(round(total / aov[d])))
            raw = rng_val.lognormal(mean=np.log(aov[d]) - 0.5 * 0.45**2, sigma=0.45, size=n)
            values = np.round(raw * (total / raw.sum()), 2)
            # Push the rounding residual onto the largest order so the region-day
            # total is preserved to the cent.
            values[int(values.argmax())] += round(total - values.sum(), 2)
            values = np.round(values, 2)  # keep every order value an exact cent

            disc_rate = 0.58 if d in promo else 0.20
            has_disc = rng_disc.random(n) < disc_rate
            pct = rng_disc.choice([0.10, 0.15, 0.20, 0.25], size=n)
            discount = np.round(values * pct * has_disc, 2)

            refunded = rng_ref.random(n) < 0.028
            full = rng_ref.random(n) < 0.60
            refund = np.round(values * np.where(full, 1.0, 0.5) * refunded, 2)

            is_new = rng_cust.random(n) < new_share[d, r]
            chunks.append(
                pd.DataFrame(
                    {
                        "order_id": [f"NG-{order_no + i:06d}" for i in range(n)],
                        "date": dates[d],
                        "region": regions[r],
                        "revenue": values,
                        "discount": discount,
                        "refund": refund,
                        "items": np.clip(np.rint(values / 42.0), 1, 8).astype("int64"),
                        "customer_type": np.where(is_new, "new", "returning"),
                    }
                )
            )
            order_no += n

    orders = pd.concat(chunks, ignore_index=True)
    orders["net_revenue"] = (orders["revenue"] - orders["discount"] - orders["refund"]).round(2)
    return orders[
        ["order_id", "date", "region", "revenue", "discount", "refund", "net_revenue", "items", "customer_type"]
    ]


# --------------------------------------------------------------------------
# GA4 sessions
# --------------------------------------------------------------------------


def build_ga4(seed: int, ad_spend: pd.DataFrame, baseline_revenue: np.ndarray) -> pd.DataFrame:
    """Sessions by date / region / source-medium / landing page."""
    dates = _dates()
    regions = _regions()
    date_pos = {d: i for i, d in enumerate(dates)}
    region_pos = {r: i for i, r in enumerate(regions)}

    rng_paid = _rng(seed, "ga4_paid")
    rng_unpaid = _rng(seed, "ga4_unpaid")
    rng_land = _rng(seed, "ga4_landing")

    # Paid sessions come from clicks. Undo the reporting lag first: the clicks
    # really happened, GA4 just has its own (different) lag.
    lag_mult = np.ones(N_DAYS)
    for back, factor in LAG_FACTORS.items():
        lag_mult[N_DAYS - 1 - back] = factor
    ga4_lag = np.ones(N_DAYS)
    for back, factor in GA4_LAG_FACTORS.items():
        ga4_lag[N_DAYS - 1 - back] = factor

    clicks = np.zeros((len(LANDING_PAGES), N_DAYS, N_REGIONS))
    source_names = list(LANDING_PAGES)
    grouped = ad_spend.groupby(["channel", "date", "region"], observed=True)["clicks"].sum()
    for channel, sm in PAID_SOURCE_MEDIUM.items():
        si = source_names.index(sm)
        sub = grouped.loc[channel].reset_index()
        di = sub["date"].map(date_pos).to_numpy()
        ri = sub["region"].map(region_pos).to_numpy()
        np.add.at(clicks[si], (di, ri), sub["clicks"].to_numpy())
    clicks /= lag_mult[None, :, None]

    sessions = np.zeros_like(clicks)
    for si, sm in enumerate(source_names):
        if sm in PAID_SOURCE_MEDIUM.values():
            sessions[si] = clicks[si] * 0.93 * _lognormal(rng_paid, (N_DAYS, N_REGIONS), 0.05)
        else:
            sessions[si] = (
                baseline_revenue
                * BASELINE_SPLIT[sm]
                / REVENUE_PER_SESSION
                * _lognormal(rng_unpaid, (N_DAYS, N_REGIONS), 0.06)
            )
    sessions = np.rint(sessions * ga4_lag[None, :, None]).astype("int64")

    rows = []
    for si, sm in enumerate(source_names):
        pages = list(LANDING_PAGES[sm])
        pvals = np.array(list(LANDING_PAGES[sm].values()))
        flat = sessions[si].ravel()
        split = rng_land.multinomial(flat, pvals / pvals.sum())
        for pi, page in enumerate(pages):
            rows.append(
                pd.DataFrame(
                    {
                        "date": np.repeat(dates.values, N_REGIONS),
                        "region": np.tile(regions, N_DAYS),
                        "source_medium": sm,
                        "landing_page": page,
                        "sessions": split[:, pi],
                    }
                )
            )

    ga4 = pd.concat(rows, ignore_index=True)
    ga4 = ga4[ga4["sessions"] > 0]

    # Two days in week 8 never made it into GA4.
    missing = [dates[_week_days(GA4_MISSING_WEEK)[o]] for o in GA4_MISSING_DAY_OFFSETS]
    ga4 = ga4[~ga4["date"].isin(missing)]

    return ga4.sort_values(["date", "region", "source_medium", "landing_page"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Truth
# --------------------------------------------------------------------------


def build_truth(
    seed: int,
    ad_spend: pd.DataFrame,
    orders: pd.DataFrame,
    true_inc: np.ndarray,
    cf_inc: np.ndarray,
    demand_noise: np.ndarray,
    meta: dict,
) -> dict:
    dates = _dates()
    regions = _regions()
    channel_names = meta["channel_names"]

    realised_inc = true_inc * demand_noise[None, :, :]
    realised_cf = cf_inc * demand_noise[None, :, :]

    shopify_revenue = float(orders["revenue"].sum())
    attributed = float(ad_spend["platform_attributed_revenue"].sum())
    spend_by_channel = ad_spend.groupby("channel")["spend"].sum()

    channel_truth = []
    for ci, channel in enumerate(channel_names):
        cfg = CHANNELS[channel]
        inc = float(realised_inc[ci].sum())
        spend = float(spend_by_channel[channel])
        att = float(ad_spend.loc[ad_spend["channel"] == channel, "platform_attributed_revenue"].sum())
        channel_truth.append(
            {
                "channel": channel,
                "spend": round(spend, 2),
                "platform_attributed_revenue": round(att, 2),
                "reported_roas": round(att / spend, 4),
                "true_incremental_revenue": round(inc, 2),
                "true_incremental_roas": round(inc / spend, 4),
                "true_over_reported": round((inc / spend) / (att / spend), 4),
                "designed_true_iroas": cfg["true_iroas"],
                "designed_reported_roas": cfg["reported_roas"],
            }
        )
    ranked = sorted(channel_truth, key=lambda c: c["true_over_reported"])

    holdouts = []
    for hold in _holdout_specs():
        ci = channel_names.index(hold["channel"])
        test_idx = [regions.index(r) for r in hold["test_regions"]]
        ctrl_idx = [regions.index(r) for r in hold["control_regions"]]
        days = hold["day_idx"]
        lost = float(realised_cf[ci][np.ix_(days, test_idx)].sum() - realised_inc[ci][np.ix_(days, test_idx)].sum())
        cf_spend_paused = float(
            (cf_inc[ci][np.ix_(days, test_idx)] / CHANNELS[hold["channel"]]["true_iroas"]).sum()
        )
        test_rev = float(
            sum(
                orders.loc[
                    orders["region"].isin(hold["test_regions"])
                    & orders["date"].between(dates[days[0]], dates[days[-1]]),
                    "revenue",
                ]
            )
        )
        holdouts.append(
            {
                "channel": hold["channel"],
                "weeks": [hold["start_week"], hold["end_week"]],
                "start_date": hold["start_date"],
                "end_date": hold["end_date"],
                "test_regions": hold["test_regions"],
                "control_regions": hold["control_regions"],
                "paused_spend_counterfactual": round(cf_spend_paused, 2),
                "true_incremental_revenue_lost": round(lost, 2),
                "true_lift_pct_of_test_region_revenue": round(100 * lost / (test_rev + lost), 3),
                "implied_true_iroas": round(lost / cf_spend_paused, 4) if cf_spend_paused else None,
            }
        )

    fatigue = ad_spend[(ad_spend["ad_set"] == FATIGUE_AD_SET)]
    f_days = [dates[d] for d in meta["fatigue_days"]]
    prior_days = [dates[d] for d in _week_days(FATIGUE_WEEK - 1)]
    f_week = fatigue[fatigue["date"].isin(f_days)]
    p_week = fatigue[fatigue["date"].isin(prior_days)]

    truth = {
        "brand": BRAND,
        "disclaimer": DISCLAIMER,
        "generator": {
            "seed": seed,
            "start_date": START_DATE.isoformat(),
            "end_date": dates[-1].date().isoformat(),
            "n_weeks": N_WEEKS,
            "n_days": N_DAYS,
            "regions": regions,
            "paid_channels": channel_names,
            "unpaid_channels": UNPAID_CHANNELS,
            "revenue_field": "shopify_orders.revenue (gross, before discount and refund)",
        },
        "attribution_gap": {
            "note": (
                "Platform-attributed revenue summed across channels far exceeds "
                "what the store actually took. Double-counted conversions, "
                "view-through credit and overlapping windows."
            ),
            "shopify_revenue": round(shopify_revenue, 2),
            "platform_attributed_revenue": round(attributed, 2),
            "ratio": round(attributed / shopify_revenue, 4),
            "designed_ratio": ATTRIBUTION_RATIO,
        },
        "channel_truth": channel_truth,
        "incrementality_ranking": {
            "note": "true_over_reported = true incremental ROAS / platform-reported ROAS. Lower means more overstated.",
            "most_overstated": [c["channel"] for c in ranked[:2]],
            "closest_to_reported": ranked[-1]["channel"],
            "order_most_to_least_overstated": [c["channel"] for c in ranked],
        },
        "geo_holdouts": holdouts,
        "anomalies": {
            "creative_fatigue": {
                "table": "ad_spend",
                "week": FATIGUE_WEEK,
                "channel": "Meta prospecting",
                "ad_set": FATIGUE_AD_SET,
                "start_date": f_days[0].date().isoformat(),
                "end_date": f_days[-1].date().isoformat(),
                "frequency_prior_week": round(float(p_week["frequency"].mean()), 3),
                "frequency_fatigue_week": round(float(f_week["frequency"].mean()), 3),
                "frequency_multiple": round(float(f_week["frequency"].mean() / p_week["frequency"].mean()), 3),
                "reported_roas_prior_week": round(
                    float(p_week["platform_attributed_revenue"].sum() / p_week["spend"].sum()), 4
                ),
                "reported_roas_fatigue_week": round(
                    float(f_week["platform_attributed_revenue"].sum() / f_week["spend"].sum()), 4
                ),
            },
            "meta_pixel_double_count": {
                "table": "ad_spend",
                "week": PIXEL_WEEK,
                "channels": META_CHANNELS,
                "dates": [dates[d].date().isoformat() for d in meta["pixel_days"]],
                "multiplier": PIXEL_MULT,
                "note": "platform_attributed_revenue only; spend, clicks and Shopify revenue are unaffected.",
            },
            "ga4_missing_days": {
                "table": "ga4_sessions",
                "week": GA4_MISSING_WEEK,
                "dates": [
                    dates[_week_days(GA4_MISSING_WEEK)[o]].date().isoformat() for o in GA4_MISSING_DAY_OFFSETS
                ],
                "note": "No rows at all for these dates. Shopify and ad_spend still have them.",
            },
            "reporting_lag": {
                "tables": ["ad_spend", "ga4_sessions"],
                "dates": [dates[-2].date().isoformat(), dates[-1].date().isoformat()],
                "ad_spend_share_of_final": [LAG_FACTORS[1], LAG_FACTORS[0]],
                "ga4_share_of_final": [GA4_LAG_FACTORS[1], GA4_LAG_FACTORS[0]],
                "note": "Numbers are still filling in. shopify_orders is complete for these dates.",
            },
        },
        "other_features": {
            "promo_window": {
                "week": PROMO_WEEK,
                "dates": [dates[d].date().isoformat() for d in _promo_day_idx()],
                "note": "Higher demand and a much higher share of discounted orders.",
            },
            "matched_region_pairs": [list(p) for p in _region_pairs()],
        },
    }
    return truth


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def generate(out_dir: str | Path = "data", seed: int = SEED) -> dict:
    """Generate every table and write it to ``out_dir``. Returns the truth dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ad_spend, true_inc, cf_inc, meta = build_ad_spend(seed)
    attributed_total = float(ad_spend["platform_attributed_revenue"].sum())
    revenue, baseline_revenue, demand_noise = build_revenue(seed, true_inc, attributed_total)
    orders = build_orders(seed, revenue, true_inc)
    ga4 = build_ga4(seed, ad_spend, baseline_revenue)
    truth = build_truth(seed, ad_spend, orders, true_inc, cf_inc, demand_noise, meta)

    orders.to_parquet(out / "shopify_orders.parquet", index=False)
    ad_spend.to_parquet(out / "ad_spend.parquet", index=False)
    ga4.to_parquet(out / "ga4_sessions.parquet", index=False)
    (out / "truth.json").write_text(json.dumps(truth, indent=2, sort_keys=False) + "\n")
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate sample data for {BRAND}.")
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    args = parser.parse_args()

    truth = generate(args.out, args.seed)
    gap = truth["attribution_gap"]
    print(f"{BRAND}: {truth['generator']['n_weeks']} weeks -> {args.out}/")
    print(f"  shopify revenue      {gap['shopify_revenue']:>14,.2f}")
    print(f"  platform attributed  {gap['platform_attributed_revenue']:>14,.2f}  ({gap['ratio']}x)")
    for c in truth["channel_truth"]:
        print(
            f"  {c['channel']:<24} reported {c['reported_roas']:.2f}  "
            f"true {c['true_incremental_roas']:.2f}  ({c['true_over_reported']:.0%} of reported)"
        )


if __name__ == "__main__":
    main()
