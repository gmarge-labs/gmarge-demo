"""Metrics over the Northfield Goods sample dataset.

Everything here is computed in pandas. No module in this package calls an AI
API -- read-outs are generated offline from these numbers and the model's only
job is to phrase them (see CLAUDE.md).

Nothing here reads an answer out of ``truth.json``. The one thing taken from it
is the geo-holdout *test plan* -- which regions were paused and when -- which is
what an analyst would have from the media plan. The lift, the incremental ROAS
and their confidence intervals are all measured from the tables.

Two things worth knowing about the data before reading these numbers:

* The final two days are still filling in on the ad platforms and in GA4, so
  daily ad spend, attributed revenue and sessions are understated there.
  Shopify is complete. Nothing in this module smooths that over.
* ``revenue`` is gross, before discounts and refunds. The reconciliation
  reports ``shopify_net_revenue`` alongside it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = "data"
DEFAULT_CONFIDENCE = 0.90
DEFAULT_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260921


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tables:
    """The three sample tables plus the truth file that ships with them."""

    orders: pd.DataFrame
    ad_spend: pd.DataFrame
    ga4: pd.DataFrame
    truth: dict


def load_tables(data_dir: str | Path = DEFAULT_DATA_DIR) -> Tables:
    path = Path(data_dir)
    return Tables(
        orders=pd.read_parquet(path / "shopify_orders.parquet"),
        ad_spend=pd.read_parquet(path / "ad_spend.parquet"),
        ga4=pd.read_parquet(path / "ga4_sessions.parquet"),
        truth=json.loads((path / "truth.json").read_text()),
    )


def _week_columns(dates: pd.Series, anchor: pd.Timestamp) -> pd.DataFrame:
    """Week start (Monday-anchored on the first day of data) and 1-based index."""
    offset = (dates - anchor).dt.days // 7
    return pd.DataFrame(
        {"week": offset + 1, "week_start": anchor + pd.to_timedelta(offset * 7, unit="D")},
        index=dates.index,
    )


# --------------------------------------------------------------------------
# Reconciliation: what the store took vs what the platforms claimed
# --------------------------------------------------------------------------


def daily_reconciliation(orders: pd.DataFrame, ad_spend: pd.DataFrame) -> pd.DataFrame:
    """Shopify revenue against summed platform-attributed revenue, by day.

    ``over_claim_ratio`` is attributed / actual. Above 1.0 means the platforms
    between them claimed more revenue than the store took -- the same order
    credited two or three times.
    """
    shop = orders.groupby("date")[["revenue", "net_revenue"]].sum()
    shop.columns = ["shopify_revenue", "shopify_net_revenue"]
    ads = ad_spend.groupby("date")[["spend", "platform_attributed_revenue"]].sum()
    ads.columns = ["ad_spend", "platform_attributed_revenue"]

    out = shop.join(ads, how="outer").fillna(0.0).reset_index()
    out["over_claimed_revenue"] = out["platform_attributed_revenue"] - out["shopify_revenue"]
    out["over_claim_ratio"] = out["platform_attributed_revenue"] / out["shopify_revenue"]
    return out


def weekly_reconciliation(
    orders: pd.DataFrame, ad_spend: pd.DataFrame, anchor: pd.Timestamp | None = None
) -> pd.DataFrame:
    """The same reconciliation rolled up to weeks.

    Ratios are recomputed on the weekly totals rather than averaged across
    days, so a partial day cannot drag the week's ratio around.
    """
    daily = daily_reconciliation(orders, ad_spend)
    anchor = pd.Timestamp(anchor) if anchor is not None else daily["date"].min()
    daily = pd.concat([daily, _week_columns(daily["date"], anchor)], axis=1)

    out = daily.groupby(["week", "week_start"], as_index=False).agg(
        week_end=("date", "max"),
        days=("date", "count"),
        shopify_revenue=("shopify_revenue", "sum"),
        shopify_net_revenue=("shopify_net_revenue", "sum"),
        ad_spend=("ad_spend", "sum"),
        platform_attributed_revenue=("platform_attributed_revenue", "sum"),
    )
    out["over_claimed_revenue"] = out["platform_attributed_revenue"] - out["shopify_revenue"]
    out["over_claim_ratio"] = out["platform_attributed_revenue"] / out["shopify_revenue"]
    return out


# --------------------------------------------------------------------------
# Reported ROAS
# --------------------------------------------------------------------------


def reported_roas_by_channel_week(
    ad_spend: pd.DataFrame, anchor: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Each channel's platform-reported ROAS, by week.

    This is the number the ad platform shows. It is not incremental -- see
    :func:`holdout_results` for what the same channels are actually worth.
    """
    anchor = pd.Timestamp(anchor) if anchor is not None else ad_spend["date"].min()
    frame = pd.concat([ad_spend, _week_columns(ad_spend["date"], anchor)], axis=1)

    out = frame.groupby(["week", "week_start", "channel"], as_index=False).agg(
        spend=("spend", "sum"),
        platform_attributed_revenue=("platform_attributed_revenue", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
    )
    out["reported_roas"] = out["platform_attributed_revenue"] / out["spend"]
    return out.sort_values(["week", "channel"]).reset_index(drop=True)


def channel_summary(ad_spend: pd.DataFrame) -> pd.DataFrame:
    """Whole-period spend and reported ROAS per channel."""
    out = ad_spend.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"),
        platform_attributed_revenue=("platform_attributed_revenue", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
    )
    out["reported_roas"] = out["platform_attributed_revenue"] / out["spend"]
    out["share_of_spend"] = out["spend"] / out["spend"].sum()
    return out.sort_values("spend", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Geo holdouts: difference-in-differences with a bootstrap interval
# --------------------------------------------------------------------------


def holdout_plan(truth: dict) -> list[dict]:
    """The test plan only -- channel, dates, and which regions were paused.

    Deliberately drops the measured truth: the estimators below must not see
    the answer they are being asked to recover.
    """
    keep = ("channel", "start_date", "end_date", "test_regions", "control_regions")
    return [{k: h[k] for k in keep} for h in truth["geo_holdouts"]]


def _pair_observations(
    orders: pd.DataFrame, ad_spend: pd.DataFrame, holdout: dict
) -> pd.DataFrame:
    """One row per matched test/control region pair.

    For each pair, the control region predicts what the test region would have
    done: scale the control's in-window figure by the pair's ratio over an
    equally long pre-period. The difference is what the pause cost. Because
    the test region's spend is zero during the window, the spend it *would*
    have had is predicted the same way.
    """
    start, end = pd.Timestamp(holdout["start_date"]), pd.Timestamp(holdout["end_date"])
    length = (end - start).days + 1
    pre_start, pre_end = start - pd.Timedelta(days=length), start - pd.Timedelta(days=1)

    revenue = orders.groupby(["date", "region"])["revenue"].sum().unstack(fill_value=0.0)
    channel = ad_spend[ad_spend["channel"] == holdout["channel"]]
    spend = channel.groupby(["date", "region"])["spend"].sum().unstack(fill_value=0.0)
    attributed = (
        channel.groupby(["date", "region"])["platform_attributed_revenue"].sum().unstack(fill_value=0.0)
    )

    rows = []
    for test, control in zip(holdout["test_regions"], holdout["control_regions"]):
        rev_pre_t = revenue.loc[pre_start:pre_end, test].sum()
        rev_pre_c = revenue.loc[pre_start:pre_end, control].sum()
        rev_post_t = revenue.loc[start:end, test].sum()
        rev_post_c = revenue.loc[start:end, control].sum()

        spend_pre_t = spend.loc[pre_start:pre_end, test].sum()
        spend_pre_c = spend.loc[pre_start:pre_end, control].sum()
        spend_post_c = spend.loc[start:end, control].sum()

        rows.append(
            {
                "test_region": test,
                "control_region": control,
                "counterfactual_revenue": rev_post_c * (rev_pre_t / rev_pre_c),
                "actual_revenue": rev_post_t,
                "counterfactual_spend": spend_post_c * (spend_pre_t / spend_pre_c),
                "control_spend": spend_post_c,
                "control_attributed_revenue": attributed.loc[start:end, control].sum(),
            }
        )

    pairs = pd.DataFrame(rows)
    pairs["incremental_revenue"] = pairs["counterfactual_revenue"] - pairs["actual_revenue"]
    return pairs


def _bootstrap_interval(
    pairs: pd.DataFrame, confidence: float, n_bootstrap: int, seed: int
) -> dict:
    """Resample the matched pairs with replacement.

    The pairs are the independent units here -- one per matched market -- so
    resampling them is what the interval should reflect. With five pairs the
    interval is wide; that is an honest reading of a five-market test, not a
    defect in the estimator.
    """
    lift = pairs["incremental_revenue"].tolist()
    cf_spend = pairs["counterfactual_spend"].tolist()
    cf_revenue = pairs["counterfactual_revenue"].tolist()
    k = len(lift)

    rng = random.Random(seed)
    lifts, roas, pcts = [], [], []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(k) for _ in range(k)]
        total_lift = sum(lift[i] for i in idx)
        total_spend = sum(cf_spend[i] for i in idx)
        total_revenue = sum(cf_revenue[i] for i in idx)
        lifts.append(total_lift)
        roas.append(total_lift / total_spend if total_spend else float("nan"))
        pcts.append(total_lift / total_revenue if total_revenue else float("nan"))

    tail = (1.0 - confidence) / 2.0
    quantiles = [tail, 1.0 - tail]
    lift_lo, lift_hi = pd.Series(lifts).quantile(quantiles)
    roas_lo, roas_hi = pd.Series(roas).quantile(quantiles)
    pct_lo, pct_hi = pd.Series(pcts).quantile(quantiles)
    return {
        "incremental_revenue_ci_low": lift_lo,
        "incremental_revenue_ci_high": lift_hi,
        "incremental_roas_ci_low": roas_lo,
        "incremental_roas_ci_high": roas_hi,
        "lift_pct_ci_low": pct_lo,
        "lift_pct_ci_high": pct_hi,
    }


def holdout_results(
    orders: pd.DataFrame,
    ad_spend: pd.DataFrame,
    plan: list[dict],
    confidence: float = DEFAULT_CONFIDENCE,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Measure every geo holdout: incremental lift, incremental ROAS, interval.

    One row per holdout. ``incremental_roas`` is the lift divided by the spend
    the paused regions would have had; ``reported_roas_in_window`` is what the
    platform claimed for the same channel over the same dates in the control
    regions, so the two are directly comparable.
    """
    rows = []
    for holdout in plan:
        pairs = _pair_observations(orders, ad_spend, holdout)
        lift = pairs["incremental_revenue"].sum()
        cf_spend = pairs["counterfactual_spend"].sum()
        cf_revenue = pairs["counterfactual_revenue"].sum()
        reported = pairs["control_attributed_revenue"].sum() / pairs["control_spend"].sum()
        incremental_roas = lift / cf_spend

        rows.append(
            {
                "channel": holdout["channel"],
                "start_date": pd.Timestamp(holdout["start_date"]),
                "end_date": pd.Timestamp(holdout["end_date"]),
                "n_pairs": len(pairs),
                "actual_revenue": pairs["actual_revenue"].sum(),
                "counterfactual_revenue": cf_revenue,
                "incremental_revenue": lift,
                "lift_pct": lift / cf_revenue,
                "paused_spend_estimate": cf_spend,
                "incremental_roas": incremental_roas,
                "reported_roas_in_window": reported,
                "over_claim_multiple": reported / incremental_roas,
                "confidence": confidence,
                **_bootstrap_interval(pairs, confidence, n_bootstrap, seed),
            }
        )

    columns = [
        "channel", "start_date", "end_date", "n_pairs",
        "actual_revenue", "counterfactual_revenue",
        "incremental_revenue", "incremental_revenue_ci_low", "incremental_revenue_ci_high",
        "lift_pct", "lift_pct_ci_low", "lift_pct_ci_high",
        "paused_spend_estimate",
        "incremental_roas", "incremental_roas_ci_low", "incremental_roas_ci_high",
        "reported_roas_in_window", "over_claim_multiple", "confidence",
    ]
    return pd.DataFrame(rows)[columns].sort_values("start_date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Everything at once
# --------------------------------------------------------------------------


def all_metrics(
    source: str | Path | Tables = DEFAULT_DATA_DIR,
    confidence: float = DEFAULT_CONFIDENCE,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Every metric the app and the read-out generator need, in one call.

    Returns a dict of plain floats, strings and DataFrames -- no AI calls, no
    hidden state. ``source`` is a data directory or an already-loaded
    :class:`Tables`.
    """
    tables = source if isinstance(source, Tables) else load_tables(source)
    orders, ad_spend, truth = tables.orders, tables.ad_spend, tables.truth
    anchor = orders["date"].min()

    daily = daily_reconciliation(orders, ad_spend)
    weekly = weekly_reconciliation(orders, ad_spend, anchor)
    roas_week = reported_roas_by_channel_week(ad_spend, anchor)
    channels = channel_summary(ad_spend)
    holdouts = holdout_results(orders, ad_spend, holdout_plan(truth), confidence, n_bootstrap, seed)

    shopify_revenue = float(orders["revenue"].sum())
    attributed = float(ad_spend["platform_attributed_revenue"].sum())
    spend = float(ad_spend["spend"].sum())
    measured_iroas = float(
        (holdouts["incremental_roas"] * holdouts["paused_spend_estimate"]).sum()
        / holdouts["paused_spend_estimate"].sum()
    )

    return {
        "brand": truth["brand"],
        "disclaimer": truth["disclaimer"],
        "window": {
            "start": orders["date"].min().date().isoformat(),
            "end": orders["date"].max().date().isoformat(),
            "n_days": int(orders["date"].nunique()),
            "n_weeks": int(weekly["week"].max()),
        },
        "headline": {
            "shopify_revenue": shopify_revenue,
            "shopify_net_revenue": float(orders["net_revenue"].sum()),
            "platform_attributed_revenue": attributed,
            "over_claimed_revenue": attributed - shopify_revenue,
            "over_claim_ratio": attributed / shopify_revenue,
            "ad_spend": spend,
            "blended_reported_roas": attributed / spend,
            "measured_incremental_roas": measured_iroas,
            "n_orders": int(len(orders)),
            "average_order_value": shopify_revenue / len(orders),
        },
        "daily_reconciliation": daily,
        "weekly_reconciliation": weekly,
        "reported_roas_by_channel_week": roas_week,
        "channel_summary": channels,
        "holdouts": holdouts,
    }
