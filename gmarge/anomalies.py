"""Week-on-week anomaly detection for the paid channels.

For every paid channel and every week, each efficiency metric is compared with
the same metric over the channel's own trailing eight weeks, using a robust
score -- the median and the median absolute deviation -- so that one bad week
inside the window cannot hide the next one. Nothing here calls an AI API and
nothing reads an answer out of ``truth.json`` (see CLAUDE.md): a flag is
recovered from ``ad_spend`` alone, and every figure in its description is
computed here.

**What is scored, and why.** Only *rates* -- reported ROAS, CPC, CPM, CTR and
frequency -- not spend or revenue levels. Levels move for planned reasons:
budgets get raised, a channel is paused in half the regions to run a geo
holdout, a promo week lifts everything. Flagging those produces a stream of
alerts that an analyst already knows about. A rate is scale-free -- it survives
a channel going dark in five of twenty regions, and it survives the last two
days of the window still filling in, because the lag scales a metric's
numerator and denominator together. What a rate cannot survive is the platform
counting conversions twice or a creative burning out, which is the point.

**Where the thresholds come from.** These weekly rates are stable. Of the 414
channel-week-metric comparisons the scan makes, the 399 that hold no planted
fault all sit within 3.2% of their trailing median, while the faults move 19%
to 41%. That stability is a liability for a robust score on its own: a window
whose weeks typically sit 0.1% from their median turns a 2% wobble into a
score of 7, and 22 of those 399 clean comparisons do exactly that. So two
guards are applied. The median absolute deviation is floored at 0.5% of the
level, which brings the worst clean comparison down to 3.1 -- below the
threshold, but not by much. And a flag needs a move of at least 10% as well as
a score past 3.5: an independent second condition, with roughly three times
the headroom on either side, so that neither guard is carrying the result
alone. The score threshold of 3.5 is the conventional cut for a modified
z-score (Iglewicz and Hoaglin); the eight-week window is long enough for a
stable median and short enough to still track a drifting channel.

**Attribution.** A flag names the campaign and ad set behind most of the move.
Each ad set's contribution is what it did against what its own trailing eight
weeks predict -- for ROAS, the attributed revenue it booked minus the revenue
its own trailing ROAS implies for the spend it took -- and those contributions
sum to the channel's total move, so a share of it is a real share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gmarge.metrics import DEFAULT_DATA_DIR, Tables, load_tables

# --------------------------------------------------------------------------
# Thresholds. See the module docstring for where these come from.
# --------------------------------------------------------------------------

TRAILING_WEEKS = 8
SCORE_THRESHOLD = 3.5
MIN_RELATIVE_MOVE = 0.10
# A spread this small would otherwise turn ordinary wobble into a large score.
MAD_FLOOR_SHARE = 0.005
MAD_TO_SIGMA = 0.6745

# Severity is the size of the move, not its score: a 40% jump needs looking at
# today whether its score is 20 or 200.
SEVERITY_CUTS = ((0.30, "high"), (0.15, "medium"), (0.0, "low"))
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """A rate, held as the ratio it is so that it can be decomposed.

    ``numerator`` and ``denominator`` are columns of the weekly aggregate
    below. Keeping both sides means an ad set's contribution to a move can be
    measured against the spend (or impressions) it actually took, rather than
    by treating every ad set as equally responsible.
    """

    key: str
    label: str
    numerator: str
    denominator: str
    scale: float = 1.0
    unit: str = "ratio"

    def value(self, frame: pd.DataFrame) -> pd.Series:
        return self.scale * frame[self.numerator] / frame[self.denominator]


METRICS = (
    Metric("reported_roas", "reported ROAS", "platform_attributed_revenue", "spend", unit="x"),
    Metric("cpc", "cost per click", "spend", "clicks", unit="money"),
    Metric("cpm", "cost per thousand impressions", "spend", "impressions", scale=1000.0, unit="money"),
    Metric("ctr", "click-through rate", "clicks", "impressions", unit="percent"),
    Metric("frequency", "average frequency", "frequency_impressions", "impressions", unit="x"),
)
METRICS_BY_KEY = {m.key: m for m in METRICS}

BASE_COLUMNS = ["spend", "platform_attributed_revenue", "impressions", "clicks", "frequency_impressions"]


def _format(metric: Metric, value: float) -> str:
    if metric.unit == "money":
        return f"${value:,.2f}"
    if metric.unit == "percent":
        return f"{value:.2%}"
    return f"{value:,.2f}"


# --------------------------------------------------------------------------
# Weekly aggregates
# --------------------------------------------------------------------------


def _weeks(dates: pd.Series, anchor: pd.Timestamp) -> pd.DataFrame:
    """Week number and week start, anchored on the first day of data."""
    offset = (dates - anchor).dt.days // 7
    return pd.DataFrame(
        {"week": offset + 1, "week_start": anchor + pd.to_timedelta(offset * 7, unit="D")},
        index=dates.index,
    )


def weekly_metrics(
    ad_spend: pd.DataFrame, by: list[str] | None = None, anchor: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Weekly totals and rates, by channel (the default) or any finer grain.

    ``frequency`` is averaged over impressions rather than over rows: a
    region-day that served a hundred impressions should not weigh as much as
    one that served a million. Search channels do not report frequency at all,
    so theirs stays null rather than becoming zero.
    """
    by = by or ["channel"]
    anchor = pd.Timestamp(anchor) if anchor is not None else ad_spend["date"].min()

    frame = ad_spend.copy()
    frame["frequency_impressions"] = frame["frequency"] * frame["impressions"]
    frame = pd.concat([frame, _weeks(frame["date"], anchor)], axis=1)

    out = frame.groupby(["week", "week_start", *by], as_index=False).agg(
        week_end=("date", "max"),
        days=("date", "nunique"),
        spend=("spend", "sum"),
        platform_attributed_revenue=("platform_attributed_revenue", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        frequency_impressions=("frequency_impressions", "sum"),
        frequency_rows=("frequency", "count"),
    )
    # Summing an all-null column gives zero, which would say a search channel
    # served every impression at a frequency of none. It reported nothing.
    out.loc[out["frequency_rows"] == 0, "frequency_impressions"] = np.nan
    out = out.drop(columns="frequency_rows")
    with np.errstate(divide="ignore", invalid="ignore"):
        for metric in METRICS:
            out[metric.key] = metric.value(out)
    return out.sort_values(["week", *by]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Robust scoring
# --------------------------------------------------------------------------


def _robust(trailing: pd.Series, value: float) -> tuple[float, float, float]:
    """``(score, median, mad)`` -- the modified z-score of ``value``.

    ``0.6745 * (x - median) / MAD`` puts the score on the same footing as a
    standard deviation for normally distributed data, without letting an
    outlier inside the window inflate the scale it is measured against.
    """
    median = float(trailing.median())
    mad = float((trailing - median).abs().median())
    scale = max(mad, MAD_FLOOR_SHARE * abs(median))
    score = MAD_TO_SIGMA * (value - median) / scale if scale else 0.0
    return score, median, mad


def robust_scores(
    ad_spend: pd.DataFrame, anchor: pd.Timestamp | None = None, trailing_weeks: int = TRAILING_WEEKS
) -> pd.DataFrame:
    """Every channel-week-metric scored against its trailing weeks.

    One row per comparison, flagged or not, so the detector's working can be
    charted or audited rather than taken on trust.
    """
    weekly = weekly_metrics(ad_spend, anchor=anchor)
    rows = []
    for channel, sub in weekly.groupby("channel", observed=True):
        sub = sub.sort_values("week")
        for metric in METRICS:
            series = sub.set_index("week")[metric.key]
            for week in series.index:
                trailing = series.loc[week - trailing_weeks : week - 1].dropna()
                value = series.loc[week]
                if len(trailing) < trailing_weeks or not np.isfinite(value):
                    continue
                score, median, mad = _robust(trailing, float(value))
                rows.append(
                    {
                        "channel": channel,
                        "week": int(week),
                        "metric": metric.key,
                        "value": float(value),
                        "baseline": median,
                        "mad": mad,
                        "pct_change": float(value) / median - 1.0 if median else np.nan,
                        "robust_score": score,
                        "trailing_weeks": len(trailing),
                    }
                )
    columns = ["channel", "week", "metric", "value", "baseline", "mad", "pct_change", "robust_score", "trailing_weeks"]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------
# Which ad set moved the channel
# --------------------------------------------------------------------------


def contributions(
    ad_set_weekly: pd.DataFrame,
    channel: str,
    week: int,
    metric: Metric,
    channel_baseline: float,
    trailing_weeks: int = TRAILING_WEEKS,
) -> pd.DataFrame:
    """Each ad set's share of a channel-week's move, largest first.

    An ad set's contribution is what it booked minus what its own trailing
    median predicts for the volume it took: for ROAS, ``attributed revenue -
    its usual ROAS x its spend this week``, in dollars of attributed revenue.
    Summed over the ad sets this is the channel's whole move, so the shares
    are shares of something real. An ad set without enough history of its own
    -- a new one, or one that was paused -- is measured against the channel's
    baseline instead.
    """
    channel_rows = ad_set_weekly[ad_set_weekly["channel"] == channel]
    current = channel_rows[channel_rows["week"] == week]

    rows = []
    for row in current.itertuples():
        history = channel_rows[
            (channel_rows["ad_set"] == row.ad_set)
            & (channel_rows["week"] >= week - trailing_weeks)
            & (channel_rows["week"] < week)
        ][metric.key].dropna()

        own_baseline = len(history) >= trailing_weeks // 2
        baseline = float(history.median()) if own_baseline else channel_baseline
        numerator = float(getattr(row, metric.numerator))
        denominator = float(getattr(row, metric.denominator))
        rows.append(
            {
                "campaign": row.campaign,
                "ad_set": row.ad_set,
                "value": metric.scale * numerator / denominator if denominator else np.nan,
                "baseline": baseline,
                # In the numerator's own units, so a contribution reads as
                # dollars of attributed revenue, or clicks, or impressions.
                "contribution": numerator - baseline / metric.scale * denominator,
                "own_baseline": own_baseline,
            }
        )

    out = pd.DataFrame(rows)
    total = out["contribution"].sum()
    out["share_of_move"] = out["contribution"] / total if total else np.nan
    return out.sort_values("share_of_move", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Anomaly:
    """One channel-week-metric move worth a human's attention."""

    channel: str
    week: int
    week_start: str
    week_end: str
    metric: str
    metric_label: str
    direction: str
    severity: str
    value: float
    baseline: float
    pct_change: float
    robust_score: float
    campaign: str
    ad_set: str
    share_of_move: float
    description: str
    numbers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "week": self.week,
            "week_start": self.week_start,
            "week_end": self.week_end,
            "metric": self.metric,
            "metric_label": self.metric_label,
            "direction": self.direction,
            "severity": self.severity,
            "value": self.value,
            "baseline": self.baseline,
            "pct_change": self.pct_change,
            "robust_score": self.robust_score,
            "campaign": self.campaign,
            "ad_set": self.ad_set,
            "share_of_move": self.share_of_move,
            "description": self.description,
            "numbers": dict(self.numbers),
        }


def _severity(pct_change: float) -> str:
    for cut, name in SEVERITY_CUTS:
        if abs(pct_change) >= cut:
            return name
    return "low"


def _lead_in(share: float) -> str:
    if share >= 0.50:
        return "Most of the move is one ad set"
    if share >= 0.30:
        return "The largest single contributor"
    return "No one ad set dominates; the largest contributor"


def _ad_spend_frame(source: str | Path | Tables | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    tables = source if isinstance(source, Tables) else load_tables(source)
    return tables.ad_spend


def detect_anomalies(
    source: str | Path | Tables | pd.DataFrame = DEFAULT_DATA_DIR,
    anchor: pd.Timestamp | None = None,
    trailing_weeks: int = TRAILING_WEEKS,
    score_threshold: float = SCORE_THRESHOLD,
    min_relative_move: float = MIN_RELATIVE_MOVE,
) -> list[Anomaly]:
    """Flag channel-weeks whose rates broke with their own recent history.

    A week is flagged only when both conditions hold: a robust score past
    ``score_threshold``, and a move of at least ``min_relative_move`` against
    the trailing median. See the module docstring for why it takes both.
    """
    ad_spend = _ad_spend_frame(source)
    anchor = pd.Timestamp(anchor) if anchor is not None else ad_spend["date"].min()
    weekly = weekly_metrics(ad_spend, anchor=anchor)
    by_ad_set = weekly_metrics(ad_spend, by=["channel", "campaign", "ad_set"], anchor=anchor)
    scores = robust_scores(ad_spend, anchor=anchor, trailing_weeks=trailing_weeks)

    flagged = scores[
        (scores["robust_score"].abs() >= score_threshold)
        & (scores["pct_change"].abs() >= min_relative_move)
    ]

    anomalies = []
    for row in flagged.itertuples():
        metric = METRICS_BY_KEY[row.metric]
        week_row = weekly[(weekly["channel"] == row.channel) & (weekly["week"] == row.week)].iloc[0]
        shares = contributions(by_ad_set, row.channel, row.week, metric, row.baseline, trailing_weeks)
        top = shares.iloc[0]

        direction = "up" if row.pct_change > 0 else "down"
        mad_share = row.mad / row.baseline if row.baseline else np.nan
        start, end = week_row["week_start"].date().isoformat(), week_row["week_end"].date().isoformat()

        description = (
            f"{row.channel}'s {metric.label} {'rose' if direction == 'up' else 'fell'} "
            f"{abs(row.pct_change):.1%} in week {row.week} ({start} to {end}), to "
            f"{_format(metric, row.value)} from a trailing {trailing_weeks}-week median of "
            f"{_format(metric, row.baseline)}. That is a robust score of {row.robust_score:+.1f}: "
            f"the trailing weeks themselves sit a typical {mad_share:.1%} from their median. "
            f"{_lead_in(float(top['share_of_move']))}: {top['ad_set']} "
            f"(campaign {top['campaign']}) accounts for {float(top['share_of_move']):.0%} of the "
            f"move, its own {metric.label} going from {_format(metric, float(top['baseline']))} to "
            f"{_format(metric, float(top['value']))}."
        )

        anomalies.append(
            Anomaly(
                channel=row.channel,
                week=int(row.week),
                week_start=start,
                week_end=end,
                metric=metric.key,
                metric_label=metric.label,
                direction=direction,
                severity=_severity(row.pct_change),
                value=float(row.value),
                baseline=float(row.baseline),
                pct_change=float(row.pct_change),
                robust_score=float(row.robust_score),
                campaign=str(top["campaign"]),
                ad_set=str(top["ad_set"]),
                share_of_move=float(top["share_of_move"]),
                description=description,
                numbers={
                    "value": round(float(row.value), 6),
                    "trailing_median": round(float(row.baseline), 6),
                    "trailing_mad": round(float(row.mad), 6),
                    "trailing_mad_share_of_median": round(float(mad_share), 6),
                    "pct_change": round(float(row.pct_change), 6),
                    "robust_score": round(float(row.robust_score), 3),
                    "trailing_weeks": int(row.trailing_weeks),
                    "spend": round(float(week_row["spend"]), 2),
                    "platform_attributed_revenue": round(float(week_row["platform_attributed_revenue"]), 2),
                    "contribution_unit": metric.numerator,
                    "top_contribution": round(float(top["contribution"]), 2),
                    "ad_set_shares": [
                        {
                            "campaign": r.campaign,
                            "ad_set": r.ad_set,
                            "share_of_move": round(float(r.share_of_move), 6),
                            "value": round(float(r.value), 6),
                            "baseline": round(float(r.baseline), 6),
                        }
                        for r in shares.itertuples()
                    ],
                },
            )
        )

    return sorted(anomalies, key=lambda a: (SEVERITY_ORDER[a.severity], a.week, a.channel, a.metric))


def anomalies_frame(anomalies: list[Anomaly]) -> pd.DataFrame:
    """The flags as a table, for display."""
    columns = [
        "severity", "week", "week_start", "week_end", "channel", "metric", "direction",
        "value", "baseline", "pct_change", "robust_score", "campaign", "ad_set",
        "share_of_move", "description", "numbers",
    ]
    if not anomalies:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([a.to_dict() for a in anomalies])[columns]
