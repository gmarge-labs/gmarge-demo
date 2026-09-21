"""Data-quality checks over the Northfield Goods sample tables.

Every check returns :class:`Finding` objects: a severity, the dates involved,
the table the problem is in, a plain-English description, and the numbers the
description is built from. Nothing here calls an AI API and nothing reads an
answer out of ``truth.json`` -- the findings are recovered from the tables the
way an analyst would recover them (see CLAUDE.md). A read-out generator may
later pass a finding's ``numbers`` to a model to phrase; the arithmetic is
already done by then.

Four things are looked for:

1. **Missing days.** A date the calendar expects but a table has no rows for.
2. **Duplicate orders.** The same order counted twice.
3. **Pixel double counting.** A platform's attributed revenue jumping against
   its own spend while the store's own revenue and order count sit still.
4. **Reporting lag.** The most recent days still filling in, so today's
   dashboard understates them.

Thresholds are set from the spread the tables actually show, and each one is
justified where it is defined. They are deliberately loose enough that a small
change of seed cannot flip them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gmarge.metrics import DEFAULT_DATA_DIR, Tables, load_tables

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------

# Missing days. A stream -- one channel, one source/medium, one region -- is
# only expected every day if it normally reports every day. Anything present on
# less than this share of the table's days is a sparse stream, not a gap.
STREAM_COVERAGE = 0.90

# Duplicate orders. Two orders can legitimately look identical: same day, same
# region, same price, same basket, two different shoppers. Across this table
# that coincidence runs at about 0.2% of rows and never exceeds 1% on any one
# day, so a flat "identical rows exist" rule would fire 160-odd times and mean
# nothing. What does mean something is a *date* where the rate explodes, which
# is what a re-ingested batch looks like. Both conditions must hold, and the
# 5% floor sits five times above the worst clean day in the data.
DUP_DATE_SHARE = 0.05
DUP_OVER_BASELINE = 10.0

# Pixel double counting. Compare a platform's attributed revenue against what
# its own spend predicts -- its trailing median revenue per dollar -- rather
# than against the store directly, because a channel going dark in a geo
# holdout moves its share of store revenue without anything being wrong. Day to
# day that figure stays inside +/-11% of its own trailing median here, so 1.40
# is roughly four times beyond the natural range while still sitting far below
# the 1.9x a doubled purchase event produces.
PIXEL_TRAILING_DAYS = 28
PIXEL_MIN_DAYS = 14
PIXEL_RATIO = 1.40
# ...and the store has to have stood still. If Shopify revenue moved with the
# platform, the extra credit is a real sales day, not a counting bug.
STORE_TOLERANCE = 0.25

# Reporting lag. Only the tail of the window is examined -- that is where a lag
# lives -- and the scan stops at the first day that looks settled, so this is a
# dozen comparisons at worst, and usually half that, against the several
# hundred the weekly anomaly scan makes. A lower bar is affordable at that count, and the output is a
# "still filling in" note rather than an alarm. Both conditions must hold: a
# day has to be both statistically odd and materially short.
LAG_TAIL_DAYS = 5
LAG_LOOKBACK_WEEKS = 4
LAG_REFERENCE_DAYS = 28
LAG_MIN_SHORTFALL = 0.08
LAG_SCORE = 3.0

# Guards against a spread so small that ordinary wobble scores as an outlier.
MAD_FLOOR_SHARE = 0.005
MAD_TO_SIGMA = 0.6745


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One data-quality problem, with the evidence behind it."""

    check: str
    severity: str
    source: str
    dates: tuple[str, ...]
    description: str
    numbers: dict = field(default_factory=dict)

    @property
    def start_date(self) -> str:
        return self.dates[0] if self.dates else ""

    @property
    def end_date(self) -> str:
        return self.dates[-1] if self.dates else ""

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "source": self.source,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_days": len(self.dates),
            "dates": list(self.dates),
            "description": self.description,
            "numbers": dict(self.numbers),
        }


def findings_frame(findings: list[Finding]) -> pd.DataFrame:
    """The findings as a table, for display. ``numbers`` is kept as a dict."""
    columns = ["check", "severity", "source", "start_date", "end_date", "n_days", "description", "numbers"]
    if not findings:
        return pd.DataFrame(columns=columns)
    rows = [{k: v for k, v in f.to_dict().items() if k != "dates"} for f in findings]
    return pd.DataFrame(rows)[columns]


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Worst first, then earliest, then by table."""
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.start_date, f.source, f.check))


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _iso(day) -> str:
    return pd.Timestamp(day).date().isoformat()


def _isos(days) -> tuple[str, ...]:
    return tuple(_iso(d) for d in days)


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _count(x: float) -> str:
    return f"{x:,.0f}"


# Columns measured in dollars. Everything else is counted in its own unit, so
# a description can say "$15,736" or "18,654 sessions" without being told which.
MONEY_COLUMNS = {"revenue", "net_revenue", "spend", "platform_attributed_revenue"}


def _volume(column: str, x: float) -> str:
    return _money(x) if column in MONEY_COLUMNS else f"{_count(x)} {column}"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _verb(names: list[str]) -> str:
    return "shows" if len(names) == 1 else "show"


def _span(dates: tuple[str, ...]) -> str:
    if len(dates) == 1:
        return dates[0]
    return f"{dates[0]} to {dates[-1]}"


def _calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Every day the tables between them claim to cover, with no holes."""
    seen = [f["date"] for f in frames.values() if len(f)]
    lo = min(s.min() for s in seen)
    hi = max(s.max() for s in seen)
    return pd.date_range(lo, hi, freq="D")


def _daily(frame: pd.DataFrame, column: str, calendar: pd.DatetimeIndex) -> pd.Series:
    """Daily totals on the full calendar. A day with no rows is NaN, not zero.

    The distinction matters: zero would say the day traded nothing, NaN says
    the day was never reported, and the checks below read them differently.
    """
    return frame.groupby("date")[column].sum().reindex(calendar)


def _runs(days: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    """Split a sorted list of days into consecutive runs."""
    out: list[list[pd.Timestamp]] = []
    for day in sorted(days):
        if out and (day - out[-1][-1]).days == 1:
            out[-1].append(day)
        else:
            out.append([day])
    return out


def _robust_score(reference: pd.Series, value: float) -> tuple[float, float, float]:
    """Modified z-score of ``value`` against ``reference``: median and MAD.

    Returns ``(score, median, mad)``. The MAD is floored at a small share of
    the level so that an unusually quiet reference window cannot turn ordinary
    wobble into a large score.
    """
    reference = reference.dropna()
    median = float(reference.median())
    mad = float((reference - median).abs().median())
    floor = MAD_FLOOR_SHARE * abs(median)
    scale = max(mad, floor)
    score = MAD_TO_SIGMA * (value - median) / scale if scale else 0.0
    return score, median, mad


# --------------------------------------------------------------------------
# 1. Missing days
# --------------------------------------------------------------------------

# The column each table is measured in, and the stream it is expected to
# report every day: a channel for the ad platform, a source/medium for GA4, a
# region for the store.
TABLE_VOLUME = {"shopify_orders": "revenue", "ad_spend": "spend", "ga4_sessions": "sessions"}
TABLE_STREAM = {"shopify_orders": "region", "ad_spend": "channel", "ga4_sessions": "source_medium"}


def _frames(tables: Tables) -> dict[str, pd.DataFrame]:
    return {"shopify_orders": tables.orders, "ad_spend": tables.ad_spend, "ga4_sessions": tables.ga4}


def _week_of(day: pd.Timestamp, calendar: pd.DatetimeIndex) -> int:
    return int((pd.Timestamp(day) - calendar[0]).days // 7) + 1


def check_missing_days(tables: Tables) -> list[Finding]:
    """Days the calendar expects that a table never reported.

    The calendar is the span the three tables cover between them, so a day is
    only called missing when some other table has it. Each unbroken run of
    missing days is one finding: two consecutive dead days are one outage, not
    two problems.
    """
    frames = _frames(tables)
    calendar = _calendar(frames)
    findings: list[Finding] = []

    for name, frame in frames.items():
        column = TABLE_VOLUME[name]
        present = set(frame["date"])
        gaps = [d for d in calendar if d not in present]
        if not gaps:
            continue

        typical = float(_daily(frame, column, calendar).median())
        elsewhere = sorted(n for n, f in frames.items() if n != name and set(gaps) <= set(f["date"]))

        for run in _runs(gaps):
            dates = _isos(run)
            weeks = sorted({_week_of(d, calendar) for d in run})
            lost = typical * len(run)
            also = " and ".join(elsewhere) if elsewhere else "no other table"
            findings.append(
                Finding(
                    check="missing_days",
                    severity="high",
                    source=name,
                    dates=dates,
                    description=(
                        f"{name} has no rows at all on {_plural(len(run), 'day')}, {_span(dates)} "
                        f"(week {weeks[0]}), while {also} cover the same dates. At about "
                        f"{_volume(column, typical)} a day either side of the gap, roughly "
                        f"{_volume(column, lost)} never arrived. Any week-{weeks[0]} "
                        f"total or rate built on {name} is running on "
                        f"{7 - len(run)} days of data, not 7."
                    ),
                    numbers={
                        "n_missing_days": len(run),
                        "days_expected": int(len(calendar)),
                        "days_present": int(len(present)),
                        "weeks": weeks,
                        f"typical_daily_{column}": round(typical, 2),
                        f"estimated_missing_{column}": round(lost, 2),
                        "present_in": elsewhere,
                    },
                )
            )

    findings.extend(_check_missing_streams(frames, calendar))
    return findings


def _check_missing_streams(frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> list[Finding]:
    """One stream stops reporting on a day the rest of the table still covers.

    Only streams that normally report daily are held to that standard; a
    stream present on fewer than ``STREAM_COVERAGE`` of the table's days is
    sparse by nature and is skipped rather than flagged every time it is quiet.
    """
    findings: list[Finding] = []
    for name, frame in frames.items():
        stream_column = TABLE_STREAM[name]
        volume = TABLE_VOLUME[name]
        table_days = sorted(set(frame["date"]))
        if not table_days:
            continue

        for stream, sub in frame.groupby(stream_column, observed=True):
            days = set(sub["date"])
            coverage = len(days) / len(table_days)
            if coverage < STREAM_COVERAGE or coverage == 1.0:
                continue

            gaps = [d for d in table_days if d not in days]
            typical = float(sub.groupby("date")[volume].sum().median())
            for run in _runs(gaps):
                dates = _isos(run)
                findings.append(
                    Finding(
                        check="missing_days",
                        severity="medium",
                        source=name,
                        dates=dates,
                        description=(
                            f"{name} reports on {_span(dates)} but {stream_column} "
                            f"'{stream}' is absent for {_plural(len(run), 'day')}, having reported on "
                            f"{coverage:.0%} of the table's days. At its usual "
                            f"{_count(typical)} {volume} a day that is about "
                            f"{_count(typical * len(run))} {volume} unaccounted for, and the "
                            f"days look complete unless you split by {stream_column}."
                        ),
                        numbers={
                            stream_column: stream,
                            "n_missing_days": len(run),
                            "coverage": round(coverage, 4),
                            f"typical_daily_{volume}": round(typical, 2),
                            f"estimated_missing_{volume}": round(typical * len(run), 2),
                        },
                    )
                )
    return findings


# --------------------------------------------------------------------------
# 2. Duplicate orders
# --------------------------------------------------------------------------

ORDER_FINGERPRINT = ["date", "region", "revenue", "discount", "refund", "items", "customer_type"]


def check_duplicate_orders(orders: pd.DataFrame) -> list[Finding]:
    """The same order counted twice.

    Two passes. A repeated ``order_id`` is unambiguous -- the store's own
    primary key cannot legitimately repeat -- and is reported whenever it
    appears. Repeated *contents* are not: identical orders happen by chance.
    That pass reports a date only when the rate of identical rows on it is far
    above the table's own background rate, which is what a batch loaded twice
    looks like and what a busy trading day does not.
    """
    findings: list[Finding] = []

    repeated_id = orders[orders.duplicated("order_id", keep=False)]
    if not repeated_id.empty:
        extra = len(repeated_id) - repeated_id["order_id"].nunique()
        overstated = float(repeated_id["revenue"].sum() - repeated_id.groupby("order_id")["revenue"].first().sum())
        dates = _isos(sorted(set(repeated_id["date"])))
        findings.append(
            Finding(
                check="duplicate_orders",
                severity="high",
                source="shopify_orders",
                dates=dates,
                description=(
                    f"{_count(repeated_id['order_id'].nunique())} order id(s) appear more than "
                    f"once, {_count(extra)} row(s) beyond the first of each, spanning "
                    f"{_span(dates)}. Order id is the store's primary key, so these are the same "
                    f"order counted twice: they add {_money(overstated)} of revenue that the "
                    f"store never took."
                ),
                numbers={
                    "n_duplicate_ids": int(repeated_id["order_id"].nunique()),
                    "n_extra_rows": int(extra),
                    "overstated_revenue": round(overstated, 2),
                    "example_order_ids": sorted(repeated_id["order_id"].unique())[:5],
                },
            )
        )

    flagged = orders.duplicated(ORDER_FINGERPRINT, keep=False)
    baseline = float(flagged.mean())
    if baseline > 0:
        by_date = pd.DataFrame({"date": orders["date"], "flagged": flagged})
        rate = by_date.groupby("date")["flagged"].agg(["sum", "count"])
        rate["share"] = rate["sum"] / rate["count"]
        hits = rate[(rate["share"] >= DUP_DATE_SHARE) & (rate["share"] >= DUP_OVER_BASELINE * baseline)]

        for day, row in hits.iterrows():
            dates = _isos([day])
            same_day = orders[(orders["date"] == day) & flagged]
            extra = len(same_day) - len(same_day.drop_duplicates(ORDER_FINGERPRINT))
            overstated = float(same_day["revenue"].sum() - same_day.drop_duplicates(ORDER_FINGERPRINT)["revenue"].sum())
            findings.append(
                Finding(
                    check="duplicate_orders",
                    severity="high",
                    source="shopify_orders",
                    dates=dates,
                    description=(
                        f"On {dates[0]}, {row['share']:.0%} of the day's {_count(row['count'])} orders "
                        f"are line-for-line identical to another order that day -- same region, "
                        f"price, discount, refund, basket and customer type -- against "
                        f"{baseline:.2%} across the whole table. Distinct order ids, so this is not "
                        f"a repeated key; it has the shape of a batch loaded twice, and about "
                        f"{_money(overstated)} of the day's revenue would be double counted if it is."
                    ),
                    numbers={
                        "duplicate_share": round(float(row["share"]), 4),
                        "table_baseline_share": round(baseline, 4),
                        "n_orders_on_date": int(row["count"]),
                        "n_rows_in_duplicate_groups": int(row["sum"]),
                        "n_extra_rows": int(extra),
                        "overstated_revenue": round(overstated, 2),
                    },
                )
            )

    return findings


# --------------------------------------------------------------------------
# 3. Pixel double counting
# --------------------------------------------------------------------------


def _platform(channel: pd.Series) -> pd.Series:
    """The ad platform behind a channel name: 'Meta prospecting' -> 'Meta'.

    A pixel belongs to a platform, not to a channel, so a purchase event fired
    twice shows up across every channel that platform runs.
    """
    return channel.str.split().str[0]


def check_pixel_double_counting(orders: pd.DataFrame, ad_spend: pd.DataFrame) -> list[Finding]:
    """Platform purchases jumping while the store's own orders sit still.

    Each platform's attributed revenue is measured against what its *own*
    spend predicts -- its trailing median revenue per dollar -- rather than
    against the store directly. A channel going dark in a geo holdout changes
    its share of store revenue without anything being wrong, and that
    normalisation is what keeps a planned test from reading as a bug.

    A day is only flagged when the store did not move with the platform: if
    Shopify revenue and orders rose too, the extra credit is a real sales day.
    """
    frame = ad_spend.copy()
    frame["platform"] = _platform(frame["channel"])
    calendar = _calendar({"ad_spend": ad_spend, "shopify_orders": orders})

    store_revenue = _daily(orders, "revenue", calendar)
    store_orders = orders.groupby("date")["order_id"].count().reindex(calendar)
    rolling = dict(window=PIXEL_TRAILING_DAYS, min_periods=PIXEL_MIN_DAYS)
    store_ratio = store_revenue / store_revenue.rolling(**rolling).median().shift(1)
    order_ratio = store_orders / store_orders.rolling(**rolling).median().shift(1)

    findings: list[Finding] = []
    for platform, sub in frame.groupby("platform", observed=True):
        spend = _daily(sub, "spend", calendar)
        attributed = _daily(sub, "platform_attributed_revenue", calendar)
        per_dollar = attributed / spend
        trailing = per_dollar.rolling(**rolling).median().shift(1)
        ratio = per_dollar / trailing
        expected = spend * trailing
        spend_ratio = spend / spend.rolling(**rolling).median().shift(1)

        store_still = (store_ratio - 1.0).abs() <= STORE_TOLERANCE
        hits = ratio[(ratio >= PIXEL_RATIO) & store_still].dropna().index

        for run in _runs(list(hits)):
            dates = _isos(run)
            claimed = float(attributed.loc[run].sum())
            predicted = float(expected.loc[run].sum())
            multiple = claimed / predicted
            channels = sorted(sub.loc[sub["date"].isin(run), "channel"].unique())
            spend_move = float((spend_ratio.loc[run] - 1.0).abs().max())
            store_move = float((store_ratio.loc[run] - 1.0).abs().max())
            order_move = float((order_ratio.loc[run] - 1.0).abs().max())

            findings.append(
                Finding(
                    check="pixel_double_counting",
                    severity="high",
                    source="ad_spend",
                    dates=dates,
                    description=(
                        f"{platform} claimed {multiple:.2f}x what its own spend predicts on "
                        f"{_plural(len(run), 'consecutive day')}, {_span(dates)}: {_money(claimed)} of "
                        f"attributed revenue against the {_money(predicted)} implied by its "
                        f"trailing {PIXEL_TRAILING_DAYS}-day revenue per dollar, an excess of "
                        f"{_money(claimed - predicted)}. {platform} spend moved at most "
                        f"{spend_move:.0%} over those days, the store's revenue at most "
                        f"{store_move:.0%} and its order count at most {order_move:.0%}, so "
                        f"nothing was sold to justify the extra credit -- it has the shape of a "
                        f"purchase event fired twice. Channels affected: {', '.join(channels)}."
                    ),
                    numbers={
                        "platform": platform,
                        "channels": channels,
                        "n_days": len(run),
                        "attributed_revenue": round(claimed, 2),
                        "expected_attributed_revenue": round(predicted, 2),
                        "excess_attributed_revenue": round(claimed - predicted, 2),
                        "implied_multiple": round(multiple, 3),
                        "max_spend_move": round(spend_move, 4),
                        "max_store_revenue_move": round(store_move, 4),
                        "max_store_order_move": round(order_move, 4),
                        "daily_multiple": {d: round(float(ratio[day]), 3) for d, day in zip(dates, run)},
                    },
                )
            )

    return findings


# --------------------------------------------------------------------------
# 4. Reporting lag
# --------------------------------------------------------------------------


def _same_weekday_expectation(series: pd.Series, weeks: int = LAG_LOOKBACK_WEEKS) -> pd.Series:
    """What a day should have been, from the same weekday in recent weeks.

    A median over the last few matching weekdays carries the level, the trend
    and the day-of-week shape without any model, and a single odd week cannot
    drag it.
    """
    shifts = [series.shift(7 * w) for w in range(1, weeks + 1)]
    return pd.concat(shifts, axis=1).median(axis=1)


def check_reporting_lag(tables: Tables) -> list[Finding]:
    """The most recent days still filling in.

    Each table's daily total is compared with the same weekday in the previous
    weeks, and the resulting ratio is scored against the 28 days before the
    tail. The scan runs backwards from the last day and stops at the first day
    that looks settled -- a lag fills in from the oldest day forward, so a
    short day in the middle of the window is a different problem, and stopping
    early keeps this to a handful of comparisons.
    """
    frames = _frames(tables)
    calendar = _calendar(frames)
    series = {name: _daily(frame, TABLE_VOLUME[name], calendar) for name, frame in frames.items()}
    ratios = {name: s / _same_weekday_expectation(s) for name, s in series.items()}
    references = {
        name: ratio.iloc[-(LAG_TAIL_DAYS + LAG_REFERENCE_DAYS) : -LAG_TAIL_DAYS].dropna()
        for name, ratio in ratios.items()
    }

    def share_of_expected(name: str, day: pd.Timestamp) -> float:
        """How much of a settled day's figure this table has so far."""
        median = float(references[name].median())
        value = ratios[name].get(day, np.nan)
        return float(value) / median if median else np.nan

    findings: list[Finding] = []
    for name, ratio in ratios.items():
        if references[name].empty:
            continue

        lagging: list[pd.Timestamp] = []
        shares: dict[str, float] = {}
        for offset in range(1, LAG_TAIL_DAYS + 1):
            day = calendar[-offset]
            value = ratio.get(day, np.nan)
            if pd.isna(value):
                break
            score, _, _ = _robust_score(references[name], float(value))
            share = share_of_expected(name, day)
            if score > -LAG_SCORE or (1.0 - share) < LAG_MIN_SHORTFALL:
                break
            lagging.append(day)
            shares[_iso(day)] = share

        if not lagging:
            continue

        column = TABLE_VOLUME[name]
        lagging = sorted(lagging)
        dates = _isos(lagging)
        reported = float(series[name].loc[lagging].sum())
        settled = reported / float(np.mean(list(shares.values())))
        complete = sorted(
            other
            for other in ratios
            if other != name
            and not references[other].empty
            and all(1.0 - share_of_expected(other, day) < LAG_MIN_SHORTFALL for day in lagging)
        )
        against = (
            f"{' and '.join(complete)} {_verb(complete)} no such shortfall on the same days"
            if complete
            else "every table is short on the same days"
        )
        shares = {d: shares[d] for d in dates}
        percents = " and ".join(f"{shares[d]:.0%} on {d}" for d in dates)

        findings.append(
            Finding(
                check="reporting_lag",
                severity="medium",
                source=name,
                dates=dates,
                description=(
                    f"{name} is still filling in over the last {_plural(len(lagging), 'day')} of the "
                    f"window, {_span(dates)}: the {column} total stands at {percents}, against what "
                    f"the same weekday in the previous {LAG_LOOKBACK_WEEKS} weeks implies -- about "
                    f"{_volume(column, settled - reported)} short of settled days. {against}, so "
                    f"this is reporting lag rather than a real fall. Read these days as "
                    f"incomplete: a week-{_week_of(lagging[0], calendar)} total built on them is "
                    f"understated, and any ratio that puts {name} against a complete table is "
                    f"distorted until they settle."
                ),
                numbers={
                    "n_days": len(lagging),
                    "share_of_expected": {d: round(v, 3) for d, v in shares.items()},
                    f"reported_{column}": round(reported, 2),
                    f"estimated_settled_{column}": round(settled, 2),
                    f"estimated_shortfall_{column}": round(settled - reported, 2),
                    "complete_tables": complete,
                    "lookback_weeks": LAG_LOOKBACK_WEEKS,
                },
            )
        )

    return findings


# --------------------------------------------------------------------------
# Everything at once
# --------------------------------------------------------------------------


def run_checks(source: str | Path | Tables = DEFAULT_DATA_DIR) -> list[Finding]:
    """Run every check. ``source`` is a data directory or a loaded ``Tables``."""
    tables = source if isinstance(source, Tables) else load_tables(source)
    findings = [
        *check_missing_days(tables),
        *check_duplicate_orders(tables.orders),
        *check_pixel_double_counting(tables.orders, tables.ad_spend),
        *check_reporting_lag(tables),
    ]
    return sort_findings(findings)
