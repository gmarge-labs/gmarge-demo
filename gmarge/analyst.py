"""Weekly read-outs: Python computes and formats the numbers, the model phrases them.

The shape of the pipeline (see CLAUDE.md) is::

    metrics + quality findings + anomalies
        -> build_facts(week)      every number the read-out may use, formatted
        -> prompt_facts(facts)    display strings only -- what the model sees
        -> the model              phrasing, and nothing else
        -> check_numbers()        every figure in the reply matched to a display
        -> readouts/week-NN.json  committed, after a human has read it

Three rules hold this together.

**Python formats.** Every number in the facts is a ``{"value", "display"}``
pair: the value for audit, the display for the read-out. ``$531.6k``,
``1.30x``, ``12.2%`` are decided here, not by the model, and the model is shown
the displays alone -- :func:`prompt_facts` strips the values, so there is no
raw float in the prompt to round, rescale or mistype.

**One source of truth for "normal".** Whether a week is outside its normal
range is the anomaly detector's verdict and nobody else's. This module does not
score levels, does not compute a band, and has no opinion of its own; it passes
on the flags ``gmarge/anomalies.py`` raised for the week. A week with a flag
leads with it and cannot be called normal.

**Incomplete weeks do not get verdicts.** When a source is still filling in,
every total that draws on it is listed in ``completeness.totals_still_filling_in``
and must be described as incomplete rather than as a rise or a fall.

``--dry-run`` writes a read-out assembled in Python from the same facts, with
no API call at all. The tests use it, and it doubles as a check on the facts:
if the template cannot be written from the facts alone, neither can the model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gmarge import llm
from gmarge.anomalies import METRICS_BY_KEY, Anomaly, detect_anomalies
from gmarge.metrics import DEFAULT_DATA_DIR, Tables, all_metrics, load_tables
from gmarge.quality import Finding, run_checks

DEFAULT_READOUT_DIR = "readouts"
DEFAULT_WEEKS = 8
WORD_LIMIT = 120

# The headline figures, and how each is written. MONEY is $531.6k, RATIO is
# 1.30x, PCT is 12.2%, COUNT is 7.
MONEY, RATIO, PCT, COUNT = "money", "ratio", "pct", "count"

HEADLINE = {
    "shopify_revenue": MONEY,
    "shopify_net_revenue": MONEY,
    "ad_spend": MONEY,
    "platform_attributed_revenue": MONEY,
    "over_claimed_revenue": MONEY,
    "over_claim_ratio": RATIO,
    "blended_reported_roas": RATIO,
}

LABELS = {
    "shopify_revenue": "store revenue",
    "shopify_net_revenue": "net store revenue",
    "ad_spend": "ad spend",
    "platform_attributed_revenue": "platform-attributed revenue",
    "over_claimed_revenue": "over-claimed revenue",
    "over_claim_ratio": "over-claim ratio",
    "blended_reported_roas": "blended reported ROAS",
}

# Which table each headline figure is read from. A figure is only as complete
# as the tables under it, which is what makes a partial week describable.
TOTAL_SOURCES = {
    "shopify_revenue": ("shopify_orders",),
    "shopify_net_revenue": ("shopify_orders",),
    "ad_spend": ("ad_spend",),
    "platform_attributed_revenue": ("ad_spend",),
    "over_claimed_revenue": ("shopify_orders", "ad_spend"),
    "over_claim_ratio": ("shopify_orders", "ad_spend"),
    "blended_reported_roas": ("ad_spend",),
}

# Week-on-week changes that can be split by channel. Store revenue cannot be:
# the platforms' attributed revenue is not the store's revenue.
DRIVEN = ("ad_spend", "platform_attributed_revenue")

# The checks that mean a day is missing or still filling in, as opposed to
# present but wrong. Only these bear on whether the week's data is complete.
INCOMPLETE_CHECKS = ("reporting_lag", "missing_days")

CHECK_LABELS = {
    "reporting_lag": "still filling in",
    "missing_days": "missing entirely",
    "duplicate_orders": "duplicate orders",
    "pixel_double_counting": "pixel double-counting",
}

# An anomaly metric's unit, in this module's formats.
UNIT_KIND = {"money": MONEY, "percent": PCT, "x": RATIO, "ratio": RATIO}

# Free text that is kept in the saved facts for the reviewer but withheld from
# the model: it carries figures in another module's formatting, and the model
# may only quote this module's displays.
PROMPT_SKIP = ("description",)


# --------------------------------------------------------------------------
# Formatting. Every number a read-out can use is written here, once.
# --------------------------------------------------------------------------


def money_display(value: float) -> str:
    """``$531.6k``, ``$1.2m``, ``$532``. Magnitude only -- direction is a word."""
    size = abs(float(value))
    if size >= 1_000_000:
        return f"${size / 1_000_000:,.1f}m"
    if size >= 1_000:
        return f"${size / 1_000:,.1f}k"
    return f"${size:,.0f}"


def ratio_display(value: float) -> str:
    """``1.30x``."""
    return f"{abs(float(value)):,.2f}x"


def pct_display(value: float) -> str:
    """``12.2%``, from a ratio."""
    return f"{abs(float(value)) * 100:.1f}%"


def count_display(value: float) -> str:
    """``7``."""
    return f"{int(round(float(value))):,}"


DISPLAY = {MONEY: money_display, RATIO: ratio_display, PCT: pct_display, COUNT: count_display}
PLACES = {MONEY: 2, RATIO: 4, PCT: 4, COUNT: 0}


def fact(value, kind: str) -> dict | None:
    """One number, as the reviewer sees it and as the read-out must write it.

    ``value`` is kept for audit and stripped before the prompt is built;
    ``display`` is the only form the model is given and the only form the
    check accepts. Displays carry magnitude, not sign: a fall is a direction
    in the prose, which keeps "down $19.3k" from reading "down -$19.3k".
    """
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return {"value": round(number, PLACES[kind]), "display": DISPLAY[kind](number)}


# --------------------------------------------------------------------------
# Everything the facts are built from, computed once
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    """One pass over the data: metrics, quality findings, anomalies."""

    metrics: dict
    findings: list[Finding]
    anomalies: list[Anomaly]

    @property
    def weeks(self) -> list[int]:
        return [int(w) for w in self.metrics["weekly_reconciliation"]["week"]]


def analyse(source: str | Path | Tables = DEFAULT_DATA_DIR) -> Analysis:
    """Run the three modules a read-out draws on. No AI calls happen here."""
    tables = source if isinstance(source, Tables) else load_tables(source)
    return Analysis(
        metrics=all_metrics(tables),
        findings=run_checks(tables),
        anomalies=detect_anomalies(tables),
    )


def week_number(facts: dict) -> int:
    return int(facts["week"]["number"]["value"])


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------


def _headline(row: pd.Series) -> dict:
    """The week's totals, straight off the weekly reconciliation row."""
    spend = float(row["ad_spend"])
    attributed = float(row["platform_attributed_revenue"])
    values = {
        "shopify_revenue": float(row["shopify_revenue"]),
        "shopify_net_revenue": float(row["shopify_net_revenue"]),
        "ad_spend": spend,
        "platform_attributed_revenue": attributed,
        "over_claimed_revenue": float(row["over_claimed_revenue"]),
        "over_claim_ratio": float(row["over_claim_ratio"]),
        "blended_reported_roas": attributed / spend if spend else None,
    }
    return {key: fact(values[key], kind) for key, kind in HEADLINE.items()}


def _change(current: dict, prior: dict, filling_in: set[str]) -> dict:
    """Week-on-week change, with a direction and whether it can be read as one.

    A total whose tables are still filling in gets its move reported as
    incomplete: the figure is real arithmetic on the rows that have landed, and
    saying it "fell" would be reading a reporting lag as a result.
    """
    out = {}
    for key, kind in HEADLINE.items():
        now, before = current.get(key), prior.get(key)
        if now is None or before is None:
            continue
        absolute = now["value"] - before["value"]
        out[key] = {
            "label": LABELS[key],
            "absolute": fact(absolute, kind),
            "pct": fact(absolute / before["value"], PCT) if before["value"] else None,
            "direction": "up" if absolute > 0 else "down" if absolute < 0 else "flat",
            "still_filling_in": key in filling_in,
        }
    return out


def _channels(roas_week: pd.DataFrame, week: int, prior_week: int | None) -> list[dict]:
    """Per channel: this week's spend and attributed revenue, and the move."""
    now = roas_week[roas_week["week"] == week].set_index("channel")
    before = roas_week[roas_week["week"] == prior_week].set_index("channel") if prior_week else None

    rows = []
    for channel, row in now.iterrows():
        prior = before.loc[channel] if before is not None and channel in before.index else None
        entry = {
            "channel": str(channel),
            "spend": fact(row["spend"], MONEY),
            "platform_attributed_revenue": fact(row["platform_attributed_revenue"], MONEY),
            "reported_roas": fact(row["reported_roas"], RATIO),
        }
        for column in DRIVEN:
            source = "spend" if column == "ad_spend" else column
            move = float(row[source]) - float(prior[source]) if prior is not None else None
            entry[f"{column}_change"] = fact(move, MONEY)
            entry[f"{column}_direction"] = None if move is None else ("up" if move > 0 else "down")
            entry[f"_{column}_move"] = move  # dropped below; ranking only
        rows.append(entry)
    return sorted(rows, key=lambda r: r["spend"]["value"], reverse=True)


def _drivers(channels: list[dict], change: dict) -> dict:
    """Which channel moved a week-on-week change, and how much of it.

    A share is the channel's own move over the total move, so shares can exceed
    100% when channels pull against each other -- a fact about the week, left
    visible rather than normalised away.
    """
    out = {}
    for column in DRIVEN:
        total = change.get(column, {}).get("absolute")
        contributions = [c for c in channels if c[f"_{column}_move"] is not None]
        if total is None or not contributions:
            out[column] = {"label": LABELS[column], "total_change": total, "channels": [], "largest": None}
            continue

        ranked = sorted(contributions, key=lambda c: abs(c[f"_{column}_move"]), reverse=True)
        listed = [
            {
                "channel": c["channel"],
                "change": c[f"{column}_change"],
                "direction": c[f"{column}_direction"],
                "share_of_change": fact(c[f"_{column}_move"] / total["value"], PCT) if total["value"] else None,
            }
            for c in ranked
        ]
        out[column] = {
            "label": LABELS[column],
            "total_change": total,
            "direction": change[column]["direction"],
            "still_filling_in": change[column]["still_filling_in"],
            "channels": listed,
            "largest": listed[0],
        }
    return out


def _anomalies(anomalies: list[Anomaly], week: int) -> dict:
    """The week's flags, from the detector, in this module's formats.

    This is the only thing in the facts that says whether the week is outside
    its normal range. The read-out gets the flags or it gets nothing; it never
    gets a band of this module's own to argue with.
    """
    flags = []
    for flag in (a for a in anomalies if a.week == week):
        kind = UNIT_KIND[METRICS_BY_KEY[flag.metric].unit]
        flags.append(
            {
                "channel": flag.channel,
                "metric": flag.metric,
                "metric_label": flag.metric_label,
                "direction": flag.direction,
                "severity": flag.severity,
                "value": fact(flag.value, kind),
                "trailing_median": fact(flag.baseline, kind),
                "trailing_weeks": fact(flag.numbers["trailing_weeks"], COUNT),
                "pct_change": fact(flag.pct_change, PCT),
                "campaign": flag.campaign,
                "ad_set": flag.ad_set,
                "share_of_move": fact(flag.share_of_move, PCT),
                "description": flag.description,
            }
        )
    return {
        "n_flags": fact(len(flags), COUNT),
        "flags": flags,
        "source": "the weekly anomaly scan in gmarge/anomalies.py",
    }


def _findings(findings: list[Finding], start: str, end: str) -> list[dict]:
    """Quality findings whose dates fall inside the week.

    A finding can straddle a week boundary, so each carries the days of its own
    that fall inside *this* week, and how many, rather than a range for a
    read-out to count across.
    """
    out = []
    for finding in findings:
        if finding.start_date > end or finding.end_date < start:
            continue
        inside = [day for day in finding.dates if start <= day <= end]
        out.append(
            {
                "check": finding.check,
                "label": CHECK_LABELS.get(finding.check, finding.check.replace("_", " ")),
                "severity": finding.severity,
                "source": finding.source,
                "start_date": finding.start_date,
                "end_date": finding.end_date,
                "dates_in_week": inside,
                "n_days_in_week": fact(len(inside), COUNT),
                "description": finding.description,
            }
        )
    return out


def _completeness(findings: list[dict], days_in_week: int) -> dict:
    """What is missing from the week, how much of it, and what it invalidates.

    Every count a read-out could want is spelled out -- days in the week, days
    with complete data, days affected, the affected days themselves -- because
    a count worked out from a date range is a count nobody checked. So is the
    list of totals that cannot be read as a rise or a fall.
    """
    incomplete = [f for f in findings if f["check"] in INCOMPLETE_CHECKS]
    affected_dates = sorted({day for f in incomplete for day in f["dates_in_week"]})
    sources = {f["source"] for f in incomplete}
    filling_in = [key for key, tables in TOTAL_SOURCES.items() if sources.intersection(tables)]

    return {
        "complete": not incomplete,
        "days_in_week": fact(days_in_week, COUNT),
        "days_affected": fact(len(affected_dates), COUNT),
        "days_with_complete_data": fact(days_in_week - len(affected_dates), COUNT),
        "affected_dates": affected_dates,
        "affected_sources": sorted(sources),
        "totals_still_filling_in": filling_in,
        "totals_still_filling_in_labels": [LABELS[key] for key in filling_in],
        "channel_figures_still_filling_in": "ad_spend" in sources,
        "affected": [
            {
                "source": f["source"],
                "check": f["check"],
                "label": f["label"],
                "start_date": f["start_date"],
                "end_date": f["end_date"],
                "n_days_in_week": f["n_days_in_week"],
                "dates_in_week": f["dates_in_week"],
            }
            for f in incomplete
        ],
    }


def _holdouts(holdouts: pd.DataFrame, start: str, end: str) -> dict:
    """Geo holdouts whose window overlaps the week.

    A holdout measures a whole four-week window, not a single week inside it,
    and the note says so: these are the only incremental figures in the facts,
    and they must not be read as this week's result.
    """
    running = []
    for row in holdouts.itertuples():
        window_start = row.start_date.date().isoformat()
        window_end = row.end_date.date().isoformat()
        if window_start > end or window_end < start:
            continue
        running.append(
            {
                "channel": row.channel,
                "window_start": window_start,
                "window_end": window_end,
                "covers_whole_week": bool(window_start <= start and window_end >= end),
                "incremental_roas": fact(row.incremental_roas, RATIO),
                "incremental_roas_ci_low": fact(row.incremental_roas_ci_low, RATIO),
                "incremental_roas_ci_high": fact(row.incremental_roas_ci_high, RATIO),
                "reported_roas_in_window": fact(row.reported_roas_in_window, RATIO),
                "over_claim_multiple": fact(row.over_claim_multiple, RATIO),
                "lift_pct": fact(row.lift_pct, PCT),
                "paused_spend_estimate": fact(row.paused_spend_estimate, MONEY),
                "confidence": fact(row.confidence, PCT),
            }
        )
    return {
        "n_running": fact(len(running), COUNT),
        "running": running,
        "note": (
            "Measured over the whole holdout window, not this week alone. "
            "Incremental ROAS is what the channel is worth; reported ROAS is what the platform claims."
        ),
    }


def build_facts(analysis: Analysis, week: int) -> dict:
    """Every number the read-out for ``week`` is allowed to use, already formatted.

    Nothing outside this dict may appear in the read-out, and nothing in it is
    a raw float by the time the model sees it -- see :func:`prompt_facts`.
    """
    weekly = analysis.metrics["weekly_reconciliation"]
    if week not in set(int(w) for w in weekly["week"]):
        raise ValueError(f"week {week} is not in the data")

    row = weekly[weekly["week"] == week].iloc[0]
    start, end = str(row["week_start"].date()), str(row["week_end"].date())
    prior_week = week - 1 if (weekly["week"] == week - 1).any() else None
    prior_row = weekly[weekly["week"] == prior_week].iloc[0] if prior_week else None

    findings = _findings(analysis.findings, start, end)
    completeness = _completeness(findings, int(row["days"]))
    filling_in = set(completeness["totals_still_filling_in"])

    headline = _headline(row)
    prior_headline = _headline(prior_row) if prior_row is not None else {}
    change = _change(headline, prior_headline, filling_in) if prior_row is not None else None
    channels = _channels(analysis.metrics["reported_roas_by_channel_week"], week, prior_week)
    drivers = _drivers(channels, change) if change else None

    facts = {
        "brand": analysis.metrics["brand"],
        "disclaimer": analysis.metrics["disclaimer"],
        "week": {
            "number": fact(week, COUNT),
            "week_start": start,
            "week_end": end,
            "days": fact(int(row["days"]), COUNT),
        },
        "prior_week": (
            {
                "number": fact(prior_week, COUNT),
                "week_start": str(prior_row["week_start"].date()),
                "week_end": str(prior_row["week_end"].date()),
            }
            if prior_row is not None
            else None
        ),
        "data_window": {
            "start": analysis.metrics["window"]["start"],
            "end": analysis.metrics["window"]["end"],
            "n_weeks": fact(analysis.metrics["window"]["n_weeks"], COUNT),
        },
        "anomalies": _anomalies(analysis.anomalies, week),
        "completeness": completeness,
        "quality_findings": findings,
        "over_claim": {
            "ratio": headline["over_claim_ratio"],
            "over_claimed_revenue": headline["over_claimed_revenue"],
            "platform_attributed_revenue": headline["platform_attributed_revenue"],
            "shopify_revenue": headline["shopify_revenue"],
            "still_filling_in": "over_claim_ratio" in filling_in,
            "meaning": (
                "The platforms between them claimed this much more revenue than the store took. "
                "A ratio above 1.00x is the same order credited more than once."
            ),
        },
        "totals": headline,
        "change_vs_prior_week": change,
        "channels": channels,
        "drivers": drivers,
        "holdouts": _holdouts(analysis.metrics["holdouts"], start, end),
        "notes": [
            "Store revenue cannot be split by channel; platform-attributed revenue is not store revenue.",
            "Reported ROAS is what the platforms claim. Only a holdout says what a channel is worth.",
        ],
    }
    return _strip_private(facts)


def _strip_private(node):
    """Drop the ``_``-prefixed working values used for ranking."""
    if isinstance(node, dict):
        return {k: _strip_private(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_strip_private(v) for v in node]
    return node


def prompt_facts(facts: dict) -> dict:
    """The facts as the model sees them: display strings, no raw numbers.

    Every ``{"value", "display"}`` pair collapses to its display, and free text
    written by another module is withheld -- it carries figures in another
    format, and the model may only quote the displays this module wrote.
    """

    def walk(node):
        if isinstance(node, dict):
            if set(node) == {"value", "display"}:
                return node["display"]
            return {k: walk(v) for k, v in node.items() if k not in PROMPT_SKIP}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(facts)


# --------------------------------------------------------------------------
# The number check
# --------------------------------------------------------------------------

DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
FIGURE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:%|x|k|m)?", re.IGNORECASE)
SENTENCE = re.compile(r"(?<=[.!?])\s+")


class NumberCheckError(RuntimeError):
    """A read-out used figures that are not in its facts."""

    def __init__(self, week: int, unsupported: list[str], text: str, facts: dict):
        self.week = week
        self.unsupported = unsupported
        self.text = text
        self.sentences = offending_sentences(text, unsupported, facts)
        super().__init__(
            f"{len(unsupported)} figure(s) not in the facts: {', '.join(unsupported)}"
            + "".join(f"\n      {line}" for line in self.sentences)
        )


def _strings(node, found: set[str]) -> set[str]:
    """Every string leaf of the facts the model was shown."""
    if isinstance(node, dict):
        for value in node.values():
            _strings(value, found)
    elif isinstance(node, list):
        for value in node:
            _strings(value, found)
    elif isinstance(node, str):
        found.add(node)
    return found


def _mask(text: str, names: list[str]) -> str:
    """Blank out names quoted verbatim from the facts.

    Names carry digits -- the ad set ``Core | Broad 25-44``, the region
    ``Region 03`` -- and quoting one is not writing a figure. Longest first, so
    a name inside a longer name cannot be half-masked.
    """
    for name in names:
        text = text.replace(name, " ")
    return text


def _written_figures(text: str, names: list[str]) -> list[str]:
    """Every date and figure written in ``text``, as the check sees them."""
    return DATE.findall(text) + FIGURE.findall(_mask(DATE.sub(" ", text), names))


def check_numbers(text: str, facts: dict) -> list[str]:
    """Every figure in ``text`` that is not a display string in ``facts``.

    The model is shown displays and nothing else, so it is held to them
    exactly: ``$531.6k`` passes and ``$531,600``, ``$532k`` and ``$0.5m`` do
    not. There is no rounding allowance, because there is nothing left to
    round -- Python already did it. An empty list means the read-out is safe
    to save.
    """
    shown = _strings(prompt_facts(facts), set())
    allowed = {s.casefold() for s in shown if FIGURE.fullmatch(s)}
    names = sorted(
        (s for s in shown if any(c.isdigit() for c in s) and not FIGURE.fullmatch(s)),
        key=len,
        reverse=True,
    )
    dates = " ".join(s for s in shown if DATE.search(s))

    unsupported = []
    for token in _written_figures(text, names):
        supported = token in dates if DATE.fullmatch(token) else token.casefold() in allowed
        if not supported:
            unsupported.append(token)
    return unsupported


def offending_sentences(text: str, unsupported: list[str], facts: dict) -> list[str]:
    """Each unsupported figure with the sentence it was written in.

    A bare list of figures says a read-out failed; it does not say what the
    model was trying to write. ``6`` means nothing on its own and everything
    once you can see it sat in "only 6 of 7 days reported" -- a model counting
    days for itself, which is fixed in the facts, not in the check.

    Sentences are searched with the check's own tokeniser rather than for the
    text of the figure, because ``6`` is a substring of ``$563,472.11``.
    """
    shown = _strings(prompt_facts(facts), set())
    names = sorted(
        (s for s in shown if any(c.isdigit() for c in s) and not FIGURE.fullmatch(s)),
        key=len,
        reverse=True,
    )
    sentences = [s.strip() for s in SENTENCE.split(text.strip()) if s.strip()]

    lines = []
    for token in unsupported:
        where = next((s for s in sentences if token in _written_figures(s, names)), text.strip())
        lines.append(f"{token!r} in: {where}")
    return lines


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------
# Asking the model
# --------------------------------------------------------------------------

SYSTEM = f"""You write the weekly read-out for a marketing-measurement demo.

Everything you need is in the FACTS the user message carries. Every figure
there is already written the way it must appear. These rules are hard:

1. At most {WORD_LIMIT} words. Plain prose in one or two short paragraphs. No
   headings, no bullet points, no markdown.
2. Write in this order, and lead with the first of these that applies:
   a. Anything flagged in `anomalies.flags`, and anything in `completeness`
      that is not complete. If there is a flag, it leads the read-out.
   b. `over_claim`: the ratio, and what it means for this week.
   c. `drivers`: the channel behind most of the week's change.
   Then, if `holdouts.running` is not empty, one sentence on what that holdout
   measured -- saying it is the holdout window, not this week.
3. Quote figures exactly as they are written in the FACTS. `$531.6k` is
   written `$531.6k`, never `$531,600`, `$532k`, `0.5m` or `531.6`. Do no
   arithmetic of any kind: no adding, no subtracting, no percentages of your
   own, no converting a unit. If a figure is not in the FACTS, you cannot use
   it. Direction is a word -- "up", "down" -- and the FACTS give it to you.
4. `anomalies` is the only thing that says whether the week is outside its
   normal range. If `anomalies.flags` is not empty, say what was flagged and
   never say the week was normal or that every metric was in range. If it is
   empty, you may say nothing was flagged -- which is not the same as saying
   everything is fine, so do not say everything is fine.
5. If `completeness.complete` is false, say so and name what is missing, using
   only the counts in `completeness`: `days_in_week`, `days_with_complete_data`,
   `days_affected`, `affected_dates` and each entry's `source`. Do not count
   days and do not subtract one count from another.
6. Any total named in `completeness.totals_still_filling_in` is incomplete. Say
   it is still filling in. Do not call its move a rise, a fall, a drop, a
   recovery or an improvement, and do not explain it -- the tables are not in
   yet.
7. This is synthetic sample data for a brand that does not exist. Do not write
   as though it were a real company's results, and do not recommend budget
   decisions."""


def build_prompt(facts: dict) -> str:
    return (
        "FACTS\n"
        + json.dumps(prompt_facts(facts), indent=2, default=str)
        + f"\n\nWrite the read-out for week {week_number(facts)}."
    )


def retry_prompt(prompt: str, unsupported: list[str], text: str, facts: dict) -> str:
    """The same request again, with the figures that failed the check named.

    The sentence each figure was written in goes back too: it is usually the
    sentence, not the figure, that shows what went wrong -- a count worked out
    from a date range, a figure rewritten in another format.
    """
    return (
        prompt
        + "\n\nYour previous read-out used "
        + ", ".join(unsupported)
        + ", which are not written that way anywhere in the FACTS above:\n"
        + "\n".join(f"  {line}" for line in offending_sentences(text, unsupported, facts))
        + "\n\nWrite the read-out again, using only figures that appear in the FACTS, "
        "copied character for character. If you need a figure the FACTS do not give you, "
        "leave it out."
    )


def generate_readout(facts: dict, *, model: str | None = None) -> str:
    """Ask the model, check the figures, ask once more if any fail.

    Raises :class:`NumberCheckError` if the second reply still uses a figure
    that is not in the facts. Nothing is saved in that case.
    """
    prompt = build_prompt(facts)
    text = llm.complete(SYSTEM, prompt, model=model)

    unsupported = check_numbers(text, facts)
    if unsupported:
        text = llm.complete(SYSTEM, retry_prompt(prompt, unsupported, text, facts), model=model)
        unsupported = check_numbers(text, facts)
        if unsupported:
            raise NumberCheckError(week_number(facts), unsupported, text, facts)
    return text


# --------------------------------------------------------------------------
# The template read-out (--dry-run)
# --------------------------------------------------------------------------


def _show(node) -> str:
    """The display string of a fact, or a dash where there is no figure."""
    return "--" if node is None else node["display"]


def _join(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def template_readout(facts: dict) -> str:
    """A read-out assembled in Python, for tests and for --dry-run.

    Built from the same facts, in the same order the model is asked for, and
    held to the same number check -- so it stays honest about what the facts
    can support, and about what they cannot.
    """
    week, completeness = facts["week"], facts["completeness"]
    flags, filling_in = facts["anomalies"]["flags"], completeness["totals_still_filling_in"]

    parts = [f"Week {_show(week['number'])} ({week['week_start']} to {week['week_end']}), {facts['brand']}."]

    # (a) what was flagged, and what is missing
    if flags:
        flag = flags[0]
        parts.append(
            f"{flag['channel']}'s {flag['metric_label']} was flagged {flag['direction']} "
            f"{_show(flag['pct_change'])} against its trailing median, to {_show(flag['value'])}, "
            f"{_show(flag['share_of_move'])} of it from ad set {flag['ad_set']}."
        )
    else:
        parts.append("Nothing was flagged this week.")

    if not completeness["complete"]:
        sources = completeness["affected_sources"]
        parts.append(
            f"{_join(sources)} {'covers' if len(sources) == 1 else 'cover'} "
            f"{_show(completeness['days_with_complete_data'])} of the week's "
            f"{_show(completeness['days_in_week'])} days -- "
            f"{_join(completeness['affected_dates'])} are "
            f"{_join(sorted({a['label'] for a in completeness['affected']}))} -- "
            + (
                f"so {_join(filling_in_labels)} are incomplete."
                if (filling_in_labels := completeness["totals_still_filling_in_labels"])
                else "which leaves the headline totals unaffected."
            )
        )

    # (b) the over-claim ratio
    over_claim = facts["over_claim"]
    if over_claim["still_filling_in"]:
        parts.append(
            f"The over-claim ratio stands at {_show(over_claim['ratio'])} on the rows in so far."
        )
    else:
        parts.append(
            f"The platforms claimed {_show(over_claim['platform_attributed_revenue'])} against "
            f"{_show(over_claim['shopify_revenue'])} of store revenue, a ratio of "
            f"{_show(over_claim['ratio'])}."
        )

    # (c) the channel behind most of the change
    driver = (facts["drivers"] or {}).get("platform_attributed_revenue")
    if driver and driver["largest"]:
        largest = driver["largest"]
        if driver["still_filling_in"]:
            # Not "moved most": the tables are not in, so there is no move yet.
            parts.append(
                f"{largest['channel']} accounts for most of the difference against last week, "
                f"{_show(largest['change'])}, on tables still filling in."
            )
        else:
            # A share over 100% is real -- channels pulling against each other --
            # but "285.2% of the change" explains nothing, so it is left to the
            # facts rather than written into a sentence.
            share = largest["share_of_change"]
            within = share is not None and abs(share["value"]) <= 1.0
            parts.append(
                f"{largest['channel']} moved attributed revenue most, {largest['direction']} "
                f"{_show(largest['change'])}" + (f", {_show(share)} of the change." if within else ".")
            )

    # the holdout, if one was running
    for holdout in facts["holdouts"]["running"][:1]:
        parts.append(
            f"The {holdout['channel']} holdout ({holdout['window_start']} to "
            f"{holdout['window_end']}) measured incremental ROAS of "
            f"{_show(holdout['incremental_roas'])} against a reported "
            f"{_show(holdout['reported_roas_in_window'])}."
        )

    parts.append("Template read-out: no model was called.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Writing read-outs
# --------------------------------------------------------------------------


def readout_record(facts: dict, text: str, model: str | None, dry_run: bool) -> dict:
    return {
        "week": week_number(facts),
        "week_start": facts["week"]["week_start"],
        "week_end": facts["week"]["week_end"],
        "brand": facts["brand"],
        "disclaimer": facts["disclaimer"],
        "mode": "dry-run" if dry_run else "model",
        "model": None if dry_run else model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "word_count": word_count(text),
        "word_limit": WORD_LIMIT,
        "text": text,
        "facts": facts,
    }


def readout_path(out_dir: str | Path, week: int) -> Path:
    return Path(out_dir) / f"week-{week:02d}.json"


def write_readout(out_dir: str | Path, record: dict) -> Path:
    path = readout_path(out_dir, record["week"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m gmarge.analyst",
        description="Generate weekly read-outs into readouts/. Offline; the app never does this.",
    )
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, help="how many of the most recent weeks (default 8)")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="data directory (default data)")
    parser.add_argument("--out", default=DEFAULT_READOUT_DIR, help="where to write (default readouts)")
    parser.add_argument("--model", default=None, help=f"model id (default {llm.MODEL_VARIABLE} or {llm.DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true", help="write template read-outs, with no API call")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.weeks < 1:
        print("--weeks must be at least 1", file=sys.stderr)
        return 2

    analysis = analyse(args.data)
    weeks = analysis.weeks[-args.weeks :]
    model = None if args.dry_run else llm.model_id(args.model)

    print(
        f"{len(weeks)} week(s): {weeks[0]} to {weeks[-1]}"
        + ("  [dry run, no API calls]" if args.dry_run else f"  [{model}]"),
        flush=True,
    )

    failed: list[int] = []
    for week in weeks:
        facts = build_facts(analysis, week)
        try:
            text = template_readout(facts) if args.dry_run else generate_readout(facts, model=model)
        except (NumberCheckError, llm.ModelError) as exc:
            print(f"  week {week:>2}: NOT SAVED -- {exc}", file=sys.stderr)
            failed.append(week)
            continue

        unsupported = check_numbers(text, facts)
        if unsupported:  # the template is held to the same bar as the model
            print(f"  week {week:>2}: NOT SAVED -- template used {', '.join(unsupported)}", file=sys.stderr)
            for line in offending_sentences(text, unsupported, facts):
                print(f"      {line}", file=sys.stderr)
            failed.append(week)
            continue

        record = readout_record(facts, text, model, args.dry_run)
        path = write_readout(args.out, record)
        over = "" if record["word_count"] <= WORD_LIMIT else f"  OVER the {WORD_LIMIT}-word limit"
        print(f"  week {week:>2}: {path} ({record['word_count']} words){over}", flush=True)

    if failed:
        print(f"\n{len(failed)} week(s) not saved: {failed}. Nothing was written for them.", file=sys.stderr)
        return 1

    print("\nRead these before committing them -- they are drafts until a human has checked them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
