"""Weekly read-outs: Python computes the numbers, the model only phrases them.

The shape of the pipeline (see CLAUDE.md) is::

    metrics + quality findings + anomalies
        -> build_facts(week)      every number the read-out may use
        -> the model              phrasing, and nothing else
        -> check_numbers()        every figure in the reply traced to a fact
        -> readouts/week-NN.json  committed, after a human has read it

The model is never asked to add, divide, compare or estimate. It is handed a
page of facts and asked for at most 120 words of prose. Anything it writes
that cannot be traced back to one of those facts is treated as a fault: the
request is made once more with the offending figures named, and if the second
reply is no better, nothing is saved.

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
from gmarge.anomalies import (
    MAD_FLOOR_SHARE,
    MAD_TO_SIGMA,
    SCORE_THRESHOLD,
    TRAILING_WEEKS,
    Anomaly,
    detect_anomalies,
    modified_z_score,
)
from gmarge.metrics import DEFAULT_DATA_DIR, Tables, all_metrics, load_tables
from gmarge.quality import Finding, run_checks

DEFAULT_READOUT_DIR = "readouts"
DEFAULT_WEEKS = 8
WORD_LIMIT = 120

MONEY_DP = 2
RATIO_DP = 4

# The headline figures a read-out is allowed to lead on, and how each is
# rounded in the facts. Rounding here and not at the point of use means the
# number the model is shown is exactly the number the check will accept.
HEADLINE = {
    "shopify_revenue": MONEY_DP,
    "shopify_net_revenue": MONEY_DP,
    "ad_spend": MONEY_DP,
    "platform_attributed_revenue": MONEY_DP,
    "over_claimed_revenue": MONEY_DP,
    "over_claim_ratio": RATIO_DP,
    "blended_reported_roas": RATIO_DP,
}

# Which of those get a trailing-window band. Levels and ratios both move for
# ordinary reasons, so the band is what says whether this week's move is one
# of them.
BANDED = ("shopify_revenue", "ad_spend", "platform_attributed_revenue", "over_claim_ratio")

LABELS = {
    "shopify_revenue": "store revenue",
    "shopify_net_revenue": "net store revenue",
    "ad_spend": "ad spend",
    "platform_attributed_revenue": "platform-attributed revenue",
    "over_claimed_revenue": "over-claimed revenue",
    "over_claim_ratio": "over-claim ratio",
    "blended_reported_roas": "blended reported ROAS",
}

# Week-on-week changes that can be split by channel. Store revenue cannot be:
# the platforms' attributed revenue is not the store's revenue.
DRIVEN = ("ad_spend", "platform_attributed_revenue")

# The checks that mean a day is missing or still filling in, as opposed to
# present but wrong. Only these bear on whether the week's data is complete.
INCOMPLETE_CHECKS = ("reporting_lag", "missing_days")


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


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------


def _round(value, places: int):
    """Round for presentation, keeping ``None`` for anything undefined."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, places)


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
    return {key: _round(values[key], places) for key, places in HEADLINE.items()}


def _change(current: dict, prior: dict) -> dict:
    """Absolute and relative week-on-week change, per headline figure."""
    out = {}
    for key, places in HEADLINE.items():
        now, before = current.get(key), prior.get(key)
        if now is None or before is None:
            out[key] = {"absolute": None, "pct": None, "direction": "unknown"}
            continue
        absolute = now - before
        out[key] = {
            "absolute": _round(absolute, places),
            "pct": _round(absolute / before, RATIO_DP) if before else None,
            "direction": "up" if absolute > 0 else "down" if absolute < 0 else "flat",
        }
    return out


def _band(series: pd.Series, week: int, value: float, places: int, trailing_weeks: int) -> dict:
    """This week against its own trailing weeks, scored the way a flag is.

    The same modified z-score the anomaly detector uses, so "outside the normal
    range" in a read-out means what it means in a flag.
    """
    trailing = series.loc[week - trailing_weeks : week - 1].dropna()
    if len(trailing) < trailing_weeks or value is None:
        return {
            "trailing_weeks": int(len(trailing)),
            "enough_history": False,
            "outside_normal_range": None,
        }

    score, median, mad = modified_z_score(trailing, float(value))
    scale = max(mad, MAD_FLOOR_SHARE * abs(median))
    half_width = SCORE_THRESHOLD * scale / MAD_TO_SIGMA
    return {
        "trailing_weeks": int(trailing_weeks),
        "enough_history": True,
        "trailing_median": _round(median, places),
        "normal_range_low": _round(median - half_width, places),
        "normal_range_high": _round(median + half_width, places),
        "robust_score": _round(score, 1),
        "score_threshold": SCORE_THRESHOLD,
        "outside_normal_range": bool(abs(score) >= SCORE_THRESHOLD),
    }


def _bands(weekly: pd.DataFrame, week: int, headline: dict, trailing_weeks: int) -> dict:
    indexed = weekly.set_index("week")
    return {
        key: _band(indexed[key], week, headline.get(key), HEADLINE[key], trailing_weeks)
        for key in BANDED
    }


def _channels(roas_week: pd.DataFrame, week: int, prior_week: int | None) -> list[dict]:
    """Per channel: this week's spend and attributed revenue, and the move."""
    now = roas_week[roas_week["week"] == week].set_index("channel")
    before = roas_week[roas_week["week"] == prior_week].set_index("channel") if prior_week else None

    rows = []
    for channel, row in now.iterrows():
        prior = before.loc[channel] if before is not None and channel in before.index else None
        entry = {
            "channel": str(channel),
            "spend": _round(row["spend"], MONEY_DP),
            "platform_attributed_revenue": _round(row["platform_attributed_revenue"], MONEY_DP),
            "reported_roas": _round(row["reported_roas"], RATIO_DP),
        }
        for column in DRIVEN:
            source = "spend" if column == "ad_spend" else column
            entry[f"{column}_change"] = (
                _round(float(row[source]) - float(prior[source]), MONEY_DP) if prior is not None else None
            )
        rows.append(entry)
    return sorted(rows, key=lambda r: r["spend"], reverse=True)


def _drivers(channels: list[dict], change: dict) -> dict:
    """Which channel moved a week-on-week change, and by how much of it.

    A share is the channel's own move over the total move, so shares can
    exceed 100% when channels pull against each other -- which is a fact about
    the week, not a rounding problem, and is left visible.
    """
    out = {}
    for column in DRIVEN:
        total = change[column]["absolute"]
        contributions = [c for c in channels if c[f"{column}_change"] is not None]
        if total is None or not contributions:
            out[column] = {"total_change": total, "channels": [], "largest": None}
            continue

        ranked = sorted(contributions, key=lambda c: abs(c[f"{column}_change"]), reverse=True)
        listed = [
            {
                "channel": c["channel"],
                "change": c[f"{column}_change"],
                "share_of_change": _round(c[f"{column}_change"] / total, RATIO_DP) if total else None,
            }
            for c in ranked
        ]
        out[column] = {"total_change": total, "channels": listed, "largest": listed[0]}
    return out


def _anomalies(anomalies: list[Anomaly], week: int) -> list[dict]:
    """The week's flags, trimmed to what a read-out could say about them."""
    keep = (
        "value", "trailing_median", "pct_change", "robust_score",
        "trailing_weeks", "spend", "platform_attributed_revenue",
    )
    out = []
    for flag in (a for a in anomalies if a.week == week):
        record = flag.to_dict()
        record["numbers"] = {k: v for k, v in record["numbers"].items() if k in keep}
        record["share_of_move"] = _round(record["share_of_move"], RATIO_DP)
        out.append(record)
    return out


def _findings(findings: list[Finding], start: str, end: str) -> list[dict]:
    """Quality findings whose dates fall inside the week.

    A finding can straddle a week boundary, so each one also carries the days
    of its own that fall inside *this* week, and how many. A read-out that
    wants to say how many days a problem covers can then quote a number
    instead of working it out.
    """
    out = []
    for finding in findings:
        if finding.start_date > end or finding.end_date < start:
            continue
        record = finding.to_dict()
        inside = [day for day in record["dates"] if start <= day <= end]
        record["dates_in_week"] = inside
        record["n_days_in_week"] = len(inside)
        out.append(record)
    return out


def _completeness(findings: list[dict], days_in_week: int) -> dict:
    """Whether anything is known to be missing from the week, and how much.

    Every count a read-out could want is spelled out here -- days in the week,
    days with complete data, days affected, and the affected days themselves.
    The model is told to quote these and never to count or subtract days of
    its own, because a day it works out for itself is a day nobody checked.
    """
    incomplete = [f for f in findings if f["check"] in INCOMPLETE_CHECKS]
    affected_dates = sorted({day for f in incomplete for day in f["dates_in_week"]})

    return {
        "complete": not incomplete,
        "days_in_week": days_in_week,
        "days_affected": len(affected_dates),
        "days_with_complete_data": days_in_week - len(affected_dates),
        "affected_dates": affected_dates,
        "affected": [
            {
                "source": f["source"],
                "check": f["check"],
                "start_date": f["start_date"],
                "end_date": f["end_date"],
                "n_days": f["n_days"],
                "n_days_in_week": f["n_days_in_week"],
                "dates_in_week": f["dates_in_week"],
            }
            for f in incomplete
        ],
    }


def build_facts(analysis: Analysis, week: int, trailing_weeks: int = TRAILING_WEEKS) -> dict:
    """Every number the read-out for ``week`` is allowed to use.

    Nothing outside this dict may appear in the read-out, which is what makes
    :func:`check_numbers` a real check rather than a formality.
    """
    weekly = analysis.metrics["weekly_reconciliation"]
    if week not in set(int(w) for w in weekly["week"]):
        raise ValueError(f"week {week} is not in the data")

    row = weekly[weekly["week"] == week].iloc[0]
    prior_week = week - 1 if (weekly["week"] == week - 1).any() else None
    prior_row = weekly[weekly["week"] == prior_week].iloc[0] if prior_week else None

    headline = _headline(row)
    prior_headline = _headline(prior_row) if prior_row is not None else {}
    change = _change(headline, prior_headline)
    channels = _channels(analysis.metrics["reported_roas_by_channel_week"], week, prior_week)
    findings = _findings(analysis.findings, str(row["week_start"].date()), str(row["week_end"].date()))

    return {
        "brand": analysis.metrics["brand"],
        "disclaimer": analysis.metrics["disclaimer"],
        "week": {
            "week": int(week),
            "week_start": str(row["week_start"].date()),
            "week_end": str(row["week_end"].date()),
            "days": int(row["days"]),
        },
        "prior_week": (
            {
                "week": int(prior_week),
                "week_start": str(prior_row["week_start"].date()),
                "week_end": str(prior_row["week_end"].date()),
            }
            if prior_row is not None
            else None
        ),
        "data_window": analysis.metrics["window"],
        "totals": headline,
        "prior_week_totals": prior_headline or None,
        "change_vs_prior_week": change if prior_row is not None else None,
        "normal_range": _bands(weekly, week, headline, trailing_weeks),
        "channels": channels,
        "drivers": _drivers(channels, change) if prior_row is not None else None,
        "anomalies": _anomalies(analysis.anomalies, week),
        "quality_findings": findings,
        "completeness": _completeness(findings, int(row["days"])),
        "notes": [
            "Store revenue cannot be split by channel; platform-attributed revenue is not store revenue.",
            "Reported ROAS is what the platforms claim, not measured incremental ROAS.",
        ],
    }


# --------------------------------------------------------------------------
# The number check
# --------------------------------------------------------------------------

DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def offending_sentences(text: str, unsupported: list[str], facts: dict) -> list[str]:
    """Each unsupported figure with the sentence it was written in.

    A bare list of figures says a read-out failed; it does not say what the
    model was trying to write. ``6`` means nothing on its own and everything
    once you can see it sat in "only 6 of 7 days reported" -- that is a model
    counting days for itself, and it is fixed in the facts, not in the check.

    A sentence is searched with the same tokeniser the check uses rather than
    for the text of the figure, because ``6`` is a substring of ``$563,472.11``
    and would otherwise point at the wrong sentence.
    """
    _, fact_strings = _fact_values(facts)
    sentences = [s.strip() for s in SENTENCE.split(text.strip()) if s.strip()]

    lines = []
    for token in unsupported:
        where = next(
            (s for s in sentences if token in _written_tokens(s, fact_strings)),
            text.strip(),
        )
        lines.append(f"{token!r} in: {where}")
    return lines


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


def _fact_values(facts: dict) -> tuple[list[float], set[str]]:
    """Every number and every string anywhere in the facts."""
    numbers: list[float] = []
    strings: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            numbers.append(float(node))
        elif isinstance(node, str):
            strings.add(node)

    walk(facts)
    return numbers, strings


def _parse_number(token: str) -> tuple[float, int]:
    """``(magnitude, decimal places written)`` for one written figure."""
    body = token.rstrip("%").replace("$", "").replace(",", "").lstrip("-")
    return abs(float(body)), len(body.partition(".")[2])


def _supported(token: str, fact_numbers: list[float]) -> bool:
    """Whether one written figure rounds to a number in the facts.

    Three allowances, and no others. A figure may be written to fewer decimal
    places than the fact it comes from; a ratio may be written as a percentage
    (0.0191 as 1.9%); and sign is carried by the wording rather than the digits,
    so magnitudes are compared. Rounding to the nearest thousand, abbreviating
    and unit conversion are all mismatches, which is deliberate -- the point is
    to catch a figure the model made up, and made-up figures are usually round.
    """
    written, places = _parse_number(token)
    tolerance = 0.5 * 10 ** -places + 1e-9
    for value in fact_numbers:
        for candidate in (abs(value), abs(value) * 100.0):
            if abs(candidate - written) <= tolerance:
                return True
    return False


def _mask(text: str, fact_strings: set[str]) -> str:
    """Blank out anything quoted verbatim from the facts.

    Names carry digits -- the ad set ``Core | Broad 25-44``, the region
    ``Region 03`` -- and so do the descriptions the quality and anomaly
    modules write, which are themselves computed in Python. Quoting one of
    those is not inventing a figure, so it is taken out of the text before the
    figures that remain are checked one by one. Longest first, so a name
    inside a description cannot be half-masked.
    """
    for string in sorted((s for s in fact_strings if any(c.isdigit() for c in s)), key=len, reverse=True):
        text = text.replace(string, " ")
    return text


def _written_tokens(text: str, fact_strings: set[str]) -> list[str]:
    """Every date and figure written in ``text``, as the check sees them."""
    return DATE.findall(text) + NUMBER.findall(_mask(DATE.sub(" ", text), fact_strings))


def check_numbers(text: str, facts: dict) -> list[str]:
    """Every figure in ``text`` that cannot be traced to ``facts``.

    An empty list means the read-out is safe to save.
    """
    fact_numbers, fact_strings = _fact_values(facts)
    haystack = " ".join(fact_strings)

    unsupported = []
    for token in _written_tokens(text, fact_strings):
        supported = token in haystack if DATE.fullmatch(token) else _supported(token, fact_numbers)
        if not supported:
            unsupported.append(token)
    return unsupported


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------
# Asking the model
# --------------------------------------------------------------------------

SYSTEM = f"""You write the weekly read-out for a marketing-measurement demo.

Every figure you could need is in the FACTS the user message carries. These
rules are hard:

1. At most {WORD_LIMIT} words. Plain prose in one or two short paragraphs. No
   headings, no bullet points, no markdown.
2. Cover three things, in this order: what changed this week, what caused most
   of it, and whether the week is outside its normal range.
3. Use only numbers that appear in the FACTS. Copy each figure as it is written
   there. You may drop decimal places ($563,472.11 as $563,472). You may not
   round to the nearest thousand, abbreviate (no "k", no "m"), convert a unit,
   or introduce a figure of your own. Do no arithmetic of any kind: if a number
   is not in the FACTS, you cannot use it.
4. If `completeness.complete` is false, say so and name what is missing. Say it
   only with the counts already in `completeness`: `days_in_week`,
   `days_with_complete_data`, `days_affected`, `affected_dates`, and the
   `source` of each entry in `affected`. Do not count days, do not subtract one
   count from another, and do not work out a day or a date from a range. If you
   want to say how many days something covers, there is a count for it; quote
   that count or leave it out.
5. This is synthetic sample data for a brand that does not exist. Do not write
   as though it were a real company's results, and do not recommend budget
   decisions."""


def build_prompt(facts: dict) -> str:
    return (
        "FACTS\n"
        + json.dumps(facts, indent=2, default=str)
        + f"\n\nWrite the read-out for week {facts['week']['week']}."
    )


def retry_prompt(prompt: str, unsupported: list[str], text: str, facts: dict) -> str:
    """The same request again, with the figures that failed the check named.

    The sentence each figure was written in goes back too: it is usually the
    sentence, not the figure, that shows what went wrong -- a count worked out
    from a date range, a total added up by hand.
    """
    return (
        prompt
        + "\n\nYour previous read-out used "
        + ", ".join(unsupported)
        + ", which are not in the FACTS above:\n"
        + "\n".join(f"  {line}" for line in offending_sentences(text, unsupported, facts))
        + "\n\nWrite the read-out again, using only figures that appear in the FACTS, "
        "copied exactly as written there. If you need a count, quote the one in the "
        "FACTS rather than working it out."
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
            raise NumberCheckError(facts["week"]["week"], unsupported, text, facts)
    return text


# --------------------------------------------------------------------------
# The template read-out (--dry-run)
# --------------------------------------------------------------------------


def _money(value: float | None) -> str:
    return "unavailable" if value is None else f"${value:,.2f}"


def _pct(value: float | None) -> str:
    return "unavailable" if value is None else f"{abs(value) * 100:.1f}%"


def template_readout(facts: dict) -> str:
    """A read-out assembled in Python, for tests and for --dry-run.

    Built from the same facts and held to the same number check, so it stays
    honest about what the facts can support.
    """
    week, totals = facts["week"], facts["totals"]
    parts = [
        f"Week {week['week']} ({week['week_start']} to {week['week_end']}), "
        f"{facts['brand']}. Template read-out: no model was called."
    ]

    change = facts["change_vs_prior_week"]
    if change:
        revenue, spend = change["shopify_revenue"], change["ad_spend"]
        parts.append(
            f"Store revenue was {_money(totals['shopify_revenue'])}, "
            f"{revenue['direction']} {_pct(revenue['pct'])} on the prior week, "
            f"on ad spend of {_money(totals['ad_spend'])}, {spend['direction']} {_pct(spend['pct'])}."
        )
        largest = (facts["drivers"] or {}).get("platform_attributed_revenue", {}).get("largest")
        if largest:
            parts.append(
                f"{largest['channel']} moved attributed revenue most, by "
                f"{_money(abs(largest['change']))}."
            )
    else:
        parts.append(f"Store revenue was {_money(totals['shopify_revenue'])}.")

    flag = next(iter(facts["anomalies"]), None)
    if flag:
        parts.append(
            f"{flag['channel']}'s {flag['metric_label']} was flagged "
            f"{flag['direction']} {_pct(flag['pct_change'])} against its trailing median, "
            f"most of it from ad set {flag['ad_set']}."
        )

    outside = [LABELS[name] for name, band in facts["normal_range"].items() if band.get("outside_normal_range")]
    parts.append(
        f"Outside its trailing {TRAILING_WEEKS}-week range: {', '.join(outside)}."
        if outside
        else f"Every headline figure sits inside its trailing {TRAILING_WEEKS}-week range."
    )

    completeness = facts["completeness"]
    if not completeness["complete"]:
        sources = sorted({a["source"] for a in completeness["affected"]})
        # Every count here is read out of the facts. Writing "7 - 2" in Python
        # would be the same mistake the model is forbidden to make.
        parts.append(
            f"Data is incomplete: {' and '.join(sources)} cover "
            f"{completeness['days_with_complete_data']} of the week's "
            f"{completeness['days_in_week']} days, with "
            f"{completeness['days_affected']} still filling in "
            f"({' and '.join(completeness['affected_dates'])})."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------
# Writing read-outs
# --------------------------------------------------------------------------


def readout_record(facts: dict, text: str, model: str | None, dry_run: bool) -> dict:
    return {
        "week": facts["week"]["week"],
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

    print(f"{len(weeks)} week(s): {weeks[0]} to {weeks[-1]}" + ("  [dry run, no API calls]" if args.dry_run else f"  [{model}]"), flush=True)

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
