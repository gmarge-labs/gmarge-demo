"""The G-Marge demo app: what the measurement looks like once someone reads it.

Run it with ``streamlit run app.py``.

This file displays. It does not measure and it does not phrase. Every number on
every page comes out of ``gmarge/metrics.py``, ``gmarge/quality.py`` or
``gmarge/anomalies.py``, and is written with the display helpers in
``gmarge/analyst.py`` -- the same ``$14.9k``, ``1.47x``, ``39.2%`` and plain
``2.88`` the weekly read-outs are held to. One formatter, so a figure on a page
and the same figure in a read-out cannot disagree.

**No AI runs here.** The app reads ``data/`` and ``readouts/`` off disk and
nothing else: no key, no network, no ``anthropic`` import anywhere in its
import graph (see CLAUDE.md, and ``tests/test_app.py``, which asserts it).
Read-outs are generated offline by ``python -m gmarge.analyst``, checked by a
human, and committed to ``readouts/``.

**Honesty rules this file implements.**

*The last two days are still filling in.* ``gmarge/quality.py`` finds them.
Every time-series chart shades that band and labels it ``provisional``.

*A lagging ratio is not a change.* Ad spend, platform-attributed revenue and
anything built on them are short over those days, so a ratio spanning them
moves for a reason that is not a result. ``gmarge/analyst.py`` withholds those
figures from a read-out rather than qualifying them; this file withholds them
from a chart the same way, and says what it withheld and why. The rule for what
is withheld is the analyst's ``TOTAL_SOURCES``, not a second opinion.

*Reported is not incremental.* Platform-reported figures are grey, measured
incremental figures are light blue, everywhere, with no exception.

*The data is invented.* The banner says so on every page, and every chart is
marked illustrative.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from gmarge import anomalies as anomaly_scan
from gmarge import quality
from gmarge.analyst import (
    DISPLAY,
    LABELS,
    METRIC_KIND,
    RATIO,
    TOTAL_SOURCES,
    UNIT_KIND,
    count_display,
    level_display,
    money_display,
    pct_display,
    ratio_display,
)
from gmarge.metrics import all_metrics, holdout_plan, load_tables

# --------------------------------------------------------------------------
# Brand
# --------------------------------------------------------------------------

CONTACT_URL = "https://www.gmarge.com/contact"
BANNER = "Sample data for a fictional brand. See what G-Marge would show you."

# The page's navy, background and surfaces are set in .streamlit/config.toml
# and in the stylesheet below. The two colours here are the ones charts encode
# with, which is why they live in Python.
#
# The two that carry the argument. Light blue is what a holdout measured; grey
# is what a platform claimed. Nothing else may use either colour.
INCREMENTAL = "#8FC0FF"
REPORTED = "#66728C"

TEXT = "#E7EEFA"
MUTED = "#9AA7C0"
GRID = "#123B78"

# Amber, for "provisional" only -- a third meaning, so it cannot be misread as
# either figure colour.
LAG = "#E3A857"

SEVERITY_COLOURS = {"high": "#FF8A80", "medium": LAG, "low": MUTED}

FONT = "Inter, sans-serif"

DATA_DIR = "data"
READOUT_DIR = "readouts"

ILLUSTRATIVE = "Illustrative. Sample data for a fictional brand -- not any real advertiser's performance."

PAGES = ["Overview", "Channels", "Holdout tests", "Agent read-outs", "Data health"]


# --------------------------------------------------------------------------
# Loading. Cached, so switching pages re-reads nothing.
# --------------------------------------------------------------------------


@st.cache_data(show_spinner="Measuring...")
def load_analysis(data_dir: str = DATA_DIR) -> dict:
    """Metrics, quality findings and anomaly flags, in one pass over the data."""
    tables = load_tables(data_dir)
    return {
        "metrics": all_metrics(tables),
        "findings": quality.run_checks(tables),
        "anomalies": anomaly_scan.detect_anomalies(tables),
        "holdout_plan": holdout_plan(tables.truth),
        "orders": tables.orders,
    }


@st.cache_data(show_spinner=False)
def load_readouts(readout_dir: str = READOUT_DIR) -> list[dict]:
    """Every committed read-out, newest week first.

    Only files. Nothing here generates one, and a week with no file is a week
    whose read-out failed its number check -- see ``readouts/README.md``.
    """
    records = []
    for path in sorted(Path(readout_dir).glob("week-*.json")):
        record = json.loads(path.read_text())
        record["path"] = path.as_posix()
        records.append(record)
    return sorted(records, key=lambda r: r["week"], reverse=True)


# --------------------------------------------------------------------------
# The reporting lag: which days, which tables, and what that spoils
# --------------------------------------------------------------------------


# What a page calls each table. The read-outs use the same names.
SOURCE_LABELS = {
    "ga4_sessions": "GA4 sessions",
    "ad_spend": "ad spend",
    "shopify_orders": "Shopify orders",
}


def lag_window(findings: list[quality.Finding]) -> dict | None:
    """The days still filling in, the tables behind them, and the totals they spoil.

    ``withheld`` is the analyst's rule and not a second opinion: a total is
    withheld when a table it is read from is one of the lagging ones. That is
    what keeps a chart here and a read-out in ``readouts/`` from disagreeing
    about whether week 26 fell.
    """
    lagging = [f for f in findings if f.check == "reporting_lag"]
    if not lagging:
        return None

    dates = sorted({day for f in lagging for day in f.dates})
    sources = {f.source for f in lagging}
    return {
        "start": pd.Timestamp(dates[0]),
        "end": pd.Timestamp(dates[-1]),
        "n_days": len(dates),
        "sources": sorted(sources),
        "source_labels": [SOURCE_LABELS.get(s, s) for s in sorted(sources)],
        "withheld": [key for key, tables in TOTAL_SOURCES.items() if sources.intersection(tables)],
    }


def complete_weeks(weekly: pd.DataFrame, lag: dict | None) -> pd.DataFrame:
    """Weeks that end before the lag starts -- the weeks a ratio can be read from."""
    if lag is None:
        return weekly
    return weekly[weekly["week_end"] < lag["start"]]


def lag_note(lag: dict | None) -> str:
    """One sentence naming the days, the tables and the figures held back."""
    if lag is None:
        return ""
    labels = [LABELS[key] for key in lag["withheld"]]
    sources = _join(lag["source_labels"])
    return (
        f"The last {count_display(lag['n_days'])} days "
        f"({lag['start'].date()} to {lag['end'].date()}) are still filling in on {sources}. "
        f"{_sentence_case(_join(labels))} are provisional there, so no total, change or ratio is "
        "drawn across those days: a move measured over a lagging table is the lag, not a result. "
        "Shopify orders are complete, so store revenue is."
    )


def _sentence_case(text: str) -> str:
    """Raise the first letter and leave the rest alone -- ``ROAS`` stays ``ROAS``."""
    return text[:1].upper() + text[1:]


def _join(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------
# Chart furniture
# --------------------------------------------------------------------------


def style(chart: alt.Chart, height: int = 260) -> alt.Chart:
    """Brand styling, applied to the finished (layered) chart and nowhere else."""
    return (
        chart.properties(height=height)
        .configure(font=FONT, background="transparent")
        .configure_view(strokeWidth=0, continuousHeight=height)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
            gridColor=GRID,
            gridOpacity=0.55,
            domainColor=GRID,
            tickColor=GRID,
        )
        .configure_legend(
            labelColor=TEXT,
            titleColor=MUTED,
            labelFontSize=11,
            titleFontSize=11,
            orient="top",
            direction="horizontal",
            offset=4,
            symbolType="square",
        )
        .configure_title(color=TEXT, fontSize=13, fontWeight=600, anchor="start")
    )


def lag_layers(lag: dict | None, covers_start, covers_end) -> list[alt.Chart]:
    """The shaded ``provisional`` band, when the chart's days actually reach it.

    ``covers_start`` and ``covers_end`` are the first and last *day* the chart
    is about, which on a weekly chart is not the first and last plotted point:
    week 26 is drawn at 2025-06-30 but covers 2025-07-06, and the lagging days
    are at the end of it. Passing the covered span rather than the marker
    positions is what puts the band over the right days.

    Exactly the lagging days are shaded -- never the whole week holding them,
    which would say a week of Shopify revenue is provisional when it is not.

    A band outside the plotted range would stretch the axis to reach days the
    chart is not about, so it is only drawn where it overlaps.
    """
    if lag is None or covers_start is None or covers_end is None:
        return []
    if pd.Timestamp(covers_end) < lag["start"] or pd.Timestamp(covers_start) > lag["end"]:
        return []

    band = pd.DataFrame(
        {"start": [lag["start"]], "end": [lag["end"] + pd.Timedelta(days=1)]}
    )
    shading = (
        alt.Chart(band)
        .mark_rect(color=LAG, opacity=0.16)
        .encode(x=alt.X("start:T", title=None), x2="end:T")
    )
    label = (
        alt.Chart(band)
        .mark_text(
            text="provisional",
            align="right",
            baseline="top",
            dx=-4,
            dy=3,
            fontSize=10,
            color=LAG,
        )
        .encode(x=alt.X("start:T", title=None), y=alt.value(0))
    )
    return [shading, label]


PROVISIONAL = (
    "The shaded days are still filling in on the ad platforms and in GA4, and are labelled "
    "provisional; nothing is read as a change across them."
)


def no_math(body: str) -> str:
    """A dollar sign is a dollar sign here, never the start of an equation.

    Streamlit reads ``$...$`` as LaTeX, so two money figures in one sentence
    turn into an equation and the sentence between them is lost. Nearly every
    figure on these pages is money, so it is escaped in one place rather than
    remembered at thirty call sites.
    """
    return body.replace("$", r"\$")


def no_math_html(body: str) -> str:
    r"""The same, for a block rendered as raw HTML.

    A markdown backslash-escape is consumed by the markdown parser; inside a
    block of HTML it is not, and ``\$563.5k`` reaches the page with the
    backslash still on it. The entity has no ``$`` in the source for the LaTeX
    pass to find, and the browser renders it as one.
    """
    return body.replace("$", "&#36;")


def md(body: str, **kwargs) -> None:
    escape = no_math_html if kwargs.get("unsafe_allow_html") else no_math
    st.markdown(escape(body), **kwargs)


def cap(body: str) -> None:
    st.caption(no_math(body))


def note(body: str) -> None:
    st.info(no_math(body))


def chart_note(text: str = "") -> None:
    """Every chart is marked illustrative, under the chart, every time."""
    cap(f"{text} {ILLUSTRATIVE}".strip())


def show(chart: alt.Chart, note: str = "", height: int = 260) -> None:
    st.altair_chart(style(chart, height), width="stretch")
    chart_note(note)


# --------------------------------------------------------------------------
# Page furniture
# --------------------------------------------------------------------------


def css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Inter is set as the theme font in .streamlit/config.toml; this only
           makes the family available. Do not set font-family broadly here --
           Streamlit's icons are ligatures in their own font, and overriding it
           renders them as the literal word "keyboard_double_arrow_left". */

        /* Readable line length on a laptop, full bleed on a phone. The top
           padding clears Streamlit's fixed toolbar, which would otherwise sit
           over the banner. */
        .block-container { max-width: 1080px; padding-top: 3.5rem; padding-bottom: 3rem; }
        @media (max-width: 640px) {
            .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
            h1 { font-size: 1.5rem !important; }
            h2 { font-size: 1.2rem !important; }
        }

        .gm-banner {
            background: linear-gradient(90deg, #002B6B 0%, #00204F 100%);
            border: 1px solid #123B78;
            border-left: 3px solid #8FC0FF;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 1.1rem;
            font-size: 0.9rem;
            line-height: 1.45;
            color: #E7EEFA;
        }
        .gm-banner a { color: #8FC0FF; font-weight: 600; text-decoration: none; }
        .gm-banner a:hover { text-decoration: underline; }

        .gm-card {
            background: #002145;
            border: 1px solid #123B78;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
        }
        .gm-card h4 { margin: 0 0 0.35rem 0; font-size: 0.95rem; color: #E7EEFA; }
        .gm-card p { margin: 0.2rem 0; font-size: 0.9rem; line-height: 1.5; color: #C9D6EC; }

        .gm-readout {
            background: #002145;
            border: 1px solid #123B78;
            border-left: 3px solid #8FC0FF;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            font-size: 1rem;
            line-height: 1.65;
            color: #E7EEFA;
        }

        /* Charts carry no element toolbar. At phone width its fullscreen
           button sits over the chart's top-right corner and the "provisional"
           label, which is the one thing on that corner that has to be read.
           Scoped to charts by :has(), so the dataframe toolbars -- search,
           download, which are useful -- are untouched. */
        [data-testid="stFullScreenFrame"]:has([data-testid="stVegaLiteChart"])
            [data-testid="stElementToolbar"] { display: none !important; }

        .gm-key { font-size: 0.82rem; color: #9AA7C0; margin: -0.3rem 0 0.9rem 0; }
        .gm-swatch { display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 2px; margin-right: 0.3rem; }
        .gm-tag {
            display: inline-block; border-radius: 4px; padding: 0.05rem 0.4rem;
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def banner() -> None:
    md(
        f'<div class="gm-banner">{BANNER} '
        f'<a href="{CONTACT_URL}" target="_blank">Talk to us &rarr;</a></div>',
        unsafe_allow_html=True,
    )


def colour_key() -> None:
    md(
        f'<div class="gm-key">'
        f'<span class="gm-swatch" style="background:{REPORTED}"></span>Platform-reported'
        f'&nbsp;&nbsp;&nbsp;'
        f'<span class="gm-swatch" style="background:{INCREMENTAL}"></span>Measured incremental'
        f'&nbsp;&nbsp;&nbsp;'
        f'<span class="gm-swatch" style="background:{LAG}"></span>Provisional (still filling in)'
        f"</div>",
        unsafe_allow_html=True,
    )


def tag(severity: str) -> str:
    colour = SEVERITY_COLOURS.get(severity, MUTED)
    return f'<span class="gm-tag" style="background:{colour}22;color:{colour}">{severity}</span>'


def card(title: str, body: str) -> None:
    md(f'<div class="gm-card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Page 1: Overview
# --------------------------------------------------------------------------


def page_overview(analysis: dict, lag: dict | None) -> None:
    metrics = analysis["metrics"]
    head = metrics["headline"]
    window = metrics["window"]
    weekly = metrics["weekly_reconciliation"]

    st.title("Where the money actually went", anchor=False)
    md(
        f"**{metrics['brand']}** &nbsp;·&nbsp; {window['start']} to {window['end']} "
        f"&nbsp;·&nbsp; {count_display(window['n_weeks'])} weeks",
        unsafe_allow_html=True,
    )
    colour_key()

    left, right = st.columns(2)
    left.metric("Store revenue (Shopify)", money_display(head["shopify_revenue"]))
    right.metric("Ad spend", money_display(head["ad_spend"]))

    left, right = st.columns(2)
    left.metric(
        "Blended reported ROAS",
        ratio_display(head["blended_reported_roas"]),
        help="What the platforms claim, added up. Not incremental.",
    )
    right.metric(
        "Measured incremental ROAS",
        ratio_display(head["measured_incremental_roas"]),
        help="Spend-weighted across the five geo holdouts. What the spend was actually worth.",
    )

    gap = head["blended_reported_roas"] / head["measured_incremental_roas"]
    md(
        f"The platforms claim **{ratio_display(head['blended_reported_roas'])}**. "
        f"The holdouts measure **{ratio_display(head['measured_incremental_roas'])}**. "
        f"Claimed is **{ratio_display(gap)}** measured — "
        f"**{money_display(head['over_claimed_revenue'])}** of revenue credited to ads that the "
        f"store never took, at an over-claim ratio of "
        f"**{ratio_display(head['over_claim_ratio'])}**."
    )

    if lag:
        note(lag_note(lag))

    st.subheader("Store revenue by week", anchor=False)
    cap("Shopify, complete for every day in the window. Gross revenue, before discounts and refunds.")
    revenue = (
        alt.Chart(weekly)
        .mark_line(color=INCREMENTAL, strokeWidth=2, point=alt.OverlayMarkDef(color=INCREMENTAL, size=28))
        .encode(
            x=alt.X("week_start:T", title="Week beginning", axis=alt.Axis(format="%d %b")),
            y=alt.Y("shopify_revenue:Q", title="Store revenue", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("week:Q", title="Week"),
                alt.Tooltip("week_start:T", title="Week beginning"),
                alt.Tooltip("shopify_revenue:Q", title="Store revenue", format="$,.0f"),
            ],
        )
    )
    show(
        alt.layer(*lag_layers(lag, weekly["week_start"].min(), weekly["week_end"].max()), revenue),
        f"{PROVISIONAL} Shopify is not one of them, so this line is complete across them.",
    )

    st.subheader("The over-claim ratio, week by week", anchor=False)
    cap(
        "Platform-attributed revenue divided by what the store actually took. "
        "1.00x would mean the platforms, between them, claimed the store's revenue exactly once."
    )
    shown = complete_weeks(weekly, lag)
    ratio_line = (
        alt.Chart(shown)
        .mark_line(color=REPORTED, strokeWidth=2, point=alt.OverlayMarkDef(color=REPORTED, size=28))
        .encode(
            x=alt.X("week_start:T", title="Week beginning", axis=alt.Axis(format="%d %b")),
            y=alt.Y(
                "over_claim_ratio:Q",
                title="Attributed / actual",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=".2f"),
            ),
            tooltip=[
                alt.Tooltip("week:Q", title="Week"),
                alt.Tooltip("over_claim_ratio:Q", title="Over-claim ratio", format=".2f"),
                alt.Tooltip("over_claimed_revenue:Q", title="Over-claimed", format="$,.0f"),
            ],
        )
    )
    once = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(color=MUTED, strokeDash=[4, 4], strokeWidth=1)
        .encode(y="y:Q")
    )
    withheld = len(weekly) - len(shown)
    show(
        alt.layer(
            *lag_layers(lag, weekly["week_start"].min(), weekly["week_end"].max()),
            once,
            ratio_line,
        ),
        (
            f"{PROVISIONAL} The last {count_display(withheld)} week is not plotted at all: its "
            f"attributed revenue is short by {count_display(lag['n_days'])} days that have not "
            "arrived while its store revenue is complete, so the ratio would fall for a reason "
            "that is not a result."
        )
        if withheld
        else "",
    )

    st.subheader("Latest read-out", anchor=False)
    readouts = load_readouts()
    if not readouts:
        st.warning("No read-outs are committed yet. Generate them with `python -m gmarge.analyst`.")
        return
    render_readout(readouts[0], expanded=False)
    cap(
        "Written offline by a model that was shown only figures Python had already computed and "
        "formatted, checked against those figures, and committed as a file. "
        "Nothing on this page calls an AI."
    )


# --------------------------------------------------------------------------
# Page 2: Channels
# --------------------------------------------------------------------------


def page_channels(analysis: dict, lag: dict | None) -> None:
    metrics = analysis["metrics"]
    channels = metrics["channel_summary"]
    holdouts = metrics["holdouts"]

    st.title("What each channel claims, and what it is worth", anchor=False)
    md(
        "Grey is what the platform reported over the holdout window. Light blue is what the "
        "holdout measured over the same window and the same control regions, so the two are "
        "directly comparable."
    )
    colour_key()

    paired = holdouts.merge(
        channels[["channel", "spend", "share_of_spend"]], on="channel", how="left"
    )
    order = paired.sort_values("over_claim_multiple", ascending=False)["channel"].tolist()

    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "channel": paired["channel"],
                    "series": "Platform-reported ROAS",
                    "roas": paired["reported_roas_in_window"],
                }
            ),
            pd.DataFrame(
                {
                    "channel": paired["channel"],
                    "series": "Measured incremental ROAS",
                    "roas": paired["incremental_roas"],
                }
            ),
        ]
    )

    bars = (
        alt.Chart(long)
        .mark_bar(height=13, cornerRadiusEnd=2)
        .encode(
            y=alt.Y("channel:N", title=None, sort=order, axis=alt.Axis(labelLimit=180)),
            yOffset=alt.YOffset("series:N", sort=["Platform-reported ROAS", "Measured incremental ROAS"]),
            # Room on the right for the gap label, which sits past the longer
            # bar. Wide enough that the label still fits at phone width.
            x=alt.X(
                "roas:Q",
                title="Return on ad spend",
                scale=alt.Scale(domain=[0, float(paired["reported_roas_in_window"].max()) * 1.45]),
            ),
            color=alt.Color(
                "series:N",
                title=None,
                sort=["Platform-reported ROAS", "Measured incremental ROAS"],
                scale=alt.Scale(
                    domain=["Platform-reported ROAS", "Measured incremental ROAS"],
                    range=[REPORTED, INCREMENTAL],
                ),
            ),
            tooltip=[
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip("series:N", title=None),
                alt.Tooltip("roas:Q", title="ROAS", format=".2f"),
            ],
        )
    )

    gaps = paired.assign(
        label=[
            f"{ratio_display(m)} over-claim" for m in paired["over_claim_multiple"]
        ]
    )
    labels = (
        alt.Chart(gaps)
        .mark_text(align="left", dx=7, fontSize=11, color=TEXT)
        .encode(
            y=alt.Y("channel:N", title=None, sort=order),
            x=alt.X("reported_roas_in_window:Q"),
            text="label:N",
        )
    )
    show(
        alt.layer(bars, labels),
        "Both figures cover the channel's own holdout window, not the whole six months.",
        height=300,
    )

    st.subheader("The gap, channel by channel", anchor=False)
    for row in order:
        item = paired[paired["channel"] == row].iloc[0]
        card(
            item["channel"],
            f"<p>Platform reported <b style='color:{REPORTED}'>"
            f"{ratio_display(item['reported_roas_in_window'])}</b>. "
            f"The holdout measured <b style='color:{INCREMENTAL}'>"
            f"{ratio_display(item['incremental_roas'])}</b> "
            f"(90% interval {ratio_display(item['incremental_roas_ci_low'])} to "
            f"{ratio_display(item['incremental_roas_ci_high'])}). "
            f"The claim is {ratio_display(item['over_claim_multiple'])} the measurement.</p>"
            f"<p style='color:{MUTED}'>"
            f"{money_display(item['spend'])} of spend over the whole window, "
            f"{pct_display(item['share_of_spend'])} of the budget. "
            f"Tested {item['start_date'].date()} to {item['end_date'].date()} across "
            f"{count_display(item['n_pairs'])} matched region pairs.</p>",
        )

    st.subheader("Whole-window spend and reported ROAS", anchor=False)
    table = channels.assign(
        Spend=[money_display(v) for v in channels["spend"]],
        **{
            "Share of budget": [pct_display(v) for v in channels["share_of_spend"]],
            "Attributed revenue": [
                money_display(v) for v in channels["platform_attributed_revenue"]
            ],
            "Reported ROAS": [ratio_display(v) for v in channels["reported_roas"]],
        },
    ).rename(columns={"channel": "Channel"})
    st.dataframe(
        table[["Channel", "Spend", "Share of budget", "Attributed revenue", "Reported ROAS"]],
        hide_index=True,
        width="stretch",
    )
    if lag:
        cap(
            f"Every figure in this table is platform-reported and includes "
            f"{lag['start'].date()} to {lag['end'].date()}, which are still filling in. "
            "They are totals, not changes; no week-on-week move is drawn across them."
        )
    cap(
        "Reported ROAS is what the platforms claim. Only a holdout says what a channel is worth. "
        f"{ILLUSTRATIVE}"
    )


# --------------------------------------------------------------------------
# Page 3: Holdout tests
# --------------------------------------------------------------------------


def holdout_series(
    orders: pd.DataFrame, holdout: dict, weeks_before: int = 4, weeks_after: int = 2
) -> pd.DataFrame:
    """Weekly revenue in the test and control regions, indexed to the pre-period.

    The two groups are different sizes, so a raw dollar comparison says nothing
    about the pause. Each group is indexed to its own average over the weeks
    before the test, which is exactly the assumption the estimate rests on: the
    two moved together until one of them was switched off. Any divergence
    inside the window is what the chart is for.

    Shape only. The lift, the incremental ROAS and the interval on the page
    come from ``metrics.holdout_results`` -- they are not read off this.
    """
    start = pd.Timestamp(holdout["start_date"])
    end = pd.Timestamp(holdout["end_date"])
    first = start - pd.Timedelta(weeks=weeks_before)
    last = min(end + pd.Timedelta(weeks=weeks_after), orders["date"].max())

    frame = orders[(orders["date"] >= first) & (orders["date"] <= last)].copy()
    groups = {"Test regions (paused)": holdout["test_regions"], "Control regions": holdout["control_regions"]}
    frame["group"] = pd.NA
    for name, regions in groups.items():
        frame.loc[frame["region"].isin(regions), "group"] = name
    frame = frame[frame["group"].notna()]

    # Weeks anchored to the test's own start, so the window edges are week edges.
    offset = (frame["date"] - start).dt.days // 7
    frame["week_start"] = start + pd.to_timedelta(offset * 7, unit="D")

    weekly = frame.groupby(["group", "week_start"], as_index=False)["revenue"].sum()
    base = (
        weekly[weekly["week_start"] < start].groupby("group")["revenue"].mean().rename("base")
    )
    weekly = weekly.merge(base, on="group")
    weekly["indexed"] = 100.0 * weekly["revenue"] / weekly["base"]
    weekly["in_test"] = (weekly["week_start"] >= start) & (weekly["week_start"] <= end)
    return weekly.sort_values(["group", "week_start"]).reset_index(drop=True)


def page_holdouts(analysis: dict, lag: dict | None) -> None:
    metrics = analysis["metrics"]
    holdouts = metrics["holdouts"]
    plan = {h["channel"]: h for h in analysis["holdout_plan"]}
    orders = analysis["orders"]

    st.title("Holdout tests", anchor=False)
    md(
        "Each test switched a channel off in five regions and left it running in five matched "
        "ones. What the paused regions did *not* earn, against what their matched controls "
        "predict they would have, is the channel's incremental revenue. Everything else is a claim."
    )
    colour_key()

    for row in holdouts.itertuples():
        st.subheader(f"{row.channel} — {row.start_date.date()} to {row.end_date.date()}", anchor=False)

        direction = "below" if row.incremental_revenue >= 0 else "above"
        md(
            f"Pausing **{row.channel}** in {count_display(row.n_pairs)} regions for "
            f"{count_display((row.end_date - row.start_date).days + 1)} days left them "
            f"**{pct_display(row.lift_pct)}** {direction} what their matched controls predict — "
            f"**{money_display(row.incremental_revenue)}** of revenue the channel had been "
            f"producing (90% interval {money_display(row.incremental_revenue_ci_low)} to "
            f"{money_display(row.incremental_revenue_ci_high)}). "
            f"Against the **{money_display(row.paused_spend_estimate)}** those regions would have "
            f"spent, that is an incremental ROAS of "
            f"**{ratio_display(row.incremental_roas)}** "
            f"({level_display(row.confidence)} interval {ratio_display(row.incremental_roas_ci_low)} "
            f"to {ratio_display(row.incremental_roas_ci_high)}). "
            f"Over the same window the platform reported "
            f"**{ratio_display(row.reported_roas_in_window)}** — "
            f"**{ratio_display(row.over_claim_multiple)}** what the test measured."
        )

        series = holdout_series(orders, plan[row.channel])
        window = pd.DataFrame({"start": [row.start_date], "end": [row.end_date + pd.Timedelta(days=1)]})
        paused = (
            alt.Chart(window)
            .mark_rect(color=INCREMENTAL, opacity=0.09)
            .encode(x=alt.X("start:T", title=None), x2="end:T")
        )
        paused_label = (
            alt.Chart(window)
            .mark_text(text="channel paused in test regions", align="left", baseline="top",
                       dx=5, dy=3, fontSize=10, color=INCREMENTAL)
            .encode(x=alt.X("start:T", title=None), y=alt.value(0))
        )
        lines = (
            alt.Chart(series)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=26))
            .encode(
                x=alt.X("week_start:T", title="Week beginning", axis=alt.Axis(format="%d %b")),
                y=alt.Y(
                    "indexed:Q",
                    title="Revenue, four weeks before the test = 100",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "group:N",
                    title=None,
                    sort=["Control regions", "Test regions (paused)"],
                    scale=alt.Scale(
                        domain=["Control regions", "Test regions (paused)"],
                        range=[REPORTED, INCREMENTAL],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("group:N", title=None),
                    alt.Tooltip("week_start:T", title="Week beginning"),
                    alt.Tooltip("revenue:Q", title="Revenue", format="$,.0f"),
                    alt.Tooltip("indexed:Q", title="Indexed", format=".1f"),
                ],
            )
        )
        covers_end = series["week_start"].max() + pd.Timedelta(days=6)
        bands = lag_layers(lag, series["week_start"].min(), covers_end)
        show(
            alt.layer(paused, paused_label, *bands, lines),
            "Both groups indexed to their own average over the four weeks before the test, so "
            "the gap that opens inside the pale window is the pause and not a size difference."
            + (f" {PROVISIONAL} This chart is Shopify revenue, which is complete across them."
               if bands else ""),
        )

        with st.expander("How this was measured"):
            md(
                f"- **{count_display(row.n_pairs)} matched pairs.** Each paused region is paired "
                f"with one that kept running. The control predicts the test region's revenue by "
                f"scaling its own in-window figure by the pair's ratio over an equally long "
                f"pre-period — a difference-in-differences.\n"
                f"- **The interval is the pairs, resampled.** The pairs are the independent units, "
                f"so the {level_display(row.confidence)} interval comes from resampling them with "
                f"replacement. Five pairs makes a wide interval; that is an honest reading of a "
                f"five-market test, not a defect.\n"
                f"- **Paused spend is predicted, not observed.** The test regions spent nothing on "
                f"{row.channel} during the window, so the "
                f"{money_display(row.paused_spend_estimate)} they *would* have spent is predicted "
                f"the same way the revenue is.\n"
                f"- **Counterfactual revenue** {money_display(row.counterfactual_revenue)} against "
                f"**actual** {money_display(row.actual_revenue)}.\n"
                f"- **Lift interval** {pct_display(row.lift_pct_ci_low)} to "
                f"{pct_display(row.lift_pct_ci_high)}."
            )

    cap(
        "A result exists only once its window has closed; it is measured over the whole window, "
        f"never over a single week inside it. {ILLUSTRATIVE}"
    )


# --------------------------------------------------------------------------
# Page 4: Agent read-outs
# --------------------------------------------------------------------------


def flatten_facts(node, prefix: str = "") -> list[tuple[str, str, object]]:
    """Every ``{value, display}`` pair in the facts, as ``path, display, value``."""
    rows: list[tuple[str, str, object]] = []
    if isinstance(node, dict):
        if set(node) == {"value", "display"}:
            return [(prefix, node["display"], node["value"])]
        for key, value in node.items():
            rows += flatten_facts(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            rows += flatten_facts(value, f"{prefix}[{index}]")
    return rows


def provenance(record: dict) -> str:
    """Who wrote this read-out: a named model, or nobody.

    ``dry run`` means the text was assembled in Python from the same facts and
    no model was called, which is what the tests use. Saying "dry run" is
    shorter and truer than printing a mode beside an empty model id.
    """
    model = record.get("model")
    if record.get("mode") == "model" and model:
        return f"model: {model}"
    return "dry run"


def render_readout(record: dict, expanded: bool = False) -> None:
    """One read-out, with the facts it was allowed to use underneath."""
    md(
        f"**Week {record['week']}** &nbsp;·&nbsp; {record['week_start']} to {record['week_end']} "
        f"&nbsp;·&nbsp; {count_display(record.get('word_count', 0))} words",
    )
    md(f'<div class="gm-readout">{record["text"]}</div>', unsafe_allow_html=True)
    cap(f"{record['disclaimer']}  \n`{record['path']}` · {provenance(record)}")

    with st.expander(f"The facts week {record['week']} was written from", expanded=expanded):
        md(
            "Python computed and formatted every one of these. The model was shown the "
            "**display** column and nothing else — no raw number to round, rescale or mistype — "
            "and every figure in the text above was matched back to one of these strings before "
            "the file was saved."
        )
        rows = flatten_facts(record["facts"])
        if rows:
            st.dataframe(
                pd.DataFrame(rows, columns=["Fact", "Display", "Value"]),
                hide_index=True,
                width="stretch",
                height=min(420, 38 * len(rows) + 40),
            )
        notes = record["facts"].get("notes") or []
        for note in notes:
            cap(f"· {note}")
        st.json(record["facts"], expanded=False)


def page_readouts(analysis: dict, lag: dict | None) -> None:
    st.title("Agent read-outs", anchor=False)
    md(
        "One read-out a week. Generated **offline**, checked by a human against the facts beside "
        "it, and committed as a file. This page reads those files. It never calls a model, and "
        "the app has no key to call one with."
    )

    readouts = load_readouts()
    if not readouts:
        st.warning("No read-outs are committed yet. Generate them with `python -m gmarge.analyst`.")
        return

    weeks = [r["week"] for r in readouts]
    cap(
        f"{count_display(len(readouts))} saved, weeks {min(weeks)} to {max(weeks)}, newest first. "
        "A missing week is a week whose read-out failed its number check and was not saved — "
        "it never means the old one still applies."
    )

    for index, record in enumerate(readouts):
        render_readout(record, expanded=False)
        if index < len(readouts) - 1:
            st.divider()


# --------------------------------------------------------------------------
# Page 5: Data health
# --------------------------------------------------------------------------


def finding_span(finding: quality.Finding) -> str:
    """``2025-02-26 to 2025-02-27``, or a single date when it is one day."""
    if finding.start_date == finding.end_date:
        return finding.start_date
    return f"{finding.start_date} to {finding.end_date}"


def short_span(finding: quality.Finding) -> str:
    """``26-27 Feb`` -- the same range, short enough to sit beside a chart row."""
    start = pd.Timestamp(finding.start_date)
    end = pd.Timestamp(finding.end_date)
    if start == end:
        return start.strftime("%d %b")
    if (start.month, start.year) == (end.month, end.year):
        return f"{start.strftime('%d')}-{end.strftime('%d %b')}"
    return f"{start.strftime('%d %b')} - {end.strftime('%d %b')}"


def format_metric(metric_key: str, value: float) -> str:
    """An anomaly's metric, in the analyst's formatting.

    Frequency is impressions per person — a count per head, not a multiple —
    so it is written 2.88 and not 2.88x. That distinction is the analyst's;
    this reads it from there rather than making it again.
    """
    metric = anomaly_scan.METRICS_BY_KEY.get(metric_key)
    kind = METRIC_KIND.get(metric_key) or UNIT_KIND.get(metric.unit if metric else "", RATIO)
    return DISPLAY[kind](value)


def page_health(analysis: dict, lag: dict | None) -> None:
    findings = analysis["findings"]
    flags = analysis["anomalies"]

    st.title("Data health", anchor=False)
    md(
        "What is wrong with the tables, and which channel-weeks broke with their own history. "
        "Both are recovered from the data the way an analyst would recover them — nothing here "
        "is read out of an answer key."
    )

    left, right = st.columns(2)
    left.metric("Data-quality findings", count_display(len(findings)))
    right.metric("Anomaly flags", count_display(len(flags)))

    st.subheader("Data-quality findings", anchor=False)
    if not findings:
        st.success("No findings.")
    else:
        rows = pd.DataFrame(
            [
                {
                    "start": pd.Timestamp(f.start_date),
                    "end": pd.Timestamp(f.end_date) + pd.Timedelta(days=1),
                    "midpoint": pd.Timestamp(f.start_date)
                    + (pd.Timestamp(f.end_date) - pd.Timestamp(f.start_date)) / 2,
                    "label": f"{SOURCE_LABELS.get(f.source, f.source)} — {f.check.replace('_', ' ')}",
                    "severity": f.severity,
                    "span": short_span(f),
                    "when": finding_span(f),
                    "description": f.description,
                }
                for f in findings
            ]
        )
        severity = alt.Color(
            "severity:N",
            title=None,
            sort=["high", "medium", "low"],
            # Under the plot, not over it: the row names occupy the top of
            # each band, and the plot is now full width so the three items fit.
            legend=alt.Legend(orient="bottom", direction="horizontal"),
            scale=alt.Scale(
                domain=["high", "medium", "low"],
                range=[SEVERITY_COLOURS["high"], SEVERITY_COLOURS["medium"], SEVERITY_COLOURS["low"]],
            ),
        )
        tips = [
            alt.Tooltip("label:N", title=None),
            alt.Tooltip("severity:N", title="Severity"),
            alt.Tooltip("when:N", title="When"),
            alt.Tooltip("description:N", title="What"),
        ]
        # The row name goes inside the plot rather than on the y axis. On the
        # axis, "ad spend - pixel double counting" took two thirds of a phone's
        # width, squeezing the plot until the legend and the last axis label
        # ran off the edge. Inline, every row gets the full width.
        position = alt.Y("label:N", title=None, sort=None, axis=None)
        axis = alt.X("start:T", title="Date", axis=alt.Axis(format="%d %b"))
        ROW = 8  # pixels below the row's centre line, where the marks sit

        names = (
            alt.Chart(rows)
            .mark_text(align="left", dy=-9, fontSize=11, color=TEXT)
            .encode(x=alt.value(0), y=position, text="label:N", tooltip=tips)
        )

        # The true span, which for most findings is two or three days out of a
        # hundred and eighty-two and so is a few pixels wide.
        spans = (
            alt.Chart(rows)
            .mark_bar(height=10, cornerRadius=2, yOffset=ROW)
            .encode(x=axis, x2="end:T", y=position, color=severity, tooltip=tips)
        )
        # A fixed-size marker on top, so a two-day finding is findable without
        # drawing it as if it lasted a fortnight. The bar still carries the
        # length; the marker only says "here".
        markers = (
            alt.Chart(rows)
            .mark_point(shape="diamond", size=90, filled=True, opacity=1.0, stroke=None, yOffset=ROW)
            .encode(x=alt.X("midpoint:T", title="Date", axis=alt.Axis(format="%d %b")),
                    y=position, color=severity, tooltip=tips)
        )
        # And the dates in words beside the marker, because a position on a
        # six-month axis is not a date anyone can read off. A finding near the
        # end of the window takes its label on the inside, so the text is not
        # cut off by the edge of the plot -- which is where the reporting lag
        # always sits.
        def dated(frame: pd.DataFrame, align: str, dx: int) -> alt.Chart:
            return (
                alt.Chart(frame)
                .mark_text(align=align, dx=dx, dy=ROW, fontSize=10, color=MUTED)
                .encode(
                    x=alt.X("midpoint:T", title="Date", axis=alt.Axis(format="%d %b")),
                    y=position,
                    text="span:N",
                    tooltip=tips,
                )
            )

        window = rows["end"].max() - rows["start"].min()
        near_end = rows["midpoint"] > rows["start"].min() + window * 0.75
        labels = [
            dated(frame, align, dx)
            for frame, align, dx in ((rows[~near_end], "left", 11), (rows[near_end], "right", -11))
            if not frame.empty
        ]
        show(
            alt.layer(
                *lag_layers(lag, rows["start"].min(), rows["end"].max()),
                names,
                spans,
                markers,
                *labels,
            ),
            f"Where each finding falls in the six-month window. {PROVISIONAL}",
            # A name and a bar per finding, plus the axis and the legend.
            height=max(150, 54 * len(findings) + 20),
        )

        for finding in findings:
            card(
                f"{tag(finding.severity)} &nbsp; {SOURCE_LABELS.get(finding.source, finding.source)} "
                f"— {finding.check.replace('_', ' ')}",
                f"<p style='color:{MUTED}'>{finding_span(finding)} · "
                f"{count_display(len(finding.dates))} days</p>"
                f"<p>{finding.description}</p>",
            )

    st.subheader("Anomaly flags", anchor=False)
    cap(
        "Only rates are scored — reported ROAS, CPC, CPM, CTR and frequency — against each "
        "channel's own trailing eight weeks. Spend and revenue levels move for planned reasons; "
        "a rate is scale-free, so it survives a channel going dark in half the regions for a "
        "holdout, and it survives the last two days still filling in."
    )
    if not flags:
        st.success("Nothing flagged.")
    else:
        for flag in flags:
            arrow = "rose" if flag.direction == "up" else "fell"
            card(
                f"{tag(flag.severity)} &nbsp; {flag.channel} — {flag.metric_label}, week {flag.week}",
                f"<p style='color:{MUTED}'>{flag.week_start} to {flag.week_end}</p>"
                f"<p>{_sentence_case(flag.metric_label)} {arrow} "
                f"<b>{pct_display(flag.pct_change)}</b>, to "
                f"<b>{format_metric(flag.metric, flag.value)}</b> from a trailing median of "
                f"<b>{format_metric(flag.metric, flag.baseline)}</b>. "
                f"{flag.ad_set} (campaign {flag.campaign}) accounts for "
                f"{pct_display(flag.share_of_move)} of the move.</p>",
            )

    if lag:
        note(lag_note(lag))


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

RENDER = {
    "Overview": page_overview,
    "Channels": page_channels,
    "Holdout tests": page_holdouts,
    "Agent read-outs": page_readouts,
    "Data health": page_health,
}


def main() -> None:
    st.set_page_config(
        page_title="G-Marge — incrementality demo",
        page_icon="📐",
        layout="centered",
        # "auto", not "expanded": on a phone an expanded sidebar opens over the
        # page, and the first thing a reader sees is the nav rather than the work.
        initial_sidebar_state="auto",
    )
    css()

    analysis = load_analysis()
    lag = lag_window(analysis["findings"])

    with st.sidebar:
        md(f"### <span style='color:{INCREMENTAL}'>G-Marge</span>", unsafe_allow_html=True)
        cap(analysis["metrics"]["brand"])
        page = st.radio("Page", PAGES, label_visibility="collapsed")
        st.divider()
        cap(analysis["metrics"]["disclaimer"])
        md(f"[See what G-Marge would show you]({CONTACT_URL})")

    banner()
    RENDER[page or PAGES[0]](analysis, lag)


if __name__ == "__main__":
    main()
