"""The Streamlit app: every page renders, and it renders honestly.

Driven with Streamlit's ``AppTest``, which runs ``app.py`` in-process. No
server is started and no browser is involved, so these run in CI like any
other test.

Three things are checked, in order of how much they matter:

1. **Every page renders with no exception.** A demo that raises on page four
   is worse than no demo.
2. **No AI call can happen.** ``anthropic`` must not be importable *into the
   app's import graph* -- not lazily, not by accident (CLAUDE.md, rule 2).
3. **The honesty rules hold.** The lagging days are shaded and labelled
   provisional on every time-series chart, and a ratio computed across them
   never reaches a chart.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"

sys.path.insert(0, str(ROOT))

import app as app_module  # noqa: E402

TIMEOUT = 120


def charts(at: AppTest) -> list:
    """Every Altair chart on the page. Each carries its Vega-Lite spec as JSON text."""
    return list(at.get("vega_lite_chart"))


def datasets(chart) -> dict[str, pd.DataFrame]:
    """The data Streamlit actually sent with a chart, by the name its spec uses.

    Altair hands the frames over as named Arrow tables rather than inlining
    them in the spec, so checking what a chart plots means decoding these --
    not reading the source that built it.
    """
    return {
        d.name: pa.ipc.open_stream(io.BytesIO(d.data.data)).read_all().to_pandas()
        for d in chart.proto.datasets
    }


def layer_data(chart, predicate) -> pd.DataFrame | None:
    """The frame behind the first layer whose spec satisfies ``predicate``."""
    spec = json.loads(chart.spec)
    frames = datasets(chart)
    for layer in spec.get("layer", [spec]):
        if predicate(layer):
            return frames.get(layer.get("data", {}).get("name"))
    return None


def band_data(chart) -> pd.DataFrame | None:
    """The provisional band's own frame, if the chart has one."""
    return layer_data(
        chart,
        lambda layer: layer.get("mark", {}).get("type") == "rect"
        and layer.get("mark", {}).get("color") == app_module.LAG,
    )


def run(page: str | None = None) -> AppTest:
    """Run the app, optionally after switching to ``page``."""
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    if page is not None:
        at.sidebar.radio[0].set_value(page).run()
    return at


# --------------------------------------------------------------------------
# Every page renders
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pages() -> dict[str, AppTest]:
    """Each page, run once. Cheaper than a run per assertion."""
    return {page: run(page) for page in app_module.PAGES}


def test_every_page_renders_without_exception(pages):
    for name, at in pages.items():
        assert not at.exception, f"{name} raised: {at.exception}"


def test_every_page_renders_something(pages):
    for name, at in pages.items():
        assert at.title, f"{name} has no title"
        assert at.markdown, f"{name} rendered no text"


def test_every_page_shows_the_banner(pages):
    for name, at in pages.items():
        body = " ".join(m.value for m in at.markdown)
        assert app_module.BANNER in body, f"{name} is missing the banner"
        assert app_module.CONTACT_URL in body, f"{name} is missing the contact link"


def test_pages_that_chart_produce_charts(pages):
    for name in ("Overview", "Channels", "Holdout tests", "Data health"):
        drawn = charts(pages[name])
        assert drawn, f"{name} rendered no chart"


def test_every_chart_is_marked_illustrative(pages):
    """A chart a reader could screenshot must carry the disclaimer with it."""
    for name, at in pages.items():
        drawn = charts(at)
        if not drawn:
            continue
        captions = [c.value for c in at.caption]
        marked = [c for c in captions if app_module.ILLUSTRATIVE in c]
        assert len(marked) >= len(drawn), (
            f"{name}: {len(drawn)} charts but only {len(marked)} illustrative captions"
        )


def test_the_default_page_is_the_overview():
    at = run()
    assert at.sidebar.radio[0].value == "Overview"
    assert "Where the money actually went" in at.title[0].value


# --------------------------------------------------------------------------
# No AI at runtime
# --------------------------------------------------------------------------


def test_the_app_never_imports_anthropic(pages):
    """Rule 2. Not at import, not lazily, not on any page."""
    assert "anthropic" not in sys.modules


def test_the_app_imports_no_sdk_and_reads_no_key():
    """Read the source, not the running process: a lazy import would still be there.

    Checked as syntax rather than as text, so that the docstring explaining
    the rule does not trip the test that enforces it.
    """
    import ast

    tree = ast.parse(APP.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "anthropic" not in imported
    assert "dotenv" not in imported
    assert not {"os", "requests", "httpx", "urllib"} & imported, (
        "the app reads files and nothing else -- no environment, no network"
    )
    assert "API_KEY" not in APP.read_text()


def test_read_outs_come_from_files_only():
    """What the page shows is what is on disk, unchanged."""
    records = app_module.load_readouts(str(ROOT / "readouts"))
    assert records, "no read-outs are committed"
    assert [r["week"] for r in records] == sorted((r["week"] for r in records), reverse=True)
    for record in records:
        saved = json.loads(Path(record["path"]).read_text())
        assert record["text"] == saved["text"]


# --------------------------------------------------------------------------
# Honesty: the reporting lag
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analysis() -> dict:
    return app_module.load_analysis(str(ROOT / "data"))


@pytest.fixture(scope="module")
def lag(analysis) -> dict:
    found = app_module.lag_window(analysis["findings"])
    assert found is not None, "the sample data is supposed to have a reporting lag"
    return found


def test_the_lag_is_read_from_the_quality_checks(analysis, lag):
    """Not hard-coded. If the data changes, the shading moves with it."""
    lagging = [f for f in analysis["findings"] if f.check == "reporting_lag"]
    assert lag["start"].date().isoformat() == min(f.start_date for f in lagging)
    assert lag["end"].date().isoformat() == max(f.end_date for f in lagging)
    assert "ad_spend" in lag["sources"]


def test_the_withheld_totals_are_the_analysts_rule(analysis, lag):
    """The app and a read-out must withhold the same figures, for the same reason."""
    from gmarge.analyst import TOTAL_SOURCES

    expected = {
        key for key, tables in TOTAL_SOURCES.items() if set(lag["sources"]).intersection(tables)
    }
    assert set(lag["withheld"]) == expected
    # Shopify is complete, so store revenue survives and the ad-side figures do not.
    assert "shopify_revenue" not in lag["withheld"]
    assert "over_claim_ratio" in lag["withheld"]
    assert "blended_reported_roas" in lag["withheld"]


def test_every_time_series_that_reaches_the_lag_shades_exactly_those_days(pages, lag):
    """The band on the rendered chart, decoded, covers the lagging days and no more.

    Read out of the chart Streamlit actually sent rather than off the source,
    because the interesting failure is a band that renders over the wrong
    days -- which reading the source would not catch.
    """
    expected = 0
    for name, at in pages.items():
        for chart in charts(at):
            band = band_data(chart)
            if band is None:
                continue
            expected += 1
            assert len(band) == 1, f"{name}: more than one provisional band"
            assert pd.Timestamp(band["start"].iloc[0]) == lag["start"], name
            # The band is drawn to the start of the next day so the last
            # lagging day is covered rather than left as a hairline.
            assert pd.Timestamp(band["end"].iloc[0]) == lag["end"] + pd.Timedelta(days=1), name
            assert "provisional" in chart.spec, f"{name}: the band carries no label"
    assert expected, "no chart shaded the lag at all"


def test_the_charts_that_reach_the_lag_are_the_ones_that_shade_it(pages, lag):
    """Both overview series reach it; the channel bars are not a time series at all."""
    banded = {name: sum(band_data(c) is not None for c in charts(at)) for name, at in pages.items()}

    assert banded["Overview"] == 2, "store revenue and the over-claim ratio both run to the end"
    assert banded["Data health"] == 1, "the findings timeline runs to the end"
    assert banded["Holdout tests"] >= 1, "the last test's window runs into the lagging days"
    assert banded["Channels"] == 0, "the channel comparison has no time axis to shade"


def test_the_rendered_ratio_chart_has_no_lagging_week_in_it(pages, lag, analysis):
    """Decoded from the chart on the page, not from the helper that built it."""
    weekly = analysis["metrics"]["weekly_reconciliation"]
    last_week = int(weekly["week"].max())

    ratio = None
    for chart in charts(pages["Overview"]):
        frame = layer_data(
            chart, lambda layer: "over_claim_ratio" in json.dumps(layer.get("encoding", {}))
        )
        if frame is not None:
            ratio = frame
    assert ratio is not None, "the over-claim ratio chart is missing"

    assert last_week not in set(ratio["week"]), "the lagging week reached the chart"
    assert pd.Timestamp(ratio["week_end"].max()) < lag["start"]


def test_the_lagging_week_is_not_plotted_as_a_ratio(analysis, lag):
    """Rule: never show a lagging ratio as a real change.

    Week 26's attributed revenue is short by two days while its store revenue
    is complete, so its over-claim ratio falls for a reason that is not a
    result. It is withheld, not qualified.
    """
    weekly = analysis["metrics"]["weekly_reconciliation"]
    shown = app_module.complete_weeks(weekly, lag)

    assert len(shown) < len(weekly), "the lagging week is still being plotted"
    assert shown["week_end"].max() < lag["start"]
    # And the withheld week really is the one that looks like a fall.
    dropped = weekly[~weekly["week"].isin(shown["week"])]
    assert (dropped["over_claim_ratio"] < shown["over_claim_ratio"].iloc[-1]).all()


def test_the_lag_note_names_the_days_the_tables_and_the_figures(lag):
    note = app_module.lag_note(lag)
    assert str(lag["start"].date()) in note and str(lag["end"].date()) in note
    assert "ad spend" in note and "GA4 sessions" in note
    assert "provisional" in note
    assert "Shopify orders are complete" in note


# --------------------------------------------------------------------------
# Honesty: reported is not incremental
# --------------------------------------------------------------------------


def test_reported_and_incremental_never_share_a_colour(pages):
    """Grey is a claim, light blue is a measurement, on every chart that pairs them."""
    assert app_module.REPORTED != app_module.INCREMENTAL
    for name in ("Channels", "Holdout tests"):
        specs = [c.spec for c in charts(pages[name])]
        paired = [s for s in specs if app_module.INCREMENTAL in s and app_module.REPORTED in s]
        assert paired, f"{name} should pair the two colours on at least one chart"


def test_the_channel_page_labels_the_gap(pages, analysis):
    """Every channel's over-claim multiple is written on the page, not left to the eye."""
    body = " ".join(m.value for m in pages["Channels"].markdown)
    from gmarge.analyst import ratio_display

    for row in analysis["metrics"]["holdouts"].itertuples():
        assert row.channel in body
        assert ratio_display(row.over_claim_multiple) in body


def test_holdout_results_are_written_with_their_interval(pages, analysis):
    from gmarge.analyst import ratio_display

    body = " ".join(m.value for m in pages["Holdout tests"].markdown)
    for row in analysis["metrics"]["holdouts"].itertuples():
        assert ratio_display(row.incremental_roas) in body
        assert ratio_display(row.incremental_roas_ci_low) in body
        assert ratio_display(row.incremental_roas_ci_high) in body
        assert "90%" in body


# --------------------------------------------------------------------------
# Honesty: one formatter, shared with the read-outs
# --------------------------------------------------------------------------


def test_the_app_uses_the_analysts_formatting():
    """``$14.9k``, ``1.47x``, ``39.2%``, and frequency as a plain number."""
    from gmarge import analyst

    assert app_module.money_display is analyst.money_display
    assert app_module.ratio_display is analyst.ratio_display
    assert app_module.pct_display is analyst.pct_display

    assert app_module.money_display(14_912.0) == "$14.9k"
    assert app_module.ratio_display(1.4712) == "1.47x"
    assert app_module.pct_display(0.3921) == "39.2%"


def test_frequency_is_a_plain_number_not_a_multiple():
    """Impressions per person is a count per head. 2.88, never 2.88x."""
    assert app_module.format_metric("frequency", 2.8765) == "2.88"
    assert app_module.format_metric("reported_roas", 2.8765) == "2.88x"
    assert app_module.format_metric("cpc", 1.234) == "$1"
    assert app_module.format_metric("ctr", 0.0123) == "1.2%"


def test_a_figure_on_a_page_matches_the_same_figure_in_a_read_out():
    """The point of sharing a formatter: the app and a read-out cannot disagree."""
    records = app_module.load_readouts(str(ROOT / "readouts"))
    by_week = {r["week"]: r for r in records}
    weekly = app_module.load_analysis(str(ROOT / "data"))["metrics"]["weekly_reconciliation"]

    checked = 0
    for row in weekly.itertuples():
        record = by_week.get(int(row.week))
        if record is None:
            continue
        saved = record["facts"].get("totals", {}).get("shopify_revenue")
        if saved is None:
            continue
        assert app_module.money_display(row.shopify_revenue) == saved["display"]
        checked += 1
    assert checked, "no week had a saved store-revenue total to compare against"


# --------------------------------------------------------------------------
# The holdout chart series
# --------------------------------------------------------------------------


def test_holdout_series_is_indexed_to_the_pre_period(analysis):
    """Both groups start at 100 on average, so the divergence is the pause."""
    import pandas as pd

    plan = {h["channel"]: h for h in analysis["holdout_plan"]}
    holdout = plan["Google shopping"]
    series = app_module.holdout_series(analysis["orders"], holdout)

    start = pd.Timestamp(holdout["start_date"])
    before = series[series["week_start"] < start]
    assert set(series["group"]) == {"Test regions (paused)", "Control regions"}
    for _, group in before.groupby("group"):
        assert group["indexed"].mean() == pytest.approx(100.0)


def test_the_paused_regions_fall_behind_their_controls_during_the_test(analysis):
    """The chart should show what the estimate says: a gap that opens in the window."""
    plan = {h["channel"]: h for h in analysis["holdout_plan"]}
    series = app_module.holdout_series(analysis["orders"], plan["Google shopping"])

    inside = series[series["in_test"]].groupby("group")["indexed"].mean()
    assert inside["Test regions (paused)"] < inside["Control regions"]


def test_money_figures_are_never_rendered_as_equations(pages, analysis):
    """Streamlit reads ``$...$`` as LaTeX.

    Two money figures in one sentence would pair up and swallow the words
    between them, so every dollar sign reaching markdown is escaped.
    """
    import re

    from gmarge.analyst import money_display

    for name, at in pages.items():
        body = " ".join(m.value for m in at.markdown)
        assert re.search(r"(?<!\\)\$", body) is None, f"{name} has an unescaped dollar sign"

    body = " ".join(m.value for m in pages["Holdout tests"].markdown)
    for row in analysis["metrics"]["holdouts"].itertuples():
        # The figure survives the escaping -- it is the sign that is escaped.
        assert "\\" + money_display(row.incremental_revenue) in body
        assert "\\" + money_display(row.paused_spend_estimate) in body


def test_the_escape_touches_nothing_but_the_dollar_sign():
    assert app_module.no_math("Store revenue $14.9k, up 1.9%") == r"Store revenue \$14.9k, up 1.9%"
    assert app_module.no_math("1.47x and 39.2%") == "1.47x and 39.2%"


def test_html_blocks_escape_as_an_entity_not_a_backslash():
    """A markdown backslash survives into a raw HTML block and shows up on the page."""
    assert app_module.no_math_html("$14.9k") == "&#36;14.9k"
    assert "\\" not in app_module.no_math_html("$14.9k and $1.2m")


def test_the_cards_and_read_outs_carry_no_stray_backslash(pages):
    """The HTML-rendered blocks: cards, the banner and the read-out panels."""
    for name, at in pages.items():
        for block in at.markdown:
            if "<div" not in block.value:
                continue
            assert "\\$" not in block.value, f"{name}: a card shows a literal backslash"
            assert "$" not in block.value, f"{name}: a card has a bare dollar sign"


# --------------------------------------------------------------------------
# Public-app chrome
# --------------------------------------------------------------------------

CONFIG = ROOT / ".streamlit" / "config.toml"


def test_streamlits_own_chrome_is_hidden():
    """No Deploy button and no hamburger on a public demo.

    "minimal" is the one mode that also hides the menu once nothing is left in
    it; "viewer" would keep the hamburger for the viewer options.
    """
    import tomllib

    config = tomllib.loads(CONFIG.read_text())
    assert config["client"]["toolbarMode"] == "minimal"


def test_every_heading_suppresses_its_anchor_link():
    """Checked as syntax, so a heading added later cannot quietly skip it."""
    import ast

    tree = ast.parse(APP.read_text())
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"title", "header", "subheader"}:
            continue
        anchor = [k for k in node.keywords if k.arg == "anchor"]
        if not anchor or not (isinstance(anchor[0].value, ast.Constant) and anchor[0].value.value is False):
            missing.append(f"line {node.lineno}: {ast.unparse(node)[:60]}")
    assert not missing, "headings without anchor=False: " + "; ".join(missing)


def test_charts_carry_no_element_toolbar_but_dataframes_still_do():
    """The rule is scoped to charts.

    A blanket hide would also take the dataframe search and download buttons,
    which are worth keeping -- the facts table under each read-out is the
    thing a reader most wants to pull out.
    """
    source = APP.read_text()
    # By rule, not by line: the selector wraps onto its own line.
    rules = [block for block in source.split("}") if "stElementToolbar" in block and "display: none" in block]
    assert rules, "no rule hides the chart element toolbar"
    for rule in rules:
        assert "stVegaLiteChart" in rule, "the rule is not scoped to charts"


# --------------------------------------------------------------------------
# Read-out provenance
# --------------------------------------------------------------------------


def test_provenance_names_the_model_or_says_dry_run():
    assert app_module.provenance({"mode": "model", "model": "claude-haiku-4-5"}) == (
        "model: claude-haiku-4-5"
    )
    assert app_module.provenance({"mode": "dry-run", "model": None}) == "dry run"
    # A mode of "model" with nothing to name is not a model read-out.
    assert app_module.provenance({"mode": "model", "model": None}) == "dry run"
    assert app_module.provenance({}) == "dry run"


def test_the_read_out_page_shows_each_files_provenance(pages):
    captions = " ".join(c.value for c in pages["Agent read-outs"].caption)
    records = app_module.load_readouts(str(ROOT / "readouts"))
    for record in records:
        assert app_module.provenance(record) in captions
    assert "mode `" not in captions, "the old mode/model pair is still being printed"


# --------------------------------------------------------------------------
# The findings timeline
# --------------------------------------------------------------------------


def severity_chart(at) -> tuple:
    """The findings timeline: the chart whose layers encode severity."""
    for chart in charts(at):
        if '"severity"' in chart.spec:
            return chart
    raise AssertionError("no chart encodes severity")


def test_the_findings_timeline_has_a_row_per_finding(pages, analysis):
    """One row, at the finding's own dates, in its own severity colour."""
    chart = severity_chart(pages["Data health"])
    frames = [f for f in datasets(chart).values() if "severity" in f.columns]
    assert frames, "the timeline sent no severity data"
    rows = frames[0]

    findings = analysis["findings"]
    assert len(rows) == len(findings)
    assert list(rows["severity"]) == [f.severity for f in findings]
    for row, finding in zip(rows.itertuples(), findings):
        assert pd.Timestamp(row.start) == pd.Timestamp(finding.start_date)
        # Drawn to the start of the next day, so the last day is covered.
        assert pd.Timestamp(row.end) == pd.Timestamp(finding.end_date) + pd.Timedelta(days=1)


def test_the_findings_timeline_keeps_its_severity_legend(pages):
    spec = severity_chart(pages["Data health"]).spec
    for level, colour in app_module.SEVERITY_COLOURS.items():
        assert colour in spec, f"{level} is not in the colour scale"
    assert '"legend"' in spec and '"orient": "bottom"' in spec


def test_the_findings_timeline_names_its_rows_inside_the_plot(pages):
    """The names are a text mark, not axis labels.

    On the y axis they took two thirds of a phone's width and pushed the
    legend and the last axis label off the edge of the chart.
    """
    import json

    spec = json.loads(severity_chart(pages["Data health"]).spec)
    layers = spec["layer"]
    named = [l for l in layers if l.get("mark", {}).get("type") == "text"
             and l.get("encoding", {}).get("text", {}).get("field") == "label"]
    assert named, "the row names are not drawn inside the plot"

    # "axis": null suppresses it; the key being absent means it is drawn. A
    # .get() cannot tell those apart, so check the key is there and is null.
    # Only the layers positioned by the row field. The provisional label is
    # pinned to a pixel offset instead, and has no axis to draw.
    with_y = [l for l in layers if "field" in l.get("encoding", {}).get("y", {})]
    assert with_y, "no layer positions anything by row"
    for layer in with_y:
        y = layer["encoding"]["y"]
        assert "axis" in y and y["axis"] is None, "a layer still draws the y axis"


def test_short_span_reads_as_dates(analysis):
    findings = {f.check: f for f in analysis["findings"]}
    lag = findings["reporting_lag"]
    assert app_module.short_span(lag) == "05-06 Jul"
    assert app_module.finding_span(lag) == f"{lag.start_date} to {lag.end_date}"
