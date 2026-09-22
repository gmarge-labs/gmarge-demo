"""The facts must carry the read-out, and the read-out must not outrun them.

Every test here runs with ``--dry-run``: no API key, no network, no model. The
template read-out is assembled from the same facts the model is given, in the
same order, and held to the same number check, so these tests prove the facts
are sufficient and the check is honest without anything leaving the machine.

Three properties are what this file is really about:

* every number is formatted in Python, and the model is shown nothing else;
* "outside the normal range" comes from the anomaly scan and nowhere else;
* a week whose tables are still filling in gets no verdict about its totals.
"""

from __future__ import annotations

import json
import sys

import pytest

from gmarge import analyst as an
from gmarge import llm

# Anything here in the facts would be this module holding a second opinion on
# what "normal" means. There is only one, and it is the anomaly scan's.
BANNED_KEYS = ("normal_range", "robust_score", "score_threshold", "enough_history", "band")

RISE_AND_FALL = ("rose", "fell", "rise", "fall", "drop", "increase", "decrease", "recovered")


@pytest.fixture(scope="session")
def analysis(dataset):
    return an.analyse(dataset.path)


@pytest.fixture(scope="session")
def last_week(analysis):
    return analysis.weeks[-1]


def facts_for(analysis, week):
    return an.build_facts(analysis, week)


def _keys(node, found=None):
    found = set() if found is None else found
    if isinstance(node, dict):
        found.update(node)
        for value in node.values():
            _keys(value, found)
    elif isinstance(node, list):
        for value in node:
            _keys(value, found)
    return found


def _leaves(node, found=None):
    """Every non-string scalar in a structure, for the no-raw-numbers check."""
    found = [] if found is None else found
    if isinstance(node, dict):
        for value in node.values():
            _leaves(value, found)
    elif isinstance(node, list):
        for value in node:
            _leaves(value, found)
    elif not isinstance(node, (str, bool)) and node is not None:
        found.append(node)
    return found


def _flagged_week(analysis):
    return next(a.week for a in analysis.anomalies)


# --------------------------------------------------------------------------
# 1. The facts carry the read-out
# --------------------------------------------------------------------------


def test_template_readout_uses_only_facts(analysis):
    """Every figure in every template read-out is a display string in its facts."""
    for week in analysis.weeks:
        facts = facts_for(analysis, week)
        text = an.template_readout(facts)
        assert an.check_numbers(text, facts) == [], f"week {week}: {text}"


def test_template_readout_is_within_the_word_limit(analysis):
    for week in analysis.weeks:
        assert an.word_count(an.template_readout(facts_for(analysis, week))) <= an.WORD_LIMIT


def test_facts_are_json_serialisable(analysis, last_week):
    facts = facts_for(analysis, last_week)
    assert an.week_number(json.loads(json.dumps(facts, default=str))) == last_week


def test_unknown_week_is_refused(analysis):
    with pytest.raises(ValueError):
        an.build_facts(analysis, max(analysis.weeks) + 1)


# --------------------------------------------------------------------------
# 2. Python formats every number; the model sees displays and nothing else
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(14_900, "$14.9k"), (531_637.44, "$531.6k"), (1_234_567.0, "$1.2m"), (532.49, "$532"), (-19_328.74, "$19.3k")],
)
def test_money_is_written_one_way(value, expected):
    assert an.money_display(value) == expected


@pytest.mark.parametrize("value,expected", [(1.3704, "1.37x"), (5.2496, "5.25x"), (-1.5, "1.50x")])
def test_ratios_are_written_one_way(value, expected):
    assert an.ratio_display(value) == expected


@pytest.mark.parametrize("value,expected", [(0.1216, "12.2%"), (-0.1216, "12.2%"), (0.392, "39.2%")])
def test_percentages_are_written_to_one_decimal_place(value, expected):
    assert an.pct_display(value) == expected


def test_a_fact_carries_both_the_value_and_the_display():
    entry = an.fact(531_637.44, an.MONEY)
    assert entry == {"value": 531_637.44, "display": "$531.6k"}
    assert an.fact(None, an.MONEY) is None
    assert an.fact(float("inf"), an.RATIO) is None


def test_every_number_in_the_facts_carries_a_display(analysis):
    for week in analysis.weeks:
        for key, node in _pairs(facts_for(analysis, week)):
            assert isinstance(node["display"], str) and node["display"], key


def _pairs(node, path="", found=None):
    """Every ``{value, display}`` fact in the structure, with its path."""
    found = [] if found is None else found
    if isinstance(node, dict):
        if set(node) == {"value", "display"}:
            found.append((path, node))
        else:
            for key, value in node.items():
                _pairs(value, f"{path}.{key}", found)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _pairs(value, f"{path}[{i}]", found)
    return found


def test_the_prompt_contains_no_raw_numbers(analysis):
    """The model cannot round, rescale or mistype a float it was never given."""
    for week in analysis.weeks:
        facts = facts_for(analysis, week)
        shown = an.prompt_facts(facts)

        assert _leaves(shown) == [], f"week {week}: raw numbers reached the prompt"
        assert _leaves(facts), "the saved facts should still carry values for audit"

        # the prompt shows the display and not the value it was rounded from
        revenue = facts["totals"]["shopify_revenue"]
        prompt = an.build_prompt(facts)
        assert revenue["display"] in prompt
        assert str(revenue["value"]) not in prompt


def test_the_prompt_withholds_text_written_by_another_module(analysis):
    """Descriptions carry figures in another format; the reviewer keeps them."""
    week = _flagged_week(analysis)
    facts = facts_for(analysis, week)

    assert facts["anomalies"]["flags"][0]["description"]
    assert "description" not in _keys(an.prompt_facts(facts))


# --------------------------------------------------------------------------
# 3. One source of truth for "normal"
# --------------------------------------------------------------------------


def test_the_facts_hold_no_band_of_their_own(analysis):
    """No level-based range, no robust score -- the scan is the only verdict."""
    for week in analysis.weeks:
        keys = _keys(facts_for(analysis, week))
        assert not [k for k in keys if any(banned in k for banned in BANNED_KEYS)]


def test_a_flagged_week_carries_its_flag_and_claims_nothing_normal(analysis):
    """Week 20's frequency flag is in the facts, and nothing says the week was fine."""
    week = next(a.week for a in analysis.anomalies if a.metric == "frequency")
    facts = facts_for(analysis, week)
    flags = facts["anomalies"]["flags"]

    assert facts["anomalies"]["n_flags"]["value"] == len(flags) >= 1
    flag = next(f for f in flags if f["metric"] == "frequency")
    assert flag["channel"] and flag["pct_change"]["display"].endswith("%")
    assert flag["ad_set"] and flag["share_of_move"]["display"].endswith("%")

    text = an.template_readout(facts).lower()
    assert "nothing was flagged" not in text
    assert "normal" not in text and "every metric" not in text


def test_a_flagged_week_leads_with_the_flag(analysis):
    week = _flagged_week(analysis)
    facts = facts_for(analysis, week)
    text = an.template_readout(facts)
    flag = facts["anomalies"]["flags"][0]

    assert text.index("flagged") < text.index("claimed")
    assert text.index(flag["channel"]) < text.index("claimed")
    assert text.index(flag["metric_label"]) < text.index("claimed")


def test_an_unflagged_week_says_so_without_calling_the_week_fine(analysis):
    flagged = {a.week for a in analysis.anomalies}
    week = next(w for w in analysis.weeks if w not in flagged)
    facts = facts_for(analysis, week)

    assert facts["anomalies"]["flags"] == []
    assert facts["anomalies"]["n_flags"]["value"] == 0

    text = an.template_readout(facts).lower()
    assert "nothing was flagged" in text
    assert "normal" not in text and "fine" not in text and "healthy" not in text


def test_every_flag_from_the_scan_reaches_its_week(analysis):
    for flag in analysis.anomalies:
        flags = facts_for(analysis, flag.week)["anomalies"]["flags"]
        assert any(f["channel"] == flag.channel and f["metric"] == flag.metric for f in flags)


# --------------------------------------------------------------------------
# 4. Order: anomaly or data issue, then over-claim, then the driver
# --------------------------------------------------------------------------


def test_the_readout_leads_with_what_matters(analysis):
    for week in analysis.weeks:
        facts = facts_for(analysis, week)
        text = an.template_readout(facts)
        if not facts["drivers"]:
            continue

        over_claim = text.index("claimed") if "claimed" in text else text.index("over-claim ratio")
        driver = text.index(facts["drivers"]["platform_attributed_revenue"]["largest"]["channel"], over_claim)
        assert text.index("flagged") < over_claim < driver


def test_a_holdout_covering_the_week_is_in_the_facts(analysis):
    """Only a holdout says what a channel is worth, so the week it covers gets it."""
    weeks_with_holdouts = 0
    for week in analysis.weeks:
        facts = facts_for(analysis, week)
        running = facts["holdouts"]["running"]
        assert facts["holdouts"]["n_running"]["value"] == len(running)

        for holdout in running:
            assert holdout["window_start"] <= facts["week"]["week_end"]
            assert holdout["window_end"] >= facts["week"]["week_start"]
            assert holdout["incremental_roas"]["display"].endswith("x")
            assert holdout["reported_roas_in_window"]["display"].endswith("x")
        if running:
            weeks_with_holdouts += 1
            assert running[0]["channel"] in an.template_readout(facts)
    assert weeks_with_holdouts, "this dataset should have weeks a holdout covers"


def test_a_week_no_holdout_covers_claims_no_incremental_roas(analysis):
    holdouts = analysis.metrics["holdouts"]
    latest = max(h.date().isoformat() for h in holdouts["end_date"])
    week = next(w for w in analysis.weeks if facts_for(analysis, w)["week"]["week_start"] > latest)
    facts = facts_for(analysis, week)

    assert facts["holdouts"]["running"] == []
    assert "incremental ROAS" not in an.template_readout(facts)


# --------------------------------------------------------------------------
# 5. An incomplete week gets no verdict
# --------------------------------------------------------------------------


def _week_of(analysis, date: str) -> int:
    weekly = analysis.metrics["weekly_reconciliation"]
    hit = weekly[
        (weekly["week_start"].astype(str).str[:10] <= date) & (weekly["week_end"].astype(str).str[:10] >= date)
    ]
    return int(hit.iloc[0]["week"])


@pytest.mark.parametrize("check", an.INCOMPLETE_CHECKS)
def test_an_incomplete_week_carries_every_day_count(analysis, check):
    """The counts are in the facts, so nothing has to be worked out."""
    finding = next((f for f in analysis.findings if f.check == check), None)
    if finding is None:
        pytest.skip(f"this dataset has no {check} finding")

    facts = facts_for(analysis, _week_of(analysis, finding.start_date))
    completeness = facts["completeness"]

    assert completeness["complete"] is False
    assert completeness["days_in_week"]["value"] == facts["week"]["days"]["value"]
    assert completeness["days_affected"]["value"] == len(completeness["affected_dates"])
    assert (
        completeness["days_with_complete_data"]["value"] + completeness["days_affected"]["value"]
        == completeness["days_in_week"]["value"]
    )
    assert set(finding.dates) & set(completeness["affected_dates"])
    assert finding.source in completeness["affected_sources"]


def test_a_clean_week_is_complete(analysis):
    dirty = {_week_of(analysis, f.start_date) for f in analysis.findings}
    clean = next(w for w in analysis.weeks if w not in dirty)
    completeness = facts_for(analysis, clean)["completeness"]

    assert completeness["complete"] is True
    assert completeness["days_affected"]["value"] == 0
    assert completeness["totals_still_filling_in"] == []


def test_only_totals_drawn_from_an_incomplete_table_are_held_back(analysis):
    """A GA4 gap does not make store revenue incomplete, and must not say it does."""
    lag = next(f for f in analysis.findings if f.check == "reporting_lag")
    lagged = facts_for(analysis, _week_of(analysis, lag.start_date))["completeness"]

    assert "ad_spend" in lagged["affected_sources"]
    assert set(lagged["totals_still_filling_in"]) == {
        key for key, tables in an.TOTAL_SOURCES.items() if "ad_spend" in tables
    }
    assert "shopify_revenue" not in lagged["totals_still_filling_in"]

    gap = next((f for f in analysis.findings if f.check == "missing_days"), None)
    if gap is not None and gap.source == "ga4_sessions":
        facts = facts_for(analysis, _week_of(analysis, gap.start_date))
        assert facts["completeness"]["complete"] is False
        assert facts["completeness"]["totals_still_filling_in"] == []
        assert "unaffected" in an.template_readout(facts)


def test_an_incomplete_total_is_never_called_a_rise_or_a_fall(analysis):
    lag = next(f for f in analysis.findings if f.check == "reporting_lag")
    facts = facts_for(analysis, _week_of(analysis, lag.start_date))
    text = an.template_readout(facts).lower()

    assert facts["completeness"]["totals_still_filling_in"]
    assert text.count("still filling in") >= 2
    assert [word for word in RISE_AND_FALL if word in text] == []


def test_the_prompt_forbids_reading_an_incomplete_total_as_a_move():
    assert "totals_still_filling_in" in an.SYSTEM
    assert "still filling in" in an.SYSTEM
    assert "anomalies" in an.SYSTEM and "normal range" in an.SYSTEM


def test_findings_carry_the_days_that_fall_inside_the_week(analysis):
    for week in analysis.weeks:
        facts = facts_for(analysis, week)
        start, end = facts["week"]["week_start"], facts["week"]["week_end"]
        for finding in facts["quality_findings"]:
            assert finding["n_days_in_week"]["value"] == len(finding["dates_in_week"])
            assert all(start <= day <= end for day in finding["dates_in_week"])


# --------------------------------------------------------------------------
# 6. The number check: displays, exactly
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def facts(analysis, last_week):
    return facts_for(analysis, last_week)


def test_a_display_string_passes(facts):
    revenue = facts["totals"]["shopify_revenue"]["display"]
    assert an.check_numbers(f"Store revenue was {revenue}.", facts) == []


@pytest.mark.parametrize("rewritten", ["$531,637.44", "$532k", "$0.5m", "531.6", "$531.60k"])
def test_the_same_number_written_another_way_is_caught(facts, rewritten):
    """There is no rounding allowance, because Python already did the rounding."""
    assert an.check_numbers(f"Store revenue was {rewritten}.", facts) == [rewritten]


def test_an_invented_figure_is_caught(facts):
    assert an.check_numbers("Store revenue was $9.9m this week.", facts) == ["$9.9m"]


def test_a_date_outside_the_facts_is_caught(facts):
    assert an.check_numbers(f"The week ended {facts['week']['week_end']}.", facts) == []
    assert an.check_numbers("The week ended 1999-12-31.", facts) == ["1999-12-31"]


def test_a_name_from_the_facts_may_be_quoted_with_its_digits(analysis):
    """Ad set names carry digits. Quoting one is not writing a figure."""
    flagged = next((a for a in analysis.anomalies if any(c.isdigit() for c in a.ad_set)), None)
    if flagged is None:
        pytest.skip("no flagged ad set has a digit in its name")
    facts = facts_for(analysis, flagged.week)
    assert an.check_numbers(f"Most of it came from ad set {flagged.ad_set}.", facts) == []


def test_prose_without_figures_passes(facts):
    assert an.check_numbers("Spend held up and the platforms claimed less.", facts) == []


def test_a_day_count_the_model_worked_out_itself_is_caught(analysis):
    """The fix for a bad count is more facts, not a looser check."""
    lag = next(f for f in analysis.findings if f.check == "reporting_lag")
    facts = facts_for(analysis, _week_of(analysis, lag.start_date))
    completeness = facts["completeness"]

    quoted = (
        f"{completeness['days_with_complete_data']['display']} of the week's "
        f"{completeness['days_in_week']['display']} days reported."
    )
    assert an.check_numbers(quoted, facts) == []

    invented = next(str(n) for n in range(2, 200) if an.check_numbers(f"{n} days reported.", facts))
    assert an.check_numbers(f"Only {invented} of the week's days reported.", facts) == [invented]


def test_the_offending_sentence_is_named(facts):
    text = "Spend fell sharply. Only 6 of 7 days reported. Revenue held up."
    assert an.offending_sentences(text, ["6"], facts) == ["'6' in: Only 6 of 7 days reported."]


def test_the_sentence_is_found_by_figure_not_by_substring(facts):
    """``6`` lives inside ``$563.5k``-style figures; the sentence named must be the real one."""
    text = "The week ran 2025-06-30 to 2025-07-06. Only 6 of 7 days reported."
    assert an.check_numbers(text, facts) == ["6"]
    assert an.offending_sentences(text, ["6"], facts) == ["'6' in: Only 6 of 7 days reported."]


# --------------------------------------------------------------------------
# 7. The retry, and what happens when it fails
# --------------------------------------------------------------------------


def test_a_bad_first_reply_is_retried_once(monkeypatch, facts):
    replies = iter(["Spend held up. Revenue was $9.9m.", "Revenue was flat."])
    prompts = []

    def fake_complete(system, prompt, *, model=None, **kwargs):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(llm, "complete", fake_complete)
    assert an.generate_readout(facts) == "Revenue was flat."
    assert len(prompts) == 2
    assert "$9.9m" in prompts[1]
    assert "Revenue was $9.9m." in prompts[1]
    assert "Spend held up." not in prompts[1]


def test_two_bad_replies_save_nothing(monkeypatch, facts, tmp_path):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Spend fell. Only 6 of 7 days reported.")

    with pytest.raises(an.NumberCheckError) as caught:
        an.generate_readout(facts)

    assert caught.value.unsupported == ["6"]
    assert caught.value.sentences == ["'6' in: Only 6 of 7 days reported."]
    assert "Only 6 of 7 days reported." in str(caught.value)
    assert not list(tmp_path.glob("*.json"))


# --------------------------------------------------------------------------
# 8. The CLI, in dry-run
# --------------------------------------------------------------------------


def test_dry_run_writes_one_file_per_week(dataset, tmp_path, capsys):
    code = an.main(["--weeks", "3", "--data", str(dataset.path), "--out", str(tmp_path), "--dry-run"])
    capsys.readouterr()

    assert code == 0
    written = sorted(tmp_path.glob("week-*.json"))
    assert len(written) == 3

    record = json.loads(written[-1].read_text())
    assert record["mode"] == "dry-run" and record["model"] is None
    assert record["word_count"] <= record["word_limit"]
    assert an.week_number(record["facts"]) == record["week"]
    assert an.check_numbers(record["text"], record["facts"]) == []
    assert "sample" in record["disclaimer"].lower()


def test_dry_run_never_calls_a_model(monkeypatch, dataset, tmp_path, capsys):
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not call the API")

    monkeypatch.setattr(llm, "complete", explode)
    assert an.main(["--weeks", "2", "--data", str(dataset.path), "--out", str(tmp_path), "--dry-run"]) == 0
    capsys.readouterr()
    assert "anthropic" not in sys.modules


def test_weeks_must_be_positive(dataset, tmp_path, capsys):
    assert an.main(["--weeks", "0", "--data", str(dataset.path), "--out", str(tmp_path), "--dry-run"]) == 2
    capsys.readouterr()


def test_more_weeks_than_the_data_has_is_capped(dataset, tmp_path, capsys):
    an.main(["--weeks", "999", "--data", str(dataset.path), "--out", str(tmp_path), "--dry-run"])
    capsys.readouterr()
    assert len(list(tmp_path.glob("week-*.json"))) == len(an.analyse(dataset.path).weeks)


# --------------------------------------------------------------------------
# 9. Configuration -- read from the environment, never echoed
# --------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch):
    """No .env loading, no inherited variables."""
    monkeypatch.setattr(llm, "_environment", lambda: None)
    monkeypatch.delenv(llm.MODEL_VARIABLE, raising=False)
    monkeypatch.delenv(llm.KEY_VARIABLE, raising=False)


def test_model_defaults_to_the_current_haiku(clean_env):
    assert llm.model_id() == llm.DEFAULT_MODEL == "claude-haiku-4-5"


def test_model_comes_from_the_environment(clean_env, monkeypatch):
    monkeypatch.setenv(llm.MODEL_VARIABLE, "claude-sonnet-5")
    assert llm.model_id() == "claude-sonnet-5"
    assert llm.model_id("claude-opus-5") == "claude-opus-5"


def test_a_missing_key_is_an_error_not_a_request(clean_env):
    assert llm.have_key() is False
    with pytest.raises(llm.ModelError) as caught:
        llm.complete("system", "prompt")
    assert llm.KEY_VARIABLE in str(caught.value)


def test_the_key_is_never_in_the_message(clean_env, monkeypatch):
    monkeypatch.setenv(llm.KEY_VARIABLE, "sk-ant-not-a-real-key")
    assert llm.have_key() is True
    assert "sk-ant-not-a-real-key" not in str(llm.model_id())


def test_a_share_over_one_hundred_percent_is_kept_out_of_the_prose(analysis):
    """Channels pulling against each other make a real share that explains nothing."""
    checked = 0
    for week in analysis.weeks:
        facts = facts_for(analysis, week)
        largest = (facts["drivers"] or {}).get("platform_attributed_revenue", {}).get("largest")
        if not largest or largest["share_of_change"] is None:
            continue
        share = largest["share_of_change"]
        if abs(share["value"]) > 1.0:
            checked += 1
            assert share["display"] not in an.template_readout(facts)
    assert checked, "this dataset should have a week whose shares pull against each other"
