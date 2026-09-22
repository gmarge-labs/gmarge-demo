"""The facts must carry the read-out, and the number check must bite.

Every test here runs with ``--dry-run``: no API key, no network, no model.
That is the point of the dry run -- the template read-out is assembled from
the same facts dict the model is given and held to the same number check, so
these tests prove the facts are sufficient and the check is honest without
anything leaving the machine.
"""

from __future__ import annotations

import json
import sys

import pytest

from gmarge import analyst as an
from gmarge import llm
from gmarge import metrics as mx


@pytest.fixture(scope="session")
def analysis(dataset):
    return an.analyse(dataset.path)


@pytest.fixture(scope="session")
def last_week(analysis):
    return analysis.weeks[-1]


def facts_for(analysis, week):
    return an.build_facts(analysis, week)


# --------------------------------------------------------------------------
# 1. The facts carry the read-out
# --------------------------------------------------------------------------


def test_template_readout_uses_only_facts(analysis):
    """Every figure in every template read-out traces back to its facts."""
    for week in analysis.weeks:
        facts = facts_for(analysis, week)
        text = an.template_readout(facts)
        assert an.check_numbers(text, facts) == [], f"week {week}: {text}"


def test_template_readout_is_within_the_word_limit(analysis):
    for week in analysis.weeks:
        text = an.template_readout(facts_for(analysis, week))
        assert an.word_count(text) <= an.WORD_LIMIT


def test_facts_are_json_serialisable(analysis, last_week):
    facts = facts_for(analysis, last_week)
    assert json.loads(json.dumps(facts, default=str))["week"]["week"] == last_week


def test_unknown_week_is_refused(analysis):
    with pytest.raises(ValueError):
        an.build_facts(analysis, max(analysis.weeks) + 1)


# --------------------------------------------------------------------------
# 2. The numbers in the facts are the numbers in the tables
# --------------------------------------------------------------------------


def test_totals_match_the_weekly_reconciliation(analysis, last_week):
    facts = facts_for(analysis, last_week)
    row = analysis.metrics["weekly_reconciliation"].set_index("week").loc[last_week]

    assert facts["totals"]["shopify_revenue"] == pytest.approx(row["shopify_revenue"], abs=0.005)
    assert facts["totals"]["ad_spend"] == pytest.approx(row["ad_spend"], abs=0.005)
    assert facts["totals"]["over_claim_ratio"] == pytest.approx(row["over_claim_ratio"], abs=5e-5)
    assert facts["week"]["week_start"] == str(row["week_start"].date())


def test_change_is_this_week_against_last(analysis, last_week):
    facts = facts_for(analysis, last_week)
    prior = facts_for(analysis, last_week - 1)

    for key in ("shopify_revenue", "ad_spend", "platform_attributed_revenue"):
        expected = facts["totals"][key] - prior["totals"][key]
        assert facts["change_vs_prior_week"][key]["absolute"] == pytest.approx(expected, abs=0.02)
    assert facts["prior_week_totals"]["ad_spend"] == prior["totals"]["ad_spend"]


def test_channel_changes_sum_to_the_total_change(analysis, last_week):
    facts = facts_for(analysis, last_week)
    for column, driver in facts["drivers"].items():
        total = sum(c["change"] for c in driver["channels"])
        assert total == pytest.approx(driver["total_change"], abs=0.05)
        assert abs(driver["largest"]["change"]) == max(abs(c["change"]) for c in driver["channels"])


def test_first_week_has_no_prior_week_and_no_band(analysis):
    facts = facts_for(analysis, analysis.weeks[0])
    assert facts["prior_week"] is None
    assert facts["change_vs_prior_week"] is None
    assert facts["drivers"] is None
    assert facts["normal_range"]["ad_spend"]["enough_history"] is False


# --------------------------------------------------------------------------
# 3. Findings and flags land in the week they belong to
# --------------------------------------------------------------------------


def _week_of(analysis, date: str) -> int:
    weekly = analysis.metrics["weekly_reconciliation"]
    hit = weekly[(weekly["week_start"].astype(str).str[:10] <= date) & (weekly["week_end"].astype(str).str[:10] >= date)]
    return int(hit.iloc[0]["week"])


def test_quality_findings_land_in_their_own_week(analysis):
    for finding in analysis.findings:
        week = _week_of(analysis, finding.start_date)
        checks = [f["check"] for f in facts_for(analysis, week)["quality_findings"]]
        assert finding.check in checks


def test_the_reporting_lag_makes_its_week_incomplete(analysis):
    lag = next(f for f in analysis.findings if f.check == "reporting_lag")
    facts = facts_for(analysis, _week_of(analysis, lag.start_date))

    assert facts["completeness"]["complete"] is False
    assert lag.source in {a["source"] for a in facts["completeness"]["affected"]}
    assert "incomplete" in an.template_readout(facts).lower()


def test_a_clean_week_is_complete(analysis):
    dirty = {_week_of(analysis, f.start_date) for f in analysis.findings}
    clean = next(w for w in analysis.weeks if w not in dirty)
    completeness = facts_for(analysis, clean)["completeness"]

    assert completeness["complete"] is True
    assert completeness["days_affected"] == 0
    assert completeness["affected_dates"] == []
    assert completeness["days_with_complete_data"] == completeness["days_in_week"]


@pytest.mark.parametrize("check", an.INCOMPLETE_CHECKS)
def test_an_incomplete_week_carries_every_day_count(analysis, check):
    """The counts a read-out needs are in the facts, so nothing has to be worked out.

    Without these the model has to subtract for itself to say how much of the
    week reported -- which is exactly the arithmetic it is not allowed to do.
    """
    finding = next((f for f in analysis.findings if f.check == check), None)
    if finding is None:
        pytest.skip(f"this dataset has no {check} finding")

    facts = facts_for(analysis, _week_of(analysis, finding.start_date))
    completeness = facts["completeness"]

    assert completeness["days_in_week"] == facts["week"]["days"]
    assert completeness["days_affected"] == len(completeness["affected_dates"])
    assert (
        completeness["days_with_complete_data"] + completeness["days_affected"]
        == completeness["days_in_week"]
    )
    assert set(finding.dates) & set(completeness["affected_dates"])
    for entry in completeness["affected"]:
        assert entry["n_days_in_week"] == len(entry["dates_in_week"])


def test_findings_carry_the_days_that_fall_inside_the_week(analysis):
    for week in analysis.weeks:
        facts = facts_for(analysis, week)
        start, end = facts["week"]["week_start"], facts["week"]["week_end"]
        for finding in facts["quality_findings"]:
            assert finding["n_days_in_week"] == len(finding["dates_in_week"])
            assert finding["n_days_in_week"] >= 1
            assert all(start <= day <= end for day in finding["dates_in_week"])


def test_a_day_count_the_model_worked_out_itself_is_caught(analysis):
    """The fix is more facts, not a looser check: an unquotable count still fails."""
    lag = next(f for f in analysis.findings if f.check == "reporting_lag")
    facts = facts_for(analysis, _week_of(analysis, lag.start_date))
    completeness = facts["completeness"]

    quoted = (
        f"{completeness['days_with_complete_data']} of the week's "
        f"{completeness['days_in_week']} days reported."
    )
    assert an.check_numbers(quoted, facts) == []

    invented = next(
        str(n) for n in range(2, 200) if an.check_numbers(f"{n} days reported.", facts)
    )
    assert an.check_numbers(f"Only {invented} of the week's days reported.", facts) == [invented]


def test_anomalies_land_in_their_own_week(analysis):
    for flag in analysis.anomalies:
        facts = facts_for(analysis, flag.week)
        assert any(a["channel"] == flag.channel and a["metric"] == flag.metric for a in facts["anomalies"])
    weeks_with_flags = {a.week for a in analysis.anomalies}
    for week in analysis.weeks:
        if week not in weeks_with_flags:
            assert facts_for(analysis, week)["anomalies"] == []


# --------------------------------------------------------------------------
# 4. The number check
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def facts(analysis, last_week):
    return facts_for(analysis, last_week)


def test_an_invented_figure_is_caught(facts):
    text = "Store revenue was $1,234,567.89 this week."
    assert an.check_numbers(text, facts) == ["$1,234,567.89"]


def test_a_figure_rounded_to_the_nearest_thousand_is_caught(facts):
    revenue = facts["totals"]["shopify_revenue"]
    rounded = f"${round(revenue, -3):,.0f}"
    assert an.check_numbers(f"Store revenue was {rounded}.", facts) == [rounded]


def test_dropping_decimal_places_is_allowed(facts):
    revenue = facts["totals"]["shopify_revenue"]
    assert an.check_numbers(f"Store revenue was ${revenue:,.2f}.", facts) == []
    assert an.check_numbers(f"Store revenue was ${revenue:,.0f}.", facts) == []


def test_a_ratio_may_be_written_as_a_percentage(facts):
    pct = facts["change_vs_prior_week"]["shopify_revenue"]["pct"]
    assert an.check_numbers(f"Revenue moved {abs(pct) * 100:.1f}%.", facts) == []
    assert an.check_numbers(f"Revenue moved {abs(pct) * 100 + 5:.1f}%.", facts) != []


def test_a_date_outside_the_facts_is_caught(facts):
    assert an.check_numbers(f"The week ended {facts['week']['week_end']}.", facts) == []
    assert an.check_numbers("The week ended 1999-12-31.", facts) == ["1999-12-31"]


def test_a_name_from_the_facts_may_be_quoted_with_its_digits(analysis):
    """Ad set names carry digits. Quoting one is not inventing a figure."""
    flagged = next((a for a in analysis.anomalies if any(c.isdigit() for c in a.ad_set)), None)
    if flagged is None:
        pytest.skip("no flagged ad set has a digit in its name")
    facts = facts_for(analysis, flagged.week)
    assert an.check_numbers(f"Most of it came from ad set {flagged.ad_set}.", facts) == []


def test_prose_without_figures_passes(facts):
    assert an.check_numbers("Spend fell and the platforms claimed less revenue.", facts) == []


# --------------------------------------------------------------------------
# 5. The retry, and what happens when it fails
# --------------------------------------------------------------------------


def test_a_bad_first_reply_is_retried_once(monkeypatch, facts):
    replies = iter(["Spend held up. Revenue was $9,999,999.00.", "Revenue was flat."])
    prompts = []

    def fake_complete(system, prompt, *, model=None, **kwargs):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(llm, "complete", fake_complete)
    assert an.generate_readout(facts) == "Revenue was flat."
    assert len(prompts) == 2
    assert "$9,999,999.00" in prompts[1]
    # the sentence goes back too, not just the figure
    assert "Revenue was $9,999,999.00." in prompts[1]
    assert "Spend held up." not in prompts[1]


def test_the_offending_sentence_is_named(facts):
    text = "Spend fell sharply. Only 6 of 7 days reported. Revenue held up."
    assert an.offending_sentences(text, ["6"], facts) == ["'6' in: Only 6 of 7 days reported."]


def test_the_sentence_is_found_by_figure_not_by_substring(facts):
    """``6`` lives inside ``$563,472.11``; the sentence named must be the real one."""
    revenue = facts["totals"]["shopify_revenue"]
    text = f"Store revenue was ${revenue:,.2f}. Only 6 of 7 days reported."

    assert an.check_numbers(text, facts) == ["6"]
    assert an.offending_sentences(text, ["6"], facts) == ["'6' in: Only 6 of 7 days reported."]


def test_a_failure_reports_the_sentence_not_just_the_figure(monkeypatch, facts):
    """A bare '6' is undiagnosable; '6' in its sentence says what went wrong."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Spend fell. Only 6 of 7 days reported.")

    with pytest.raises(an.NumberCheckError) as caught:
        an.generate_readout(facts)

    error = caught.value
    assert error.unsupported == ["6"]
    assert error.sentences == ["'6' in: Only 6 of 7 days reported."]
    assert "Only 6 of 7 days reported." in str(error)


def test_two_bad_replies_save_nothing(monkeypatch, facts, tmp_path):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Revenue was $9,999,999.00.")

    with pytest.raises(an.NumberCheckError) as caught:
        an.generate_readout(facts)
    assert caught.value.unsupported == ["$9,999,999.00"]
    assert not list(tmp_path.glob("*.json"))


# --------------------------------------------------------------------------
# 6. The CLI, in dry-run
# --------------------------------------------------------------------------


def test_dry_run_writes_one_file_per_week(dataset, tmp_path, capsys):
    code = an.main(["--weeks", "3", "--data", str(dataset.path), "--out", str(tmp_path), "--dry-run"])
    capsys.readouterr()

    assert code == 0
    written = sorted(tmp_path.glob("week-*.json"))
    assert len(written) == 3

    record = json.loads(written[-1].read_text())
    assert record["mode"] == "dry-run"
    assert record["model"] is None
    assert record["word_count"] <= record["word_limit"]
    assert record["text"] and record["facts"]["week"]["week"] == record["week"]
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
# 7. Configuration -- read from the environment, never echoed
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
