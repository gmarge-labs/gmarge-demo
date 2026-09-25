"""The keep-awake check: the parts that can be tested without a browser.

The browser half is exercised by the workflow itself every six hours, against
the real app, which is the only place it can be exercised honestly. What is
tested here is the logic around it -- which is where the mistakes that fail
*silently* live:

* checking for text the app no longer renders, so a healthy app reads as down;
* a wake pattern loose enough to match something the app itself renders, so the
  script clicks around inside a working app;
* looking for the app's content on the top-level page, when Streamlit Cloud
  runs the app inside an iframe and the top-level page never contains it.

``scripts/keep_awake.py`` imports ``playwright`` inside ``main()`` rather than
at module scope, the same way ``gmarge/llm.py`` defers ``anthropic``, so this
file can import it with neither the package nor a browser installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "keep_awake.py"
WORKFLOW = ROOT / ".github" / "workflows" / "keep-awake.yml"

sys.path.insert(0, str(ROOT / "scripts"))

import keep_awake  # noqa: E402


# --------------------------------------------------------------------------
# A page, as much of one as these functions touch
# --------------------------------------------------------------------------


class FakeFrame:
    """One frame, with text and optionally a button."""

    def __init__(self, text: str = "", button: str | None = None, detached: bool = False):
        self.text = text
        self.button = button
        self.detached = detached
        self.clicked = False

    def evaluate(self, _script: str) -> str:
        if self.detached:
            raise RuntimeError("Frame was detached")
        return self.text

    def get_by_role(self, role: str, name):
        assert role == "button"
        matches = self.button is not None and name.search(self.button) is not None
        return FakeLocator(self, matches)


class FakeLocator:
    def __init__(self, frame: FakeFrame, matches: bool):
        self.frame = frame
        self.matches = matches

    def count(self) -> int:
        return 1 if self.matches else 0

    @property
    def first(self):
        return self

    def click(self, timeout: int = 0) -> None:
        self.frame.clicked = True


class FakePage:
    def __init__(self, *frames: FakeFrame):
        self.frames = list(frames)


# The page Streamlit Cloud actually serves: a host frame that does not contain
# the app's text, and the app in an iframe underneath it.
HOST_TEXT = "Manage app · Streamlit"
APP_FRAME_TEXT = (
    "G-Marge\nNorthfield Goods (sample brand)\n"
    "Sample data for a fictional brand. See what G-Marge would show you. Talk to us →\n"
    "Where the money actually went"
)

# Streamlit Cloud's sleep screen, as it words it.
SLEEP_TEXT = "This app has gone to sleep due to inactivity. Would you like to wake it back up?"
SLEEP_BUTTON = "Yes, get this app back up!"


# --------------------------------------------------------------------------
# What counts as loaded
# --------------------------------------------------------------------------


def test_the_text_it_waits_for_is_the_banner_the_app_renders():
    """The one that keeps this honest.

    If the banner is reworded and this string is not, the check looks for text
    that no longer exists and reports a healthy app as down, every six hours,
    for as long as it takes someone to read the email.
    """
    sys.path.insert(0, str(ROOT))
    import app

    assert keep_awake.APP_TEXT == app.BANNER


def test_an_app_rendering_in_its_iframe_counts_as_loaded():
    page = FakePage(FakeFrame(HOST_TEXT), FakeFrame(APP_FRAME_TEXT))
    assert keep_awake.is_loaded(page)


def test_the_host_page_alone_does_not_count_as_loaded():
    """HTTP 200 and a host page are what a sleeping app also serves."""
    assert not keep_awake.is_loaded(FakePage(FakeFrame(HOST_TEXT)))
    assert not keep_awake.is_loaded(FakePage(FakeFrame(HOST_TEXT), FakeFrame(SLEEP_TEXT)))
    assert not keep_awake.is_loaded(FakePage())


def test_a_frame_that_detaches_mid_read_is_skipped_not_fatal():
    page = FakePage(FakeFrame(detached=True), FakeFrame(APP_FRAME_TEXT))
    assert keep_awake.is_loaded(page)


def test_a_half_rendered_app_does_not_count():
    """Streamlit paints the sidebar before it has run the script."""
    partial = "G-Marge\nNorthfield Goods (sample brand)\nOverview\nChannels"
    assert not keep_awake.is_loaded(FakePage(FakeFrame(HOST_TEXT), FakeFrame(partial)))


# --------------------------------------------------------------------------
# Finding the wake button
# --------------------------------------------------------------------------


def test_the_wake_button_is_found_and_clicked():
    sleeping = FakeFrame(SLEEP_TEXT, button=SLEEP_BUTTON)
    assert keep_awake.click_wake(FakePage(FakeFrame(HOST_TEXT), sleeping))
    assert sleeping.clicked


@pytest.mark.parametrize(
    "wording",
    [
        "Yes, get this app back up!",
        "get this app back up",
        "Wake this app back up",
        "Yes, wake the app up",
    ],
)
def test_the_pattern_survives_streamlits_wording(wording):
    """Streamlit's wording is not ours and has changed before."""
    assert keep_awake.WAKE.search(wording)


@pytest.mark.parametrize(
    "wording",
    [
        "Sample data for a fictional brand. See what G-Marge would show you.",
        "Talk to us",
        "Data health",
        "Manage app",
        "Deploy",
        "Share",
    ],
)
def test_the_pattern_never_matches_something_the_app_renders(wording):
    """A loose pattern would have the script clicking around a working app."""
    assert not keep_awake.WAKE.search(wording)


def test_nothing_is_clicked_on_a_healthy_app():
    app_frame = FakeFrame(APP_FRAME_TEXT, button="Talk to us")
    assert not keep_awake.click_wake(FakePage(FakeFrame(HOST_TEXT), app_frame))
    assert not app_frame.clicked


def test_a_sleeping_page_is_recognised_for_the_log():
    assert keep_awake.looks_asleep(FakePage(FakeFrame(SLEEP_TEXT)))
    assert not keep_awake.looks_asleep(FakePage(FakeFrame(APP_FRAME_TEXT)))


# --------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------


def test_it_gives_up_after_three_minutes():
    """The point of the run is the email that follows the failure."""
    assert keep_awake.TIMEOUT == 180.0
    assert keep_awake.POLL < keep_awake.RELOAD_AFTER < keep_awake.TIMEOUT


def test_a_page_that_never_renders_fails_within_the_budget():
    page = FakePage(FakeFrame(HOST_TEXT), FakeFrame(SLEEP_TEXT))
    page.goto = lambda *a, **k: None

    started = __import__("time").monotonic()
    assert not keep_awake.wait_for_app(page, "https://example.invalid", timeout=1.0)
    assert __import__("time").monotonic() - started < 10


def test_an_app_that_is_already_awake_returns_at_once():
    page = FakePage(FakeFrame(HOST_TEXT), FakeFrame(APP_FRAME_TEXT))
    page.goto = lambda *a, **k: None
    assert keep_awake.wait_for_app(page, "https://example.invalid", timeout=5.0)


# --------------------------------------------------------------------------
# The workflow
# --------------------------------------------------------------------------


def test_the_workflow_runs_on_a_schedule_and_on_demand():
    workflow = WORKFLOW.read_text()
    assert "schedule:" in workflow
    assert 'cron: "0 */6 * * *"' in workflow
    assert "workflow_dispatch:" in workflow


def test_playwright_is_pinned_exactly():
    workflow = WORKFLOW.read_text()
    assert 'PLAYWRIGHT_VERSION: "1.63.0"' in workflow
    assert "playwright==" in workflow
    for loose in (">=", "~=", "playwright\n", "playwright "):
        assert f"pip install {loose}" not in workflow


def test_playwright_is_not_an_app_dependency():
    """CLAUDE.md rule 5: the app's dependency set is exactly those eight."""
    requirements = (ROOT / "requirements.txt").read_text().lower()
    assert "playwright" not in requirements


def test_the_workflow_installs_chromium_and_runs_the_one_script():
    workflow = WORKFLOW.read_text()
    assert "playwright install --with-deps chromium" in workflow
    assert "python scripts/keep_awake.py" in workflow


def test_the_failure_screenshot_is_kept():
    workflow = WORKFLOW.read_text()
    assert keep_awake.FAILURE_SCREENSHOT in workflow
    assert "if: failure()" in workflow
