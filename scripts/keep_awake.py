"""Keep the public demo awake, and fail loudly when it is not.

Streamlit Community Cloud puts an app to sleep after a stretch with no
visitors. Waking it is a click on a page, not an HTTP request, so this opens
the app in a real browser, clicks the wake button if one is there, and then
waits until the app has *rendered* -- not until the server returns 200, which
it does while asleep, while waking, and while showing an error.

Run by ``.github/workflows/keep-awake.yml`` every six hours. It is CI tooling
and not part of the app: ``playwright`` is pinned in the workflow and is
deliberately not in ``requirements.txt``, which stays exactly the eight
packages CLAUDE.md lists.

Two things about the page are worth knowing before changing any of this.

**The app is in an iframe.** Streamlit Cloud serves the host page and runs the
app inside ``<iframe title="streamlitApp">``. Looking for the app's text on the
top-level page finds nothing even when the app is perfectly healthy, so every
check here runs over ``page.frames``.

**What counts as loaded is the app's own words.** :data:`APP_TEXT` is the
banner from ``app.py``, which only appears once Streamlit has run the script
and rendered. ``tests/test_keep_awake.py`` asserts the two strings are still
the same, so changing the banner cannot leave this silently checking for text
that no longer exists.
"""

from __future__ import annotations

import os
import re
import sys
import time

APP_URL = "https://gmarge-demo.streamlit.app"

# The banner app.py puts on every page. Kept identical to app.BANNER by a test.
APP_TEXT = "Sample data for a fictional brand. See what G-Marge would show you."

# The button Streamlit Cloud shows on a sleeping app ("Yes, get this app back
# up!"). Matched on the accessible name of a button, not on page text, and
# written with word boundaries so it cannot fire on something the app itself
# renders. The alternatives are there because this is someone else's wording
# and it has changed before.
WAKE = re.compile(r"get this app back up|\bwake\b.{0,20}\bapp\b|\bapp back up\b", re.I)

# Only used to say something useful in the log. Nothing branches on it.
ASLEEP = re.compile(r"gone to sleep|is asleep|waking up|spinning up", re.I)

# The whole budget, from the first navigation to the app rendering. A cold
# start on the community tier is usually well under a minute; three minutes is
# the point at which something is wrong and a human should hear about it.
TIMEOUT = 180.0
POLL = 3.0
# A wake sometimes finishes without the host page noticing. Re-navigating costs
# nothing and unsticks it.
RELOAD_AFTER = 45.0
NAVIGATION_TIMEOUT = 60_000
FAILURE_SCREENSHOT = "keep-awake-failure.png"


def log(message: str) -> None:
    """One line, flushed, so the Actions log reads in order."""
    print(message, flush=True)


def frame_texts(page) -> list[str]:
    """The visible text of every frame, the app's own frame included.

    A frame can be detached mid-read while the page is still settling, which
    raises rather than returning nothing; that is a frame that no longer
    matters, so it is skipped.
    """
    texts = []
    for frame in page.frames:
        try:
            text = frame.evaluate("() => document.body ? document.body.innerText : ''")
        except Exception:
            continue
        if text:
            texts.append(text)
    return texts


def is_loaded(page) -> bool:
    """Whether the app has rendered its own content somewhere on the page."""
    return any(APP_TEXT in text for text in frame_texts(page))


def looks_asleep(page) -> bool:
    return any(ASLEEP.search(text) for text in frame_texts(page))


def click_wake(page) -> bool:
    """Click the wake button if the page is showing one. True if clicked."""
    for frame in page.frames:
        try:
            button = frame.get_by_role("button", name=WAKE)
            if button.count() == 0:
                continue
            button.first.click(timeout=15_000)
            return True
        except Exception:
            continue
    return False


def wait_for_app(page, url: str, timeout: float = TIMEOUT) -> bool:
    """Open the app and wait for it to render. True if it did, inside budget."""
    deadline = time.monotonic() + timeout
    page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT)
    last_reload = time.monotonic()
    woken = False

    while time.monotonic() < deadline:
        if is_loaded(page):
            return True

        if not woken and click_wake(page):
            log("The app was asleep. Clicked the wake button.")
            woken = True
            last_reload = time.monotonic()
        elif not woken and looks_asleep(page):
            log("The app looks asleep but shows no wake button yet; waiting.")

        if time.monotonic() - last_reload > RELOAD_AFTER:
            log(f"Still not rendered after {RELOAD_AFTER:.0f}s. Reloading.")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT)
            except Exception as exc:
                log(f"Reload failed: {exc}")
            last_reload = time.monotonic()

        time.sleep(POLL)

    return is_loaded(page)


def report_failure(page, url: str, timeout: float) -> None:
    """Say what was on the screen instead, and keep a picture of it."""
    log(f"FAILED: {url} did not render the app within {timeout:.0f}s.")
    log(f"Looked for: {APP_TEXT!r}")
    for index, text in enumerate(frame_texts(page)):
        excerpt = " ".join(text.split())[:300]
        log(f"  frame {index}: {excerpt or '(empty)'}")
    try:
        page.screenshot(path=FAILURE_SCREENSHOT, full_page=True)
        log(f"Screenshot written to {FAILURE_SCREENSHOT}.")
    except Exception as exc:
        log(f"Could not take a screenshot: {exc}")


def main(argv: list[str] | None = None) -> int:
    # Imported here, not at module scope, so the tests can import this module
    # and check its logic without the browser or the package being installed.
    from playwright.sync_api import sync_playwright

    argv = sys.argv[1:] if argv is None else argv
    url = argv[0] if argv else os.environ.get("APP_URL", APP_URL)

    log(f"Opening {url}")
    started = time.monotonic()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            loaded = wait_for_app(page, url)
            if loaded:
                log(f"The app rendered after {time.monotonic() - started:.1f}s. Awake.")
                return 0
            report_failure(page, url, TIMEOUT)
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
