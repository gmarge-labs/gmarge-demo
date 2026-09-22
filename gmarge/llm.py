"""The one place in this package that talks to a model.

Nothing else imports ``anthropic``. The Streamlit app never reaches this
module: read-outs are generated here, offline, checked by a human and saved to
``readouts/``, and the app only ever reads those files (see CLAUDE.md).

Configuration comes from a git-ignored ``.env`` loaded with ``python-dotenv``:

``ANTHROPIC_API_KEY``
    The key. It is read from the environment and handed straight to the SDK --
    never logged, never echoed, never written into a read-out.
``GMARGE_MODEL``
    Optional model id. Defaults to :data:`DEFAULT_MODEL`.

The model is asked to phrase numbers that Python has already computed. It is
never asked to do arithmetic, so the request is deliberately plain: one system
prompt, one user message, no tools, and a token ceiling a little above the
length of a read-out. Nothing model-specific is sent either -- no sampling
parameters, no thinking configuration -- so ``GMARGE_MODEL`` can name any
current model without this module needing to know which.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Claude Haiku 4.5. Small and cheap is the right size for the job here --
# rephrasing a page of facts in under 120 words.
DEFAULT_MODEL = "claude-haiku-4-5"

KEY_VARIABLE = "ANTHROPIC_API_KEY"
MODEL_VARIABLE = "GMARGE_MODEL"

MAX_TOKENS = 1024


class ModelError(RuntimeError):
    """A request to the model could not be made, or came back unusable."""


def _environment() -> None:
    """Load ``.env`` once, without overriding anything already exported."""
    load_dotenv(override=False)


def model_id(override: str | None = None) -> str:
    """The model to call: an explicit override, then ``GMARGE_MODEL``, then the default."""
    if override:
        return override
    _environment()
    return os.environ.get(MODEL_VARIABLE, "").strip() or DEFAULT_MODEL


def have_key() -> bool:
    """Whether a key is present. Does not return it, and does not log it."""
    _environment()
    return bool(os.environ.get(KEY_VARIABLE, "").strip())


def complete(system: str, prompt: str, *, model: str | None = None, max_tokens: int = MAX_TOKENS) -> str:
    """Send one request and return the text of the reply.

    Raises :class:`ModelError` rather than letting an SDK exception escape, so
    that callers have one thing to catch and no key material can surface in a
    traceback frame.
    """
    if not have_key():
        raise ModelError(
            f"{KEY_VARIABLE} is not set. Copy .env.example to .env and fill it in; "
            ".env is git-ignored and must stay that way."
        )

    # Imported here, not at module scope: importing this module must not
    # require the SDK, and --dry-run must not touch it at all.
    import anthropic

    chosen = model_id(model)
    try:
        response = anthropic.Anthropic().messages.create(
            model=chosen,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as exc:
        raise ModelError(f"{KEY_VARIABLE} was rejected by the API.") from exc
    except anthropic.NotFoundError as exc:
        raise ModelError(f"No such model: {chosen!r}. Set {MODEL_VARIABLE} in .env.") from exc
    except anthropic.APIError as exc:
        raise ModelError(f"Request to {chosen!r} failed: {exc}") from exc

    if response.stop_reason == "refusal":
        raise ModelError(f"{chosen!r} declined the request.")

    text = "\n".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise ModelError(f"{chosen!r} returned no text (stop reason {response.stop_reason!r}).")
    return text
