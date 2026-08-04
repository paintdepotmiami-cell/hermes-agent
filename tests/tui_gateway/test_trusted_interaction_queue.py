from __future__ import annotations

import threading

from agent._trusted_interaction_issuer import _issue_trusted_interaction
from tui_gateway import server


def _interaction(message_id: str):
    return _issue_trusted_interaction(
        surface="desktop",
        platform="local",
        actor_id="transport-1",
        channel_id="transport-1",
        chat_id="runtime-session-1",
        thread_id=None,
        session_key="durable-session-1",
        session_id="runtime-session-1",
        message_id=message_id,
        received_at=1_754_000_000.0,
        profile_id="default",
    )


def test_merged_local_prompts_keep_latest_accepted_interaction():
    """Merged text is one model turn, governed by its latest accepted input."""
    session = {
        "history_lock": threading.Lock(),
        "queued_prompt": None,
    }
    first = _interaction("local-message-1")
    latest = _interaction("local-message-2")

    server._enqueue_prompt(
        session,
        "first fragment",
        object(),
        trusted_interaction=first,
    )
    server._enqueue_prompt(
        session,
        "latest fragment",
        object(),
        trusted_interaction=latest,
    )

    queued = session["queued_prompt"]
    assert queued["text"] == "first fragment\n\nlatest fragment"
    assert queued["trusted_interaction"] is latest
    assert queued["trusted_interaction"].message_id == "local-message-2"
