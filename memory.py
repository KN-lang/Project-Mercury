"""
memory.py — Project Mercury
Manages persistent conversation history within the Streamlit session.
All state lives in st.session_state so it survives reruns but resets on page refresh.
"""
from datetime import datetime
from typing import Any
import streamlit as st


# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "history":               [],     # list of pipeline result dicts
    "pending_action":        None,   # action dict awaiting human confirmation
    "awaiting_confirmation": False,  # True when we're paused for HITL
    "last_result":           None,   # most recent pipeline result
    "transcription":         "",     # last transcribed text (for display)
    "intent_result":         None,   # raw intent dict from agent
    "processing":            False,  # True while pipeline is running
}


def init_memory() -> None:
    """Initialise all required session_state keys on first run."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def add_to_history(entry: dict) -> None:
    """Append a completed pipeline run to the session history."""
    entry = dict(entry)  # shallow copy — don't mutate caller's dict
    entry.setdefault("timestamp", datetime.now().strftime("%H:%M:%S"))
    st.session_state.history.append(entry)


def get_history() -> list[dict]:
    """Return the full history list (read-only intent)."""
    return st.session_state.history


def clear_history() -> None:
    """Wipe conversation history and last result."""
    st.session_state.history      = []
    st.session_state.last_result  = None
    st.session_state.transcription = ""
    st.session_state.intent_result = None


def set_pending(action: dict) -> None:
    """Park an action for human-in-the-loop review."""
    st.session_state.pending_action        = action
    st.session_state.awaiting_confirmation = True


def clear_pending() -> None:
    """Clear the pending action gate."""
    st.session_state.pending_action        = None
    st.session_state.awaiting_confirmation = False
