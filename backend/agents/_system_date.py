# agents/_system_date.py — date preamble for agent system prompts
# Author: Hridam Biswas | Project: Helios
#
# Shared helpers that prepend today's date to every agent's system prompt.
# Computed at invoke time (not module import) so a long-running server
# doesn't drift from UTC day to day, and so training-cutoff guidance uses
# the current date rather than whenever the process booted.

from __future__ import annotations
from datetime import date

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def today_iso() -> str:
    return date.today().isoformat()


def today_long() -> str:
    d = date.today()
    return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"


def date_preamble() -> str:
    """Standard date preamble prepended to every agent's system prompt."""
    return (
        f"Today's date is {today_long()} ({today_iso()}). "
        "Your training data has a knowledge cutoff; events that happened "
        "after that cutoff may still appear in the web results provided "
        "to you below.\n\n"
    )


def date_preamble_with_web_trust() -> str:
    """For the judge and the skeptic — tells the model to trust dated web
    results over its own memory for anything that happened after the
    training cutoff. Without this, the skeptic keeps flagging post-cutoff
    events as 'has not happened yet'."""
    return date_preamble() + (
        "When web results describe an event that is dated after your "
        "training cutoff, TRUST THE DATED WEB RESULTS over your own "
        "memory. Never claim an event 'has not happened yet' or 'is in "
        "the future' if a web result shows it already happened.\n\n"
    )
