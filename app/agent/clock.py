"""
One frozen "now" per request (the temporal anchor).

An instruction sent at 23:59 IST can take a minute or two to plan and run. Without
this, the planner could read "today" as Sunday and a worker, after midnight, as
Monday, so "tomorrow" would mean two different days. The moment a request enters
Socio (the Telegram message's own timestamp, a CLI call, an email being handled),
frozen() fixes the date; every prompt built inside it uses the same today and
tomorrow, written out as exact dates in a <temporal_anchor> block.

It is a contextvar: worker threads get it through contextvars.copy_context()
(supervisor.run_plan). Tool checks that must use the real time (is a slot in the
past? is it free now?) keep calling datetime.now themselves.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Iterator

from app.calendar.tools import INDIA_TZ


TIMEZONE_NAME = "Asia/Kolkata"

_frozen: ContextVar[datetime | None] = ContextVar("socio_frozen_now", default=None)


def now() -> datetime:
    """The request's frozen time in IST, or the real time outside a request."""

    return _frozen.get() or datetime.now(INDIA_TZ)


@contextmanager
def frozen(at: datetime | None = None) -> Iterator[datetime]:
    """Freeze now() for everything inside the block (an outer freeze wins: it came in first)."""

    if _frozen.get() is not None:
        yield _frozen.get()
        return
    moment = (at or datetime.now(INDIA_TZ)).astimezone(INDIA_TZ)
    token = _frozen.set(moment)
    try:
        yield moment
    finally:
        _frozen.reset(token)


def temporal_context(at: datetime | None = None) -> dict[str, str]:
    """The anchor's fields (for logs and tests)."""

    current = (at or now()).astimezone(INDIA_TZ)
    tomorrow = current + timedelta(days=1)
    return {
        "origin_timestamp_utc": current.astimezone(timezone.utc).isoformat(),
        "user_local_time": current.strftime("%Y-%m-%d %H:%M:%S"),
        "user_timezone": TIMEZONE_NAME,
        "target_date_today": current.strftime("%Y-%m-%d"),
        "target_date_tomorrow": tomorrow.strftime("%Y-%m-%d"),
        "day_of_week": current.strftime("%A"),
        "day_of_week_tomorrow": tomorrow.strftime("%A"),
    }


def temporal_anchor(at: datetime | None = None) -> str:
    """The block every planner / agent / worker prompt carries."""

    c = temporal_context(at)
    return (
        f"<temporal_anchor>Today is {c['target_date_today']} ({c['day_of_week']}), "
        f"now {c['user_local_time'][11:16]}. Tomorrow is {c['target_date_tomorrow']} ({c['day_of_week_tomorrow']}). "
        f"Timezone: {TIMEZONE_NAME} (IST, UTC+05:30). Resolve every relative date (today, tomorrow, "
        f"next Friday, in 2 hours) against this, never against another clock.</temporal_anchor>"
    )
