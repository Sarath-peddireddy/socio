"""
Scheduled routines: when they run and how they are stored.

A routine is something Socio does for the owner on a schedule ("send me the top AI
news every day at 9 AM"): its prompt_query is run like an owner instruction and the
answer is sent to Telegram (app.scheduler.service).

Times:
- The owner gives a time of day in a timezone (default Asia/Kolkata, IST). zoneinfo
  (with the tzdata package on Windows) handles any IANA zone, daylight saving included.
- next_run_at is stored as UTC text in one fixed format (UTC_FORMAT), so the "due"
  query compares strings correctly. (SQLite's datetime('now') uses a space instead of
  the 'T', so it must never be compared with these values directly.)
- cron_expr is "minute hour * * day-of-week" in the job's own timezone; only the forms
  this module writes are read back (no general cron parser).
"""

from __future__ import annotations

import re
import threading
import uuid
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.agent.state import AgentStateStore


DEFAULT_TIMEZONE = "Asia/Kolkata"
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FREQUENCIES = ("daily", "weekdays", "weekly", "one_off")
MAX_ACTIVE_JOBS = 20
MAX_TITLE_CHARS = 80
MAX_QUERY_CHARS = 1000
# A one-off job may be set at most this far ahead.
MAX_DAYS_AHEAD = 366

DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
# cron counts days from Sunday = 0; Python's weekday() from Monday = 0.
_CRON_DAY = {python_day: (python_day + 1) % 7 for python_day in range(7)}
_PYTHON_DAY = {cron_day: python_day for python_day, cron_day in _CRON_DAY.items()}
_RRULE_DAY = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")

_TIME = re.compile(r"^\s*(\d{1,2})(?:[:.](\d{2}))?\s*([ap])?\.?\s*m?\.?\s*$", re.IGNORECASE)

# Set while a scheduled routine runs, so the routine cannot schedule more routines.
# A plain flag (not thread-local): a routine's worker subagents run in other threads.
_ROUTINE_RUNNING = threading.Event()


def routine_running() -> bool:
    return _ROUTINE_RUNNING.is_set()


def utc_text(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime(UTC_FORMAT)


def parse_utc(text: str) -> datetime:
    return datetime.strptime(text, UTC_FORMAT).replace(tzinfo=timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def get_zone(name: str | None) -> ZoneInfo:
    name = (name or DEFAULT_TIMEZONE).strip()
    if name.upper() in {"IST", "INDIA"}:
        name = DEFAULT_TIMEZONE
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"I don't know the timezone {name!r}. Use a name like Asia/Kolkata or America/New_York.") from exc


def parse_time_of_day(text: str) -> time:
    """'09:00', '9', '9am', '9:30 PM', '21.15' -> a time. ValueError with a plain message otherwise."""

    match = _TIME.match(text or "")
    if not match:
        raise ValueError(f"I couldn't read the time {text!r}. Say it like 09:00 or 9:30 PM.")
    hour, minute, half = int(match.group(1)), int(match.group(2) or 0), (match.group(3) or "").lower()
    if half:
        if not 1 <= hour <= 12:
            raise ValueError(f"I couldn't read the time {text!r}. Say it like 09:00 or 9:30 PM.")
        hour = hour % 12 + (12 if half == "p" else 0)
    if hour > 23 or minute > 59:
        raise ValueError(f"I couldn't read the time {text!r}. Say it like 09:00 or 9:30 PM.")
    return time(hour, minute)


def parse_day_of_week(text: str) -> int:
    """'Monday' / 'mon' -> 0 (Python weekday)."""

    word = (text or "").strip().lower()
    for index, name in enumerate(DAY_NAMES):
        if word and name.startswith(word[:3]) and len(word) >= 3:
            return index
    raise ValueError(f"I couldn't read the day {text!r}. Use a weekday name like Monday.")


def cron_for(frequency: str, at: time, weekday: int | None = None) -> str | None:
    if frequency == "daily":
        return f"{at.minute} {at.hour} * * *"
    if frequency == "weekdays":
        return f"{at.minute} {at.hour} * * 1-5"
    if frequency == "weekly":
        return f"{at.minute} {at.hour} * * {_CRON_DAY[weekday]}"
    return None


def rrule_for(frequency: str, at: time, weekday: int | None = None) -> str | None:
    by_time = f"BYHOUR={at.hour};BYMINUTE={at.minute}"
    if frequency == "daily":
        return f"FREQ=DAILY;{by_time}"
    if frequency == "weekdays":
        return f"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;{by_time}"
    if frequency == "weekly":
        return f"FREQ=WEEKLY;BYDAY={_RRULE_DAY[weekday]};{by_time}"
    return None


def parse_cron(cron_expr: str) -> tuple[time, set[int]]:
    """A cron_expr written by cron_for -> (time of day, allowed Python weekdays)."""

    minute, hour, _, _, days = cron_expr.split()
    if days == "*":
        allowed = set(range(7))
    elif "-" in days:
        first, last = (int(part) for part in days.split("-"))
        allowed = {_PYTHON_DAY[day % 7] for day in range(first, last + 1)}
    else:
        allowed = {_PYTHON_DAY[int(day) % 7] for day in days.split(",")}
    return time(int(hour), int(minute)), allowed


def next_occurrence(cron_expr: str, zone: ZoneInfo, after: datetime) -> datetime:
    """The first time strictly after `after` (aware) that matches cron_expr in `zone`, in UTC."""

    at, allowed = parse_cron(cron_expr)
    local_day = after.astimezone(zone).date()
    for offset in range(8):
        day = local_day + timedelta(days=offset)
        if day.weekday() not in allowed:
            continue
        # A wall-clock time skipped by a daylight-saving jump (02:30 on the spring-forward
        # night) resolves to the same instant an hour later; a repeated one to its first pass.
        candidate = datetime.combine(day, at, tzinfo=zone).astimezone(timezone.utc)
        if candidate > after:
            return candidate
    raise ValueError(f"No run time found for {cron_expr!r}.")  # unreachable for valid expressions


def tz_label(zone_name: str) -> str:
    return "IST" if zone_name == DEFAULT_TIMEZONE else zone_name


def describe_schedule(job: dict) -> str:
    """'daily at 09:00 AM IST', 'every weekday at ...', 'every Monday at ...', 'once on Mon 12 Oct at ...'."""

    zone_name = job.get("timezone") or DEFAULT_TIMEZONE
    zone = get_zone(zone_name)
    label = tz_label(zone_name)
    if job.get("cron_expr"):
        at, allowed = parse_cron(job["cron_expr"])
        clock = at.strftime("%I:%M %p")
        if job["frequency"] == "daily":
            return f"daily at {clock} {label}"
        if job["frequency"] == "weekdays":
            return f"every weekday at {clock} {label}"
        day = DAY_NAMES[min(allowed)].capitalize()
        return f"every {day} at {clock} {label}"
    local = parse_utc(job["next_run_at"]).astimezone(zone)
    return f"once on {local.strftime('%a %d %b')} at {local.strftime('%I:%M %p')} {label}"


def next_run_text(job: dict) -> str:
    """When the job runs next, in its own timezone ('Mon 12 Oct, 09:00 AM IST')."""

    zone_name = job.get("timezone") or DEFAULT_TIMEZONE
    local = parse_utc(job["next_run_at"]).astimezone(get_zone(zone_name))
    return f"{local.strftime('%a %d %b, %I:%M %p')} {tz_label(zone_name)}"


def _clean(text: str, limit: int, what: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        raise ValueError(f"The routine needs a {what}.")
    if len(text) > limit:
        raise ValueError(f"The {what} is too long (at most {limit} characters).")
    return text


def create_job(
    title: str,
    query: str,
    time_of_day: str,
    frequency: str,
    timezone_name: str = DEFAULT_TIMEZONE,
    day_of_week: str = "",
    on_date: str = "",
    *,
    store: AgentStateStore | None = None,
    now: datetime | None = None,
) -> dict:
    """Validate and store a routine; returns the stored job. ValueError (plain words) if invalid."""

    store = store or AgentStateStore()
    now = now or now_utc()
    title = _clean(title, MAX_TITLE_CHARS, "title")
    query = _clean(query, MAX_QUERY_CHARS, "description of what to do")
    frequency = (frequency or "").strip().lower().replace("-", "_").replace(" ", "_")
    frequency = {"once": "one_off", "oneoff": "one_off", "weekday": "weekdays", "every_day": "daily"}.get(frequency, frequency)
    if frequency not in FREQUENCIES:
        raise ValueError("How often should it run? I can do daily, every weekday, weekly on one day, or once.")

    zone = get_zone(timezone_name)
    zone_name = zone.key
    at = parse_time_of_day(time_of_day)

    if len(store.scheduled_jobs()) >= MAX_ACTIVE_JOBS:
        raise ValueError(f"You already have {MAX_ACTIVE_JOBS} routines. Cancel one first (/schedules).")

    weekday = None
    if frequency == "weekly":
        weekday = parse_day_of_week(day_of_week)
    if frequency == "one_off":
        try:
            day = date.fromisoformat((on_date or "").strip()) if on_date else now.astimezone(zone).date()
        except ValueError as exc:
            raise ValueError(f"I couldn't read the date {on_date!r}. Use YYYY-MM-DD.") from exc
        moment = datetime.combine(day, at, tzinfo=zone).astimezone(timezone.utc)
        if moment <= now:
            if on_date:
                raise ValueError("That time has already passed. Pick a time in the future.")
            moment = datetime.combine(day + timedelta(days=1), at, tzinfo=zone).astimezone(timezone.utc)
        if moment > now + timedelta(days=MAX_DAYS_AHEAD):
            raise ValueError("That is too far ahead. Pick a date within a year.")
        cron_expr = None
        next_run = moment
    else:
        cron_expr = cron_for(frequency, at, weekday)
        next_run = next_occurrence(cron_expr, zone, now)

    job = {
        "id": f"job_{uuid.uuid4().hex[:10]}",
        "title": title,
        "prompt_query": query,
        "cron_expr": cron_expr,
        "rrule": rrule_for(frequency, at, weekday),
        "timezone": zone_name,
        "next_run_at": utc_text(next_run),
        "frequency": frequency,
    }
    store.add_scheduled_job(job)
    return store.get_scheduled_job(job["id"])


def following_run(job: dict, after: datetime) -> str | None:
    """The run after this one (UTC text), or None for a one-off job."""

    if not job.get("cron_expr"):
        return None
    return utc_text(next_occurrence(job["cron_expr"], get_zone(job.get("timezone")), after))
