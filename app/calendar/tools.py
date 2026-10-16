import re
from datetime import date, datetime, time, timedelta, timezone
from uuid import uuid4


from langchain_core.tools import tool

from app.calendar.availability import find_free_slots
from app.calendar.client import get_calendar_service
from app.calendar.events import meet_link
from app.calendar.holds import held_periods
from app.google_api import execute_with_retry


# Where a booking's confirmation email puts the Google Meet link that Calendar creates
# when the owner approves (the model never writes a meeting link itself).
MEET_LINK_PLACEHOLDER = "[meeting link]"
_SIGN_OFF = re.compile(
    r"(?im)^(best|best regards|regards|kind regards|warm regards|thanks|thank you|many thanks|cheers|sincerely)[,.!]?[ \t]*$"
)


def with_meet_placeholder(body: str) -> str:
    """The reply with a line for the Meet link (before the sign-off) unless it already has one."""

    if not body.strip() or MEET_LINK_PLACEHOLDER in body:
        return body
    line = f"Google Meet link: {MEET_LINK_PLACEHOLDER}"
    sign_offs = list(_SIGN_OFF.finditer(body))
    if sign_offs:
        at = sign_offs[-1].start()
        return f"{body[:at].rstrip()}\n\n{line}\n\n{body[at:]}"
    return f"{body.rstrip()}\n\n{line}"


# Clock times in text: "3 PM", "3:00 pm", "9.30 a.m.", "15:00", "at 3". Each mention gives the
# (hour, minute) readings it may mean ("3:00" without AM/PM could be 03:00 or 15:00).
_CLOCK = re.compile(
    r"(?i)\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)(?![a-z])"   # 3 pm, 3:30 pm, 9.30 a.m.
    r"|\b(\d{1,2})[:.](\d{2})\b(?![:.]\d)"                                    # 15:00, 3:30 (not 15:00:00 / 16.10.2026)
    r"|\bat\s+(\d{1,2})\b(?![:.\d])"                                          # at 3
)
# Machine timestamps ("2026-10-16T09:30:00+05:30", "+0000") are not times anyone wrote as such.
_MACHINE_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ][\d:.]+(?:Z|[+-]\d{2}:?\d{2})?|[+-]\d{2}:?\d{2}\b")


def clock_mentions(text: str) -> list[set[tuple[int, int]]]:
    """Every clock time written in text, each as the set of (hour, minute) it could mean."""

    mentions = []
    for match in _CLOCK.finditer(_MACHINE_TIME.sub(" ", text or "")):
        if match.group(3):
            hour, minute = int(match.group(1)), int(match.group(2) or 0)
            if not (1 <= hour <= 12 and minute < 60):
                continue
            pm = match.group(3).lower().startswith("p")
            mentions.append({(hour % 12 + (12 if pm else 0), minute)})
        elif match.group(4):
            hour, minute = int(match.group(4)), int(match.group(5))
            if hour > 23 or minute > 59:
                continue
            mentions.append({(hour, minute), (hour + 12, minute)} if 1 <= hour < 12 else {(hour, minute)})
        else:
            hour = int(match.group(6))
            if 1 <= hour <= 12:
                mentions.append({(hour % 12, 0), (hour % 12 + 12, 0)})
    return mentions


def in_working_hours(start: datetime, end: datetime) -> bool:
    """Whether [start, end) lies on one weekday inside the 11:00-16:00 IST window."""

    start, end = start.astimezone(INDIA_TZ), end.astimezone(INDIA_TZ)
    window_start = datetime.combine(start.date(), time(WORK_START_HOUR), tzinfo=INDIA_TZ)
    window_end = datetime.combine(start.date(), time(WORK_END_HOUR), tzinfo=INDIA_TZ)
    return start.weekday() < 5 and window_start <= start and end <= window_end


def check_reply_times(reply_body: str, *slots: datetime) -> None:
    """
    ValueError when the reply names a clock time that is none of the given slot times (IST). A
    booking once said "3:00 PM IST" in the email while the slot was 9:30 AM IST.
    """

    allowed = {(moment.astimezone(INDIA_TZ).hour, moment.astimezone(INDIA_TZ).minute) for moment in slots if moment}
    for mention in clock_mentions(reply_body):
        if not mention & allowed:
            times = " / ".join(sorted({f"{moment.astimezone(INDIA_TZ):%H:%M}" for moment in slots if moment}))
            raise ValueError(
                f"The reply names a time that is not the meeting's ({times} IST). Use start_time / end_time "
                "exactly as the time the reply states (IST, e.g. 2026-10-16T15:00:00+05:30), or fix the reply. "
                "Never convert times to UTC."
            )


def fill_meet_link(body: str, link: str) -> str:
    """The placeholder replaced by the real link; with no link, the lines that mention it are dropped."""

    if MEET_LINK_PLACEHOLDER not in body:
        return body
    if link:
        return body.replace(MEET_LINK_PLACEHOLDER, link)
    kept = "\n".join(line for line in body.split("\n") if MEET_LINK_PLACEHOLDER not in line)
    return re.sub(r"\n{3,}", "\n\n", kept).strip()


# All interview times are interpreted in India time. A fixed offset is used
# instead of zoneinfo to avoid a tzdata dependency on Windows (India has no DST).
INDIA_TZ = timezone(timedelta(hours=5, minutes=30))


# Socio only offers weekday slots inside this window (India time).
WORK_START_HOUR, WORK_END_HOUR = 11, 16
MAX_DAYS_AHEAD = 60


def _day_label(day: date) -> str:
    return f"{day:%A} {day.day} {day:%b %Y}"


def _next_weekday(day: date) -> date:
    day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _parse_day(value: str, today: date) -> date:
    """'2026-10-12' (or an ISO datetime) -> that date; ValueError with a plain reason otherwise."""

    try:
        day = date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise ValueError(f"'{value}' is not a date. Use YYYY-MM-DD, e.g. {_next_weekday(today).isoformat()}.") from exc
    if day < today:
        raise ValueError(f"{_day_label(day)} has already passed.")
    if day > today + timedelta(days=MAX_DAYS_AHEAD):
        raise ValueError(f"I can only check up to {MAX_DAYS_AHEAD} days ahead.")
    return day


def _check_day(day: date, now: datetime, duration_minutes: int, held: list) -> dict:
    """Free slots on one day, or a note saying why there are none."""

    entry: dict = {"date": day.isoformat(), "day": _day_label(day), "slots": [], "free": []}
    if day.weekday() >= 5:
        entry["note"] = f"{_day_label(day)} is a weekend day; I only offer weekdays."
        return entry

    window_start = datetime.combine(day, time(WORK_START_HOUR), tzinfo=INDIA_TZ)
    window_end = datetime.combine(day, time(WORK_END_HOUR), tzinfo=INDIA_TZ)
    # Today: never offer a time that has started; begin at the next quarter hour.
    earliest = (now + timedelta(minutes=14)).replace(second=0, microsecond=0)
    earliest -= timedelta(minutes=earliest.minute % 15)
    search_start = max(window_start, earliest)
    if search_start + timedelta(minutes=duration_minutes) > window_end:
        entry["note"] = f"Today's {WORK_START_HOUR}:00-{WORK_END_HOUR}:00 window is over." if day == now.date() else "No time left."
        return entry

    for slot in find_free_slots(start_time=search_start, end_time=window_end, duration_minutes=duration_minutes, extra_busy=held):
        start, end = slot["start"], slot["end"]
        if start >= window_start and end <= window_end:
            gap_end = min(slot.get("window_end") or end, window_end)
            entry["slots"].append({"start": start.isoformat(), "end": end.isoformat()})
            entry["free"].append(f"{start:%H:%M}-{gap_end:%H:%M}")
    if not entry["slots"]:
        entry["note"] = (f"Fully booked between {WORK_START_HOUR}:00 and {WORK_END_HOUR}:00 "
                         f"(no free {duration_minutes}-minute gap).")
    return entry


@tool
def find_calendar_availability(
    duration_minutes: int = 30,
    date: str = "",
    days_ahead: int = 7,
    at_time: str = "",
) -> dict:
    """
    Find free meeting slots in the owner's primary Google Calendar.

    Socio only offers weekdays, 11:00 AM-4:00 PM India time (UTC+05:30).
    - For one specific day ("tomorrow", "Friday", "12 Oct"), pass date="YYYY-MM-DD":
      exactly that day is checked.
    - Otherwise the next days_ahead days are checked, starting today (max 21).
    - Is ONE exact time free ("am I free at 2 PM?", "if I'm free at 2, book it")? Pass date
      and at_time="14:00" (24-hour IST): the answer is "requested" {"start", "end", "free":
      true/false, "why"}. Use it, not "slots", to answer about that time.

    Returns {"slots": [{"start", "end"}, ...] (ONLY the earliest start of each free gap, not
    every free time: a free day shows one slot at 11:00), "days": [{"date", "day", "free":
    ["11:00-16:00", ...], "note"?}] (the whole free windows), "note"?}.
    An empty "slots" always comes with a "note" saying why (weekend, day over, fully
    booked) and the next weekday to try. Slots already proposed to someone and waiting
    for the owner's approval count as busy.
    """

    duration_minutes = int(duration_minutes)
    longest = (WORK_END_HOUR - WORK_START_HOUR) * 60
    if not 5 <= duration_minutes <= longest:
        raise ValueError(f"A meeting must be 5 to {longest} minutes: I only offer {WORK_START_HOUR}:00-{WORK_END_HOUR}:00.")

    now = datetime.now(INDIA_TZ)
    today = now.date()
    held = [(hold["start"], hold["end"]) for hold in held_periods()]

    if str(at_time or "").strip() and not str(date or "").strip():
        raise ValueError("at_time needs date=YYYY-MM-DD: the day to check that time on.")
    if str(date or "").strip():
        days = [_parse_day(date, today)]
    else:
        days = [today + timedelta(days=offset) for offset in range(max(1, min(int(days_ahead), 21)))]

    checked = [_check_day(day, now, duration_minutes, held) for day in days]
    # Several days: list only the days with free time (weekends would just be noise), unless none has any.
    shown = [entry for entry in checked if entry["slots"]] or checked
    result: dict = {
        "slots": [slot for entry in checked for slot in entry["slots"]],
        "days": [{key: value for key, value in entry.items() if key != "slots"} for entry in shown],
    }
    if not result["slots"]:
        reasons = "; ".join(entry["note"] for entry in checked if entry.get("note")) or "No free time."
        nxt = _next_weekday(days[-1])
        result["note"] = f"{reasons} The next weekday to try is {_day_label(nxt)} (date={nxt.isoformat()})."
    if str(at_time or "").strip():
        result["requested"] = _check_time(days[0], at_time, duration_minutes, now, held)
    return result


_CLOCK_ARG = re.compile(r"(?i)^\s*(\d{1,2})(?:[:.](\d{2})(?::00)?)?\s*(a\.?m\.?|p\.?m\.?)?\s*$")  # "12:00:00" too


def _check_time(day: date, at_time: str, duration_minutes: int, now: datetime, held: list) -> dict:
    """
    Whether one exact time is free (live test C1, 2026-10-16: "If I'm free Mon 2 PM, book it" was
    answered from "slots", which only lists 11:00 for a free day: "you're not free at 2 PM", or
    a booking at 11:00). Outside the usual window is reported, not refused: the owner named it.
    """

    from app.calendar.events import find_conflicts

    match = _CLOCK_ARG.match(str(at_time))
    hour, minute = (int(match.group(1)), int(match.group(2) or 0)) if match else (99, 0)
    if match and match.group(3):
        if not 1 <= hour <= 12:
            hour = 99
        else:
            hour = hour % 12 + (12 if match.group(3).lower().startswith("p") else 0)
    if hour > 23 or minute > 59:
        raise ValueError(f"'{at_time}' is not a time. Use 24-hour HH:MM in IST, e.g. 14:00.")
    start = datetime.combine(day, time(hour, minute), tzinfo=INDIA_TZ)
    end = start + timedelta(minutes=duration_minutes)
    answer: dict = {"start": start.isoformat(), "end": end.isoformat()}
    if start <= now:
        return {**answer, "free": False, "why": "That time has already passed."}
    busy = [event.get("title") or "an event" for event in find_conflicts(start, end)]
    held_here = [(hold_start, hold_end) for hold_start, hold_end in held if hold_start < end and hold_end > start]
    if busy:
        answer.update(free=False, why=f"It overlaps {', '.join(busy)}.")
    elif held_here:
        answer.update(free=False, why="It is held for another meeting proposal waiting for the owner's approval.")
    else:
        answer.update(free=True, why="Nothing is booked then.")
    if day.weekday() >= 5 or not in_working_hours(start, end):
        answer["note"] = "Outside the usual weekdays 11:00-16:00 window: book it only because the owner named this time."
    return answer

@tool
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    description: str = "",
    attendee_email: str = "",
    meeting_link: str = "",
    attendee_emails: list[str] | None = None,
    video_call: bool = False,
) -> dict:
    """
    Create an interview or meeting event in the user's primary
    Google Calendar.

    video_call=True asks Google to add a Google Meet call; its link is
    returned as meet_link ("" if Google did not create one).

    attendee_emails (optional) invites several people; attendee_email is
    the single-person form and is merged with it.

    The requested time must be in the future and must not overlap
    with an existing calendar event.

    start_time and end_time must be ISO 8601 datetime strings.

    attendee_email is optional. If provided, the person is added
    as a Google Calendar attendee.

    meeting_link is optional. If provided, it is included
    in the event description.
    """

    try:
        start_dt = datetime.fromisoformat(
            start_time.replace("Z", "+00:00")
    )

        end_dt = datetime.fromisoformat(
            end_time.replace("Z", "+00:00")
    )

    except ValueError as exc:
        raise ValueError(
        "Invalid datetime format. "
        "Use ISO 8601 format."
        )   from exc

    india_tz = INDIA_TZ

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=india_tz)

    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=india_tz)

# Normalize both times to the user's local timezone.
    start_dt = start_dt.astimezone(india_tz)
    end_dt = end_dt.astimezone(india_tz)

    now = datetime.now().astimezone()

    if start_dt <= now:
        raise ValueError(
            "Cannot create a calendar event in the past. "
            f"Requested start: {start_time}. "
            f"Current time: {now.isoformat()}."
        )

    if end_dt <= start_dt:
        raise ValueError(
            "Event end time must be after start time."
        )

    # Check for existing calendar events that overlap
    # with the requested time.
    existing_events = execute_with_retry(
        get_calendar_service()
        .events()
        .list(
            calendarId="primary",
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ),
        operation_name="calendar_conflict_check",
    ).get("items", [])

    for existing_event in existing_events:
        existing_start = (
            existing_event
            .get("start", {})
            .get("dateTime")
        )

        existing_end = (
            existing_event
            .get("end", {})
            .get("dateTime")
        )

        # Ignore all-day events because they don't represent
        # a specific interview time slot.
        if not existing_start or not existing_end:
            continue

        existing_start_dt = datetime.fromisoformat(
            existing_start.replace("Z", "+00:00")
        )

        existing_end_dt = datetime.fromisoformat(
            existing_end.replace("Z", "+00:00")
        )

        # Overlap condition:
        # existing_start < requested_end
        # AND
        # existing_end > requested_start
        if (
            existing_start_dt < end_dt
            and existing_end_dt > start_dt
        ):
            existing_title = existing_event.get(
                "summary",
                "Untitled event",
            )

            raise ValueError(
                "Cannot create the calendar event because "
                "the requested time overlaps with an existing "
                f"calendar event: '{existing_title}'. "
                f"Existing event: "
                f"{existing_start} → {existing_end}."
            )

    service = get_calendar_service()

    event_description = description

    if meeting_link:
        if event_description:
            event_description += "\n\n"

        event_description += (
            f"Meeting link: {meeting_link}"
        )

    event = {
    "summary": title,
    "description": event_description,
    "start": {
        "dateTime": start_dt.isoformat(),
        "timeZone": "Asia/Kolkata",
    },
    "end": {
        "dateTime": end_dt.isoformat(),
        "timeZone": "Asia/Kolkata",
    },
}

    invitees = list(dict.fromkeys(
        address.strip().lower() for address in [attendee_email, *(attendee_emails or [])] if address and address.strip()
    ))
    if invitees:
        event["attendees"] = [{"email": address} for address in invitees]
    if video_call:
        event["conferenceData"] = {
            "createRequest": {"requestId": uuid4().hex, "conferenceSolutionKey": {"type": "hangoutsMeet"}}
        }

    # Not retried: a timed-out insert may still have created the event.
    created_event = execute_with_retry(
        service.events()
        .insert(
            calendarId="primary",
            body=event,
            sendUpdates="all" if invitees else "none",
            **({"conferenceDataVersion": 1} if video_call else {}),
        ),
        operation_name="calendar_create_event",
        retry=False,
    )

    link = meet_link(created_event)
    if video_call and not link and created_event.get("id"):
        # Google may finish setting up the call just after the insert answers: look once more.
        # The event exists either way, so a failed look only means no link in the reply.
        try:
            link = meet_link(execute_with_retry(
                service.events().get(calendarId="primary", eventId=created_event["id"]),
                operation_name="calendar_get_event",
            ))
        except Exception:
            link = ""

    return {
        "meet_link": link,
        "event_id": created_event.get("id"),
        "title": created_event.get("summary"),
        "start": created_event.get("start", {}).get("dateTime"),
        "end": created_event.get("end", {}).get("dateTime"),
        "attendee": attendee_email or (invitees[0] if invitees else ""),
        "attendees": invitees,
        "html_link": created_event.get("htmlLink"),
        "status": created_event.get("status"),
        # The insert response is Google's confirmation of the stored event.
        "verified": bool(created_event.get("id")) and created_event.get("status") == "confirmed",
    }