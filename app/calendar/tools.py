from datetime import datetime, timedelta, timezone


from langchain_core.tools import tool

from app.calendar.availability import find_free_slots
from app.calendar.client import get_calendar_service
from app.google_api import execute_with_retry


# All interview times are interpreted in India time. A fixed offset is used
# instead of zoneinfo to avoid a tzdata dependency on Windows (India has no DST).
INDIA_TZ = timezone(timedelta(hours=5, minutes=30))


@tool
def find_calendar_availability(
    duration_minutes: int = 30,
    days_ahead: int = 7,
) -> list[dict]:
    """
    Find available interview or meeting slots in the user's primary Google Calendar.

    Checks the next days_ahead days (default 7, max 21) and only returns
    weekday slots between 11:00 AM and 4:00 PM India time (UTC+05:30).
    Each free gap yields its earliest slot.
    """

    now = datetime.now(INDIA_TZ)
    days_ahead = max(1, min(int(days_ahead), 21))

    suitable_slots = []

    for day_offset in range(days_ahead):
        current_date = (
            now + timedelta(days=day_offset)
        ).date()

        # Monday = 0 ... Sunday = 6
        if current_date.weekday() >= 5:
            continue

        window_start = datetime.combine(
            current_date,
            datetime.min.time(),
        ).replace(
            hour=11,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=now.tzinfo,
        )

        window_end = datetime.combine(
            current_date,
            datetime.min.time(),
        ).replace(
            hour=16,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=now.tzinfo,
        )

        # If today's interview window has already passed,
        # skip today.
        if window_end <= now:
            continue

        # If we're currently inside today's interview window,
        # don't offer a slot that has already started.
        search_start = max(
            window_start,
            now,
        )

        raw_slots = find_free_slots(
            start_time=search_start,
            end_time=window_end,
            duration_minutes=duration_minutes,
        )

        for slot in raw_slots:
            start = slot["start"]
            end = slot["end"]

            if start >= window_start and end <= window_end:
                suitable_slots.append(
                    {
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                    }
                )

    return suitable_slots

@tool
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    description: str = "",
    attendee_email: str = "",
    meeting_link: str = "",
) -> dict:
    """
    Create an interview or meeting event in the user's primary
    Google Calendar.

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

    if attendee_email:
        event["attendees"] = [
            {
                "email": attendee_email,
            }
        ]

    # Not retried: a timed-out insert may still have created the event.
    created_event = execute_with_retry(
        service.events()
        .insert(
            calendarId="primary",
            body=event,
            sendUpdates="all" if attendee_email else "none",
        ),
        operation_name="calendar_create_event",
        retry=False,
    )

    return {
        "event_id": created_event.get("id"),
        "title": created_event.get("summary"),
        "start": created_event.get("start", {}).get("dateTime"),
        "end": created_event.get("end", {}).get("dateTime"),
        "attendee": attendee_email,
        "html_link": created_event.get("htmlLink"),
        "status": created_event.get("status"),
        # The insert response is Google's confirmation of the stored event.
        "verified": bool(created_event.get("id")) and created_event.get("status") == "confirmed",
    }