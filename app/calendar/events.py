"""
Read, move, and cancel existing Google Calendar events.

move_event / cancel_event change the owner's calendar and notify attendees,
so they are only called from approved requests (app.agent.approvals).
"""

from datetime import datetime, timedelta, timezone

from app.calendar.client import get_calendar_service
from app.google_api import execute_with_retry


INDIA_TZ = timezone(timedelta(hours=5, minutes=30))


def parse_ist(value: str) -> datetime:
    """Parse an ISO 8601 time; naive times are interpreted as IST. Returns IST-aware."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid datetime {value!r}. Use ISO 8601, e.g. 2026-10-16T11:00:00.") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=INDIA_TZ)

    return parsed.astimezone(INDIA_TZ)


def meet_link(event: dict) -> str:
    """The event's Google Meet (video call) link, or "" when it has none."""

    if not isinstance(event, dict):
        return ""
    if isinstance(event.get("hangoutLink"), str) and event["hangoutLink"]:
        return event["hangoutLink"]
    for point in (event.get("conferenceData") or {}).get("entryPoints", []):
        if point.get("entryPointType") == "video" and point.get("uri"):
            return point["uri"]
    return ""


def summarize_event(event: dict) -> dict:
    """Compact, LLM-friendly view of an event."""

    attendees = event.get("attendees", [])

    return {
        "event_id": event.get("id"),
        "title": event.get("summary", "Untitled event"),
        "start": event.get("start", {}).get("dateTime") or event.get("start", {}).get("date"),
        "end": event.get("end", {}).get("dateTime") or event.get("end", {}).get("date"),
        "status": event.get("status"),
        # The only meeting link Socio may give out for this event ("" = it has no video call).
        "meet_link": meet_link(event),
        # Attendees other than the owner ("self" is set by Google for the owner).
        "other_attendees": [a.get("email", "").lower() for a in attendees if not a.get("self") and a.get("email")],
        "organizer_is_owner": bool(event.get("organizer", {}).get("self")),
    }


def get_event(event_id: str) -> dict:
    return execute_with_retry(
        get_calendar_service().events().get(calendarId="primary", eventId=event_id),
        operation_name="calendar_get_event",
    )


def list_events(start: datetime, end: datetime, query: str = "") -> list[dict]:
    request_args = {
        "calendarId": "primary",
        "timeMin": start.isoformat(),
        "timeMax": end.isoformat(),
        "singleEvents": True,
        "orderBy": "startTime",
        "maxResults": 50,
    }
    if query:
        request_args["q"] = query

    return execute_with_retry(
        get_calendar_service().events().list(**request_args),
        operation_name="calendar_list_events",
    ).get("items", [])


def find_conflicts(start: datetime, end: datetime, ignore_event_id: str | None = None) -> list[dict]:
    """Timed events overlapping [start, end), excluding ignore_event_id. All-day events are ignored."""

    conflicts = []

    for event in list_events(start, end):
        if event.get("id") == ignore_event_id or event.get("status") == "cancelled":
            continue

        event_start = event.get("start", {}).get("dateTime")
        event_end = event.get("end", {}).get("dateTime")
        if not event_start or not event_end:
            continue

        if parse_ist(event_start) < end and parse_ist(event_end) > start:
            conflicts.append(summarize_event(event))

    return conflicts


def validate_new_time(start: datetime, end: datetime, ignore_event_id: str | None = None) -> None:
    """Raise ValueError unless [start, end) is in the future, well-formed, and free."""

    if start <= datetime.now(INDIA_TZ):
        raise ValueError(f"The new start time {start.isoformat()} is in the past.")

    if end <= start:
        raise ValueError("The new end time must be after the start time.")

    conflicts = find_conflicts(start, end, ignore_event_id=ignore_event_id)
    if conflicts:
        titles = ", ".join(f"'{c['title']}' ({c['start']})" for c in conflicts)
        raise ValueError(f"The new time overlaps with: {titles}.")


def move_event(event_id: str, new_start: datetime, new_end: datetime) -> dict:
    """Move an event (attendees are notified). Re-validates the slot first."""

    event = get_event(event_id)
    if event.get("status") == "cancelled":
        raise ValueError("That event has already been cancelled.")

    validate_new_time(new_start, new_end, ignore_event_id=event_id)

    # Not retried: a timed-out patch may still have been applied.
    updated = execute_with_retry(
        get_calendar_service().events().patch(
            calendarId="primary",
            eventId=event_id,
            body={
                "start": {"dateTime": new_start.isoformat(), "timeZone": "Asia/Kolkata"},
                "end": {"dateTime": new_end.isoformat(), "timeZone": "Asia/Kolkata"},
            },
            sendUpdates="all",
        ),
        operation_name="calendar_move_event",
        retry=False,
    )

    return summarize_event(updated)


def cancel_event(event_id: str) -> dict:
    """Cancel (delete) an event and notify attendees."""

    event = get_event(event_id)
    if event.get("status") == "cancelled":
        raise ValueError("That event has already been cancelled.")

    execute_with_retry(
        get_calendar_service().events().delete(calendarId="primary", eventId=event_id, sendUpdates="all"),
        operation_name="calendar_cancel_event",
        retry=False,
    )

    return {**summarize_event(event), "status": "cancelled"}
