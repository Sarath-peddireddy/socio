from datetime import datetime, timedelta, timezone

from app.calendar.client import get_calendar_service
from app.google_api import execute_with_retry


def get_calendar_events(
    start_time: datetime,
    end_time: datetime,
) -> list[dict]:
    """
    Retrieve calendar events between two timestamps.
    """

    service = get_calendar_service()

    response = execute_with_retry(
        service.events()
        .list(
            calendarId="primary",
            timeMin=start_time.isoformat(),
            timeMax=end_time.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ),
        operation_name="calendar_list_events",
    )

    return response.get("items", [])


def find_free_slots(
    start_time: datetime,
    end_time: datetime,
    duration_minutes: int = 30,
) -> list[dict]:
    """
    Find available time slots between start_time and end_time.

    This function only reads the calendar.
    It does not create or modify events.
    """

    events = get_calendar_events(
        start_time=start_time,
        end_time=end_time,
    )

    busy_periods = []

    for event in events:
        event_start = event.get("start", {}).get("dateTime")
        event_end = event.get("end", {}).get("dateTime")

        if not event_start or not event_end:
            continue

        busy_periods.append(
            (
                datetime.fromisoformat(
                    event_start.replace("Z", "+00:00")
                ),
                datetime.fromisoformat(
                    event_end.replace("Z", "+00:00")
                ),
            )
        )

    busy_periods.sort(
        key=lambda period: period[0]
    )

    free_slots = []

    current_time = start_time

    for busy_start, busy_end in busy_periods:

        if current_time + timedelta(
            minutes=duration_minutes
        ) <= busy_start:

            free_slots.append(
                {
                    "start": current_time,
                    "end": current_time
                    + timedelta(minutes=duration_minutes),
                }
            )

        if busy_end > current_time:
            current_time = busy_end

    if current_time + timedelta(
        minutes=duration_minutes
    ) <= end_time:

        free_slots.append(
            {
                "start": current_time,
                "end": current_time
                + timedelta(minutes=duration_minutes),
            }
        )

    return free_slots