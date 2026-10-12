"""
Google Sheets: read, append rows, and create. Nothing is ever deleted or overwritten.

Values are written as RAW text (never USER_ENTERED), so content that came from
an email can never become a formula. Works on any Sheet the owner can access
(or only the AGENT_DRIVE_FOLDER folder if set); new spreadsheets go into
Socio's own "Socio" folder. Owner mode only.
"""

from app.files.drive import move_into_output_folder, resolve_drive_file
from app.google_api import completed_write, execute_with_retry
from app.workspace.client import get_workspace_service


SHEET_MIME = "application/vnd.google-apps.spreadsheet"
MAX_ROWS = 200


def _service():
    return get_workspace_service("sheets", "v4")


def _sheet_url(sheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"


def _quote(tab: str) -> str:
    return "'" + tab.replace("'", "''") + "'"


def _tab_names(spreadsheet_id: str) -> list[str]:
    meta = execute_with_retry(
        _service().spreadsheets().get(spreadsheetId=spreadsheet_id, fields="sheets.properties.title"),
        operation_name="sheets_get_meta",
    )
    return [sheet["properties"]["title"] for sheet in meta.get("sheets", [])]


def read_sheet(reference: str, tab: str = "") -> dict:
    """Rows of one tab (default: the first) of a spreadsheet."""

    file = resolve_drive_file(reference, SHEET_MIME)
    tabs = _tab_names(file["id"])
    tab = tab or (tabs[0] if tabs else "Sheet1")
    if tab not in tabs:
        raise ValueError(f"No tab named '{tab}'. Tabs: {', '.join(tabs)}")

    response = execute_with_retry(
        _service().spreadsheets().values().get(spreadsheetId=file["id"], range=f"{_quote(tab)}!A1:Z{MAX_ROWS}"),
        operation_name="sheets_read",
    )
    rows = response.get("values", [])

    return {
        "spreadsheet_id": file["id"],
        "title": file["name"],
        "url": _sheet_url(file["id"]),
        "tab": tab,
        "tabs": tabs,
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(rows) >= MAX_ROWS,
    }


def append_sheet_row(reference: str, values: list[str], tab: str = "") -> dict:
    """Append one row after the last row of a tab. Existing rows are never changed."""

    if not values:
        raise ValueError("The row is empty.")

    file = resolve_drive_file(reference, SHEET_MIME)
    tabs = _tab_names(file["id"])
    tab = tab or (tabs[0] if tabs else "Sheet1")
    if tab not in tabs:
        raise ValueError(f"No tab named '{tab}'. Tabs: {', '.join(tabs)}")

    response = execute_with_retry(
        _service().spreadsheets().values().append(
            spreadsheetId=file["id"],
            range=f"{_quote(tab)}!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [[str(value) for value in values]]},
        ),
        operation_name="sheets_append",
        retry=False,
    )

    return {
        "spreadsheet_id": file["id"],
        "title": file["name"],
        "url": _sheet_url(file["id"]),
        "updated_range": response.get("updates", {}).get("updatedRange"),
        "verified": bool(response.get("updates", {}).get("updatedRows")),
    }


def create_spreadsheet(title: str, headers: list[str] | None = None) -> dict:
    """Create a spreadsheet (optionally with a header row) in Socio's folder."""

    if not title.strip():
        raise ValueError("A spreadsheet needs a title.")

    service = _service()
    body: dict = {"properties": {"title": title.strip()}}
    if headers:
        body["sheets"] = [{"data": [{"rowData": [{"values": [
            {"userEnteredValue": {"stringValue": str(header)}} for header in headers
        ]}]}]}]

    # Not retried: a timed-out create may still have created the spreadsheet.
    spreadsheet = execute_with_retry(
        service.spreadsheets().create(body=body, fields="spreadsheetId"),
        operation_name="sheets_create",
        retry=False,
    )
    sheet_id = spreadsheet["spreadsheetId"]
    # It exists from here on: a failure below must not be reported as "nothing changed".
    completed_write(f"created the Google Sheet '{title.strip()}' ({_sheet_url(sheet_id)})")

    move_into_output_folder(sheet_id)
    return {"spreadsheet_id": sheet_id, "title": title.strip(), "url": _sheet_url(sheet_id), "verified": True}
