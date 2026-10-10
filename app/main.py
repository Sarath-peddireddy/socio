"""
Run Socio, the autonomous email and calendar agent.

    python -m app.main                       # process new inbox emails (live)
    python -m app.main --dry-run             # same, but never send/create anything
    python -m app.main --query "subject:test" --max 5
    python -m app.main --status              # show stored outcomes, approvals, failures
    python -m app.main --resolve 3           # close pending approval #3
    python -m app.main --ask "trash the Swiggy email"   # direct instruction (owner mode)
    python -m app.main --google-login        # one-time sign-in for Drive, Docs, Sheets, Tasks
    python -m app.main --approve 4           # approve and execute request #4
    python -m app.main --reject 4            # reject request #4 (nothing changes)

Output shows sender, subject, and decisions only, never email bodies.
"""

import argparse
import logging
import sys

from app.agent.approvals import approve, describe, reject
from app.agent.followups import complete_followup_task
from app.agent.runner import DEFAULT_QUERY, handle_owner_command, process_inbox
from app.agent.state import AgentStateStore


def _short(text: str, width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def print_outcomes(outcomes) -> None:
    if not outcomes:
        print("No matching emails.")
        return

    skipped = [o for o in outcomes if o.status == "skipped_already_handled"]
    if skipped:
        print(f"Skipped {len(skipped)} email(s) already handled in earlier runs.")

    for outcome in outcomes:
        if outcome in skipped:
            continue
        print(f"\n[{outcome.status}] {_short(outcome.subject, 70)}")
        print(f"  from: {outcome.sender}   message_id: {outcome.message_id}")
        if outcome.intent:
            print(f"  intent: {outcome.intent}")
        if outcome.reason:
            print(f"  reason: {_short(outcome.reason, 160)}")
        for action in outcome.actions:
            print(f"  action: {action['type']} -> {action['status']}")
        if outcome.summary:
            print(f"  agent summary: {_short(outcome.summary, 300)}")


def print_status() -> None:
    store = AgentStateStore()
    print("Message outcomes:", store.status_counts() or "none yet")

    interrupted = store.interrupted_messages()
    if interrupted:
        print(f"Interrupted (will be retried): {len(interrupted)}")

    approvals = store.pending_approvals()
    print(f"\nOpen pending approvals: {len(approvals)}")
    for approval in approvals:
        print(f"\n  #{approval['id']} {approval['type']}")
        for line in describe(store.get_approval(approval["id"])).splitlines():
            print(f"     {line}")
    if approvals:
        print("\n  Approve with --approve ID, reject with --reject ID.")

    failures = store.failures()[:5]
    print(f"\nRecent failures: {len(failures)} shown")
    for failure in failures:
        print(f"  {failure['created_at']} {failure['source']}: {_short(failure['message'], 140)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Socio - autonomous email and calendar agent")
    parser.add_argument("--query", default=DEFAULT_QUERY, help=f"Gmail search query (default: {DEFAULT_QUERY!r})")
    parser.add_argument("--max", type=int, default=10, help="Maximum number of messages to list")
    parser.add_argument("--dry-run", action="store_true", help="Decide and plan, but do not send, draft, or create events")
    parser.add_argument("--status", action="store_true", help="Show stored state and exit")
    parser.add_argument("--resolve", type=int, metavar="ID", help="Close pending approval #ID and exit")
    parser.add_argument("--approve", type=int, metavar="ID", help="Approve and execute pending approval #ID")
    parser.add_argument("--reject", type=int, metavar="ID", help="Reject pending approval #ID (nothing changes)")
    parser.add_argument("--ask", metavar="TEXT", help="Give the agent a direct instruction (owner mode)")
    parser.add_argument("--google-login", "--drive-login", dest="google_login", action="store_true",
                        help="Sign in to Google Drive, Docs, Sheets and Tasks, and list the Socio folder")
    args = parser.parse_args(argv)

    # Windows consoles default to cp1252, which cannot print emoji in subjects.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.status:
        print_status()
        return 0

    if args.resolve is not None:
        store = AgentStateStore()
        approval = store.get_approval(args.resolve)
        closed = store.resolve_pending_approval(args.resolve, "Resolved by owner")
        if closed and approval:
            complete_followup_task(store, approval.get("message_id"))
        print(f"Approval #{args.resolve} {'closed' if closed else 'not found or already closed'}.")
        return 0 if closed else 1

    if args.google_login:
        from app.drive.client import get_drive_service
        from app.files.drive import drive_configured, list_drive_documents
        from app.files.local import ALLOWED_EXTENSIONS

        get_drive_service(allow_browser_login=True)
        print("Google Drive, Docs, Sheets and Tasks connected.")
        if not drive_configured():
            print("Set AGENT_DRIVE_FOLDER=<folder name> in .env to choose the documents folder.")
            return 1
        documents = list_drive_documents(ALLOWED_EXTENSIONS)
        print(f"The agent can see {len(documents)} file(s):")
        for document in documents:
            print(f"  {document['folder'] + '/' if document['folder'] else ''}{document['name']}")
        return 0

    if args.approve is not None or args.reject is not None:
        outcome = approve(args.approve) if args.approve is not None else reject(args.reject)
        print(outcome["message"])
        return 0 if outcome["ok"] else 1

    if args.ask:
        result = handle_owner_command(args.ask, dry_run=args.dry_run)
        print(result.reply)
        for approval_id in result.approval_ids:
            print(f"\nWaiting for your approval (#{approval_id}):")
            print(describe(AgentStateStore().get_approval(approval_id)))
        return 0

    print(f"{'DRY RUN - ' if args.dry_run else ''}Processing: {args.query!r} (max {args.max})")
    outcomes = process_inbox(query=args.query, max_messages=args.max, dry_run=args.dry_run)
    print_outcomes(outcomes)

    return 1 if any(o.status == "failed" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
