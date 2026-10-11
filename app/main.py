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
    python -m app.main --draft 4             # move held email #4 to Gmail Drafts (not sent)
    python -m app.main --remember "no calls before noon"   # teach a preference (long-term memory)
    python -m app.main --memories            # list what Socio has learned
    python -m app.main --forget 7            # forget memory #7
    python -m app.main --skills              # list saved skills (./skills)
    python -m app.main --save-skill NAME     # save the last successful multi-step instruction as a skill

Output shows sender, subject, and decisions only, never email bodies.
Warnings and full error details are also written to logs/agent.log.
"""

import argparse
import sys

from app.agent.approvals import approve, describe, reject, to_draft
from app.agent.followups import complete_followup_task
from app.agent.runner import DEFAULT_QUERY, handle_owner_command, process_inbox
from app.agent.state import AgentStateStore
from app.errors import setup_logging


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
        print("\n  Approve with --approve ID, reject with --reject ID, or move an email to Drafts with --draft ID.")

    interrupted_runs = store.interrupted_supervisor_runs()
    if interrupted_runs:
        print(f"\nInterrupted instructions (not re-run): {', '.join('#' + str(r['id']) for r in interrupted_runs)}")

    stuck = store.stuck_approvals()
    if stuck:
        # Never retried automatically: the email / event change may already have happened.
        print("\nApprovals cut off while running (check Gmail / Calendar before doing them again):")
        for approval in stuck:
            print(f"  #{approval['id']} [{approval['type']}] {approval['target']} (tapped {approval['locked_at']} UTC)")

    failures = store.failures()[:5]
    print(f"\nRecent failures: {len(failures)} shown")
    for failure in failures:
        print(f"  {failure['created_at']} {failure['source']}: {_short(failure['message'], 140)}")


def _memory_command(args) -> int:
    from app.memory import skills
    from app.memory.learning import learn_from_owner_note
    from app.memory.semantic import SemanticMemory, memory_enabled

    if args.save_skill:
        from app.agent.supervisor import save_last_plan

        try:
            saved = save_last_plan(args.save_skill)
        except skills.SkillError as exc:
            print(exc)
            return 1
        print(f"Saved skill '{saved.name}' ({saved.tasks} steps; parameters: {', '.join(saved.parameters) or 'none'})"
              + ("" if saved.generalized else " exactly as it ran (Gemini could not generalize it)") + ".")
        return 0

    if args.skills:
        found = skills.list_skills()
        print("Saved skills:" if found else f"No skills in {skills.skills_dir()}.")
        for skill in found:
            kind = f"script {skill.script}" if skill.script else f"{len(skill.plan)} steps"
            print(f"  {skill.name} ({kind}, by {skill.author}): {skill.description}")
        return 0

    if not memory_enabled():
        print("Memory is turned off (AGENT_MEMORY=0).")
        return 1

    memory = SemanticMemory()
    if args.remember:
        for item in learn_from_owner_note(args.remember):
            print(f"{'Noted' if item['created'] else 'Updated'} [{item['id']}]: {item['entity']}: {item['preference']}")
        return 0
    if args.forget is not None:
        forgotten = memory.forget(args.forget)
        print(f"Memory #{args.forget} {'forgotten' if forgotten else 'not found'}.")
        return 0 if forgotten else 1

    memories = memory.all(limit=100)
    print(f"Memories: {len(memories)}" + ("" if memories else " (use --remember, or tell Socio 'remember: ...' in Telegram)"))
    for item in memories:
        print(f"  [{item.id}] {item.line()}  <{item.source}, x{item.hits}>")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Socio - autonomous email and calendar agent")
    parser.add_argument("--query", default=DEFAULT_QUERY, help=f"Gmail search query (default: {DEFAULT_QUERY!r})")
    parser.add_argument("--max", type=int, default=10, help="Maximum number of messages to list")
    parser.add_argument("--dry-run", action="store_true", help="Decide and plan, but do not send, draft, or create events")
    parser.add_argument("--status", action="store_true", help="Show stored state and exit")
    parser.add_argument("--resolve", type=int, metavar="ID", help="Close pending approval #ID and exit")
    parser.add_argument("--approve", type=int, metavar="ID", help="Approve and execute pending approval #ID")
    parser.add_argument("--reject", type=int, metavar="ID", help="Reject pending approval #ID (nothing changes)")
    parser.add_argument("--draft", type=int, metavar="ID", help="Move held email #ID to Gmail Drafts instead of sending it")
    parser.add_argument("--ask", metavar="TEXT", help="Give the agent a direct instruction (owner mode)")
    parser.add_argument("--remember", metavar="TEXT", help="Teach Socio a preference (long-term memory)")
    parser.add_argument("--memories", action="store_true", help="List what Socio has learned")
    parser.add_argument("--forget", type=int, metavar="ID", help="Forget memory #ID")
    parser.add_argument("--skills", action="store_true", help="List saved skills")
    parser.add_argument("--save-skill", metavar="NAME", help="Save the last successful multi-step instruction as skill NAME")
    parser.add_argument("--google-login", "--drive-login", dest="google_login", action="store_true",
                        help="Sign in to Google Drive, Docs, Sheets and Tasks, and list the Socio folder")
    args = parser.parse_args(argv)

    # Windows consoles default to cp1252, which cannot print emoji in subjects.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    setup_logging()

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
        from app.files.drive import folder_limited, list_drive_documents
        from app.files.local import ALLOWED_EXTENSIONS

        service = get_drive_service(allow_browser_login=True)
        user = service.about().get(fields="user(displayName, emailAddress)").execute()["user"]
        print(f"Google Drive, Docs, Sheets and Tasks connected for {user['emailAddress']}.")
        if folder_limited():
            documents = list_drive_documents(ALLOWED_EXTENSIONS)
            print(f"Limited to the AGENT_DRIVE_FOLDER folder: {len(documents)} file(s) visible.")
        else:
            print("Socio can search your whole Drive. Files Socio creates go into a 'Socio' folder.")
        return 0

    if args.approve is not None or args.reject is not None or args.draft is not None:
        if args.approve is not None:
            outcome = approve(args.approve)
        elif args.reject is not None:
            outcome = reject(args.reject)
        else:
            outcome = to_draft(args.draft)
        print(f"#{args.approve if args.approve is not None else args.reject if args.reject is not None else args.draft}: {outcome['message']}")
        if outcome.get("error"):
            print(f"  detail (local only): {outcome['error']}")
        return 0 if outcome["ok"] else 1

    if args.remember or args.memories or args.forget is not None or args.skills or args.save_skill:
        return _memory_command(args)

    if args.ask:
        result = handle_owner_command(args.ask, dry_run=args.dry_run)
        if result.reply:
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
