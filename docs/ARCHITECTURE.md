# Socio architecture

## Principles

1. **Deterministic where possible, LLM where needed.** Fetching, rule-based
   triage, verification, and bookkeeping are plain Python. The LLM is used for
   judgement: classifying intent, choosing tools, writing emails.
2. **Two trust levels.** Instructions from the owner (Telegram / CLI) are
   trusted. Email content is untrusted data and gets a restricted toolset.
   Every prompt carries email content only inside `<untrusted_email_body>`
   tags (`app/agent/untrusted.py`).
3. **Nothing reaches other people without the owner.** In email mode every
   reply (with or without files) is held as an approval; booking, moving, or
   cancelling events and trashing email are proposals too. Only the owner's
   approval executes them. The agent has no direct booking tool at all
   (`propose_book_meeting` only). In owner mode, emails the owner asks for are
   sent directly.
4. **Never claim what didn't happen.** An action counts as done only when the
   API result confirms it; sends are re-checked in the thread.

## Components

```
app/telegram_bot.py      owner-only Telegram front end, approval buttons, scheduled inbox checks
app/main.py              CLI front end
        │
app/agent/runner.py      process_inbox / handle_message (email mode), handle_owner_command (owner mode)
        │
app/agent/decision.py    metadata rules → keyword pre-check → Gemini structured decision (+ intent)
app/agent/graph.py       LangGraph: agent node ⇄ tools node; mode-specific toolsets and guards
app/agent/prompts.py     system prompts and per-intent playbooks
app/agent/approvals.py   execute / reject approval requests (re-validated at execution time)
app/agent/state.py       SQLite: message outcomes, attempts, actions, approvals, slot holds, failures
app/agent/untrusted.py   sanitize + wrap email content in <untrusted_email_body> tags
        │
app/gmail/               read, search, reply, send, draft, trash (never permanent delete)
app/calendar/            availability, create, find, move, cancel; holds.py: local soft holds
app/drive/, app/files/   document search: owner's Drive (owned files) + local shareable folder
app/workspace/           shared Workspace sign-in; Google Docs and Sheets (append-only)
app/tasks/               Google Tasks; app/agent/followups.py adds tasks when the owner is needed
app/google_auth.py       shared OAuth credential loading
app/google_api.py        shared timeouts, bounded retries, token refresh
```

## Email mode (per incoming email)

```
fetch thread ─► decide ─► no action / low confidence ─► record
                   │
                   └─► agent ⇄ tools ─► verify ─► record outcome
```

Guards in the tools node:

- every `send_gmail_reply` becomes a `send_reply` / `send_reply_with_attachment`
  approval; the recipient, thread, and message come from the runner, not the model
- only the conversation being handled can be replied to, only to its participants
- calendar invites only to conversation participants
- an event can only be moved/cancelled on request of one of its attendees
- one reply / draft per email, even across restarts
- once a reply is held in an approval (reply, booking, move), no other reply or
  draft can be made for that email, including by a restarted run
- a booking/move whose time overlaps another email's held slot is refused
- no Trash, no new emails, no mailbox search

Untrusted content: `untrusted.py` strips HTML (scripts, styles, comments,
tags), invisible and control characters, then escapes `<`/`>` so an email
cannot close the `<untrusted_email_body>` tag early; headers become escaped
attributes. Used by the runner's task prompt, the triage prompt, and the
Gmail read tools (`get_thread`, `get_email`, `search_gmail`,
`get_recent_emails`). This lowers injection risk; the hard limits are the
toolset and the approvals above.

## Owner mode (Telegram / `--ask`)

Full toolset (search mail, send new email, calendar, documents, Google Docs,
Sheets, and Tasks) plus `propose_trash_email`. Bookings, event changes, and
trashing still go through approvals, so the owner sees the exact parsed time
before anything is sent to attendees.

## Google Workspace

Drive, Docs, Sheets, and Tasks share one OAuth token (`workspace_token.json`,
`python -m app.main --google-login`); Gmail and Calendar keep their own.
Scopes: `drive.readonly`, `drive.file`, `documents`, `spreadsheets`, `tasks`.
Socio searches the owner's whole Drive by default (`'me' in owners`, so files
shared by others are never offered as the owner's); `AGENT_DRIVE_FOLDER`
optionally limits it to one folder. Files Socio creates go into a "Socio"
folder it creates and tags (`appProperties`), so it never writes into one of
the owner's folders. Edits are append-only; Sheets values are written with
`valueInputOption=RAW`.

Sharing files by email: in email mode a reply with any attachment (Drive or
local) becomes a `send_reply_with_attachment` approval (Telegram: Approve &
Send / Edit in Gmail / Reject).

Follow-up tasks: when an email ends as `needs_review`, `needs_attention`, or
`awaiting_approval`, a Google Task is created (once per email) and completed
when the owner approves, rejects, or resolves it. Failures never affect email handling.

## Approvals

`propose_*` tools (and, in email mode, `send_gmail_reply`) validate the request
and the graph records an approval (`pending_approvals` table). The Telegram bot
pushes each new request with Approve / Reject buttons (replies also get Edit in
Gmail). `approvals.approve()` claims the request atomically (so a double tap
cannot run it twice), re-validates (event exists, new slot still free),
executes, then sends the follow-up reply if one was attached.

| Type | Executes |
|---|---|
| `send_reply` | the reply in the thread |
| `send_reply_with_attachment` | the reply with its files |
| `book_meeting` | create the event (conflict re-checked, attendee invited), then the reply |
| `move_event` / `cancel_event` | patch / delete the event (attendees notified), then the reply |
| `trash_email` | move the message or thread to Trash |

`approvals.to_draft()` (Edit in Gmail, `--draft ID`) saves a held reply as a
Gmail draft instead (one attachment carried over; with several, the owner
attaches them) and marks the email `needs_review`. A failed execution changes
nothing further and marks the email `needs_attention`; a rejection marks it
`needs_review` for the owner.

Soft holds: when a `book_meeting` or `move_event` approval is recorded, its new
time goes into `slot_holds` for 24 hours (`SLOT_HOLD_HOURS`).
`find_calendar_availability` treats active holds as busy, and the graph refuses
another email's booking/move that overlaps one (an email's own hold never
blocks it, so a retried run still works). A hold is released as soon as its
approval is approved, rejected, failed, or resolved. After it
expires the approval may still be open; approving then still re-checks the
real calendar.

## Reliability

- Google APIs: 30 s socket timeout, up to 3 retries for timeouts, DNS errors,
  HTTP 429/5xx; non-idempotent calls (send, insert, patch, delete, draft) are
  never retried. Persistent outages raise `GoogleApiUnavailableError`, which
  stops a batch without counting against an email's retry budget.
- Gemini: 60 s timeout; per-minute rate limits are waited out using the
  server's suggested delay; daily quota / unreachable API raises
  `LLMUnavailableError`.
- Restart recovery: interrupted emails are retried from the database; actions
  already completed are shown to the agent and cannot be repeated.
- Concurrency: the Telegram bot is one sequential loop (messages, button taps,
  and scheduled inbox checks never overlap). Across processes (bot + CLI),
  SQLite runs in WAL mode with a 30 s busy timeout, and a live inbox run takes
  a 30-minute lease (`run_locks` table, renewed per email, released at the
  end or on a crash's expiry). A second live run returns `skipped_busy`; dry
  runs need no lock. Approvals are claimed atomically, so a double tap from
  Telegram and the CLI cannot execute twice.

## Outcome statuses

`replied`, `replied_unverified`, `no_action`, `needs_review`,
`needs_attention`, `awaiting_approval`, `superseded`, `failed` (retried),
`failed_permanently`. A run that found another one in progress reports
`skipped_busy` (not stored).
