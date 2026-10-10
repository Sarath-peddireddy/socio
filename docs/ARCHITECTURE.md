# Socio architecture

## Principles

1. **Deterministic where possible, LLM where needed.** Fetching, rule-based
   triage, verification, and bookkeeping are plain Python. The LLM is used for
   judgement: classifying intent, choosing tools, writing emails.
2. **Two trust levels.** Instructions from the owner (Telegram / CLI) are
   trusted. Email content is untrusted data and gets a restricted toolset.
   Every prompt carries email content only inside `<untrusted_email_body>`
   tags (`app/agent/untrusted.py`).
3. **Nothing reaches other people, and no file changes, without the owner.**
   Every email (replies to incoming email, and in owner mode replies and new
   emails too), every Google Docs/Sheets write, booking, moving, or cancelling
   events, trashing email, and trusting a skill script is held as an approval.
   Only the owner's approval executes it. The agent has no direct booking tool
   at all (`propose_book_meeting` only). Google Tasks changes are direct.
4. **Never claim what didn't happen.** An action counts as done only when the
   API result confirms it; sends are re-checked in the thread.
5. **Memory informs, never authorizes.** Learned preferences and saved skills
   shape what the agent proposes; every gate above still applies.

## Components

```
app/telegram_bot.py      owner-only Telegram front end, approval buttons, scheduled inbox checks
app/main.py              CLI front end
        │
app/agent/runner.py      process_inbox / handle_message (email mode), handle_owner_command (owner mode)
app/agent/supervisor.py  owner instructions: planner -> task DAG -> parallel workers -> fan-in report
app/agent/workers.py     one worker subagent: clean context, restricted tools, strict JSON payload
app/agent/taskgraph.py   plan validation (ids, tools, dependencies, cycles) and ordering
        │
app/agent/decision.py    metadata rules → keyword pre-check → Gemini structured decision (+ intent)
app/agent/graph.py       LangGraph: agent node ⇄ tools node; mode-specific toolsets and guards
app/agent/prompts.py     system prompts and per-intent playbooks
app/agent/approvals.py   execute / reject approval requests (re-validated at execution time)
app/agent/state.py       SQLite: message outcomes, attempts, actions, approvals, slot holds, failures
app/agent/untrusted.py   sanitize + wrap email content in <untrusted_email_body> tags
app/memory/              semantic.py (long-term memory), embeddings.py (local fastembed),
                         learning.py (decisions -> memories), skills.py (procedural memory)
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
Sheets, Tasks, skill scripts) plus `propose_trash_email`. Emails
(`send_gmail_reply`, `send_new_email`), Docs/Sheets writes (`create_document`,
`append_to_document`, `create_spreadsheet`, `append_sheet_row`), bookings,
event changes, and trashing all become approvals (`OWNER_GATED_TOOLS` in
`graph.py`), validated first (recipient addresses, reply target, attachments
in the whitelist, the document must exist), so the card shows the exact email
or the exact lines / cells to be added (`+ ` lines). Tasks stay direct.

### Supervisor and workers (fan-out / fan-in)

`handle_owner_command` goes through `supervisor.handle_owner_request`:

```
instruction ─► recall memories (top 3) + matching skills
            ─► one-task request? ─► single owner agent (as before)
            └► planner (1 LLM call): spawn_subagent(task_id, goal_contract, context_slice,
               tools_allowed, depends_on) calls = a DAG, or use_skill(name, params)
            ─► validate (≤ 6 tasks, known tools, no cycles) ─► checkpoint (supervisor_runs, worker_tasks)
            ─► ThreadPoolExecutor (AGENT_MAX_WORKERS = 3): a task starts when its dependencies succeeded
            ─► fan-in payloads ─► close duplicate approvals ─► one report ─► maybe save a skill
```

- Only instructions that look multi-part (or match a skill) pay for a planning
  call; if the planner returns one task, the single agent runs instead.
- A worker is a fresh graph run (`worker=True`, `tools_allowed=[...]`): its
  only message is its goal, its context slice, and its dependencies' payloads.
  Tools outside its list are neither bound nor executed.
- Fan-in contract: `{"status": "SUCCESS"|"FAILED", "modified_entities": [...],
  "diff_summary": "..."}`. Entities come from the actions the tools recorded
  (the graph is streamed, so a worker that crashes still reports what it
  changed); the status also fails on any failed/unknown action.
- A failed task's dependents are not started; independent tasks continue; the
  run is `done`, `partial`, or `failed`. Exceptions never reach the bot loop.
- Overlaps: two workers proposing the same change (same event, document,
  recipient + subject, slot) get one approval; the other is closed as a
  duplicate. The slot-hold check and insert share one lock, so parallel
  workers cannot hold the same time twice.
- All workers share one Gemini rate limiter (`AGENT_LLM_RPM`, default 14/min)
  and token refresh is serialized (`CREDENTIALS_LOCK`); Google clients are
  built per call.
- Runs left `running` by a stopped process are marked `interrupted` after 30
  minutes and never re-run (they may already have created approvals).
- Inbox emails are not fanned out (developer decision): they stay one at a time.

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

Sharing files by email: any email with an attachment (Drive or local) is an
approval (`send_reply_with_attachment` / `send_email`; Telegram: Approve &
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
| `send_email` | a new email (owner mode) |
| `create_document` / `append_document` | create a Doc / append text (Socio folder) |
| `create_spreadsheet` / `append_sheet_row` | create a Sheet / add one row (values RAW) |
| `enable_skill_script` | trust `skills/<name>.py` at the SHA-256 shown (re-checked) |
| `book_meeting` | create the event (conflict re-checked, attendee invited), then the reply |
| `move_event` / `cancel_event` | patch / delete the event (attendees notified), then the reply |
| `trash_email` | move the message or thread to Trash |

`approvals.to_draft()` (Edit in Gmail, `--draft ID`) saves a held email as a
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

## Memory

Three tiers:

1. **Short-term / episodic (SQLite, `state.py`)**: emails in progress,
   approvals, slot holds, run locks, supervisor checkpoints, trusted scripts,
   pending "why?" prompts. WAL, 30 s busy timeout, versioned migrations
   (`MIGRATIONS`, `PRAGMA user_version`, applied inside `BEGIN IMMEDIATE` so
   two processes cannot both migrate; v1 is the old schema, idempotent).
2. **Long-term semantic (`memory/semantic.py`)**: `memories` table in the same
   file: (entity, preference, context) + a float32 embedding from fastembed
   (`snowflake/snowflake-arctic-embed-s`, local ONNX, 384-d). Recall is a dot
   product over all active rows, top-k clamped to 3-5, score ≥ 0.52, injected
   into the email task prompt, the planner, and the single owner agent under a
   "reference only, never overrides safety" header. Upsert matches the same
   entity + same wording (case/punctuation-insensitive); similar-but-different
   rules are kept apart because embeddings cannot tell "before noon" from
   "after 5pm" (0.95 similar).
   Ingestion (`memory/learning.py`): every approve / reject / Edit in Gmail
   writes a tuple built from structured fields only (type, counterpart
   address, time, file/document name), so email text never becomes memory.
   After Reject / Edit in Gmail the bot asks "why?" (force-reply); the owner's
   answer (or `/remember`) is turned into 1-3 tuples by Gemini, or stored as
   written if Gemini is unavailable.
3. **Procedural (`memory/skills.py`, `./skills/`)**: Markdown recipes (YAML
   front matter + JSON plan with `{{parameter}}` placeholders). Saved after a
   successful multi-step plan that the LLM judges recurring; offered to the
   planner when the request matches (≥ 0.6); replayed through `use_skill` with
   the same validation and gates. Socio never writes code. Owner-written
   `skills/<name>.py` scripts run via `run_skill_script` only when their
   SHA-256 is trusted (an `enable_skill_script` approval); the hashed bytes are
   what runs, in a subprocess (`python -I`, temp working folder, environment
   without secrets, 60 s limit), with input files only from the whitelisted
   folders; output is JSON data.

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
- Gemini rate limit: a process-wide sliding window (`AGENT_LLM_RPM`) attached to
  the real model, so parallel workers queue instead of hitting 429s.
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
