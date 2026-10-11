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
   at all (`propose_book_meeting` only). Creating and completing Google Tasks are
   approvals too (`propose_create_task`, `propose_complete_task`, developer decision
   2026-10-11); Socio never adds tasks on its own (automatic follow-ups are off).
4. **Never claim what didn't happen.** An action counts as done only when the
   API result confirms it; sends are re-checked in the thread.
5. **Memory informs, never authorizes.** Learned preferences and saved skills
   shape what the agent proposes; every gate above still applies.

## Components

```
app/telegram_bot.py      owner-only Telegram front end, approval buttons, scheduled inbox checks
app/telegram_format.py   Telegram HTML: esc(), agent Markdown -> HTML, HTML-safe clipping, cards
app/errors.py            error gateway: logs/agent.log + predefined owner messages / status cards
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
                         learning.py (explicit owner rules), episodic.py (24 h notes),
                         skills.py (procedural memory)
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
or the exact lines / cells to be added (`+ ` lines). `append_sheet_row` is refused
(no card) when an identical row is already in the sheet; the model gets the row
number and link to tell the owner. Tasks are created and completed only through
`propose_create_task` / `propose_complete_task` (the tool refuses an empty title or a bad
due date before any card). Rules the owner states in chat go to
memory through the `remember_rule` tool (never to Docs/Sheets/Tasks).

Owner-facing text never shows internal ids: cards have no approval numbers,
`approvals` messages are plain ("Meeting moved.", "This request was already
approved."; the CLI prefixes `#ID:`), the supervisor report has no task ids,
and the prompts tell the model to name things by subject, name, or time. The
bot's follow-up context (last 3 exchanges) stores what each card proposed and
how the owner decided ("The owner rejected: ..."), so "create a task for this"
refers to it.

The owner sees each approval request as its own card, so the owner-mode prompt
(not workers) tells the model not to narrate them and to answer exactly
`CARDS_ONLY` when approval requests are the only outcome.
`supervisor.owner_reply()` turns that into an empty reply and the bot sends
only the cards (no "I have sent an approval request..." preamble). If the
model says `CARDS_ONLY` but also did something directly (a Google Task) or a
tool failed, those are listed instead, so nothing done goes unreported.
`/start`, `/help`, and capability questions ("what can you do?",
`telegram_format.is_capability_question`) get a fixed welcome card without a
model call.

### Supervisor and workers (fan-out / fan-in)

`handle_owner_command` goes through `supervisor.handle_owner_request`:

```
instruction ─► recall memories (top 3) + matching skills
            ─► one-task request? ─► single owner agent (as before)
            └► planner (1 LLM call): spawn_subagent(task_id, goal_contract, context_slice,
               tools_allowed, depends_on) calls = a DAG, or use_skill(name, params)
            ─► validate (≤ 6 tasks, known tools, no cycles) ─► checkpoint (supervisor_runs, worker_tasks)
            ─► ThreadPoolExecutor (AGENT_MAX_WORKERS = 3): a task starts when its dependencies succeeded
            ─► fan-in payloads ─► close duplicate approvals ─► one report
            ─► count the workflow's success (3rd time: suggest /save_skill once)
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
- A worker crash is logged in full by the error gateway; its `diff_summary`
  carries only a predefined message, so the report never shows raw errors.

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

Task tracking (`app/tasks/tracker.py`, `tracked_tasks` table, schema v4): on the
bot's schedule `check_tasks()` reads the default Google Tasks list once (no AI
call) and keeps a local copy of the open tasks. It never changes a task itself;
it creates `complete_task` approvals: overdue tasks get one "Is this done?" card
(the day after the due date, 09:00-21:00 IST, at most 3 per check, once per due
date; "Not yet" = rejected, not asked again), and tasks ticked or deleted in the
Tasks app are untracked silently (any open card about them is resolved). A full
page of 100 tasks closes nothing (the rest may be on the next page). The
owner-mode single agent gets the open tasks in its instruction
(`open_tasks_block`, ids for tool calls only) and proposes closing one when the
owner says it is done. Old automatic reminders (titles starting "Socio: ") get
one `close_old_reminders` card per batch ("Close all" / "Keep them"); approving
completes each, skipping ones already gone. Approving `create_task` starts
tracking the new task; approving `complete_task` stops it.

Follow-up tasks (`app/agent/followups.py`) are OFF by default (developer decision
2026-10-11: Socio creates tasks only when asked). With `AGENT_AUTO_TASKS=1`, an
email that ends as `needs_review`, `needs_attention`, or `awaiting_approval` gets a
Google Task (once per email), completed when the owner approves, rejects, or
resolves it. With it off, Socio neither creates nor completes them. Failures never
affect email handling.

## Approvals

`propose_*` tools (and, in email mode, `send_gmail_reply`) validate the request
and the graph records an approval (`pending_approvals` table). The Telegram bot
pushes each new request as one HTML card (`telegram_format.approval_card`):
🗑️ Trash, ✉️ Outgoing Email Proposal (To, Subject, the full draft in a
blockquote), 📅 Meeting Proposal (title, attendee, slot in IST, "Local 24h
soft-hold active" only while a hold exists, the reply to send afterwards),
🔁 Move / 🚫 Cancel Meeting, 📝 Workspace Update (target, action, `+` diff),
🔐 Trust Skill Script. Buttons: Approve & Trash / Book / Move / Cancel, or
Approve; emails get Approve & Send / Edit in Gmail / Reject. Every dynamic
value is HTML-escaped. After a tap the card is rendered again from the stored
request with the result line. `approvals.approve()` claims the request atomically (so a double tap
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
`needs_review` for the owner. In Telegram a rejection ends silently: the card
is edited to "❌ Proposal rejected. No changes were made." (Edit in Gmail:
"✏️ Saved to Gmail Drafts. Nothing was sent."), with no buttons, and nothing is
learned (the "+ Add Permanent Rule" button was removed, developer decision
2026-10-11; old buttons only explain how to add a rule). Failure messages on the card
keep our own validation reasons (e.g. "overlaps 'Dentist'"); anything else is
a predefined message, while `pending_approvals.resolution`, `failures`, and
`logs/agent.log` keep the full error.

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
   pending rule prompts (`feedback_requests`), workflow success counts
   (`workflow_stats`), and situational owner notes (`episodic_notes`, 24 h,
   `memory/episodic.py`; injected into the email, planner, and owner prompts
   under a "notes for today, reference only" header while active). WAL, 30 s
   busy timeout, versioned migrations (`MIGRATIONS`, `PRAGMA user_version`,
   applied inside `BEGIN IMMEDIATE` so two processes cannot both migrate; v1 is
   the old schema, idempotent; v3 adds the notes, workflow counts, and
   `supervisor_runs.skill`).
2. **Long-term semantic (`memory/semantic.py`)**: `memories` table in the same
   file: (entity, preference, context) + a float32 embedding from fastembed
   (`snowflake/snowflake-arctic-embed-s`, local ONNX, 384-d). Recall is a dot
   product over all active rows, top-k clamped to 3-5, score ≥ 0.52, injected
   into the email task prompt, the planner, and the single owner agent under a
   "reference only, never overrides safety" header. Upsert matches the same
   entity + same wording (case/punctuation-insensitive); similar-but-different
   rules are kept apart because embeddings cannot tell "before noon" from
   "after 5pm" (0.95 similar).
   Ingestion (`memory/learning.py`) is explicit only; a decision alone writes
   nothing. `/remember` text is turned into 1-3 tuples by Gemini (stored as
   written if Gemini is unavailable). When the owner states a rule in chat, the
   agent calls `remember_rule`; the text goes through one Gemini call that classifies
   it (`STANDING_POLICY` → upsert as `owner_rule`, confirmed "✓ Rule saved: …";
   `SITUATIONAL` → `episodic_notes` for 24 h, confirmed "Understood for
   today.") and extracts the tuples. Without Gemini a keyword check decides
   (never/always/only/before/after → standing; today/already/tired → situational;
   unsure → standing, since the owner asked for a permanent rule). Request
   context comes from structured fields only, so email text never becomes memory.
3. **Procedural (`memory/skills.py`, `./skills/`)**: Markdown recipes (YAML
   front matter + JSON plan with `{{parameter}}` placeholders). Never saved
   automatically: `/save_skill NAME` / `--save-skill NAME`
   (`supervisor.save_last_plan`) packages the newest `done` run with ≥ 2 tasks
   that was not itself a skill replay; one Gemini call generalizes it into
   parameters (validated; saved verbatim if that fails). `workflow_stats` counts
   successes per tool-graph signature (each task's tools + dependency edges,
   goals ignored); on the 3rd success the report suggests `/save_skill` once.
   Offered to the planner when the request matches (≥ 0.6); replayed through
   `use_skill` with the same validation and gates. Socio never writes code. Owner-written
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

## Error gateway (`app/errors.py`)

Every Telegram update and button handler, the scheduled inbox check, the poll
loop, workers and the supervisor's fan-out, approval execution, the inbox
runner, and graph tool calls go through `report_failure(exc, where, **ids)`. It
writes the stack trace, the location, the ids (never email bodies or tool
argument values), and a timestamp to `logs/agent.log` (`AGENT_LOG_DIR`;
rotating, 2 MB x 3) and returns one predefined message: plain text (CLI,
worker summaries, inbox reasons) or, with `html=True` (the bot), a status card
"⚠️ <b>Title</b>" + newline + the sentence (Telegram HTML has no `<br>`).

| Exception | Card title | Message |
|---|---|---|
| Google `HttpError` 429/5xx, `GoogleApiUnavailableError`, OAuth/DNS transport errors | Google Service Busy | Google services are busy or unreachable right now. Please try again in a few minutes. |
| `sqlite3.OperationalError` (locked / busy) | Workspace Busy | The system is completing another task. Please try again in a moment. |
| Gemini timeout / unreachable / rate limit (`LLMUnavailableError`, `httpx.TimeoutException`) | AI Reasoning Timeout | The AI model took too long to respond. Tap /run or send the request again. |
| Gemini daily quota exhausted | AI Daily Limit Reached | The AI model's daily limit is used up. Please try again later. |
| anything else | Task Paused | Unable to complete that action right now. The details have been logged for review. |

The wording only promises what happens (developer decision): nothing is
queued or re-run behind the owner's back, except that an email interrupted by
an outage is retried on the next inbox check (its reason says so). Our own
validation errors (`ValueError` family, e.g. "that slot overlaps 'Dentist'")
keep their message on approval cards and in tool results the model sees;
everything else is replaced. `/status` shows only a failure count, and
`Bot.send` drops any traceback-looking line as a last line of defence. If
Telegram ever rejects a message's HTML (a formatting bug), the bot logs it and
resends the same text without tags instead of losing it. The CLI
still prints `detail (local only)` for failed approvals.

## Outcome statuses

`replied`, `replied_unverified`, `no_action`, `needs_review`,
`needs_attention`, `awaiting_approval`, `superseded`, `failed` (retried),
`failed_permanently`. A run that found another one in progress reports
`skipped_busy` (not stored).
