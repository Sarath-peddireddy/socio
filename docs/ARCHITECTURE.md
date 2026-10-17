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
   API result confirms it; sends are re-checked in the thread. Meeting links are
   never written by the model: a booking's Google Meet link comes from Google on
   approval, an existing meeting's from its calendar event, and any other video-call
   link in an outgoing write must appear in what Socio read
   (`graph._invented_meeting_link`).
5. **Memory informs, never authorizes.** Learned preferences and saved skills
   shape what the agent proposes; every gate above still applies.

## Components

```
app/telegram_bot.py      owner-only Telegram front end, approval buttons, inbox checks on request (/check)
app/telegram_format.py   Telegram HTML: esc(), agent Markdown -> HTML, HTML-safe clipping, cards
app/errors.py            error gateway: logs/agent.log + predefined owner messages / status cards
app/observability.py     optional Langfuse tracing: roots, spans, per-attempt LLM callback, scrubbing
app/main.py              CLI front end
        │
app/agent/runner.py      process_inbox / handle_message (email mode), handle_owner_command (owner mode)
app/agent/supervisor.py  owner instructions: planner -> task DAG -> parallel workers -> fan-in report
app/agent/workers.py     one worker subagent: clean context, restricted tools, strict JSON payload
app/agent/taskgraph.py   plan validation (ids, tools, dependencies, cycles) and ordering
        │
app/agent/decision.py    metadata rules → keyword pre-check → LLM structured decision (+ intent)
app/agent/graph.py       LangGraph: agent node ⇄ tools node; mode-specific toolsets and guards
app/agent/llm.py         get_llm(role) / invoke_llm; routing and failover in llm_router.py
app/agent/prompts.py     system prompts and per-intent playbooks
app/agent/approvals.py   execute / reject approval requests (re-validated at execution time)
app/agent/state.py       SQLite: message outcomes, attempts, actions, approvals, slot holds, failures
app/agent/untrusted.py   sanitize + wrap email content in <untrusted_email_body> tags
app/memory/              semantic.py (long-term memory), embeddings.py (local fastembed),
                         learning.py (explicit owner rules), episodic.py (24 h notes),
                         context_assembler.py / context_anchor.py (chat context),
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
`append_to_document`, `create_spreadsheet`, `append_sheet_row`), sharing
(`create_and_share_file`, `share_drive_file`), bookings,
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

That context lives in SQLite (schema v5), so a restart does not lose it:
`conversation_turns` holds each finished turn as plain text (the owner's
instruction and Socio's final reply with its card descriptions; raw tool output
never goes into history). `memory/context_assembler.py` loads the last 3 turns
from the past 24 h, in whole turns only: a turn is an owner message plus
everything that answered it, so an assistant tool call is never sent without its
tool results. While the estimate (characters / 4) is over 2,000 tokens the oldest
whole turn is dropped; a single oversized turn has its final reply shortened.
`conversation_anchors` (`memory/context_anchor.py`) remembers the thread,
contact (email addresses only), and document the chat was last about, taken from
the agent's tool calls, its proposals, and approved cards. It is added to the
owner instruction and the planner request as an `<active_context>` block
(reference only) and ignored after 24 h. Workers never see either: they get only
their task. `/new` archives the turns and clears the anchor.

The owner sees each approval request as its own card, so the owner-mode prompt
(not workers) tells the model not to narrate them and to answer exactly
`CARDS_ONLY` when approval requests are the only outcome.
`supervisor.owner_reply()` turns that into an empty reply and the bot sends
only the cards (no "I have sent an approval request..." preamble). If the
model says `CARDS_ONLY` but also did something directly (a Google Task) or a
tool failed, those are listed instead, so nothing done goes unreported.
`/help` and capability questions ("what can you do?",
`telegram_format.is_capability_question`) get a fixed welcome card without a
model call; so does `/start`, which Telegram sends when the bot is first opened
(not in the menu). One command name per job (`telegram_format.BOT_COMMANDS`, developer
request 2026-10-17): the former duplicates `/run`, `/reset`, `/clear`, `/schedules`
(`RETIRED_COMMANDS`) only answer "That command is now /check" (or /new, /routines),
and an unknown `/command` gets "I don't know that command" instead of going to the agent.

Status reactions on the owner's own message (developer request 2026-10-17, "like WhatsApp"; `setMessageReaction`,
one bot reaction per message, each call replaces the last; best-effort like "typing…"). `Bot._status_reactions`
wraps the work: 👀 (`REACTION_LOOKING`) for `/check`, `/dryrun` and instructions that are lookups
(`telegram_format.LOOKUP_REQUEST`: research / look up / find / search / check ...) or questions
(`not supervisor.is_action_request`); 👨‍💻 for other instructions; ✍ while a gap answer fills a reply. Then
`outcome_reaction`: 🤷 when the result has only PROBLEM parts (or the inbox check itself failed, or an exception:
the error gateway's card follows as before), 🤔 when it ends with a QUESTION, else 👌. Acknowledgements get
`acknowledgement_reaction` (thanks 🙏, bye / take care 🫡, good night 😴, perfect 👌, great 🔥, cool 🆒, hmm 🤔,
❤ ❤, else 👍); bye phrases count as acknowledgements now. Bots may only use Telegram's fixed `ReactionTypeEmoji`
list: no 🔍 / 👋, so 👀 / 🫡 stand in. `tests/test_reactions.py`.

Replies when Telegram is unreachable (2026-10-17 live bug: a "remember: ..." rule was saved, its "✓ Rule saved"
failed on a DNS outage, so the turn was never stored and "Is it done?" was answered about the request before).
`_handle_instruction` now stores the turn and the anchor BEFORE sending. Replies go through `Bot._deliver`: a
request that certainly never reached Telegram (`_never_reached`: DNS failure, connection refused, connect
timeout) is held in `pending_messages` (max 20) and sent first on the next poll whose `getUpdates` succeeds,
in order, up to `MAX_EDIT_TRIES` polls; a read timeout (the reply may have arrived) still raises to the error
gateway and is never resent. Cards were already safe (unsent cards stay `notified = 0`). `tests/test_lost_reply.py`.

Plain language (developer requirement: the reader is an end user, not a
developer): the owner/worker prompts (`PLAIN_LANGUAGE`) say to address the
owner as "you", never write "the owner" / "the user", never mention APIs,
tools, endpoints, JSON, ids, or how the work was split, never promise a later
action ("once approved I'll share the link"), and answer a question in one
natural paragraph. `app/agent/plain_text.plain_reply()` is the last filter on
the agent's own text (single agent and supervisor report; never on cards,
which must show exactly what will be sent): it removes raw JSON, approval
numbers, Gmail / Drive / task ids (links and email addresses are kept, readable
names like `Q3_Budget_Report_2026` too), and turns "for the owner" into "for
you". Tool refusals the model sees are plain sentences too ("That task is
already marked completed.", "I couldn't find that task in Google Tasks.").

### Supervisor and workers (fan-out / fan-in)

`handle_owner_command` goes through `supervisor.handle_owner_request`:

```
instruction ─► question / status check? (is_action_request, regex, no model call)
               ─► single owner agent with the conversation (never planned)
            ─► recall memories (top 3) + matching skills
            ─► one-task request? ─► single owner agent (as before)
            └► planner (1 LLM call, structured output PlannerOutput): tasks = [SubTask(task_id,
               target_domain, goal_contract, context_slice, tools_allowed, depends_on)] = a DAG,
               or use_skill + params
            ─► validate (≥ 2 tasks if the request names ≥ 2 domains, ≤ 6 tasks, known tools and
               domains, no cycles; a rejected plan goes back once with the reason)
            ─► checkpoint (supervisor_runs, worker_tasks)
            ─► ThreadPoolExecutor (AGENT_MAX_WORKERS = 3): a task starts when its dependencies succeeded
            ─► fan-in payloads ─► close duplicate approvals ─► short plain report (often empty)
            ─► count the workflow's success (3rd time: suggest /save_skill once)
```

- Intent routing: a message is an action command if a clause starts with an
  action verb ("Create ...", "..., and send it to X", "If not, send it",
  "Can you book ..."); otherwise, if it reads like a question or status check
  ("Have you sent ...", "What's ...", "Check ...", "Tell me ...", a trailing
  "?"), it goes to the single agent, which has the conversation. A status
  question was once split into a Drive task and a Gmail task that each lacked
  the context ("the sheet is empty", "no sheet was sent"). A missed action only
  loses the fan-out (the single agent has every tool).
- Only action commands that look multi-part, name two or more domains (email,
  calendar, workspace = Drive/Docs/Sheets, tasks), or match a skill pay for a
  planning call; if the planner returns one task, the single agent runs instead.
- Create & share: "create a sheet/doc and send it to X" is ONE workspace task with
  `create_and_share_file` (`taskgraph.COMPOUND_DOMAINS` counts it as covering
  email too, so the ≥ 2-domain rule accepts it). It was once split into a
  create-only task (no email tool) whose worker promised to share the link
  "once approved", which nothing did.
- Replies are split by intent (`ReplyPart`: answer / done / problem / question;
  `OwnerCommandResult.parts`). The bot (`telegram_format.reply_messages`) sends one
  message per kind with a short header (📋 Here's what I found / ✅ Done / ⚠️ Couldn't
  finish / ❓ I need your answer), then the approval cards, then the question last; a
  reply of one part is one plain message with no header. The single agent's text is
  split by `split_reply` (✅ / ⚠️ lines; a closing question after a long answer).
- Questions: workers choose sensible defaults (titles, subjects, messages: shown on
  the card) and only report `NEEDS_INPUT` with one question when blocked; that blocks
  their dependents (not logged as a failure). A worker that ends with a question is
  treated as asking. The planner may return only `question` when nothing can start.
  A placeholder title ("Untitled Document") is refused for create_document /
  create_spreadsheet and replaced from the email subject for create_and_share.
- One email to several people: prompts and the `send_new_email` description say one
  email with everyone in To; `approvals.merge_same_emails` merges new-email approvals
  from one instruction with the same subject, text, and files into one card before
  anything is shown (personalised texts, different files, or a card already shown stay
  separate). A "meeting invite" is one `propose_book_meeting` with `attendee_emails`.
- The report (`build_report`) has no header, task list, or goals: a part that
  produced an approval card says nothing (the card is the answer), a step whose
  result fed a later step says nothing; failures always show (with the
  planner's plain `label`, e.g. "⚠️ The email to Priya: ..."), dependents of a
  failure get one "I didn't do the steps that depended on it" line, and final
  findings show as plain sentences. Empty when the cards say it all ("Done."
  only if there is neither a card nor anything to say).
- Decomposition rule (§4c): the planner prompt forbids monolithic tasks, with
  examples. The whole plan comes back as one structured answer (a list), not
  one tool call per task: Gemini tended to emit only the first call and wait,
  which collapsed multi-part plans into one task. If the request names ≥ 2
  domains and the plan has < 2 tasks (pydantic `PlannedTasks`), or the answer
  does not fit the schema, the planner is asked once more with the reason;
  a second failure falls back to the single agent. The planner gets today's
  date and writes exact dates into the tasks, so workers never work them out.
- Least privilege: each task names one `target_domain`; `taskgraph.DOMAIN_TOOLS`
  caps its `tools_allowed` to that domain's tools (others are dropped and
  logged; a task with none left is refused). Email includes `find_documents`
  for attachments. Older saved skills without a domain get the narrowest one
  that covers their tools.
- A worker is a fresh graph run (`worker=True`, `tools_allowed=[...]`): its
  only message is its goal, its context slice, and its dependencies' payloads.
  Tools outside its list are neither bound nor executed. Worker summaries in
  the report go through `plain_reply` (no JSON, ids, or approval numbers).
- Fan-in contract: `{"status": "SUCCESS"|"FAILED", "modified_entities": [...],
  "diff_summary": "..."}`. A closing status object is always cut from the
  worker's text; "WAITING" / "PENDING" (a change waiting for approval) counts
  as SUCCESS (once leaked as raw `{"status": "WAITING", ...}` into Telegram).
  Entities come from the actions the tools recorded
  (the graph is streamed, so a worker that crashes still reports what it
  changed); the status also fails on any failed/unknown action.
- A failed task's dependents are not started; independent tasks continue; the
  run is `done`, `partial`, or `failed`. Exceptions never reach the bot loop.
- Overlaps: two workers proposing the same change (same event, document,
  recipient + subject, slot) get one approval; the other is closed as a
  duplicate. The slot-hold check and insert share one lock, so parallel
  workers cannot hold the same time twice.
- Workers run on Mistral, then Groq (`get_llm(WORKER)`), so parallel subtasks
  do not spend Gemini's daily quota; each provider key has one shared rate
  limiter (`AGENT_LLM_RPM` per Gemini key, `AGENT_MISTRAL_RPM`, `AGENT_GROQ_RPM`).
  Workers are threads (the stack is synchronous, so not `asyncio.gather`); each
  call fails over on its own, and a failed worker never stops its siblings.
  Google token refresh is serialized (`CREDENTIALS_LOCK`); Google clients are
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

Sharing access (`app/workspace/sharing.py`, Drive `permissions.create`): only
files in Socio's output folder (the `drive.file` scope covers files Socio
created; no new sign-in needed). `create_and_share_file` -> one
`create_and_share` approval ("📊 Create & Share Spreadsheet" / "📄 ... Document":
title, recipient (Editor), columns or content, subject, message; Approve & Send /
Cancel). Approving creates the file, grants the role without Google's own
notification (Google insists on it for non-Google addresses, then it is sent),
and emails the live link from Gmail (so it is in Sent), in that order; each step
runs only if the one before succeeded, the email is never retried, and the card
says exactly how far it got ("Created ... and sent edit link to X." /
"Created ..., but I couldn't share it ..., so no email was sent." / "... but the
email didn't go through ... You can send them this link"). `share_drive_file` ->
`share_file` approval for a file Socio created earlier (Google emails the link).

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

## Routines and news (`app/scheduler/`, `app/news/`)

The owner can ask for something on a schedule ("send me the top AI news every day
at 9 AM"). Owner-mode tools: `schedule_task(title, query, time_of_day, frequency,
timezone="Asia/Kolkata", day_of_week, date)`, `list_scheduled_tasks`,
`cancel_scheduled_task(job_id)`. Creating or cancelling a routine is a local write
(like `remember_rule`), not an approval; anything the routine later proposes
(emails, events, writes) still becomes an approval card. Not available in email mode.

- **Storage:** `scheduled_jobs` (migration v6): `id`, `title`, `prompt_query`,
  `cron_expr` ("0 9 * * 1-5", in the job's own timezone), `rrule` (for reference),
  `timezone`, `next_run_at`, `frequency` (`daily` / `weekdays` / `weekly` /
  `one_off`), `is_active`, `last_run_at`, `last_status`. `next_run_at` is UTC text in
  one fixed format (`2026-10-12T03:30:00Z`) and "now" is passed in from Python:
  SQLite's `datetime('now')` uses a space instead of the `T`, so comparing the two
  as text would be wrong. Index `idx_scheduled_jobs_next_run (is_active,
  next_run_at)`: the due query is one index range search (~0.005 ms at 5,000 jobs).
- **Times:** `zoneinfo` + `tzdata` (already installed), so any IANA zone works,
  daylight saving included; IST is the default. Only the four frequencies above are
  supported, so `jobs.py` computes the next run itself (no cron library).
- **Runner:** the bot's loop calls `service.run_due_jobs` every 60 s (routines are the owner's own
  instructions, so they run although the inbox is never checked on a timer). For each due job it first
  *claims* the slot (conditional UPDATE: next slot, or inactive for a one-off), then
  runs `prompt_query` through `handle_owner_command` with no chat history, then the
  bot sends "⏰ **Scheduled Briefing:** title" + the answer (approval cards after it)
  and saves the turn so follow-up questions work. At most once per slot: a crash or a
  second process cannot repeat it, and a failed run is not retried (the owner gets the
  error gateway's status card; the next slot runs normally). A slot missed while the
  bot was off runs once if it is at most `AGENT_SCHEDULE_GRACE_HOURS` (6) late,
  otherwise it is skipped with a one-line notice; missed days never pile up.
- **No loops:** while a routine runs, `schedule_task` / `cancel_scheduled_task`
  refuse (a process-wide flag, since workers run in other threads). Requests with a
  recurrence ("every day", "each Monday", "daily at 7") skip the planner
  (`supervisor.RECURRING_REQUEST`); "daily" / "weekly" alone ("the weekly prep") do not
  count.
- **Confirmation:** the reply to a new routine is the tool's own text ("✅ Scheduled! I
  will prepare and send your **X** daily at 09:00 AM IST."), added by
  `supervisor.owner_reply`; the model answers `CARDS_ONLY` when that was all.
- **Telegram:** `/routines` lists active routines with one
  Cancel button each (`unsched:<job_id>`, owner only); the message is redrawn after a
  tap. No ids in the visible text.
- **News:** `get_news(topic, max_items)` reads public RSS / Atom feeds (Hacker
  News, TechCrunch, The Verge, Ars Technica, MIT Technology Review; override with
  `AGENT_NEWS_FEEDS`) with `requests` and the standard library XML parser: 10 s
  timeout, 3 MB cap, documents declaring entities refused, only http(s) links kept,
  items older than 4 days dropped, duplicates merged, newest first; a topic matches
  whole words ("AI" also matches LLM / OpenAI / ..., but not "said"). Headlines and
  summaries are sanitized and wrapped in `<untrusted_news_item>` tags (`<` / `>`
  escaped), covered by the same untrusted-data rule as emails. Planner domains
  `news` and `routines` hold these tools.

## Approvals

`propose_*` tools (and, in email mode, `send_gmail_reply`) validate the request
and the graph records an approval (`pending_approvals` table). The Telegram bot
pushes each new request as one HTML card (`telegram_format.approval_card`):
🗑️ Trash, ✉️ Outgoing Email Proposal (To, Subject, the full draft in a
blockquote), 📅 Meeting Proposal (title, attendee, slot in IST on the 12-hour clock, the
reply to send afterwards; no hold status and no reason line on any card, developer decision
2026-10-11: the hold still works in the background),
🔁 Move / 🚫 Cancel Meeting, 📝 Workspace Update (target, action, `+` diff),
📊 / 📄 Create & Share (Approve & Send / Cancel), 🔗 Share File, 🔐 Trust Skill Script. Buttons: Approve & Trash / Book / Move / Cancel, or
Approve; emails get Approve & Send / Edit in Gmail / Reject. Every dynamic
value is HTML-escaped. After a tap the card is rendered again from the stored
request with the result line. `approvals.approve()` claims the request atomically (so a double tap
cannot run it twice), re-validates (event exists, new slot still free),
executes, then sends the follow-up reply if one was attached.

A tap is answered the moment it arrives (toast "Working on it…") and the card is edited to
"⏳ Working on it…" without buttons before any Google call; the result line replaces it (Google
calls take ~0.6 s each, so answering after the work felt like 3-5 s). An unexpected error puts
the buttons back (`Bot._restore_card`). A tap on a card that is no longer open gets a short
toast ("This was already approved."), never a pop-up and never a card edit.

Inbox summary: an email that has a card is shown by its card alone; the others get one line in
fixed wording (`telegram_bot.INBOX_NOTES`, or the error gateway's own sentence for an outage),
never the agent's internal summary. With nothing to say, no summary is sent. The "Reply Drafted
in Gmail" card shows To / Subject / what you need to do / the draft text itself.

Replies Socio can't finish (something only the owner knows): there are no Gmail drafts in email
mode. `create_reply_draft` is gated like `send_gmail_reply` and becomes the same reply card; with
`[OWNER: ...]` gaps the card is "✉️ Reply Needs Your Input" with the question (the tool's `reason`)
and only Edit in Gmail / Reject, and `approve()` refuses while a gap remains. The owner answers in
chat, as a Telegram reply to the card (`pending_approvals.card_message_id`, migration v8) or as the
next message after it: `gap_fill.fill_gaps` (one model call; it also decides whether the message is
an answer at all, otherwise it runs as an instruction) -> `approvals.fill_reply` (old request
resolved, a new one with the same recipient, thread, files and snapshot) -> a fresh card with
Approve & Send / Edit in Gmail / Reject. In owner mode `create_reply_draft` still saves a draft.

Access requests from an email (`propose_file_access`, email mode only, conditional proposal):
someone in the conversation asks for access to a Google Doc / Sheet whose link is in the
conversation (`runner._thread_file_ids` -> `state.thread_file_ids`; anyone else or any other
file is refused in `graph._scope_violation`). The tool reads the file's permissions first
(`sharing.current_access`; drive.file can read them for Socio's files): already enough access ->
no card, the agent replies telling them to open it signed in as that address; otherwise one 🔗
Share File card (`share_file` + `notify: False` + the reply) -> approve grants access, then sends
the reply in the thread (`[link]` replaced by the file's link).

Taps are idempotent: the claim is one `UPDATE pending_approvals SET status = 'executing',
locked_at = now WHERE id = ? AND status = 'open'`. Only the tap that changes the row runs; any
other (double tap, repeated Telegram update, CLI at the same moment) gets `duplicate`, and the
bot answers it with a pop-up only, so the card keeps the first tap's result. A request still
`executing` 15 minutes after its `locked_at` was cut off by a crash: `/status` and `--status`
list it ("may or may not have happened"); it is never retried automatically.

A failure while Google is carrying the action out is reported by what it proves
(`google_api.outcome_unknown`). A dropped or timed-out connection, or a 5xx, after the write
was sent does not prove it failed (a timed-out send may have been delivered): the request
becomes `unknown`, the email `needs_attention`, the action is recorded as `unknown`, and the
owner is told "I can't tell whether the email went out ... Check your Gmail Sent folder before
trying again", never "nothing was changed". 429 / 4xx / validation errors / a failure before
`_execute` keep "That didn't go through, so nothing was changed." Writes that took effect
before a later step failed (a created Doc or Sheet, access given) are collected with
`google_api.recording_writes()` / `completed_write()` and listed ("Already done: created the
Google Doc 'X' (link)"). The same applies to the reply after a booking / share and to the
email inside Create & Share (`reply_unknown`, `email_unknown`).

Hours can pass between card and tap, so `check_still_current()` runs right after the claim:
- an email reply (also the reply after a booking): the thread must have no message newer than
  `reply.last_seen_message_id`, the newest message when the card was made (whoever wrote it:
  the owner answering in Gmail counts). Gmail's `historyId` is not used: reading or labelling
  a message changes it too, which would make nearly every card stale.
- a booking or move: the slot must still be in the future and free in Calendar.

If not, nothing is done, the request becomes `stale` (its slot hold is released, the email
`needs_review`), and the card is edited to "⚠️ Things changed since this was proposed" with the
reason ("Priya wrote again in this conversation after I prepared this."). Cards from an
incoming email get **🔄 Re-check**: `runner.recheck_stale_approval` (under the inbox lock;
`stale` -> `rechecked` once) handles the thread's newest incoming email again, unless an inbox
check already handled that newer email, and the fresh card follows. Owner-mode cards say
"Ask me again if you still want it." Cards made before this check (no snapshot) skip the
thread part.

| Type | Executes |
|---|---|
| `send_reply` | the reply in the thread |
| `send_reply_with_attachment` | the reply with its files |
| `send_email` | a new email (owner mode) |
| `create_document` / `append_document` | create a Doc / append text (Socio folder) |
| `create_spreadsheet` / `append_sheet_row` | create a Sheet / add one row (values RAW) |
| `create_and_share` | create a Sheet/Doc, give the recipient access, email them the link |
| `share_file` | give someone access to a file Socio created (Google emails them) |
| `enable_skill_script` | trust `skills/<name>.py` at the SHA-256 shown (re-checked) |
| `book_meeting` | create the event with a Google Meet call (conflict re-checked, every attendee invited; from an email the sender is always invited), then the reply with the real Meet link in place of `[meeting link]` |
| `move_event` / `cancel_event` | patch / delete the event (Google notifies attendees), then the reply in the email thread; on the owner's instruction instead one email from the owner to all attendees (old -> new time, or cancelled; shown on the card as "Then email") |
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

Availability (`find_calendar_availability`): weekdays 11:00-16:00 IST only. `date=YYYY-MM-DD`
checks exactly one day; without it, `days_ahead` days from today (today = day 1). It once missed
a free Monday because the worker asked for "tomorrow" as `days_ahead=1` (= only today, a Sunday).
Returns `{"slots", "days": [{"date", "day", "free": ["11:00-16:00"]}], "note"}`; an empty result
always has a note with the reason (weekend, today's window over, fully booked) and the next
weekday to try. Refused with a plain reason: a past date, more than 60 days ahead, an unreadable
date, a meeting shorter than 5 or longer than 300 minutes. Today starts at the next quarter hour.

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
   `supervisor_runs.skill`; v7 adds `pending_approvals.locked_at`).
2. **Long-term semantic (`memory/semantic.py`)**: `memories` table in the same
   file: (entity, preference, context) + a float32 embedding from fastembed
   (`snowflake/snowflake-arctic-embed-s`, local ONNX, 384-d; onnxruntime's own telemetry is off:
   `ORT_DISABLE_TELEMETRY=1` when `embeddings.py` loads, `disable_telemetry_events()` before the session). Recall is a dot
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
   **Conflicts (`memory/rule_engine.py`)**: before a standing rule is upserted, one
   supervisor-chain call compares it with the active memories (structured
   `ConflictCheck`: has_conflict, rule number, explanation, merged rule; the number
   must be an active memory, an identical rule is an update). On a conflict nothing
   is saved: a `rule_conflict` approval (from `/remember` directly, from chat via
   `remember_rule` returning `{"proposal": ...}`, a conditional proposal tool).
   `approvals.decide_rule_conflict`: replace (old memory → `replaced`, new saved),
   merge (merged rule saved under the old rule's entity, old → `replaced`), both (new
   saved with "Takes priority over: <old>" in its context); Cancel = `reject()`.
   Model unavailable → saved unchecked (warning logged). Situational notes are not checked.
4. **Learned corrections (`agent/eval_feedback.py`, `failure_cases`, schema v9)**:
   failures are recorded locally — `rejected` (Reject on a proposal; not on
   questions such as task "Is this done?", rule conflicts, script trust), `stale`
   (approve found the thread/slot changed), `worker_timeout`, `plan_invalid` — with
   the plan id, the owner's request and the plan steps, taken from
   `pending_approvals.origin` (stamped by `remember_origin` when the request ends:
   mode, request, run id, plan, trace id; first stamp wins). A Langfuse score
   `owner_feedback` = 0 (metadata `eval_status: negative_feedback`) goes to the
   card's original trace only if tracing is enabled. A failure alone teaches
   nothing: a Telegram reply to a rejected/stale card goes through one model call
   (`Correction`: feedback → a one-sentence "When ..." lesson; a new instruction →
   handled as usual, nothing stored). The newest 2 lessons for the mode (owner /
   email) whose domain matches the request (`request_domains`) are injected as
   `<learned_corrections>` (≤ 250 tokens, sanitized) into the planner, the single
   owner agent, and the email task prompt. `/memories` lists them; `/forget_lesson`.

**Retention (`retention_manager.py`)**: `Bot.maintain_database` (startup, then
every 24 h) runs `purge_expired_records` in one `BEGIN IMMEDIATE` transaction,
then `wal_checkpoint(TRUNCATE)`. Operational (30 d): finished supervisor runs and
their worker tasks, decided approvals (approved/rejected/resolved/stale/rechecked/
failed), `actions`, `failures`, released holds, expired notes, uncorrected failure
cases. Messages (90 d): `conversation_turns`. Anchors (1 d). Kept: open/executing/
unknown approvals, `processed_messages` (stops old emails being handled again),
memories, lessons, routines, tracked tasks. All compared timestamps are SQLite UTC
text, so `datetime('now', '-N days')` is valid (unlike `scheduled_jobs.next_run_at`).
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
- Models (`app/agent/llm_router.py`): two chains. Supervisor (email mode,
  owner agent, planner, triage, rules, skills): Gemini primary key → Gemini
  fallback key → Mistral. Worker: Mistral → Groq → Gemini fallback → Gemini
  primary (the supervisor's key last).
  Each client makes one attempt (no SDK retries); on 429, 5xx, a network
  error, or a rejected key/model the same request goes to the next provider
  at once and a warning is logged (`⚠️ [ModelRouter] Rate limit hit on Worker
  priority 1 (Mistral). Seamlessly failing over to priority 2 (Groq).`, no
  stack trace). The failed provider cools down (suggested delay for
  per-minute limits, 1 h for daily quota or a rejected key, 10 s for outages)
  so later calls skip it without a request; Groq's 413 (request over its
  per-minute token budget) and `tool_use_failed` skip it for that request
  only. If every provider is cooling down, the call waits for the first to
  return (at most 70 s, 3 times), else raises `LLMUnavailableError`
  ("daily quota" when quota is the cause). Model calls have no side effects,
  so repeating one elsewhere is safe. langchain-google-genai adds the
  documented skip-validator thought signature to tool calls that came from
  another provider, so a mid-run failover to Gemini 3 is accepted;
  langchain-mistralai rewrites other providers' tool-call ids into Mistral's
  9-character format. Mistral's free tier gives `mistral-small`/`medium` 0
  requests/minute, so the default is `ministral-14b-latest` (30/min).
- Restart recovery: interrupted emails are retried from the database; actions
  already completed are shown to the agent and cannot be repeated.
- Rate limits: a process-wide sliding window per provider key (`AGENT_LLM_RPM`,
  `AGENT_MISTRAL_RPM`, `AGENT_GROQ_RPM`) attached to the real model, so parallel workers queue
  instead of hitting 429s.
- Inbox checks on request only (developer decision 2026-10-17): `/check`, or a whole message such as
  "check my emails" / "run inbox sweep" (`telegram_format.is_inbox_check_request`, regex, no model call) ->
  `Bot.check_now`: `process_inbox`, then `tracker.check_tasks` (failures logged only). `AGENT_POLL_MINUTES`
  defaults to 0, so the loop never reads the inbox or Google Tasks on its own; a number > 0 opts back in. The
  daily retention purge + `wal_checkpoint(TRUNCATE)` (`maintain_database`) is on its own 24 h clock in the same
  loop, which always runs because it is Telegram's long poll. `tests/test_on_demand_inbox.py`.
- Concurrency: the Telegram bot is one sequential loop (messages, button taps,
  and inbox checks never overlap). Across processes (bot + CLI),
  SQLite runs in WAL mode with a 30 s busy timeout, and a live inbox run takes
  a 30-minute lease (`run_locks` table, renewed per email, released at the
  end or on a crash's expiry). A second live run returns `skipped_busy`; dry
  runs need no lock. Approvals are claimed atomically, so a double tap from
  Telegram and the CLI cannot execute twice.
- Connections are short-lived: `state._connect()` commits (or rolls back) and closes at the end
  of each block, so no reader holds a snapshot that would stop SQLite from emptying the WAL
  file; `synchronous=NORMAL` on each connection (safe with WAL). The bot runs
  `PRAGMA wal_checkpoint(TRUNCATE)` at startup and once a day (`maintain_database`). The busy
  timeout stays 30 s (a shorter one brings back "database is locked" between bot and CLI).
- Time limits: every model call has a 60 s socket timeout (a timeout fails over to the next
  provider) and every Google call 30 s, so no step hangs forever. On top, each supervisor
  worker has a wall-clock budget (`AGENT_WORKER_TIMEOUT`, 180 s) checked after every graph
  step; past it the worker stops before its next model call or tool round and reports
  FAILED ("It took too long, so I stopped it."). Threads cannot be killed, so a running step
  finishes first. Its slot frees for the next task, dependents are not started, independent
  siblings finish (developer decision 2026-10-11: not 25 s, and siblings are not cancelled).
- One "today" per request (`app/agent/clock.py`): the Telegram message's own timestamp (CLI /
  routine / email: the start of the run) is frozen in a contextvar, copied into each worker
  thread (`contextvars.copy_context()` in `run_plan`). The owner, email, worker, and planner
  prompts carry `<temporal_anchor>Today is 2026-10-11 (Sunday) ... Tomorrow is 2026-10-12
  (Monday) ... Asia/Kolkata</temporal_anchor>`, so a request sent at 23:59 and run after
  midnight, or on a UTC server, still means the same dates. Tool checks that need the real
  time (is a slot in the past or free now?) use the real clock.

## Observability (`app/observability.py`, Langfuse)

Optional and OFF by default: on only when `LANGFUSE_TRACING_ENABLED=true` AND
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set (developer decision 2026-10-12: no
personal data leaves the machine by accident; keys alone do nothing). Langfuse Python SDK v4
(OpenTelemetry).

```
telegram_interaction       root per owner message: session = chat id, user = Telegram user id,
                           input = the message, output = every message sent back, metadata =
                           ingress_utc, temporal_anchor, intent, route; tag telegram
  supervisor_plan          (chain) planner call(s); output = the validated task DAG
  owner_agent              (agent) the single agent, when the request is not planned
  worker_<domain>_<task>   (agent) one per worker thread; input = goal contract + tools,
                           output = the fan-in payload; ERROR if FAILED
    worker_llm             (generation) one per provider attempt: model, tokens, messages;
                           metadata provider / priority / attempt / failover_from
    model_failover         (event, WARNING) after a 429 / outage / timeout, before the retry
    <tool name>            (tool) every tool call: args, result, metadata outcome =
                           read / completed / proposed / dry_run / refused / failed
email_run                  root per email (session = gmail-thread-<id>, tag email)
scheduled_routine          root per routine run (session = chat id, tag routine)
owner_command              root for CLI --ask (tag cli); nothing inside the bot's trace
```

- Generations come only from Langfuse's LangChain `CallbackHandler`, attached per
  attempt in `llm_router.route` (only for routed models; test fakes are untouched). No
  `@observe` decorators, so no model call is recorded twice. A failed attempt is an ERROR
  generation (the provider's status in its status message); the next provider's
  generation follows under the same parent.
- Worker spans nest under the request's root because `run_plan` already starts each
  worker in `contextvars.copy_context()` (OpenTelemetry's current span is a contextvar).
- Every observation's input/output/metadata goes through `scrub()` (also Langfuse's
  `mask` hook): base64 runs >= 400 chars and `data:` URIs replaced, strings clipped at
  20,000 chars. Prompts and email text are still recorded (that is what makes a trace
  useful), which is why tracing is opt-in.
- Never in the way: off / no keys -> every call a no-op; any SDK error ->
  one warning, no-op; spans export in a background thread, so an unreachable Langfuse
  does not delay replies (the CLI's final `flush()` waits up to the exporter's timeout,
  ~5 s); the bot checks the keys at startup in the background and turns tracing off if
  they are rejected (401/403). The bot does not flush per message; `shutdown()` on exit.
- `scripts/verify_langfuse.py` sends a parent span + generation + tool span and reads it
  back through the v2 observations API (new Langfuse organizations have no
  `GET /api/public/traces`; it answers 410).

## Error gateway (`app/errors.py`)

Every Telegram update and button handler, the inbox check, the poll
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
| No AI provider answered: timeout / unreachable / rate limit (`LLMUnavailableError`, `httpx.TimeoutException`) | AI Reasoning Timeout | The AI model took too long to respond. Tap /run or send the request again. |
| Daily quota exhausted on every configured provider | AI Daily Limit Reached | The AI model's daily limit is used up. Please try again later. |
| anything else | Task Paused | Unable to complete that action right now. The details have been logged for review. |

The wording only promises what happens (developer decision): nothing is
queued or re-run behind the owner's back, except that an email interrupted by
an outage is retried the next time the inbox is checked (its reason says so). Our own
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
