# Socio architecture

## Principles

1. **Deterministic where possible, LLM where needed.** Fetching, rule-based
   triage, verification, and bookkeeping are plain Python. The LLM is used for
   judgement: classifying intent, choosing tools, writing emails.
2. **Two trust levels.** Instructions from the owner (Telegram / CLI) are
   trusted. Email content is untrusted data and gets a restricted toolset.
3. **Human-in-the-loop for destructive changes.** Trashing email and
   moving/cancelling events are proposals; only the owner's approval executes
   them.
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
app/agent/state.py       SQLite: message outcomes, attempts, actions, approvals, failures
        │
app/gmail/               read, search, reply, send, draft, trash (never permanent delete)
app/calendar/            availability, create, find, move, cancel
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

- only the conversation being handled can be replied to, only to its participants
- calendar invites only to conversation participants
- an event can only be moved/cancelled on request of one of its attendees
- one booking / reply / draft per email, even across restarts
- no Trash, no new emails, no mailbox search

## Owner mode (Telegram / `--ask`)

Full toolset (search mail, send new email, calendar, documents, Google Docs,
Sheets, and Tasks) plus `propose_trash_email`. Destructive changes still go
through approvals.

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

Sharing files by email: when the agent (email mode) replies with a Drive
attachment the owner has not marked *Always allow*, the reply becomes a
`send_reply_with_attachment` approval (Telegram: Send / Reject / Send & always
allow). Files in the local shareable folder are sent directly.

Follow-up tasks: when an email ends as `needs_review`, `needs_attention`, or
`awaiting_approval`, a Google Task is created (once per email) and completed
when the owner approves, rejects, or resolves it. Failures never affect email handling.

## Approvals

`propose_*` tools validate the request and the graph records an approval
(`pending_approvals` table). The Telegram bot pushes each new request with
Approve / Reject buttons. `approvals.approve()` claims the request atomically
(so a double tap cannot run it twice), re-validates (event exists, new slot
still free), executes, then sends the follow-up reply if one was attached.

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

## Outcome statuses

`replied`, `replied_unverified`, `no_action`, `needs_review`,
`needs_attention`, `awaiting_approval`, `superseded`, `failed` (retried),
`failed_permanently`.
