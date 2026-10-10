# Socio

**Your inbox and calendar, handled.** Socio is an AI assistant that reads
your Gmail, decides what needs a response, and takes care of it: scheduling
meetings, answering requests, sending the documents people ask for, and
asking you first before anything is deleted or changed. You talk to it from
Telegram.

> Status: early access. Run it against a test Google account while you
> evaluate it.

---

## What Socio does

| Incoming email | What Socio does |
|---|---|
| "Can we schedule a 30-min interview?" | Finds a free slot, books it with the sender invited, replies with the exact time |
| "Could you send your resume?" | Finds it in your Google Drive and asks you on Telegram before sending it (tap *Always allow* once and it won't ask again) |
| "What's your notice period?" (Socio doesn't know) | Saves a ready-to-send Gmail draft with the gaps marked for you |
| "Can we move our call to Friday?" | Finds the meeting and asks you on Telegram; once you approve, it moves the event and replies |
| Newsletters, promotions, receipts, "thanks!", FYIs | Nothing. No reply is sent |

And from Telegram, you can just ask:

- *"send my resume to hr@company.com"*
- *"what's on my calendar this week?"*
- *"move my 3pm with Raj to Friday 11am"*
- *"trash the Swiggy email"*
- *"add Acme, PM, applied today to my applications sheet"*
- *"write a cover letter for Acme in a doc"*
- *"remind me to send the invoice on Friday"*

When something needs you (a draft to review, a change to approve), Socio
also adds it to **Google Tasks**, and completes it once you've acted.

## Safety model

- **You approve anything destructive.** Moving email to Trash and moving or
  cancelling events are only *proposed*. They run after you tap **Approve** in
  Telegram, and are re-validated at that moment. Trash is never a permanent
  delete (Gmail keeps it 30 days, and Socio's Gmail permission cannot
  permanently delete).
- **Emails are untrusted.** When handling an incoming email, Socio gets a
  restricted toolset: it can only reply inside that conversation, only to
  people already in it, and can never delete, search your other mail, or
  email new people.
- **Your Drive, your call.** Socio searches your whole Drive (only files you
  own, never ones others shared with you). It never sends a Drive file to
  someone who emailed asking for it without your approval, unless you've
  marked that file *Always allow*.
- **Append-only edits.** Socio can read Docs and Sheets and *add* to them
  (append text, add rows) but never deletes or overwrites anything. Sheet
  values are written as plain text, so content from an email can never
  become a formula. Docs, Sheets, and Tasks are only available on your
  instructions, never while handling an incoming email. Files Socio creates
  go into a "Socio" folder it creates in your Drive.
- **No duplicates.** Sends, bookings, and drafts are never retried
  blindly; every outcome is recorded so a restart never repeats work.
- **Owner-only bot.** The Telegram bot only responds to your chat id.

## Quick start

Requirements: Python 3.11+, a Google Cloud project with the Gmail, Google
Calendar, Google Drive, Google Docs, Google Sheets, and Google Tasks APIs enabled, an OAuth client (`credentials.json`,
type "Desktop app"), a Gemini API key, and a Telegram bot token.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # then fill in the values
# put credentials.json in the project root

python -m app.main --dry-run      # first run opens Google sign-in; shows what Socio would do
python -m app.main --google-login # one-time sign-in for Drive, Docs, Sheets, Tasks
python -m app.telegram_bot        # start the assistant
```

### Telegram setup

1. In Telegram, message **@BotFather** → `/newbot` → copy the token into
   `.env` as `TELEGRAM_BOT_TOKEN`.
2. Run `python -m app.telegram_bot` and send your bot any message. It replies
   with your chat id.
3. Put it in `.env` as `TELEGRAM_OWNER_CHAT_ID` and restart the bot.

Bot commands: `/run` (check inbox now), `/dryrun`, `/status`, `/approvals`,
`/new`, `/help`. The inbox is also checked every `AGENT_POLL_MINUTES`.

## Configuration

See [`.env.example`](.env.example).

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | LLM used for decisions and writing |
| `AGENT_OWNER_NAME` | Name used to sign emails |
| `AGENT_DRIVE_FOLDER` | Optional: limit Socio to one Drive folder (default: your whole Drive) |
| `AGENT_AUTO_TASKS` | Add a Google Task when something needs you (default on; `0` = off) |
| `AGENT_FILES_DIR` | Optional local documents folder (default `./documents`) |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_OWNER_CHAT_ID` | The only chat allowed to use the bot |
| `AGENT_POLL_MINUTES` | Inbox check interval for the bot (0 = off) |
| `AGENT_INBOX_QUERY` | Which emails to process (default: inbox, last 7 days, not from you) |

## Command line

```bash
python -m app.main                         # process new email now
python -m app.main --dry-run               # plan only: nothing is sent, booked, or saved
python -m app.main --status                # outcomes, open approvals, recent failures
python -m app.main --ask "trash the Swiggy email"   # same as typing in Telegram
python -m app.main --approve 3             # approve request #3
python -m app.main --reject 3              # reject request #3 (nothing changes)
```

## How it works

```
Telegram  ─┐
CLI       ─┼─► Runner ─► Decide (rules → LLM) ─► Agent (LangGraph) ─► Tools ─► Verify ─► State
Schedule  ─┘                                         │
                                                     └─► Approval requests ─► you ─► Execute
```

- **Runner**: for each new email, fetch the full thread, decide whether
  action is needed, let the agent act, verify the result, record it.
- **Decision**: Gmail metadata rules first (spam, promotions, mailing lists,
  auto-replies, no-reply senders) so obvious cases never reach the LLM.
- **Agent**: a single LangGraph tool-calling agent (Gemini) with per-intent
  playbooks for scheduling, information requests, documents, and follow-ups.
- **Reliability**: timeouts and bounded retries on every Google API call,
  rate-limit handling for the LLM, and restart recovery from local state.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project structure

```
app/
  main.py           CLI entry point
  telegram_bot.py   Telegram interface: instructions, approvals, scheduled checks
  google_api.py     Shared Google API timeouts, retries, token refresh
  agent/            Runner, LangGraph agent, decisions, prompts, approvals, state
  gmail/            Gmail auth, messages, threads, drafts, parsing
  calendar/         Availability, booking, find/move/cancel events
  drive/            Google Drive client
  workspace/        Shared Workspace sign-in, Google Docs, Google Sheets
  tasks/            Google Tasks
  files/            Document search: Drive folder + local folder
docs/
  ARCHITECTURE.md
```

## Roadmap

- Always-on hosting (service / cloud)
- Configurable working hours and time zones
