# Socio

**Your inbox and calendar, handled.** Socio is an AI assistant that reads
your Gmail, decides what needs a response, and prepares it: scheduling
meetings, answering requests, sending the documents people ask for. Nothing
reaches anyone else, and none of your files changes, until you tap once to
approve it. It learns from those taps. You talk to it from Telegram.

> Status: early access. Run it against a test Google account while you
> evaluate it.

---

## What Socio does

| Incoming email | What Socio does |
|---|---|
| "Can we schedule a 30-min interview?" | Finds a free slot and asks you on Telegram; once you approve, it books it with the sender invited and replies with the exact time |
| "Could you send your resume?" | Finds it in your Google Drive and asks you on Telegram before sending it |
| "Just to confirm, you join on 1 Nov?" | Writes the reply and asks you on Telegram: *Approve & Send*, *Edit in Gmail*, or *Reject* |
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

Several things in one message ("send my resume to X, add a row to my sheet,
and remind me Friday") are split into separate tasks that run in parallel,
and you get one combined report.

When something needs you (a draft to review, a change to approve), Socio
also adds it to **Google Tasks**, and completes it once you've acted.

## It learns from you

- **Only when you say so.** Approving, rejecting, or editing a request
  teaches Socio nothing by itself. A rejection ends quietly ("❌ Proposal
  rejected. No changes were made.") with an optional **+ Add Permanent Rule**
  button. Tap it and type why: a lasting rule ("I never take calls before
  noon") is saved and recalled whenever it's relevant; a one-off reason
  ("already talked to him", "I'm tired today") is kept for 24 hours only.
  Teach it directly with `/remember`, see what it knows with `/memories`, and
  remove something with `/forget`. Memories are stored on your machine and
  embedded with a small local model, so they never leave it.
- **Skills.** After a multi-step instruction works, `/save_skill NAME` saves
  its plan as a recipe in `skills/` (Gemini turns the names and dates into
  parameters), and a matching request reuses it. Socio never saves one on its
  own; after you've run the same kind of workflow 3 times it suggests it once.
  Recipes are plain Markdown, contain no code, and go through the same
  approvals. You can also add your own Python scripts there; each one runs
  only after you've approved its exact content. See
  [`skills/README.md`](skills/README.md).
- Memory only informs Socio. It never skips an approval.

When something fails, Telegram only gets a short message (for example
"⚠️ Google services are busy or unreachable right now. Please try again in a
few minutes."). The full error with its stack trace goes to `logs/agent.log`
on your computer.

## Safety model

- **Socio does the work; nothing reaches anyone else without your tap.**

  | Action | What Socio does |
  |---|---|
  | Any email: replies to incoming email, and emails you ask for in Telegram | Asks you (*Approve & Send* / *Edit in Gmail* / *Reject*); *Edit in Gmail* saves it as a draft for you to change and send |
  | Booking a new meeting | Asks you; the booking and its confirmation reply go out together |
  | Moving or cancelling a meeting | Asks you; then updates the event and replies |
  | Moving email to Trash | Asks you (only when you asked for it) |
  | Creating or adding to Google Docs / Sheets | Asks you, showing exactly what will be added |
  | Running one of your skill scripts for the first time (or after it changed) | Asks you, showing its code |
  | Google Tasks | Does it |

  Approved actions are re-validated at the moment you tap (for example, the
  slot must still be free). Trash is never a permanent delete (Gmail keeps it
  30 days, and Socio's Gmail permission cannot permanently delete).
- **Emails are untrusted.** When handling an incoming email, Socio gets a
  restricted toolset: it can only reply inside that conversation, only to
  people already in it, and can never delete, search your other mail, or
  email new people. Email content reaches the model only inside
  `<untrusted_email_body>` tags (sanitized, with tags in the email escaped),
  and the model is told never to follow instructions found there.
- **No double-booked slots.** A time proposed for a booking or a move is held
  locally for 24 hours (or until you decide), so Socio won't propose the same
  slot to two people while you haven't approved either yet.
- **Your Drive, your call.** Socio searches your whole Drive (only files you
  own, never ones others shared with you). It never emails a file to anyone
  without your approval.
- **Append-only edits.** Socio can read Docs and Sheets and *add* to them
  (append text, add rows, after your approval) but never deletes or
  overwrites anything. Sheet
  values are written as plain text, so content from an email can never
  become a formula. Docs, Sheets, and Tasks are only available on your
  instructions, never while handling an incoming email. Files Socio creates
  go into a "Socio" folder it creates in your Drive.
- **No duplicates.** Sends, bookings, and drafts are never retried
  blindly; every outcome is recorded so a restart never repeats work. Only
  one inbox run happens at a time, even if you run the CLI while the bot is
  running.
- **Parallel tasks stay fenced.** Each task of a split instruction runs with
  only the tools it needs and none of your conversation, and its changes go
  through the same approvals. If two tasks propose the same change, you get
  one card.
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

The first time Socio needs its memory, it downloads a small embedding model
(snowflake-arctic-embed-s, about 130 MB) into `.cache/fastembed`. After that
everything runs offline on your CPU.

### Telegram setup

1. In Telegram, message **@BotFather** → `/newbot` → copy the token into
   `.env` as `TELEGRAM_BOT_TOKEN`.
2. Run `python -m app.telegram_bot` and send your bot any message. It replies
   with your chat id.
3. Put it in `.env` as `TELEGRAM_OWNER_CHAT_ID` and restart the bot.

Bot commands: `/run` (check inbox now), `/dryrun`, `/status`, `/approvals`,
`/new`, `/remember TEXT`, `/memories`, `/forget ID`, `/save_skill NAME`,
`/skills`, `/forget_skill NAME`, `/help`. The inbox is also checked every
`AGENT_POLL_MINUTES`.

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
| `AGENT_MEMORY` | Remember the rules you add and recall them (default on; `0` = off) |
| `AGENT_MAX_WORKERS` | Tasks of one instruction run in parallel (default 3) |
| `AGENT_LLM_RPM` | Gemini requests per minute, shared by all tasks (default 14; free tier is 15) |
| `AGENT_SKILLS_DIR` | Where skills live (default `./skills`) |
| `AGENT_MODEL_CACHE` | Where the embedding model is stored (default `./.cache/fastembed`) |
| `AGENT_LOG_DIR` | Where `agent.log` goes (default `./logs`, gitignored) |

## Command line

```bash
python -m app.main                         # process new email now
python -m app.main --dry-run               # plan only: nothing is sent, booked, or saved
python -m app.main --status                # outcomes, open approvals, recent failures
python -m app.main --ask "trash the Swiggy email"   # same as typing in Telegram
python -m app.main --approve 3             # approve request #3
python -m app.main --reject 3              # reject request #3 (nothing changes)
python -m app.main --draft 3               # move held email #3 to Gmail Drafts (not sent)
python -m app.main --remember "no calls before noon"   # teach a preference
python -m app.main --memories              # what Socio has learned
python -m app.main --forget 7              # forget memory #7
python -m app.main --skills                # saved skills
python -m app.main --save-skill weekly-prep   # save the last successful multi-step instruction
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
- **Agent**: a LangGraph tool-calling agent (Gemini) with per-intent
  playbooks for scheduling, information requests, documents, and follow-ups.
- **Supervisor**: splits a multi-part instruction into a task graph, runs
  independent tasks in parallel worker agents (each with a clean context and
  only its tools), and merges their results into one report.
- **Memory**: SQLite for short-term state; your learned preferences with local
  embeddings (top 3 recalled per email or instruction); reusable skills in
  `skills/`.
- **Reliability**: timeouts and bounded retries on every Google API call,
  rate-limit handling for the LLM, and restart recovery from local state.
  An error gateway logs full details to `logs/agent.log` and shows the owner
  only predefined messages.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project structure

```
app/
  main.py           CLI entry point
  telegram_bot.py   Telegram interface: instructions, approvals, scheduled checks
  google_api.py     Shared Google API timeouts, retries, token refresh
  errors.py         Error gateway: logs/agent.log + short owner-facing messages
  agent/            Runner, LangGraph agent, supervisor + workers, decisions, prompts, approvals, state
  memory/           Long-term memory (local embeddings), 24 h notes, owner rules, skills
  gmail/            Gmail auth, messages, threads, drafts, parsing
  calendar/         Availability, booking, find/move/cancel events
  drive/            Google Drive client
  workspace/        Shared Workspace sign-in, Google Docs, Google Sheets
  tasks/            Google Tasks
  files/            Document search: Drive folder + local folder
skills/             Saved procedures (recipes) and your own scripts; personal, gitignored
logs/               agent.log (full error details); gitignored
docs/
  ARCHITECTURE.md
```

## Roadmap

- Always-on hosting (service / cloud)
- Configurable working hours and time zones
