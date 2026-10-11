"""
Supervisor-worker harness for owner instructions (fan-out / fan-in).

    owner instruction (Telegram / --ask)
      -> recall long-term memories (top 3) and matching skills
      -> planner: one LLM call returning a structured PlannerOutput: a list of
         SubTask(task_id, target_domain, goal_contract, context_slice,
         tools_allowed, depends_on) forming a DAG, or use_skill (name, params)
         to reuse a saved recipe. A request that touches several domains must
         come back as several tasks (one corrective retry, then the single agent)
      -> workers run concurrently (thread pool of AGENT_MAX_WORKERS, default 3),
         each in a clean context with only its domain's tools, on the worker
         model chain (llm_router: Mistral -> Groq -> Gemini)
      -> fan-in: strict JSON payloads -> aggregator closes duplicate approvals,
         checkpoints the run in SQLite, and writes a short plain report: only failures
         and findings the approval cards do not show (often nothing: cards only)
      -> skills are never saved automatically: /save_skill NAME packages the last
         successful plan, and after the same tool graph succeeds 3 times the report
         offers it once

Routing by intent (no model call): questions and status checks ("Have you sent the
sheet?") always run the single owner-mode agent, which has the conversation; only
action commands with several parts are planned. A one-task instruction runs the
single agent as before. Inbox emails are never fanned out (developer decision): they
stay one at a time in app.agent.runner.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.agent.approvals import merge_same_emails
from app.agent.graph import AGENT_TOOLS, create_agent_graph
from app.calendar.tools import INDIA_TZ
from app.agent.llm import get_llm, invoke_llm
from app.agent.prompts import CARDS_ONLY
from app.agent.state import AgentStateStore
from app.agent.plain_text import plain_reply, without_internal_ids, without_sent_repeat  # noqa: F401  (re-exported for callers and tests)
from app.agent.taskgraph import DOMAIN_TOOLS, MAX_TASKS, PlanError, covered_domains, normalize_plan, topological_order
from app.agent.workers import NEEDS_INPUT, run_worker
from app.errors import plain_message, report_failure
from app.memory.episodic import notes_block
from app.memory.semantic import Memory, format_memories, recall
from app.memory.skills import Skill, SkillError, expand_skill, find_skills, get_skill, save_recipe, slugify


logger = logging.getLogger(__name__)

MAX_WORKERS = max(1, int(os.getenv("AGENT_MAX_WORKERS", "3")))
OWNER_RECURSION_LIMIT = 30
# A rejected plan goes back to the planner once with the reason.
PLAN_ATTEMPTS = 2
# Tools a worker may be given (all owner-mode tools; gates still apply inside them).
WORKER_TOOLS = {agent_tool.name for agent_tool in AGENT_TOOLS}

# Only instructions that look like several jobs are worth a planning call.
MULTI_PART = re.compile(
    r"(\n|;|\b(and then|then|also|after that|as well as|plus)\b|\d[.)]\s|,\s*and\b|\band\s+(send|email|reply|add|append|create|write|"
    r"move|book|schedule|cancel|trash|remind|find|check|tell|share|draft|update|list)\b)",
    re.IGNORECASE,
)

# Which domains an instruction mentions. Two or more means the plan must have two
# or more tasks (one per domain). A file to attach is part of the email.
REQUEST_DOMAINS = {
    "email": re.compile(r"\b(e-?mails?|mail|inbox|repl(y|ies|ied)|send|sent|forward|threads?)\b", re.IGNORECASE),
    "calendar": re.compile(r"\b(calendar|meetings?|events?|schedul\w*|availab\w*|free (time|slots?)|slots?|book|reschedul\w*|"
                           r"appointments?|invites?)\b", re.IGNORECASE),
    "workspace": re.compile(r"\b(docs?|documents?|sheets?|spreadsheets?|drive)\b", re.IGNORECASE),
    "tasks": re.compile(r"\b(tasks?|remind\w*|to-?dos?)\b", re.IGNORECASE),
    "news": re.compile(r"\b(news|headlines?)\b", re.IGNORECASE),
}

# "every day at 9", "each Monday", "daily at 7", "at 9 am on weekdays": a request to set up a
# routine. It is one schedule_task call for the single agent, never a plan: the words inside
# it ("send me ... news every day") would otherwise read as several domains. "daily" / "weekly"
# alone are not enough ("the weekly prep", "my daily standup" are names): they need a time.
_TIME_WORD = r"\d{1,2}(?::\d\d)?\s*(?:am|pm|a\.m\.|p\.m\.)?"
RECURRING_REQUEST = re.compile(
    r"\b(?:every|each)\s+(?:other\s+)?(?:day|morning|evening|night|week|weekday|weekend|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b"
    rf"|\b(?:daily|weekly|on weekdays)\s+at\s+{_TIME_WORD}"
    rf"|\bat\s+{_TIME_WORD}\s+(?:daily|weekly|on weekdays)\b",
    re.IGNORECASE,
)


# Work ON a Doc / Sheet (not just sending it): "create a sheet", "add a row to the sheet".
WORKSPACE_WORK = re.compile(
    r"\b(create|make|new|add|append|write|update|edit|insert|put|read|summari[sz]e|rename|copy)\b", re.IGNORECASE
)


def request_domains(text: str) -> set[str]:
    domains = {domain for domain, pattern in REQUEST_DOMAINS.items() if pattern.search(text or "")}
    # "Send the budget sheet to Sam": the sheet is what the email carries (one email task that
    # finds the file and sends its link), not a second job. Developer report 2026-10-11: split
    # into two tasks, the email worker never got the file and wrote "[link]" with no link.
    if {"email", "workspace"} <= domains and not WORKSPACE_WORK.search(text or ""):
        domains.discard("workspace")
    return domains


# Intent: an action command has a clause that starts with an action verb ("Create ...",
# "... and send it to X", "Can you book ..."). Anything else is a question or a status
# check ("Have you sent the sheet?", "What's on my calendar?"): it goes to the single
# agent, which has the conversation, and never to the planner (splitting a question into
# tasks loses the context it is about). No model call: a missed action only costs the
# fan-out (the single agent has every tool); a question taken for an action would be split.
_CLAUSE_START = r"(?:^|[\n.;:!?,]|\b(?:and|then|also|but|plus|after that|if not|otherwise)\b)"
_POLITE = (
    r"(?:(?:please|pls|plz|kindly|just|now|also|then|and|ok|okay)\s+|"
    r"(?:can|could|would|will)\s+(?:you|u)\s+(?:please\s+)?|"
    r"i\s+(?:want|need|would like|'d like|wanted)\s+(?:you\s+)?to\s+|help\s+me\s+|go ahead and\s+)*"
)
_ACTION_VERB = (
    r"(?:create|make|draft|write|send|e-?mail|mail|reply|respond|forward|schedule|book|reschedule|move|cancel|"
    r"delete|remove|trash|archive|add|append|put|insert|remind|share|set\s?up|mark|complete|close|update|edit|"
    r"save|remember|invite|attach|run|prepare|organi[sz]e|fill|notify|ask|tell(?!\s+me\b)|let\s+\w+\s+know|"
    r"block|rename|upload)"
)
ACTION_REQUEST = re.compile(rf"{_CLAUSE_START}\s*{_POLITE}{_ACTION_VERB}\b", re.IGNORECASE)
# Reads like a question or a request for information ("Have you sent ...", "What's on ...",
# "Check whether ...", "Tell me ...", or a trailing "?").
QUESTION_LIKE = re.compile(
    r"^\s*(?:(?:hey|hi|hello|ok|okay|so|socio)\b[\s,!.]*)*"
    r"(?:what|what's|whats|when|where|who|whom|whose|which|why|how|is|are|am|was|were|"
    r"(?:do|does|did|have|has|had)\s+(?:i|you|we|they|he|she|it|my|your|socio|anyone|there)|"
    r"can i|could i|should|shall|any|tell me|show me|list|give me|check|find|search|look|let me know|status)\b",
    re.IGNORECASE,
)


def is_action_request(text: str) -> bool:
    """
    False for a question or status check, True for anything else. An action clause
    ("..., and send it to X", "If not, send it") makes it an action even inside a question.
    """

    text = (text or "").strip()
    if ACTION_REQUEST.search(text):
        return True
    return not (QUESTION_LIKE.match(text) or text.endswith("?"))


# What a reply message is about. The bot shows each kind as its own message so an answer,
# a problem, and a question never share one text box (developer request 2026-10-11).
ANSWER, DONE, PROBLEM, QUESTION = "answer", "done", "problem", "question"
KIND_PREFIX = {ANSWER: "", DONE: "✅ ", PROBLEM: "⚠️ ", QUESTION: "❓ "}


@dataclass
class ReplyPart:
    kind: str   # ANSWER, DONE, PROBLEM, or QUESTION
    text: str


def reply_text(parts: list[ReplyPart]) -> str:
    """All parts as one plain text (CLI, follow-up context), each line marked by its kind."""

    lines = []
    for part in parts:
        prefix = KIND_PREFIX.get(part.kind, "")
        lines.append(part.text if not prefix or part.text.startswith(prefix.strip()) else prefix + part.text)
    return "\n".join(lines)


@dataclass
class OwnerCommandResult:
    reply: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    approval_ids: list[int] = field(default_factory=list)
    run_id: int | None = None
    parts: list[ReplyPart] = field(default_factory=list)
    # The single agent's tool calls ({"name", "args"}), for the conversation anchor; never shown.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


# ----- planner output (one structured answer: the whole plan) ----------------------------
# Not one tool call per task: Gemini tends to emit only the first call and wait for its
# result, which silently collapsed multi-part plans into one task (§4c).


class SubTask(BaseModel):
    """One atomic worker task: one domain, only that domain's tools."""

    task_id: str = Field(description="Short unique id such as t1, t2.")
    label: str = Field(
        default="",
        description="A few everyday words naming this part for the person, e.g. 'the email to Priya', "
        "'the rent reminder'. Used only if it fails.",
    )
    target_domain: Literal["email", "calendar", "workspace", "tasks", "memory", "script", "news", "routines"] = Field(
        description="The one domain this task works in. The worker only gets that domain's tools."
    )
    goal_contract: str = Field(description="What the worker must achieve and what counts as done. Self-contained.")
    context_slice: str = Field(
        default="",
        description="Every fact the worker needs (names, email addresses, dates, file or document names, which "
        "email or event, the owner's relevant preferences). The worker sees nothing else.",
    )
    tools_allowed: list[str] = Field(description="Only the tool names this task needs, from its domain's tool list.")
    depends_on: list[str] = Field(default_factory=list, description="task_ids whose results this task needs. Empty if independent.")


class PlannerOutput(BaseModel):
    """Socio's plan for the owner's instruction."""

    tasks: list[SubTask] = Field(
        default_factory=list,
        description="Every atomic task, one per domain or step. Empty only when use_skill is set.",
    )
    question: str = Field(
        default="",
        description="Only when NOTHING in the instruction can start without an answer (e.g. who to send it to): one "
        "short question for the person, and tasks empty. Never for details that have a sensible default.",
    )
    use_skill: str = Field(default="", description="The name of a saved skill that fits the whole instruction, or empty.")
    skill_params_json: str = Field(
        default="{}", description='With use_skill: its parameters as a JSON object, e.g. {"week": "5-9 Oct 2026"}.'
    )


class PlannedTasks(BaseModel):
    """The planner's tasks for one instruction. Rejects a monolithic plan for a multi-domain request."""

    request_domains: set[str]
    tasks: list[SubTask]

    @model_validator(mode="after")
    def _decomposed(self) -> "PlannedTasks":
        covered = set().union(*(covered_domains(task.target_domain, task.tools_allowed) for task in self.tasks))
        if len(self.request_domains) >= 2 and len(self.tasks) < 2 and len(covered) < 2:
            raise ValueError(
                f"The instruction touches {len(self.request_domains)} domains ({', '.join(sorted(self.request_domains))}) "
                f"but the plan has {len(self.tasks)} task. Monolithic tasks are forbidden: split it into one task per domain."
            )
        return self


PLANNER_PROMPT = """You are Socio's supervisor. You plan; you never act yourself.
Return the plan for the owner's instruction: the full list of worker tasks, all at once.
Today is {today} (India Standard Time).

CRITICAL ARCHITECTURAL RULE (§4c):
You are a pure Coordinator and Dispatcher, NEVER a monolithic executor.
When any part of the instruction touches more than one domain (email, calendar, workspace = Drive
files / Docs / Sheets, tasks = Google Tasks reminders), you MUST decompose it into distinct, atomic
tasks in the tasks list: one task per domain. A task that bundles several domains is STRICTLY
FORBIDDEN and the plan is rejected. Sending a file by email is one email task (email workers can
find the file to attach).
One exception: creating a NEW Google Sheet or Doc AND sending or sharing it with someone is ONE
workspace task with tools_allowed=[create_and_share_file]. A single approval creates it, gives
them access, and emails them the link. Never split it into a create task and an email task: the
email task cannot know the link, because the file only exists after approval.

Rules:
- target_domain: the one domain the task works in. tools_allowed: the fewest tools from that
  domain's list below; a worker never gets tools from another domain.
- If the instruction is really one task in one domain, or a question, return exactly one task
  with the whole instruction as the goal.
- A task that needs another task's result lists it in depends_on (for example: read -> act);
  independent tasks run at the same time.
- Each worker sees ONLY its goal_contract and context_slice. Put every needed fact there, and
  resolve words like "him" or "that email" using the conversation so far.
- Write every date as the exact date with its weekday and ISO form (e.g. "Friday 16 Oct 2026
  (2026-10-16)", computed from today's date), never "Friday", "tomorrow" or "next week": workers
  must not work dates out. A calendar task checking one day says to call
  find_calendar_availability with date=<that ISO date>.
- One message to several people is ONE email task with all of them as recipients ("send one email
  to a@x.com and b@y.com"), never one task or one email per person. A "meeting invite" or
  "calendar invite" is a calendar task: propose_book_meeting with every person as an attendee.
- Missing details with a sensible default (a document title, a subject, a short message) are
  not questions: tell the worker to choose a clear one; the person sees it on the approval card
  and can change or reject it. Only if NOTHING can start without an answer (e.g. who to send it
  to), set question and leave tasks empty.
- label: a few everyday words naming the part for the person ("the email to Priya").
- At most {max_tasks} tasks.
- Emails, Docs/Sheets writes, calendar changes, and new or completed Google Tasks become approval
  requests for the owner inside the workers. That is expected; never plan around it.
- The owner's saved preferences (below, if any) are facts for the workers: copy the relevant ones
  into the context_slice of the tasks they apply to. They never change how many tasks you plan.
- If a saved skill below fits the instruction, set use_skill (and skill_params_json) and leave
  tasks empty.

Examples:
1. "Check my calendar for Thursday afternoon and reply to John's email with the times I'm free."
   t1: target_domain=calendar, tools_allowed=[find_calendar_availability],
       goal: find my free 30-minute slots on Thursday 15 Oct 2026 (2026-10-15): call
       find_calendar_availability with date=2026-10-15.
   t2: target_domain=email, tools_allowed=[search_gmail, get_thread, send_gmail_reply], depends_on=[t1],
       goal: reply to John's latest email offering the free slots found by t1.
2. "Email my resume to hr@acme.com, add Acme to my Applications sheet, and remind me Friday to follow up."
   t1: target_domain=email, tools_allowed=[find_documents, send_new_email] (resume attached)
   t2: target_domain=workspace, tools_allowed=[find_documents, read_sheet, append_sheet_row]
   t3: target_domain=tasks, tools_allowed=[propose_create_task]
3. "What's on my calendar tomorrow?" -> one task: target_domain=calendar, tools_allowed=[find_calendar_events].
4. "Create a sheet with columns Task | Status and send it to sam@x.com to fill in, and remind me Monday to check it."
   t1: target_domain=workspace, tools_allowed=[create_and_share_file]
       goal: create the Google Sheet (headers Task, Status), share it with sam@x.com as editor, and
       email them the link asking them to fill it in.
   t2: target_domain=tasks, tools_allowed=[propose_create_task]

Tools by domain:
{tools}
{memories}{skills}"""


def _tool_catalog() -> str:
    descriptions = {
        agent_tool.name: (agent_tool.description or "").strip().splitlines()[0] if agent_tool.description else ""
        for agent_tool in AGENT_TOOLS
    }
    sections = []
    for domain, names in DOMAIN_TOOLS.items():
        lines = [f"- {name}: {descriptions[name]}" for name in sorted(names) if name in descriptions]
        if lines:
            sections.append(f"{domain}:\n" + "\n".join(lines))
    return "\n".join(sections)


def _skill_catalog(skills: list[tuple[Skill, float]]) -> str:
    if not skills:
        return ""
    lines = [
        f"- {skill.name}: {skill.description} (parameters: {', '.join(skill.parameters) or 'none'})"
        for skill, _ in skills
    ]
    return "\n\nSaved skills that may fit:\n" + "\n".join(lines)


def _history_text(history: list | None) -> str:
    if not history:
        return ""
    turns = []
    for message in history[-6:]:
        who = "Owner" if isinstance(message, HumanMessage) else "Socio"
        turns.append(f"{who}: {str(message.content)[:500]}")
    return "Conversation so far:\n" + "\n".join(turns) + "\n\n"


@dataclass
class Plan:
    tasks: list[dict[str, Any]]
    skill: str = ""
    question: str = ""   # the planner needs an answer before anything can start


def plan_request(
    text: str,
    history: list | None = None,
    memories: list[Memory] | None = None,
    skills: list[tuple[Skill, float]] | None = None,
    context: str = "",
) -> Plan:
    """
    The planner call. A plan that fails validation (a monolithic task for a
    multi-domain request, an unknown tool or domain, a cycle) is sent back once
    with the reason. Raises PlanError (or SkillError) if it still cannot be run.
    context is the chat's <active_context> block (app.memory.context_anchor), if any.
    """

    memory_block = "\n\n".join(block for block in (format_memories(memories or []), notes_block()) if block)
    system = PLANNER_PROMPT.format(
        today=datetime.now(INDIA_TZ).strftime("%A %d %B %Y"),
        max_tasks=MAX_TASKS,
        tools=_tool_catalog(),
        memories=f"\n{memory_block}" if memory_block else "",
        skills=_skill_catalog(skills or []),
    )
    planner = get_llm().with_structured_output(PlannerOutput)
    request = f"{_history_text(history)}Instruction from the owner:\n{text}" + (f"\n\n{context}" if context else "")
    messages = [SystemMessage(content=system), HumanMessage(content=request)]

    for attempt in range(PLAN_ATTEMPTS):
        try:
            return _plan_from_output(invoke_llm(planner, messages), text, skills)
        except (PlanError, OutputParserException, ValidationError) as exc:
            reason = _rejection_reason(exc)
            if attempt + 1 == PLAN_ATTEMPTS:
                raise PlanError(reason) from exc
            logger.warning("Plan rejected (%s); asking the planner once more.", reason)
            messages = [*messages[:2], HumanMessage(content=(
                f"Your previous plan was rejected: {reason}\nReturn the corrected plan."
            ))]
    raise PlanError("No plan.")  # unreachable: the loop returns or raises


def _rejection_reason(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors())
    return str(exc)[:500]


def _plan_from_output(output: Any, text: str, skills: list[tuple[Skill, float]] | None) -> Plan:
    if output is None:
        return Plan([])
    if not isinstance(output, PlannerOutput):
        output = PlannerOutput.model_validate(output)

    if output.use_skill.strip():
        by_name = {skill.name: skill for skill, _ in skills or []}
        skill = by_name.get(output.use_skill.strip())
        if skill is None:
            raise PlanError(f"The planner chose an unknown skill {output.use_skill!r}.")
        try:
            params = json.loads(output.skill_params_json or "{}")
        except json.JSONDecodeError as exc:
            raise PlanError(f"Skill parameters are not valid JSON: {exc}") from exc
        if not isinstance(params, dict):
            raise PlanError("Skill parameters must be a JSON object.")
        return Plan(expand_skill(skill, params, WORKER_TOOLS), skill=skill.name)

    if not output.tasks:
        return Plan([], question=" ".join((output.question or "").split()))
    PlannedTasks(request_domains=request_domains(text), tasks=output.tasks)   # raises ValidationError if monolithic
    return Plan(normalize_plan([task.model_dump() for task in output.tasks], WORKER_TOOLS))


# ----- fan-out ---------------------------------------------------------------------------


def run_plan(
    tasks: list[dict[str, Any]],
    *,
    run_id: int,
    store: AgentStateStore,
    dry_run: bool = False,
    worker: Callable[..., dict[str, Any]] = run_worker,
    max_workers: int = MAX_WORKERS,
) -> dict[str, dict[str, Any]]:
    """
    Run the DAG: every task whose dependencies succeeded starts as soon as a worker
    slot is free. A task whose dependency failed is not started (FAILED, nothing changed).
    Returns {task_id: payload}; never raises for a worker's failure.
    """

    by_id = {task["task_id"]: task for task in tasks}
    waiting = dict(by_id)
    results: dict[str, dict[str, Any]] = {}

    def finish(task_id: str, payload: dict[str, Any]) -> None:
        results[task_id] = payload
        store.update_worker_task(run_id, task_id, payload["status"], payload)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="socio-worker") as pool:
        running: dict[Any, str] = {}

        while waiting or running:
            for task_id, task in list(waiting.items()):
                # A step waiting for the person's answer blocks what depends on it, like a failure.
                failed = [d for d in task["depends_on"] if results.get(d, {}).get("status") in {"FAILED", NEEDS_INPUT}]
                if failed:
                    del waiting[task_id]
                    asked = all(results[d]["status"] == NEEDS_INPUT for d in failed)
                    finish(task_id, {
                        "status": "FAILED",
                        "modified_entities": [],
                        "diff_summary": f"Not started: it needs {', '.join(failed)}, which "
                                        f"{'is waiting for an answer' if asked else 'failed'}. Nothing was changed.",
                        "not_started": "question" if asked else "failure",
                    })
                elif all(d in results for d in task["depends_on"]) and len(running) < max_workers:
                    del waiting[task_id]
                    upstream = {d: results[d] for d in task["depends_on"]}
                    store.update_worker_task(run_id, task_id, "running")
                    running[pool.submit(worker, task, upstream, dry_run=dry_run)] = task_id

            if not running:
                if waiting:  # cannot happen for a validated DAG; never spin
                    for task_id in list(waiting):
                        finish(task_id, {"status": "FAILED", "modified_entities": [], "diff_summary": "Not started: unresolvable dependencies.", "not_started": True})
                    waiting.clear()
                break

            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = running.pop(future)
                try:
                    payload = future.result()
                except Exception as exc:  # a worker function that raised anyway
                    report_failure(exc, "worker.crash", run_id=run_id, task_id=task_id)
                    payload = {"status": "FAILED", "modified_entities": [], "diff_summary": f"It stopped because of an error. {plain_message(exc)}"}
                if not dry_run and payload["status"] == "FAILED":   # a question is not a failure
                    store.add_failure(f"worker:{task_id}", payload["diff_summary"])
                finish(task_id, payload)

    return results


# ----- fan-in ------------------------------------------------------------------------------


def resolve_overlaps(results: dict[str, dict[str, Any]], store: AgentStateStore, dry_run: bool = False) -> list[str]:
    """
    Two workers proposing the same change (same event, document, recipient...) would
    show the owner two cards for one thing: keep the first approval, close the others
    quietly (the owner sees one card; how the work was split is not their concern).
    Completed changes cannot be undone, so overlaps among them are reported.
    """

    notes: list[str] = []
    first_approval: dict[str, tuple[str, int]] = {}
    first_change: dict[str, str] = {}

    for task_id in sorted(results):
        for entity in results[task_id]["modified_entities"]:
            key = entity.get("key")
            if entity.get("kind") == "approval" and entity.get("approval_id"):
                if key in first_approval and first_approval[key][0] != task_id:
                    keep = first_approval[key][1]
                    entity["duplicate_of"] = keep
                    if not dry_run:
                        store.resolve_pending_approval(entity["approval_id"], f"Duplicate of #{keep} from the same instruction")
                else:
                    first_approval.setdefault(key, (task_id, entity["approval_id"]))
            elif entity.get("status") == "completed":
                if key in first_change and first_change[key] != task_id:
                    notes.append("I changed the same item twice; please check it.")
                first_change.setdefault(key, task_id)
    return notes


def build_parts(tasks: list[dict[str, Any]], results: dict[str, dict[str, Any]], notes: list[str], dry_run: bool = False) -> list[ReplyPart]:
    """
    What the owner gets next to the approval cards, one part per item, each marked by
    intent (answer / problem / question) so the bot can show them as separate messages.

    No header, no list of parts, no goals: the owner cares whether it got done. A part
    that produced an approval card says nothing (the card is the answer); a step whose
    result fed a later step says nothing either. What is left: failures (always),
    questions a step needs answered, and what a final step found or did directly.
    Empty when the cards say it all.
    """

    needed_later = {dependency for task in tasks for dependency in task["depends_on"]}
    parts: list[ReplyPart] = []
    skipped = {"failure": False, "question": False}

    def named(task: dict[str, Any], text: str) -> str:
        label = task.get("label", "")
        return f"{label[:1].upper() + label[1:]}: {text}" if label else text

    for task in tasks:
        payload = results.get(task["task_id"], {"status": "FAILED", "diff_summary": "It didn't run.", "modified_entities": []})
        summary = plain_reply(payload.get("diff_summary", ""))
        proposed = any(entity.get("kind") == "approval" for entity in payload.get("modified_entities", []))

        if payload.get("not_started"):
            skipped["question" if payload["not_started"] == "question" else "failure"] = True
        elif payload["status"] == NEEDS_INPUT or (payload["status"] == "SUCCESS" and not proposed and "?" in summary):
            # A worker that ended by asking something is asking, whatever status it wrote.
            parts.append(ReplyPart(QUESTION, named(task, summary) if summary else named(task, "What should I do here?")))
        elif payload["status"] == "FAILED":
            parts.append(ReplyPart(PROBLEM, named(task, summary)))
        elif not proposed and task["task_id"] not in needed_later and summary:
            parts.append(ReplyPart(ANSWER, summary))

    if skipped["failure"]:
        parts.append(ReplyPart(PROBLEM, "I didn't do the steps that depended on it, so nothing else was changed."))
    if skipped["question"]:
        parts.append(ReplyPart(QUESTION, "I haven't done the steps that need this answer yet; reply and I'll continue."))
    parts += [ReplyPart(PROBLEM, note) for note in notes]
    if dry_run:
        parts.insert(0, ReplyPart(ANSWER, "🧪 Dry run: nothing was sent or changed."))
    return parts


def build_report(tasks: list[dict[str, Any]], results: dict[str, dict[str, Any]], notes: list[str], skill: str = "", dry_run: bool = False) -> str:
    """build_parts() as one text (CLI, the run record, the follow-up context)."""

    return reply_text(build_parts(tasks, results, notes, dry_run=dry_run))


# ----- procedural memory: skills are saved only when the owner asks -------------------------

SUGGEST_AFTER_SUCCESSES = 3


def workflow_signature(tasks: list[dict[str, Any]]) -> str:
    """
    The shape of a plan: each task's tools and its dependencies, in dependency order.

    Goals and task ids are left out, so "weekly prep for week 41" and "... week 42"
    count as the same workflow.
    """

    order = topological_order(tasks)
    position = {task_id: index for index, task_id in enumerate(order)}
    by_id = {task["task_id"]: task for task in tasks}
    shape = [
        [sorted(by_id[task_id]["tools_allowed"]), sorted(position[d] for d in by_id[task_id]["depends_on"])]
        for task_id in order
    ]
    return hashlib.sha256(json.dumps(shape).encode()).hexdigest()[:16]


_NAME_STOPWORDS = {
    "a", "an", "the", "and", "then", "also", "to", "for", "of", "in", "on", "my", "me", "please", "with",
    "it", "this", "that", "i", "is", "at", "from", "by", "as", "plus", "after",
}


def suggest_skill_name(request: str, tasks: list[dict[str, Any]]) -> str:
    """A short kebab-case name from the request's first meaningful words (no LLM call)."""

    words = [w for w in re.findall(r"[a-z]+", (request or "").lower()) if w not in _NAME_STOPWORDS and len(w) > 2]
    tools = sorted({tool for task in tasks for tool in task["tools_allowed"]})
    for candidate in ("-".join(words[:3]), "-".join(tools[:2])):
        try:
            return slugify(candidate)
        except SkillError:
            continue
    return "my-workflow"


def note_workflow_success(tasks: list[dict[str, Any]], request: str, run_id: int, store: AgentStateStore) -> str:
    """Count a successful multi-step run; on the third success of the same shape, offer /save_skill once."""

    try:
        signature = workflow_signature(tasks)
        stats = store.record_workflow_success(signature, run_id)
        if stats["successes"] >= SUGGEST_AFTER_SUCCESSES and not stats["suggested"] and store.mark_workflow_suggested(signature):
            name = suggest_skill_name(request, tasks)
            return f"\n💡 You've run this workflow {stats['successes']} times. Save as a skill? [/save_skill {name}]"
    except Exception as exc:
        logger.warning("Could not count the workflow: %s", exc)
    return ""


class SkillDraft(BaseModel):
    description: str = Field(description="One sentence: when to use this skill.")
    parameters: list[str] = Field(default_factory=list, description="snake_case names of the parts that change each time.")
    plan_json: str = Field(description="The plan as a JSON list of tasks, with {{parameter}} placeholders.")


SKILL_PROMPT = """The owner asked to save the procedure below as a reusable skill.
Generalize it: replace the parts that change each time (dates, names, amounts, addresses,
subjects) with {{{{parameter}}}} placeholders, list those parameters, and keep each task's
task_id, tools_allowed, and depends_on unchanged. Do not add or remove tasks.

Instruction: {request}
Plan: {plan}"""


@dataclass
class SavedSkill:
    name: str
    parameters: list[str]
    generalized: bool
    tasks: int


def save_last_plan(name: str, store: AgentStateStore | None = None) -> SavedSkill:
    """
    /save_skill NAME: package the last fully successful multi-step plan as skills/NAME.md.

    One Gemini call turns concrete values into {{parameters}}; if that fails or the
    result does not validate, the plan is saved exactly as it ran. Raises SkillError
    (with a message for the owner) when there is nothing to save or the name is taken.
    """

    store = store or AgentStateStore()
    slug = slugify(name)
    if get_skill(slug) is not None:
        raise SkillError(f"A skill named {slug} already exists. Pick another name or /forget_skill {slug} first.")

    run = store.last_successful_plan_run()
    if run is None:
        raise SkillError("There is no successful multi-step instruction to save yet.")

    request, plan = run["request"], run["plan"]
    notes = f"Saved by the owner from: {request[:300]}"
    try:
        draft = invoke_llm(
            get_llm().with_structured_output(SkillDraft),
            SKILL_PROMPT.format(request=request, plan=json.dumps(plan, ensure_ascii=False)),
        )
        generalized = json.loads(draft.plan_json)
        if not isinstance(generalized, list) or len(generalized) != len(plan):
            raise PlanError("The generalized plan changed the number of tasks.")
        skill = save_recipe(slug, draft.description or request, draft.parameters, generalized, WORKER_TOOLS, notes=notes, source_run=run["id"])
        return SavedSkill(skill.name, skill.parameters, True, len(skill.plan))
    except Exception as exc:
        logger.warning("Saving the plan as it ran (generalizing failed: %s)", exc)

    skill = save_recipe(slug, request, [], plan, WORKER_TOOLS, notes=notes, source_run=run["id"])
    return SavedSkill(skill.name, [], False, len(skill.plan))


# ----- entry point -----------------------------------------------------------------------------


def run_single_agent(
    text: str,
    *,
    history: list | None = None,
    dry_run: bool = False,
    memories: list[Memory] | None = None,
    context: str = "",
) -> OwnerCommandResult:
    """The plain owner-mode agent (one context, full owner toolset)."""

    from app.tasks.tracker import open_tasks_block

    instruction = f"Instruction from the owner:\n{text}"
    for block in (context, format_memories(memories or []), notes_block(), open_tasks_block()):
        if block:
            instruction += f"\n\n{block}"

    result = create_agent_graph().invoke(
        {
            "messages": list(history or []) + [HumanMessage(content=instruction)],
            "processed_message_ids": [],
            "actions": [],
            "pending_approvals": [],
            "failures": [],
            "dry_run": dry_run,
        },
        config={"recursion_limit": OWNER_RECURSION_LIMIT},
    )
    messages = result.get("messages", [])
    approval_ids = [approval["id"] for approval in result.get("pending_approvals", []) if "id" in approval]
    if not dry_run:
        approval_ids = merge_same_emails(approval_ids, AgentStateStore())
    actions = result.get("actions", [])
    reply = owner_reply(messages[-1].text if messages else "", approval_ids, dry_run=dry_run, actions=actions)
    reply = without_sent_repeat(reply, text)
    tool_calls = [
        {"name": call.get("name", ""), "args": call.get("args") or {}}
        for message in messages[len(history or []):]
        for call in (getattr(message, "tool_calls", None) or [])
    ]
    return OwnerCommandResult(reply=reply, actions=actions, approval_ids=approval_ids, parts=split_reply(reply), tool_calls=tool_calls)


# A paragraph that ends by asking the person something ("Which one should I reply to?").
_TRAILING_QUESTION = re.compile(r"(?s)^(?P<body>.*[.!:)\]])\s+(?P<question>[^.!?\n]*\?)\s*$")


def split_reply(reply: str) -> list[ReplyPart]:
    """
    The single agent's text as parts by intent: ✅ / ⚠️ lines are done / problem, a
    closing question becomes its own question part (so it is not buried under the
    answer), the rest is the answer. A reply that is only one kind stays one part.
    """

    reply = (reply or "").strip()
    if not reply:
        return []
    parts: list[ReplyPart] = []
    answer: list[str] = []
    for line in reply.splitlines():
        if line.startswith("✅"):
            parts.append(ReplyPart(DONE, line.removeprefix("✅").strip()))
        elif line.startswith("⚠️"):
            parts.append(ReplyPart(PROBLEM, line.removeprefix("⚠️").strip()))
        else:
            answer.append(line)
    text = "\n".join(answer).strip()
    paragraphs = re.split(r"\n\s*\n", text) if text else []
    question = ""
    if paragraphs:
        last = paragraphs[-1].strip()
        if len(paragraphs) > 1 and last.endswith("?"):
            question, paragraphs = last, paragraphs[:-1]
        else:
            match = _TRAILING_QUESTION.match(last)
            # Only a real answer followed by a question is split; "Not sure yet. Which one?" stays one message.
            if match and len(match.group("body")) >= 120:
                question, paragraphs[-1] = match.group("question").strip(), match.group("body").strip()
    text = "\n\n".join(p.strip() for p in paragraphs).strip()
    ordered = ([ReplyPart(ANSWER, text)] if text else []) + parts + ([ReplyPart(QUESTION, question)] if question else [])
    return ordered


_CARDS_ONLY = re.compile(rf"[\s`*\"'(\[]*{CARDS_ONLY}[\s`*\"'.)\]]*")


# Everyday names for direct actions, for "⚠️ Saving the rule didn't work."
ACTION_WORDS = {
    "remember_rule": "Saving the rule",
    "create_reply_draft": "Saving the reply draft",
    "send_gmail_reply": "Sending the reply",
    "send_new_email": "Sending the email",
    "create_document": "Creating the document",
    "append_to_document": "Adding to the document",
    "create_spreadsheet": "Creating the spreadsheet",
    "append_sheet_row": "Adding the row",
    "create_and_share_file": "Creating and sharing the file",
    "share_drive_file": "Sharing the file",
    "schedule_task": "Setting up the routine",
    "cancel_scheduled_task": "Stopping the routine",
}


def _direct_outcomes(actions: list[dict[str, Any]]) -> list[str]:
    """What the agent did without an approval card (or failed to do), one line each."""

    lines = []
    for action in actions:
        kind, status, args = action.get("type", ""), action.get("status", ""), action.get("args") or {}
        if status in {"failed", "unknown"}:
            what = ACTION_WORDS.get(kind, "One step")
            lines.append(f"⚠️ {what} didn't work." if status == "failed" else f"⚠️ {what} may not have worked (I can't tell whether it happened).")
        elif status == "completed" and kind == "remember_rule":
            lines.append(f"✅ Rule saved: {(action.get('result') or {}).get('summary') or args.get('rule', '')}")
        elif status == "completed" and kind == "create_reply_draft":
            lines.append("✅ Saved a reply draft in Gmail Drafts.")
        elif status == "completed" and kind == "schedule_task":
            lines.append((action.get("result") or {}).get("confirmation") or "✅ Routine scheduled.")
        elif status == "completed" and kind == "cancel_scheduled_task":
            lines.append(f"✅ Stopped the routine {(action.get('result') or {}).get('title', '')}.".replace("  ", " "))
        elif status == "completed":
            lines.append(f"✅ Done: {ACTION_WORDS.get(kind, kind.replace('_', ' ')).lower()}.")
    return lines


def owner_reply(text: str, approval_ids: list[int], dry_run: bool = False, actions: list[dict[str, Any]] | None = None) -> str:
    """
    The single agent's answer for the owner. Empty when the approval cards say it all
    (the model answered CARDS_ONLY), so the owner gets the cards and no preamble.
    If the model said CARDS_ONLY but also changed something directly (a Google Task) or
    a tool failed, those are listed instead: nothing done is ever left unreported.
    """

    text = plain_reply(text)
    said_cards_only = bool(_CARDS_ONLY.search(text))
    reply = _CARDS_ONLY.sub(" ", text).strip() if said_cards_only else text
    if not reply and said_cards_only:
        reply = "\n".join(_direct_outcomes(actions or []))
    # A new routine is confirmed in Socio's own words ("✅ Scheduled! I will prepare and send
    # your **X** daily at 09:00 AM IST."), not the model's paraphrase.
    confirmations = [
        (action.get("result") or {}).get("confirmation", "")
        for action in actions or []
        if action.get("type") == "schedule_task" and action.get("status") == "completed"
    ]
    confirmations = [line for line in confirmations if line and line not in reply]
    if confirmations:
        return "\n".join(confirmations + ([reply] if reply else []))
    if reply or approval_ids:
        return reply
    if said_cards_only:
        # The model claimed approval requests that do not exist: nothing was done.
        return (
            "🧪 Dry run: nothing was proposed or changed." if dry_run
            else "I'm not sure what you'd like me to do yet. Could you give me a few more details?"
        )
    return "Done."


def should_plan(text: str, skills: list[tuple[Skill, float]]) -> bool:
    """
    Only action commands are planned. Questions and status checks never are. A one-step
    action skips the planner too: it would return one task, which runs the single agent
    anyway (one model call saved, same result).
    """

    if not is_action_request(text) or RECURRING_REQUEST.search(text or ""):
        return False
    return bool(skills) or bool(MULTI_PART.search(text or "")) or len(request_domains(text)) >= 2


def handle_owner_request(
    text: str,
    *,
    dry_run: bool = False,
    history: list | None = None,
    store: AgentStateStore | None = None,
    context: str = "",
) -> OwnerCommandResult:
    """
    Plan, fan out, fan in, report. Questions and one-task requests run the single agent.

    context is the chat's <active_context> block (app.memory.context_anchor). It reaches the
    single agent and the planner; workers never see it (or the history), only their task.
    """

    store = store or AgentStateStore()
    memories = recall(text, store=store)
    skills = find_skills(text) if is_action_request(text) else []

    if not should_plan(text, skills):
        return run_single_agent(text, history=history, dry_run=dry_run, memories=memories, context=context)

    try:
        plan = plan_request(text, history, memories, skills, context=context)
    except (PlanError, SkillError) as exc:
        logger.warning("Plan rejected (%s); running the single agent instead.", exc)
        plan = Plan([])

    if plan.question and not plan.tasks and not plan.skill:
        # Nothing can start without an answer: ask first, do nothing.
        question = plain_reply(plan.question)
        return OwnerCommandResult(reply=question, parts=[ReplyPart(QUESTION, question)])

    if len(plan.tasks) <= 1 and not plan.skill:
        return run_single_agent(text, history=history, dry_run=dry_run, memories=memories, context=context)

    store.close_stale_supervisor_runs()
    run_id = store.start_supervisor_run(text)
    store.save_supervisor_plan(run_id, plan.tasks, skill=plan.skill)

    results = run_plan(plan.tasks, run_id=run_id, store=store, dry_run=dry_run)
    notes = resolve_overlaps(results, store, dry_run=dry_run)
    parts = build_parts(plan.tasks, results, notes, dry_run=dry_run)

    statuses = {payload["status"] for payload in results.values()}
    run_status = "done" if statuses == {"SUCCESS"} else ("failed" if statuses == {"FAILED"} else "partial")

    if run_status == "done" and not plan.skill and not dry_run:
        hint = note_workflow_success(plan.tasks, text, run_id, store).strip()
        if hint:
            parts.append(ReplyPart(ANSWER, hint))

    entities = [entity for payload in results.values() for entity in payload["modified_entities"]]
    approval_ids = [
        e["approval_id"] for e in entities
        if e.get("kind") == "approval" and e.get("approval_id") and not e.get("duplicate_of")
    ]
    approval_ids = merge_same_emails(list(dict.fromkeys(approval_ids)), store) if not dry_run else approval_ids
    if not parts and not approval_ids:
        parts = [ReplyPart(ANSWER, "Done.")]  # never silent: no card and nothing to report

    report = reply_text(parts)
    store.finish_supervisor_run(run_id, "dry_run" if dry_run else run_status, report)
    return OwnerCommandResult(reply=report, actions=entities, approval_ids=approval_ids, run_id=run_id, parts=parts)
