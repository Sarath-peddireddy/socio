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
         checkpoints the run in SQLite, and writes one report
      -> skills are never saved automatically: /save_skill NAME packages the last
         successful plan, and after the same tool graph succeeds 3 times the report
         offers it once

A one-task instruction (or a follow-up that needs the conversation) runs the
single owner-mode agent as before. Inbox emails are never fanned out
(developer decision): they stay one at a time in app.agent.runner.
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

from app.agent.graph import AGENT_TOOLS, create_agent_graph
from app.calendar.tools import INDIA_TZ
from app.agent.llm import get_llm, invoke_llm
from app.agent.prompts import CARDS_ONLY
from app.agent.state import AgentStateStore
from app.agent.taskgraph import DOMAIN_TOOLS, MAX_TASKS, PlanError, normalize_plan, topological_order
from app.agent.workers import run_worker
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
}


def request_domains(text: str) -> set[str]:
    return {domain for domain, pattern in REQUEST_DOMAINS.items() if pattern.search(text or "")}


@dataclass
class OwnerCommandResult:
    reply: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    approval_ids: list[int] = field(default_factory=list)
    run_id: int | None = None


# ----- planner output (one structured answer: the whole plan) ----------------------------
# Not one tool call per task: Gemini tends to emit only the first call and wait for its
# result, which silently collapsed multi-part plans into one task (§4c).


class SubTask(BaseModel):
    """One atomic worker task: one domain, only that domain's tools."""

    task_id: str = Field(description="Short unique id such as t1, t2.")
    target_domain: Literal["email", "calendar", "workspace", "tasks", "memory", "script"] = Field(
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
        if len(self.request_domains) >= 2 and len(self.tasks) < 2:
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

Rules:
- target_domain: the one domain the task works in. tools_allowed: the fewest tools from that
  domain's list below; a worker never gets tools from another domain.
- If the instruction is really one task in one domain, or a question, return exactly one task
  with the whole instruction as the goal.
- A task that needs another task's result lists it in depends_on (for example: read -> act);
  independent tasks run at the same time.
- Each worker sees ONLY its goal_contract and context_slice. Put every needed fact there, and
  resolve words like "him" or "that email" using the conversation so far.
- Write every date as the exact date with its weekday (e.g. "Friday 16 Oct 2026", computed from
  today's date), never "Friday", "tomorrow" or "next week": workers must not work dates out.
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
       goal: find my free 30-minute slots on Thursday 15 Oct 2026, 14:00-18:00 IST.
   t2: target_domain=email, tools_allowed=[search_gmail, get_thread, send_gmail_reply], depends_on=[t1],
       goal: reply to John's latest email offering the free slots found by t1.
2. "Email my resume to hr@acme.com, add Acme to my Applications sheet, and remind me Friday to follow up."
   t1: target_domain=email, tools_allowed=[find_documents, send_new_email] (resume attached)
   t2: target_domain=workspace, tools_allowed=[find_documents, read_sheet, append_sheet_row]
   t3: target_domain=tasks, tools_allowed=[propose_create_task]
3. "What's on my calendar tomorrow?" -> one task: target_domain=calendar, tools_allowed=[find_calendar_events].

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


def plan_request(
    text: str,
    history: list | None = None,
    memories: list[Memory] | None = None,
    skills: list[tuple[Skill, float]] | None = None,
) -> Plan:
    """
    The planner call. A plan that fails validation (a monolithic task for a
    multi-domain request, an unknown tool or domain, a cycle) is sent back once
    with the reason. Raises PlanError (or SkillError) if it still cannot be run.
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
    messages = [SystemMessage(content=system), HumanMessage(content=f"{_history_text(history)}Instruction from the owner:\n{text}")]

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
        return Plan([])
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
                failed = [d for d in task["depends_on"] if results.get(d, {}).get("status") == "FAILED"]
                if failed:
                    del waiting[task_id]
                    finish(task_id, {
                        "status": "FAILED",
                        "modified_entities": [],
                        "diff_summary": f"Not started: it needs {', '.join(failed)}, which failed. Nothing was changed.",
                    })
                elif all(d in results for d in task["depends_on"]) and len(running) < max_workers:
                    del waiting[task_id]
                    upstream = {d: results[d] for d in task["depends_on"]}
                    store.update_worker_task(run_id, task_id, "running")
                    running[pool.submit(worker, task, upstream, dry_run=dry_run)] = task_id

            if not running:
                if waiting:  # cannot happen for a validated DAG; never spin
                    for task_id in list(waiting):
                        finish(task_id, {"status": "FAILED", "modified_entities": [], "diff_summary": "Not started: unresolvable dependencies."})
                    waiting.clear()
                break

            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = running.pop(future)
                try:
                    payload = future.result()
                except Exception as exc:  # a worker function that raised anyway
                    report_failure(exc, "worker.crash", run_id=run_id, task_id=task_id)
                    payload = {"status": "FAILED", "modified_entities": [], "diff_summary": f"Worker crashed. {plain_message(exc)}"}
                if not dry_run and payload["status"] == "FAILED":
                    store.add_failure(f"worker:{task_id}", payload["diff_summary"])
                finish(task_id, payload)

    return results


# ----- fan-in ------------------------------------------------------------------------------


def resolve_overlaps(results: dict[str, dict[str, Any]], store: AgentStateStore, dry_run: bool = False) -> list[str]:
    """
    Two workers proposing the same change (same event, document, recipient...) would
    show the owner two cards for one thing: keep the first approval, close the others.
    Completed changes cannot be undone, so overlaps among them are only reported.
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
                    notes.append("Two parts of your request proposed the same change; you'll see it only once.")
                else:
                    first_approval.setdefault(key, (task_id, entity["approval_id"]))
            elif entity.get("status") == "completed":
                if key in first_change and first_change[key] != task_id:
                    notes.append("Two parts of your request changed the same item; please check it.")
                first_change.setdefault(key, task_id)
    return notes


# Tool results say "Approval request #12"; workers sometimes copy that despite the prompt.
_APPROVAL_NUMBER = re.compile(r"\b(approval request|approval|request)\s*#\d+", re.IGNORECASE)
_ASIDE_NUMBER = re.compile(r"\s*\((?:approval request|approval|request)?\s*#\d+\)", re.IGNORECASE)
_BARE_NUMBER = re.compile(r"\s*#\d+")


def without_internal_ids(text: str) -> str:
    """Owner-facing worker summaries carry no approval numbers (plain language, no ids)."""

    text = _ASIDE_NUMBER.sub("", text or "")
    text = _APPROVAL_NUMBER.sub(lambda m: "The approval request" if m.group(0)[0].isupper() else "the approval request", text)
    return _BARE_NUMBER.sub("", text)


def build_report(tasks: list[dict[str, Any]], results: dict[str, dict[str, Any]], notes: list[str], skill: str = "", dry_run: bool = False) -> str:
    header = f"{'🧪 Dry run - ' if dry_run else ''}🧭 I handled your request in {len(tasks)} parts"
    if skill:
        header += f" (using your saved skill '{skill}')"
    lines = [header + ":"]

    approvals: list[int] = []
    for task in tasks:
        payload = results.get(task["task_id"], {"status": "FAILED", "diff_summary": "Did not run."})
        mark = "✅" if payload["status"] == "SUCCESS" else "❌"
        goal = " ".join(task["goal_contract"].split())
        goal = goal if len(goal) <= 70 else goal[:69] + "…"
        lines.append(f"{mark} {goal}\n   {without_internal_ids(payload['diff_summary'])}")
        approvals += [
            e["approval_id"] for e in payload.get("modified_entities", [])
            if e.get("kind") == "approval" and e.get("approval_id") and not e.get("duplicate_of")
        ]

    if approvals:
        count = len(set(approvals))
        lines.append(f"⏳ {count} {'item is' if count == 1 else 'items are'} waiting for your approval (see the cards).")
    lines += [f"⚠️ {note}" for note in notes]
    return "\n".join(lines)


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


def run_single_agent(text: str, *, history: list | None = None, dry_run: bool = False, memories: list[Memory] | None = None) -> OwnerCommandResult:
    """The plain owner-mode agent (one context, full owner toolset)."""

    from app.tasks.tracker import open_tasks_block

    instruction = f"Instruction from the owner:\n{text}"
    for block in (format_memories(memories or []), notes_block(), open_tasks_block()):
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
    actions = result.get("actions", [])
    return OwnerCommandResult(
        reply=owner_reply(messages[-1].text if messages else "", approval_ids, dry_run=dry_run, actions=actions),
        actions=actions,
        approval_ids=approval_ids,
    )


_CARDS_ONLY = re.compile(rf"[\s`*\"'(\[]*{CARDS_ONLY}[\s`*\"'.)\]]*")


def _direct_outcomes(actions: list[dict[str, Any]]) -> list[str]:
    """What the agent did without an approval card (or failed to do), one line each."""

    lines = []
    for action in actions:
        kind, status, args = action.get("type", ""), action.get("status", ""), action.get("args") or {}
        if status in {"failed", "unknown"}:
            lines.append(f"⚠️ {kind.replace('_', ' ')} did not complete" + (" (it is unknown whether it happened)." if status == "unknown" else "."))
        elif status == "completed" and kind == "remember_rule":
            lines.append(f"✅ Rule saved: {(action.get('result') or {}).get('summary') or args.get('rule', '')}")
        elif status == "completed" and kind == "create_reply_draft":
            lines.append("✅ Saved a reply draft in Gmail Drafts.")
        elif status == "completed":
            lines.append(f"✅ Done: {kind.replace('_', ' ')}.")
    return lines


def owner_reply(text: str, approval_ids: list[int], dry_run: bool = False, actions: list[dict[str, Any]] | None = None) -> str:
    """
    The single agent's answer for the owner. Empty when the approval cards say it all
    (the model answered CARDS_ONLY), so the owner gets the cards and no preamble.
    If the model said CARDS_ONLY but also changed something directly (a Google Task) or
    a tool failed, those are listed instead: nothing done is ever left unreported.
    """

    text = (text or "").strip()
    said_cards_only = bool(_CARDS_ONLY.search(text))
    reply = _CARDS_ONLY.sub(" ", text).strip() if said_cards_only else text
    if not reply and said_cards_only:
        reply = "\n".join(_direct_outcomes(actions or []))
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
    return bool(skills) or bool(MULTI_PART.search(text or "")) or len(request_domains(text)) >= 2


def handle_owner_request(text: str, *, dry_run: bool = False, history: list | None = None, store: AgentStateStore | None = None) -> OwnerCommandResult:
    """Plan, fan out, fan in, report. Falls back to the single agent for one-task requests."""

    store = store or AgentStateStore()
    memories = recall(text, store=store)
    skills = find_skills(text)

    if not should_plan(text, skills):
        return run_single_agent(text, history=history, dry_run=dry_run, memories=memories)

    try:
        plan = plan_request(text, history, memories, skills)
    except (PlanError, SkillError) as exc:
        logger.warning("Plan rejected (%s); running the single agent instead.", exc)
        plan = Plan([])

    if len(plan.tasks) <= 1 and not plan.skill:
        return run_single_agent(text, history=history, dry_run=dry_run, memories=memories)

    store.close_stale_supervisor_runs()
    run_id = store.start_supervisor_run(text)
    store.save_supervisor_plan(run_id, plan.tasks, skill=plan.skill)

    results = run_plan(plan.tasks, run_id=run_id, store=store, dry_run=dry_run)
    notes = resolve_overlaps(results, store, dry_run=dry_run)
    report = build_report(plan.tasks, results, notes, skill=plan.skill, dry_run=dry_run)

    statuses = {payload["status"] for payload in results.values()}
    run_status = "done" if statuses == {"SUCCESS"} else ("failed" if statuses == {"FAILED"} else "partial")

    if run_status == "done" and not plan.skill and not dry_run:
        report += note_workflow_success(plan.tasks, text, run_id, store)

    store.finish_supervisor_run(run_id, "dry_run" if dry_run else run_status, report)

    entities = [entity for payload in results.values() for entity in payload["modified_entities"]]
    return OwnerCommandResult(
        reply=report,
        actions=entities,
        approval_ids=[
            e["approval_id"] for e in entities
            if e.get("kind") == "approval" and e.get("approval_id") and not e.get("duplicate_of")
        ],
        run_id=run_id,
    )
