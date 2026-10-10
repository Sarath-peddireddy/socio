"""
Supervisor-worker harness for owner instructions (fan-out / fan-in).

    owner instruction (Telegram / --ask)
      -> recall long-term memories (top 3) and matching skills
      -> planner: one LLM call emitting spawn_subagent(task_id, goal_contract,
         context_slice, tools_allowed, depends_on) calls that form a DAG,
         or use_skill(name, params) to reuse a saved recipe
      -> workers run concurrently (thread pool of AGENT_MAX_WORKERS, default 3;
         all share one Gemini rate limiter), each in a clean context
      -> fan-in: strict JSON payloads -> aggregator closes duplicate approvals,
         checkpoints the run in SQLite, and writes one report
      -> a successful multi-step plan that is likely to recur is saved as a skill

A one-task instruction (or a follow-up that needs the conversation) runs the
single owner-mode agent as before. Inbox emails are never fanned out
(developer decision): they stay one at a time in app.agent.runner.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.graph import AGENT_TOOLS, create_agent_graph
from app.agent.llm import get_llm, invoke_llm
from app.agent.state import AgentStateStore
from app.agent.taskgraph import MAX_TASKS, PlanError, normalize_plan, parallel_width
from app.agent.workers import run_worker
from app.memory.semantic import Memory, format_memories, memory_enabled, recall
from app.memory.skills import Skill, SkillError, expand_skill, find_skills, save_recipe, similar_skill


logger = logging.getLogger(__name__)

MAX_WORKERS = max(1, int(os.getenv("AGENT_MAX_WORKERS", "3")))
OWNER_RECURSION_LIMIT = 30
# Tools a worker may be given (all owner-mode tools; gates still apply inside them).
WORKER_TOOLS = {agent_tool.name for agent_tool in AGENT_TOOLS}

# Only instructions that look like several jobs are worth a planning call.
MULTI_PART = re.compile(
    r"(\n|;|\b(and then|then|also|after that|as well as|plus)\b|\d[.)]\s|,\s*and\b|\band\s+(send|email|reply|add|append|create|write|"
    r"move|book|schedule|cancel|trash|remind|find|check|tell|share|draft|update|list)\b)",
    re.IGNORECASE,
)


@dataclass
class OwnerCommandResult:
    reply: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    approval_ids: list[int] = field(default_factory=list)
    run_id: int | None = None


# ----- planner tools (schemas only: the planner's tool calls ARE the plan) -------------


class spawn_subagent(BaseModel):
    """Start one worker subagent for one piece of the owner's instruction."""

    task_id: str = Field(description="Short unique id such as t1, t2.")
    goal_contract: str = Field(description="What the worker must achieve and what counts as done. Self-contained.")
    context_slice: str = Field(
        default="",
        description="Every fact the worker needs (names, email addresses, dates, file or document names, which "
        "email or event). The worker sees nothing else.",
    )
    tools_allowed: list[str] = Field(description="Only the tool names this task needs, from the tool list.")
    depends_on: list[str] = Field(default_factory=list, description="task_ids whose results this task needs. Empty if independent.")


class use_skill(BaseModel):
    """Reuse a saved skill's plan instead of planning from scratch."""

    name: str = Field(description="The skill name from the list of saved skills.")
    params_json: str = Field(default="{}", description='The skill parameters as a JSON object, e.g. {"week": "5-9 Oct 2026"}.')


PLANNER_PROMPT = """You are Socio's supervisor. You plan; you never act yourself.
Split the owner's instruction into worker tasks by calling spawn_subagent once per task.

Rules:
- If the instruction is really one task, or a question, call spawn_subagent exactly once with the
  whole instruction as the goal.
- Use several tasks only for separate pieces of work. A task that needs another task's result
  lists it in depends_on; independent tasks run at the same time.
- Each worker sees ONLY its goal_contract and context_slice. Put every needed fact there, and
  resolve words like "him" or "that email" using the conversation so far.
- tools_allowed: the fewest tools from the list below that can do the task.
- At most {max_tasks} tasks.
- Emails, Docs/Sheets writes, and calendar changes become approval requests for the owner inside
  the workers. That is expected; never plan around it.
- If a saved skill below fits the instruction, call use_skill instead, with its parameters.

Tools:
{tools}
{memories}{skills}"""


def _tool_catalog() -> str:
    lines = []
    for agent_tool in AGENT_TOOLS:
        first_line = (agent_tool.description or "").strip().splitlines()[0] if agent_tool.description else ""
        lines.append(f"- {agent_tool.name}: {first_line}")
    return "\n".join(lines)


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
    """One planner call. Raises PlanError (or SkillError) if the result cannot be run."""

    memory_block = format_memories(memories or [])
    system = PLANNER_PROMPT.format(
        max_tasks=MAX_TASKS,
        tools=_tool_catalog(),
        memories=f"\n{memory_block}" if memory_block else "",
        skills=_skill_catalog(skills or []),
    )
    planner = get_llm().bind_tools([spawn_subagent, use_skill])
    response = invoke_llm(
        planner,
        [SystemMessage(content=system), HumanMessage(content=f"{_history_text(history)}Instruction from the owner:\n{text}")],
    )
    calls = getattr(response, "tool_calls", []) or []

    skill_calls = [call for call in calls if call["name"] == "use_skill"]
    if skill_calls:
        by_name = {skill.name: skill for skill, _ in skills or []}
        args = skill_calls[0].get("args", {})
        skill = by_name.get(str(args.get("name", "")))
        if skill is None:
            raise PlanError(f"The planner chose an unknown skill {args.get('name')!r}.")
        try:
            params = json.loads(args.get("params_json") or "{}")
        except json.JSONDecodeError as exc:
            raise PlanError(f"Skill parameters are not valid JSON: {exc}") from exc
        if not isinstance(params, dict):
            raise PlanError("Skill parameters must be a JSON object.")
        return Plan(expand_skill(skill, params, WORKER_TOOLS), skill=skill.name)

    tasks = [call.get("args", {}) for call in calls if call["name"] == "spawn_subagent"]
    if not tasks:
        return Plan([])
    return Plan(normalize_plan(tasks, WORKER_TOOLS))


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
                    payload = {"status": "FAILED", "modified_entities": [], "diff_summary": f"Worker crashed: {type(exc).__name__}: {exc}"}
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
                    notes.append(f"#{entity['approval_id']} ({task_id}) duplicated #{keep} and was closed.")
                else:
                    first_approval.setdefault(key, (task_id, entity["approval_id"]))
            elif entity.get("status") == "completed":
                if key in first_change and first_change[key] != task_id:
                    notes.append(f"{task_id} and {first_change[key]} both changed {key}.")
                first_change.setdefault(key, task_id)
    return notes


def build_report(tasks: list[dict[str, Any]], results: dict[str, dict[str, Any]], notes: list[str], skill: str = "", dry_run: bool = False) -> str:
    width = parallel_width(tasks)
    header = f"{'🧪 Dry run - ' if dry_run else ''}🧭 Split into {len(tasks)} tasks" + (f" (up to {min(width, MAX_WORKERS)} at once)" if width > 1 else "")
    if skill:
        header += f", using skill '{skill}'"
    lines = [header + ":"]

    approvals: list[int] = []
    for task in tasks:
        payload = results.get(task["task_id"], {"status": "FAILED", "diff_summary": "Did not run."})
        mark = "✅" if payload["status"] == "SUCCESS" else "❌"
        goal = " ".join(task["goal_contract"].split())
        goal = goal if len(goal) <= 70 else goal[:69] + "…"
        lines.append(f"{mark} {task['task_id']} · {goal}\n   {payload['diff_summary']}")
        approvals += [
            e["approval_id"] for e in payload.get("modified_entities", [])
            if e.get("kind") == "approval" and e.get("approval_id") and not e.get("duplicate_of")
        ]

    if approvals:
        lines.append("⏳ Waiting for your approval: " + ", ".join(f"#{a}" for a in approvals))
    lines += [f"⚠️ {note}" for note in notes]
    return "\n".join(lines)


# ----- procedural memory: save a successful plan as a skill ----------------------------------


class SkillDraft(BaseModel):
    save: bool = Field(description="True only if this procedure is likely to be repeated.")
    name: str = Field(default="", description="Short kebab-case name, e.g. weekly-applications-update.")
    description: str = Field(default="", description="One sentence: when to use this skill.")
    parameters: list[str] = Field(default_factory=list, description="snake_case names of the parts that change each time.")
    plan_json: str = Field(default="[]", description="The plan as a JSON list of tasks, with {{parameter}} placeholders.")


SKILL_PROMPT = """The owner's instruction below was split into tasks and every task succeeded.
Decide whether this procedure is likely to be repeated (for example a weekly report, a regular
vendor's invoice, a recurring spreadsheet update). If it is a one-off, set save to false.
If it will recur, generalize it: replace the parts that change each time (dates, names,
amounts, addresses) with {{{{parameter}}}} placeholders, list those parameters, and keep each
task's task_id, tools_allowed, and depends_on unchanged.

Instruction: {request}
Plan: {plan}"""


def maybe_save_skill(request: str, tasks: list[dict[str, Any]], run_id: int) -> str | None:
    """Ask the LLM to generalize a successful multi-step plan into a recipe. Returns its name."""

    if len(tasks) < 2 or not memory_enabled():
        return None

    try:
        draft = invoke_llm(
            get_llm().with_structured_output(SkillDraft),
            SKILL_PROMPT.format(request=request, plan=json.dumps(tasks, ensure_ascii=False)),
        )
        if not draft or not draft.save:
            return None
        existing = similar_skill(draft.description)
        if existing is not None and existing.name != draft.name:
            logger.info("Not saving a skill: '%s' already covers it.", existing.name)
            return None
        skill = save_recipe(
            draft.name, draft.description, draft.parameters, json.loads(draft.plan_json),
            WORKER_TOOLS, source_run=run_id, notes=f"Learned from: {request[:300]}",
        )
        return skill.name
    except (SkillError, PlanError, json.JSONDecodeError) as exc:
        logger.info("Not saving a skill: %s", exc)
    except Exception as exc:
        logger.warning("Skill saving failed: %s", exc)
    return None


# ----- entry point -----------------------------------------------------------------------------


def run_single_agent(text: str, *, history: list | None = None, dry_run: bool = False, memories: list[Memory] | None = None) -> OwnerCommandResult:
    """The plain owner-mode agent (one context, full owner toolset)."""

    instruction = f"Instruction from the owner:\n{text}"
    memory_block = format_memories(memories or [])
    if memory_block:
        instruction += f"\n\n{memory_block}"

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
    return OwnerCommandResult(
        reply=(messages[-1].text.strip() if messages else "") or "Done.",
        actions=result.get("actions", []),
        approval_ids=[approval["id"] for approval in result.get("pending_approvals", []) if "id" in approval],
    )


def should_plan(text: str, skills: list[tuple[Skill, float]]) -> bool:
    return bool(skills) or bool(MULTI_PART.search(text or ""))


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
    store.save_supervisor_plan(run_id, plan.tasks)

    results = run_plan(plan.tasks, run_id=run_id, store=store, dry_run=dry_run)
    notes = resolve_overlaps(results, store, dry_run=dry_run)
    report = build_report(plan.tasks, results, notes, skill=plan.skill, dry_run=dry_run)

    statuses = {payload["status"] for payload in results.values()}
    run_status = "done" if statuses == {"SUCCESS"} else ("failed" if statuses == {"FAILED"} else "partial")

    if run_status == "done" and not plan.skill and not dry_run:
        saved = maybe_save_skill(text, plan.tasks, run_id)
        if saved:
            report += f"\n🧩 Saved as skill '{saved}': next time I'll reuse this plan (/skills to see it)."

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
