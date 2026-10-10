"""
Task DAGs for the supervisor: validation and dependency order.

A plan is a list of tasks:
    {"task_id", "goal_contract", "context_slice", "tools_allowed", "depends_on"}
Plans come from the LLM planner or a saved skill, so they are checked before
anything runs.
"""

from __future__ import annotations

import re
from typing import Any


MAX_TASKS = 6
TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_GOAL_CHARS = 1500
MAX_CONTEXT_CHARS = 4000


class PlanError(ValueError):
    """The plan cannot be run as given."""


def normalize_plan(tasks: list[dict[str, Any]], allowed_tools: set[str]) -> list[dict[str, Any]]:
    """
    Check a plan and return a cleaned copy.

    Rules: 1 to MAX_TASKS tasks; unique simple ids; a non-empty goal; tools from
    allowed_tools only (at least one); dependencies only on other tasks of the
    same plan, without cycles.
    """

    if not tasks:
        raise PlanError("The plan has no tasks.")
    if len(tasks) > MAX_TASKS:
        raise PlanError(f"The plan has {len(tasks)} tasks; the limit is {MAX_TASKS}.")

    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()

    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        if not TASK_ID.match(task_id):
            raise PlanError(f"Invalid task id {task_id!r}.")
        if task_id in seen:
            raise PlanError(f"Duplicate task id {task_id!r}.")
        seen.add(task_id)

        goal = str(task.get("goal_contract") or "").strip()
        if not goal:
            raise PlanError(f"Task {task_id} has no goal.")

        tools = list(dict.fromkeys(str(name) for name in task.get("tools_allowed") or []))
        unknown = [name for name in tools if name not in allowed_tools]
        if unknown:
            raise PlanError(f"Task {task_id} asks for tools that are not available: {', '.join(unknown)}.")
        if not tools:
            raise PlanError(f"Task {task_id} has no tools.")

        cleaned.append({
            "task_id": task_id,
            "goal_contract": goal[:MAX_GOAL_CHARS],
            "context_slice": str(task.get("context_slice") or "").strip()[:MAX_CONTEXT_CHARS],
            "tools_allowed": tools,
            "depends_on": list(dict.fromkeys(str(d) for d in task.get("depends_on") or [])),
        })

    for task in cleaned:
        missing = [d for d in task["depends_on"] if d not in seen]
        if missing:
            raise PlanError(f"Task {task['task_id']} depends on unknown task(s): {', '.join(missing)}.")
        if task["task_id"] in task["depends_on"]:
            raise PlanError(f"Task {task['task_id']} depends on itself.")

    topological_order(cleaned)  # raises on a cycle
    return cleaned


def topological_order(tasks: list[dict[str, Any]]) -> list[str]:
    """Task ids in an order where every task comes after its dependencies (Kahn's algorithm)."""

    remaining = {task["task_id"]: set(task["depends_on"]) for task in tasks}
    order: list[str] = []

    while remaining:
        ready = [task_id for task_id, deps in remaining.items() if not deps]
        if not ready:
            raise PlanError(f"The plan has a dependency cycle among: {', '.join(sorted(remaining))}.")
        for task_id in ready:
            order.append(task_id)
            del remaining[task_id]
        for deps in remaining.values():
            deps.difference_update(ready)

    return order


def parallel_width(tasks: list[dict[str, Any]]) -> int:
    """How many tasks can run at the same time at most (the widest level of the DAG)."""

    level: dict[str, int] = {}
    by_id = {task["task_id"]: task for task in tasks}
    for task_id in topological_order(tasks):
        level[task_id] = 1 + max((level[d] for d in by_id[task_id]["depends_on"]), default=0)
    counts: dict[int, int] = {}
    for value in level.values():
        counts[value] = counts.get(value, 0) + 1
    return max(counts.values(), default=0)
