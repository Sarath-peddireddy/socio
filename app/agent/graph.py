from typing import Annotated, Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agent.llm import get_llm, invoke_llm
from app.agent.prompts import build_system_prompt
from app.agent.state import AgentStateStore
from app.agent.approvals import describe
from app.agent.tools import (
    append_sheet_row,
    append_to_document,
    check_reply_needed,
    complete_task,
    create_document,
    create_spreadsheet,
    create_task,
    list_tasks,
    read_document,
    read_sheet,
    create_reply_draft,
    find_calendar_events,
    find_documents,
    get_email,
    get_recent_emails,
    get_thread,
    propose_cancel_calendar_event,
    propose_move_calendar_event,
    propose_trash_email,
    search_gmail,
    send_gmail_reply,
    send_new_email,
)
from app.calendar.tools import create_calendar_event, find_calendar_availability
from app.google_api import is_retryable_error


# Single source of truth: these tools are both bound to the model and dispatched below.
AGENT_TOOLS = [
    get_recent_emails,
    get_email,
    search_gmail,
    get_thread,
    check_reply_needed,
    find_calendar_availability,
    find_calendar_events,
    create_calendar_event,
    find_documents,
    send_gmail_reply,
    send_new_email,
    create_reply_draft,
    propose_move_calendar_event,
    propose_cancel_calendar_event,
    propose_trash_email,
    read_document,
    create_document,
    append_to_document,
    read_sheet,
    append_sheet_row,
    create_spreadsheet,
    list_tasks,
    create_task,
    complete_task,
]
TOOLS_BY_NAME = {agent_tool.name: agent_tool for agent_tool in AGENT_TOOLS}

# Tools with external side effects. Only these are recorded as actions.
SIDE_EFFECT_TOOLS = {
    "send_gmail_reply",
    "send_new_email",
    "create_calendar_event",
    "create_reply_draft",
    "create_document",
    "append_to_document",
    "create_spreadsheet",
    "append_sheet_row",
    "create_task",
    "complete_task",
}

# Tools that only create an approval request for the owner (see app.agent.approvals).
PROPOSAL_TOOLS = {"propose_move_calendar_event", "propose_cancel_calendar_event", "propose_trash_email"}

# Tools that must stay inside the thread being handled.
THREAD_SCOPED_TOOLS = {"send_gmail_reply", "create_reply_draft"}

# When handling an incoming email, the agent acts on behalf of an untrusted
# sender, so it gets a narrower toolset: no reading other mail, no new emails
# to arbitrary people, and no trashing.
EMAIL_MODE_TOOLS = {
    "get_thread",
    "find_calendar_availability",
    "find_calendar_events",
    "create_calendar_event",
    "find_documents",
    "send_gmail_reply",
    "create_reply_draft",
    "propose_move_calendar_event",
    "propose_cancel_calendar_event",
}


def _is_email_mode(state: "AgentState") -> bool:
    return bool(state.get("target_thread_id"))


def _allowed_tools(state: "AgentState") -> list:
    if _is_email_mode(state):
        return [agent_tool for agent_tool in AGENT_TOOLS if agent_tool.name in EMAIL_MODE_TOOLS]
    return AGENT_TOOLS


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    processed_message_ids: list[str]
    actions: list[dict[str, Any]]
    pending_approvals: list[dict[str, Any]]
    failures: list[str]

    # Set by the runner when handling one specific email (optional otherwise).
    target_message_id: str
    target_thread_id: str
    reply_recipient: str
    participants: list[str]
    dry_run: bool


def call_model(state: AgentState):
    """Ask the LLM what to do next."""

    llm = get_llm()
    llm_with_tools = llm.bind_tools(_allowed_tools(state))

    # The system prompt is not stored in state so it is always current.
    mode = "email" if _is_email_mode(state) else "owner"
    messages = [SystemMessage(content=build_system_prompt(mode))] + list(state["messages"])
    response = invoke_llm(llm_with_tools, messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Continue only when the model requested a tool call."""

    if not state.get("messages"):
        return END

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    return "tools" if tool_calls else END


def _execute_tool(tool_call: dict[str, Any]):
    tool_name = tool_call["name"]

    if tool_name not in TOOLS_BY_NAME:
        raise ValueError(f"Unknown tool requested: {tool_name}")

    return TOOLS_BY_NAME[tool_name].invoke(tool_call.get("args", {}))


def _action_target(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name in {"send_gmail_reply", "create_reply_draft"}:
        return args.get("message_id", "")
    if tool_name == "send_new_email":
        return args.get("to", "")
    if tool_name in {"create_document", "create_spreadsheet", "create_task"}:
        return args.get("title", "")
    if tool_name == "append_to_document":
        return args.get("document", "")
    if tool_name == "append_sheet_row":
        return args.get("spreadsheet", "")
    if tool_name == "complete_task":
        return args.get("task_id", "")
    if tool_name == "create_calendar_event":
        return f"{args.get('title', '')} @ {args.get('start_time', '')}"
    return ""


def _scope_violation(tool_name: str, args: dict[str, Any], state: AgentState) -> str | None:
    """Return a refusal message if a tool call reaches outside the email being handled."""

    target_thread_id = state.get("target_thread_id")

    if target_thread_id and tool_name not in EMAIL_MODE_TOOLS:
        return f"Refused: {tool_name} is not available while handling an incoming email."

    if target_thread_id and tool_name in THREAD_SCOPED_TOOLS and args.get("thread_id") != target_thread_id:
        return (
            f"Refused: you may only act on thread {target_thread_id}, "
            f"not {args.get('thread_id')}."
        )

    participants = {address.lower() for address in state.get("participants", [])}
    attendee = (args.get("attendee_email") or "").strip().lower()

    if tool_name == "create_calendar_event" and participants and attendee and attendee not in participants:
        return (
            f"Refused: {attendee} is not part of this conversation, "
            "so they cannot be invited."
        )

    return None


def _dry_run_result(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dry_run": True,
        "executed": False,
        "note": (
            f"DRY RUN: {tool_name} was NOT executed. Continue as if it succeeded, "
            "but say in your summary that this was a dry run."
        ),
    }

    if tool_name == "create_calendar_event":
        result.update(start=args.get("start_time"), end=args.get("end_time"), title=args.get("title"))

    return result


def _tool_message(content: Any, tool_call: dict[str, Any]) -> ToolMessage:
    return ToolMessage(content=str(content), tool_call_id=tool_call.get("id"))


def _record_proposal(
    proposal: dict[str, Any],
    state: AgentState,
    store: AgentStateStore,
    dry_run: bool,
    actions: list[dict[str, Any]],
    pending_approvals: list[dict[str, Any]],
    failures: list[str],
) -> str:
    """Turn a validated proposal into an approval request for the owner. Returns the tool message."""

    proposal = dict(proposal)
    kind = proposal.pop("type")
    reply_body = (proposal.pop("reply_body", "") or "").strip()

    if _is_email_mode(state):
        # A sender may only ask to change an event they are part of.
        participants = {address.lower() for address in state.get("participants", [])}
        if not set(proposal.get("other_attendees", [])) & participants:
            refusal = (
                "Refused: nobody from this email conversation is an attendee of that event, "
                "so it cannot be changed on their request. Use create_reply_draft instead."
            )
            failures.append(f"{kind}: {refusal}")
            return refusal

        if reply_body:
            proposal["reply"] = {
                "thread_id": state.get("target_thread_id"),
                "message_id": state.get("target_message_id"),
                "to": state.get("reply_recipient", ""),
                "body": reply_body,
            }

    summary = describe({"type": kind, "payload": proposal})

    if dry_run:
        actions.append({"type": kind, "status": "dry_run", "args": proposal})
        return f"DRY RUN: would ask the owner to approve:\n{summary}"

    approval_id = store.add_pending_approval(
        kind,
        summary,
        proposal,
        message_id=state.get("target_message_id"),
        thread_id=state.get("target_thread_id"),
    )
    actions.append({"type": kind, "status": "proposed", "approval_id": approval_id, "args": proposal})
    pending_approvals.append({"id": approval_id, "type": kind})

    return (
        f"Approval request #{approval_id} sent to the owner. NOTHING has changed yet. "
        + ("Do not send a reply now: your reply_body is sent automatically after approval. " if proposal.get("reply") else "")
        + "Say in your summary that this is waiting for the owner's approval."
    )


def execute_tools(state: AgentState):
    """Execute tools requested by the LLM and preserve state for restart recovery."""

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    tool_results = []
    store = AgentStateStore()
    dry_run = bool(state.get("dry_run"))
    source_message_id = state.get("target_message_id")

    processed_message_ids = list(state.get("processed_message_ids", []))
    actions = list(state.get("actions", []))
    pending_approvals = list(state.get("pending_approvals", []))
    failures = list(state.get("failures", []))

    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "unknown_tool")
        args = tool_call.get("args", {})
        target = _action_target(tool_name, args)

        refusal = _scope_violation(tool_name, args, state)

        # Never reply twice to the same message, including across restarts.
        if refusal is None and tool_name == "send_gmail_reply" and (
            store.is_processed(args.get("message_id", ""))
            or args.get("message_id") in processed_message_ids
        ):
            refusal = (
                f"Not sent: a reply to message {args['message_id']} was already "
                "sent. Do not send it again."
            )

        # One event / reply / draft per email, even if an earlier attempt was interrupted.
        if refusal is None and source_message_id and tool_name in SIDE_EFFECT_TOOLS:
            earlier = store.actions_for_message(source_message_id, status="completed")
            earlier = [action for action in earlier if action["type"] == tool_name]
            if earlier:
                refusal = (
                    f"Refused: {tool_name} was already completed for this email "
                    f"(result: {earlier[-1]['result']}). Do not repeat it; use that result."
                )

        if refusal:
            failures.append(f"{tool_name}: {refusal}")
            tool_results.append(_tool_message(refusal, tool_call))
            continue

        if dry_run and tool_name in SIDE_EFFECT_TOOLS:
            result = _dry_run_result(tool_name, args)
            actions.append({"type": tool_name, "target": target, "args": args, "status": "dry_run", "result": result})
            if tool_name == "send_gmail_reply":
                processed_message_ids.append(args.get("message_id"))
            tool_results.append(_tool_message(result, tool_call))
            continue

        try:
            result = _execute_tool(tool_call)
        except Exception as exc:  # pragma: no cover - exercised in live runs and recorded for follow-up
            failures.append(f"{tool_name}: {exc}")

            if not dry_run:
                store.add_failure(tool_name, str(exc))

            message = f"Tool failed: {exc}"

            if tool_name in SIDE_EFFECT_TOOLS:
                status = "unknown" if is_retryable_error(exc) else "failed"
                actions.append({"type": tool_name, "target": target, "args": args, "status": status, "error": str(exc)})

                if not dry_run:
                    store.add_action(tool_name, target, args, status=status, message_id=source_message_id)

                if status == "unknown":
                    message += (
                        " This was a network error, so it is unknown whether the action "
                        "happened. Do not retry it; report it in your summary."
                    )

            tool_results.append(_tool_message(message, tool_call))
            continue

        if tool_name in PROPOSAL_TOOLS:
            message = _record_proposal(result["proposal"], state, store, dry_run, actions, pending_approvals, failures)
            tool_results.append(_tool_message(message, tool_call))
            continue

        tool_results.append(_tool_message(result, tool_call))

        if tool_name not in SIDE_EFFECT_TOOLS:
            continue

        actions.append({"type": tool_name, "target": target, "args": args, "status": "completed", "result": result})

        if dry_run:
            continue

        store.add_action(tool_name, target, args, status="completed", message_id=source_message_id, result=result)

        if tool_name == "send_gmail_reply":
            store.mark_processed(
                args["message_id"],
                thread_id=result.get("thread_id"),
                status="replied",
                summary="Reply sent",
                metadata={"sent_message_id": result.get("message_id"), "verified": result.get("verified")},
            )
            processed_message_ids.append(args["message_id"])

        if tool_name == "create_reply_draft":
            approval = {
                "type": "review_draft",
                "target": args.get("reason", ""),
                "draft_id": result.get("draft_id"),
                "message_id": args.get("message_id"),
                "thread_id": args.get("thread_id"),
            }
            store.add_pending_approval(
                "review_draft",
                approval["target"],
                approval,
                message_id=source_message_id or args.get("message_id"),
                thread_id=args.get("thread_id"),
            )
            pending_approvals.append(approval)

    return {
        "messages": tool_results,
        "processed_message_ids": processed_message_ids,
        "actions": actions,
        "pending_approvals": pending_approvals,
        "failures": failures,
    }


def create_agent_graph():
    """Build and compile the Socio agent graph."""

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_email_workflow(user_prompt: str):
    """Convenience entry point for a free-form autonomous run (see app.agent.runner for per-email processing)."""

    agent = create_agent_graph()
    initial_state = {
        "messages": [HumanMessage(content=user_prompt)],
        "processed_message_ids": [],
        "actions": [],
        "pending_approvals": [],
        "failures": [],
    }
    return agent.invoke(initial_state)
