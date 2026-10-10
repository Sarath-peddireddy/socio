"""
Socio's long-lived memory.

Three tiers:

1. Short-term / episodic state: SQLite (`app.agent.state`): emails in progress,
   approvals, slot holds, run locks, supervisor checkpoints.
2. Long-term semantic memory (`semantic.py`): (entity, preference, context)
   tuples learned from the owner's approvals, rejections, and corrections,
   embedded locally (`embeddings.py`) and recalled by similarity (top 3-5).
3. Procedural memory (`skills.py`): reusable recipes in ./skills/ saved after a
   multi-step instruction succeeded, so the next one reuses the plan.

Memory only informs the agent. It never grants a permission: every approval
gate stays in force whatever a memory or skill says.
"""
