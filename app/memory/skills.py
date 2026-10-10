"""
Procedural memory: reusable skills in ./skills/ (or AGENT_SKILLS_DIR).

A skill is a Markdown recipe with YAML front matter and a JSON plan:

    ---
    name: weekly-applications-update
    description: Add this week's job applications to the applications sheet
    parameters: [week]
    author: socio
    created: 2026-10-10
    ---
    # Weekly applications update
    ## Plan
    ```json
    [{"task_id": "t1", "goal_contract": "Find applications sent in {{week}} ...",
      "context_slice": "", "tools_allowed": ["search_gmail"], "depends_on": []}]
    ```

- Socio saves a recipe after a multi-step owner instruction succeeded, and the
  supervisor reuses its plan next time instead of planning from scratch.
  Recipes contain no code: replaying one runs the same guarded tools and approvals.
- Socio never writes Python. A script (skills/<name>.py) is written by the owner
  and runs only after the owner approved its exact SHA-256 (enable_skill_script),
  in a subprocess with no secrets in its environment, on files from the
  whitelisted folders only. It returns JSON; it cannot send or change anything
  itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.agent.taskgraph import PlanError, normalize_plan


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")
PLACEHOLDER = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")
PLAN_BLOCK = re.compile(r"## Plan\s*```json\s*(.*?)```", re.DOTALL)
FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)

SKILL_MATCH_SCORE = 0.6       # request -> skill description similarity needed to offer a skill to the planner
SKILL_DUPLICATE_SCORE = 0.95  # description -> description similarity at which a new recipe repeats an old one
SCRIPT_TIMEOUT_SECONDS = 60
MAX_SCRIPT_OUTPUT_CHARS = 20_000


class SkillError(ValueError):
    pass


def skills_dir() -> Path:
    return Path(os.getenv("AGENT_SKILLS_DIR") or PROJECT_ROOT / "skills")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:48].strip("-")
    if not SKILL_NAME.match(slug):
        raise SkillError(f"Invalid skill name {name!r}.")
    return slug


@dataclass
class Skill:
    name: str
    description: str
    parameters: list[str]
    plan: list[dict[str, Any]]
    path: Path
    author: str = "socio"
    script: str = ""
    created: str = ""
    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# ----- reading ------------------------------------------------------------------


def parse_skill(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if not match:
        raise SkillError(f"{path.name}: missing front matter.")

    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise SkillError(f"{path.name}: front matter is not a mapping.")
    body = match.group(2)

    name = slugify(str(meta.get("name") or path.stem))
    plan_match = PLAN_BLOCK.search(body)
    plan = json.loads(plan_match.group(1)) if plan_match else []
    if not isinstance(plan, list):
        raise SkillError(f"{path.name}: the plan must be a JSON list.")

    script = str(meta.get("script") or "")
    if script and Path(script).name != script:
        raise SkillError(f"{path.name}: script must be a file name in the skills folder.")

    return Skill(
        name=name,
        description=str(meta.get("description") or "").strip(),
        parameters=[str(p) for p in meta.get("parameters") or []],
        plan=plan,
        path=path,
        author=str(meta.get("author") or "owner"),
        script=script,
        created=str(meta.get("created") or ""),
        notes=PLAN_BLOCK.sub("", body).strip(),
        meta=meta,
    )


def list_skills() -> list[Skill]:
    folder = skills_dir()
    if not folder.is_dir():
        return []

    skills = []
    for path in sorted(folder.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            skills.append(parse_skill(path))
        except Exception as exc:
            logger.warning("Skipping skill %s: %s", path.name, exc)
    return skills


def get_skill(name: str) -> Skill | None:
    slug = slugify(name)
    return next((skill for skill in list_skills() if skill.name == slug), None)


_vector_cache: dict[tuple[str, float], np.ndarray] = {}
_vector_lock = threading.Lock()


def _skill_vector(skill: Skill, embedder) -> np.ndarray:
    key = (str(skill.path), skill.path.stat().st_mtime)
    with _vector_lock:
        if key not in _vector_cache:
            _vector_cache[key] = embedder.embed_documents([skill.description])[0]
        return _vector_cache[key]


def find_skills(request: str, k: int = 3, min_score: float = SKILL_MATCH_SCORE) -> list[tuple[Skill, float]]:
    """Skills whose description matches the request, best first. [] on any problem."""

    from app.memory.embeddings import get_embedder
    from app.memory.semantic import memory_enabled

    skills = [skill for skill in list_skills() if skill.plan]
    if not skills or not memory_enabled():
        return []
    try:
        embedder = get_embedder()
        query = np.asarray(embedder.embed_query(request), dtype=np.float32)
        scored = [(skill, float(_skill_vector(skill, embedder) @ query)) for skill in skills]
    except Exception as exc:
        logger.warning("Skill lookup failed: %s", exc)
        return []
    scored = [(skill, round(score, 3)) for skill, score in scored if score >= min_score]
    return sorted(scored, key=lambda item: -item[1])[:k]


def similar_skill(description: str, min_score: float = SKILL_DUPLICATE_SCORE) -> Skill | None:
    """An existing recipe that describes the same procedure, if any."""

    from app.memory.embeddings import get_embedder

    skills = [skill for skill in list_skills() if skill.plan]
    if not skills:
        return None
    embedder = get_embedder()
    vector = embedder.embed_documents([description])[0]
    scored = [(skill, float(_skill_vector(skill, embedder) @ vector)) for skill in skills]
    best = max(scored, key=lambda item: item[1])
    return best[0] if best[1] >= min_score else None


def expand_skill(skill: Skill, params: dict[str, Any], allowed_tools: set[str]) -> list[dict[str, Any]]:
    """The skill's plan with {{parameters}} filled in, validated like any other plan."""

    params = {str(key): " ".join(str(value).split())[:300] for key, value in (params or {}).items()}
    missing = [name for name in skill.parameters if not params.get(name)]
    if missing:
        raise SkillError(f"Skill {skill.name} needs: {', '.join(missing)}.")

    def fill(text: str) -> str:
        return PLACEHOLDER.sub(lambda m: params.get(m.group(1), m.group(0)), text or "")

    tasks = [
        {**task, "goal_contract": fill(task.get("goal_contract", "")), "context_slice": fill(task.get("context_slice", ""))}
        for task in skill.plan
    ]
    leftover = {m.group(1) for task in tasks for m in PLACEHOLDER.finditer(task["goal_contract"] + task["context_slice"])}
    if leftover:
        raise SkillError(f"Skill {skill.name} uses undeclared parameters: {', '.join(sorted(leftover))}.")

    return normalize_plan(tasks, allowed_tools)


# ----- writing (recipes only) -------------------------------------------------------


def save_recipe(
    name: str,
    description: str,
    parameters: list[str],
    plan: list[dict[str, Any]],
    allowed_tools: set[str],
    notes: str = "",
    source_run: int | None = None,
) -> Skill:
    """
    Write a recipe to skills/<name>.md. Never writes code.

    The plan must be valid for allowed_tools, every {{placeholder}} must be a
    declared parameter, and an owner-written skill is never overwritten.
    """

    slug = slugify(name)
    description = " ".join((description or "").split())[:300]
    if not description:
        raise SkillError("A skill needs a description.")
    parameters = [p for p in dict.fromkeys(str(p).strip() for p in parameters or []) if p]
    bad = [p for p in parameters if not re.fullmatch(r"[a-z_][a-z0-9_]*", p)]
    if bad:
        raise SkillError(f"Invalid parameter names: {', '.join(bad)}.")

    try:
        plan = normalize_plan(plan, allowed_tools)
    except PlanError as exc:
        raise SkillError(f"Invalid plan: {exc}") from exc

    used = {m.group(1) for task in plan for m in PLACEHOLDER.finditer(task["goal_contract"] + task["context_slice"])}
    undeclared = used - set(parameters)
    if undeclared:
        raise SkillError(f"Plan uses undeclared parameters: {', '.join(sorted(undeclared))}.")

    folder = skills_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug}.md"
    if path.exists():
        existing = parse_skill(path)
        if existing.author != "socio":
            raise SkillError(f"A skill named {slug} was written by the owner; not overwriting it.")

    meta = {
        "name": slug,
        "description": description,
        "parameters": parameters,
        "author": "socio",
        "created": date.today().isoformat(),
    }
    if source_run is not None:
        meta["source_run"] = int(source_run)

    title = slug.replace("-", " ").capitalize()
    notes = (notes or "").strip()
    text = (
        f"---\n{yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()}\n---\n"
        f"# {title}\n\n"
        + (f"{notes}\n\n" if notes else "")
        + "Saved by Socio after this procedure succeeded. Replaying it uses the same tools and\n"
        "approval gates as any instruction. Edit or delete this file freely.\n\n"
        f"## Plan\n```json\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n```\n"
    )

    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return parse_skill(path)


def archive_skill(name: str) -> bool:
    """Move a skill out of use (into skills/.archive/); nothing is deleted."""

    slug = slugify(name)
    path = skills_dir() / f"{slug}.md"
    if not path.exists():
        return False
    archive = skills_dir() / ".archive"
    archive.mkdir(exist_ok=True)
    shutil.move(str(path), str(archive / f"{slug}-{date.today().isoformat()}.md"))
    return True


# ----- owner-written scripts (hash-pinned) --------------------------------------------


def script_path(name: str) -> Path:
    path = (skills_dir() / f"{slugify(name)}.py").resolve()
    if path.parent != skills_dir().resolve() or not path.is_file():
        raise SkillError(f"No skill script named {name!r} in {skills_dir()}.")
    return path


def script_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def script_preview(content: bytes, lines: int = 25) -> str:
    text = content.decode("utf-8", errors="replace").splitlines()
    return "\n".join(text[:lines]) + ("\n…" if len(text) > lines else "")


def run_script(name: str, file_paths: list[str] | None = None, params: dict[str, Any] | None = None, store=None) -> dict[str, Any]:
    """
    Run an owner-approved skill script and return its JSON output.

    If this exact file has not been approved, nothing runs: the result carries a
    proposal for an enable_skill_script approval instead.
    """

    from app.agent.state import AgentStateStore
    from app.files.local import resolve_shareable_file

    store = store or AgentStateStore()
    slug = slugify(name)
    content = script_path(slug).read_bytes()
    digest = script_sha256(content)

    if not store.is_skill_script_trusted(slug, digest):
        return {
            "executed": False,
            "proposal": {
                "type": "enable_skill_script",
                "name": slug,
                "sha256": digest,
                "preview": script_preview(content),
                "reason": "This script has not been approved in its current form.",
            },
        }

    # Only files from the whitelisted folders (documents/, AGENT_FILES_DIR, Drive downloads).
    files = [str(resolve_shareable_file(path)) for path in file_paths or []]

    with tempfile.TemporaryDirectory(prefix="socio-skill-") as work:
        # Run the bytes that were hashed, not whatever the file holds a moment later.
        runnable = Path(work) / f"{slug}.py"
        runnable.write_bytes(content)
        env = {key: os.environ[key] for key in ("SYSTEMROOT", "PATH", "PATHEXT", "COMSPEC") if key in os.environ}
        env.update({"TEMP": work, "TMP": work, "PYTHONIOENCODING": "utf-8"})

        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(runnable)],
                input=json.dumps({"files": files, "params": params or {}, "output_dir": work}),
                capture_output=True, text=True, encoding="utf-8", timeout=SCRIPT_TIMEOUT_SECONDS, cwd=work, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"executed": True, "ok": False, "error": f"Timed out after {SCRIPT_TIMEOUT_SECONDS}s."}

    stdout = completed.stdout[:MAX_SCRIPT_OUTPUT_CHARS]
    try:
        output: Any = json.loads(stdout) if stdout.strip() else None
    except json.JSONDecodeError:
        output = stdout

    return {
        "executed": True,
        "ok": completed.returncode == 0,
        "output": output,
        "error": completed.stderr[-1500:] if completed.returncode else "",
    }


def trust_script(name: str, sha256: str, approval_id: int | None = None, store=None) -> dict[str, Any]:
    """Record the owner's approval for this exact script content (re-checked now)."""

    from app.agent.state import AgentStateStore

    store = store or AgentStateStore()
    slug = slugify(name)
    current = script_sha256(script_path(slug).read_bytes())
    if current != sha256:
        raise SkillError(f"skills/{slug}.py changed after the request was made; nothing was approved.")
    store.trust_skill_script(slug, sha256, approval_id)
    return {"trusted": slug, "sha256": sha256}
