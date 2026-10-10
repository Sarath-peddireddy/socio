# Skills (procedural memory)

This folder holds Socio's reusable procedures. Everything here except this
README is gitignored: skills can contain names, addresses, and file names.

## Recipes (`*.md`)

Socio never saves a recipe on its own. After a multi-step instruction
succeeds, send `/save_skill NAME` in Telegram (or run
`python -m app.main --save-skill NAME`): the last successful plan is saved
here, with names, dates, and similar details turned into `{{parameters}}` by
Gemini (or exactly as it ran if Gemini is unavailable). After you've run the
same kind of workflow 3 times, the report suggests it once. The next time a
matching instruction arrives (a weekly report, a regular vendor's invoice),
the supervisor reuses the plan instead of planning from scratch.

```markdown
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
[
  {"task_id": "t1", "goal_contract": "Find job applications I sent during {{week}}",
   "context_slice": "", "tools_allowed": ["search_gmail", "get_thread"], "depends_on": []},
  {"task_id": "t2", "goal_contract": "Add one row per application to the Applications sheet",
   "context_slice": "", "tools_allowed": ["read_sheet", "append_sheet_row"], "depends_on": ["t1"]}
]
```
```

- `{{parameter}}` placeholders are filled from the new instruction.
- Recipes contain no code. Replaying one goes through the same tools and
  approval gates as any instruction (every email and Docs/Sheets write still
  asks you first).
- Edit or delete recipes freely. Socio never overwrites one you wrote
  (`author` other than `socio`). `/forget_skill NAME` moves one to `.archive/`.

## Scripts (`*.py`): you write these, Socio never does

A script handles a job that's better done in code, such as parsing a vendor's
invoice PDF. Socio runs it with `run_skill_script` only after you approved
**that exact file content** (a Telegram card shows its first lines and its
SHA-256). If you change the file, it needs approval again.

The script reads JSON on stdin and writes JSON to stdout:

```python
import json, sys

request = json.load(sys.stdin)   # {"files": [...], "params": {...}, "output_dir": "..."}
# files: absolute paths, only from your documents folder (or downloaded Drive files)
print(json.dumps({"total": 1234.5, "vendor": "Acme"}))
```

It runs in a separate Python process with a temporary working folder, no
secrets in its environment, and a 60-second limit. Its output is treated as
data, never as instructions. It cannot send or change anything through Socio.
It is not a security sandbox, though: it runs with your user's permissions, so
only approve code you wrote or reviewed.

A recipe can point at a script with `script: name.py` in its front matter.
