"""
Check the Docker setup without starting the bot: no Telegram, Google, or model calls.

    python scripts/verify_docker.py           compose file, host files, then a smoke test in the image
    python scripts/verify_docker.py --build   rebuild the image first
    python scripts/verify_docker.py --config  only the compose file and host files

The smoke test runs one throwaway container (`docker compose run --rm`) with its database in /tmp, so the
real data volume is never written. It checks: non-root user, read-only code, the database is created in
SOCIO_DATA_DIR with WAL and the current schema, the app imports, the time zone, and the mounted sign-in files.
Exit code 1 if anything fails. The resolved compose file is never printed (it contains .env values).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HOST_FILES = ["credentials.json", "token.json", "calendar_token.json", "workspace_token.json"]
HOST_DIRS = ["documents", "skills"]

SMOKE = r"""
import os, sqlite3, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL") + "  " + name + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)

results = []
results.append(check("runs as a non-root user", os.getuid() != 0, f"uid {os.getuid()}"))
results.append(check("Python version", sys.version_info >= (3, 12), sys.version.split()[0]))
results.append(check("code is read-only", not os.access("/app/app", os.W_OK)))
results.append(check("data volume is writable", os.access("/app/data", os.W_OK)))

from app.agent.state import AgentStateStore, BUSY_TIMEOUT_SECONDS, SCHEMA_VERSION
store = AgentStateStore()
expected = os.path.join(os.environ["SOCIO_DATA_DIR"], "agent_state.sqlite3")
results.append(check("database created in SOCIO_DATA_DIR", str(store.db_path) == expected and store.db_path.exists(), str(store.db_path)))
with store._connect() as conn:
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    sync = conn.execute("PRAGMA synchronous").fetchone()[0]
results.append(check("WAL journal mode", mode == "wal", mode))
results.append(check("schema migrated", version == SCHEMA_VERSION, f"v{version}"))
results.append(check("synchronous=NORMAL", sync == 1, str(sync)))
results.append(check("busy timeout", BUSY_TIMEOUT_SECONDS >= 5, f"{BUSY_TIMEOUT_SECONDS} s"))

import app.telegram_bot, app.main, fastembed, onnxruntime  # noqa: F401
results.append(check("app, fastembed and onnxruntime import", True))
offset = datetime(2026, 1, 1, tzinfo=ZoneInfo("Asia/Kolkata")).utcoffset()
results.append(check("Asia/Kolkata time zone", offset == timedelta(hours=5, minutes=30), str(offset)))

for name in ("credentials.json", "token.json", "calendar_token.json", "workspace_token.json"):
    path = "/app/" + name
    readable = os.path.isfile(path) and os.access(path, os.R_OK)
    if name == "credentials.json":
        results.append(check(f"{name} mounted read-only", readable and not os.access(path, os.W_OK)))
    else:
        results.append(check(f"{name} mounted and writable (token refresh)", readable and os.access(path, os.W_OK)))
results.append(check("downloads folder is writable", os.access("/app/downloads", os.W_OK)))
sys.exit(0 if all(results) else 1)
"""


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, text=True, encoding="utf-8", errors="replace", **kwargs)


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(("PASS" if ok else "FAIL") + "  " + name + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="rebuild the image before the smoke test")
    parser.add_argument("--config", action="store_true", help="only check the compose file and the host files")
    options = parser.parse_args()

    results: list[bool] = []
    print("== Docker")
    if not shutil.which("docker"):
        report("docker is installed", False, "install Docker Desktop or Docker Engine")
        return 1
    engine = run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True)
    if not report("Docker engine is running", engine.returncode == 0, engine.stdout.strip() or "start Docker Desktop"):
        return 1
    compose = run(["docker", "compose", "version", "--short"], capture_output=True)
    results.append(report("docker compose is available", compose.returncode == 0, compose.stdout.strip()))

    print("== Host files")
    results.append(report(".env exists", (ROOT / ".env").is_file(), "copy .env.example to .env"))
    for name in HOST_FILES:
        path = ROOT / name
        hint = "run: python -m app.main --google-login (on this machine, not in Docker)" if name != "credentials.json" else "OAuth client file from Google Cloud"
        results.append(report(f"{name} is a file", path.is_file(), "" if path.is_file() else hint))
    for name in HOST_DIRS:
        results.append(report(f"{name}/ folder exists", (ROOT / name).is_dir(), "" if (ROOT / name).is_dir() else f"create it: mkdir {name}"))

    print("== Compose file")
    config = run(["docker", "compose", "config", "--quiet"], capture_output=True)
    results.append(report("docker-compose.yml is valid", config.returncode == 0, config.stderr.strip()[:300]))
    services = run(["docker", "compose", "config", "--services"], capture_output=True)
    names = set(services.stdout.split())
    for service in ("socio-bot", "langfuse-web", "langfuse-db"):
        results.append(report(f"service {service} defined", service in names))

    if options.config or not all(results):
        if not all(results):
            print("\nFix the failures above before the smoke test.")
        return 0 if all(results) else 1

    if options.build:
        print("== Build")
        build = run(["docker", "compose", "build", "socio-bot"])
        if not report("image builds", build.returncode == 0):
            return 1

    print("== Smoke test (throwaway container, database in /tmp)")
    smoke = run([
        "docker", "compose", "run", "--rm", "--no-deps",
        "-e", "SOCIO_DATA_DIR=/tmp/socio-smoke", "-e", "AGENT_STATE_DB=",
        "socio-bot", "python", "-c", SMOKE,
    ])
    results.append(report("smoke test", smoke.returncode == 0))

    print("\nAll checks passed." if all(results) else "\nSome checks failed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
