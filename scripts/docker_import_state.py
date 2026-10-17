"""
Copy Socio's local database (app/agent_state.sqlite3) into the Docker data volume, once, when moving the
bot into Docker. Without it the container starts empty: it would handle the last week of email again and
forget your rules, routines, and open cards.

    python scripts/docker_import_state.py            refuse if the volume already has a database
    python scripts/docker_import_state.py --force    replace the database in the volume

Stop the bot first (both the one outside Docker and the container). The local file is copied with SQLite's
backup API, so the copy is consistent even with a WAL file next to it; the snapshot is kept in backups/.
"""

import argparse
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app" / "agent_state.sqlite3"

# Runs inside a throwaway container as appuser, so the file in the volume is owned by appuser.
COPY_IN = r"""
import os, sqlite3, sys
target = os.path.join(os.environ["SOCIO_DATA_DIR"], "agent_state.sqlite3")
force = sys.argv[1] == "force"
if os.path.exists(target) and not force:
    print("The data volume already has a database. Nothing copied (use --force to replace it).")
    sys.exit(2)
for suffix in ("", "-wal", "-shm"):
    if os.path.exists(target + suffix):
        os.remove(target + suffix)
source = sqlite3.connect("file:/import/state.sqlite3?mode=ro&immutable=1", uri=True)
dest = sqlite3.connect(target)
source.backup(dest)
dest.execute("PRAGMA journal_mode=WAL")
version = dest.execute("PRAGMA user_version").fetchone()[0]
ok = dest.execute("PRAGMA integrity_check").fetchone()[0]
dest.close()
print(f"Copied into the volume: schema v{version}, integrity {ok}.")
sys.exit(0 if ok == "ok" else 1)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="replace a database already in the volume")
    options = parser.parse_args()

    if not SOURCE.is_file():
        print(f"No local database at {SOURCE.relative_to(ROOT)}: nothing to import.")
        return 1

    running = subprocess.run(["docker", "compose", "ps", "--status", "running", "-q", "socio-bot"],
                             cwd=ROOT, capture_output=True, text=True)
    if running.returncode != 0:
        print("Docker isn't reachable. Start Docker Desktop and try again.")
        return 1
    if running.stdout.strip():
        print("The socio-bot container is running. Stop it first: docker compose stop socio-bot")
        return 1

    if not options.force:
        exists = subprocess.run(["docker", "compose", "run", "--rm", "--no-deps", "socio-bot", "python", "-c",
                                 "import os, sys; sys.exit(0 if os.path.exists(os.path.join(os.environ['SOCIO_DATA_DIR'], 'agent_state.sqlite3')) else 1)"],
                                cwd=ROOT, capture_output=True, text=True)
        if exists.returncode == 0:
            print("The data volume already has a database. Nothing copied (use --force to replace it).")
            return 2

    backups = ROOT / "backups"
    backups.mkdir(exist_ok=True)
    snapshot = backups / f"agent_state-docker-import-{datetime.now():%Y%m%d-%H%M%S}.sqlite3"
    source = sqlite3.connect(f"file:{SOURCE.as_posix()}?mode=ro", uri=True)
    dest = sqlite3.connect(snapshot)
    source.backup(dest)
    check = dest.execute("PRAGMA integrity_check").fetchone()[0]
    dest.close()
    source.close()
    if check != "ok":
        print(f"The local database failed its integrity check ({check}). Nothing copied.")
        return 1
    print(f"Snapshot of the local database: {snapshot.relative_to(ROOT)}")

    result = subprocess.run([
        "docker", "compose", "run", "--rm", "--no-deps",
        "-v", f"{snapshot}:/import/state.sqlite3:ro",
        "socio-bot", "python", "-c", COPY_IN, "force" if options.force else "keep",
    ], cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
