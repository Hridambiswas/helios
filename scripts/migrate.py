#!/usr/bin/env python3
# scripts/migrate.py — Alembic migration helper CLI
# Author: Hridam Biswas | Project: Helios
"""
Thin wrapper around Alembic for running, generating, and inspecting migrations.

Usage:
    python scripts/migrate.py upgrade          # Apply all pending migrations
    python scripts/migrate.py downgrade -1     # Roll back one revision
    python scripts/migrate.py current          # Show current DB revision
    python scripts/migrate.py history          # Show revision history
    python scripts/migrate.py generate "msg"   # Auto-generate a new revision
"""
import subprocess
import sys
from pathlib import Path

ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"


def _run(args: list[str]) -> int:
    cmd = ["alembic", "-c", str(ALEMBIC_INI), *args]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    action = sys.argv[1]

    if action == "upgrade":
        target = sys.argv[2] if len(sys.argv) > 2 else "head"
        sys.exit(_run(["upgrade", target]))

    elif action == "downgrade":
        target = sys.argv[2] if len(sys.argv) > 2 else "-1"
        sys.exit(_run(["downgrade", target]))

    elif action == "current":
        sys.exit(_run(["current"]))

    elif action == "history":
        sys.exit(_run(["history", "--verbose"]))

    elif action == "generate":
        if len(sys.argv) < 3:
            print("Usage: migrate.py generate <message>")
            sys.exit(1)
        msg = sys.argv[2]
        sys.exit(_run(["revision", "--autogenerate", "-m", msg]))

    else:
        print(f"Unknown action: {action}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
