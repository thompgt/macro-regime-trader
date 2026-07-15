"""PostToolUse hook: auto-format/fix the just-edited file if it's a .py file.

Reads the hook input JSON from stdin, extracts tool_input.file_path, and runs
`ruff format` + `ruff check --fix` on it. No-ops (exit 0) for non-.py files or
missing paths so it never blocks the tool call it hooks.
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    file_path = payload.get("tool_input", {}).get("file_path", "")
    if not file_path.endswith(".py"):
        return 0

    subprocess.run(["ruff", "format", file_path], capture_output=True)
    subprocess.run(["ruff", "check", "--fix", file_path], capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
