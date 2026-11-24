"""Paths, config loading and results saving shared by every stage."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def load_config(name="default"):
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def git_commit():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def save_results(name, results, configs):
    """Write results/<name>.json with the configs used and the git commit, as CLAUDE.md requires."""
    RESULTS.mkdir(exist_ok=True)
    payload = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_commit(),
        "configs": configs,
        "results": results,
    }
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path
