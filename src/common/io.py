"""Paths and config loading shared by every stage."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name="default"):
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)
