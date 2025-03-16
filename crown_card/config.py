"""Config loading helpers. All parameters live in configs/*.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repository root: crown-card/
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
DEFAULT_DB_PATH = ROOT / "outputs" / "crown.db"
OUTPUT_DIR = ROOT / "outputs"


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML config by base name (with or without .yaml extension)."""
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    path = CONFIG_DIR / name
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def economics_config() -> dict[str, Any]:
    return load_yaml("economics")


def underwriting_config() -> dict[str, Any]:
    return load_yaml("underwriting")


def simulation_config() -> dict[str, Any]:
    return load_yaml("simulation")


def scenarios_config() -> dict[str, Any]:
    return load_yaml("scenarios")


def db_url(path: str | Path | None = None) -> str:
    """SQLite URL for the Crown database."""
    p = Path(path) if path else DEFAULT_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{p}"
