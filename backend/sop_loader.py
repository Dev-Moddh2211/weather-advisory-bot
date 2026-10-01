from pathlib import Path
from typing import Any
import yaml


def load_sops(path: str | Path = "sops/sops.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or not isinstance(data.get("sops"), list) or not isinstance(data.get("severity_order"), dict):
        raise ValueError("Invalid SOP configuration")
    for sop in data["sops"]:
        if not all(key in sop for key in ("id", "name", "conditions", "severity", "advice", "priority")):
            raise ValueError("Every SOP requires id, name, conditions, severity, advice, and priority")
    return data
