# hc_ia_retail/utils/manifest.py
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_jsonable(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    # fallback (repr) for objects not serializable
    return repr(x)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def collect_git(repo_root: Path) -> Dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    dirty = (status is not None and status != "")
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_json(path: Path, patch: Dict[str, Any]) -> None:
    base = read_json(path) if path.exists() else {}
    # shallow merge is enough for our manifests
    base.update(patch)
    write_json(path, base)


def collect_runtime_env() -> Dict[str, Any]:
    return {
        "generated_at_utc": utc_now_z(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }


def pip_freeze_text() -> str | None:
    out = _run([sys.executable, "-m", "pip", "freeze"])
    return out


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")