# hc_ia_retail/utils.py
from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Best-effort seeding across Python/NumPy/(PyTorch if available).

    This improves run-to-run reproducibility but cannot guarantee full determinism
    across all hardware/drivers/kernels.
    """

    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))

    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        # Best-effort determinism flags (may raise depending on version/device)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        # torch not installed / not usable
        pass


def _run_git(args: list[str], project_root: str | None = None) -> subprocess.CompletedProcess[str] | None:
    """Run a git command and return the CompletedProcess, or None if git fails."""

    cwd = None
    if project_root:
        cwd = str(Path(project_root).expanduser().resolve())

    try:
        cp = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return cp
    except Exception:
        return None


def get_git_commit(project_root: str | None = None) -> str | None:
    """Return current git commit hash (HEAD) or None if unavailable."""

    cp = _run_git(["rev-parse", "HEAD"], project_root=project_root)
    if cp is None or cp.returncode != 0:
        return None
    out = (cp.stdout or "").strip()
    return out or None


def is_git_dirty(project_root: str | None = None) -> bool | None:
    """Return True if git working tree is dirty, False if clean, None if unavailable."""

    cp = _run_git(["status", "--porcelain"], project_root=project_root)
    if cp is None or cp.returncode != 0:
        return None
    out = (cp.stdout or "").strip()
    return bool(out)


def get_git_state(project_root: str | None = None) -> dict[str, Any]:
    """Return git state info: commit, dirty, availability, and error message if any."""
    commit = get_git_commit(project_root=project_root)
    dirty = is_git_dirty(project_root=project_root)
    available = True
    error = None
    if commit is None and dirty is None:
        available = False
        error = "git unavailable"
    return {
        "commit": commit,
        "dirty": dirty,
        "available": available,
        "error": error,
    }


def get_pip_freeze() -> str | None:
    """Return `pip freeze` output or None if all attempts fail."""
    result = pip_freeze()
    if result is not None:
        return result

    # fallback: try python -m pip freeze
    try:
        cp = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if cp.returncode == 0:
            out = (cp.stdout or "").strip()
            if out:
                return out
    except Exception:
        pass

    # fallback: try pip freeze directly
    try:
        cp = subprocess.run(
            ["pip", "freeze"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if cp.returncode == 0:
            out = (cp.stdout or "").strip()
            if out:
                return out
    except Exception:
        pass

    return None


def sha256_file(path: str) -> str:
    """Compute SHA256 hash of a file (alias to hash_file_sha256)."""
    return hash_file_sha256(path)


def hash_file_sha256(path: str) -> str:
    """Compute SHA256 of a file (buffered)."""

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found for sha256: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_env_versions() -> dict[str, Any]:
    """Return environment info + key package versions (best-effort)."""

    base: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {},
    }

    pkgs = [
        "numpy",
        "pandas",
        "gymnasium",
        "stable_baselines3",
        "torch",
    ]

    versions: dict[str, str] = {}
    for name in pkgs:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            # package not installed / not discoverable
            pass

    base["packages"] = versions
    return base


def pip_freeze() -> str | None:
    """Return `pip freeze` output (string) or None if it fails."""

    try:
        cp = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if cp.returncode != 0:
            return None
        out = (cp.stdout or "").strip()
        return out if out else None
    except Exception:
        return None