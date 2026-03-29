# scripts/train/train.py
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hc_ia_retail.run_registry import append_run_index, safe_float  # noqa: E402

from hc_ia_retail.callbacks import RunControlCallback  # noqa: E402
from hc_ia_retail.config import (  # noqa: E402
    data_cfg,
    env_cfg,
    model_cfg,
    paths_cfg,
    regime_cfg,
    sac_cfg,
)
from hc_ia_retail.data import load_training_dataframe, split_train_val_test  # noqa: E402
from hc_ia_retail.env import RetailTradingEnv  # noqa: E402
from hc_ia_retail.models import GRUWindowExtractor  # noqa: E402
from hc_ia_retail.utils import set_global_seeds  # noqa: E402


# -----------------------------
# Helpers (json / audit)
# -----------------------------
def json_sanitize(obj: Any) -> Any:
    """Recursively convert objects into JSON-serializable values (audit-safe)."""
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, type):
        mod = getattr(obj, "__module__", "")
        qual = getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
        if mod and qual:
            return f"{mod}.{qual}"
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    try:
        if isinstance(obj, pd.Timestamp):
            if obj.tzinfo is not None:
                return obj.tz_convert("UTC").to_pydatetime().replace(tzinfo=None).isoformat() + "Z"
            return obj.to_pydatetime().isoformat()
    except Exception:
        pass

    try:
        if isinstance(obj, set):
            return [json_sanitize(x) for x in sorted(list(obj), key=lambda x: str(x))]
        if hasattr(obj, "value") and not isinstance(obj, (str, bytes, bytearray)):
            v = getattr(obj, "value")
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def info_get(info: dict, keys: list[str], default=np.nan):
    for k in keys:
        if k in info and info[k] is not None:
            return info[k]
    return default


def compute_north_star_score(
    sharpe: float,
    maxdd: float,
    turnover_mean: float,
    d: float = 0.20,
    tau: float = 0.01,
    lambda_dd: float = 2.0,
    lambda_to: float = 1.0,
) -> float:
    """
    score = sharpe - lambda_dd*max(0, maxdd - d) - lambda_to*max(0, turnover_mean - tau)
    - maxdd in [0,1]
    - turnover_mean is mean(|Δpos|)
    """
    sharpe = float(sharpe)
    maxdd = float(maxdd)
    turnover_mean = float(turnover_mean)
    dd_pen = max(0.0, maxdd - float(d))
    to_pen = max(0.0, turnover_mean - float(tau))
    return float(sharpe - float(lambda_dd) * dd_pen - float(lambda_to) * to_pen)




def _best_effort_git_state(project_root: Path) -> dict[str, Any]:
    try:
        from hc_ia_retail.utils import get_git_state  # type: ignore

        st = get_git_state(str(project_root))
        if isinstance(st, dict):
            return st
    except Exception:
        pass
    try:
        from hc_ia_retail.utils import get_git_commit, is_git_dirty  # type: ignore

        commit = get_git_commit(str(project_root))
        dirty = is_git_dirty(str(project_root))
        return {"available": commit is not None or dirty is not None, "commit": commit, "dirty": dirty, "error": None}
    except Exception as e:
        return {"available": False, "commit": None, "dirty": None, "error": f"{type(e).__name__}: {e}"}


def _best_effort_pip_freeze() -> str | None:
    try:
        from hc_ia_retail.utils import get_pip_freeze  # type: ignore

        out = get_pip_freeze()
        return out if out else None
    except Exception:
        pass
    try:
        from hc_ia_retail.utils import pip_freeze  # type: ignore

        out = pip_freeze()
        return out if out else None
    except Exception:
        pass

    import subprocess

    try:
        cp = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if cp.returncode == 0 and (cp.stdout or "").strip():
            return (cp.stdout or "").strip()
    except Exception:
        pass

    try:
        cp = subprocess.run(
            ["pip", "freeze"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if cp.returncode == 0 and (cp.stdout or "").strip():
            return (cp.stdout or "").strip()
    except Exception:
        pass

    return None


# -----------------------------
# Unified manifest helpers
# -----------------------------
def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")
    except Exception:
        pass


def _snapshot_config_py(run_dir: Path) -> str | None:
    """Copy repo root config.py into run_dir/snapshots for audit reproducibility."""
    try:
        snap_dir = run_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        src = (PROJECT_ROOT / "config.py").resolve()
        if src.exists():
            dst = snap_dir / "config_snapshot.py"
            shutil.copy2(str(src), str(dst))
            return str(dst)
    except Exception:
        pass
    return None


def _write_train_manifest(
    *,
    run_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    seed: int,
    data_path_used: str,
    dataset_sha: str | None,
    feature_cols: list[str],
    per_step_dim: int,
    window_size: int,
    window_dim: int,
    df_train,
    df_val,
    df_test,
    policy_kwargs: dict[str, Any],
    checkpoint_freq_used: int,
    sanity_report: dict[str, Any] | None,
    git_state: dict[str, Any],
    pip_freeze_path: str | None,
    pip_freeze_txt: str | None,
    data_source_meta: dict[str, Any] | None = None,
) -> Path:
    """Write a single audit-grade manifest for the run."""

    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best"

    snap_path = _snapshot_config_py(run_dir)

    manifest_path = run_dir / "train_manifest.json"

    payload: dict[str, Any] = {
        "kind": "train_manifest_v1",
        "generated_at_utc": _utc_now_z(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "command_line": " ".join([sys.executable] + sys.argv),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "device": str(getattr(args, "device", "")),
            "timesteps": int(getattr(args, "timesteps", 0)),
            "eval_freq": int(getattr(args, "eval_freq", 0)),
            "chunk": int(getattr(args, "chunk", 0)),
            "resume": bool(getattr(args, "resume", False)),
        },
        "seed": int(seed),
        "git": git_state,
        "data": {
            "dataset_path": str(data_path_used),
            "dataset_sha256": dataset_sha,
            "rows": {
                "train": int(len(df_train)) if df_train is not None else 0,
                "val": int(len(df_val)) if df_val is not None else 0,
                "test": int(len(df_test)) if df_test is not None else 0,
            },
            "source_meta": data_source_meta or {},
        },
        "features": {
            "feature_cols": list(feature_cols),
        },
        "observation": {
            "per_step_dim": int(per_step_dim),
            "window_size": int(window_size),
            "window_dim": int(window_dim),
        },
        "configs": {
            "data_cfg": asdict(data_cfg),
            "env_cfg": asdict(env_cfg),
            "sac_cfg": asdict(sac_cfg),
            "model_cfg": asdict(model_cfg),
            "paths_cfg": asdict(paths_cfg),
            "regime_cfg": asdict(regime_cfg),
        },
        "algo": {
            "name": "SAC",
            "policy_kwargs": json_sanitize(policy_kwargs),
        },
        "sanity": sanity_report,
        "artifacts": {
            "tb_dir": str(tb_dir),
            "checkpoints_dir": str(ckpt_dir),
            "best_dir": str(best_dir),
            "vecnormalize_path": str(run_dir / "vecnormalize.pkl"),
            "data_report_path": str(run_dir / "data_report.json"),
            "pip_freeze_path": pip_freeze_path,
            "pip_freeze_included": bool(pip_freeze_txt),
            "config_snapshot_path": snap_path,
            "checkpoint_freq_used": int(checkpoint_freq_used),
        },
    }

    _safe_write_json(manifest_path, payload)
    return manifest_path


def _patch_train_manifest_finalize(
    *,
    run_dir: Path,
    trained_steps_final: int,
    total_timesteps: int,
    final_model_path: Path,
    best_model_path: Path,
    best_metrics_path: Path,
    last_checkpoint: str,
) -> None:
    """Update the manifest with final pointers (best model, final export, last checkpoint)."""
    p = run_dir / "train_manifest.json"
    if not p.exists():
        return
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return

    j["final"] = {
        "trained_steps_final": int(trained_steps_final),
        "target_timesteps": int(total_timesteps),
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
        "best_metrics_val_path": str(best_metrics_path),
        "last_checkpoint": str(last_checkpoint),
        "finalized_at_utc": _utc_now_z(),
    }

    _safe_write_json(p, j)


# -----------------------------
# Data report (data quality)
# -----------------------------
def _best_effort_cfg_value(*objs: Any, keys: tuple[str, ...]) -> Any:
    for o in objs:
        if o is None:
            continue
        for k in keys:
            try:
                v = getattr(o, k)
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                return v
            except Exception:
                continue
    return None


def _parse_interval_to_hours(x: Any) -> float:
    if x is None:
        return 1.0
    if isinstance(x, (int, float)):
        try:
            xf = float(x)
            if xf > 0:
                return xf
        except Exception:
            return 1.0

    s = str(x).strip().lower()
    if not s:
        return 1.0

    aliases = {
        "1h": 1.0,
        "60m": 1.0,
        "60min": 1.0,
        "1hour": 1.0,
        "1hr": 1.0,
        "4h": 4.0,
        "1d": 24.0,
        "24h": 24.0,
    }
    if s in aliases:
        return float(aliases[s])

    m = re.match(r"^\s*(\d+)\s*([smhd])\s*$", s)
    if m:
        n = float(m.group(1))
        u = m.group(2)
        if u == "s":
            return n / 3600.0
        if u == "m":
            return n / 60.0
        if u == "h":
            return n
        if u == "d":
            return n * 24.0

    m2 = re.match(r"^\s*(\d+)\s*(sec|secs|second|seconds|min|mins|minute|minutes|hour|hours|day|days)\s*$", s)
    if m2:
        n = float(m2.group(1))
        u = m2.group(2)
        if u.startswith("sec"):
            return n / 3600.0
        if u.startswith("min"):
            return n / 60.0
        if u.startswith("hour"):
            return n
        if u.startswith("day"):
            return n * 24.0

    return 1.0


def _infer_utc_timestamps(df_in) -> pd.Series | None:
    try:
        if hasattr(df_in, "columns"):
            for col in ("timestamp", "datetime", "date", "time"):
                if col in df_in.columns:
                    s = pd.to_datetime(df_in[col], utc=True, errors="coerce")
                    s = pd.Series(s)
                    s = s.dropna()
                    if len(s) == 0:
                        return None
                    return s
        idx = getattr(df_in, "index", None)
        if isinstance(idx, pd.DatetimeIndex):
            s = pd.to_datetime(idx, utc=True, errors="coerce")
            s = pd.Series(s)
            s = s.dropna()
            if len(s) == 0:
                return None
            return s
    except Exception:
        return None
    return None


def _count_nonfinite(df_in, cols: list[str]) -> int:
    if df_in is None:
        return 0
    n = 0
    for c in cols:
        if c not in getattr(df_in, "columns", []):
            continue
        try:
            arr = pd.to_numeric(df_in[c], errors="coerce").to_numpy(dtype=np.float64)
            n += int(np.sum(~np.isfinite(arr)))
        except Exception:
            continue
    return int(n)


def _build_data_quality_report(
    *,
    df_raw,
    df_feat,
    df_train,
    df_val,
    df_test,
    feature_cols: list[str],
    data_path_used: str,
    dataset_sha: str | None,
    data_source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    instrument = _best_effort_cfg_value(data_cfg, env_cfg, model_cfg, keys=("instrument", "symbol", "pair", "ticker"))
    interval = _best_effort_cfg_value(data_cfg, env_cfg, keys=("interval", "timeframe", "bar", "bar_size", "freq"))
    expected_hours = _parse_interval_to_hours(interval)

    n_rows_raw = int(len(df_raw)) if df_raw is not None else 0
    n_rows_feat = int(len(df_feat)) if df_feat is not None else 0

    pct_dropped_by_dropna = None
    if n_rows_raw > 0:
        pct_dropped_by_dropna = float(max(0, n_rows_raw - n_rows_feat) / float(n_rows_raw))

    ts = _infer_utc_timestamps(df_feat)
    if ts is None:
        ts = _infer_utc_timestamps(df_raw)
    time_min = None
    time_max = None
    duplicates_count = 0
    gap_count = 0
    max_gap_hours = 0.0

    if ts is not None and len(ts) > 0:
        try:
            ts_sorted = ts.sort_values()
            duplicates_count = int(ts_sorted.duplicated().sum())

            time_min = ts_sorted.iloc[0].to_pydatetime().replace(tzinfo=None).isoformat() + "Z"
            time_max = ts_sorted.iloc[-1].to_pydatetime().replace(tzinfo=None).isoformat() + "Z"

            diffs = ts_sorted.diff().dropna()
            if len(diffs) > 0:
                diff_hours = diffs.dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0
                gap_mask = ~np.isclose(diff_hours, float(expected_hours))
                gap_count = int(np.sum(gap_mask))
                if diff_hours.size:
                    max_gap_hours = float(np.max(diff_hours))
        except Exception:
            pass

    base_cols = [c for c in ("open", "high", "low", "close", "volume") if df_feat is not None and c in df_feat.columns]
    feat_cols_present = [c for c in feature_cols if df_feat is not None and c in df_feat.columns]
    cols_checked = list(dict.fromkeys(base_cols + feat_cols_present))
    nonfinite_count = _count_nonfinite(df_feat, cols_checked)

    ret_col = None
    for cand in ("log_ret_1", "ret_fwd", "log_ret_1_fwd", "log_ret_1_fwd_used", "ret_fwd_used"):
        if df_feat is not None and cand in df_feat.columns:
            ret_col = cand
            break

    ret_quantiles = None
    if ret_col is not None:
        try:
            r = pd.to_numeric(df_feat[ret_col], errors="coerce").to_numpy(dtype=np.float64)
            r = r[np.isfinite(r)]
            if r.size:
                qs = np.quantile(r, [0.01, 0.05, 0.50, 0.95, 0.99])
                ret_quantiles = {
                    "col": ret_col,
                    "p1": float(qs[0]),
                    "p5": float(qs[1]),
                    "p50": float(qs[2]),
                    "p95": float(qs[3]),
                    "p99": float(qs[4]),
                }
        except Exception:
            ret_quantiles = None

    report: dict[str, Any] = {
        "instrument": instrument,
        "interval": interval,
        "data_path": data_path_used,
        "sha256": dataset_sha,
        "n_rows_raw": int(n_rows_raw),
        "n_rows_feat": int(n_rows_feat),
        "n_rows_train": int(len(df_train)) if df_train is not None else 0,
        "n_rows_val": int(len(df_val)) if df_val is not None else 0,
        "n_rows_test": int(len(df_test)) if df_test is not None else 0,
        "time_min": time_min,
        "time_max": time_max,
        "duplicates_count": int(duplicates_count),
        "gap_count": int(gap_count),
        "max_gap_hours": float(max_gap_hours),
        "nonfinite_count": int(nonfinite_count),
        "pct_dropped_by_dropna": pct_dropped_by_dropna,
        "ret_quantiles": ret_quantiles,
        "regime_enabled": bool(getattr(regime_cfg, "enabled", False)),
        "data_source_meta": data_source_meta or {},
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    return report




# -----------------------------
# Window stacking wrapper
# -----------------------------
class WindowStackWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, window_size: int):
        super().__init__(env)
        self.window_size = int(window_size)
        self.buf: list[np.ndarray] = []
        orig = env.observation_space
        assert isinstance(orig, gym.spaces.Box)
        self.per_step = int(orig.shape[0])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.per_step * self.window_size,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buf = [obs.copy() for _ in range(self.window_size)]
        return self.observation(obs), info

    def observation(self, obs):
        self.buf.append(obs.copy())
        self.buf = self.buf[-self.window_size :]
        return np.concatenate(self.buf, axis=0).astype(np.float32)


# -----------------------------
# Env builders
# -----------------------------
def make_env(df, feature_cols, max_steps: int | None):
    def _init():
        e = RetailTradingEnv(df, feature_cols=feature_cols, max_steps=max_steps)
        e = WindowStackWrapper(e, window_size=int(data_cfg.window_size))
        e = Monitor(e)
        return e

    return _init


def make_raw_env_for_sanity(df, feature_cols, max_steps: int | None):
    e = RetailTradingEnv(df, feature_cols=feature_cols, max_steps=max_steps)
    e = WindowStackWrapper(e, window_size=int(data_cfg.window_size))
    e = Monitor(e)
    return e


# -----------------------------
# Callbacks
# -----------------------------
class ValNorthStarBestCallback(BaseCallback):
    """
    At each eval_freq:
      - rollout on env_val (VecNormalize frozen: training=False)
      - compute Sharpe / MaxDD / turnover_mean / mean_abs_pos / action_std
      - compute score_val using compute_north_star_score
      - compute multi-window aggregates (median) if enabled:
            score_val_agg = median(score_val_i)
            score_val_worst = min(score_val_i)
      - select best checkpoint ONLY among checkpoints that satisfy anti-collapse guard:
            action_std_val >= eps_action AND turnover_mean_val >= eps_to
        (only model selection; does not change learning/MDP)
      - if improved, save best/best_model.zip and best/metrics_val.json

    Keeps VecNormalize consistent: we also save vecnormalize.pkl (and copy into best/).
    """

    def __init__(
        self,
        eval_env: VecNormalize,
        vec_train: VecNormalize,
        vec_path: Path,
        best_dir: Path,
        eval_freq_steps: int,
        freq_per_year: float,
        north_star_params: dict[str, float] | None = None,
        deterministic: bool = True,
        max_rollout_steps: int | None = None,
        eps_action_std: float = 0.01,
        eps_turnover: float = 1e-6,
        use_val_windows: bool = True,
        window_len: int = 512,
        window_stride: int = 256,
        n_windows: int = 3,
        val_windows: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.vec_train = vec_train
        self.vec_path = Path(vec_path)
        self.best_dir = Path(best_dir)
        self.eval_freq_steps = int(eval_freq_steps)
        self.freq_per_year = float(freq_per_year)
        self.deterministic = bool(deterministic)
        self.max_rollout_steps = int(max_rollout_steps) if max_rollout_steps is not None else None

        self.eps_action_std = float(eps_action_std)
        self.eps_turnover = float(eps_turnover)

        self.use_val_windows = bool(use_val_windows)
        self.window_len = int(max(16, window_len))
        self.window_stride = int(max(1, window_stride))
        self.n_windows = int(max(1, n_windows))

        if val_windows is not None:
            self.val_windows = [(int(a), int(b)) for (a, b) in val_windows if int(b) > int(a)]
        else:
            self.val_windows = None

        ns = north_star_params or {"d": 0.20, "tau": 0.01, "lambda_dd": 2.0, "lambda_to": 1.0}
        self.ns_params = {k: float(v) for k, v in ns.items()}

        self._last_eval_step = -1
        self.best_score = -float("inf")
        self.best_step = 0
        self.metrics_path = self.best_dir / "metrics_val.json"
        self.best_model_path = self.best_dir / "best_model.zip"
        self.best_vec_path = self.best_dir / "vecnormalize.pkl"
        self.best_score_val = float("nan")

    def _on_training_start(self) -> None:
        self.best_dir.mkdir(parents=True, exist_ok=True)
        if self.metrics_path.exists():
            try:
                j = json.loads(self.metrics_path.read_text(encoding="utf-8"))
                prev = j.get("north_star", {}).get("score_val_agg", None)
                if prev is None:
                    prev = j.get("north_star", {}).get("score_val", None)
                if prev is None:
                    prev = j.get("score_val_agg", j.get("score_val", None))
                if prev is not None and np.isfinite(float(prev)):
                    self.best_score = float(prev)
                    self.best_score_val = float(prev)
                self.best_step = int(j.get("step", j.get("best_step", 0)) or 0)
            except Exception:
                pass

    @staticmethod
    def _compute_metrics(
        equity: np.ndarray, pos: np.ndarray, actions: np.ndarray, freq_per_year: float
    ) -> tuple[float, float, float, float, float]:
        eq = np.asarray(equity, dtype=np.float64)
        if eq.size < 3:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        eq = np.nan_to_num(eq, nan=0.0, posinf=0.0, neginf=0.0)
        eq = np.where(eq <= 0.0, 1e-12, eq)

        log_eq = np.log(eq + 1e-12)
        rets = np.diff(log_eq)
        mu = float(np.mean(rets)) if rets.size else 0.0
        sig = float(np.std(rets) + 1e-12) if rets.size else 1e-12
        sharpe = float((mu / sig) * math.sqrt(float(freq_per_year))) if rets.size else 0.0

        peak = np.maximum.accumulate(eq)
        peak = np.where(peak <= 0.0, 1e-12, peak)
        dd = 1.0 - (eq / peak)
        max_dd = float(np.max(dd)) if dd.size else 0.0

        p = np.nan_to_num(np.asarray(pos, dtype=np.float64), nan=0.0)
        turnover_mean = float(np.mean(np.abs(np.diff(p)))) if p.size >= 2 else 0.0
        p_abs = np.abs(p[np.isfinite(p)])
        mean_abs_pos = float(np.mean(p_abs)) if p_abs.size else 0.0

        a = np.asarray(actions, dtype=np.float64).reshape(-1)
        a = a[np.isfinite(a)]
        action_std = float(np.std(a)) if a.size else 0.0

        return sharpe, max_dd, turnover_mean, mean_abs_pos, action_std

    def _build_default_windows(self) -> list[tuple[int, int]]:
        wins: list[tuple[int, int]] = []
        start = 0
        for _ in range(self.n_windows):
            end = start + self.window_len
            wins.append((int(start), int(end)))
            start += self.window_stride
        return wins

    def _rollout_val_window(self, start_offset: int, end_offset: int) -> dict[str, Any]:
        self.eval_env.training = False
        self.eval_env.norm_reward = False

        obs = self.eval_env.reset()

        equities: list[float] = []
        positions: list[float] = []
        actions: list[float] = []

        done = False
        steps = 0

        start_offset = int(max(0, start_offset))
        end_offset = int(max(start_offset + 1, end_offset))
        record_len = int(end_offset - start_offset)

        cap = self.max_rollout_steps
        if cap is None:
            cap = 10_000_000

        last_eq = 1.0

        while not done and steps < cap and steps < start_offset:
            act, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _reward, dones, infos = self.eval_env.step(act)
            done = bool(np.asarray(dones).reshape(-1)[0])
            steps += 1

        rec_steps = 0
        while not done and steps < cap and rec_steps < record_len:
            act, _ = self.model.predict(obs, deterministic=self.deterministic)

            try:
                a0 = float(np.asarray(act).reshape(-1)[0])
            except Exception:
                a0 = float("nan")
            actions.append(a0)

            obs, _reward, dones, infos = self.eval_env.step(act)
            info0 = infos[0] if isinstance(infos, list) else infos

            eq = info_get(info0, ["equity", "eq", "nav", "portfolio_value"], default=np.nan)
            pos = info_get(info0, ["position", "pos", "exposure", "pos_new", "pos_prev"], default=np.nan)

            if np.isfinite(eq):
                last_eq = float(eq)
            equities.append(float(last_eq))
            positions.append(float(pos) if np.isfinite(pos) else float("nan"))

            done = bool(np.asarray(dones).reshape(-1)[0])
            steps += 1
            rec_steps += 1

        if not equities:
            equities = [1.0, 1.0, 1.0]
        if not positions:
            positions = [0.0 for _ in range(len(equities))]
        if not actions:
            actions = [0.0 for _ in range(max(1, len(equities) - 1))]

        sharpe, max_dd, turnover_mean, mean_abs_pos, action_std = self._compute_metrics(
            np.asarray(equities), np.asarray(positions), np.asarray(actions), self.freq_per_year
        )

        score_val = compute_north_star_score(
            sharpe=sharpe,
            maxdd=max_dd,
            turnover_mean=turnover_mean,
            **self.ns_params,
        )

        return {
            "steps_total": int(steps),
            "steps_window": int(rec_steps),
            "window": {"start": int(start_offset), "end": int(end_offset)},
            "val_sharpe": float(sharpe),
            "val_max_dd": float(max_dd),
            "val_turnover_mean": float(turnover_mean),
            "val_mean_abs_pos": float(mean_abs_pos),
            "val_action_std": float(action_std),
            "score_val": float(score_val),
        }

    def _evaluate_val(self) -> dict[str, Any]:
        if self.use_val_windows:
            wins = self.val_windows if self.val_windows is not None else self._build_default_windows()
            ms = [self._rollout_val_window(a, b) for (a, b) in wins]

            scores = np.asarray([m["score_val"] for m in ms], dtype=np.float64)
            to = np.asarray([m["val_turnover_mean"] for m in ms], dtype=np.float64)
            astd = np.asarray([m["val_action_std"] for m in ms], dtype=np.float64)
            dd = np.asarray([m["val_max_dd"] for m in ms], dtype=np.float64)
            sh = np.asarray([m["val_sharpe"] for m in ms], dtype=np.float64)
            mapos = np.asarray([m["val_mean_abs_pos"] for m in ms], dtype=np.float64)

            def _med(x: np.ndarray) -> float:
                x = x[np.isfinite(x)]
                return float(np.median(x)) if x.size else float("nan")

            def _min(x: np.ndarray) -> float:
                x = x[np.isfinite(x)]
                return float(np.min(x)) if x.size else float("nan")

            def _max(x: np.ndarray) -> float:
                x = x[np.isfinite(x)]
                return float(np.max(x)) if x.size else float("nan")

            return {
                "mode": "val_windows_v1",
                "windows": ms,
                "score_val_agg": _med(scores),
                "score_val_worst": _min(scores),
                "val_sharpe": _med(sh),
                "val_max_dd": _med(dd),
                "val_turnover_mean": _med(to),
                "val_mean_abs_pos": _med(mapos),
                "val_action_std": _med(astd),
                "val_sharpe_best_window": _max(sh),
                "val_max_dd_worst_window": _max(dd),
            }

        cap = self.max_rollout_steps
        if cap is None:
            cap = 10_000_000
        m = self._rollout_val_window(0, cap)
        return {
            "mode": "single_rollout",
            **m,
            "score_val_agg": float(m["score_val"]),
            "score_val_worst": float(m["score_val"]),
        }

    def _save_best(self, cur_step: int, val_pack: dict[str, Any]) -> None:
        self.best_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.model.save(str(self.best_model_path))
        except Exception:
            pass

        try:
            self.vec_train.save(str(self.vec_path))
        except Exception:
            pass
        try:
            if self.vec_path.exists():
                shutil.copy2(str(self.vec_path), str(self.best_vec_path))
        except Exception:
            pass

        score_val = float(val_pack.get("score_val_agg", val_pack.get("score_val", float("nan"))))

        payload = {
            "step": int(cur_step),
            "north_star": {
                "name": "north_star_v1",
                "definition": "sharpe - lambda_dd*max(0, max_dd-d) - lambda_to*max(0, turnover_mean-tau)",
                "params": dict(self.ns_params),
                "val_sharpe": float(val_pack.get("val_sharpe", float("nan"))),
                "val_max_dd": float(val_pack.get("val_max_dd", float("nan"))),
                "val_turnover_mean": float(val_pack.get("val_turnover_mean", float("nan"))),
                "val_mean_abs_pos": float(val_pack.get("val_mean_abs_pos", float("nan"))),
                "val_action_std": float(val_pack.get("val_action_std", float("nan"))),
                "score_val": float(val_pack.get("score_val", float("nan"))),
                "score_val_agg": float(score_val),
                "score_val_worst": float(val_pack.get("score_val_worst", float("nan"))),
                "mode": str(val_pack.get("mode", "")),
            },
            "selection_guard": {
                "type": "anti_collapse_v1",
                "min_action_std_val": float(self.eps_action_std),
                "min_turnover_mean_val": float(self.eps_turnover),
                "eligible": True,
            },
            "artifacts": {
                "best_model_path": str(self.best_model_path),
                "vecnormalize_path": str(self.best_vec_path) if self.best_vec_path.exists() else str(self.vec_path),
            },
            "val_windows": val_pack.get("windows", None),
            "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }
        try:
            self.metrics_path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_step(self) -> bool:
        cur = int(self.num_timesteps)

        if self.eval_freq_steps <= 0:
            return True
        if cur <= 0 or (cur % self.eval_freq_steps) != 0:
            return True
        if cur == self._last_eval_step:
            return True

        self._last_eval_step = cur
        pack = self._evaluate_val()

        score_agg = float(pack.get("score_val_agg", pack.get("score_val", float("nan"))))
        score_worst = float(pack.get("score_val_worst", pack.get("score_val", float("nan"))))
        sharpe = float(pack.get("val_sharpe", float("nan")))
        max_dd = float(pack.get("val_max_dd", float("nan")))
        turnover_mean = float(pack.get("val_turnover_mean", float("nan")))
        mean_abs_pos = float(pack.get("val_mean_abs_pos", float("nan")))
        action_std = float(pack.get("val_action_std", float("nan")))

        eligible = bool(
            np.isfinite(action_std)
            and np.isfinite(turnover_mean)
            and (action_std >= self.eps_action_std)
            and (turnover_mean >= self.eps_turnover)
        )

        self.logger.record("val/sharpe", float(sharpe))
        self.logger.record("val/max_dd", float(max_dd))
        self.logger.record("val/turnover_mean", float(turnover_mean))
        self.logger.record("val/mean_abs_pos", float(mean_abs_pos))
        self.logger.record("val/action_std", float(action_std))
        self.logger.record("val/eligible", 1.0 if eligible else 0.0)
        self.logger.record("val/eps_action_std", float(self.eps_action_std))
        self.logger.record("val/eps_turnover", float(self.eps_turnover))
        self.logger.record("val/score_agg", float(score_agg))
        self.logger.record("val/score_worst", float(score_worst))
        self.logger.record("val/best_score", float(self.best_score))
        self.logger.record("val/mode", 0.0 if str(pack.get("mode", "")) == "single_rollout" else 1.0)
        self.logger.dump(cur)

        if eligible and np.isfinite(score_agg) and (score_agg > self.best_score + 1e-12):
            self.best_score = float(score_agg)
            self.best_step = cur
            self.best_score_val = float(score_agg)
            pack["score_val"] = float(pack.get("score_val", score_agg))
            self._save_best(cur, pack)

        return True


class TrainingStabilityCallback(BaseCallback):
    """Live training stability diagnostics (TensorBoard) from rollout infos."""

    def __init__(self, log_every_steps: int, run_dir: Path, dsr_clip: float | None):
        super().__init__()
        self.log_every_steps = int(max(1, log_every_steps))
        self.run_dir = Path(run_dir)
        self.dsr_clip = float(dsr_clip) if dsr_clip is not None and float(dsr_clip) > 0 else None

        self._actions: list[float] = []
        self._rewards: list[float] = []
        self._rewards_bps: list[float] = []
        self._turnover: list[float] = []
        self._costs: list[float] = []
        self._pnl_gross: list[float] = []
        self._pnl_net: list[float] = []
        self._clipped_count: int = 0

        self._jsonl_path = self.run_dir / "training_stability.jsonl"

    def _on_training_start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", None)
        if actions is not None:
            a = np.asarray(actions).reshape(-1)
            for x in a:
                if np.isfinite(x):
                    self._actions.append(float(x))

        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            r = np.asarray(rewards).reshape(-1)
            for x in r:
                if np.isfinite(x):
                    self._rewards.append(float(x))

        infos = self.locals.get("infos", None)
        if infos is not None:
            infos_iter = [infos] if isinstance(infos, dict) else list(infos)
            for info in infos_iter:
                if not isinstance(info, dict):
                    continue

                if "turnover" in info:
                    try:
                        self._turnover.append(float(info["turnover"]))
                    except Exception:
                        pass
                if "cost" in info:
                    try:
                        self._costs.append(float(info["cost"]))
                    except Exception:
                        pass
                if "pnl_log_gross" in info:
                    try:
                        self._pnl_gross.append(float(info["pnl_log_gross"]))
                    except Exception:
                        pass
                if "pnl_log_net" in info:
                    try:
                        self._pnl_net.append(float(info["pnl_log_net"]))
                    except Exception:
                        pass
                if "reward_final_bps" in info:
                    try:
                        self._rewards_bps.append(float(info["reward_final_bps"]))
                    except Exception:
                        pass

                if self.dsr_clip is not None:
                    rr = info.get("reward_raw", None)
                    rf = info.get("reward_final", None)
                    try:
                        if rr is not None and rf is not None:
                            rr_f = float(rr)
                            if abs(rr_f) >= (self.dsr_clip - 1e-12):
                                self._clipped_count += 1
                    except Exception:
                        pass

        cur = int(self.num_timesteps)
        if (cur % self.log_every_steps) != 0:
            return True

        a = np.asarray(self._actions, dtype=np.float64) if self._actions else np.array([], dtype=np.float64)
        action_mean = float(a.mean()) if a.size else 0.0
        action_std = float(a.std()) if a.size else 0.0
        pct_sat = float(np.mean(np.abs(a) > 0.95)) if a.size else 0.0

        r = np.asarray(self._rewards, dtype=np.float64) if self._rewards else np.array([], dtype=np.float64)
        reward_mean = float(r.mean()) if r.size else 0.0
        rbps = np.asarray(self._rewards_bps, dtype=np.float64) if self._rewards_bps else np.array([], dtype=np.float64)
        reward_bps_mean = float(rbps.mean()) if rbps.size else 0.0

        to = np.asarray(self._turnover, dtype=np.float64) if self._turnover else np.array([], dtype=np.float64)
        turnover_mean = float(to.mean()) if to.size else 0.0

        c = np.asarray(self._costs, dtype=np.float64) if self._costs else np.array([], dtype=np.float64)
        cost_mean = float(c.mean()) if c.size else 0.0

        pg = np.asarray(self._pnl_gross, dtype=np.float64) if self._pnl_gross else np.array([], dtype=np.float64)
        pnl_gross_mean = float(pg.mean()) if pg.size else 0.0

        pn = np.asarray(self._pnl_net, dtype=np.float64) if self._pnl_net else np.array([], dtype=np.float64)
        pnl_net_mean = float(pn.mean()) if pn.size else 0.0

        pct_reward_clipped = 0.0
        if self.dsr_clip is not None and r.size:
            pct_reward_clipped = float(self._clipped_count / max(1, r.size))

        self.logger.record("train_stability/action_mean", action_mean)
        self.logger.record("train_stability/action_std", action_std)
        self.logger.record("train_stability/pct_action_sat", pct_sat)
        self.logger.record("train_stability/turnover_mean", turnover_mean)
        self.logger.record("train_stability/cost_mean", cost_mean)
        self.logger.record("train_stability/pnl_gross_mean", pnl_gross_mean)
        self.logger.record("train_stability/pnl_net_mean", pnl_net_mean)
        self.logger.record("train_stability/reward_mean_logret", reward_mean)
        if rbps.size:
            self.logger.record("train_stability/reward_mean_bps", reward_bps_mean)
        if self.dsr_clip is not None:
            self.logger.record("train_stability/pct_reward_clipped", pct_reward_clipped)

        try:
            alpha = None
            if hasattr(self.model, "ent_coef") and isinstance(self.model.ent_coef, (int, float)):
                alpha = float(self.model.ent_coef)
            if hasattr(self.model, "ent_coef_tensor") and self.model.ent_coef_tensor is not None:
                try:
                    alpha = float(self.model.ent_coef_tensor.detach().cpu().numpy().reshape(-1)[0])
                except Exception:
                    pass
            if alpha is not None:
                self.logger.record("train_stability/alpha", float(alpha))
        except Exception:
            pass

        self.logger.dump(cur)

        rec = dict(
            t=cur,
            action_mean=action_mean,
            action_std=action_std,
            pct_action_sat=pct_sat,
            turnover_mean=turnover_mean,
            cost_mean=cost_mean,
            pnl_gross_mean=pnl_gross_mean,
            pnl_net_mean=pnl_net_mean,
            reward_mean_logret=reward_mean,
            reward_mean_bps=reward_bps_mean if rbps.size else None,
            pct_reward_clipped=pct_reward_clipped if self.dsr_clip is not None else None,
        )
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(json_sanitize(rec)) + "\n")
        except Exception:
            pass

        self._actions.clear()
        self._rewards.clear()
        self._rewards_bps.clear()
        self._turnover.clear()
        self._costs.clear()
        self._pnl_gross.clear()
        self._pnl_net.clear()
        self._clipped_count = 0

        return True


class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize stats on a fixed cadence AND on training end."""

    def __init__(self, vec_env: VecNormalize, save_path: Path, save_freq: int):
        super().__init__()
        self.vec_env = vec_env
        self.save_path = Path(save_path)
        self.save_freq = int(max(1, save_freq))

    def _on_step(self) -> bool:
        cur = int(self.num_timesteps)
        if cur > 0 and (cur % self.save_freq) == 0:
            try:
                self.vec_env.save(str(self.save_path))
            except Exception:
                pass
        return True

    def _on_training_end(self) -> None:
        try:
            self.vec_env.save(str(self.save_path))
        except Exception:
            pass


class SaveResumeStateCallback(BaseCallback):
    """Persist resume state (trained_steps) frequently for robust recovery."""

    def __init__(self, run_dir: Path, save_freq: int):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.save_freq = int(max(1, save_freq))
        self.path = self.run_dir / "resume_state.json"

    def _write(self, cur: int) -> None:
        try:
            self.path.write_text(json.dumps({"trained_steps": int(cur)}, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_step(self) -> bool:
        cur = int(self.num_timesteps)
        if cur > 0 and (cur % self.save_freq) == 0:
            self._write(cur)
        return True

    def _on_training_end(self) -> None:
        self._write(int(self.num_timesteps))


def _read_resume_state(run_dir: Path) -> int:
    p = Path(run_dir) / "resume_state.json"
    if not p.exists():
        return 0
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        return int(j.get("trained_steps", 0))
    except Exception:
        return 0


def _pick_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob("sac_*_steps.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _parse_steps_from_ckpt_name(p: Path) -> int:
    stem = p.stem
    try:
        parts = stem.split("_")
        return int(parts[1])
    except Exception:
        return 0


# -----------------------------
# Sanity (GO/NO-GO)
# -----------------------------
def _run_policy_rollout(
    env: gym.Env,
    policy_name: str,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    max_steps: int,
    reset_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    obs, info = env.reset(options=reset_options)
    rows: list[dict[str, Any]] = []
    done = False
    steps = 0
    eq_recomp_net = 1.0
    eq_recomp_gross = 1.0

    while not done and steps < max_steps:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)

        pnl_ln = info.get("pnl_log_net", None)
        if pnl_ln is not None:
            try:
                eq_recomp_net *= float(np.exp(float(pnl_ln)))
            except Exception:
                pass

        pnl_lg = info.get("pnl_log_gross", None)
        if pnl_lg is not None:
            try:
                eq_recomp_gross *= float(np.exp(float(pnl_lg)))
            except Exception:
                pass

        row = {
            "policy": policy_name,
            "step": steps,
            "timestamp": info.get("timestamp", info.get("t_index", None)),
            "action": float(np.asarray(action).reshape(-1)[0]),
            "reward": float(np.asarray(reward).reshape(-1)[0]) if np.ndim(reward) else float(reward),
            "equity": float(info.get("equity", np.nan)),
            "equity_recomp_net": float(eq_recomp_net),
            "equity_recomp_gross": float(eq_recomp_gross),
            "pos_prev": info.get("pos_prev", info.get("position", None)),
            "pos_target": info.get("pos_target", info.get("pos_new", None)),
            "turnover": info.get("turnover", None),
            "cost": info.get("cost", None),
            "funding_rate_used": info.get("funding_rate_used", None),
            "funding_cost": info.get("funding_cost", None),
            "pnl_log_gross": info.get("pnl_log_gross", None),
            "pnl_log_net": info.get("pnl_log_net", None),
            "ret_fwd_used": info.get("ret_fwd_used", None),
            "reward_raw": info.get("reward_raw", None),
            "reward_final": info.get("reward_final", None),
        }
        rows.append(row)
        steps += 1

    return rows


def _sanity_go_no_go(
    run_dir: Path,
    df_train,
    feature_cols: list[str],
    sanity_steps: int,
    tol_log_final: float,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    sanity_dir = run_dir / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    n = int(min(len(df_train), sanity_steps + 512))
    if n < 128:
        raise ValueError(f"Not enough train rows for sanity: have {len(df_train)} need >=128")
    df_s = df_train.iloc[:n].copy()

    saved = {}
    for k in (
        "fee_bps",
        "spread_bps",
        "funding_mode",
        "reward_mode",
        "action_mode",
        "bench_mode",
        "include_bench_pos_in_obs",
    ):
        if hasattr(env_cfg, k):
            saved[k] = getattr(env_cfg, k)

    if hasattr(env_cfg, "fee_bps"):
        setattr(env_cfg, "fee_bps", 0.0)
    if hasattr(env_cfg, "spread_bps"):
        setattr(env_cfg, "spread_bps", 0.0)
    if hasattr(env_cfg, "funding_mode"):
        setattr(env_cfg, "funding_mode", "none")
    if hasattr(env_cfg, "reward_mode"):
        setattr(env_cfg, "reward_mode", "pnl")

    # Sanity is a low-level accounting/mechanics test.
    # Force the legacy direct-action / no-benchmark formulation so that
    # always_long / always_short identities remain meaningful.
    if hasattr(env_cfg, "action_mode"):
        setattr(env_cfg, "action_mode", "direct")
    if hasattr(env_cfg, "bench_mode"):
        setattr(env_cfg, "bench_mode", "none")
    if hasattr(env_cfg, "include_bench_pos_in_obs"):
        setattr(env_cfg, "include_bench_pos_in_obs", False)

    try:
        env = make_raw_env_for_sanity(df_s, feature_cols, max_steps=int(sanity_steps))
        rng = np.random.default_rng(123)

        def pol_flat(_obs: np.ndarray) -> np.ndarray:
            return np.asarray([0.0], dtype=np.float32)

        def pol_long(_obs: np.ndarray) -> np.ndarray:
            return np.asarray([1.0], dtype=np.float32)

        def pol_short(_obs: np.ndarray) -> np.ndarray:
            return np.asarray([-1.0], dtype=np.float32)

        def pol_rand(_obs: np.ndarray) -> np.ndarray:
            return np.asarray([float(rng.uniform(-1.0, 1.0))], dtype=np.float32)

        L = float(getattr(env_cfg, "max_leverage", 1.0))
        rows_flat = _run_policy_rollout(
            env,
            "always_flat",
            pol_flat,
            max_steps=int(sanity_steps),
            reset_options={"initial_position": 0.0},
        )
        rows_long = _run_policy_rollout(
            env,
            "always_long",
            pol_long,
            max_steps=int(sanity_steps),
            reset_options={"initial_position": +L},
        )
        rows_short = _run_policy_rollout(
            env,
            "always_short",
            pol_short,
            max_steps=int(sanity_steps),
            reset_options={"initial_position": -L},
        )
        rows_rand = _run_policy_rollout(
            env,
            "random",
            pol_rand,
            max_steps=int(sanity_steps),
            reset_options={"initial_position": 0.0},
        )

        eq_env = np.asarray([r["equity"] for r in rows_long], dtype=np.float64)
        eq_recomp = np.asarray([r.get("equity_recomp_net", np.nan) for r in rows_long], dtype=np.float64)

        mask = np.isfinite(eq_env) & np.isfinite(eq_recomp) & (eq_env > 0) & (eq_recomp > 0)
        idx = np.where(mask)[0]
        idx = idx[idx >= 1]

        pass_long_recomp = False
        final_log_diff = None
        max_abs_log_diff = None
        if idx.size >= 10:
            log_diff = np.log(eq_env[idx] / eq_recomp[idx])
            final_log_diff = float(log_diff[-1])
            max_abs_log_diff = float(np.max(np.abs(log_diff)))
            pass_long_recomp = bool(abs(final_log_diff) <= float(tol_log_final))

        eq_short_env = np.asarray([r["equity"] for r in rows_short], dtype=np.float64)
        eq_short_recomp = np.asarray([r.get("equity_recomp_net", np.nan) for r in rows_short], dtype=np.float64)

        mask_ls = np.isfinite(eq_env) & np.isfinite(eq_short_env) & (eq_env > 0) & (eq_short_env > 0)
        idx_ls = np.where(mask_ls)[0]
        idx_ls = idx_ls[idx_ls >= 1]

        pass_long_short_identity = False
        max_abs_log_prod = None
        if idx_ls.size >= 10:
            log_prod = np.log(eq_env[idx_ls] * eq_short_env[idx_ls])
            max_abs_log_prod = float(np.max(np.abs(log_prod)))
            pass_long_short_identity = bool(max_abs_log_prod <= float(tol_log_final))

        mask_sh = np.isfinite(eq_short_env) & np.isfinite(eq_short_recomp) & (eq_short_env > 0) & (eq_short_recomp > 0)
        idx_sh = np.where(mask_sh)[0]
        idx_sh = idx_sh[idx_sh >= 1]

        pass_short_recomp = False
        short_final_log_diff = None
        short_max_abs_log_diff = None
        if idx_sh.size >= 10:
            sh_log_diff = np.log(eq_short_env[idx_sh] / eq_short_recomp[idx_sh])
            short_final_log_diff = float(sh_log_diff[-1])
            short_max_abs_log_diff = float(np.max(np.abs(sh_log_diff)))
            pass_short_recomp = bool(abs(short_final_log_diff) <= float(tol_log_final))

        trace_csv = sanity_dir / "sanity_trace_always_long.csv"
        cols = list(rows_long[0].keys()) if rows_long else []
        with trace_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in rows_long:
                f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

        trace_csv_short = sanity_dir / "sanity_trace_always_short.csv"
        cols_s = list(rows_short[0].keys()) if rows_short else []
        with trace_csv_short.open("w", encoding="utf-8") as f:
            f.write(",".join(cols_s) + "\n")
            for r in rows_short:
                f.write(",".join("" if r[c] is None else str(r[c]) for c in cols_s) + "\n")

        report = {
            "sanity_steps": int(sanity_steps),
            "tol_log_final": float(tol_log_final),
            "sanity_env_overrides": {
                "action_mode": "direct",
                "bench_mode": "none",
                "include_bench_pos_in_obs": False,
                "fee_bps": 0.0,
                "spread_bps": 0.0,
                "funding_mode": "none",
                "reward_mode": "pnl",
            },
            "always_long_vs_recomp": {
                "pass": bool(pass_long_recomp),
                "final_log_diff": final_log_diff,
                "max_abs_log_diff": max_abs_log_diff,
                "note": "Comparison ignores first step (pos_prev convention). Uses equity recomposed from exp(cumsum(pnl_log_net)).",
            },
            "always_short_vs_recomp": {
                "pass": bool(pass_short_recomp),
                "final_log_diff": short_final_log_diff,
                "max_abs_log_diff": short_max_abs_log_diff,
                "note": "Comparison ignores first step (pos_prev convention). Uses equity recomposed from exp(cumsum(pnl_log_net)).",
            },
            "long_short_identity": {
                "pass": bool(pass_long_short_identity),
                "max_abs_log(eq_long*eq_short)": max_abs_log_prod,
                "note": "In frictionless log-return world (no costs/funding), eq_long*eq_short≈1.",
            },
            "policies_run": {
                "always_flat_steps": int(len(rows_flat)),
                "always_long_steps": int(len(rows_long)),
                "always_short_steps": int(len(rows_short)),
                "random_steps": int(len(rows_rand)),
            },
            "artifacts": {
                "trace_always_long_csv": str(trace_csv),
                "trace_always_short_csv": str(trace_csv_short),
            },
            "go_no_go": bool(pass_long_recomp and pass_short_recomp and pass_long_short_identity),
        }

        (sanity_dir / "sanity_report.json").write_text(json.dumps(json_sanitize(report), indent=2), encoding="utf-8")
        return report
    finally:
        for k, v in saved.items():
            try:
                setattr(env_cfg, k, v)
            except Exception:
                pass


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="", help="Override dataset path (parquet/csv). Default: config.")
    ap.add_argument("--timesteps", type=int, default=int(sac_cfg.total_timesteps))
    ap.add_argument("--eval_freq", type=int, default=25_000)
    ap.add_argument("--n_eval_episodes", type=int, default=3)
    ap.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--chunk", type=int, default=10_000, help="Progress chunk size (learn steps per loop)")
    ap.add_argument("--checkpoint_freq", type=int, default=50_000, help="Save checkpoint every N steps")
    ap.add_argument("--seed", type=int, default=int(sac_cfg.seed), help="Override seed (default from config)")
    ap.add_argument("--split_json", type=str, default="", help="Optional JSON file with explicit train/val/test indices")

    ap.add_argument(
        "--include_drawdown_in_obs",
        action="store_true",
        help="Include drawdown state in observation (training/eval runs only; tests keep default).",
    )
    ap.add_argument(
        "--include_bench_pos_in_obs",
        action="store_true",
        help="Include benchmark target position in observation (training/eval runs only; tests keep default).",
    )

    ap.add_argument("--run_sanity", action="store_true", help="Run sanity GO/NO-GO before training")
    ap.add_argument("--sanity_only", action="store_true", help="Run sanity and exit (no training)")
    ap.add_argument("--sanity_steps", type=int, default=2000, help="Max steps for sanity rollouts")
    ap.add_argument("--sanity_tol_log_final", type=float, default=1e-4, help="Tol on final log(eq_long/eq_bh)")

    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in a run_dir")
    ap.add_argument("--run_dir", type=str, default="", help="Run directory to resume from (required with --resume)")

    args = ap.parse_args()

    if bool(getattr(args, "include_drawdown_in_obs", False)):
        env_cfg.include_drawdown_in_obs = True
    if bool(getattr(args, "include_bench_pos_in_obs", False)):
        env_cfg.include_bench_pos_in_obs = True

    seed = int(args.seed)
    set_global_seeds(seed)

    data_path_clean = args.data_path.strip() if args.data_path.strip() else None

    try:
        if data_path_clean:
            df_final, feature_cols, data_source_meta = load_training_dataframe(data_path_arg=data_path_clean)
        else:
            df_final, feature_cols, data_source_meta = load_training_dataframe()
    except TypeError:
        try:
            if data_path_clean:
                df_final, feature_cols, data_source_meta = load_training_dataframe(data_path_clean)
            else:
                df_final, feature_cols, data_source_meta = load_training_dataframe()
        except TypeError:
            if data_path_clean:
                raise TypeError(
                    "Current load_training_dataframe() implementation does not accept a data_path override. "
                    "Either remove --data_path usage or update hc_ia_retail.data.load_training_dataframe()."
                )
            df_final, feature_cols, data_source_meta = load_training_dataframe()

    data_path_used = str(
        data_source_meta.get("data_path_used", data_source_meta.get("data_path", "")) or ""
    )
    dataset_sha = data_source_meta.get("dataset_sha256", data_source_meta.get("sha256"))

    Path(paths_cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(paths_cfg.models_dir).mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not args.run_dir.strip():
            raise ValueError("--resume requires --run_dir")
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        run_id = run_dir.name.replace("run_", "") if run_dir.name.startswith("run_") else run_dir.name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(paths_cfg.out_dir) / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_run_name = f"train_{run_id}"

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    vec_path = run_dir / "vecnormalize.pkl"

    git_state = _best_effort_git_state(PROJECT_ROOT)
    pip_freeze_txt = _best_effort_pip_freeze()
    pip_freeze_path = None
    if pip_freeze_txt:
        pip_freeze_path = str(run_dir / "pip_freeze.txt")
        (run_dir / "pip_freeze.txt").write_text(pip_freeze_txt.strip() + "\n", encoding="utf-8")

    def _apply_explicit_split(df_in, split_json_path: str):
        p = Path(split_json_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"split_json not found: {p}")
        spec = json.loads(p.read_text(encoding="utf-8"))

        def _sl(name: str):
            a = int(spec[name]["start"])
            b = int(spec[name]["end"])
            if not (0 <= a < b <= len(df_in)):
                raise ValueError(f"Invalid split bounds for {name}: {a}:{b} with n={len(df_in)}")
            return df_in.iloc[a:b].copy()

        return _sl("train"), _sl("val"), _sl("test")

    if args.split_json.strip():
        df_train, df_val, df_test = _apply_explicit_split(df_final, args.split_json.strip())
    else:
        df_train, df_val, df_test = split_train_val_test(df_final)

    data_report = _build_data_quality_report(
        df_raw=df_final,
        df_feat=df_final,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        feature_cols=list(feature_cols),
        data_path_used=data_path_used,
        dataset_sha=dataset_sha,
        data_source_meta=data_source_meta,
    )
    data_report_path = run_dir / "data_report.json"
    try:
        data_report_path.write_text(json.dumps(json_sanitize(data_report), indent=2), encoding="utf-8")
    except Exception as e:
        try:
            (run_dir / "data_report_error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
        except Exception:
            pass

    _tmp_env = RetailTradingEnv(df_train, feature_cols=feature_cols, max_steps=2)
    try:
        per_step_dim = int(_tmp_env.observation_space.shape[0])
    finally:
        try:
            _tmp_env.close()
        except Exception:
            pass

    window_size = int(data_cfg.window_size)
    window_dim = int(per_step_dim * window_size)

    total = int(args.timesteps)
    requested_ckpt = int(args.checkpoint_freq)
    if requested_ckpt <= 0:
        requested_ckpt = max(1000, total // 10)
    checkpoint_freq = int(min(requested_ckpt, max(1, total)))
    if checkpoint_freq == total and total >= 2:
        checkpoint_freq = int(max(1, total // 2))

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] dataset={data_path_used}")
    print(f"[INFO] dataset_sha256={dataset_sha}")
    print(f"[INFO] regime_enabled={bool(getattr(regime_cfg, 'enabled', False))}")
    print(f"[INFO] regime_feature_mode={getattr(regime_cfg, 'regime_feature_mode', 'market_plus_regime')}")
    print(f"[INFO] rows: train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")
    print(
        f"[INFO] features={len(feature_cols)} | per_step_dim={per_step_dim} | window={window_size} | window_dim={window_dim}"
    )
    print(
        f"[INFO] obs_overrides: include_drawdown_in_obs={bool(getattr(env_cfg, 'include_drawdown_in_obs', False))} | "
        f"include_bench_pos_in_obs={bool(getattr(env_cfg, 'include_bench_pos_in_obs', False))}"
    )
    print(f"[INFO] device={args.device}")
    print(f"[INFO] tb_dir={tb_dir}")
    print(f"[INFO] checkpoint_freq(requested={requested_ckpt}) -> using {checkpoint_freq}")

    sanity_report = None
    if args.run_sanity or args.sanity_only:
        print("[SANITY] Running GO/NO-GO sanity checks...")
        sanity_report = _sanity_go_no_go(
            run_dir=run_dir,
            df_train=df_train,
            feature_cols=list(feature_cols),
            sanity_steps=int(args.sanity_steps),
            tol_log_final=float(args.sanity_tol_log_final),
        )
        print(f"[SANITY] go_no_go={sanity_report.get('go_no_go')} (see {run_dir / 'sanity' / 'sanity_report.json'})")
        if args.sanity_only:
            print(f"[OK] sanity_only -> exiting. run_dir={run_dir}")
            return
        if not sanity_report.get("go_no_go", False):
            raise RuntimeError(
                "SANITY NO-GO: always-long did not match Buy&Hold within tolerance. "
                "Fix timing/cost/funding before training."
            )

    env_train = DummyVecEnv([make_env(df_train, feature_cols, getattr(env_cfg, "max_steps_train", None))])

    if args.resume and vec_path.exists():
        env_train = VecNormalize.load(str(vec_path), env_train)
        env_train.training = True
        env_train.norm_reward = False
    else:
        env_train = VecNormalize(env_train, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env_train.seed(seed)

    env_val = DummyVecEnv([make_env(df_val, feature_cols, max_steps=None)])
    if args.resume and vec_path.exists():
        env_val = VecNormalize.load(str(vec_path), env_val)
        env_val.training = False
        env_val.norm_reward = False
    else:
        env_val = VecNormalize(env_val, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
        env_val.obs_rms = env_train.obs_rms
        env_val.training = False
        env_val.norm_reward = False
    env_val.seed(seed + 1)

    best_score_val = float("nan")
    funding_mode_str = str(getattr(env_cfg, "funding_mode", "")) or ""

    policy_kwargs = dict(
        features_extractor_class=GRUWindowExtractor,
        features_extractor_kwargs=dict(
            window_size=window_size,
            per_step_dim=per_step_dim,
            gru_hidden=int(model_cfg.gru_hidden),
            gru_layers=int(model_cfg.gru_layers),
            out_dim=int(model_cfg.mlp_latent),
        ),
        net_arch=dict(pi=[128, 128], qf=[128, 128]),
    )

    resume_ckpt = _pick_latest_checkpoint(ckpt_dir) if args.resume else None
    resume_trained_steps = _read_resume_state(run_dir) if args.resume else 0

    if args.resume and resume_ckpt is not None:
        step_from_name = _parse_steps_from_ckpt_name(resume_ckpt)
        print(f"[RESUME] loading checkpoint: {resume_ckpt} (steps≈{step_from_name})")
        model = SAC.load(str(resume_ckpt), env=env_train, device=str(args.device))
        trained = int(max(step_from_name, resume_trained_steps))
        try:
            model.num_timesteps = trained
        except Exception:
            pass
    else:
        model = SAC(
            "MlpPolicy",
            env_train,
            learning_rate=sac_cfg.learning_rate,
            buffer_size=sac_cfg.buffer_size,
            batch_size=sac_cfg.batch_size,
            gamma=sac_cfg.gamma,
            tau=sac_cfg.tau,
            train_freq=sac_cfg.train_freq,
            gradient_steps=sac_cfg.gradient_steps,
            ent_coef=sac_cfg.ent_coef,
            target_entropy=sac_cfg.target_entropy,
            use_sde=True,
            sde_sample_freq=1,
            seed=seed,
            verbose=0,
            policy_kwargs=policy_kwargs,
            device=str(args.device),
            tensorboard_log=str(tb_dir),
        )
        trained = 0

    print(f"[INFO] starting trained_steps={trained} / target={total}")

    freq = float(getattr(data_cfg, "freq_per_year", 24 * 365))

    val_best_cb = ValNorthStarBestCallback(
        eval_env=env_val,
        vec_train=env_train,
        vec_path=vec_path,
        best_dir=best_dir,
        eval_freq_steps=int(args.eval_freq),
        freq_per_year=freq,
        north_star_params={"d": 0.20, "tau": 0.01, "lambda_dd": 2.0, "lambda_to": 1.0},
        deterministic=True,
        max_rollout_steps=None,
        eps_action_std=float(getattr(env_cfg, "val_eps_action_std", 0.01)),
        eps_turnover=float(getattr(env_cfg, "val_eps_turnover", 1e-6)),
        use_val_windows=bool(getattr(env_cfg, "val_use_windows", True)),
        window_len=int(getattr(env_cfg, "val_window_len", 512)),
        window_stride=int(getattr(env_cfg, "val_window_stride", 256)),
        n_windows=int(getattr(env_cfg, "val_n_windows", 3)),
        val_windows=None,
    )

    dsr_clip = getattr(env_cfg, "dsr_clip", None)
    stability_cb = TrainingStabilityCallback(
        log_every_steps=int(getattr(sac_cfg, "log_every_steps", 2048)) if hasattr(sac_cfg, "log_every_steps") else 2048,
        run_dir=run_dir,
        dsr_clip=float(dsr_clip) if dsr_clip is not None else None,
    )

    save_vecnorm_cb = SaveVecNormalizeCallback(vec_env=env_train, save_path=vec_path, save_freq=checkpoint_freq)
    save_resume_cb = SaveResumeStateCallback(run_dir=run_dir, save_freq=max(1000, min(10_000, checkpoint_freq)))

    meta_path = run_dir / "meta.json"
    meta: dict[str, Any] = {
        "run_id": run_id,
        "seed": int(seed),
        "command_line": list(sys.argv),
        "project_root": str(PROJECT_ROOT),
        "dataset": {"path": data_path_used, "sha256": dataset_sha},
        "data_source_meta": data_source_meta,
        "git": {
            "commit": git_state.get("commit"),
            "dirty": git_state.get("dirty"),
            "available": git_state.get("available"),
            "error": git_state.get("error"),
        },
        "pip_freeze_path": pip_freeze_path,
        "feature_cols": list(feature_cols),
        "dims": {"per_step_dim": per_step_dim, "window_size": window_size, "window_dim": window_dim},
        "splits": {"train_rows": int(len(df_train)), "val_rows": int(len(df_val)), "test_rows": int(len(df_test))},
        "configs": {
            "data_cfg": asdict(data_cfg),
            "env_cfg": asdict(env_cfg),
            "sac_cfg": asdict(sac_cfg),
            "model_cfg": asdict(model_cfg),
            "paths_cfg": asdict(paths_cfg),
            "regime_cfg": asdict(regime_cfg),
        },
        "policy_kwargs": json_sanitize(policy_kwargs),
        "runtime": {
            "device": str(args.device),
            "timesteps": int(args.timesteps),
            "eval_freq": int(args.eval_freq),
            "n_eval_episodes_arg_kept": int(args.n_eval_episodes),
            "checkpoint_freq_requested": int(args.checkpoint_freq),
            "checkpoint_freq_used": int(checkpoint_freq),
            "chunk": int(args.chunk),
            "resume": bool(args.resume),
            "run_dir": str(run_dir),
        },
        "sanity": sanity_report,
        "artifacts": {
            "run_dir": str(run_dir),
            "data_report_path": str(run_dir / "data_report.json"),
            "tb_dir": str(tb_dir),
            "tb_run_name": tb_run_name,
            "checkpoints_dir": str(ckpt_dir),
            "best_dir": str(best_dir),
            "best_model_path": str(best_dir / "best_model.zip"),
            "best_metrics_val_path": str(best_dir / "metrics_val.json"),
            "vecnormalize_path": str(vec_path),
            "resume_state_path": str(run_dir / "resume_state.json"),
        },
    }
    meta_path.write_text(json.dumps(json_sanitize(meta), indent=2), encoding="utf-8")

    manifest_path = _write_train_manifest(
        run_dir=run_dir,
        run_id=run_id,
        args=args,
        seed=seed,
        data_path_used=data_path_used,
        dataset_sha=dataset_sha,
        feature_cols=list(feature_cols),
        per_step_dim=per_step_dim,
        window_size=window_size,
        window_dim=window_dim,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        policy_kwargs=policy_kwargs,
        checkpoint_freq_used=checkpoint_freq,
        sanity_report=sanity_report,
        git_state=git_state,
        pip_freeze_path=pip_freeze_path,
        pip_freeze_txt=pip_freeze_txt,
        data_source_meta=data_source_meta,
    )
    print(f"[OK] manifest_train -> {manifest_path}")

    chunk = int(args.chunk)
    if chunk <= 0:
        chunk = max(total // 100, 1000)

    pbar = tqdm(total=total, desc="Training SAC (steps)", unit="step", dynamic_ncols=True)
    if trained > 0:
        pbar.update(min(trained, total))

    def _save_checkpoint(step_now: int) -> None:
        ckpt_path = ckpt_dir / f"sac_{int(step_now)}_steps.zip"
        try:
            model.save(str(ckpt_path))
        except Exception:
            pass
        try:
            model.save_replay_buffer(str(ckpt_dir / f"replay_{int(step_now)}_steps.pkl"))
        except Exception:
            pass
        try:
            env_train.save(str(vec_path))
        except Exception:
            pass

    callbacks = [
        val_best_cb,
        stability_cb,
        save_vecnorm_cb,
        save_resume_cb,
        RunControlCallback(run_dir, check_freq=1000),
    ]

    if trained <= 0:
        next_ckpt = checkpoint_freq
    else:
        next_ckpt = ((trained // checkpoint_freq) + 1) * checkpoint_freq
    next_ckpt = int(min(next_ckpt, total))

    while trained < total:
        step = min(chunk, total - trained)

        model.learn(
            total_timesteps=step,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=tb_run_name,
            progress_bar=False,
        )

        trained += step
        pbar.update(step)

        while next_ckpt > 0 and trained >= next_ckpt:
            _save_checkpoint(next_ckpt)
            next_ckpt += checkpoint_freq
            if next_ckpt > total:
                break

    pbar.close()

    final_path = Path(paths_cfg.models_dir) / f"hc_retail_sac_gru_final_{run_id}"
    model.save(str(final_path))

    env_train.save(str(vec_path))
    _save_checkpoint(trained)

    meta["artifacts"]["model_final_path"] = str(final_path)
    meta["artifacts"]["vecnormalize_path"] = str(vec_path)
    meta["artifacts"]["last_checkpoint"] = str(_pick_latest_checkpoint(ckpt_dir) or "")
    meta["runtime"]["trained_steps_final"] = int(trained)
    meta_path.write_text(json.dumps(json_sanitize(meta), indent=2), encoding="utf-8")

    _patch_train_manifest_finalize(
        run_dir=run_dir,
        trained_steps_final=int(trained),
        total_timesteps=int(total),
        final_model_path=final_path,
        best_model_path=(best_dir / "best_model.zip"),
        best_metrics_path=(best_dir / "metrics_val.json"),
        last_checkpoint=str(_pick_latest_checkpoint(ckpt_dir) or ""),
    )

    try:
        best_score_val = float(getattr(val_best_cb, "best_score_val", float("nan")))
    except Exception:
        best_score_val = float("nan")

    try:
        runs_csv = run_dir / "runs" / "runs_index.csv"
        runs_csv.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "run_id": str(run_dir.name),
            "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "seed": safe_float(seed),
            "score_val": safe_float(best_score_val),
            "score_oos": float("nan"),
            "sharpe_oos": float("nan"),
            "maxdd_oos": float("nan"),
            "turnover_oos": float("nan"),
            "funding_mode": funding_mode_str,
        }
        append_run_index(runs_csv, row)
    except Exception:
        pass

    print(f"[OK] run_dir -> {run_dir}")
    print(f"[OK] best_model -> {best_dir / 'best_model.zip'}")
    print(f"[OK] best_metrics -> {best_dir / 'metrics_val.json'}")
    print(f"[OK] checkpoints -> {ckpt_dir}")
    print(f"[OK] vecnorm -> {vec_path}")
    print(f"[OK] final_model -> {final_path}")
    print(f"[OK] tensorboard -> tensorboard --logdir {tb_dir} --port 6006")
    print(
        f"[HINT] If TensorBoard shows 'No dashboards', double-check RUN_DIR and use an absolute path: "
        f"tensorboard --logdir '{tb_dir.resolve()}' --port 6006"
    )
    print(
        f"[OK] resume with: python scripts/train/train.py --resume --run_dir {run_dir} "
        f"--timesteps {total} --device {args.device}"
    )


if __name__ == "__main__":
    main()