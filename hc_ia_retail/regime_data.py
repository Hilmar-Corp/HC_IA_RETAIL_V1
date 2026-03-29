from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported table format: {p}")


@dataclass(frozen=True)
class RegimeDatasetHandle:
    rl_dataset_run_id: str
    run_dir: Path
    observation_pack_path: Path
    audit_pack_path: Optional[Path]
    feature_contract_path: Path
    manifest_path: Path
    quality_summary_path: Optional[Path]
    observation_pack_sha256: str
    manifest: dict
    feature_contract: dict
    quality_summary: Optional[dict]


def _is_rl_dataset_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    required = [
        p / "rl_observation_pack.parquet",
        p / "feature_contract.json",
        p / "manifest.json",
    ]
    return all(x.exists() for x in required)


def resolve_latest_rl_dataset_run(rl_dataset_root: str | Path) -> Path:
    root = Path(rl_dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"rl_dataset_root not found: {root}")

    candidates = [p for p in root.iterdir() if _is_rl_dataset_dir(p)]
    if not candidates:
        raise FileNotFoundError(f"No RL dataset run found under: {root}")

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_rl_dataset_run(
    rl_dataset_root: str | Path,
    *,
    rl_dataset_run_id: Optional[str],
    use_latest: bool,
) -> Path:
    root = Path(rl_dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"rl_dataset_root not found: {root}")

    if rl_dataset_run_id:
        run_dir = root / rl_dataset_run_id
        if not _is_rl_dataset_dir(run_dir):
            raise FileNotFoundError(
                f"Requested rl_dataset_run_id={rl_dataset_run_id} not found or incomplete under {root}"
            )
        return run_dir

    if use_latest:
        return resolve_latest_rl_dataset_run(root)

    raise ValueError(
        "No RL dataset selected. Provide rl_dataset_run_id or set use_latest_rl_dataset=True."
    )


def load_rl_regime_contract(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    return read_json(run_dir / "feature_contract.json")


def load_rl_regime_manifest(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    return read_json(run_dir / "manifest.json")


def load_rl_regime_quality_summary(run_dir: str | Path) -> Optional[dict]:
    run_dir = Path(run_dir).expanduser().resolve()
    p = run_dir / "quality_summary.json"
    return read_json(p) if p.exists() else None


def load_rl_regime_dataset_handle(
    rl_dataset_root: str | Path,
    *,
    rl_dataset_run_id: Optional[str] = None,
    use_latest: bool = True,
) -> RegimeDatasetHandle:
    run_dir = resolve_rl_dataset_run(
        rl_dataset_root,
        rl_dataset_run_id=rl_dataset_run_id,
        use_latest=use_latest,
    )

    observation_pack_path = run_dir / "rl_observation_pack.parquet"
    audit_pack_path = run_dir / "rl_audit_pack.parquet"
    feature_contract_path = run_dir / "feature_contract.json"
    manifest_path = run_dir / "manifest.json"
    quality_summary_path = run_dir / "quality_summary.json"

    manifest = read_json(manifest_path)
    feature_contract = read_json(feature_contract_path)
    quality_summary = read_json(quality_summary_path) if quality_summary_path.exists() else None

    return RegimeDatasetHandle(
        rl_dataset_run_id=run_dir.name,
        run_dir=run_dir,
        observation_pack_path=observation_pack_path,
        audit_pack_path=audit_pack_path if audit_pack_path.exists() else None,
        feature_contract_path=feature_contract_path,
        manifest_path=manifest_path,
        quality_summary_path=quality_summary_path if quality_summary_path.exists() else None,
        observation_pack_sha256=sha256_file(observation_pack_path),
        manifest=manifest,
        feature_contract=feature_contract,
        quality_summary=quality_summary,
    )


def _normalize_timestamp_col(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_col}")
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    if out[timestamp_col].isna().any():
        n_bad = int(out[timestamp_col].isna().sum())
        raise ValueError(f"Timestamp parse failure in {timestamp_col}: n_bad={n_bad}")
    out = out.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    if out[timestamp_col].duplicated().any():
        dupes = int(out[timestamp_col].duplicated().sum())
        raise ValueError(f"Duplicate timestamps detected in {timestamp_col}: n_duplicates={dupes}")
    return out


def validate_regime_pack_basic(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    segment_col: str = "segment",
    expected_freq: str = "4h",
) -> dict:
    d = _normalize_timestamp_col(df, timestamp_col=timestamp_col)

    if segment_col not in d.columns:
        raise KeyError(f"Missing segment column: {segment_col}")

    ts = d[timestamp_col]
    diffs = ts.diff().dropna()

    expected_delta = pd.Timedelta(expected_freq)
    n_gap_breaks = 0
    min_gap_seconds = 0.0
    max_gap_seconds = 0.0
    median_gap_seconds = 0.0

    if len(diffs) > 0:
        diff_seconds = diffs.dt.total_seconds().to_numpy(dtype=float)
        exp_seconds = expected_delta.total_seconds()
        gap_mask = ~np.isclose(diff_seconds, exp_seconds)
        n_gap_breaks = int(gap_mask.sum())
        min_gap_seconds = float(np.min(diff_seconds))
        max_gap_seconds = float(np.max(diff_seconds))
        median_gap_seconds = float(np.median(diff_seconds))

    return {
        "n_rows": int(len(d)),
        "n_unique_timestamps": int(d[timestamp_col].nunique()),
        "duplicate_timestamp_count": int(d[timestamp_col].duplicated().sum()),
        "min_timestamp": d[timestamp_col].min().isoformat(),
        "max_timestamp": d[timestamp_col].max().isoformat(),
        "expected_freq": expected_freq,
        "n_gap_breaks": n_gap_breaks,
        "min_gap_seconds": min_gap_seconds,
        "median_gap_seconds": median_gap_seconds,
        "max_gap_seconds": max_gap_seconds,
        "segments": sorted(d[segment_col].astype(str).unique().tolist()),
    }


def load_rl_regime_observation_pack(
    rl_dataset_root: str | Path,
    *,
    rl_dataset_run_id: Optional[str] = None,
    use_latest: bool = True,
    timestamp_col: str = "timestamp",
    segment_col: str = "segment",
    expected_freq: str = "4h",
) -> tuple[pd.DataFrame, RegimeDatasetHandle, dict]:
    handle = load_rl_regime_dataset_handle(
        rl_dataset_root=rl_dataset_root,
        rl_dataset_run_id=rl_dataset_run_id,
        use_latest=use_latest,
    )
    df = read_table(handle.observation_pack_path)
    df = _normalize_timestamp_col(df, timestamp_col=timestamp_col)
    validation = validate_regime_pack_basic(
        df,
        timestamp_col=timestamp_col,
        segment_col=segment_col,
        expected_freq=expected_freq,
    )
    return df, handle, validation


def _extract_policy_columns(feature_contract: dict, key: str) -> List[str]:
    cols = feature_contract.get(key, [])
    if cols is None:
        return []
    if not isinstance(cols, list):
        raise ValueError(f"feature_contract[{key}] must be a list")
    return [str(x) for x in cols]


def get_regime_feature_columns(
    feature_contract: dict,
    *,
    include_timestamp: bool = False,
    include_segment: bool = False,
    include_z_hat_filter: bool = False,
) -> List[str]:
    agent_cols = _extract_policy_columns(feature_contract, "agent_observable_columns_v1")

    if not include_z_hat_filter:
        agent_cols = [c for c in agent_cols if c != "z_hat_filter"]

    ordered = []
    if include_timestamp:
        ordered.append("timestamp")
    if include_segment:
        ordered.append("segment")
    ordered.extend(agent_cols)
    return ordered


def get_regime_provenance_columns(feature_contract: dict) -> List[str]:
    return _extract_policy_columns(feature_contract, "provenance_only_columns_v1")


def get_regime_audit_only_columns(feature_contract: dict) -> List[str]:
    return _extract_policy_columns(feature_contract, "audit_only_columns_v1")


def get_regime_forbidden_columns(feature_contract: dict) -> List[str]:
    return _extract_policy_columns(feature_contract, "forbidden_columns_v1")


def build_regime_metadata(handle: RegimeDatasetHandle, validation: dict, regime_feature_columns: Sequence[str]) -> dict:
    manifest = handle.manifest
    upstream_full_history_run_id = (
        manifest.get("inputs", {}).get("upstream_full_history_run_id")
        or manifest.get("upstream_full_history_run_id")
    )

    return {
        "rl_dataset_run_id": handle.rl_dataset_run_id,
        "rl_dataset_run_dir": str(handle.run_dir),
        "rl_observation_pack_path": str(handle.observation_pack_path),
        "rl_observation_pack_sha256": handle.observation_pack_sha256,
        "feature_contract_path": str(handle.feature_contract_path),
        "manifest_path": str(handle.manifest_path),
        "quality_summary_path": str(handle.quality_summary_path) if handle.quality_summary_path else None,
        "upstream_full_history_run_id": upstream_full_history_run_id,
        "regime_feature_columns": list(regime_feature_columns),
        "n_regime_feature_columns": int(len(regime_feature_columns)),
        "validation": validation,
    }