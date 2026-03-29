from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def generate_data_report(
    *,
    df_raw: pd.DataFrame,
    df_feat: pd.DataFrame,
    n_rows_train: int,
    n_rows_val: int,
    n_rows_test: int,
    data_path: str,
    instrument: str,
    interval: str,
) -> Dict[str, Any]:
    """
    Dependency-light generator for data_report.json (matches tests schema).
    Returns JSON-serializable primitives only.
    """
    n_rows_raw = int(len(df_raw))
    n_rows_feat = int(len(df_feat))

    # time stats / gaps on raw timestamps (tests provide "timestamp")
    time_min: Optional[str] = None
    time_max: Optional[str] = None
    duplicates_count = 0
    gap_count = 0
    max_gap_hours = 0.0

    if "timestamp" in df_raw.columns:
        ts = pd.to_datetime(df_raw["timestamp"], utc=True, errors="coerce")
        if ts.notna().any():
            time_min = ts.min().isoformat()
            time_max = ts.max().isoformat()

        duplicates_count = int(ts.duplicated().sum())

        ts_sorted = ts.sort_values()
        dt_hours = ts_sorted.diff().dt.total_seconds() / 3600.0
        if dt_hours.notna().any():
            gaps = dt_hours[dt_hours > 1.0000001]
            gap_count = int(gaps.shape[0])
            max_gap_hours = float(gaps.max()) if gaps.shape[0] else 0.0

    # nonfinite on numeric columns of df_feat
    nonfinite_count = 0
    try:
        arr = df_feat.select_dtypes(include=["number"]).to_numpy(dtype=float)
        if arr.size:
            nonfinite_count = int(arr.size - np.isfinite(arr).sum())
    except Exception:
        nonfinite_count = 0

    # best-effort "dropped by dropna" proxy
    pct_dropped_by_dropna = 0.0
    if n_rows_raw > 0:
        pct_dropped_by_dropna = float(max(0.0, 1.0 - (n_rows_feat / float(n_rows_raw))))

    # return quantiles if a return-like col exists (tests accept None or dict)
    ret_quantiles = None
    ret_col = None
    for cand in ("log_ret_1_fwd", "ret_fwd_used", "ret_fwd", "log_ret_1"):
        if cand in df_feat.columns:
            ret_col = cand
            break

    if ret_col is not None:
        try:
            r = pd.to_numeric(df_feat[ret_col], errors="coerce").to_numpy(dtype=float)
            r = r[np.isfinite(r)]
            if r.size:
                qs = [0.01, 0.05, 0.50, 0.95, 0.99]
                vals = np.quantile(r, qs)
                ret_quantiles = {f"p{int(q*100):02d}": float(v) for q, v in zip(qs, vals)}
        except Exception:
            ret_quantiles = None

    # sha256 of a stable representation (doesn't need the real dataset file)
    try:
        payload = df_feat.to_json(orient="records", date_format="iso")
        sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        sha256 = hashlib.sha256(f"{n_rows_feat}".encode("utf-8")).hexdigest()

    return {
        "instrument": str(instrument),
        "interval": str(interval),
        "data_path": str(data_path),
        "n_rows_raw": int(n_rows_raw),
        "n_rows_feat": int(n_rows_feat),
        "n_rows_train": int(n_rows_train),
        "n_rows_val": int(n_rows_val),
        "n_rows_test": int(n_rows_test),
        "time_min": time_min,
        "time_max": time_max,
        "duplicates_count": int(duplicates_count),
        "gap_count": int(gap_count),
        "max_gap_hours": float(max_gap_hours),
        "nonfinite_count": int(nonfinite_count),
        "pct_dropped_by_dropna": float(pct_dropped_by_dropna),
        "ret_quantiles": ret_quantiles,
        "sha256": str(sha256),
    }


def write_data_report(
    run_dir,
    df_raw: pd.DataFrame,
    df_feat: pd.DataFrame,
    n_rows_train: int,
    n_rows_val: int,
    n_rows_test: int,
    data_path: str,
    instrument: str,
    interval: str,
    **_ignored,
) -> Dict[str, Any]:
    """
    Writer expected by tests. Writes run_dir/data_report.json and returns the dict.
    Accepts extra kwargs for compatibility.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rep = generate_data_report(
        df_raw=df_raw,
        df_feat=df_feat,
        n_rows_train=n_rows_train,
        n_rows_val=n_rows_val,
        n_rows_test=n_rows_test,
        data_path=data_path,
        instrument=instrument,
        interval=interval,
    )
    (run_dir / "data_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep