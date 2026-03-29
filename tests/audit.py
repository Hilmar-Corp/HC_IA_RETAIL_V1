# hc_ia_retail/audit.py
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd


def _iso(dt) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.isoformat()
    return str(dt)


def _df_fingerprint(df: pd.DataFrame) -> str:
    """
    Stable-ish fingerprint for small audit artifacts.
    Enough for tests + quick traceability.
    """
    try:
        payload = df.to_csv(index=True).encode("utf-8")
    except Exception:
        payload = repr(df).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_data_report(
    data: Union[pd.DataFrame, Dict[str, Any]],
    out_dir: Union[str, Path],
    *,
    filename: str = "data_report.json",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Writes an audit-friendly JSON report about the dataset.

    Flexible signature on purpose: tests may pass a DataFrame directly or wrap it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Accept either a DataFrame directly or a dict-like wrapper
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        # common patterns: {"df": df} or {"data": df}
        cand = data.get("df", None) or data.get("data", None)
        if isinstance(cand, pd.DataFrame):
            df = cand
        else:
            raise TypeError("write_data_report expected a DataFrame or a dict containing a DataFrame under 'df'/'data'.")
    else:
        raise TypeError("write_data_report expected a pandas.DataFrame (or a dict wrapper).")

    # Basic dataset stats
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    cols = [str(c) for c in df.columns.tolist()]
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    nulls = {str(c): int(df[c].isna().sum()) for c in df.columns}

    # Index info (nice if it's a DatetimeIndex)
    idx = df.index
    index_info: Dict[str, Any] = {"type": type(idx).__name__}
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
        index_info.update(
            {
                "tz": str(idx.tz) if idx.tz is not None else None,
                "start": _iso(idx.min()),
                "end": _iso(idx.max()),
                "is_monotonic_increasing": bool(idx.is_monotonic_increasing),
                "has_duplicates": bool(idx.has_duplicates),
            }
        )
        # Try infer freq (can be None)
        try:
            index_info["inferred_freq"] = idx.inferred_freq
        except Exception:
            index_info["inferred_freq"] = None

    report: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": cols,
        "dtypes": dtypes,
        "null_counts": nulls,
        "index": index_info,
        "fingerprint_sha256": _df_fingerprint(df),
    }

    if extra:
        report["extra"] = dict(extra)

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def generate_data_report(*args, **kwargs) -> Path:
    # Alias expected by some candidates/tests
    return write_data_report(*args, **kwargs)