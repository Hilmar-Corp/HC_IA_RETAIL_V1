# hc_ia_retail/run_registry.py
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict


def safe_float(x: Any, default: float = float("nan")) -> float:
    """Best-effort float conversion (robust to None/np/pd types/strings)."""
    if x is None:
        return default
    try:
        # fast path
        f = float(x)
        if math.isfinite(f):
            return f
        return default
    except Exception:
        pass
    try:
        s = str(x).strip()
        if not s:
            return default
        # handle common string nulls
        if s.lower() in {"nan", "none", "null", "inf", "+inf", "-inf"}:
            return default
        f = float(s)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def append_run_index(csv_path: str | Path, row_dict: Dict[str, Any]) -> Path:
    """
    Append a row to a CSV run registry. Creates file + parent dirs if missing.
    Ensures stable column order: existing header if file exists, else from row_dict keys.
    Missing fields are written as empty strings.
    Extra fields not in header are ignored (to keep schema stable).
    """
    p = Path(csv_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    # Determine header
    header: list[str]
    file_exists = p.exists() and p.stat().st_size > 0
    if file_exists:
        try:
            with p.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                header = [h.strip() for h in header if str(h).strip()]
        except Exception:
            header = []
    else:
        header = []

    if not header:
        header = list(row_dict.keys())

    # Build row aligned to header
    row_out: dict[str, str] = {}
    for k in header:
        v = row_dict.get(k, "")
        if v is None:
            row_out[k] = ""
        elif isinstance(v, float):
            # keep human readable, avoid "nan"
            row_out[k] = "" if not math.isfinite(v) else f"{v:.10g}"
        else:
            s = str(v)
            row_out[k] = "" if s.lower() == "nan" else s

    # Write
    write_header = not file_exists
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row_out)

    return p