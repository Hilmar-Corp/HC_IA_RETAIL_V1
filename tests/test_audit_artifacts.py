# tests/test_audit_artifacts.py
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import pytest


# ---------------------------
# Helpers
# ---------------------------
def _make_synth_df(n: int = 50, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    """
    Minimal OHLCV + timestamp (+ forward return + 1 dummy feature) for fast env tests.
    Hourly UTC timestamps.
    """
    n = int(n)
    assert n >= 10
    ts = pd.date_range(start=start, periods=n, freq="H", tz="UTC")

    rng = np.random.default_rng(123)
    rets = rng.normal(0.0, 0.001, size=n).astype(np.float64)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + 0.0005)
    low = np.minimum(open_, close) * (1.0 - 0.0005)
    vol = rng.uniform(1.0, 10.0, size=n)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": vol.astype(float),
        }
    )

    # forward log return for step convention (t -> t+1)
    fwd = np.zeros(n, dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    fwd[:-1] = np.log((c[1:] + 1e-12) / (c[:-1] + 1e-12))
    fwd[-1] = 0.0
    df["log_ret_1_fwd"] = fwd.astype(float)

    # dummy feature
    df["feat_dummy"] = pd.Series(rng.normal(0.0, 1.0, size=n), index=df.index).astype(float)
    return df


def _load_env_class():
    mod = importlib.import_module("hc_ia_retail.env")
    return getattr(mod, "RetailTradingEnv")


def _find_data_report_writer() -> Callable:
    """
    Tries to find the function you added in TÂCHE 1/3 that writes data_report.json.

    If your writer is in another module/name, add it to candidates.
    """
    candidates: list[Tuple[str, str]] = [
        ("hc_ia_retail.train", "write_data_report"),
        ("hc_ia_retail.train", "generate_data_report"),
        ("hc_ia_retail.audit", "write_data_report"),
        ("hc_ia_retail.audit", "generate_data_report"),
        ("hc_ia_retail.utils.audit", "write_data_report"),
        ("hc_ia_retail.utils.audit", "generate_data_report"),
        ("hc_ia_retail.reporting", "write_data_report"),
        ("hc_ia_retail.reporting", "generate_data_report"),
    ]

    for module_name, fn_name in candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return fn

    pytest.fail(
        "Impossible de trouver la fonction qui écrit data_report.json.\n"
        "Expose une fonction importable (ex: hc_ia_retail.audit.write_data_report)\n"
        "ou ajoute son chemin dans 'candidates' dans ce test."
    )


def _assert_min_report_keys(d: dict):
    required = {
        "instrument",
        "interval",
        "data_path",
        "n_rows_raw",
        "n_rows_feat",
        "n_rows_train",
        "n_rows_val",
        "n_rows_test",
        "time_min",
        "time_max",
        "duplicates_count",
        "gap_count",
        "max_gap_hours",
        "nonfinite_count",
        "pct_dropped_by_dropna",
        "ret_quantiles",
        "sha256",
    }
    missing = sorted([k for k in required if k not in d])
    assert not missing, f"Missing keys in data_report.json: {missing}"

    # light sanity checks
    assert isinstance(d["n_rows_raw"], (int, float))
    assert isinstance(d["duplicates_count"], (int, float))
    assert isinstance(d["gap_count"], (int, float))
    assert isinstance(d["nonfinite_count"], (int, float))
    assert d["ret_quantiles"] is None or isinstance(d["ret_quantiles"], dict)


def _is_json_primitive(x) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


# ---------------------------
# Tests
# ---------------------------
def test_data_report_json_created(tmp_path: Path):
    """
    No training. Just calls the data_report writer on a tiny synthetic dataset.
    """
    write_fn = _find_data_report_writer()

    df_raw = _make_synth_df(n=40)
    df_feat = df_raw.copy()

    # split sizes
    n = len(df_feat)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    called = False
    last_err: Optional[Exception] = None

    # Pattern A: keyword-style writer
    try:
        write_fn(
            run_dir=run_dir,
            df_raw=df_raw,
            df_feat=df_feat,
            n_rows_train=n_train,
            n_rows_val=n_val,
            n_rows_test=n_test,
            data_path=str(run_dir / "dummy.csv"),
            instrument="TEST",
            interval="1h",
        )
        called = True
    except TypeError as e:
        last_err = e

    # Pattern B: positional
    if not called:
        try:
            write_fn(
                run_dir,
                df_raw,
                df_feat,
                n_train,
                n_val,
                n_test,
                str(run_dir / "dummy.csv"),
                "TEST",
                "1h",
            )
            called = True
        except TypeError as e:
            last_err = e

    # Pattern C: generator returns dict
    if not called:
        try:
            rep = write_fn(
                df_raw=df_raw,
                df_feat=df_feat,
                n_rows_train=n_train,
                n_rows_val=n_val,
                n_rows_test=n_test,
                data_path=str(run_dir / "dummy.csv"),
                instrument="TEST",
                interval="1h",
            )
            assert isinstance(rep, dict)
            (run_dir / "data_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
            called = True
        except TypeError as e:
            last_err = e

    assert called, f"Could not call data report writer (signature mismatch). Last error: {last_err}"

    p = run_dir / "data_report.json"
    assert p.exists(), "data_report.json not created in run_dir"

    d = json.loads(p.read_text(encoding="utf-8"))
    _assert_min_report_keys(d)


def test_env_step_info_audit_keys():
    """
    No vecenv, no training: instantiate env and do a few steps.
    Checks the audit-grade keys you implemented in RetailTradingEnv.
    """
    RetailTradingEnv = _load_env_class()

    df = _make_synth_df(n=30)
    feature_cols = ["feat_dummy"]

    env = RetailTradingEnv(df, feature_cols=feature_cols, max_steps=10)

    obs, info0 = env.reset()
    assert isinstance(info0, dict)

    action = np.array([0.0], dtype=np.float32)

    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(info, dict)

        # Minimal contract requested
        must_have = [
            "ret_fwd_used",
            "turnover",
            "equity",
            "drawdown",
            "pnl_log_net",
        ]
        for k in must_have:
            assert k in info, f"Missing info['{k}']"

        # Stronger audit contract (your implementation)
        audit_keys = [
            "t_index",
            "timestamp",
            "pos_prev",
            "pos_target",
            "pos_new",
            "cost",
            "fee_bps",
            "spread_bps",
            "funding_rate_used",
            "funding_cost",
            "pnl_log_gross",
            "reward_raw",
            "reward",
            "is_clipped",
        ]
        for k in audit_keys:
            assert k in info, f"Missing audit info['{k}']"

        # Types should be JSON-serializable primitives
        for k in must_have + audit_keys:
            assert _is_json_primitive(info[k]), f"info['{k}'] is not JSON-primitive: {type(info[k])}"

        # Basic sanity on values (not too strict)
        assert np.isfinite(float(info["equity"])), "equity must be finite"
        assert float(info["equity"]) > 0.0, "equity must stay > 0 in this synthetic run"
        assert 0.0 <= float(info["drawdown"]) <= 1.0, "drawdown should be in [0,1]"
        assert float(info["turnover"]) >= 0.0, "turnover should be >= 0"

        if bool(terminated) or bool(truncated):
            break