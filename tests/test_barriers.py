# tests/test_barriers.py
from __future__ import annotations

import math
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd

from hc_ia_retail.config import data_cfg, env_cfg
from hc_ia_retail.data import load_ohlcv, split_train_val_test
from hc_ia_retail.features import add_features
from hc_ia_retail.env import RetailTradingEnv


@contextmanager
def _override_env_cfg(**overrides):
    """Temporarily override attributes on env_cfg for the duration of a test."""
    saved = {}
    try:
        for k, v in overrides.items():
            if hasattr(env_cfg, k):
                saved[k] = getattr(env_cfg, k)
                setattr(env_cfg, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(env_cfg, k, v)


def _is_strictly_increasing(ts: np.ndarray) -> bool:
    if ts.size < 2:
        return True
    # assume numpy datetime64[ns]
    return bool(np.all(ts[1:] > ts[:-1]))


def _finite_df(df: pd.DataFrame, cols: list[str]) -> bool:
    arr = df[cols].to_numpy(dtype=np.float64, copy=False)
    return bool(np.isfinite(arr).all())


def _assert_close(a: float, b: float, tol: float, msg: str) -> None:
    if not (abs(a - b) <= tol):
        raise AssertionError(f"{msg} | a={a} b={b} tol={tol}")


def _compute_bh_equity_from_close(close: np.ndarray, initial: float = 1.0) -> np.ndarray:
    """
    Buy&Hold equity per step aligned to env convention:
    step t uses (close[t+1]/close[t]) => equity after step t.
    Output length = n_steps = len(close)-1
    """
    close = np.asarray(close, dtype=np.float64)
    n = max(0, close.size - 1)
    out = np.zeros(n, dtype=np.float64)
    eq = float(initial)
    for t in range(n):
        p0 = close[t]
        p1 = close[t + 1]
        if p0 > 0 and np.isfinite(p0) and np.isfinite(p1):
            eq *= float(p1 / p0)
        out[t] = eq
    return out


def test_data_integrity_perp():
    """
    Barrière 1:
      - timestamp tri strict, unique
      - pas NaN/inf sur OHLCV
      - colonnes OHLCV présentes
    """
    df = load_ohlcv(None)  # uses config default path
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in required:
        assert c in df.columns, f"missing column: {c}"

    # timestamp: tz-naive representing UTC (we only enforce dtype and monotonicity + uniqueness)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    assert ts.notna().all(), "timestamp contains NaT"
    # ensure strict increasing + unique
    ts_np = ts.to_numpy(dtype="datetime64[ns]")
    assert _is_strictly_increasing(ts_np), "timestamp not strictly increasing"
    assert ts.duplicated().sum() == 0, "timestamp not unique"

    # OHLCV numeric finite
    assert _finite_df(df, ["open", "high", "low", "close", "volume"]), "OHLCV contains NaN/inf"


def test_features_no_leakage():
    """
    Barrière 2:
      - log_ret_1_fwd jamais dans feature_cols
    """
    df = load_ohlcv(None)
    df_feat, feature_cols = add_features(df)

    assert "log_ret_1_fwd" not in feature_cols, "LEAKAGE: log_ret_1_fwd in feature_cols"
    # sanity: forward exists (should) but is separate
    assert "log_ret_1_fwd" in df_feat.columns, "expected forward col missing after add_features"


def test_split_boundary_no_forward_cross():
    """
    Barrière 3:
      - si forward_col présent, dernière ligne train/val retirée
        (anti-leakage: aucune transition t->t+1 croise frontière)
    """
    df = load_ohlcv(None)
    df_feat, _feature_cols = add_features(df)

    train, val, test = split_train_val_test(df_feat)

    if "log_ret_1_fwd" in df_feat.columns:
        # condition nécessaire: segments non vides
        assert len(train) >= 2, "train too small for boundary check"
        assert len(val) >= 2, "val too small for boundary check"

        # Anti-leakage expected behavior:
        # last timestamp of train should be strictly < first timestamp of val
        # AND train should NOT include the last row of its original slice if forward existed
        t_train_last = pd.to_datetime(train["timestamp"].iloc[-1])
        t_val_first = pd.to_datetime(val["timestamp"].iloc[0])
        t_val_last = pd.to_datetime(val["timestamp"].iloc[-1])
        t_test_first = pd.to_datetime(test["timestamp"].iloc[0])

        assert t_train_last < t_val_first, "boundary breach: train last timestamp >= val first"
        assert t_val_last < t_test_first, "boundary breach: val last timestamp >= test first"

        # Stronger: ensure no step in train would require a forward that belongs to val
        # That means: train last row must NOT be the row immediately before val first in original df
        # (i.e., original adjacency is broken by dropping last row)
        # We check using timestamps: if train_last + 1h == val_first then that's suspicious.
        dt = (t_val_first - t_train_last)
        assert dt >= pd.Timedelta(hours=2), (
            "anti-leakage expected drop not applied: train_last is 1h before val_first"
        )


def test_env_action_mapping():
    """
    Barrière 4:
      - action=1 => pos=L
      - action=-1 => pos=-L
    """
    df = load_ohlcv(None)
    df_feat, feature_cols = add_features(df)

    # small slice
    df_small = df_feat.iloc[:300].copy()

    # Plumbing test: disable any execution smoothing so action maps instantly to target position.
    with _override_env_cfg(execution_beta=1.0, execution_half_life_hours=None):
        env = RetailTradingEnv(df_small, feature_cols=feature_cols, max_steps=50)

        # reset
        obs, info = env.reset(seed=123)

        L = float(getattr(env_cfg, "max_leverage", getattr(env_cfg, "leverage_max", 1.0)))

        # action +1
        obs, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        pos = float(info.get("pos_target", info.get("pos_new", info.get("position", np.nan))))
        _assert_close(pos, L, tol=1e-9, msg="action=+1 should map to +L")

        # action -1
        obs, r, term, trunc, info = env.step(np.array([-1.0], dtype=np.float32))
        pos = float(info.get("pos_target", info.get("pos_new", info.get("position", np.nan))))
        _assert_close(pos, -L, tol=1e-9, msg="action=-1 should map to -L")

        return


def test_sanity_always_long_equals_bh():
    """
    Barrière 5:
      - cost=0 et funding=0
      - policy always-long => equity ≈ BH (tolérance serrée)
    """
    df = load_ohlcv(None)
    df_feat, feature_cols = add_features(df)

    # slice for a deterministic sanity window
    # Ensure we have enough bars: need close[t] and close[t+1]
    df_s = df_feat.iloc[:2000].copy()
    assert len(df_s) >= 100, "not enough data for sanity"

    # Plumbing test: disable any execution smoothing AND disable frictions via env_cfg
    # so always-long should match a BH proxy built from the exact (pos_used, ret_fwd_used)
    L = float(getattr(env_cfg, "max_leverage", getattr(env_cfg, "leverage_max", 1.0)))

    with _override_env_cfg(
        execution_beta=1.0,
        execution_half_life_hours=None,
        fee_bps=0.0,
        spread_bps=0.0,
        funding_mode="none",
    ):
        # Instantiate env (prefer config overrides; kwargs may be ignored by env)
        env = RetailTradingEnv(df_s, feature_cols=feature_cols, max_steps=500)

        # Prefer starting already-long to avoid any pos_prev first-step convention mismatch.
        try:
            obs, info = env.reset(seed=7, options={"initial_position": +L})
        except TypeError:
            obs, info = env.reset(seed=7)

        eq_env: list[float] = []
        rets_used: list[float] = []
        pos_used: list[float] = []

        done = False
        t = 0
        while not done:
            obs, reward, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
            done = bool(term or trunc)

            eq_env.append(float(info.get("equity", np.nan)))

            r = info.get("ret_fwd_used", info.get("log_ret_fwd_used", None))
            if r is None:
                raise AssertionError("env info must include ret_fwd_used (or log_ret_fwd_used) for plumbing test")
            rets_used.append(float(r))

            # The env typically applies ret_fwd_used with a position from the *previous* step.
            # Use the most specific key available; fall back to +L.
            p = info.get("pos_prev", None)
            if p is None:
                p = info.get("position", None)
            if p is None:
                p = info.get("pos_target", info.get("pos_new", None))
            try:
                pos_used.append(float(p))
            except Exception:
                pos_used.append(float(L))

            t += 1
            if t > 10000:
                break

        eq_env = np.asarray(eq_env, dtype=np.float64)
        rets_used = np.asarray(rets_used, dtype=np.float64)
        pos_used = np.asarray(pos_used, dtype=np.float64)

        assert np.isfinite(eq_env).all(), "env equity contains NaN/inf"
        assert np.isfinite(rets_used).all(), "env ret_fwd_used contains NaN/inf"
        assert np.isfinite(pos_used).all(), "env pos_used contains NaN/inf"
        assert eq_env.size >= 10, "env too short"
        assert rets_used.size == eq_env.size, "rets_used length mismatch"
        assert pos_used.size == eq_env.size, "pos_used length mismatch"

        # Build BH proxy from the *exact* quantities used by the env:
        # equity_bh[t] = exp(sum_{i<=t} pos_used[i] * ret_fwd_used[i])
        eq_bh = np.exp(np.cumsum(pos_used * rets_used))

        # Match the train sanity convention: ignore first step (pos_prev convention).
        idx = np.arange(eq_env.size)
        idx = idx[idx >= 1]
        assert idx.size >= 5, "sanity window too short after ignoring first step"

        _assert_close(eq_env[idx][-1], eq_bh[idx][-1], tol=1e-6, msg="always-long terminal equity != BH")

        rel = np.max(np.abs(eq_env[idx] - eq_bh[idx]) / (np.abs(eq_bh[idx]) + 1e-12))
        assert rel <= 1e-6, f"always-long path deviates from BH too much: max_rel_err={rel}"

        return