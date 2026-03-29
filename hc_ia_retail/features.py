# HC_IA_RETAIL/features.py
"""hc_ia_retail.features

Core feature engineering for HC_IA_RETAIL.

Design principles:
- All features in `feature_cols` are *causal*: at time t they depend only on information up to t.
- A separate forward-return column (default: `log_ret_1_fwd`) is created for the environment reward.
  Convention: reward at step t uses the forward return from t -> t+1.
- The forward column is NEVER part of `feature_cols` (anti-leakage by construction).
- Volatility (`sigma_1h`) is computed causally from past returns and exposed as a first-class feature (name kept for compatibility; bar size may change).
- Funding feature is optional and must be aligned causally (backward merge) and scaled 8h→1h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import feat_cfg, data_cfg
from typing import Optional, Tuple


def _interval_to_hours(interval: str) -> float:
    """Convert config interval strings like '1h', '4h', '30m', '1d' to hours."""
    s = str(interval).strip().lower()
    if s.endswith("h"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) / 60.0
    if s.endswith("d"):
        return float(s[:-1]) * 24.0
    # Fallback: assume hours
    return float(s)


def _attach_funding_rate_1h_scaled(
    df: pd.DataFrame,
    funding_df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    funding_ts_col: str = "timestamp",
    funding_rate_col: str = "funding_rate",
    scale_hours: float = 8.0,
) -> pd.Series:
    """Attach a causal funding-rate feature aligned to each row of `df`.

    Alignment contract (anti-leakage): for each row timestamp t, we attach the *latest*
    funding value with timestamp <= t (merge_asof backward). We then scale the 8h rate
    to a 1h-equivalent rate by dividing by `scale_hours`.

    This helper is intentionally shared-friendly: the environment and feature pipeline
    should use the same alignment logic to avoid silent mismatches.
    """

    if ts_col not in df.columns:
        raise KeyError(f"funding feature requires '{ts_col}' column in df")
    if funding_ts_col not in funding_df.columns:
        raise KeyError(f"funding_df missing required timestamp column '{funding_ts_col}'")
    if funding_rate_col not in funding_df.columns:
        raise KeyError(f"funding_df missing required rate column '{funding_rate_col}'")

    d = df[[ts_col]].copy()
    f = funding_df[[funding_ts_col, funding_rate_col]].copy()

    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="raise")
    f[funding_ts_col] = pd.to_datetime(f[funding_ts_col], utc=True, errors="raise")

    d = d.sort_values(ts_col)
    f = f.sort_values(funding_ts_col)

    merged = pd.merge_asof(
        d,
        f,
        left_on=ts_col,
        right_on=funding_ts_col,
        direction="backward",
        allow_exact_matches=True,
    )

    rate = merged[funding_rate_col].astype(float)
    if rate.isna().any():
        # If your dataset starts before the first available funding point,
        # you must either extend the funding history or trim early rows.
        n = int(rate.isna().sum())
        raise ValueError(
            f"funding feature alignment produced {n} NaNs. "
            "Ensure funding history covers the dataset start (or trim early rows)."
        )

    return rate / float(scale_hours)


def add_features(
    df: pd.DataFrame,
    forward_col_name: str = "log_ret_1_fwd",
    *,
    include_funding_feature: bool = False,
    funding_df: Optional[pd.DataFrame] = None,
    ts_col: str = "timestamp",
    funding_ts_col: str = "timestamp",
    funding_rate_col: str = "funding_rate",
    funding_scale_hours: float = 8.0,
    funding_feature_name: str = "funding_rate_1h_scaled",
):
    """Add causal features and a separated 1-step forward return for reward.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least: close, volume. (Volatility is computed causally from returns.)
    forward_col_name : str
        Name of the forward-return column used for reward. Default: "log_ret_1_fwd".
    include_funding_feature : bool
        Whether to add a causal funding-rate feature (backward aligned and scaled).
    funding_df : Optional[pd.DataFrame]
        DataFrame with funding data if `include_funding_feature` is True.
    ts_col : str
        Timestamp column name in `df`.
    funding_ts_col : str
        Timestamp column name in `funding_df`.
    funding_rate_col : str
        Funding rate column name in `funding_df`.
    funding_scale_hours : float
        Hours to scale funding rate from (default 8h) to 1h.
    funding_feature_name : str
        Name of the funding feature column to add.

    Returns
    -------
    df_feat : pd.DataFrame
        DataFrame with features + forward-return column, with rows dropped where features
        are undefined and with the last row removed (no forward return available).
    feature_cols : list[str]
        Stable ordered list of causal feature column names (does NOT include forward col).

    Notes
    -----
    Reward convention: at time t, the environment uses `forward_col_name` computed as
    log_ret_1 shifted by -1, i.e. forward return for t -> t+1.
    Funding feature is optional and causal via backward alignment.
    It must match the environment’s funding alignment to avoid mismatches.
    """

    if df is None or len(df) == 0:
        raise ValueError("add_features: input dataframe is empty")

    df = df.copy()

    # --- Required columns ---
    for col in ("close", "volume"):
        if col not in df.columns:
            raise KeyError(f"add_features: missing required column '{col}'")

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    # --- Basic data validity (no silent garbage) ---
    if not np.isfinite(close.to_numpy()).all():
        raise ValueError("add_features: 'close' contains NaN/inf")
    if not np.isfinite(vol.to_numpy()).all():
        raise ValueError("add_features: 'volume' contains NaN/inf")

    if (close <= 0).any():
        bad = int((close <= 0).sum())
        raise ValueError(f"add_features: invalid prices detected (close <= 0) count={bad}")

    if (vol < 0).any():
        bad = int((vol < 0).sum())
        raise ValueError(f"add_features: invalid volumes detected (volume < 0) count={bad}")

    # --- Causal returns ---
    # log_ret_1 = 1-bar return (so it becomes 4h when interval='4h')
    df["log_ret_1"] = np.log(close / close.shift(1))

    # Keep 'log_ret_24' meaning 24h return (time-consistent across bar sizes)
    step_h = _interval_to_hours(getattr(data_cfg, "interval", "1h"))
    shift_24h = max(1, int(round(24.0 / step_h)))
    df["log_ret_24"] = np.log(close / close.shift(shift_24h))

    # --- Rolling volatility (causal) ---
    # Interpret feat_cfg.vol_window as *hours* and convert to bars.
    # sigma_1h name is kept for backward compatibility in the rest of the codebase;
    # semantically, it is 'sigma_per_bar' in log-return units of the current bar size.
    vol_win_bars = max(2, int(round(float(feat_cfg.vol_window) / step_h)))
    df["vol_roll"] = df["log_ret_1"].rolling(vol_win_bars).std()
    df["sigma_1h"] = df["vol_roll"].astype(float)

    # --- RSI (EWMA, causal) ---
    w = feat_cfg.rsi_window
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / w, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / w, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- Volume z-score (causal) ---
    m = vol.rolling(feat_cfg.z_window).mean()
    s = vol.rolling(feat_cfg.z_window).std()
    df["vol_z"] = (vol - m) / (s + 1e-8)

    # --- MA ratio (causal) ---
    # Interpret ma_fast/ma_slow as *hours* and convert to bars.
    ma_fast_bars = max(2, int(round(float(feat_cfg.ma_fast) / step_h)))
    ma_slow_bars = max(ma_fast_bars + 1, int(round(float(feat_cfg.ma_slow) / step_h)))
    ma_f = close.rolling(ma_fast_bars).mean()
    ma_s = close.rolling(ma_slow_bars).mean()
    df["ma_ratio"] = ma_f / (ma_s + 1e-8)

    # --- Optional funding-rate feature (causal, backward-aligned) ---
    if include_funding_feature:
        if funding_df is None:
            raise ValueError(
                "include_funding_feature=True requires funding_df to be provided. "
                "Pass a DataFrame with columns (timestamp, funding_rate)."
            )

        df[funding_feature_name] = _attach_funding_rate_1h_scaled(
            df,
            funding_df,
            ts_col=ts_col,
            funding_ts_col=funding_ts_col,
            funding_rate_col=funding_rate_col,
            scale_hours=funding_scale_hours,
        )

    # MUST stay stable, ordered, and NOT include forward column.
    feature_cols = ["log_ret_1", "log_ret_24", "vol_roll", "sigma_1h", "rsi", "vol_z", "ma_ratio"]
    if include_funding_feature:
        feature_cols.append(funding_feature_name)

    # Anti-leakage guardrail (explicit)
    assert forward_col_name not in feature_cols, (
        f"forward_col_name='{forward_col_name}' must never be part of feature_cols"
    )
    assert "log_ret_1_fwd" not in feature_cols, "'log_ret_1_fwd' must not be in feature_cols"
    assert funding_feature_name != forward_col_name, (
        "funding_feature_name must not equal forward_col_name (anti-leakage guardrail)"
    )

    # Drop rows where any causal feature is undefined (rolling warmup)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "add_features: dataframe became empty after dropna(warmup). "
            "Check data length vs rolling windows."
        )

    # Forward return for reward ONLY (never part of observation)
    # Convention: at time t, forward return corresponds to t -> t+1.
    df[forward_col_name] = df["log_ret_1"].shift(-1)

    # Last row has no forward label (t+1 not available) -> drop it.
    df = df.iloc[:-1].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "add_features: dataframe became empty after dropping last row for forward return. "
            "Not enough data after warmup."
        )

    return df, feature_cols