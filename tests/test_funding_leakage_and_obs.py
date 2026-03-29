# tests/test_funding_leakage_and_obs.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hc_ia_retail.features import add_features
from hc_ia_retail.env import RetailTradingEnv
from hc_ia_retail.config import env_cfg


def _make_hourly_ohlcv(start: str = "2024-01-01", n_hours: int = 40) -> pd.DataFrame:
    """Deterministic hourly OHLCV with timestamp, close, volume."""
    ts = pd.date_range(start=start, periods=n_hours, freq="h", tz="UTC").tz_localize(None)
    # simple monotone close to avoid weird edge cases
    close = np.linspace(100.0, 120.0, n_hours).astype(float)
    vol = np.full(n_hours, 1000.0, dtype=float)
    df = pd.DataFrame({"timestamp": ts, "close": close, "volume": vol})
    # optional OHLC not required by add_features(), but harmless
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    return df


def _make_funding_8h(start: str = "2024-01-01", n_points: int = 6) -> pd.DataFrame:
    """Deterministic 8h funding points. Column names match add_features() defaults."""
    ts = pd.date_range(start=start, periods=n_points, freq="8h", tz="UTC").tz_localize(None)
    # alternating signs to detect wrong forward/backward carry
    # ensure we always return exactly n_points rates (tests may request n_points > 6)
    base = np.array([0.0008, -0.0004, 0.0002, -0.0006, 0.0001, 0.0003], dtype=float)
    rates = np.resize(base, int(n_points)).astype(float)
    return pd.DataFrame({"timestamp": ts, "funding_rate": rates})


def _expected_funding_1h_scaled(df_raw: pd.DataFrame, funding_df: pd.DataFrame, scale_hours: float = 8.0) -> np.ndarray:
    """Independent expected: backward asof + divide by 8."""
    d = df_raw[["timestamp"]].copy()
    f = funding_df[["timestamp", "funding_rate"]].copy()

    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="raise")
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="raise")
    d = d.sort_values("timestamp")
    f = f.sort_values("timestamp")

    merged = pd.merge_asof(
        d,
        f,
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    if merged["funding_rate"].isna().any():
        raise AssertionError("Expected funding alignment produced NaNs; funding history must cover dataset start.")

    return (merged["funding_rate"].astype(float) / float(scale_hours)).to_numpy(dtype=np.float64)


def test_funding_feature_is_causal_backward_and_scaled_8h_to_1h():
    df_raw = _make_hourly_ohlcv(n_hours=300)
    funding_df = _make_funding_8h(n_points=6)

    df_feat, feature_cols = add_features(
        df_raw,
        include_funding_feature=True,
        funding_df=funding_df,
        funding_feature_name="funding_rate_1h_scaled",
        funding_scale_hours=8.0,
        ts_col="timestamp",
        funding_ts_col="timestamp",
        funding_rate_col="funding_rate",
    )

    assert "funding_rate_1h_scaled" in df_feat.columns
    assert "funding_rate_1h_scaled" in feature_cols

    # Build expected on the same raw timeline, then apply the same row drops as add_features():
    # - warmup dropna on feature cols
    # - drop last row (forward return shift)
    expected_full = _expected_funding_1h_scaled(df_raw, funding_df, scale_hours=8.0)

    # Recompute the exact kept timestamps from df_feat to compare by timestamp join
    # (robust vs warmup window sizes).
    exp_map = pd.Series(expected_full, index=pd.to_datetime(df_raw["timestamp"], utc=True))
    got_ts = pd.to_datetime(df_feat["timestamp"], utc=True)

    expected_on_kept = exp_map.loc[got_ts].to_numpy(dtype=np.float64)
    got = pd.to_numeric(df_feat["funding_rate_1h_scaled"], errors="coerce").to_numpy(dtype=np.float64)

    assert np.isfinite(got).all()
    assert np.allclose(got, expected_on_kept, rtol=0.0, atol=1e-15), "Funding feature mismatch vs backward asof /8 scaling."


def test_funding_feature_does_not_peek_future_event():
    """
    Strong anti-leakage: funding carried for an hour must come from funding_time <= hour timestamp.
    We verify this by checking which 8h bucket each hour maps to (backward carry), not forward.
    """
    df_raw = _make_hourly_ohlcv(n_hours=300)
    funding_df = _make_funding_8h(n_points=6)

    df_feat, _ = add_features(
        df_raw,
        include_funding_feature=True,
        funding_df=funding_df,
        funding_feature_name="funding_rate_1h_scaled",
        funding_scale_hours=8.0,
    )

    # For each kept t, the funding rate must equal the most recent funding point <= t divided by 8.
    f_ts = pd.to_datetime(funding_df["timestamp"], utc=True).to_numpy()
    f_rt = funding_df["funding_rate"].astype(float).to_numpy()

    got_ts = pd.to_datetime(df_feat["timestamp"], utc=True).to_numpy()
    got = df_feat["funding_rate_1h_scaled"].astype(float).to_numpy()

    for t, r1h in zip(got_ts, got):
        # index of last funding timestamp <= t
        idx = np.where(f_ts <= t)[0]
        assert idx.size > 0, "Funding history must cover dataset start for causal backward merge."
        j = int(idx[-1])
        expected = float(f_rt[j] / 8.0)
        assert abs(float(r1h) - expected) <= 1e-15, "Detected potential forward/peek (not backward carry)."


def test_env_uses_same_funding_column_and_obs_contains_it_when_switch_on():
    df_raw = _make_hourly_ohlcv(n_hours=300)
    funding_df = _make_funding_8h(n_points=10)

    df_feat, feature_cols = add_features(
        df_raw,
        include_funding_feature=True,
        funding_df=funding_df,
        funding_feature_name="funding_rate_1h_scaled",
        funding_scale_hours=8.0,
    )
    assert "funding_rate_1h_scaled" in df_feat.columns

    # Patch env config temporarily (and restore after)
    saved = {
        "include_funding_in_obs": getattr(env_cfg, "include_funding_in_obs", False),
        "funding_mode": getattr(env_cfg, "funding_mode", "none"),
        "execution_model": getattr(env_cfg, "execution_model", "instant"),
        "max_leverage": getattr(env_cfg, "max_leverage", 1.0),
        "fee_bps": getattr(env_cfg, "fee_bps", 0.0),
        "spread_bps": getattr(env_cfg, "spread_bps", 0.0),
    }
    try:
        setattr(env_cfg, "include_funding_in_obs", True)
        setattr(env_cfg, "funding_mode", "binance_8h")
        setattr(env_cfg, "execution_model", "instant")
        setattr(env_cfg, "max_leverage", 1.0)
        setattr(env_cfg, "fee_bps", 0.0)
        setattr(env_cfg, "spread_bps", 0.0)

        env = RetailTradingEnv(df_feat, feature_cols=list(feature_cols), max_steps=10)

        obs, _ = env.reset(options={"initial_position": 0.0})
        # funding must now be part of env.feature_cols (thus present in obs slice)
        assert any(c in env.feature_cols for c in ("funding_rate_1h", "funding_rate_1h_scaled")), \
            "Env switch ON but funding feature not appended to feature_cols."

        # Step once with action=+1 so pos_interval ~ 1.0 (instant exec, L=1)
        obs2, reward, terminated, truncated, info = env.step(np.asarray([1.0], dtype=np.float32))

        t = int(info["t_index"])
        # funding_rate_used must equal the df row funding at step t (same alignment contract)
        expected_rate = float(df_feat.iloc[t]["funding_rate_1h_scaled"])
        got_rate = float(info.get("funding_rate_used", np.nan))
        assert np.isfinite(got_rate)
        assert abs(got_rate - expected_rate) <= 1e-15, "Env funding_rate_used != dataset funding feature at step t."

        # funding_cost should be pos_interval * funding_rate_used ; with instant exec + action=1 => pos_interval ~ 1
        pos_interval = float(info.get("pos_interval", np.nan))
        funding_cost = float(info.get("funding_cost", np.nan))
        assert np.isfinite(pos_interval) and np.isfinite(funding_cost)
        assert abs(funding_cost - (pos_interval * got_rate)) <= 1e-12, "funding_cost not consistent with pos_interval * rate."
    finally:
        for k, v in saved.items():
            try:
                setattr(env_cfg, k, v)
            except Exception:
                pass