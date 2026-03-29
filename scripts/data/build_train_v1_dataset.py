#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("/Users/clovishilmarcher/Documents/2026/HilmarCorp/HC_IA_RETAIL")
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "train_v1_sac_dataset_4h.parquet"


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_timestamp(df: pd.DataFrame, name: str) -> pd.DataFrame:
    ts_candidates = [
        "timestamp",
        "date",
        "open_time",
        "close_time",
        "datetime",
        "funding_time_utc",
        "fundingTime",
    ]
    ts_col = None
    for c in ts_candidates:
        if c in df.columns:
            ts_col = c
            break

    if ts_col is None:
        raise ValueError(f"[{name}] aucune colonne timestamp-like trouvée. Colonnes={list(df.columns)}")

    s = df[ts_col]

    if pd.api.types.is_datetime64_any_dtype(s):
        ts = pd.to_datetime(s, utc=True, errors="coerce")
    elif pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        med = s_num.dropna().median()
        if pd.isna(med):
            ts = pd.to_datetime(s_num, utc=True, errors="coerce")
        elif med > 1e14:
            ts = pd.to_datetime(s_num, utc=True, errors="coerce")
        elif med > 1e11:
            ts = pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
        elif med > 1e9:
            ts = pd.to_datetime(s_num, unit="s", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(s_num, utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(s, utc=True, errors="coerce")

    out = df.copy()
    out[ts_col] = ts
    out = out.dropna(subset=[ts_col]).sort_values(ts_col).drop_duplicates(subset=[ts_col]).reset_index(drop=True)

    if ts_col != "timestamp":
        out = out.rename(columns={ts_col: "timestamp"})

    return out


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(20, window // 4)
    mu = series.rolling(window=window, min_periods=min_periods).mean()
    sd = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return (series - mu) / sd.replace(0.0, np.nan)


def build_btc_features(btc: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in btc.columns]
    if missing:
        raise ValueError(f"[btc] colonnes manquantes: {missing}")

    df = btc[required].copy().sort_values("timestamp").reset_index(drop=True)

    eps = 1e-12

    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_ret_24"] = np.log(df["close"] / df["close"].shift(6))  # 6 x 4h = 24h
    df["range_frac"] = (df["high"] - df["low"]) / (df["close"].abs() + eps)
    df["close_to_high"] = (df["high"] - df["close"]) / (df["high"] - df["low"] + eps)
    df["close_to_low"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + eps)
    df["volume_z_63"] = rolling_zscore(np.log1p(df["volume"]), window=63)
    df["realized_vol_24"] = df["log_ret_1"].rolling(window=6, min_periods=6).std(ddof=0)
    ma_24 = df["close"].rolling(window=6, min_periods=6).mean()
    df["ma_ratio_24"] = df["close"] / (ma_24 + eps) - 1.0

    keep = [
        "timestamp",
        "log_ret_1",
        "log_ret_24",
        "range_frac",
        "close_to_high",
        "close_to_low",
        "volume_z_63",
        "realized_vol_24",
        "ma_ratio_24",
    ]
    return df[keep].copy()


def keep_columns(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] colonnes manquantes: {missing}")
    return df[cols].copy()


def assert_no_duplicate_columns(df: pd.DataFrame) -> None:
    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        raise ValueError(f"Colonnes dupliquées après merge: {dup}")


def main() -> None:
    log("[V1] loading sources...")

    btc_raw = pd.read_parquet(DATA_DIR / "BTCUSDT_PERP_4h.parquet")
    funding_raw = pd.read_parquet(DATA_DIR / "BTCUSDT_PERP_funding_8h.parquet")
    reg_raw = pd.read_parquet(DATA_DIR / "rl_observation_pack.parquet copie")
    alt_raw = pd.read_csv(DATA_DIR / "alt_indices_4h.csv")
    stb_raw = pd.read_csv(DATA_DIR / "stablecoins_indices_4h.csv")
    mac_raw = pd.read_csv(DATA_DIR / "macro_hc_regime_4h_ready.csv")

    btc_raw = normalize_timestamp(btc_raw, "btc")
    funding_raw = normalize_timestamp(funding_raw, "funding")
    reg_raw = normalize_timestamp(reg_raw, "regime")
    alt_raw = normalize_timestamp(alt_raw, "alts")
    stb_raw = normalize_timestamp(stb_raw, "stablecoins")
    mac_raw = normalize_timestamp(mac_raw, "macro")

    log(f"[V1] btc_raw shape={btc_raw.shape} ts={btc_raw['timestamp'].min()} -> {btc_raw['timestamp'].max()}")
    log(f"[V1] funding_raw shape={funding_raw.shape} ts={funding_raw['timestamp'].min()} -> {funding_raw['timestamp'].max()}")
    log(f"[V1] reg_raw shape={reg_raw.shape} ts={reg_raw['timestamp'].min()} -> {reg_raw['timestamp'].max()}")
    log(f"[V1] alt_raw shape={alt_raw.shape} ts={alt_raw['timestamp'].min()} -> {alt_raw['timestamp'].max()}")
    log(f"[V1] stb_raw shape={stb_raw.shape} ts={stb_raw['timestamp'].min()} -> {stb_raw['timestamp'].max()}")
    log(f"[V1] mac_raw shape={mac_raw.shape} ts={mac_raw['timestamp'].min()} -> {mac_raw['timestamp'].max()}")

    funding_candidates = [c for c in ["funding_rate", "fundingRate", "rate", "funding"] if c in funding_raw.columns]
    if not funding_candidates:
        raise ValueError(f"[funding] no funding rate column found. Columns={list(funding_raw.columns)}")

    funding = funding_raw[["timestamp", funding_candidates[0]]].copy()
    funding = funding.rename(columns={funding_candidates[0]: "funding_rate"})
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
    funding = funding.dropna(subset=["timestamp", "funding_rate"]).sort_values("timestamp").reset_index(drop=True)
    funding["funding_rate_1h_scaled"] = funding["funding_rate"] / 8.0

    log("[V1] building btc features...")
    btc = build_btc_features(btc_raw)
    btc = pd.merge_asof(
        btc.sort_values("timestamp"),
        funding[["timestamp", "funding_rate_1h_scaled"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    regime_cols = [
        "timestamp",
        "p_max_filter",
        "p_margin_filter",
        "effective_num_states_filter",
        "expected_state_filter",
        "p_filter_0",
        "p_filter_1",
        "p_filter_2",
        "dp_filter_0",
        "dp_filter_1",
        "dp_filter_2",
    ]
    reg = keep_columns(reg_raw, regime_cols, "regime")

    alt_cols = [
        "timestamp",
        "alt_eq_ret_4h",
        "alt_eq_log_ret_24h",
        "alt_breadth_4h_pos",
        "alt_breadth_24h_pos",
        "alt_disp_ret_4h",
        "alt_eq_vol_logret_24h",
        "alt_tail_down_share_4h",
        "alt_frac_crash",
    ]
    alt = keep_columns(alt_raw, alt_cols, "alts")

    stb_cols = [
        "timestamp",
        "stab_peg_stress_mean",
        "stab_peg_stress_max",
        "stab_tail_intensity",
        "stab_trust_flow_7d",
        "stab_activity_volume_z_365d",
        "stab_effective_n",
    ]
    stb = keep_columns(stb_raw, stb_cols, "stablecoins")

    mac_cols = [
        "timestamp",
        "hc_risk_aversion",
        "hc_risk_aversion_d1",
        "hc_dollar_tightening",
        "hc_dollar_tightening_d1",
        "hc_liquidity_impulse",
        "hc_liquidity_impulse_d1",
        "hc_growth_cycle",
    ]
    mac = keep_columns(mac_raw, mac_cols, "macro")

    log("[V1] merging blocks on timestamp...")
    df = btc.merge(reg, on="timestamp", how="inner")
    df = df.merge(alt, on="timestamp", how="inner")
    df = df.merge(stb, on="timestamp", how="inner")
    df = df.merge(mac, on="timestamp", how="inner")

    assert_no_duplicate_columns(df)

    log(f"[V1] merged shape before dropna = {df.shape}")
    log(f"[V1] merged ts window = {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # Audit NaN
    nan_ratio = df.isna().mean().sort_values(ascending=False)
    log("[V1] top NaN ratios before final drop:")
    for c, v in nan_ratio.head(20).items():
        log(f"    {c}: {v:.4%}")

    # Final clean for SAC
    before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    after = len(df)

    log(f"[V1] rows before dropna = {before}")
    log(f"[V1] rows after  dropna = {after}")
    log(f"[V1] final ts window = {df['timestamp'].min()} -> {df['timestamp'].max()}")

    feature_cols = [c for c in df.columns if c != "timestamp"]
    log(f"[V1] n_features = {len(feature_cols)}")
    for i, c in enumerate(feature_cols, 1):
        log(f"    {i:02d}. {c}")

    df.to_parquet(OUT_PATH, index=False)
    log(f"[V1] saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()