# HC_IA_RETAIL/data.py

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Optional

import numpy as np
import pandas as pd

from .config import data_cfg, env_cfg, regime_cfg
from .features import add_features


def _default_dataset_path() -> Path:
    """Resolve the default dataset path.

    Goal: use the PERP dataset by default when available (e.g. data/BTCUSDT_PERP_1h.parquet).
    We prefer (in order):
      1) data_cfg.csv_path (expected to be set to the perp dataset in config)
      2) repo_root/data/BTCUSDT_PERP_1h.parquet if it exists (safe fallback)
    """
    # Config-driven default (preferred)
    cfg_path = getattr(data_cfg, "csv_path", None)
    if cfg_path:
        return Path(cfg_path)

    # Safe fallback: look for the canonical perp file in repo_root/data/
    repo_root = Path(__file__).resolve().parents[1]
    fallback = repo_root / "data" / "BTCUSDT_PERP_1h.parquet"
    return fallback


def _infer_expected_timedelta() -> Optional[pd.Timedelta]:
    """Best-effort expected bar frequency for gap reporting.

    Tries to infer from config fields if present, without hard dependencies on config schema.
    Returns None if it cannot infer.
    """
    # Common patterns we might have in config
    for attr in ("interval", "freq", "freq_str", "bar_freq"):
        val = getattr(data_cfg, attr, None)
        if not val:
            continue
        try:
            # Handle Binance-like intervals (e.g. "1h", "4h", "1d")
            s = str(val).strip()
            # Normalize a few common forms
            s_norm = s.replace("H", "h").replace("D", "d").replace("M", "m")
            if s_norm.endswith("h"):
                return pd.Timedelta(hours=int(s_norm[:-1]))
            if s_norm.endswith("d"):
                return pd.Timedelta(days=int(s_norm[:-1]))
            if s_norm.endswith("m"):
                return pd.Timedelta(minutes=int(s_norm[:-1]))
            # Pandas can parse many strings (e.g. "1H")
            return pd.Timedelta(s)
        except Exception:
            continue

    # If nothing in config, we cannot infer safely
    return None



def _detect_forward_col(df: pd.DataFrame) -> Optional[str]:
    """Detect the forward-return column name used for rewards (anti-boundary leakage).

    Priority order:
      1) Explicit config on data_cfg: forward_col_name / forward_col
      2) Canonical default: 'log_ret_1_fwd'
      3) Heuristic fallback: first column (sorted) that ends with '_fwd'

    Returns None if no forward-like column exists.
    """
    cfg_name = getattr(data_cfg, "forward_col_name", None) or getattr(data_cfg, "forward_col", None)
    if cfg_name and cfg_name in df.columns:
        return str(cfg_name)

    if "log_ret_1_fwd" in df.columns:
        return "log_ret_1_fwd"

    # Heuristic fallback (kept conservative): columns ending with '_fwd'
    candidates = sorted([c for c in df.columns if isinstance(c, str) and c.endswith("_fwd")])
    return candidates[0] if candidates else None



def _raw_market_cols_to_exclude() -> set[str]:
    """Raw market columns excluded from RL observation feature lists.

    Rationale:
      - raw OHLCV levels are weakly stationary / scale-dependent
      - downstream RL should consume engineered market features, not raw bars
    """
    return {"open", "high", "low", "close", "volume"}


def _build_market_feature_dataframe(
    df_market_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Build engineered market features with the same funding contract as train/eval.

    This keeps `load_training_dataframe()` aligned with the downstream scripts:
      - optional funding feature is added only when enabled in env_cfg
      - early market rows are trimmed to the first available funding timestamp
      - timestamps are normalized to UTC tz-naive before feature construction
    """
    df_market = df_market_raw.copy()

    use_funding_feature = bool(getattr(env_cfg, "include_funding_in_obs", False)) and str(
        getattr(env_cfg, "funding_mode", "none")
    ).lower() != "none"

    funding_df = None
    funding_ts_col_used = ""
    n_rows_before_funding_trim = int(len(df_market))
    n_rows_dropped_for_funding_overlap = 0
    funding_min_ts = None

    if use_funding_feature:
        fpath = getattr(env_cfg, "funding_path", None)
        if not fpath:
            raise ValueError(
                "include_funding_in_obs=True requires env_cfg.funding_path to be set (path to funding parquet)."
            )

        fpath = str(Path(str(fpath)).expanduser().resolve())
        funding_df = pd.read_parquet(fpath)

        if "funding_time_utc" in funding_df.columns:
            funding_ts_col_used = "funding_time_utc"
        elif "timestamp" in funding_df.columns:
            funding_ts_col_used = "timestamp"
        else:
            raise ValueError(
                f"Funding parquet missing timestamp column. Expected 'funding_time_utc' or 'timestamp'. cols={list(funding_df.columns)}"
            )

        funding_df = funding_df.copy()
        funding_df[funding_ts_col_used] = (
            pd.to_datetime(funding_df[funding_ts_col_used], utc=True, errors="coerce")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        funding_df = funding_df.dropna(subset=[funding_ts_col_used]).reset_index(drop=True)
        if funding_df.empty:
            raise ValueError("Funding timestamps could not be parsed (all NaT).")

        if "timestamp" not in df_market.columns:
            raise ValueError(f"OHLCV dataframe missing 'timestamp' column. cols={list(df_market.columns)}")

        df_market = df_market.copy()
        df_market["timestamp"] = (
            pd.to_datetime(df_market["timestamp"], utc=True, errors="coerce")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        df_market = df_market.dropna(subset=["timestamp"]).reset_index(drop=True)
        if df_market.empty:
            raise ValueError("OHLCV timestamps could not be parsed (all NaT).")

        funding_min_ts = funding_df[funding_ts_col_used].min()
        keep_mask = df_market["timestamp"] >= funding_min_ts
        df_market = df_market.loc[keep_mask].reset_index(drop=True)
        n_rows_dropped_for_funding_overlap = int(n_rows_before_funding_trim - len(df_market))

    df_market_feat, market_feature_cols = add_features(
        df_market,
        include_funding_feature=use_funding_feature,
        funding_df=funding_df,
        ts_col="timestamp",
        funding_ts_col=(funding_ts_col_used or "timestamp"),
        funding_rate_col="funding_rate",
        funding_scale_hours=8.0,
        funding_feature_name="funding_rate_1h_scaled",
    )

    market_meta = {
        "use_funding_feature": bool(use_funding_feature),
        "funding_path": str(Path(str(getattr(env_cfg, "funding_path", ""))).expanduser().resolve())
        if use_funding_feature and getattr(env_cfg, "funding_path", None)
        else None,
        "funding_ts_col": funding_ts_col_used or None,
        "funding_min_ts": str(funding_min_ts) if funding_min_ts is not None else None,
        "n_rows_before_funding_trim": int(n_rows_before_funding_trim),
        "n_rows_dropped_for_funding_overlap": int(n_rows_dropped_for_funding_overlap),
        "market_feature_cols": list(market_feature_cols),
        "n_rows_market_feat": int(len(df_market_feat)),
    }
    return df_market_feat, list(market_feature_cols), market_meta


def dataset_sha256(path: str | Path) -> str:
    """Compute SHA256 fingerprint of a dataset file (binary).

    This is used for audit-grade run tracing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_rl_dataset_run_dir() -> Path:
    """Resolve the RL dataset run directory produced by hc-regime-engine.

    Selection policy:
      1) if regime_cfg.rl_dataset_run_id is set, use it exactly
      2) else if regime_cfg.use_latest_rl_dataset is True, pick the most recent j16_2_rl_* directory
    """
    root = Path(getattr(regime_cfg, "rl_dataset_root", "")).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"RL dataset root not found: {root}")

    run_id = getattr(regime_cfg, "rl_dataset_run_id", None)
    if run_id:
        run_dir = root / str(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Requested RL dataset run_id not found: {run_dir}")
        return run_dir

    use_latest = bool(getattr(regime_cfg, "use_latest_rl_dataset", True))
    if not use_latest:
        raise ValueError(
            "regime_cfg.enabled=True but no rl_dataset_run_id provided and use_latest_rl_dataset=False"
        )

    candidates = sorted(
        [p for p in root.glob("j16_2_rl_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No RL dataset directories found under: {root}")
    return candidates[0]


def load_regime_observation_pack() -> tuple[pd.DataFrame, dict]:
    """Load the causal RL observation pack exported by hc-regime-engine.

    Returns
    -------
    df_regime : pd.DataFrame
        Observation pack with timestamp parsed as UTC tz-naive and sorted uniquely.
    meta : dict
        Minimal metadata for audit/replay.
    """
    run_dir = _resolve_rl_dataset_run_dir()
    obs_path = run_dir / "rl_observation_pack.parquet"
    manifest_path = run_dir / "manifest.json"
    contract_path = run_dir / "feature_contract.json"
    quality_path = run_dir / "quality_summary.json"

    if not obs_path.exists():
        raise FileNotFoundError(f"Missing rl_observation_pack.parquet: {obs_path}")

    df = pd.read_parquet(obs_path)
    if "timestamp" not in df.columns:
        raise ValueError("RL observation pack missing required 'timestamp' column")

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.copy()
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("RL observation pack timestamps are not strictly sorted")
    if not df["timestamp"].is_unique:
        raise ValueError("RL observation pack timestamps are not unique")

    meta = {
        "run_dir": str(run_dir),
        "obs_path": str(obs_path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "feature_contract_path": str(contract_path) if contract_path.exists() else None,
        "quality_summary_path": str(quality_path) if quality_path.exists() else None,
        "obs_sha256": dataset_sha256(obs_path),
    }

    if manifest_path.exists():
        try:
            meta["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            meta["manifest"] = None
    if contract_path.exists():
        try:
            meta["feature_contract"] = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception:
            meta["feature_contract"] = None
    if quality_path.exists():
        try:
            meta["quality_summary"] = json.loads(quality_path.read_text(encoding="utf-8"))
        except Exception:
            meta["quality_summary"] = None

    return df, meta


def _select_regime_columns(df_regime: pd.DataFrame) -> list[str]:
    """Select admissible regime columns for downstream merge according to config."""
    mode = str(getattr(regime_cfg, "regime_feature_mode", "market_plus_regime")).lower()
    include_z = bool(getattr(regime_cfg, "include_z_hat_filter", False))

    base_cols: list[str] = []
    for c in sorted(df_regime.columns):
        if c in {"timestamp", "segment"}:
            continue
        if c.startswith("p_filter_"):
            base_cols.append(c)
        elif c.startswith("dp_filter_"):
            base_cols.append(c)
        elif c in {"p_max_filter", "p_margin_filter", "effective_num_states_filter", "expected_state_filter"}:
            base_cols.append(c)
        elif include_z and c == "z_hat_filter":
            base_cols.append(c)

    if not base_cols:
        raise ValueError("No admissible regime observation columns found in rl_observation_pack")

    if mode not in {"market_only", "regime_only", "market_plus_regime"}:
        raise ValueError(
            f"Invalid regime_feature_mode={mode!r}. Expected market_only, regime_only or market_plus_regime."
        )

    return base_cols



# --- Helper: Trim market dataframe to overlap with regime timestamps ---
def _trim_market_to_regime_overlap(
    df_market: pd.DataFrame,
    df_regime: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Trim market dataframe to the exact time support covered by the regime pack.

    This avoids failing strict exact-timestamp merges simply because the market dataset
    starts earlier (or ends later) than the exported causal regime observation pack.
    """
    if "timestamp" not in df_market.columns:
        raise ValueError("_trim_market_to_regime_overlap requires df_market['timestamp']")
    if "timestamp" not in df_regime.columns:
        raise ValueError("_trim_market_to_regime_overlap requires df_regime['timestamp']")

    left = df_market.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    left = left.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    left = left.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    regime_ts = pd.to_datetime(df_regime["timestamp"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    regime_ts = regime_ts.dropna()
    if regime_ts.empty:
        raise ValueError("Regime observation pack has no valid timestamps")

    regime_min_ts = regime_ts.min()
    regime_max_ts = regime_ts.max()

    before = int(len(left))
    mask = (left["timestamp"] >= regime_min_ts) & (left["timestamp"] <= regime_max_ts)
    left = left.loc[mask].reset_index(drop=True)
    after = int(len(left))

    meta = {
        "market_min_ts_before_trim": str(pd.to_datetime(df_market["timestamp"], errors="coerce").min()) if len(df_market) else None,
        "market_max_ts_before_trim": str(pd.to_datetime(df_market["timestamp"], errors="coerce").max()) if len(df_market) else None,
        "regime_min_ts": str(regime_min_ts),
        "regime_max_ts": str(regime_max_ts),
        "n_rows_market_before_trim": before,
        "n_rows_market_after_trim": after,
        "n_rows_dropped_pre_overlap": int(before - after),
    }
    return left, meta


def merge_market_and_regime_features(
    df_market: pd.DataFrame,
    *,
    strict_timestamp_merge: bool | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Merge market dataframe with causal regime observation pack.

    Contract:
      - merge key is exact UTC timestamp
      - no forward fill
      - join is inner by default to keep only timestamps present in both sources
      - returned feature list respects regime_cfg.regime_feature_mode
    """
    if "timestamp" not in df_market.columns:
        raise ValueError("merge_market_and_regime_features requires df_market['timestamp']")

    df_regime, regime_meta = load_regime_observation_pack()
    regime_cols = _select_regime_columns(df_regime)

    left_raw = df_market.copy()
    left_raw["timestamp"] = pd.to_datetime(left_raw["timestamp"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    left_raw = left_raw.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    left_raw = left_raw.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    left, overlap_meta = _trim_market_to_regime_overlap(left_raw, df_regime)

    right_cols = ["timestamp"] + regime_cols
    if "segment" in df_regime.columns:
        right_cols.append("segment")
    right = df_regime.loc[:, right_cols].copy()

    allow_inner_only = bool(getattr(regime_cfg, "allow_inner_join_only", True))
    strict_merge = bool(
        getattr(regime_cfg, "strict_timestamp_merge", True)
        if strict_timestamp_merge is None
        else strict_timestamp_merge
    )

    how = "inner" if allow_inner_only else "left"
    merged = left.merge(right, on="timestamp", how=how, validate="one_to_one")

    if strict_merge:
        if how == "inner" and len(merged) != len(left):
            missing = int(len(left) - len(merged))
            raise ValueError(
                f"Strict regime merge failed: {missing} market rows have no exact regime timestamp match"
            )
        if how == "left":
            missing_any = merged[regime_cols].isna().any(axis=1)
            if bool(missing_any.any()):
                n_missing = int(missing_any.sum())
                raise ValueError(
                    f"Strict regime merge failed: {n_missing} rows have missing regime features after left join"
                )

    mode = str(getattr(regime_cfg, "regime_feature_mode", "market_plus_regime")).lower()
    excluded_raw_market_cols = _raw_market_cols_to_exclude()
    market_feature_cols = [
        c for c in df_market.columns
        if c not in {"timestamp", "segment"}
        and c not in excluded_raw_market_cols
        and not (isinstance(c, str) and c.endswith("_fwd"))
    ]

    if mode == "market_only":
        feature_cols = list(market_feature_cols)
    elif mode == "regime_only":
        feature_cols = list(regime_cols)
    else:
        feature_cols = list(market_feature_cols) + list(regime_cols)

    merge_meta = {
        "regime_run_dir": regime_meta.get("run_dir"),
        "regime_obs_sha256": regime_meta.get("obs_sha256"),
        "n_rows_market": int(len(left)),
        "n_rows_regime": int(len(df_regime)),
        "n_rows_merged": int(len(merged)),
        "regime_feature_mode": mode,
        "regime_feature_cols": list(regime_cols),
        "market_feature_cols": list(market_feature_cols),
        "excluded_raw_market_cols": sorted(excluded_raw_market_cols),
        "feature_cols": list(feature_cols),
        "join_how": how,
        "strict_timestamp_merge": bool(strict_merge),
        **overlap_meta,
    }
    return merged, feature_cols, {**regime_meta, **merge_meta}


def load_ohlcv(path: str | Path | None = None) -> pd.DataFrame:
    """Load an OHLCV dataset from parquet/csv with audit-grade invariants.

    Invariants enforced:
      - timestamp is parsed as UTC, then made tz-naive (but representing UTC)
      - strict chronological sort
      - unique timestamps (drop_duplicates)
      - OHLCV columns exist and are finite floats (NaN/inf rows dropped)

    Notes:
      - Gap reporting (diff(timestamp) != expected) is optional and best-effort.
        If data_cfg.report_gaps is True and we can infer an expected timedelta, gaps are printed.
    """
    if bool(getattr(regime_cfg, "enabled", False)) and path is None:
        # Market OHLCV remains the base dataset. Regime features are merged downstream,
        # not substituted for the market file path.
        pass

    p = Path(path) if path else _default_dataset_path()
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # --- Timestamp parsing (UTC) ---
    time_col = getattr(data_cfg, "time_col", None)
    if time_col and time_col in df.columns:
        ts_raw = df[time_col]
    elif "timestamp" in df.columns:
        ts_raw = df["timestamp"]
    else:
        raise ValueError("No timestamp column (expected data_cfg.time_col or 'timestamp')")

    # Parse as UTC, then store tz-naive timestamps representing UTC
    ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    # Ensure UTC then drop tz info (tz-naive but UTC)
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.copy()
    df["timestamp"] = ts

    # Drop invalid timestamps early
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # --- Column contract: OHLCV ---
    ohlcv_cols_raw = getattr(data_cfg, "ohlcv_cols", None)
    if not ohlcv_cols_raw:
        raise ValueError("data_cfg.ohlcv_cols is missing/empty")

    # Normalize to a list of column names (pandas treats a tuple key as a single label)
    ohlcv_cols = list(ohlcv_cols_raw)

    missing = [c for c in ohlcv_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV col(s): {missing}")

    # Cast OHLCV to float, coerce invalid to NaN
    for c in ohlcv_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Drop non-finite rows (NaN/inf) on OHLCV
    mask_finite = np.isfinite(df.loc[:, ohlcv_cols].to_numpy()).all(axis=1)
    if not mask_finite.all():
        n_bad = int((~mask_finite).sum())
        df = df.loc[mask_finite].reset_index(drop=True)
        print(f"[data] Dropped {n_bad} row(s) with non-finite OHLCV values (NaN/inf)")

    # Sort strictly by timestamp and deduplicate
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Drop duplicate timestamps (keep first after sort)
    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"[data] Dropped {n_dupes} duplicate timestamp row(s)")

    # --- Strict time invariants (barrière) ---
    if not df["timestamp"].is_unique:
        raise ValueError("Timestamp column is not unique after deduplication")

    diffs = df["timestamp"].diff().dropna()
    if len(diffs) > 0 and not (diffs > pd.Timedelta(0)).all():
        # This should never happen after sort+dedupe; if it does, STOP.
        raise ValueError("Timestamp column is not strictly increasing after sort/dedup")

    # Optional: gap reporting
    if bool(getattr(data_cfg, "report_gaps", False)):
        expected = _infer_expected_timedelta()
        if expected is None:
            print("[data] Gap report requested but could not infer expected frequency from config.")
        else:
            diffs = df["timestamp"].diff()
            # Ignore first NaT
            gap_mask = diffs.notna() & (diffs != expected)
            n_gaps = int(gap_mask.sum())
            if n_gaps > 0:
                ex = df.loc[gap_mask, ["timestamp"]].head(5)
                print(f"[data] Detected {n_gaps} gap(s) where diff(timestamp) != {expected}. Examples:")
                print(ex.to_string(index=False))
            else:
                print(f"[data] Gap report: no gaps detected (expected diff {expected}).")

    return df


def split_train_val_test(df: pd.DataFrame):
    """Découpage train/val/test strictement chronologique.

    Anti-leakage (frontière) : si une colonne forward du type `log_ret_1_fwd` (shift(-1))
    est présente, alors la dernière observation d'un segment (train ou val) consomme
    mécaniquement un forward appartenant au segment suivant (transition t->t+1).

    Règle institutionnelle : après split, retirer la dernière ligne de train ET la dernière
    ligne de val si une colonne forward est présente, afin qu'aucune transition ne traverse
    les frontières train->val ou val->test.
    """
    if "timestamp" not in df.columns:
        raise ValueError("split_train_val_test expects a 'timestamp' column")

    # Barrière: df must already be time-ordered and unique by the time it reaches splitting
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("split_train_val_test requires df['timestamp'] to be sorted ascending")
    if not df["timestamp"].is_unique:
        raise ValueError("split_train_val_test requires df['timestamp'] to be unique")

    n = len(df)
    if n == 0:
        return df.copy(), df.copy(), df.copy()

    train_frac = float(getattr(data_cfg, "train_frac", 0.7))
    val_frac = float(getattr(data_cfg, "val_frac", 0.15))

    if not (0.0 < train_frac < 1.0) or not (0.0 < val_frac < 1.0) or (train_frac + val_frac) >= 1.0:
        raise ValueError(f"Invalid split fractions: train_frac={train_frac}, val_frac={val_frac} (must sum < 1)")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    if n_train <= 0 or n_val <= 0 or (n_train + n_val) >= n:
        raise ValueError(
            f"Split produces empty/invalid segments: n={n}, n_train={n_train}, n_val={n_val}. "
            "Adjust data_cfg.train_frac/val_frac or provide more data."
        )

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val :].reset_index(drop=True)

    # Anti-leakage frontière si un forward return est présent
    forward_col = _detect_forward_col(df)
    if forward_col is not None:
        if len(df_train) > 0:
            df_train = df_train.iloc[:-1].reset_index(drop=True)
        if len(df_val) > 0:
            df_val = df_val.iloc[:-1].reset_index(drop=True)

    return df_train, df_val, df_test


def load_training_dataframe(path: str | Path | None = None) -> tuple[pd.DataFrame, list[str], dict]:
    """Load the full training dataframe according to the active data contract.

    Pipeline
    --------
    1) load raw market OHLCV
    2) build causal engineered market features with add_features(...)
    3) if regime_cfg.enabled=False: return market-feature dataframe + engineered market feature list
    4) if regime_cfg.enabled=True: exact-timestamp merge market-feature dataframe with regime pack

    Returns
    -------
    df_train_input : pd.DataFrame
        Market-feature dataframe when regime is disabled, else merged market+regime dataframe.
    feature_cols : list[str]
        Observation feature list actually intended for the RL agent.
    meta : dict
        Audit metadata describing the load path and merge policy.
    """
    if str(getattr(data_cfg, "price_mode", "close")).lower() == "feature_dataset":
        p = Path(path) if path is not None else _default_dataset_path()
        if not p.exists():
            raise FileNotFoundError(p)

        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        time_col = getattr(data_cfg, "time_col", None)
        if time_col and time_col in df.columns:
            ts_raw = df[time_col]
        elif "timestamp" in df.columns:
            ts_raw = df["timestamp"]
            time_col = "timestamp"
        else:
            raise ValueError("No timestamp column (expected data_cfg.time_col or 'timestamp')")

        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.copy()
        df[time_col] = ts
        if time_col != "timestamp":
            df = df.rename(columns={time_col: "timestamp"})

        df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

        forward_col = _detect_forward_col(df)
        exclude = {"timestamp"}
        if forward_col is not None:
            exclude.add(forward_col)

        feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not feature_cols:
            raise ValueError("Feature dataset mode found no numeric feature columns to feed the agent")

        meta = {
            "regime_enabled": False,
            "data_path": str(p.resolve()),
            "price_mode": str(getattr(data_cfg, "price_mode", "feature_dataset")),
            "feature_cols": list(feature_cols),
            "forward_col": forward_col,
            "n_rows": int(len(df)),
            "min_ts": str(df["timestamp"].min()) if len(df) else None,
            "max_ts": str(df["timestamp"].max()) if len(df) else None,
        }
        return df, feature_cols, meta
    df_market_raw = load_ohlcv(path)

    # Build causal engineered market features first.
    # This is the market observation contract we want RL to consume.
    df_market_feat, market_feature_cols, market_build_meta = _build_market_feature_dataframe(df_market_raw)
    market_feature_cols = list(market_feature_cols)

    excluded_raw_market_cols = _raw_market_cols_to_exclude()

    base_meta = {
        "regime_enabled": bool(getattr(regime_cfg, "enabled", False)),
        "data_path": str(Path(path).resolve()) if path is not None else str(_default_dataset_path().resolve()),
        "excluded_raw_market_cols": sorted(excluded_raw_market_cols),
        "market_feature_cols": list(market_feature_cols),
        "n_rows_market_raw": int(len(df_market_raw)),
        "n_rows_market_feat": int(len(df_market_feat)),
        **market_build_meta,
    }

    if not bool(getattr(regime_cfg, "enabled", False)):
        meta = {
            **base_meta,
            "feature_cols": list(market_feature_cols),
            "n_rows": int(len(df_market_feat)),
        }
        return df_market_feat, market_feature_cols, meta

    df_merged, feature_cols, merge_meta = merge_market_and_regime_features(df_market_feat)
    meta = {
        **base_meta,
        "n_rows": int(len(df_merged)),
        "n_rows_dropped_pre_overlap": int(merge_meta.get("n_rows_dropped_pre_overlap", 0)),
        "regime_min_ts": merge_meta.get("regime_min_ts"),
        "regime_max_ts": merge_meta.get("regime_max_ts"),
        **merge_meta,
    }
    return df_merged, feature_cols, meta