# tests/test_data_integrity.py
from __future__ import annotations

import numpy as np
import pandas as pd


def test_timestamp_exists(df_raw: pd.DataFrame):
    assert "timestamp" in df_raw.columns, "Colonne timestamp manquante."


def test_timestamp_sorted_strict(df_raw: pd.DataFrame):
    ts = pd.to_datetime(df_raw["timestamp"], utc=True, errors="coerce")
    assert ts.notna().all(), "Timestamps invalides (NaT)."
    # strictement croissant (pas de doublons ni retours en arrière)
    diffs = ts.astype("int64").diff().iloc[1:]
    assert (diffs > 0).all(), "Timestamp non strictement croissant (doublons ou désordre)."


def test_ohlcv_columns_present(df_raw: pd.DataFrame):
    needed = ["open", "high", "low", "close", "volume"]
    for c in needed:
        assert c in df_raw.columns, f"Colonne {c} manquante."


def test_ohlcv_numeric_and_finite(df_raw: pd.DataFrame):
    cols = ["open", "high", "low", "close", "volume"]
    x = df_raw[cols].astype(float)
    assert np.isfinite(x.to_numpy()).all(), "OHLCV contient NaN/Inf."
    assert (x["close"] > 0).all(), "close <= 0 détecté (problème data)."
    assert (x["high"] >= x["low"]).all(), "high < low détecté (problème data)."