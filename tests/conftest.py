# tests/conftest.py
from __future__ import annotations

import os
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from hc_ia_retail.config import data_cfg
from hc_ia_retail.data import load_ohlcv, split_train_val_test
from hc_ia_retail.features import add_features


@pytest.fixture(scope="session")
def df_raw() -> pd.DataFrame:
    """
    Charge le dataset officiel (par défaut via config) ou via env var HC_DATA_PATH.
    """
    data_path = os.environ.get("HC_DATA_PATH", "").strip() or None
    df = load_ohlcv(data_path)
    assert isinstance(df, pd.DataFrame) and len(df) > 100, "Dataset trop petit ou invalide."
    return df


@pytest.fixture(scope="session")
def df_feat_and_cols(df_raw):
    """
    Calcule les features et renvoie (df_feat, feature_cols).
    """
    df_feat, feature_cols = add_features(df_raw)
    assert len(feature_cols) > 0, "feature_cols vide."
    return df_feat, feature_cols


@pytest.fixture(scope="session")
def splits(df_feat_and_cols):
    """
    Split chronologique (train/val/test) sur df_feat.
    """
    df_feat, _ = df_feat_and_cols
    df_train, df_val, df_test = split_train_val_test(df_feat)
    assert len(df_train) > 100 and len(df_val) > 10 and len(df_test) > 10
    return df_train, df_val, df_test