# tests/test_split_no_leakage.py
from __future__ import annotations

import pandas as pd


def _minmax_ts(df: pd.DataFrame):
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return ts.min(), ts.max()


def test_split_chronological_no_overlap(splits):
    df_train, df_val, df_test = splits

    tr_min, tr_max = _minmax_ts(df_train)
    va_min, va_max = _minmax_ts(df_val)
    te_min, te_max = _minmax_ts(df_test)

    assert tr_max < va_min, "Frontière train→val violée (chevauchement temporel)."
    assert va_max < te_min, "Frontière val→test violée (chevauchement temporel)."


def test_split_disjoint_timestamps(splits):
    df_train, df_val, df_test = splits

    tr = set(pd.to_datetime(df_train["timestamp"], utc=True))
    va = set(pd.to_datetime(df_val["timestamp"], utc=True))
    te = set(pd.to_datetime(df_test["timestamp"], utc=True))

    assert tr.isdisjoint(va), "Train et Val partagent des timestamps."
    assert tr.isdisjoint(te), "Train et Test partagent des timestamps."
    assert va.isdisjoint(te), "Val et Test partagent des timestamps."