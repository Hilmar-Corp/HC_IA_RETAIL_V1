# tests/test_features_causality.py
from __future__ import annotations

import numpy as np
import pandas as pd


def test_forward_column_not_in_features(df_feat_and_cols):
    df_feat, feature_cols = df_feat_and_cols
    assert "log_ret_1_fwd" not in feature_cols, "Fuite majeure: log_ret_1_fwd dans feature_cols."


def test_features_defined_only_past(df_raw):
    """
    Test de causalité (heuristique) :
    Les features à un timestamp ts0 ne doivent pas changer si on enlève le futur.
    Note: add_features() construit aussi log_ret_1_fwd via shift(-1) et drop la dernière ligne.
    Pour pouvoir comparer à ts0, on inclut ts0+1 dans le préfixe, puis on compare à ts0.
    """
    import numpy as np
    import pandas as pd
    from hc_ia_retail.features import add_features

    # Calcul full (référence)
    df_full, feature_cols = add_features(df_raw)

    # Choisir un point après warmup et pas trop proche de la fin
    t0 = min(500, len(df_full) // 2)
    ts0 = pd.to_datetime(df_full["timestamp"].iloc[t0], utc=True)

    # Retrouver l'index de ts0 dans le RAW (robuste tz-aware / dtype)
    raw_ts = pd.to_datetime(df_raw["timestamp"], utc=True)
    raw_ns = raw_ts.astype("int64")  # nanosecondes depuis epoch
    ts0_ns = ts0.value             # nanosecondes depuis epoch

    idx0 = int(np.searchsorted(raw_ns, ts0_ns))
    assert idx0 < len(raw_ns) and raw_ns[idx0] == ts0_ns, (
        "ts0 introuvable dans df_raw (conversion timezone/dtype ou dataset incohérent)."
    )

    # IMPORTANT: inclure une bougie supplémentaire (ts0+1) pour éviter que add_features drop ts0
    assert idx0 + 1 < len(df_raw), "ts0 trop proche de la fin du dataset raw."
    df_prefix_raw = df_raw.iloc[: idx0 + 2].copy()

    # Recompute sur préfixe
    df_pref, cols2 = add_features(df_prefix_raw)
    assert list(feature_cols) == list(cols2), "feature_cols instable entre full et prefix."

    # Aligner exactement sur ts0
    full_ts = pd.to_datetime(df_full["timestamp"], utc=True)
    pref_ts = pd.to_datetime(df_pref["timestamp"], utc=True)

    assert (pref_ts == ts0).any(), "ts0 absent dans df_pref (problème de warmup/dropna inattendu)."

    row_full = df_full.loc[full_ts == ts0].iloc[0]
    row_pref = df_pref.loc[pref_ts == ts0].iloc[0]

    a = row_full[feature_cols].to_numpy(dtype=float)
    b = row_pref[feature_cols].to_numpy(dtype=float)

    diff = np.nan_to_num(a - b, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.max(np.abs(diff)) < 1e-8, (
        "Suspicion look-ahead: features diffèrent entre full et prefix au même timestamp."
    )

def test_forward_shift_is_consistent_if_present(df_feat_and_cols):
    df_feat, _ = df_feat_and_cols
    if "log_ret_1" in df_feat.columns and "log_ret_1_fwd" in df_feat.columns:
        a = df_feat["log_ret_1"].shift(-1).to_numpy()
        b = df_feat["log_ret_1_fwd"].to_numpy()
        # ignore last (NaN)
        mask = np.isfinite(a[:-1]) & np.isfinite(b[:-1])
        assert np.allclose(a[:-1][mask], b[:-1][mask], atol=1e-10), "log_ret_1_fwd mal défini."