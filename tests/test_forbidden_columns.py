# tests/test_forbidden_columns.py
from __future__ import annotations

import re


def test_forbidden_feature_column_names(df_feat_and_cols):
    """
    Empêche l'introduction accidentelle de colonnes de fuite (forward/label/target)
    dans le vecteur d'observation.
    """
    _, feature_cols = df_feat_and_cols

    # patterns interdits (ajoute/retire selon tes conventions)
    forbidden_patterns = [
        r"fwd", r"forward", r"future",
        r"label", r"target", r"y_", r"ground_truth",
        r"next_", r"t\+1",
        r"reward", r"pnl", r"equity",  # ces grandeurs doivent rester dans env/info, pas dans features
    ]

    bad = []
    for c in feature_cols:
        cl = c.lower()
        for pat in forbidden_patterns:
            if re.search(pat, cl):
                bad.append((c, pat))
                break

    assert not bad, (
        "Colonnes interdites détectées dans feature_cols (risque de leakage / triche). "
        f"Exemples: {bad[:8]}"
    )