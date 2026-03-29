# tests/test_no_forward_across_split.py
from __future__ import annotations

import numpy as np
import pandas as pd


def test_no_forward_crosses_split_boundary(df_raw):
    """
    Vérifie qu'aucune observation d'entraînement n'utilise un forward return (t->t+1)
    qui traverse la frontière train->val ou val->test.

    Hypothèse raisonnable: si log_ret_1_fwd est présent, il est défini via shift(-1).
    Donc l'index i utilise i+1 pour scorer le pas.
    """
    from hc_ia_retail.features import add_features
    from hc_ia_retail.data import split_train_val_test

    df_full, _ = add_features(df_raw)

    # Split sur df_full (tel que fait dans le pipeline actuel)
    df_train, df_val, df_test = split_train_val_test(df_full)

    # Si pas de colonne forward, rien à tester ici
    if "log_ret_1_fwd" not in df_full.columns:
        return

    # On travaille sur timestamps pour être explicite et audit-able
    t_train = pd.to_datetime(df_train["timestamp"], utc=True)
    t_val = pd.to_datetime(df_val["timestamp"], utc=True)
    t_test = pd.to_datetime(df_test["timestamp"], utc=True)

    # frontières
    assert t_train.max() < t_val.min(), "Split invalide: train et val se chevauchent."
    assert t_val.max() < t_test.min(), "Split invalide: val et test se chevauchent."

    # Condition anti-leakage forward :
    # pour tout i dans train, le pas i+1 doit rester dans train
    # -> équivalent à: max timestamp train doit être <= timestamp avant le premier de val
    # mais on le vérifie de façon opérationnelle:
    # - le dernier train ne doit pas être immédiatement suivi par un timestamp en validation.
    # (à 1h près dans ton cas, mais on ne suppose pas l'intervalle exact: on compare l'index dans df_full)

    # Reconstituer l'index global (position dans df_full trié)
    full_ts = pd.to_datetime(df_full["timestamp"], utc=True)
    full_ns = full_ts.astype("int64").to_numpy()

    # indices des lignes train dans df_full
    train_ns = t_train.astype("int64").to_numpy()
    idx_train = np.searchsorted(full_ns, train_ns)
    assert np.all(full_ns[idx_train] == train_ns), "Alignement timestamps train vs full impossible."

    # Pour chaque idx i en train, vérifier que i+1 n'est PAS en val/test
    # (on ne check pas tous pour le coût; on check la zone frontière: les derniers points du train)
    tail = idx_train[-10:]  # suffisant, c'est à la frontière que ça casse
    bad = []
    for i in tail:
        j = i + 1
        if j >= len(full_ns):
            continue
        ts_j = full_ts.iloc[j]
        if (ts_j >= t_val.min()) or (ts_j >= t_test.min()):
            bad.append((full_ts.iloc[i], ts_j))

    assert len(bad) == 0, (
        "Fuite expérimentale: une transition (t->t+1) du TRAIN pointe hors train "
        f"(exemples: {bad[:3]}). "
        "Mitigation: calculer log_ret_1_fwd APRES split, ou dropper la dernière ligne du train."
    )