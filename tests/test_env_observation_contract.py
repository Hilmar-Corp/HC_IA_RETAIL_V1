# tests/test_env_observation_contract.py
from __future__ import annotations

import numpy as np
import gymnasium as gym

from hc_ia_retail.env import RetailTradingEnv

def test_env_observation_dim(splits, df_feat_and_cols):
    _df_train, _df_val, df_test = splits
    _df_feat, feature_cols = df_feat_and_cols

    env = RetailTradingEnv(df_test, feature_cols=feature_cols, max_steps=200)
    obs, info = env.reset()

    assert obs.ndim == 1, "Observation doit être un vecteur 1D."
    expected = int(env.observation_space.shape[0])
    assert obs.shape[0] == expected, f"Dim obs attendue {expected}, reçu {obs.shape[0]}."


def test_env_obs_has_no_forward_leak(splits, df_feat_and_cols):
    _df_train, _df_val, df_test = splits
    df_feat, feature_cols = df_feat_and_cols

    assert "log_ret_1_fwd" not in feature_cols

    env = RetailTradingEnv(df_test, feature_cols=feature_cols, max_steps=50)
    obs, info = env.reset()

    # juste un filet: vérifier que l'env ne rajoute pas une colonne forward en dur
    assert obs.shape[0] == int(env.observation_space.shape[0])