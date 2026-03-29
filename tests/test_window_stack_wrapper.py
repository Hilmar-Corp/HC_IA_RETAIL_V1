# tests/test_window_stack_wrapper.py
from __future__ import annotations

import numpy as np
import gymnasium as gym

from hc_ia_retail.env import RetailTradingEnv
from scripts.eval.eval_oos import WindowStackWrapper  # réutilise la classe existante


def test_window_stack_shapes(splits, df_feat_and_cols):
    _df_train, _df_val, df_test = splits
    _df_feat, feature_cols = df_feat_and_cols

    base = RetailTradingEnv(df_test, feature_cols=feature_cols, max_steps=10)
    W = 8
    env = WindowStackWrapper(base, window_size=W)

    obs, info = env.reset()
    per_step = int(base.observation_space.shape[0])
    assert obs.shape == (per_step * W,)

    obs2, r, done, trunc, info2 = env.step(np.array([0.0], dtype=np.float32))
    assert obs2.shape == (per_step * W,)


def test_window_stack_is_causal_on_reset(splits, df_feat_and_cols):
    _df_train, _df_val, df_test = splits
    _df_feat, feature_cols = df_feat_and_cols

    base = RetailTradingEnv(df_test, feature_cols=feature_cols, max_steps=10)
    W = 4
    env = WindowStackWrapper(base, window_size=W)
    obs, info = env.reset()

    per_step = int(base.observation_space.shape[0])
    stacked = obs.reshape(W, per_step)

    # au reset, toutes les lignes doivent être identiques (s0 répété)
    for i in range(1, W):
        assert np.allclose(stacked[i], stacked[0], atol=1e-12), "Reset stacking non conforme (buffer init)."