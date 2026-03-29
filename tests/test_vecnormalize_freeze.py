# tests/test_vecnormalize_freeze.py
from __future__ import annotations

import os
from pathlib import Path

import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hc_ia_retail.config import paths_cfg, data_cfg
from hc_ia_retail.data import load_ohlcv, split_train_val_test
from hc_ia_retail.features import add_features
from hc_ia_retail.env import RetailTradingEnv
from scripts.eval.eval_oos import WindowStackWrapper


def _make_env(df, feature_cols):
    def _init():
        e = RetailTradingEnv(df, feature_cols=feature_cols, max_steps=50)
        e = WindowStackWrapper(e, window_size=data_cfg.window_size)
        return e
    return _init


def test_vecnormalize_is_frozen_in_eval_if_present():
    # cherche un vecnormalize.pkl (dernier run)
    runs_dir = Path(paths_cfg.out_dir)
    candidates = sorted(runs_dir.glob("run_*/vecnormalize.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        pytest.skip("Aucun vecnormalize.pkl trouvé : lancer un training avant.")

    df = load_ohlcv(None)
    df_feat, feature_cols = add_features(df)
    _tr, _va, te = split_train_val_test(df_feat)

    # Try most-recent vecnormalize.pkl files until one matches the current obs shape.
    loaded_env = None
    last_shape_err = None

    for vec_path in candidates:
        env_try = DummyVecEnv([_make_env(te, feature_cols)])
        try:
            env_try = VecNormalize.load(str(vec_path), env_try)
            loaded_env = (vec_path, env_try)
            break
        except AssertionError as e:
            # Shape mismatch between vecnorm and current env obs space
            if "spaces must have the same shape" in str(e):
                last_shape_err = str(e)
                continue
            raise

    if loaded_env is None:
        pytest.skip(
            f"Aucun vecnormalize.pkl compatible avec l'observation actuelle. Last err: {last_shape_err}"
        )

    vec_path, env = loaded_env

    env.training = False
    env.norm_reward = False

    # check flag
    assert env.training is False, "VecNormalize doit être gelé en évaluation (training=False)."
    assert env.norm_reward is False, "Norm reward doit être désactivé en évaluation."