# scripts/debug/diagnose_agent_dead.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repo root is importable when running as a script (fixes: ModuleNotFoundError: hc_ia_retail)
ROOT = Path(__file__).resolve().parents[2]  # HC_IA_RETAIL/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np

# Adapte ces imports à ton repo si besoin (selon ton journal ils existent)
from hc_ia_retail.data import load_ohlcv, split_train_val_test
from hc_ia_retail.features import add_features
from hc_ia_retail.env import RetailTradingEnv

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception as e:
    raise RuntimeError("stable-baselines3 requis pour ce script") from e


# --- petit wrapper fenêtre (copie le même que train.py si tu en as déjà un) ---
import gymnasium as gym
from collections import deque

class WindowStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, window: int):
        super().__init__(env)
        self.window = int(window)
        self.buf = deque(maxlen=self.window)

        low = np.repeat(self.env.observation_space.low, self.window, axis=0)
        high = np.repeat(self.env.observation_space.high, self.window, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buf.clear()
        for _ in range(self.window):
            self.buf.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.buf.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.concatenate(list(self.buf), axis=0).astype(np.float32)


def _info_get(info: dict, keys: list[str], default=np.nan):
    for k in keys:
        if k in info:
            return info[k]
    return default


def rollout(env, policy: str, steps: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    obs = env.reset()
    # SB3 VecEnv reset returns only obs
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, _ = obs
    # DummyVecEnv => reset returns obs only
    actions = []
    poss = []
    turnovers = []
    equities = []
    flat_count = 0

    for _ in range(steps):
        if policy == "random":
            a = rng.uniform(-1.0, 1.0, size=(1, env.action_space.shape[0]))
        elif policy == "flat":
            a = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)
        elif policy == "long":
            a = np.ones((1, env.action_space.shape[0]), dtype=np.float32)
        else:
            raise ValueError("policy must be: random|flat|long")

        obs, reward, done, info = env.step(a)
        if isinstance(obs, tuple) and len(obs) == 5:
            # gymnasium-style tuple slipped through
            obs, reward, terminated, truncated, info = obs
            done = np.logical_or(terminated, truncated)

        # VecEnv: info est une liste de dicts (1 env)
        inf = info[0] if isinstance(info, (list, tuple)) else info

        pos = _info_get(inf, ["pos", "position", "pos_prev", "pos_new"], default=np.nan)
        turnover = _info_get(inf, ["turnover", "to", "turnover_abs"], default=np.nan)
        equity = _info_get(inf, ["equity", "eq"], default=np.nan)

        actions.append(float(a[0, 0]))
        poss.append(float(pos) if pos is not None else np.nan)
        turnovers.append(float(turnover) if turnover is not None else np.nan)
        equities.append(float(equity) if equity is not None else np.nan)

        if np.isfinite(pos) and abs(pos) < 1e-3:
            flat_count += 1

        if done:
            obs = env.reset()

    actions = np.asarray(actions, dtype=float)
    poss = np.asarray(poss, dtype=float)
    turnovers = np.asarray(turnovers, dtype=float)
    equities = np.asarray(equities, dtype=float)

    out = {
        "policy": policy,
        "steps": steps,
        "mean_abs_action": float(np.nanmean(np.abs(actions))),
        "mean_abs_pos": float(np.nanmean(np.abs(poss))),
        "pct_flat": float(flat_count / max(1, steps)),
        "turnover_mean": float(np.nanmean(turnovers)),
        "equity_start": float(equities[0]) if len(equities) else np.nan,
        "equity_end": float(equities[-1]) if len(equities) else np.nan,
        "equity_std": float(np.nanstd(equities)),
        "pos_nan_pct": float(np.mean(~np.isfinite(poss))),
        "turnover_nan_pct": float(np.mean(~np.isfinite(turnovers))),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--vecnorm_path", type=str, default="")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", type=str, default="diagnose_agent_dead.json")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_ohlcv(args.data_path)
    df_feat, feature_cols = add_features(df)

    df_train, df_val, df_test = split_train_val_test(df_feat)

    # ⚠️ important : df_test doit contenir la colonne forward utilisée par l'env (si ton env l'attend)
    max_steps = int(args.max_steps)
    if max_steps <= 0:
        # auto: take a bounded slice so the rollout is fast and reproducible
        max_steps = min(int(args.steps), max(1000, len(df_test) - 1))

    env = RetailTradingEnv(df=df_test, feature_cols=feature_cols, max_steps=max_steps)
    env = WindowStackWrapper(env, window=args.window)

    venv = DummyVecEnv([lambda: env])

    if args.vecnorm_path:
        vn = VecNormalize.load(args.vecnorm_path, venv)
        vn.training = False
        vn.norm_reward = False
        venv = vn

    results = []
    for pol in ["flat", "long", "random"]:
        results.append(rollout(venv, pol, steps=args.steps, seed=args.seed))

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()