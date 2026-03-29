from __future__ import annotations

import argparse
import json
import sys
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hc_ia_retail.run_registry import append_run_index, safe_float  # noqa: E402

from hc_ia_retail.config import data_cfg, env_cfg, paths_cfg, regime_cfg  # noqa: E402
from hc_ia_retail.data import load_training_dataframe, split_train_val_test  # noqa: E402
from hc_ia_retail.env import RetailTradingEnv  # noqa: E402


# ----------------- JSON helper (avoid Path / numpy types) -----------------
def json_sanitize(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    return obj


# ----------------- Robust info getter (fix "agent mort" due to wrong keys) -----------------
def info_get(info: dict, keys: list[str], default=np.nan):
    for k in keys:
        if k in info and info[k] is not None:
            return info[k]
    return default


# ----------------- North Star score (run comparator) -----------------
def compute_north_star_score(
    sharpe: float,
    maxdd: float,
    turnover_mean: float,
    d: float = 0.20,
    tau: float = 0.01,
    lambda_dd: float = 2.0,
    lambda_to: float = 1.0,
) -> float:
    """
    North Star score (single scalar) for comparing runs.

    score = sharpe - lambda_dd*max(0, maxdd - d) - lambda_to*max(0, turnover_mean - tau)

    Notes:
    - maxdd is in [0, 1] (e.g., 0.20 = 20% drawdown)
    - turnover_mean is a proxy (here mean(|Δpos|)), unitless per step
    """
    sharpe = float(sharpe)
    maxdd = float(maxdd)
    turnover_mean = float(turnover_mean)
    dd_pen = max(0.0, maxdd - float(d))
    to_pen = max(0.0, turnover_mean - float(tau))
    return float(sharpe - float(lambda_dd) * dd_pen - float(lambda_to) * to_pen)


# ----------------- Regime-aware data build -----------------
def build_eval_dataframe(data_path: str | None = None) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """
    Build evaluation dataframe using the exact same causal pipeline as training.

    Compatible with both versions of load_training_dataframe:
      - load_training_dataframe(data_path_arg=...)
      - load_training_dataframe(...)
      - load_training_dataframe() with no override support
    """
    data_path_clean = str(data_path).strip() if data_path else None

    try:
        if data_path_clean:
            df_final, feature_cols, data_source_meta = load_training_dataframe(data_path_arg=data_path_clean)
        else:
            df_final, feature_cols, data_source_meta = load_training_dataframe()
    except TypeError:
        try:
            if data_path_clean:
                df_final, feature_cols, data_source_meta = load_training_dataframe(data_path_clean)
            else:
                df_final, feature_cols, data_source_meta = load_training_dataframe()
        except TypeError:
            if data_path_clean:
                raise TypeError(
                    "Current load_training_dataframe() implementation does not accept a data_path override. "
                    "Either remove --data_path / data_path usage or update hc_ia_retail.data.load_training_dataframe()."
                )
            df_final, feature_cols, data_source_meta = load_training_dataframe()

    return df_final, list(feature_cols), dict(data_source_meta)


# ----------------- Wrapper: stack last window_size obs -----------------
class WindowStackWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, window_size: int):
        super().__init__(env)
        self.window_size = int(window_size)
        self.buf: list[np.ndarray] = []
        orig = env.observation_space
        assert isinstance(orig, gym.spaces.Box)
        self.per_step = int(orig.shape[0])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.per_step * self.window_size,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buf = [obs.copy() for _ in range(self.window_size)]
        return self.observation(obs), info

    def observation(self, obs):
        self.buf.append(obs.copy())
        self.buf = self.buf[-self.window_size:]
        return np.concatenate(self.buf, axis=0).astype(np.float32)


def make_env(df, feature_cols):
    def _init():
        e = RetailTradingEnv(df, feature_cols=feature_cols, max_steps=None)
        e = WindowStackWrapper(e, window_size=data_cfg.window_size)
        return e

    return _init


# ----------------- Metrics helpers -----------------
def compute_equity_metrics(equity: np.ndarray, freq_per_year: float = 24 * 365) -> dict:
    equity = np.asarray(equity, dtype=np.float64)
    if len(equity) < 3:
        return {"sharpe": 0.0, "max_dd": 0.0, "cumulative_return": 0.0, "ann_return": 0.0, "ann_vol": 0.0}

    log_eq = np.log(equity + 1e-12)
    rets = np.diff(log_eq)

    mu = float(rets.mean())
    sigma = float(rets.std() + 1e-12)
    sharpe = float((mu / sigma) * np.sqrt(freq_per_year))

    ann_vol = float(sigma * np.sqrt(freq_per_year))
    ann_ret = float(np.exp(mu * freq_per_year) - 1.0)
    cum_ret = float(equity[-1] / equity[0] - 1.0)

    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    return {
        "sharpe": sharpe,
        "max_dd": float(max_dd),
        "cumulative_return": cum_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
    }


def _to_numpy_finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def compute_drawdown_series(equity: np.ndarray) -> tuple[np.ndarray, float]:
    eq = _to_numpy_finite(equity)
    if eq.size == 0:
        return np.array([], dtype=np.float64), 0.0
    peak = np.maximum.accumulate(eq)
    peak = np.where(peak <= 0, 1e-12, peak)
    dd = 1.0 - (eq / peak)
    max_dd = float(np.max(dd)) if dd.size else 0.0
    return dd, max_dd


def compute_returns(equity: np.ndarray) -> np.ndarray:
    eq = _to_numpy_finite(equity)
    if eq.size < 2:
        return np.array([], dtype=np.float64)
    log_eq = np.log(eq + 1e-12)
    return np.diff(log_eq)


def compute_advanced_metrics(
    equity: np.ndarray,
    freq_per_year: float = 24 * 365,
    rf: float = 0.0,
) -> dict:
    eq = _to_numpy_finite(equity)
    base = compute_equity_metrics(eq, freq_per_year=freq_per_year)
    rets = compute_returns(eq)

    if rets.size == 0:
        out = dict(base)
        out.update(
            {
                "sortino": 0.0,
                "calmar": 0.0,
                "skew": 0.0,
                "kurtosis": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "var_99": 0.0,
                "cvar_99": 0.0,
                "tail_ratio": 0.0,
            }
        )
        return out

    downside = rets[rets < 0]
    dd = float(np.sqrt(np.mean(downside**2))) if downside.size else 0.0
    mu = float(rets.mean())
    sortino = float((mu / (dd + 1e-12)) * np.sqrt(freq_per_year))

    _dd_series, max_dd = compute_drawdown_series(eq)
    calmar = float(base["ann_return"] / (max_dd + 1e-12))

    centered = rets - mu
    m2 = float(np.mean(centered**2) + 1e-12)
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = float(m3 / (m2**1.5))
    kurt = float(m4 / (m2**2) - 3.0)

    q = float(np.quantile(rets, 0.05))
    tail = rets[rets <= q]
    cvar = float(tail.mean()) if tail.size else q

    q99 = float(np.quantile(rets, 0.01))
    tail99 = rets[rets <= q99]
    cvar99 = float(tail99.mean()) if tail99.size else q99

    top5 = rets[rets >= np.quantile(rets, 0.95)]
    bot5 = rets[rets <= np.quantile(rets, 0.05)]
    eps = 1e-12
    if top5.size > 0 and bot5.size > 0:
        tail_ratio = abs(np.mean(top5)) / (abs(np.mean(bot5)) + eps)
    else:
        tail_ratio = 0.0

    out = dict(base)
    out.update(
        {
            "sortino": sortino,
            "calmar": calmar,
            "skew": skew,
            "kurtosis": kurt,
            "var_95": q,
            "cvar_95": cvar,
            "var_99": q99,
            "cvar_99": cvar99,
            "tail_ratio": tail_ratio,
        }
    )
    return out


def plot_equity(x, eq_sac, eq_bh, eq_bh_vt, eq_naive, outpath: Path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, eq_sac, label="Equity SAC OOS")
    ax.plot(x, eq_bh, label="Equity Buy & Hold OOS", linestyle="--")
    if eq_bh_vt is not None:
        ax.plot(x, eq_bh_vt, label="Equity Buy & Hold Vol-Target OOS", linestyle=":")
    if eq_naive is not None:
        ax.plot(x, eq_naive, label="Equity Naive baseline OOS", linestyle="-.")
    ax.set_title("HC_IA_RETAIL — Equity OOS (SAC vs baselines)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_drawdown(x, equity, outpath: Path):
    dd, _ = compute_drawdown_series(equity)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, dd, label="Drawdown")
    ax.set_title("HC_IA_RETAIL — Drawdown (OOS)")
    ax.set_ylim(0, max(0.01, float(np.nanmax(dd)) * 1.05))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_underwater(x, equity, outpath: Path):
    eq = _to_numpy_finite(equity)
    peak = np.maximum.accumulate(eq)
    uw = (eq / np.where(peak <= 0, 1e-12, peak)) - 1.0
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, uw, label="Underwater")
    ax.axhline(0.0)
    ax.set_title("HC_IA_RETAIL — Underwater plot (OOS)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_positions(x, pos, outpath: Path):
    pos = _to_numpy_finite(pos)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, pos, label="Position")
    ax.set_title("HC_IA_RETAIL — Position (OOS)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_turnover(x, pos, outpath: Path):
    pos = _to_numpy_finite(pos)
    if pos.size < 2:
        turn = np.zeros_like(pos)
        x2 = x
    else:
        turn = np.abs(np.diff(pos))
        x2 = x[1:] if hasattr(x, "__len__") and len(x) == len(pos) else np.arange(turn.size)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x2, turn, label="|Δpos|")
    ax.set_title("HC_IA_RETAIL — Turnover proxy |Δposition| (OOS)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_returns_hist(rets, outpath: Path):
    r = _to_numpy_finite(rets)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.hist(r, bins=80)
    ax.set_title("HC_IA_RETAIL — Returns distribution (log-returns)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_qq(rets, outpath: Path):
    r = _to_numpy_finite(rets)
    if r.size < 10:
        return

    r_sorted = np.sort(r)
    n = r_sorted.size
    p = (np.arange(1, n + 1) - 0.5) / n

    from numpy import sqrt

    try:
        from numpy.special import erfinv  # type: ignore
    except Exception:
        try:
            from scipy.special import erfinv  # type: ignore
        except Exception:
            print("[WARN] erfinv unavailable (no numpy.special.erfinv / scipy). Skipping QQ plot.")
            return

    z = sqrt(2.0) * erfinv(2.0 * p - 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(z, r_sorted, s=8)

    a = float(np.median(r_sorted) - np.median(z) * (np.std(r_sorted) / (np.std(z) + 1e-12)))
    b = float(np.std(r_sorted) / (np.std(z) + 1e-12))
    ax.plot(z, a + b * z)

    ax.set_title("HC_IA_RETAIL — QQ plot (Normal)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def rolling_sharpe(rets: np.ndarray, window: int, freq_per_year: float) -> np.ndarray:
    r = _to_numpy_finite(rets)
    if r.size < window + 2:
        return np.array([], dtype=np.float64)
    out = np.full(r.size, np.nan, dtype=np.float64)
    for i in range(window, r.size + 1):
        w = r[i - window : i]
        mu = w.mean()
        sig = w.std() + 1e-12
        out[i - 1] = (mu / sig) * np.sqrt(freq_per_year)
    return out


def plot_rolling_sharpe(x, rets, outpath: Path, window: int = 24 * 30, freq_per_year: float = 24 * 365):
    rs = rolling_sharpe(rets, window=window, freq_per_year=freq_per_year)
    if rs.size == 0:
        return
    x_r = x[1:] if hasattr(x, "__len__") and len(x) >= (len(rs) + 1) else np.arange(rs.size)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x_r, rs, label=f"Rolling Sharpe (window={window})")
    ax.set_title("HC_IA_RETAIL — Rolling Sharpe (OOS)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def compute_buyhold_equity(df_test: pd.DataFrame, n_points: int, initial: float = 1.0) -> np.ndarray:
    n = int(n_points)
    if n <= 0:
        return np.array([], dtype=np.float64)

    # Feature-dataset path: buy&hold equity can be reconstructed directly from
    # forward log-returns when raw close prices are not available.
    if "log_ret_1_fwd" in df_test.columns:
        lr = pd.to_numeric(df_test["log_ret_1_fwd"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)[:n]
        lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
        return float(initial) * np.exp(np.cumsum(lr))
    if "log_ret_1" in df_test.columns:
        lr = pd.to_numeric(df_test["log_ret_1"], errors="coerce").shift(-1).fillna(0.0).to_numpy(dtype=np.float64)[:n]
        lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
        return float(initial) * np.exp(np.cumsum(lr))

    if "close" not in df_test.columns:
        return np.full(n, np.nan, dtype=np.float64)

    close = df_test["close"].astype(float).to_numpy()

    if close.size >= n + 1:
        bh = np.zeros(n, dtype=np.float64)
        eq = float(initial)
        for i in range(n):
            prev = close[i]
            cur = close[i + 1]
            if prev <= 0 or (not np.isfinite(prev)) or (not np.isfinite(cur)):
                pass
            else:
                eq *= float(cur / prev)
            bh[i] = eq
        return bh

    close = close[:n]
    bh = np.ones(n, dtype=np.float64) * float(initial)
    for i in range(1, n):
        prev = close[i - 1]
        cur = close[i]
        if prev <= 0 or not np.isfinite(prev) or not np.isfinite(cur):
            bh[i] = bh[i - 1]
        else:
            bh[i] = bh[i - 1] * (cur / prev)
    return bh


def compute_naive_position(df_test: pd.DataFrame, n_points: int) -> np.ndarray:
    """Naive baseline position rule using available market features only."""
    L = float(getattr(env_cfg, "max_leverage", 1.0))
    n = int(n_points)
    if n <= 0:
        return np.array([], dtype=np.float64)

    # Support both legacy raw-feature names and the new V1 feature dataset names.
    ma = None
    if "ma_ratio" in df_test.columns:
        ma = pd.to_numeric(df_test["ma_ratio"], errors="coerce").to_numpy(dtype=np.float64)[:n]
    elif "ma_ratio_24" in df_test.columns:
        ma = pd.to_numeric(df_test["ma_ratio_24"], errors="coerce").to_numpy(dtype=np.float64)[:n]

    rsi = pd.to_numeric(df_test["rsi"], errors="coerce").to_numpy(dtype=np.float64)[:n] if "rsi" in df_test.columns else None
    mom24 = pd.to_numeric(df_test["log_ret_24"], errors="coerce").to_numpy(dtype=np.float64)[:n] if "log_ret_24" in df_test.columns else None

    pos = np.zeros(n, dtype=np.float64)

    # Legacy rule when both ma_ratio and rsi are available.
    if ma is not None and rsi is not None:
        # Legacy ma_ratio uses ratio around 1.0. V1 ma_ratio_24 is centered around 0.0.
        if "ma_ratio" in df_test.columns:
            long_mask = (ma > 1.0) & (rsi >= 50.0)
            short_mask = (ma < 1.0) & (rsi <= 50.0)
        else:
            long_mask = (ma > 0.0) & (rsi >= 50.0)
            short_mask = (ma < 0.0) & (rsi <= 50.0)
        pos[long_mask] = 1.0
        pos[short_mask] = -1.0
        return np.clip(pos * L, -L, L)

    # V1 fallback: simple continuous sign baseline from trend/momentum only.
    if ma is not None:
        anchor = ma
    elif mom24 is not None:
        anchor = mom24
    else:
        return pos

    anchor = np.nan_to_num(anchor, nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.sign(anchor)
    return np.clip(pos * L, -L, L)


def compute_equity_from_positions(
    df_test: pd.DataFrame,
    positions: np.ndarray,
    initial: float,
    fee_bps: float,
    spread_bps: float,
) -> tuple[np.ndarray, dict]:
    pos = _to_numpy_finite(positions).astype(np.float64)
    n = int(pos.size)
    if n == 0:
        return (
            np.array([], dtype=np.float64),
            {
                "turnover": np.array([], dtype=np.float64),
                "cost": np.array([], dtype=np.float64),
                "pnl_log_gross": np.array([], dtype=np.float64),
                "pnl_log_net": np.array([], dtype=np.float64),
            },
        )

    if "log_ret_1_fwd" in df_test.columns:
        ret_fwd = df_test["log_ret_1_fwd"].astype(float).to_numpy()[:n]
    elif "log_ret_1" in df_test.columns:
        ret_fwd = pd.to_numeric(df_test["log_ret_1"], errors="coerce").shift(-1).fillna(0.0).to_numpy(dtype=np.float64)[:n]
    else:
        close = df_test["close"].astype(float).to_numpy() if "close" in df_test.columns else np.array([], dtype=float)
        ret_fwd = np.zeros(n, dtype=np.float64)
        if close.size >= n + 1:
            for i in range(n):
                prev = close[i]
                cur = close[i + 1]
                if prev > 0 and np.isfinite(prev) and np.isfinite(cur):
                    ret_fwd[i] = float(np.log(cur / prev))
                else:
                    ret_fwd[i] = 0.0
        else:
            ret_fwd[:] = 0.0

    ret_fwd = _to_numpy_finite(ret_fwd)

    k_cost = (float(fee_bps) + float(spread_bps)) / 1e4
    p_prev = 0.0

    turnover = np.zeros(n, dtype=np.float64)
    cost = np.zeros(n, dtype=np.float64)
    pnl_gross = np.zeros(n, dtype=np.float64)
    pnl_net = np.zeros(n, dtype=np.float64)

    eq = float(initial)
    eq_plot = np.zeros(n, dtype=np.float64)

    for i in range(n):
        p_t = float(pos[i])
        to = abs(p_t - p_prev)
        c = k_cost * to
        pg = p_t * float(ret_fwd[i])
        pn = pg - c

        eq = eq * float(np.exp(pn))
        eq_plot[i] = eq

        turnover[i] = to
        cost[i] = c
        pnl_gross[i] = pg
        pnl_net[i] = pn

        p_prev = p_t

    details = {
        "turnover": turnover,
        "cost": cost,
        "pnl_log_gross": pnl_gross,
        "pnl_log_net": pnl_net,
    }
    return eq_plot, details


# ----------------- Institutional OOS helpers -----------------
def _ensure_datetime_index(ts_like) -> pd.DatetimeIndex:
    s = pd.to_datetime(ts_like, utc=True, errors="coerce")
    return pd.DatetimeIndex(s)


def compute_monthly_table(timestamps, equity: np.ndarray) -> pd.DataFrame:
    eq = _to_numpy_finite(equity)
    if eq.size < 3:
        return pd.DataFrame(columns=["month", "return", "vol", "sharpe", "max_dd"])

    idx = _ensure_datetime_index(timestamps)
    df = pd.DataFrame({"equity": eq}, index=idx)

    eq_m = df["equity"].resample("ME").last().dropna()
    if len(eq_m) < 2:
        return pd.DataFrame(columns=["month", "return", "vol", "sharpe", "max_dd"])

    ret_m = eq_m.pct_change().dropna()
    logret = np.log(df["equity"].clip(lower=1e-12)).diff().dropna()

    rows = []
    for month_end, r in ret_m.items():
        month_start = (month_end - pd.offsets.MonthEnd(1)) + pd.Timedelta(seconds=1)
        lr = logret.loc[(logret.index > month_start) & (logret.index <= month_end)]
        if len(lr) >= 5:
            mu = float(lr.mean())
            sig = float(lr.std() + 1e-12)
            freq = float(getattr(data_cfg, "freq_per_year", 24 * 365))
            sharpe = (mu / sig) * math.sqrt(freq)
            vol = sig * math.sqrt(freq)
        else:
            sharpe = 0.0
            vol = 0.0

        eq_month = df.loc[(df.index > month_start) & (df.index <= month_end), "equity"].to_numpy(dtype=float)
        _, mdd = compute_drawdown_series(eq_month)

        rows.append(
            {
                "month": month_end.strftime("%Y-%m"),
                "return": float(r),
                "vol": float(vol),
                "sharpe": float(sharpe),
                "max_dd": float(mdd),
            }
        )

    return pd.DataFrame(rows)


def plot_monthly_heatmap(monthly_df: pd.DataFrame, outpath: Path, title: str = "Monthly returns heatmap"):
    if monthly_df is None or monthly_df.empty:
        return
    tmp = monthly_df.copy()
    tmp["year"] = tmp["month"].str.slice(0, 4).astype(int)
    tmp["m"] = tmp["month"].str.slice(5, 7).astype(int)
    pivot = tmp.pivot(index="year", columns="m", values="return").sort_index()

    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto")

    ax.set_title(title)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.to_list())
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    arr = pivot.to_numpy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v*100:.1f}%", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def compute_drawdown_events(timestamps, equity: np.ndarray) -> pd.DataFrame:
    eq = _to_numpy_finite(equity)
    if eq.size < 3:
        return pd.DataFrame(columns=["start", "trough", "recovery", "depth", "duration_bars", "recovery_bars"])

    idx = _ensure_datetime_index(timestamps)
    dd, _ = compute_drawdown_series(eq)

    events = []
    in_dd = False
    start_i = 0
    trough_i = 0
    trough_dd = 0.0

    for i in range(len(dd)):
        if not in_dd and dd[i] > 0:
            in_dd = True
            start_i = i
            trough_i = i
            trough_dd = dd[i]
        if in_dd:
            if dd[i] >= trough_dd:
                trough_dd = dd[i]
                trough_i = i
            if dd[i] <= 1e-12:
                recovery_i = i
                events.append(
                    {
                        "start": str(idx[start_i]),
                        "trough": str(idx[trough_i]),
                        "recovery": str(idx[recovery_i]),
                        "depth": float(trough_dd),
                        "duration_bars": int(trough_i - start_i + 1),
                        "recovery_bars": int(recovery_i - trough_i + 1),
                    }
                )
                in_dd = False

    if in_dd:
        events.append(
            {
                "start": str(idx[start_i]),
                "trough": str(idx[trough_i]),
                "recovery": None,
                "depth": float(trough_dd),
                "duration_bars": int(trough_i - start_i + 1),
                "recovery_bars": None,
            }
        )

    return pd.DataFrame(events).sort_values("depth", ascending=False)


def compute_trade_blotter(
    timestamps, equity: np.ndarray, position: np.ndarray, eps: float = 1e-3
) -> tuple[pd.DataFrame, dict]:
    eq = _to_numpy_finite(equity)
    pos = _to_numpy_finite(position)
    idx = _ensure_datetime_index(timestamps)

    n = min(len(eq), len(pos), len(idx))
    eq = eq[:n]
    pos = pos[:n]
    idx = idx[:n]

    rebalance_eps = 0.02

    def _sign(x: float) -> int:
        if x > eps:
            return 1
        if x < -eps:
            return -1
        return 0

    trades: list[dict[str, Any]] = []
    open_trade: dict[str, Any] | None = None
    cur_sign = 0
    rebalance_events: list[dict[str, Any]] = []
    prev_pos = float(pos[0]) if n else 0.0

    for i in range(n):
        s = _sign(float(pos[i]))

        if i > 0:
            dpos = float(pos[i] - prev_pos)
            if abs(dpos) >= rebalance_eps:
                rebalance_events.append(
                    {
                        "time": str(idx[i]),
                        "i": int(i),
                        "type": "rebalance",
                        "pos_prev": float(prev_pos),
                        "pos": float(pos[i]),
                        "dpos": float(dpos),
                    }
                )
            prev_pos = float(pos[i])

        if open_trade is None:
            if s != 0:
                open_trade = {
                    "entry_time": str(idx[i]),
                    "entry_i": int(i),
                    "side": "long" if s > 0 else "short",
                    "entry_equity": float(eq[i]),
                    "max_fav": float(eq[i]),
                    "max_adv": float(eq[i]),
                    "entry_pos": float(pos[i]),
                }
                cur_sign = s
        else:
            open_trade["max_fav"] = float(max(open_trade["max_fav"], float(eq[i])))
            open_trade["max_adv"] = float(min(open_trade["max_adv"], float(eq[i])))

            if s == 0 or (s != 0 and s != cur_sign):
                entry_i = int(open_trade["entry_i"])
                exit_i = int(i)
                pnl = float(eq[exit_i] - float(open_trade["entry_equity"]))
                ret = float(eq[exit_i] / (float(open_trade["entry_equity"]) + 1e-12) - 1.0)
                mfe = float(open_trade["max_fav"] / (float(open_trade["entry_equity"]) + 1e-12) - 1.0)
                mae = float(open_trade["max_adv"] / (float(open_trade["entry_equity"]) + 1e-12) - 1.0)

                trades.append(
                    {
                        "type": "directional",
                        "entry_time": open_trade["entry_time"],
                        "exit_time": str(idx[exit_i]),
                        "side": open_trade["side"],
                        "duration_bars": int(exit_i - entry_i + 1),
                        "entry_equity": float(open_trade["entry_equity"]),
                        "exit_equity": float(eq[exit_i]),
                        "pnl": float(pnl),
                        "return": float(ret),
                        "mfe": float(mfe),
                        "mae": float(mae),
                        "entry_pos": float(open_trade.get("entry_pos", 0.0)),
                        "exit_pos": float(pos[exit_i]),
                    }
                )

                open_trade = None
                cur_sign = 0

                if s != 0:
                    open_trade = {
                        "entry_time": str(idx[i]),
                        "entry_i": int(i),
                        "side": "long" if s > 0 else "short",
                        "entry_equity": float(eq[i]),
                        "max_fav": float(eq[i]),
                        "max_adv": float(eq[i]),
                        "entry_pos": float(pos[i]),
                    }
                    cur_sign = s

    df_dir = pd.DataFrame(trades)
    df_reb = pd.DataFrame(rebalance_events)

    if df_dir.empty and df_reb.empty:
        df_trades = pd.DataFrame()
    elif df_dir.empty:
        df_trades = df_reb.copy()
    elif df_reb.empty:
        df_trades = df_dir.copy()
    else:
        df_trades = pd.concat([df_dir, df_reb], ignore_index=True, sort=False)

    n_dir = int(len(df_dir))
    n_reb = int(len(df_reb))
    n_total = int(len(df_trades))

    stats = {
        "n_trades_total": n_total,
        "n_trades_directional": n_dir,
        "n_trades_rebalance": n_reb,
        "n_trades": n_total,
        "win_rate": float((df_dir["pnl"] > 0).mean()) if n_dir else 0.0,
        "avg_trade_return": float(df_dir["return"].mean()) if n_dir else 0.0,
        "avg_duration_bars": float(df_dir["duration_bars"].mean()) if n_dir else 0.0,
        "profit_factor": float(
            df_dir.loc[df_dir["pnl"] > 0, "pnl"].sum()
            / (abs(df_dir.loc[df_dir["pnl"] < 0, "pnl"].sum()) + 1e-12)
        )
        if n_dir
        else 0.0,
    }

    return df_trades, stats


def compute_vol_target_bh_equity(bh_equity: np.ndarray, target_rets: np.ndarray, initial: float) -> np.ndarray:
    bh_eq = _to_numpy_finite(bh_equity)
    if bh_eq.size < 3 or target_rets.size < 5:
        return bh_eq

    bh_rets = compute_returns(np.r_[initial, bh_eq])
    if bh_rets.size < 5:
        return bh_eq

    vol_bh = float(np.std(bh_rets) + 1e-12)
    vol_t = float(np.std(target_rets) + 1e-12)
    k = vol_t / vol_bh
    k = float(np.clip(k, 0.0, 3.0))

    eq = np.ones_like(bh_eq, dtype=np.float64) * float(initial)
    lr = bh_rets[: len(eq)] * k
    eq[:] = float(initial) * np.exp(np.cumsum(lr))
    return eq


def plot_gross_net_and_costs(x, eq_net, eq_gross, cum_cost, outpath: Path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, eq_net, label="Equity net")
    if eq_gross is not None:
        ax.plot(x, eq_gross, label="Equity gross (proxy)", linestyle="--")
    if cum_cost is not None:
        ax2 = ax.twinx()
        ax2.plot(x, cum_cost, label="Cum costs", linestyle=":")
        ax2.set_ylabel("Cumulative costs")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")
    ax.set_title("HC_IA_RETAIL — Equity net vs gross and cumulative costs (OOS)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_exposure_hist(pos, outpath: Path):
    p = _to_numpy_finite(pos)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.hist(p, bins=80)
    ax.set_title("HC_IA_RETAIL — Exposure distribution (position)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_rolling_vol(x, rets, outpath: Path, window: int = 24 * 30, freq_per_year: float = 24 * 365):
    r = _to_numpy_finite(rets)
    if r.size < window + 2:
        return
    out = np.full(r.size, np.nan, dtype=np.float64)
    for i in range(window, r.size + 1):
        w = r[i - window : i]
        out[i - 1] = float(np.std(w) * math.sqrt(freq_per_year))
    x_r = x[1:] if hasattr(x, "__len__") and len(x) >= (len(out) + 1) else np.arange(out.size)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x_r, out, label=f"Rolling vol (window={window})")
    ax.set_title("HC_IA_RETAIL — Rolling volatility (OOS)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_scatter_vs_benchmark(rets_sac, rets_bench, outpath: Path, title: str):
    r1 = _to_numpy_finite(rets_sac)
    r2 = _to_numpy_finite(rets_bench)
    n = min(len(r1), len(r2))
    if n < 10:
        return
    r1 = r1[:n]
    r2 = r2[:n]
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.scatter(r2, r1, s=6)
    ax.set_xlabel("Benchmark returns")
    ax.set_ylabel("Strategy returns")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def pick_latest_run(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No runs found in {runs_dir}")
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir", type=str, default="", help="Run directory containing vecnormalize.pkl + best/best_model.zip"
    )
    ap.add_argument("--model_path", type=str, default="", help="Override model path (zip)")
    ap.add_argument("--vecnorm_path", type=str, default="", help="Override vecnormalize.pkl path")
    ap.add_argument("--data_path", type=str, default="", help="Override dataset path")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--split_json",
        type=str,
        default="",
        help="Optional JSON file with explicit train/val/test indices (walk-forward)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else pick_latest_run(Path(paths_cfg.out_dir))
    vec_path = Path(args.vecnorm_path).expanduser().resolve() if args.vecnorm_path else (run_dir / "vecnormalize.pkl")

    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
    else:
        cand = run_dir / "best" / "best_model.zip"
        if cand.exists():
            model_path = cand
        else:
            models = sorted(
                Path(paths_cfg.models_dir).glob("hc_retail_sac_gru_final_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            model_path = models[0] if models else cand

    df_feat, feature_cols, data_source_meta = build_eval_dataframe(args.data_path if args.data_path else None)

    def _apply_explicit_test_split(df_feat, split_json_path: str) -> pd.DataFrame:
        p = Path(split_json_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"split_json not found: {p}")
        spec = json.loads(p.read_text(encoding="utf-8"))
        a = int(spec["test"]["start"])
        b = int(spec["test"]["end"])
        if not (0 <= a < b <= len(df_feat)):
            raise ValueError(f"Invalid test bounds: {a}:{b} with n={len(df_feat)}")
        return df_feat.iloc[a:b].copy()

    if args.split_json.strip():
        df_test = _apply_explicit_test_split(df_feat, args.split_json.strip())
        _df_train, _df_val = pd.DataFrame(), pd.DataFrame()
    else:
        _df_train, _df_val, df_test = split_train_val_test(df_feat)

    env_test = DummyVecEnv([make_env(df_test, feature_cols)])
    if vec_path.exists():
        env_test = VecNormalize.load(str(vec_path), env_test)
        env_test.training = False
        env_test.norm_reward = False
    else:
        env_test = VecNormalize(env_test, training=False, norm_obs=False, norm_reward=False)

    model = SAC.load(str(model_path))
    obs = env_test.reset()

    initial_eq_fallback = float(getattr(env_cfg, "initial_cash", 1.0))
    equities = [initial_eq_fallback]
    positions: list[float] = []
    rewards: list[float] = []
    trace_rows: list[dict] = []

    t = 0
    while True:
        action, _ = model.predict(obs, deterministic=bool(args.deterministic))
        obs, reward, dones, infos = env_test.step(action)

        info = infos[0] if isinstance(infos, list) else infos
        trace = dict(info)
        trace["t"] = int(t)
        trace["action_pred"] = float(np.asarray(action).reshape(-1)[0])
        trace["reward"] = float(np.asarray(reward).reshape(-1)[0])
        trace_rows.append(trace)

        eq = info_get(info, ["equity", "eq", "nav", "portfolio_value"], default=np.nan)
        pos = info_get(info, ["position", "pos", "exposure", "pos_new", "pos_prev"], default=np.nan)

        if np.isfinite(eq):
            equities.append(float(eq))
        else:
            equities.append(float(equities[-1]))

        positions.append(float(pos) if np.isfinite(pos) else float("nan"))
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))

        t += 1
        if bool(np.asarray(dones).reshape(-1)[0]):
            break
        if t > len(df_test) + 5:
            break

    equities = np.asarray(equities, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)

    eq_plot = equities[1:]
    n_points = len(eq_plot)

    x = df_test["timestamp"].iloc[:n_points] if "timestamp" in df_test.columns else np.arange(n_points)
    trace_df = pd.DataFrame(trace_rows)

    funding_keys = ["funding", "funding_fee", "funding_cost", "funding_paid", "funding_pnl", "funding_rate_used"]
    _funding_present = any(k in trace_df.columns for k in funding_keys)
    _funding_sum_abs = 0.0
    if _funding_present:
        for k in ["funding", "funding_fee", "funding_cost", "funding_paid", "funding_pnl"]:
            if k in trace_df.columns:
                v = pd.to_numeric(trace_df[k], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                _funding_sum_abs += float(np.sum(np.abs(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))))

    if not _funding_present:
        funding_mode = "disabled"
    else:
        funding_mode = "enabled" if _funding_sum_abs > 0.0 else "zeroed"

    funding_rate_used_mean = 0.0
    if "funding_rate_used" in trace_df.columns:
        fr = pd.to_numeric(trace_df["funding_rate_used"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        fr = np.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0)
        funding_rate_used_mean = float(np.mean(fr)) if fr.size else 0.0

    funding_cost_total = 0.0
    for k in ["funding_cost", "funding_fee", "funding", "funding_paid"]:
        if k in trace_df.columns:
            v = pd.to_numeric(trace_df[k], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if np.nanmedian(v) < 0:
                v = -v
            funding_cost_total += float(np.sum(v))
    if funding_cost_total == 0.0 and "funding_pnl" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if np.nanmedian(v) < 0:
            funding_cost_total = float(np.sum(-v))
        else:
            funding_cost_total = float(np.sum(np.abs(v)))

    non_funding_cost_candidates = [
        "fee",
        "fees",
        "commission",
        "trade_cost",
        "cost",
        "costs",
        "slippage",
    ]

    nonfund_cost_per_step = None
    for c in non_funding_cost_candidates:
        if c in trace_df.columns:
            vals = pd.to_numeric(trace_df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if nonfund_cost_per_step is None:
                nonfund_cost_per_step = np.zeros_like(vals, dtype=float)
            nonfund_cost_per_step = nonfund_cost_per_step + vals

    if nonfund_cost_per_step is not None:
        if np.nanmedian(nonfund_cost_per_step) < 0:
            nonfund_cost_per_step = -nonfund_cost_per_step
        nonfund_cost_per_step = np.nan_to_num(nonfund_cost_per_step, nan=0.0, posinf=0.0, neginf=0.0)

    funding_pnl_log_per_step = None
    funding_cost_abs_per_step = None

    if "funding_pnl_log" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding_pnl_log"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        funding_pnl_log_per_step = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    elif "funding_pnl" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        funding_pnl_log_per_step = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    elif "funding_cost" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding_cost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        funding_pnl_log_per_step = -v
    elif "funding_fee" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding_fee"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        funding_pnl_log_per_step = -v
    elif "funding" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        funding_pnl_log_per_step = -v

    if "funding_cost_abs" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding_cost_abs"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        funding_cost_abs_per_step = np.abs(v)
    elif "funding_cost" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding_cost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        funding_cost_abs_per_step = np.abs(v)
    elif "funding_fee" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding_fee"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        funding_cost_abs_per_step = np.abs(v)
    elif "funding" in trace_df.columns:
        v = pd.to_numeric(trace_df["funding"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        funding_cost_abs_per_step = np.abs(v)

    cum_cost = None
    eq_gross = None
    if nonfund_cost_per_step is not None:
        c = nonfund_cost_per_step[:n_points]
        cum_cost = np.cumsum(c)
        eq_gross = eq_plot + cum_cost

    costs_breakdown_df = None
    cols = {}

    for c in non_funding_cost_candidates:
        if c in trace_df.columns:
            vals = pd.to_numeric(trace_df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)[:n_points]
            if np.nanmedian(vals) < 0:
                vals = -vals
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            cols[f"{c}_cum"] = np.cumsum(vals)

    if funding_cost_abs_per_step is not None:
        cols["funding_cost_abs_cum"] = np.cumsum(funding_cost_abs_per_step[:n_points])
    if funding_pnl_log_per_step is not None:
        cols["funding_pnl_log_cum"] = np.cumsum(funding_pnl_log_per_step[:n_points])

    if cols:
        costs_breakdown_df = pd.DataFrame(cols)

    initial_eq = float(equities[0])

    bh_equity = compute_buyhold_equity(df_test, n_points=n_points, initial=initial_eq)

    rets_sac = compute_returns(np.r_[initial_eq, eq_plot])
    rets_bh = compute_returns(np.r_[initial_eq, bh_equity])

    bh_volt = compute_vol_target_bh_equity(bh_equity, target_rets=rets_sac, initial=initial_eq)
    rets_bh_vt = compute_returns(np.r_[initial_eq, bh_volt])

    pos_naive = compute_naive_position(df_test, n_points=n_points)
    eq_naive, naive_details = compute_equity_from_positions(
        df_test,
        positions=pos_naive,
        initial=initial_eq,
        fee_bps=float(getattr(env_cfg, "fee_bps", 0.0)),
        spread_bps=float(getattr(env_cfg, "spread_bps", 0.0)),
    )
    rets_naive = compute_returns(np.r_[initial_eq, eq_naive])

    freq = float(getattr(data_cfg, "freq_per_year", 24 * 365))

    sac_metrics = compute_advanced_metrics(equities, freq_per_year=freq)
    bh_metrics = compute_advanced_metrics(np.r_[initial_eq, bh_equity], freq_per_year=freq)
    bh_volt_metrics = compute_advanced_metrics(np.r_[initial_eq, bh_volt], freq_per_year=freq)
    naive_metrics = compute_advanced_metrics(np.r_[initial_eq, eq_naive], freq_per_year=freq)

    mean_abs_action = (
        float(np.mean(np.abs(pd.to_numeric(trace_df["action_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float))))
        if "action_pred" in trace_df.columns and len(trace_df)
        else float("nan")
    )
    pos_nan_pct = float(np.mean(~np.isfinite(positions))) if len(positions) else 1.0

    pos_clean = np.nan_to_num(positions[:n_points], nan=0.0)
    pos_abs_mean = float(np.abs(pos_clean).mean())
    pos_mean = float(pos_clean.mean())
    turnover = np.abs(np.diff(pos_clean)) if len(pos_clean) > 1 else np.array([0.0])
    turnover_mean = float(turnover.mean()) if turnover.size else 0.0
    turnover_sum = float(turnover.sum()) if turnover.size else 0.0

    eps = 1e-3
    pct_long = float((pos_clean > eps).mean())
    pct_short = float((pos_clean < -eps).mean())
    pct_flat = float((np.abs(pos_clean) <= eps).mean())

    if n_points >= 2:
        dpos = np.diff(pos_clean[:n_points])
        eq_for_turn = eq_plot[: len(dpos)]
        turnover_notional = np.abs(dpos) * np.maximum(eq_for_turn, 0.0)
        turnover_notional_sum = float(np.sum(turnover_notional))
        turnover_notional_mean = float(np.mean(turnover_notional))
    else:
        turnover_notional_sum = 0.0
        turnover_notional_mean = 0.0

    total_cost = float(cum_cost[-1]) if cum_cost is not None and len(cum_cost) else 0.0
    cost_per_notional = float(total_cost / (turnover_notional_sum + 1e-12)) if turnover_notional_sum > 0 else 0.0

    north_star_name = "north_star_v1"
    north_star_params = {"d": 0.20, "tau": 0.01, "lambda_dd": 2.0, "lambda_to": 1.0}
    north_star_definition = "sharpe - lambda_dd*max(0, max_dd-d) - lambda_to*max(0, turnover_mean-tau)"

    score_oos_sac = compute_north_star_score(
        sharpe=float(sac_metrics.get("sharpe", 0.0)),
        maxdd=float(sac_metrics.get("max_dd", 0.0)),
        turnover_mean=float(turnover_mean),
        **north_star_params,
    )

    score_oos_bh = compute_north_star_score(
        sharpe=float(bh_metrics.get("sharpe", 0.0)),
        maxdd=float(bh_metrics.get("max_dd", 0.0)),
        turnover_mean=0.0,
        **north_star_params,
    )
    score_oos_bh_vt = compute_north_star_score(
        sharpe=float(bh_volt_metrics.get("sharpe", 0.0)),
        maxdd=float(bh_volt_metrics.get("max_dd", 0.0)),
        turnover_mean=0.0,
        **north_star_params,
    )

    if len(pos_naive) > 1:
        naive_turnover_mean = float(np.mean(np.abs(np.diff(np.nan_to_num(pos_naive, nan=0.0)))))
    else:
        naive_turnover_mean = 0.0
    score_oos_naive = compute_north_star_score(
        sharpe=float(naive_metrics.get("sharpe", 0.0)),
        maxdd=float(naive_metrics.get("max_dd", 0.0)),
        turnover_mean=float(naive_turnover_mean),
        **north_star_params,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = run_dir / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if costs_breakdown_df is not None:
        costs_breakdown_df.to_csv(out_dir / "costs_breakdown.csv", index=False)

    trace_df.to_csv(out_dir / "trace_oos.csv", index=False)
    try:
        trace_df.to_parquet(out_dir / "trace_oos.parquet", index=False)
    except Exception:
        pass

    eq_df = pd.DataFrame(
        {
            "timestamp": x if isinstance(x, pd.Series) else pd.Series(x),
            "equity_sac": eq_plot,
            "equity_buyhold": bh_equity,
            "equity_buyhold_voltarget": bh_volt,
            "equity_naive": eq_naive,
            "position_sac": pos_clean[:n_points],
            "position_naive": pos_naive[:n_points],
        }
    )
    eq_df.to_csv(out_dir / "equity_oos.csv", index=False)

    monthly_sac = compute_monthly_table(x, eq_plot)
    monthly_bh = compute_monthly_table(x, bh_equity)
    monthly_volt = compute_monthly_table(x, bh_volt)
    monthly_naive = compute_monthly_table(x, eq_naive)
    monthly_sac.to_csv(out_dir / "monthly_returns_sac.csv", index=False)
    monthly_bh.to_csv(out_dir / "monthly_returns_buyhold.csv", index=False)
    monthly_volt.to_csv(out_dir / "monthly_returns_bh_voltarget.csv", index=False)
    monthly_naive.to_csv(out_dir / "monthly_returns_naive.csv", index=False)

    dd_events = compute_drawdown_events(x, eq_plot)
    dd_events.to_csv(out_dir / "drawdown_events.csv", index=False)

    trades_df, trades_stats = compute_trade_blotter(x, eq_plot, pos_clean[:n_points])
    trades_df.to_csv(out_dir / "trades.csv", index=False)

    r = rets_sac
    pos_for_r = pos_clean[: len(r)]
    regime = np.where(pos_for_r > 1e-3, "long", np.where(pos_for_r < -1e-3, "short", "flat"))
    attrib = pd.DataFrame({"regime": regime, "logret": r})
    attrib_summary = (
        attrib.groupby("regime")
        .agg(
            n=("logret", "size"),
            mean=("logret", "mean"),
            sum=("logret", "sum"),
            vol=("logret", "std"),
        )
        .reset_index()
    )
    attrib_summary.to_csv(out_dir / "attribution_by_regime.csv", index=False)

    out = {
        "eval_id": ts,
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "vecnorm_path": str(vec_path) if vec_path.exists() else None,
        "feature_cols": list(feature_cols),
        "window_size": int(data_cfg.window_size),
        "env_cfg": asdict(env_cfg),
        "regime_cfg": asdict(regime_cfg),
        "data_source_meta": data_source_meta,
        "metrics": {
            "sac": sac_metrics,
            "buyhold": bh_metrics,
            "buyhold_voltarget": bh_volt_metrics,
            "naive": naive_metrics,
            "delta_vs_buyhold": {
                "delta_sharpe": float(sac_metrics["sharpe"] - bh_metrics["sharpe"]),
                "delta_cumret": float(sac_metrics["cumulative_return"] - bh_metrics["cumulative_return"]),
                "delta_maxdd": float(sac_metrics["max_dd"] - bh_metrics["max_dd"]),
            },
            "delta_vs_buyhold_voltarget": {
                "delta_sharpe": float(sac_metrics["sharpe"] - bh_volt_metrics["sharpe"]),
                "delta_cumret": float(sac_metrics["cumulative_return"] - bh_volt_metrics["cumulative_return"]),
                "delta_maxdd": float(sac_metrics["max_dd"] - bh_volt_metrics["max_dd"]),
            },
            "delta_vs_naive": {
                "delta_sharpe": float(sac_metrics["sharpe"] - naive_metrics["sharpe"]),
                "delta_cumret": float(sac_metrics["cumulative_return"] - naive_metrics["cumulative_return"]),
                "delta_maxdd": float(sac_metrics["max_dd"] - naive_metrics["max_dd"]),
            },
            "north_star": {
                "name": north_star_name,
                "definition": north_star_definition,
                "params": dict(north_star_params),
                "score_oos": {
                    "sac": float(score_oos_sac),
                    "buyhold": float(score_oos_bh),
                    "buyhold_voltarget": float(score_oos_bh_vt),
                    "naive": float(score_oos_naive),
                },
            },
            "funding": {
                "funding_mode": str(funding_mode),
                "funding_rate_used_mean": float(funding_rate_used_mean),
                "funding_cost_total": float(funding_cost_total),
            },
        },
        "position_stats": {
            "mean_pos": pos_mean,
            "mean_abs_pos": pos_abs_mean,
            "pct_flat_abs_lt_1e-3": pct_flat,
            "turnover_mean_abs_dpos": turnover_mean,
            "turnover_sum_abs_dpos": turnover_sum,
            "pct_long": pct_long,
            "pct_short": pct_short,
            "pct_flat": pct_flat,
            "turnover_notional_sum": turnover_notional_sum,
            "turnover_notional_mean": turnover_notional_mean,
            "total_cost": total_cost,
            "cost_per_notional": cost_per_notional,
            "pos_nan_pct": float(pos_nan_pct),
            "mean_abs_action": float(mean_abs_action),
        },
        "trade_stats": trades_stats,
        "cost_available": bool(nonfund_cost_per_step is not None),
        "naive_details": {
            "fee_bps_used": float(getattr(env_cfg, "fee_bps", 0.0)),
            "spread_bps_used": float(getattr(env_cfg, "spread_bps", 0.0)),
            "turnover_sum": float(np.sum(naive_details["turnover"])) if naive_details["turnover"].size else 0.0,
            "cost_sum": float(np.sum(naive_details["cost"])) if naive_details["cost"].size else 0.0,
        },
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(json_sanitize(out), f, indent=2)

    report = []
    report.append("# HC_IA_RETAIL — Evaluation OOS\n")
    report.append(f"- eval_id: `{ts}`\n")
    report.append(f"- run_dir: `{run_dir}`\n")
    report.append(f"- model: `{model_path}`\n")
    report.append(f"- vecnorm: `{vec_path if vec_path.exists() else None}`\n")
    report.append(f"- window_size: `{int(data_cfg.window_size)}`\n")
    report.append(f"- regime_enabled: `{bool(getattr(regime_cfg, 'enabled', False))}`\n")
    report.append(f"- regime_feature_mode: `{getattr(regime_cfg, 'regime_feature_mode', 'market_plus_regime')}`\n")

    report.append("\n## Metrics (SAC)\n")
    for k, v in sac_metrics.items():
        report.append(f"- {k}: {v:.6g}\n")

    report.append("\n## North Star (OOS)\n")
    report.append(f"- name: {north_star_name}\n")
    report.append(f"- definition: {north_star_definition}\n")
    report.append(f"- params: {north_star_params}\n")
    report.append(f"- score_oos_sac: {score_oos_sac:.6g}\n")
    report.append(f"- score_oos_buyhold: {score_oos_bh:.6g}\n")
    report.append(f"- score_oos_buyhold_voltarget: {score_oos_bh_vt:.6g}\n")
    report.append(f"- score_oos_naive: {score_oos_naive:.6g}\n")

    report.append("\n## Funding (OOS)\n")
    report.append(f"- funding_mode: {funding_mode}\n")
    report.append(f"- funding_rate_used_mean: {funding_rate_used_mean:.6g}\n")
    report.append(f"- funding_cost_total: {funding_cost_total:.6g}\n")

    report.append("\n## Metrics (Buy & Hold)\n")
    for k, v in bh_metrics.items():
        report.append(f"- {k}: {v:.6g}\n")

    report.append("\n## Metrics (Buy & Hold VolTarget)\n")
    for k, v in bh_volt_metrics.items():
        report.append(f"- {k}: {v:.6g}\n")

    report.append("\n## Metrics (Naive baseline)\n")
    for k, v in naive_metrics.items():
        report.append(f"- {k}: {v:.6g}\n")

    report.append("\n## Position stats\n")
    report.append(f"- mean|action|: {mean_abs_action:.6g}\n")
    report.append(f"- pos_nan_pct: {pos_nan_pct:.6g}\n")
    report.append(f"- mean_pos: {pos_mean:.6g}\n")
    report.append(f"- mean_abs_pos: {pos_abs_mean:.6g}\n")
    report.append(f"- turnover_mean_abs_dpos: {turnover_mean:.6g}\n")
    report.append(f"- turnover_sum_abs_dpos: {turnover_sum:.6g}\n")
    report.append(f"- pct_flat_abs_lt_1e-3: {pct_flat:.6g}\n")
    report.append(f"- pct_long: {pct_long:.6g}\n")
    report.append(f"- pct_short: {pct_short:.6g}\n")
    report.append(f"- turnover_notional_sum: {turnover_notional_sum:.6g}\n")
    report.append(f"- turnover_notional_mean: {turnover_notional_mean:.6g}\n")
    report.append(f"- total_cost: {total_cost:.6g}\n")
    report.append(f"- cost_per_notional: {cost_per_notional:.6g}\n")

    report.append("\n## Plots\n")
    report.append("- `plots/equity_oos.png` (SAC, BH, BH vol-target, naive)\n")
    report.append("- `plots/drawdown_oos.png`\n")
    report.append("- `plots/underwater_oos.png`\n")
    report.append("- `plots/position_oos.png`\n")
    report.append("- `plots/turnover_oos.png`\n")
    report.append("- `plots/returns_hist.png`\n")
    report.append("- `plots/qq_plot.png`\n")
    report.append("- `plots/rolling_sharpe.png`\n")
    report.append("- `plots/equity_position_3d.png`\n")
    report.append("- `plots/exposure_hist.png`\n")
    report.append("- `plots/equity_gross_net_costs.png` (if available)\n")
    report.append("- `plots/monthly_heatmap_sac.png`\n")
    report.append("- `plots/rolling_vol.png`\n")
    report.append("- `plots/scatter_vs_buyhold.png`\n")

    report.append("\n## Tables / Artifacts\n")
    report.append("- `equity_oos.csv`\n")
    report.append("- `trace_oos.csv`\n")
    report.append("- `metrics.json`\n")
    report.append("- `monthly_returns_sac.csv`\n")
    report.append("- `monthly_returns_buyhold.csv`\n")
    report.append("- `monthly_returns_bh_voltarget.csv`\n")
    report.append("- `monthly_returns_naive.csv`\n")
    report.append("- `drawdown_events.csv`\n")
    report.append("- `trades.csv`\n")
    report.append("- `attribution_by_regime.csv`\n")

    (out_dir / "report.md").write_text("".join(report), encoding="utf-8")

    try:
        runs_csv = run_dir / "runs" / "runs_index.csv"

        funding_mode_row = None
        if "funding_mode" in trace_df.columns and len(trace_df):
            try:
                funding_mode_row = str(trace_df["funding_mode"].iloc[0])
            except Exception:
                funding_mode_row = None
        if not funding_mode_row:
            funding_mode_row = str(getattr(env_cfg, "funding_mode", "")) or ""

        seed_value = float("nan")
        try:
            meta_p = run_dir / "meta.json"
            if meta_p.exists():
                meta_j = json.loads(meta_p.read_text(encoding="utf-8"))
                seed_value = safe_float(meta_j.get("seed", float("nan")))
        except Exception:
            pass
        if not np.isfinite(seed_value):
            seed_value = safe_float(getattr(env_cfg, "seed", float("nan")))

        row = {
            "run_id": str(run_dir.name),
            "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "seed": seed_value,
            "score_val": float("nan"),
            "score_oos": safe_float(score_oos_sac),
            "sharpe_oos": safe_float(
                sac_metrics.get("sharpe", float("nan")) if isinstance(sac_metrics, dict) else float("nan")
            ),
            "maxdd_oos": safe_float(
                sac_metrics.get("max_dd", float("nan")) if isinstance(sac_metrics, dict) else float("nan")
            ),
            "turnover_oos": safe_float(turnover_mean),
            "funding_mode": funding_mode_row,
        }
        append_run_index(runs_csv, row)
    except Exception:
        pass

    plot_equity(x, eq_plot, bh_equity, bh_volt, eq_naive, plots_dir / "equity_oos.png")
    plot_drawdown(x, eq_plot, plots_dir / "drawdown_oos.png")
    plot_underwater(x, eq_plot, plots_dir / "underwater_oos.png")
    plot_positions(x, pos_clean[:n_points], plots_dir / "position_oos.png")
    plot_turnover(x, pos_clean[:n_points], plots_dir / "turnover_oos.png")

    plot_exposure_hist(pos_clean[:n_points], plots_dir / "exposure_hist.png")

    if eq_gross is not None and cum_cost is not None:
        plot_gross_net_and_costs(x, eq_plot, eq_gross, cum_cost, plots_dir / "equity_gross_net_costs.png")

    plot_monthly_heatmap(
        monthly_sac,
        plots_dir / "monthly_heatmap_sac.png",
        title="HC_IA_RETAIL — Monthly returns heatmap (SAC OOS)",
    )

    plot_rolling_vol(
        x,
        rets_sac,
        plots_dir / "rolling_vol.png",
        window=int(getattr(data_cfg, "rolling_vol_window", 24 * 30)),
        freq_per_year=freq,
    )

    plot_scatter_vs_benchmark(
        rets_sac,
        rets_bh,
        plots_dir / "scatter_vs_buyhold.png",
        title="HC_IA_RETAIL — Returns scatter (SAC vs Buy&Hold)",
    )

    plot_returns_hist(rets_sac, plots_dir / "returns_hist.png")
    plot_qq(rets_sac, plots_dir / "qq_plot.png")
    plot_rolling_sharpe(
        x,
        rets_sac,
        plots_dir / "rolling_sharpe.png",
        window=int(getattr(data_cfg, "rolling_sharpe_window", 24 * 30)),
        freq_per_year=freq,
    )

    try:
        from matplotlib import cm

        t_idx = np.arange(n_points, dtype=float)
        r_step = rewards[:n_points]
        norm = (
            plt.Normalize(vmin=float(np.nanmin(r_step)), vmax=float(np.nanmax(r_step)))
            if np.nanmax(r_step) > np.nanmin(r_step)
            else plt.Normalize(-1e-6, 1e-6)
        )
        colors = cm.viridis(norm(np.nan_to_num(r_step, nan=0.0)))

        fig3d = plt.figure(figsize=(11, 7))
        ax3d = fig3d.add_subplot(111, projection="3d")
        ax3d.scatter(t_idx, pos_clean[:n_points], eq_plot, c=colors, s=6, depthshade=True)
        ax3d.set_xlabel("Step")
        ax3d.set_ylabel("Position")
        ax3d.set_zlabel("Equity")
        ax3d.set_title("HC_IA_RETAIL — OOS 3D view")
        fig3d.tight_layout()
        fig3d.savefig(plots_dir / "equity_position_3d.png")
        plt.close(fig3d)
    except Exception:
        pass

    print(f"[OK] eval_dir -> {out_dir}")
    print(f"[OK] regime_enabled={bool(getattr(regime_cfg, 'enabled', False))}")
    print(f"[OK] regime_feature_mode={getattr(regime_cfg, 'regime_feature_mode', 'market_plus_regime')}")
    print(
        "[OK] SAC   | Sharpe ≈ "
        f"{sac_metrics['sharpe']:.3f} | CumRet ≈ {sac_metrics['cumulative_return']:.2%} | MaxDD ≈ {sac_metrics['max_dd']:.2%}"
    )
    print(
        f"[OK] NS    | {north_star_name}={score_oos_sac:.6g} | d={north_star_params['d']}, tau={north_star_params['tau']}, "
        f"lambda_dd={north_star_params['lambda_dd']}, lambda_to={north_star_params['lambda_to']}"
    )
    print(
        f"[OK] FUND  | mode={funding_mode} | rate_mean={funding_rate_used_mean:.6g} | cost_total={funding_cost_total:.6g}"
    )
    print(
        "[OK] BH    | Sharpe ≈ "
        f"{bh_metrics['sharpe']:.3f} | CumRet ≈ {bh_metrics['cumulative_return']:.2%} | MaxDD ≈ {bh_metrics['max_dd']:.2%}"
    )
    print(
        "[OK] BH_VT | Sharpe ≈ "
        f"{bh_volt_metrics['sharpe']:.3f} | CumRet ≈ {bh_volt_metrics['cumulative_return']:.2%} | MaxDD ≈ {bh_volt_metrics['max_dd']:.2%}"
    )
    print(
        "[OK] NAIVE | Sharpe ≈ "
        f"{naive_metrics['sharpe']:.3f} | CumRet ≈ {naive_metrics['cumulative_return']:.2%} | MaxDD ≈ {naive_metrics['max_dd']:.2%}"
    )
    print(f"[OK] ACT   | mean|action| ≈ {mean_abs_action:.6f}")
    print(
        "[OK] POS   | mean|pos| ≈ "
        f"{pos_abs_mean:.4f} | turnover_mean ≈ {turnover_mean:.6f} | pct_flat ≈ {pct_flat:.2%} | pos_nan_pct ≈ {pos_nan_pct:.2%}"
    )


if __name__ == "__main__":
    main()