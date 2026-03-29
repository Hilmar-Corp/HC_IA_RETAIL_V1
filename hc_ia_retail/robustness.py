import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# IO helpers
# -------------------------

def _read_json(p: Path) -> dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def find_latest_dir(parent: Path, pattern: str) -> Path | None:
    candidates = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


# -------------------------
# Metrics (equity -> stats)
# -------------------------

def _equity_to_logrets(equity: np.ndarray) -> np.ndarray:
    equity = np.asarray(equity, dtype=float)
    equity = np.maximum(equity, 1e-12)
    log_eq = np.log(equity)
    r = np.diff(log_eq)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r


def _max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=float)
    e = np.maximum(e, 1e-12)
    peak = np.maximum.accumulate(e)
    peak = np.where(peak <= 0, 1e-12, peak)
    dd = 1.0 - (e / peak)
    if dd.size == 0:
        return 0.0
    return float(np.max(dd))


def _tail_ratio(logrets: np.ndarray, q_hi: float = 0.95, q_lo: float = 0.05) -> float:
    r = np.asarray(logrets, dtype=float)
    if r.size < 10:
        return 0.0
    hi = np.quantile(r, q_hi)
    lo = np.quantile(r, q_lo)
    lo = abs(lo) + 1e-12
    return float(hi / lo)


def _var_cvar(logrets: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    r = np.asarray(logrets, dtype=float)
    if r.size < 10:
        return 0.0, 0.0
    q = float(np.quantile(r, 1.0 - alpha))
    tail = r[r <= q]
    cvar = float(tail.mean()) if tail.size else q
    return q, cvar


def metrics_from_equity(
    equity: np.ndarray,
    freq_per_year: float = 24 * 365,
    downside_target: float = 0.0,
) -> dict[str, float]:
    """
    Compute standard quant metrics from an equity curve.
    - returns are log-returns r_t = log(E_t) - log(E_{t-1})
    - annualization uses freq_per_year
    """
    e = np.asarray(equity, dtype=float)
    if e.size < 3:
        return {
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "cumulative_return": 0.0,
            "max_dd": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "var_99": 0.0,
            "cvar_99": 0.0,
            "tail_ratio": 0.0,
        }

    r = _equity_to_logrets(e)
    mu = float(np.mean(r))
    sig = float(np.std(r) + 1e-12)

    ann_return = float(math.exp(mu * freq_per_year) - 1.0)
    ann_vol = float(sig * math.sqrt(freq_per_year))
    sharpe = float((mu / sig) * math.sqrt(freq_per_year))

    # Sortino
    downside = r - float(downside_target)
    downside = downside[downside < 0]
    dd_sig = float(np.sqrt(np.mean(downside * downside)) + 1e-12) if downside.size else 1e-12
    sortino = float((mu / dd_sig) * math.sqrt(freq_per_year))

    cumret = float(e[-1] / e[0] - 1.0)

    maxdd = _max_drawdown(e)
    calmar = float(ann_return / (maxdd + 1e-12))

    var95, cvar95 = _var_cvar(r, alpha=0.95)
    var99, cvar99 = _var_cvar(r, alpha=0.99)
    tail = _tail_ratio(r)

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cumulative_return": cumret,
        "max_dd": float(maxdd),
        "var_95": float(var95),
        "cvar_95": float(cvar95),
        "var_99": float(var99),
        "cvar_99": float(cvar99),
        "tail_ratio": float(tail),
    }


# -------------------------
# Parse eval outputs
# -------------------------

def parse_eval_metrics(eval_dir: Path) -> dict[str, Any]:
    """
    Reads eval_dir/metrics.json produced by scripts/eval/eval_oos.py (your evaluation brick).
    Returns a flat row for robust aggregation across seeds.
    """
    m = _read_json(eval_dir / "metrics.json")
    sac = (m.get("metrics") or {}).get("sac") or {}
    ps = m.get("position_stats") or {}
    ts = m.get("trade_stats") or {}

    row = {
        "eval_dir": str(eval_dir),
        "run_dir": str(eval_dir.parent),
        "eval_id": m.get("eval_id"),
        "model_path": m.get("model_path"),
        "vecnorm_path": m.get("vecnorm_path"),
        "window_size": m.get("window_size"),

        "sharpe": _safe_float(sac.get("sharpe")),
        "sortino": _safe_float(sac.get("sortino")),
        "calmar": _safe_float(sac.get("calmar")),
        "ann_return": _safe_float(sac.get("ann_return")),
        "ann_vol": _safe_float(sac.get("ann_vol")),
        "cumulative_return": _safe_float(sac.get("cumulative_return")),
        "max_dd": _safe_float(sac.get("max_dd")),
        "var_95": _safe_float(sac.get("var_95")),
        "cvar_95": _safe_float(sac.get("cvar_95")),
        "var_99": _safe_float(sac.get("var_99")),
        "cvar_99": _safe_float(sac.get("cvar_99")),
        "tail_ratio": _safe_float(sac.get("tail_ratio")),

        "mean_abs_pos": _safe_float(ps.get("mean_abs_pos")),
        "turnover_mean": _safe_float(ps.get("turnover_mean_abs_dpos")),
        "turnover_sum": _safe_float(ps.get("turnover_sum_abs_dpos")),
        "pct_flat": _safe_float(ps.get("pct_flat_abs_lt_1e-3")),
        "total_cost": _safe_float(ps.get("total_cost")),
        "cost_per_notional": _safe_float(ps.get("cost_per_notional")),

        "n_trades": int(ts.get("n_trades", 0) or 0),
        "win_rate": _safe_float(ts.get("win_rate")),
        "profit_factor": _safe_float(ts.get("profit_factor")),
        "avg_trade_return": _safe_float(ts.get("avg_trade_return")),
        "avg_duration_bars": _safe_float(ts.get("avg_duration_bars")),
    }
    return row


def load_eval_equities(eval_dir: Path) -> pd.DataFrame:
    """
    Expected columns (from your eval script):
      - timestamp (optional)
      - equity_sac
      - equity_bh (optional)
    """
    eq_path = eval_dir / "equity_oos.csv"
    if not eq_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(eq_path)
    # normalize column names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    return df


def load_eval_trace(eval_dir: Path) -> pd.DataFrame:
    tr_path = eval_dir / "trace_oos.csv"
    if not tr_path.exists():
        return pd.DataFrame()
    return pd.read_csv(tr_path)


# -------------------------
# Behavioral diagnostics
# -------------------------

def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_behavior_summary_from_trace(eval_dir: Path) -> dict[str, float]:
    """
    Pull robust behavior features directly from trace_oos.csv.
    This is complementary to metrics.json position_stats (mean-based).
    We add tails (p90/p95/p99) which matter institutionally.
    """
    tr = load_eval_trace(eval_dir)
    if tr.empty:
        return {}

    pos_col = _pick_first_existing(tr, ["position", "pos", "position_t", "exposure"])
    act_col = _pick_first_existing(tr, ["action", "a", "target_pos", "target_position"])
    cost_col = _pick_first_existing(tr, ["cost", "trade_cost", "fees", "fee", "commission", "slippage", "funding", "funding_fee"])
    pnl_col = _pick_first_existing(tr, ["pnl_log", "log_pnl", "pnl", "reward_pnl_log"])

    out: dict[str, float] = {}

    if pos_col is not None:
        pos = pd.to_numeric(tr[pos_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        abs_pos = np.abs(pos)
        if abs_pos.size:
            out["abs_pos_p50"] = float(np.quantile(abs_pos, 0.50))
            out["abs_pos_p90"] = float(np.quantile(abs_pos, 0.90))
            out["abs_pos_p95"] = float(np.quantile(abs_pos, 0.95))
            out["abs_pos_p99"] = float(np.quantile(abs_pos, 0.99))
            # turnover based on positions
            if abs_pos.size >= 2:
                dpos = np.abs(np.diff(pos))
                out["turnover_p50"] = float(np.quantile(dpos, 0.50))
                out["turnover_p90"] = float(np.quantile(dpos, 0.90))
                out["turnover_p95"] = float(np.quantile(dpos, 0.95))
                out["turnover_p99"] = float(np.quantile(dpos, 0.99))

    # cost tails
    if cost_col is not None:
        cost = pd.to_numeric(tr[cost_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.nanmedian(cost) < 0:
            cost = -cost
        cost = np.maximum(cost, 0.0)
        if cost.size:
            out["cost_p95"] = float(np.quantile(cost, 0.95))
            out["cost_p99"] = float(np.quantile(cost, 0.99))
            out["cost_mean"] = float(np.mean(cost))
            out["cost_sum"] = float(np.sum(cost))

    # pnl tails (log)
    if pnl_col is not None:
        pnl = pd.to_numeric(tr[pnl_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if pnl.size:
            out["pnl_p05"] = float(np.quantile(pnl, 0.05))
            out["pnl_p01"] = float(np.quantile(pnl, 0.01))
            out["pnl_mean"] = float(np.mean(pnl))
            out["pnl_std"] = float(np.std(pnl))

    # action vs position mismatch (if both present)
    if pos_col is not None and act_col is not None:
        pos = pd.to_numeric(tr[pos_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        act = pd.to_numeric(tr[act_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        n = min(pos.size, act.size)
        if n:
            gap = np.abs(act[:n] - pos[:n])
            out["act_pos_gap_mean"] = float(np.mean(gap))
            out["act_pos_gap_p95"] = float(np.quantile(gap, 0.95))

    return out


# -------------------------
# Cost sensitivity (log-space correct if possible)
# -------------------------

def compute_cost_sensitivity_from_eval(eval_dir: Path, multipliers: list[float], freq_per_year: float = 24 * 365) -> pd.DataFrame:
    """
    Post-hoc cost sensitivity WITHOUT re-running env.

    Preferred (correct) path:
      If trace has pnl_log and cost_step (positive),
      pnl_log = core - cost_step,
      pnl_log(m) = pnl_log - (m-1)*cost_step
      => rebuild equity via log accumulation.

    Fallback:
      If only equity and a cost-like column exist, we do a rough additive approximation (kept as last resort).
    """
    eq = load_eval_equities(eval_dir)
    tr = load_eval_trace(eval_dir)
    if eq.empty or tr.empty:
        return pd.DataFrame()

    # Equity column
    if "equity_sac" not in eq.columns:
        return pd.DataFrame()
    net = pd.to_numeric(eq["equity_sac"], errors="coerce").fillna(method="ffill").fillna(1.0).to_numpy(dtype=float)

    # Align
    n = min(len(net), len(tr))
    net = net[:n]
    tr = tr.iloc[:n].copy()

    pnl_col = _pick_first_existing(tr, ["pnl_log", "log_pnl"])
    cost_col = _pick_first_existing(tr, ["cost", "trade_cost", "fees", "fee", "commission", "slippage", "funding", "funding_fee"])

    def _metrics(equity: np.ndarray) -> dict[str, float]:
        m = metrics_from_equity(equity, freq_per_year=freq_per_year)
        return {"sharpe": m["sharpe"], "cumret": m["cumulative_return"], "maxdd": m["max_dd"]}

    rows = []

    # Preferred (log-space consistent)
    if pnl_col is not None and cost_col is not None:
        pnl = pd.to_numeric(tr[pnl_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        cost = pd.to_numeric(tr[cost_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        # normalize cost to positive
        if np.nanmedian(cost) < 0:
            cost = -cost
        cost = np.maximum(cost, 0.0)

        # infer initial from net
        e0 = float(net[0]) if net.size else 1.0
        log_e0 = math.log(max(e0, 1e-12))

        for mlt in multipliers:
            pnl_m = pnl - (float(mlt) - 1.0) * cost
            log_eq = log_e0 + np.cumsum(pnl_m)
            eq_m = np.exp(log_eq)
            mm = _metrics(eq_m)
            rows.append({"cost_multiplier": float(mlt), **mm, "method": "log_pnl_adjust"})

        return pd.DataFrame(rows)

    # Fallback (approx) if we only have net equity + some cost proxy
    if cost_col is None:
        return pd.DataFrame()

    cost = pd.to_numeric(tr[cost_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if np.nanmedian(cost) < 0:
        cost = -cost
    cost = np.maximum(cost, 0.0)
    cum_cost = np.cumsum(cost)
    gross_approx = net + cum_cost

    for mlt in multipliers:
        net_m = gross_approx - float(mlt) * cum_cost
        net_m = np.maximum(net_m, 1e-12)
        mm = _metrics(net_m)
        rows.append({"cost_multiplier": float(mlt), **mm, "method": "additive_fallback"})

    return pd.DataFrame(rows)


# -------------------------
# Block bootstrap (statistical robustness)
# -------------------------

@dataclass
class BootstrapConfig:
    n_boot: int = 500
    block_len: int = 48  # ~2 days at 1h
    ci_lo: float = 0.05
    ci_hi: float = 0.95


def _moving_block_bootstrap_indices(T: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Standard moving block bootstrap indices.
    Build a resampled series length T by concatenating random contiguous blocks.
    """
    if T <= 0:
        return np.array([], dtype=int)
    L = max(1, int(block_len))
    out = []
    while len(out) < T:
        start = int(rng.integers(0, max(1, T)))
        block = [(start + i) % T for i in range(L)]
        out.extend(block)
    return np.array(out[:T], dtype=int)


def bootstrap_equity_metrics(
    equity: np.ndarray,
    freq_per_year: float,
    cfg: BootstrapConfig,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Bootstraps log-returns in blocks, reconstructs equity, computes metrics.
    Returns a dataframe with columns:
      - boot_id, sharpe, ann_return, ann_vol, max_dd, cumret
    """
    e = np.asarray(equity, dtype=float)
    if e.size < 10:
        return pd.DataFrame()

    r = _equity_to_logrets(e)
    T = r.size
    if T < 10:
        return pd.DataFrame()

    rng = np.random.default_rng(int(seed))
    rows = []
    for b in range(int(cfg.n_boot)):
        idx = _moving_block_bootstrap_indices(T, cfg.block_len, rng=rng)
        r_b = r[idx]
        # rebuild equity starting from 1 (scale-free)
        log_eq = np.cumsum(r_b)
        eq_b = np.exp(np.concatenate([[0.0], log_eq]))  # length T+1
        m = metrics_from_equity(eq_b, freq_per_year=freq_per_year)
        rows.append({
            "boot_id": b,
            "sharpe": m["sharpe"],
            "ann_return": m["ann_return"],
            "ann_vol": m["ann_vol"],
            "max_dd": m["max_dd"],
            "cumret": m["cumulative_return"],
            "tail_ratio": m["tail_ratio"],
        })
    return pd.DataFrame(rows)


def summarize_bootstrap(df_boot: pd.DataFrame, metric_cols: list[str], cfg: BootstrapConfig) -> dict[str, float]:
    """
    Produce median + CI bounds for each metric.
    """
    out: dict[str, float] = {}
    if df_boot is None or df_boot.empty:
        return out

    q_lo = float(cfg.ci_lo)
    q_hi = float(cfg.ci_hi)
    for c in metric_cols:
        s = pd.to_numeric(df_boot.get(c), errors="coerce").dropna()
        if s.empty:
            continue
        out[f"{c}_median"] = float(s.quantile(0.50))
        out[f"{c}_ci_lo"] = float(s.quantile(q_lo))
        out[f"{c}_ci_hi"] = float(s.quantile(q_hi))
    return out


# -------------------------
# Regime slicing (market robustness)
# -------------------------

def compute_regimes_from_benchmark(
    equity_bh: np.ndarray,
    vol_window: int = 168,   # 7 days @1h
    ret_window: int = 168,   # 7 days @1h
) -> np.ndarray:
    """
    Defines regimes using benchmark (buy&hold) log-returns:
      - trend sign: rolling mean
      - vol state: rolling std relative to median

    Returns regime labels per step (length = len(equity_bh)).
    Labels:
      bull_lowvol, bull_highvol, bear_lowvol, bear_highvol
    """
    e = np.asarray(equity_bh, dtype=float)
    e = np.maximum(e, 1e-12)
    r = _equity_to_logrets(e)  # length n-1
    if r.size < max(vol_window, ret_window) + 10:
        # fallback: single regime
        return np.array(["unknown"] * e.size)

    r_s = pd.Series(r)
    mu = r_s.rolling(int(ret_window)).mean()
    vol = r_s.rolling(int(vol_window)).std()

    mu = mu.fillna(method="bfill").fillna(0.0).to_numpy()
    vol = vol.fillna(method="bfill").fillna(vol.median()).to_numpy()

    vol_med = float(np.nanmedian(vol)) if np.isfinite(np.nanmedian(vol)) else 0.0
    trend = mu >= 0.0
    highvol = vol >= vol_med

    labels = []
    for t, hv in zip(trend, highvol):
        if t and not hv:
            labels.append("bull_lowvol")
        elif t and hv:
            labels.append("bull_highvol")
        elif (not t) and (not hv):
            labels.append("bear_lowvol")
        else:
            labels.append("bear_highvol")

    # align to equity length (equity has one more point than returns)
    labels_eq = ["start"] + labels
    return np.array(labels_eq, dtype=object)


def slice_metrics_by_regime(
    equity_sac: np.ndarray,
    regimes: np.ndarray,
    freq_per_year: float,
) -> pd.DataFrame:
    """
    Compute metrics on each regime slice (using only steps belonging to that regime).
    """
    e = np.asarray(equity_sac, dtype=float)
    regimes = np.asarray(regimes, dtype=object)
    n = min(e.size, regimes.size)
    e = e[:n]
    regimes = regimes[:n]

    uniq = [x for x in pd.unique(regimes) if x is not None]
    rows = []
    for rg in uniq:
        mask = (regimes == rg)
        idx = np.where(mask)[0]
        if idx.size < 20:
            continue
        # take the equity points in that regime, keep ordering
        e_rg = e[idx]
        m = metrics_from_equity(e_rg, freq_per_year=freq_per_year)
        rows.append({
            "regime": str(rg),
            "n_points": int(idx.size),
            **m,
        })
    return pd.DataFrame(rows)


# -------------------------
# Aggregation helpers
# -------------------------

def describe_distribution(df: pd.DataFrame, cols: list[str], quantiles=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    out_rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        q = s.quantile(list(quantiles)).to_dict()
        out_rows.append({
            "metric": c,
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
            **{f"p{int(k*100)}": float(v) for k, v in q.items()},
        })
    return pd.DataFrame(out_rows)


# -------------------------
# Plot helpers
# -------------------------

def plot_box(df: pd.DataFrame, col: str, outpath: Path, title: str):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.boxplot(s.to_numpy(), vert=True)
    ax.set_title(title)
    ax.set_xticks([1])
    ax.set_xticklabels([col])
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str):
    dx = pd.to_numeric(df[x], errors="coerce")
    dy = pd.to_numeric(df[y], errors="coerce")
    m = dx.notna() & dy.notna()
    if m.sum() < 3:
        return
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(dx[m].to_numpy(), dy[m].to_numpy(), s=18)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_lines_cost_sensitivity(df_cost: pd.DataFrame, outpath: Path, title: str):
    """
    Expects columns: seed, cost_multiplier, sharpe/cumret/maxdd
    Plots median across seeds with p10/p90 band (simple, robust).
    """
    if df_cost is None or df_cost.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for metric in ["sharpe", "cumret", "maxdd"]:
        if metric not in df_cost.columns:
            continue
        g = df_cost.groupby("cost_multiplier")[metric]
        x = np.array(sorted(g.groups.keys()), dtype=float)
        med = np.array([float(g.get_group(v).median()) for v in x], dtype=float)
        p10 = np.array([float(g.get_group(v).quantile(0.10)) for v in x], dtype=float)
        p90 = np.array([float(g.get_group(v).quantile(0.90)) for v in x], dtype=float)

        ax.plot(x, med, label=f"{metric} median")
        ax.fill_between(x, p10, p90, alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("cost_multiplier")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
#
# -------------------------
# Multiple testing control (institutional)
# -------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def deflated_sharpe_ratio(sr: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Approx Deflated Sharpe Ratio.

    Conservative approximation that penalizes multiple trials.
    Returns a dict you can serialize to JSON.
    """
    n_trials = max(1, int(n_trials))
    n_obs = max(5, int(n_obs))

    sr_star = math.sqrt(2.0 * math.log(float(n_trials))) / math.sqrt(float(n_obs - 1))

    denom = 1.0 - (float(skew) * float(sr)) + ((float(kurt) - 1.0) / 4.0) * (float(sr) ** 2)
    denom = max(1e-12, float(denom))

    z = (float(sr) - float(sr_star)) * math.sqrt(float(n_obs - 1)) / math.sqrt(denom)
    p = 1.0 - _norm_cdf(float(z))

    return {
        "sr": float(sr),
        "sr_star": float(sr_star),
        "z": float(z),
        "p_value_one_sided": float(p),
        "n_trials": int(n_trials),
        "n_obs": int(n_obs),
        "skew": float(skew),
        "kurt": float(kurt),
    }


def white_reality_check(diff_returns: np.ndarray, n_boot: int = 2000, block_len: int = 48, seed: int = 0) -> dict:
    """White Reality Check via fixed-length moving block bootstrap.

    Tests H0: mean(diff_returns) <= 0 (one-sided).
    """
    rng = np.random.default_rng(int(seed))
    d = np.asarray(diff_returns, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < max(10, int(block_len) + 2):
        return {"ok": False, "reason": f"too_few_points: n={len(d)}"}

    T = int(len(d))
    obs = float(d.mean())

    boots = np.empty(int(n_boot), dtype=float)
    L = max(1, int(block_len))
    for b in range(int(n_boot)):
        idx = []
        while len(idx) < T:
            start = int(rng.integers(0, T))
            end = min(T, start + L)
            idx.extend(range(start, end))
        idx = idx[:T]
        boots[b] = float(d[idx].mean())

    p = float(np.mean(boots <= 0.0))

    return {
        "ok": True,
        "obs_mean": float(obs),
        "p_value_one_sided": float(p),
        "n": int(T),
        "n_boot": int(n_boot),
        "block_len": int(block_len),
        "boot_p05": float(np.quantile(boots, 0.05)),
        "boot_p50": float(np.quantile(boots, 0.50)),
        "boot_p95": float(np.quantile(boots, 0.95)),
    }


# -------------------------
# Stress tests (worst windows)
# -------------------------

def stress_windows_from_equity(eq: np.ndarray, window: int = 24 * 30, topk: int = 5) -> pd.DataFrame:
    """Identify worst rolling windows by cumulative return and by max drawdown."""
    e = np.asarray(eq, dtype=float)
    e = np.maximum(e, 1e-12)
    n = int(len(e))
    w = int(window)
    if n < w + 5:
        return pd.DataFrame()

    rows = []
    for i in range(0, n - w):
        seg = e[i:i + w]
        cumret = float(seg[-1] / seg[0] - 1.0)
        peak = np.maximum.accumulate(seg)
        dd = (peak - seg) / np.maximum(peak, 1e-12)
        maxdd = float(np.max(dd))
        rows.append((i, i + w, cumret, maxdd))

    df = pd.DataFrame(rows, columns=["start", "end", "cumret", "max_dd"])
    df1 = df.sort_values("cumret").head(int(topk)).assign(rank_type="worst_cumret")
    df2 = df.sort_values("max_dd", ascending=False).head(int(topk)).assign(rank_type="worst_maxdd")
    return pd.concat([df1, df2], ignore_index=True)