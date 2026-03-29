# scripts/eval/sanity_run.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from hc_ia_retail.config import env_cfg
from hc_ia_retail.data import load_ohlcv
from hc_ia_retail.features import add_features
from hc_ia_retail.env import RetailTradingEnv


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / np.maximum(peak, 1e-12))
    return float(np.max(dd)) if equity.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, default="", help="Override dataset path (else HC_DATA_PATH or config default)")
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--policy", type=str, default="all", choices=["all", "flat", "long", "random"])
    ap.add_argument("--outdir", type=str, default="", help="Output dir (default: runs/sanity_<timestamp>)")
    args = ap.parse_args()

    # --- Output dir ---
    outdir = Path(args.outdir) if args.outdir else Path("runs") / f"sanity_{_now_tag()}"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load + features ---
    data_path = args.data_path.strip() or os.environ.get("HC_DATA_PATH", "").strip() or None
    df = load_ohlcv(data_path)
    # R7 (one-change, opposable): if funding is enabled in env_cfg AND included in observation,
    # then attach the causal 1h-scaled funding feature to the dataset BEFORE splitting.
    use_funding_feature = bool(getattr(env_cfg, "include_funding_in_obs", False)) and str(
        getattr(env_cfg, "funding_mode", "none")
    ).lower() != "none"

    funding_df = None
    if use_funding_feature:
        fpath = getattr(env_cfg, "funding_path", None)
        if not fpath:
            raise ValueError(
                "include_funding_in_obs=True requires env_cfg.funding_path to be set (path to funding parquet)."
            )
        fpath = str(Path(str(fpath)).expanduser().resolve())
        funding_df = pd.read_parquet(fpath)

    df_feat, feature_cols = add_features(
        df,
       include_funding_feature=use_funding_feature,
       funding_df=funding_df,
       funding_ts_col="funding_time_utc",  # from fetch_binance_data.py
       funding_rate_col="funding_rate",
       funding_scale_hours=8.0,
       funding_feature_name="funding_rate_1h_scaled",
    )

    # --- Force “cost=0 + funding=none” for sanity accounting checks ---
    # (Your env reads from env_cfg; this avoids needing env kwargs.)
    orig_fee_bps = float(getattr(env_cfg, "fee_bps", 0.0))
    orig_spread_bps = float(getattr(env_cfg, "spread_bps", 0.0))
    orig_funding_mode = str(getattr(env_cfg, "funding_mode", "none"))

    setattr(env_cfg, "fee_bps", 0.0)
    setattr(env_cfg, "spread_bps", 0.0)
    setattr(env_cfg, "funding_mode", "none")

    def run_one(policy_name: str) -> Dict[str, Any]:
        env = RetailTradingEnv(
            df_feat,
            feature_cols=feature_cols,
            max_steps=int(args.n_steps),
        )

        obs, info0 = env.reset(seed=int(args.seed))

        rng = np.random.default_rng(int(args.seed))
        trace_rows: List[Dict[str, Any]] = []

        eq_path: List[float] = []
        bh_path: List[float] = []

        eq_bh = 1.0
        L = float(getattr(env_cfg, "max_leverage", getattr(env_cfg, "leverage_max", 1.0)))

        done = False
        steps = 0
        start_t_index = None

        while not done:
            if policy_name == "flat":
                a = 0.0
            elif policy_name == "long":
                a = 1.0
            elif policy_name == "random":
                a = float(rng.uniform(-1.0, 1.0))
            else:
                raise ValueError(policy_name)

            obs, reward, term, trunc, info = env.step(np.array([a], dtype=np.float32))
            done = bool(term or trunc)
            steps += 1

            if start_t_index is None:
                start_t_index = int(info.get("t_index", -1))

            # Equity path
            eq = float(info.get("equity", np.nan))
            eq_path.append(eq)

            # BH path computed from the *same* closes the env used (t -> t+1), in multiplicative form
            # eq *= (close[t+1]/close[t])**L  (equivalent to exp(L*logret))
            t_idx = int(info.get("t_index", -1))
            if 0 <= t_idx < (len(df_feat) - 1):
                p0 = float(df_feat.iloc[t_idx]["close"])
                p1 = float(df_feat.iloc[t_idx + 1]["close"])
                if p0 > 0 and np.isfinite(p0) and np.isfinite(p1) and p1 > 0:
                    eq_bh *= float((p1 / p0) ** L)
            bh_path.append(eq_bh)

            trace_rows.append(
                {
                    "step": steps,
                    "t_index": int(info.get("t_index", -1)),
                    "idx_next": int(info.get("idx_next", -1)),
                    "close_t": float(df_feat.iloc[t_idx]["close"]) if 0 <= t_idx < len(df_feat) else np.nan,
                    "close_t1": float(df_feat.iloc[t_idx + 1]["close"]) if 0 <= t_idx + 1 < len(df_feat) else np.nan,
                    "ret_fwd_used": float(info.get("ret_fwd_used", np.nan)),
                    "action": float(info.get("action_clipped", a)),
                    "pos_prev": float(info.get("pos_prev", np.nan)),
                    "pos_target": float(info.get("pos_target", np.nan)),
                    "cost": float(info.get("cost", np.nan)),
                    "funding_cost": float(info.get("funding_cost", np.nan)),
                    "pnl_log_net": float(info.get("pnl_log_net", np.nan)),
                    "reward": float(reward),
                    "equity": float(eq),
                    "equity_bh": float(eq_bh),
                    "equity_err": float(eq - eq_bh) if (np.isfinite(eq) and np.isfinite(eq_bh)) else np.nan,
                }
            )

            if steps >= int(args.n_steps):
                break

        eq_arr = np.asarray(eq_path, dtype=np.float64)
        bh_arr = np.asarray(bh_path, dtype=np.float64)

        # Summary checks
        max_abs_err = float(np.nanmax(np.abs(eq_arr - bh_arr))) if (eq_arr.size and bh_arr.size) else np.nan
        terminal_err = float(eq_arr[-1] - bh_arr[-1]) if (eq_arr.size and bh_arr.size) else np.nan

        summary = {
            "policy": policy_name,
            "seed": int(args.seed),
            "n_steps_requested": int(args.n_steps),
            "n_steps_done": int(steps),
            "start_t_index": int(start_t_index) if start_t_index is not None else None,
            "L": float(L),
            "equity_final": float(eq_arr[-1]) if eq_arr.size else None,
            "equity_bh_final": float(bh_arr[-1]) if bh_arr.size else None,
            "terminal_equity_err": terminal_err,
            "max_abs_equity_err": max_abs_err,
            "max_drawdown": _max_drawdown(eq_arr) if eq_arr.size else None,
        }

        # Save artifacts
        trace_df = pd.DataFrame(trace_rows)
        trace_df.to_csv(outdir / f"trace_{policy_name}.csv", index=False)

        return summary

    try:
        reports: List[Dict[str, Any]] = []
        policies = ["flat", "long", "random"] if args.policy == "all" else [args.policy]

        for pol in policies:
            reports.append(run_one(pol))

        # Global report
        report = {
            "outdir": str(outdir),
            "data_path": data_path,
            "env_cfg_snapshot": {k: getattr(env_cfg, k) for k in dir(env_cfg) if k.islower() and not k.startswith("_")},
            "reports": reports,
        }

        with open(outdir / "sanity_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"[OK] sanity artifacts written to: {outdir}")
        for r in reports:
            if r["policy"] == "long":
                print(
                    f"[always-long vs BH] equity={r['equity_final']:.10f} bh={r['equity_bh_final']:.10f} "
                    f"max_abs_err={r['max_abs_equity_err']:.3e}"
                )

    finally:
        # Restore config
        setattr(env_cfg, "fee_bps", orig_fee_bps)
        setattr(env_cfg, "spread_bps", orig_spread_bps)
        setattr(env_cfg, "funding_mode", orig_funding_mode)


if __name__ == "__main__":
    main()