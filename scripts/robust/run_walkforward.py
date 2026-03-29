from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hc_ia_retail.config import paths_cfg, sac_cfg, data_cfg  # noqa: E402
from hc_ia_retail.robustness import (  # noqa: E402
    find_latest_dir,
    parse_eval_metrics,
    describe_distribution,
    plot_box,
    plot_scatter,
    plot_lines_cost_sensitivity,
    compute_cost_sensitivity_from_eval,
    load_eval_equities,
    compute_behavior_summary_from_trace,
    BootstrapConfig,
    bootstrap_equity_metrics,
    summarize_bootstrap,
    compute_regimes_from_benchmark,
    slice_metrics_by_regime,
    deflated_sharpe_ratio,
    white_reality_check,
    stress_windows_from_equity,
)


def run_cmd(cmd: list[str]) -> None:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def snapshot_runs(out_dir: Path) -> set[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return set(out_dir.glob("run_*"))


def read_ohlcv_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_any(df: pd.DataFrame, path_no_suffix: Path) -> Path:
    """
    Prefer parquet. Fallback to csv.
    Returns the final written path.
    """
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    pq = path_no_suffix.with_suffix(".parquet")
    try:
        df.to_parquet(pq, index=False)
        return pq
    except Exception:
        csv = path_no_suffix.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


def fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["sharpe", "ann_return", "max_dd", "cumulative_return"]
    keys = [k for k in keys if k in df.columns]
    out = []
    for fold, g in df.groupby("fold"):
        row = {"fold": int(fold), "n": int(len(g))}
        for k in keys:
            s = pd.to_numeric(g[k], errors="coerce").dropna()
            if s.empty:
                continue
            row[f"{k}_mean"] = float(s.mean())
            row[f"{k}_std"] = float(s.std(ddof=0))
            row[f"{k}_p10"] = float(s.quantile(0.10))
            row[f"{k}_p50"] = float(s.quantile(0.50))
            row[f"{k}_p90"] = float(s.quantile(0.90))
        out.append(row)
    return pd.DataFrame(out).sort_values("fold")


# -----------------
# Anti cherry-pick / multiple-testing helpers
# -----------------
def _effective_trials(
    selection_mode: str,
    n_model_trials: int,
    n_seeds: int,
    n_runs: int,
) -> int:
    """Effective number of trials for multiple-testing corrections."""
    n_model_trials = max(1, int(n_model_trials))
    if selection_mode == "config":
        return n_model_trials
    if selection_mode == "seed":
        return n_model_trials * max(1, int(n_seeds))
    # seed_fold
    return n_model_trials * max(1, int(n_runs))


def _observed_sharpe_for_selection(df: pd.DataFrame, selection_mode: str) -> float:
    """Observed Sharpe consistent with the selection_mode."""
    s_all = pd.to_numeric(df.get("sharpe", pd.Series(dtype=float)), errors="coerce").dropna()
    if s_all.empty:
        return float("nan")

    if selection_mode == "config":
        # No cherry-pick: report a robust central tendency over all folds×seeds
        return float(s_all.median())

    if selection_mode == "seed":
        # Dishonest selection: pick the best seed based on average performance across folds
        if "seed" not in df.columns:
            return float(s_all.max())
        g = df.copy()
        g["sharpe"] = pd.to_numeric(g["sharpe"], errors="coerce")
        sr_by_seed = g.dropna(subset=["sharpe"]).groupby("seed")["sharpe"].mean()
        return float(sr_by_seed.max()) if not sr_by_seed.empty else float(s_all.max())

    # seed_fold
    return float(s_all.max())


def _block_bootstrap_mean(x: np.ndarray, block_len: int, rng: np.random.Generator) -> float:
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n <= 1:
        return float(np.nan)
    block_len = max(1, int(block_len))
    # number of blocks needed to reach length n
    k = int(np.ceil(n / block_len))
    starts = rng.integers(0, max(1, n - block_len + 1), size=k)
    out = []
    for s in starts:
        out.append(x[s:s + block_len])
    y = np.concatenate(out, axis=0)[:n]
    return float(np.nanmean(y))


def white_reality_check_multi(
    diffs: list[np.ndarray],
    n_boot: int,
    block_len: int,
    seed: int,
) -> dict:
    """White-style Reality Check on the max mean over multiple candidate series.

    We recenter each candidate (subtract its sample mean) to enforce the null, then
    block-bootstrap the mean and take the max across candidates.

    Returns a dict with observed max-mean and a bootstrap p-value.
    """
    cand = [np.asarray(d, dtype=float) for d in diffs if d is not None and np.asarray(d).size > 10]
    if not cand:
        return {"ok": False, "reason": "no candidate series"}

    # Observed statistic: max over candidates of mean(d)
    obs_means = [float(np.mean(d)) for d in cand]
    obs_stat = float(np.max(obs_means))

    # Null: recenter each series to mean 0
    cand0 = [d - float(np.mean(d)) for d in cand]

    rng = np.random.default_rng(int(seed))
    boot_stats = []
    for _ in range(int(n_boot)):
        bs_means = [_block_bootstrap_mean(d0, block_len=block_len, rng=rng) for d0 in cand0]
        boot_stats.append(float(np.nanmax(bs_means)))

    boot = np.asarray(boot_stats, dtype=float)
    # one-sided p-value for outperforming 0 (max-mean)
    p = float((np.sum(boot >= obs_stat) + 1.0) / (boot.size + 1.0))

    return {
        "ok": True,
        "n_candidates": int(len(cand)),
        "n_boot": int(n_boot),
        "block_len": int(block_len),
        "obs_stat_max_mean": obs_stat,
        "obs_means": {
            "min": float(np.min(obs_means)),
            "median": float(np.median(obs_means)),
            "max": float(np.max(obs_means)),
        },
        "boot_stat_quantiles": {
            "q05": float(np.quantile(boot, 0.05)),
            "q50": float(np.quantile(boot, 0.50)),
            "q95": float(np.quantile(boot, 0.95)),
        },
        "p_value_one_sided": p,
    }


def main():
    ap = argparse.ArgumentParser()

    # Walk-forward geometry
    ap.add_argument("--wf_folds", type=int, default=6, help="Number of walk-forward folds")
    ap.add_argument("--wf_train_frac", type=float, default=0.60, help="Train fraction inside each fold")
    ap.add_argument("--wf_val_frac", type=float, default=0.10, help="Val fraction inside each fold")
    ap.add_argument("--wf_test_frac", type=float, default=0.10, help="Test fraction inside each fold")
    ap.add_argument("--wf_step_frac", type=float, default=0.10, help="Step/slide fraction vs total dataset length")
    ap.add_argument("--purge_bars", type=int, default=1, help="Bars purged at split boundaries")
    ap.add_argument("--embargo_bars", type=int, default=24, help="Gap bars between val end and test start")
    ap.add_argument("--stress_window", type=int, default=24*30, help="Stress rolling window length in bars")

    ap.add_argument("--stress_topk", type=int, default=5, help="Top-K worst stress windows")

    # Anti cherry-pick / multiple-testing controls
    ap.add_argument(
        "--n_model_trials",
        type=int,
        default=1,
        help="How many distinct model variants/configs were tried in this R&D cycle (used for multiple-testing corrections).",
    )
    ap.add_argument(
        "--selection_mode",
        type=str,
        default="config",
        choices=["config", "seed", "seed_fold"],
        help=(
            "What you would cherry-pick if you were dishonest: "
            "config=none (report distribution), seed=pick best seed, seed_fold=pick best run among all folds×seeds. "
            "This changes the effective n_trials and the observed statistic used for DSR/WRC."
        ),
    )

    # Orchestration
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds")
    ap.add_argument("--timesteps", type=int, default=int(sac_cfg.total_timesteps), help="Training timesteps per seed per fold")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|mps|cuda")
    ap.add_argument("--deterministic_eval", action="store_true", help="Eval in deterministic mode")

    # Cost sensitivity
    ap.add_argument("--cost_multipliers", type=str, default="0.0,1.0,2.0,4.0", help="Cost multipliers post-hoc")

    # Statistical robustness (bootstrap)
    ap.add_argument("--bootstrap", type=int, default=300, help="Block bootstrap replications per seed per fold")
    ap.add_argument("--block_len", type=int, default=48, help="Block length (bars) for bootstrap")
    ap.add_argument("--ci_lo", type=float, default=0.05, help="CI lower quantile (e.g., 0.05)")
    ap.add_argument("--ci_hi", type=float, default=0.95, help="CI upper quantile (e.g., 0.95)")

    # Data override (optional)
    ap.add_argument("--data_path", type=str, default="", help="Override base dataset path (otherwise data_cfg.csv_path)")

    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    cost_multipliers = [float(x.strip()) for x in args.cost_multipliers.split(",") if x.strip() != ""]
    freq = float(getattr(data_cfg, "freq_per_year", 24 * 365))

    base_path = Path(args.data_path) if args.data_path else Path(getattr(data_cfg, "csv_path"))
    df_raw = read_ohlcv_any(base_path)
    if len(df_raw) < 500:
        raise RuntimeError(f"Dataset too small for walk-forward: n={len(df_raw)}")

    # fold span and step in absolute rows (based on total dataset)
    total_n = len(df_raw)
    fold_span_frac = float(args.wf_train_frac + args.wf_val_frac + args.wf_test_frac)
    if fold_span_frac <= 0.20:
        raise RuntimeError("Fold span too small. Increase train/val/test fractions.")
    fold_span = int(round(fold_span_frac * total_n))
    step = int(round(float(args.wf_step_frac) * total_n))
    fold_span = max(200, fold_span)
    step = max(1, step)

    wf_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    robust_dir = Path(paths_cfg.out_dir) / f"walkforward_{wf_id}"
    (robust_dir / "plots").mkdir(parents=True, exist_ok=True)
    wf_dir = robust_dir / "walkforward"
    wf_dir.mkdir(parents=True, exist_ok=True)

    # bootstrap config
    boot_cfg = BootstrapConfig(
        n_boot=int(args.bootstrap),
        block_len=int(args.block_len),
        ci_lo=float(args.ci_lo),
        ci_hi=float(args.ci_hi),
    )

    # outputs
    rows = []
    behavior_rows = []
    boot_rows = []
    boot_summary_rows = []
    regime_rows = []
    cost_rows = []
    manifest_rows = []
    wrc_series = []
    stress_rows = []

    # Build fold files
    for k in range(int(args.wf_folds)):
        start = k * step
        end = start + fold_span
        if end > total_n:
            break

        df_fold = df_raw.iloc[start:end].copy()

        # Capture timestamps if present
        ts0 = None
        ts1 = None
        if "timestamp" in df_fold.columns:
            try:
                ts0 = str(df_fold["timestamp"].iloc[0])
                ts1 = str(df_fold["timestamp"].iloc[-1])
            except Exception:
                ts0, ts1 = None, None

        n = int(len(df_fold))
        purge = max(0, int(args.purge_bars))
        embargo = max(0, int(args.embargo_bars))

        train_end = int(round(float(args.wf_train_frac) * n))
        val_end = int(round((float(args.wf_train_frac) + float(args.wf_val_frac)) * n))

        train_end_p = max(1, train_end - purge)
        val_start = train_end  # keep original boundary start for val
        val_end_p = max(val_start + 1, val_end - purge)

        test_start = min(n, val_end + embargo)
        test_len = int(round(float(args.wf_test_frac) * n))
        test_end = min(n, test_start + max(1, test_len))

        if train_end_p < 50 or (val_end_p - val_start) < 20 or (test_end - test_start) < 20:
            continue

        split_spec = {
            "train": {"start": 0, "end": int(train_end_p)},
            "val": {"start": int(val_start), "end": int(val_end_p)},
            "test": {"start": int(test_start), "end": int(test_end)},
            "purge_bars": int(purge),
            "embargo_bars": int(embargo),
            "n_rows_fold": int(n),
        }
        split_path_no_suffix = wf_dir / f"fold_{k:02d}_split"
        split_path = split_path_no_suffix.with_suffix(".json")
        split_path.write_text(json.dumps(split_spec, indent=2), encoding="utf-8")

        fold_path = write_any(df_fold, wf_dir / f"fold_{k:02d}")
        manifest_rows.append({
            "fold": int(k),
            "start_idx": int(start),
            "end_idx": int(end),
            "n_rows": int(len(df_fold)),
            "fold_path": str(fold_path),
            "ts_start": ts0,
            "ts_end": ts1,
            "split_path": str(split_path),
            "purge_bars": int(purge),
            "embargo_bars": int(embargo),
            "test_start_rel": int(test_start),
            "test_end_rel": int(test_end),
        })

    if not manifest_rows:
        raise RuntimeError("No folds could be constructed. Reduce wf_folds or wf_step_frac.")

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(wf_dir / "manifest.csv", index=False)

    # Run folds × seeds
    for m in manifest_rows:
        fold = int(m["fold"])
        fold_path = str(m["fold_path"])
        split_path = str(m.get("split_path", ""))

        for seed in seeds:
            before = snapshot_runs(Path(paths_cfg.out_dir))

            # TRAIN on fold dataset (internal split train/val/test stays identical to your pipeline)
            cmd_train = [
                sys.executable, "scripts/train/train.py",
                "--device", str(args.device),
                "--timesteps", str(int(args.timesteps)),
                "--seed", str(int(seed)),
                "--data_path", str(fold_path),
            ]
            if split_path:
                cmd_train.extend(["--split_json", str(split_path)])
            run_cmd(cmd_train)

            after = snapshot_runs(Path(paths_cfg.out_dir))
            new_runs = sorted(list(after - before), key=lambda p: p.stat().st_mtime, reverse=True)
            run_dir = new_runs[0] if new_runs else find_latest_dir(Path(paths_cfg.out_dir), "run_*")
            if run_dir is None:
                raise FileNotFoundError("Could not resolve run_dir after training")

            # EVAL OOS on fold dataset (eval will take the fold's internal OOS/test)
            cmd_eval = [
                sys.executable, "scripts/eval/eval_oos.py",
                "--run_dir", str(run_dir),
                "--data_path", str(fold_path),
            ]
            if split_path:
                cmd_eval.extend(["--split_json", str(split_path)])
            if args.deterministic_eval:
                cmd_eval.append("--deterministic")
            run_cmd(cmd_eval)

            eval_dir = find_latest_dir(run_dir, "eval_*")
            if eval_dir is None:
                raise FileNotFoundError(f"No eval_* dir found in {run_dir}")

            # 1) base metrics (metrics.json)
            r = parse_eval_metrics(eval_dir)
            r["wf_id"] = wf_id
            r["fold"] = int(fold)
            r["seed"] = int(seed)
            r["fold_start_idx"] = int(m["start_idx"])
            r["fold_end_idx"] = int(m["end_idx"])
            r["fold_path"] = str(fold_path)
            r["fold_ts_start"] = m.get("ts_start")
            r["fold_ts_end"] = m.get("ts_end")
            rows.append(r)

            # 2) behavior tails (trace)
            beh = compute_behavior_summary_from_trace(eval_dir)
            if beh:
                behavior_rows.append({
                    "wf_id": wf_id,
                    "fold": int(fold),
                    "seed": int(seed),
                    "eval_dir": str(eval_dir),
                    **beh,
                })

            # 3) cost sensitivity (post-hoc)
            cs = compute_cost_sensitivity_from_eval(eval_dir, multipliers=list(cost_multipliers), freq_per_year=freq)
            if cs is not None and not cs.empty:
                cs.insert(0, "wf_id", wf_id)
                cs.insert(1, "fold", int(fold))
                cs.insert(2, "seed", int(seed))
                cs.insert(3, "eval_dir", str(eval_dir))
                cost_rows.append(cs)

            # 4) bootstrap on OOS equity
            eq_df = load_eval_equities(eval_dir)
            if (eq_df is not None) and (not eq_df.empty) and ("equity_sac" in eq_df.columns):
                eq = pd.to_numeric(eq_df["equity_sac"], errors="coerce").ffill().fillna(1.0).to_numpy(dtype=float)

                df_boot = bootstrap_equity_metrics(eq, freq_per_year=freq, cfg=boot_cfg, seed=int(seed))
                if df_boot is not None and not df_boot.empty:
                    df_boot.insert(0, "wf_id", wf_id)
                    df_boot.insert(1, "fold", int(fold))
                    df_boot.insert(2, "seed", int(seed))
                    df_boot.insert(3, "eval_dir", str(eval_dir))
                    boot_rows.append(df_boot)

                    summ = summarize_bootstrap(
                        df_boot,
                        metric_cols=["sharpe", "ann_return", "ann_vol", "max_dd", "cumret", "tail_ratio"],
                        cfg=boot_cfg,
                    )
                    if summ:
                        boot_summary_rows.append({
                            "wf_id": wf_id,
                            "fold": int(fold),
                            "seed": int(seed),
                            "eval_dir": str(eval_dir),
                            **summ,
                            "block_len": int(boot_cfg.block_len),
                            "n_boot": int(boot_cfg.n_boot),
                            "ci_lo": float(boot_cfg.ci_lo),
                            "ci_hi": float(boot_cfg.ci_hi),
                        })

                # 5) regime slicing if benchmark present
                if "equity_bh" in eq_df.columns:
                    bh = pd.to_numeric(eq_df["equity_bh"], errors="coerce").ffill().fillna(1.0).to_numpy(dtype=float)
                    regimes = compute_regimes_from_benchmark(bh, vol_window=168, ret_window=168)
                    df_rg = slice_metrics_by_regime(eq, regimes=regimes, freq_per_year=freq)
                    if df_rg is not None and not df_rg.empty:
                        df_rg.insert(0, "wf_id", wf_id)
                        df_rg.insert(1, "fold", int(fold))
                        df_rg.insert(2, "seed", int(seed))
                        df_rg.insert(3, "eval_dir", str(eval_dir))
                        regime_rows.append(df_rg)

            # White Reality Check and stress tests
            if (eq_df is not None) and (not eq_df.empty) and ("equity_sac" in eq_df.columns) and ("equity_bh" in eq_df.columns):
                es = pd.to_numeric(eq_df["equity_sac"], errors="coerce").ffill().fillna(1.0).to_numpy(dtype=float)
                eb = pd.to_numeric(eq_df["equity_bh"], errors="coerce").ffill().fillna(1.0).to_numpy(dtype=float)
                es = np.maximum(es, 1e-12)
                eb = np.maximum(eb, 1e-12)
                rs = np.diff(np.log(es))
                rb = np.diff(np.log(eb))
                d = rs - rb
                if d.size > 10:
                    wrc_series.append({"fold": int(fold), "seed": int(seed), "d": d})
            if (eq_df is not None) and (not eq_df.empty) and ("equity_sac" in eq_df.columns):
                es2 = pd.to_numeric(eq_df["equity_sac"], errors="coerce").ffill().fillna(1.0).to_numpy(dtype=float)
                df_st = stress_windows_from_equity(es2, window=int(args.stress_window), topk=int(args.stress_topk))
                if df_st is not None and not df_st.empty:
                    df_st.insert(0, "wf_id", wf_id)
                    df_st.insert(1, "fold", int(fold))
                    df_st.insert(2, "seed", int(seed))
                    df_st.insert(3, "eval_dir", str(eval_dir))
                    stress_rows.append(df_st)

    # -----------------
    # Exports
    # -----------------
    df = pd.DataFrame(rows)
    df.to_csv(robust_dir / "walkforward_runs.csv", index=False)

    dist = describe_distribution(
        df,
        cols=[c for c in [
            "sharpe", "sortino", "calmar", "ann_return", "ann_vol", "cumulative_return", "max_dd",
            "cvar_95", "cvar_99", "tail_ratio",
            "mean_abs_pos", "turnover_mean", "pct_flat",
            "n_trades", "win_rate", "profit_factor",
        ] if c in df.columns],
        quantiles=(0.1, 0.5, 0.9),
    )
    dist.to_csv(robust_dir / "walkforward_summary_overall.csv", index=False)

    df_fold = fold_summary(df)
    df_fold.to_csv(robust_dir / "walkforward_summary_by_fold.csv", index=False)

    if behavior_rows:
        pd.DataFrame(behavior_rows).to_csv(robust_dir / "walkforward_behavior.csv", index=False)

    if boot_rows:
        pd.concat(boot_rows, ignore_index=True).to_csv(robust_dir / "bootstrap_samples.csv", index=False)

    if boot_summary_rows:
        pd.DataFrame(boot_summary_rows).to_csv(robust_dir / "bootstrap_summary.csv", index=False)

    if regime_rows:
        pd.concat(regime_rows, ignore_index=True).to_csv(robust_dir / "regime_metrics.csv", index=False)

    if cost_rows:
        df_cost = pd.concat(cost_rows, ignore_index=True)
        df_cost.to_csv(robust_dir / "cost_sensitivity.csv", index=False)

    if stress_rows:
        pd.concat(stress_rows, ignore_index=True).to_csv(robust_dir / "stress_tests.csv", index=False)

    # -----------------
    # Anti cherry-pick / multiple-testing controls
    # -----------------
    mt_out = None
    if "sharpe" in df.columns:
        s_all = pd.to_numeric(df["sharpe"], errors="coerce").dropna()
        if not s_all.empty:
            n_runs = int(len(s_all))
            n_seeds_eff = int(df["seed"].nunique()) if "seed" in df.columns else int(len(seeds))

            sr_obs = _observed_sharpe_for_selection(df, selection_mode=str(args.selection_mode))
            # approximate n_obs by median OOS steps if present, else fallback
            if "n_steps_oos" in df.columns:
                n_obs = int(pd.to_numeric(df["n_steps_oos"], errors="coerce").dropna().median())
            else:
                n_obs = 1000

            n_trials_eff = _effective_trials(
                selection_mode=str(args.selection_mode),
                n_model_trials=int(args.n_model_trials),
                n_seeds=n_seeds_eff,
                n_runs=n_runs,
            )

            dsr = deflated_sharpe_ratio(sr=float(sr_obs), n_trials=int(n_trials_eff), n_obs=int(max(10, n_obs)))
            mt_out = {
                "ok": True,
                "selection_mode": str(args.selection_mode),
                "n_model_trials": int(max(1, args.n_model_trials)),
                "n_trials_effective": int(n_trials_eff),
                "n_runs": int(n_runs),
                "n_seeds": int(n_seeds_eff),
                "sr_observed_for_selection": float(sr_obs),
                "sr_all": {
                    "mean": float(s_all.mean()),
                    "median": float(s_all.median()),
                    "p10": float(s_all.quantile(0.10)),
                    "p90": float(s_all.quantile(0.90)),
                    "max": float(s_all.max()),
                },
                "n_obs_median": int(n_obs),
                "dsr": dsr,
                "note": (
                    "DSR corrects for multiple testing *only* if n_model_trials reflects how many distinct variants you compared. "
                    "If you also cherry-pick seeds/runs, use selection_mode=seed or seed_fold."
                ),
            }
            (robust_dir / "multiple_testing.json").write_text(json.dumps(mt_out, indent=2), encoding="utf-8")

    # White Reality Check vs Buy&Hold, consistent with selection_mode
    wrc_out = None
    if wrc_series:
        # Build candidate diff-return series depending on selection_mode
        if str(args.selection_mode) == "config":
            # single candidate = concatenation across folds×seeds
            d_all = np.concatenate([x["d"] for x in wrc_series], axis=0)
            wrc = white_reality_check(d_all, n_boot=2000, block_len=int(args.block_len), seed=0)
            wrc_out = {
                "ok": True,
                "selection_mode": "config",
                "n_candidates": 1,
                "n_model_trials": int(max(1, args.n_model_trials)),
                "result": wrc,
                "note": (
                    "This tests outperformance vs BH for the *given* configuration. "
                    "If you cherry-pick seeds/runs, use selection_mode=seed or seed_fold to test the max-over-candidates."
                ),
            }
        elif str(args.selection_mode) == "seed":
            by_seed = {}
            for x in wrc_series:
                by_seed.setdefault(int(x["seed"]), []).append(x["d"])
            candidates = [np.concatenate(v, axis=0) for v in by_seed.values() if len(v) > 0]
            wrc = white_reality_check_multi(candidates, n_boot=2000, block_len=int(args.block_len), seed=0)
            wrc_out = {
                "ok": True,
                "selection_mode": "seed",
                "n_candidates": int(len(candidates)),
                "n_model_trials": int(max(1, args.n_model_trials)),
                "result": wrc,
                "note": "This is a max-over-seeds test: would the best seed still be significant after selection?",
            }
        else:
            candidates = [x["d"] for x in wrc_series]
            wrc = white_reality_check_multi(candidates, n_boot=2000, block_len=int(args.block_len), seed=0)
            wrc_out = {
                "ok": True,
                "selection_mode": "seed_fold",
                "n_candidates": int(len(candidates)),
                "n_model_trials": int(max(1, args.n_model_trials)),
                "result": wrc,
                "note": "This is a max-over-runs test: would the best run among folds×seeds be significant after selection?",
            }

        (robust_dir / "white_reality_check.json").write_text(json.dumps(wrc_out, indent=2), encoding="utf-8")

    # -----------------
    # Plots
    # -----------------
    plot_box(df, "sharpe", robust_dir / "plots" / "box_sharpe.png", "Walk-forward — Sharpe (all folds × seeds)")
    plot_box(df, "max_dd", robust_dir / "plots" / "box_maxdd.png", "Walk-forward — MaxDD (all folds × seeds)")
    plot_box(df, "ann_return", robust_dir / "plots" / "box_ann_return.png", "Walk-forward — Annualized return (all folds × seeds)")

    if "turnover_mean" in df.columns and "sharpe" in df.columns:
        plot_scatter(df, "turnover_mean", "sharpe", robust_dir / "plots" / "scatter_turnover_vs_sharpe.png",
                    "Walk-forward — Turnover vs Sharpe (all folds × seeds)")

    if cost_rows:
        plot_lines_cost_sensitivity(
            df_cost,
            robust_dir / "plots" / "cost_sensitivity_lines.png",
            "Walk-forward — Cost sensitivity (median + p10/p90 across folds×seeds)",
        )

    # fold-by-fold sharpe line plot
    if "fold" in df.columns and "sharpe" in df.columns:
        g = df.groupby("fold")["sharpe"]
        x = np.array(sorted(g.groups.keys()), dtype=float)
        med = np.array([float(g.get_group(v).median()) for v in x], dtype=float)
        p10 = np.array([float(g.get_group(v).quantile(0.10)) for v in x], dtype=float)
        p90 = np.array([float(g.get_group(v).quantile(0.90)) for v in x], dtype=float)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(x, med)
        ax.fill_between(x, p10, p90, alpha=0.15)
        ax.set_title("Walk-forward — Sharpe by fold (median + p10/p90 across seeds)")
        ax.set_xlabel("fold")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(robust_dir / "plots" / "wf_fold_sharpe.png")
        plt.close(fig)

    # -----------------
    # Meta + report
    # -----------------
    meta = {
        "anti_cherrypick": {
            "n_model_trials": int(max(1, args.n_model_trials)),
            "selection_mode": str(args.selection_mode),
            "multiple_testing_json": bool((robust_dir / "multiple_testing.json").exists()),
            "white_reality_check_json": bool((robust_dir / "white_reality_check.json").exists()),
        },
        "wf_id": wf_id,
        "base_data_path": str(base_path),
        "total_rows": int(total_n),
        "fold_span_rows": int(fold_span),
        "step_rows": int(step),
        "wf_folds_requested": int(args.wf_folds),
        "wf_train_frac": float(args.wf_train_frac),
        "wf_val_frac": float(args.wf_val_frac),
        "wf_test_frac": float(args.wf_test_frac),
        "wf_step_frac": float(args.wf_step_frac),
        "seeds": seeds,
        "timesteps_per_seed_per_fold": int(args.timesteps),
        "device": str(args.device),
        "deterministic_eval": bool(args.deterministic_eval),
        "freq_per_year": float(freq),
        "cost_multipliers": list(cost_multipliers),
        "bootstrap": {
            "n_boot": int(boot_cfg.n_boot),
            "block_len": int(boot_cfg.block_len),
            "ci_lo": float(boot_cfg.ci_lo),
            "ci_hi": float(boot_cfg.ci_hi),
        },
        "exports": {
            "walkforward_runs.csv": True,
            "walkforward_summary_overall.csv": True,
            "walkforward_summary_by_fold.csv": True,
            "walkforward_behavior.csv": bool(behavior_rows),
            "bootstrap_samples.csv": bool(boot_rows),
            "bootstrap_summary.csv": bool(boot_summary_rows),
            "regime_metrics.csv": bool(regime_rows),
            "cost_sensitivity.csv": bool(cost_rows),
        },
    }
    with (robust_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = []
    report.append("# HC_IA_RETAIL — Walk-forward / Rolling OOS Outputs\n\n")
    report.append(f"- wf_id: `{wf_id}`\n")
    report.append(f"- base_data_path: `{base_path}`\n")
    report.append(f"- seeds: `{seeds}`\n")
    report.append(f"- timesteps per seed per fold: `{int(args.timesteps)}`\n")
    report.append(f"- device: `{args.device}`\n")
    report.append(f"- deterministic_eval: `{bool(args.deterministic_eval)}`\n")
    report.append(f"- freq_per_year: `{freq}`\n")
    report.append(f"- folds built: `{len(manifest_rows)}`\n\n")

    report.append("## Core\n")
    report.append("- `walkforward_runs.csv` (1 ligne = 1 fold × 1 seed)\n")
    report.append("- `walkforward_summary_by_fold.csv` (distribution par fold)\n")
    report.append("- `walkforward_summary_overall.csv` (distribution globale)\n")
    report.append("- `stress_tests.csv` (worst rolling windows)\n")
    report.append("- `multiple_testing.json` (Deflated Sharpe Ratio; requires setting --n_model_trials honestly)\n")
    report.append("- `white_reality_check.json` (Reality-check vs BH; selection-aware via --selection_mode)\n\n")

    report.append("## Plots\n")
    report.append("- `plots/wf_fold_sharpe.png`\n")
    report.append("- `plots/box_sharpe.png`\n")
    report.append("- `plots/box_maxdd.png`\n")
    report.append("- `plots/box_ann_return.png`\n")
    report.append("- `plots/scatter_turnover_vs_sharpe.png`\n")
    if cost_rows:
        report.append("- `plots/cost_sensitivity_lines.png`\n")

    (robust_dir / "report.md").write_text("".join(report), encoding="utf-8")

    print(f"[OK] walk-forward dir -> {robust_dir}")
    print("[OK] Walk-forward complete: distribution across folds × seeds generated")


if __name__ == "__main__":
    main()