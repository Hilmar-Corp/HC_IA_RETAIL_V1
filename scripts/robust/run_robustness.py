from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
)


def run_cmd(cmd: list[str]) -> None:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def snapshot_runs(out_dir: Path) -> set[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return set(out_dir.glob("run_*"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds")

    ap.add_argument("--timesteps", type=int, default=int(sac_cfg.total_timesteps), help="Training timesteps per seed")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|mps|cuda")

    ap.add_argument("--deterministic_eval", action="store_true", help="Eval in deterministic mode")

    # Cost sensitivity
    ap.add_argument("--cost_multipliers", type=str, default="0.0,1.0,2.0,4.0", help="Cost multipliers post-hoc")

    # Statistical robustness (bootstrap)
    ap.add_argument("--bootstrap", type=int, default=500, help="Block bootstrap replications per seed")
    ap.add_argument("--block_len", type=int, default=48, help="Block length (bars) for bootstrap")
    ap.add_argument("--ci_lo", type=float, default=0.05, help="CI lower quantile (e.g., 0.05)")
    ap.add_argument("--ci_hi", type=float, default=0.95, help="CI upper quantile (e.g., 0.95)")

    # Regime slicing
    ap.add_argument("--regime_vol_window", type=int, default=168, help="Vol window (bars) for regime split")
    ap.add_argument("--regime_ret_window", type=int, default=168, help="Trend window (bars) for regime split")

    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    cost_multipliers = [float(x.strip()) for x in args.cost_multipliers.split(",") if x.strip() != ""]

    freq = float(getattr(data_cfg, "freq_per_year", 24 * 365))

    robust_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    robust_dir = Path(paths_cfg.out_dir) / f"robust_{robust_id}"
    robust_dir.mkdir(parents=True, exist_ok=True)
    (robust_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Core outputs
    rows = []
    behavior_rows = []
    boot_rows = []
    boot_summary_rows = []
    regime_rows = []
    cost_rows = []

    # bootstrap config
    boot_cfg = BootstrapConfig(
        n_boot=int(args.bootstrap),
        block_len=int(args.block_len),
        ci_lo=float(args.ci_lo),
        ci_hi=float(args.ci_hi),
    )

    for seed in seeds:
        before = snapshot_runs(Path(paths_cfg.out_dir))

        # -----------------
        # TRAIN
        # -----------------
        cmd_train = [
            sys.executable, "scripts/train/train.py",
            "--device", str(args.device),
            "--timesteps", str(int(args.timesteps)),
            "--seed", str(int(seed)),
        ]
        run_cmd(cmd_train)

        after = snapshot_runs(Path(paths_cfg.out_dir))
        new_runs = sorted(list(after - before), key=lambda p: p.stat().st_mtime, reverse=True)
        run_dir = new_runs[0] if new_runs else find_latest_dir(Path(paths_cfg.out_dir), "run_*")
        if run_dir is None:
            raise FileNotFoundError("Could not resolve run_dir after training")

        # -----------------
        # EVAL (OOS)
        # -----------------
        cmd_eval = [sys.executable, "scripts/eval/eval_oos.py", "--run_dir", str(run_dir)]
        if args.deterministic_eval:
            cmd_eval.append("--deterministic")
        run_cmd(cmd_eval)

        eval_dir = find_latest_dir(run_dir, "eval_*")
        if eval_dir is None:
            raise FileNotFoundError(f"No eval_* dir found in {run_dir}")

        # -----------------
        # 1) Base metrics (from metrics.json)
        # -----------------
        r = parse_eval_metrics(eval_dir)
        r["seed"] = int(seed)
        rows.append(r)

        # -----------------
        # 2) Behavioral tails (from trace_oos.csv)
        # -----------------
        beh = compute_behavior_summary_from_trace(eval_dir)
        if beh:
            beh_row = {"seed": int(seed), "eval_dir": str(eval_dir), **beh}
            behavior_rows.append(beh_row)

        # -----------------
        # 3) Cost sensitivity (post-hoc)
        # -----------------
        cs = compute_cost_sensitivity_from_eval(eval_dir, multipliers=cost_multipliers, freq_per_year=freq)
        if cs is not None and not cs.empty:
            cs.insert(0, "seed", int(seed))
            cs.insert(1, "eval_dir", str(eval_dir))
            cost_rows.append(cs)

        # -----------------
        # 4) Statistical robustness (block bootstrap on OOS equity)
        # -----------------
        eq_df = load_eval_equities(eval_dir)
        if (eq_df is not None) and (not eq_df.empty) and ("equity_sac" in eq_df.columns):
            eq = pd.to_numeric(eq_df["equity_sac"], errors="coerce").fillna(method="ffill").fillna(1.0).to_numpy(dtype=float)

            df_boot = bootstrap_equity_metrics(eq, freq_per_year=freq, cfg=boot_cfg, seed=int(seed))
            if df_boot is not None and not df_boot.empty:
                df_boot.insert(0, "seed", int(seed))
                df_boot.insert(1, "eval_dir", str(eval_dir))
                boot_rows.append(df_boot)

                summ = summarize_bootstrap(
                    df_boot,
                    metric_cols=["sharpe", "ann_return", "ann_vol", "max_dd", "cumret", "tail_ratio"],
                    cfg=boot_cfg,
                )
                if summ:
                    boot_summary_rows.append({
                        "seed": int(seed),
                        "eval_dir": str(eval_dir),
                        **summ,
                        "block_len": int(boot_cfg.block_len),
                        "n_boot": int(boot_cfg.n_boot),
                        "ci_lo": float(boot_cfg.ci_lo),
                        "ci_hi": float(boot_cfg.ci_hi),
                    })

            # -----------------
            # 5) Regime slicing (using benchmark if present)
            # -----------------
            if "equity_bh" in eq_df.columns:
                bh = pd.to_numeric(eq_df["equity_bh"], errors="coerce").fillna(method="ffill").fillna(1.0).to_numpy(dtype=float)
                regimes = compute_regimes_from_benchmark(
                    bh,
                    vol_window=int(args.regime_vol_window),
                    ret_window=int(args.regime_ret_window),
                )
                df_rg = slice_metrics_by_regime(eq, regimes=regimes, freq_per_year=freq)
                if df_rg is not None and not df_rg.empty:
                    df_rg.insert(0, "seed", int(seed))
                    df_rg.insert(1, "eval_dir", str(eval_dir))
                    regime_rows.append(df_rg)

    # -----------------
    # Exports
    # -----------------
    df = pd.DataFrame(rows)
    df.to_csv(robust_dir / "robust_runs.csv", index=False)

    # distribution summary (across seeds)
    metrics_cols = [
        "sharpe", "sortino", "calmar", "ann_return", "ann_vol", "cumulative_return", "max_dd",
        "cvar_95", "cvar_99", "tail_ratio",
        "mean_abs_pos", "turnover_mean", "pct_flat",
        "n_trades", "win_rate", "profit_factor",
    ]
    dist = describe_distribution(df, cols=[c for c in metrics_cols if c in df.columns], quantiles=(0.1, 0.5, 0.9))
    dist.to_csv(robust_dir / "robust_summary.csv", index=False)

    # behavior exports
    if behavior_rows:
        df_beh = pd.DataFrame(behavior_rows)
        df_beh.to_csv(robust_dir / "robust_behavior.csv", index=False)

    # bootstrap exports
    if boot_rows:
        df_boot_all = pd.concat(boot_rows, ignore_index=True)
        df_boot_all.to_csv(robust_dir / "bootstrap_samples.csv", index=False)

    if boot_summary_rows:
        df_boot_sum = pd.DataFrame(boot_summary_rows)
        df_boot_sum.to_csv(robust_dir / "bootstrap_summary.csv", index=False)

    # regime exports
    if regime_rows:
        df_rg_all = pd.concat(regime_rows, ignore_index=True)
        df_rg_all.to_csv(robust_dir / "regime_metrics.csv", index=False)

    # cost exports
    if cost_rows:
        df_cost = pd.concat(cost_rows, ignore_index=True)
        df_cost.to_csv(robust_dir / "cost_sensitivity.csv", index=False)

    # -----------------
    # Plots
    # -----------------
    plot_box(df, "sharpe", robust_dir / "plots" / "box_sharpe.png", "Robustness — Sharpe (across seeds)")
    plot_box(df, "max_dd", robust_dir / "plots" / "box_maxdd.png", "Robustness — MaxDD (across seeds)")
    plot_box(df, "ann_return", robust_dir / "plots" / "box_ann_return.png", "Robustness — Annualized return (across seeds)")
    if "turnover_mean" in df.columns and "sharpe" in df.columns:
        plot_scatter(df, "turnover_mean", "sharpe", robust_dir / "plots" / "scatter_turnover_vs_sharpe.png", "Robustness — Turnover vs Sharpe")

    if cost_rows:
        plot_lines_cost_sensitivity(
            pd.concat(cost_rows, ignore_index=True),
            robust_dir / "plots" / "cost_sensitivity_lines.png",
            "Robustness — Cost sensitivity (median + p10/p90 across seeds)",
        )

    # -----------------
    # Meta + report
    # -----------------
    meta = {
        "robust_id": robust_id,
        "seeds": seeds,
        "timesteps": int(args.timesteps),
        "device": str(args.device),
        "deterministic_eval": bool(args.deterministic_eval),
        "freq_per_year": float(freq),
        "cost_multipliers": cost_multipliers,
        "bootstrap": {
            "n_boot": int(boot_cfg.n_boot),
            "block_len": int(boot_cfg.block_len),
            "ci_lo": float(boot_cfg.ci_lo),
            "ci_hi": float(boot_cfg.ci_hi),
        },
        "regime": {
            "vol_window": int(args.regime_vol_window),
            "ret_window": int(args.regime_ret_window),
        },
        "exports": {
            "robust_runs.csv": True,
            "robust_summary.csv": True,
            "robust_behavior.csv": bool(behavior_rows),
            "bootstrap_samples.csv": bool(boot_rows),
            "bootstrap_summary.csv": bool(boot_summary_rows),
            "regime_metrics.csv": bool(regime_rows),
            "cost_sensitivity.csv": bool(cost_rows),
        },
    }
    with (robust_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = []
    report.append("# HC_IA_RETAIL — Robustness Protocol Outputs\n\n")
    report.append(f"- robust_id: `{robust_id}`\n")
    report.append(f"- seeds: `{seeds}`\n")
    report.append(f"- timesteps per seed: `{int(args.timesteps)}`\n")
    report.append(f"- device: `{args.device}`\n")
    report.append(f"- deterministic_eval: `{bool(args.deterministic_eval)}`\n")
    report.append(f"- freq_per_year: `{freq}`\n\n")

    report.append("## Seed-distribution (core)\n")
    report.append("- `robust_runs.csv` (1 ligne = 1 seed)\n")
    report.append("- `robust_summary.csv` (mean/std/p10/p50/p90)\n\n")

    report.append("## Behavior tails (institutional)\n")
    report.append("- `robust_behavior.csv` (turnover/abs_pos tails, cost tails, pnl tails)\n\n")

    report.append("## Statistical robustness (block bootstrap)\n")
    report.append(f"- `bootstrap_summary.csv` (median + CI [{boot_cfg.ci_lo:.2f},{boot_cfg.ci_hi:.2f}] par seed)\n")
    report.append(f"- `bootstrap_samples.csv` (samples bruts, n_boot={boot_cfg.n_boot}, block_len={boot_cfg.block_len})\n\n")

    report.append("## Regime robustness\n")
    report.append("- `regime_metrics.csv` (métriques par régime bull/bear x low/high vol)\n\n")

    report.append("## Cost sensitivity\n")
    report.append("- `cost_sensitivity.csv` (+ `plots/cost_sensitivity_lines.png`)\n\n")

    report.append("## Plots\n")
    report.append("- `plots/box_sharpe.png`\n")
    report.append("- `plots/box_maxdd.png`\n")
    report.append("- `plots/box_ann_return.png`\n")
    report.append("- `plots/scatter_turnover_vs_sharpe.png`\n")
    if cost_rows:
        report.append("- `plots/cost_sensitivity_lines.png`\n")

    (robust_dir / "report.md").write_text("".join(report), encoding="utf-8")

    print(f"[OK] robust_dir -> {robust_dir}")
    print("[OK] Robustness artifacts generated (seed-dist + behavior tails + bootstrap + regimes + cost sensitivity)")


if __name__ == "__main__":
    main()