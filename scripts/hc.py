#!/usr/bin/env python3
"""
scripts/hc.py

Single entrypoint for HC_IA_RETAIL:
- train
- resume
- pause (via run_dir/PAUSE file)
- eval
- diag (runs eval twice: deterministic vs stochastic)

Works by calling existing scripts:
- scripts/train/train.py
- scripts/eval/eval_oos.py
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


def _py() -> str:
    return sys.executable


def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _latest_run_dir() -> Path:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"runs/ not found at: {RUNS_DIR}")
    candidates = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("No run directories found in runs/")
    # lexicographic sort works if your run dirs are timestamped
    return sorted(candidates)[-1]


def _resolve_run_dir(run_dir: Optional[str]) -> Path:
    if run_dir is None or run_dir.strip() == "":
        return _latest_run_dir()
    p = Path(run_dir)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"run_dir not found: {p}")
    return p


def cmd_train(args: argparse.Namespace) -> None:
    train_script = REPO_ROOT / "scripts" / "train" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing: {train_script}")

    cmd = [
        _py(),
        str(train_script),
        "--device", args.device,
    ]
    if args.timesteps is not None:
        cmd += ["--timesteps", str(args.timesteps)]
    if args.data_path is not None:
        cmd += ["--data_path", args.data_path]
    if args.eval_freq is not None:
        cmd += ["--eval_freq", str(args.eval_freq)]
    if args.n_eval_episodes is not None:
        cmd += ["--n_eval_episodes", str(args.n_eval_episodes)]
    if args.checkpoint_freq is not None:
        cmd += ["--checkpoint_freq", str(args.checkpoint_freq)]
    if args.chunk is not None:
        cmd += ["--chunk", str(args.chunk)]

    _run(cmd)

    # Auto-eval/diag on latest run
    if args.auto_eval or args.diag:
        run_dir = _latest_run_dir()
        if args.auto_eval:
            _eval(run_dir=run_dir, deterministic=args.deterministic_eval)
        if args.diag:
            _diag(run_dir=run_dir)


def cmd_resume(args: argparse.Namespace) -> None:
    train_script = REPO_ROOT / "scripts" / "train" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing: {train_script}")

    run_dir = _resolve_run_dir(args.run_dir)

    cmd = [
        _py(),
        str(train_script),
        "--resume", str(run_dir),
        "--device", args.device,
    ]
    if args.timesteps is not None:
        cmd += ["--timesteps", str(args.timesteps)]
    if args.eval_freq is not None:
        cmd += ["--eval_freq", str(args.eval_freq)]
    if args.n_eval_episodes is not None:
        cmd += ["--n_eval_episodes", str(args.n_eval_episodes)]
    if args.checkpoint_freq is not None:
        cmd += ["--checkpoint_freq", str(args.checkpoint_freq)]
    if args.chunk is not None:
        cmd += ["--chunk", str(args.chunk)]

    _run(cmd)

    if args.auto_eval or args.diag:
        if args.auto_eval:
            _eval(run_dir=run_dir, deterministic=args.deterministic_eval)
        if args.diag:
            _diag(run_dir=run_dir)


def cmd_pause(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir(args.run_dir)
    pause_file = run_dir / "PAUSE"
    pause_file.write_text("pause requested\n")
    print(f"[OK] Pause requested: created {pause_file}")
    print("Training will stop gracefully at next callback check, saving a checkpoint/vecnorm/replay (if enabled).")


def _eval(run_dir: Path, deterministic: bool) -> None:
    eval_script = REPO_ROOT / "scripts" / "eval" / "eval_oos.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Missing: {eval_script}")

    cmd = [_py(), str(eval_script), "--run_dir", str(run_dir)]
    if deterministic:
        cmd += ["--deterministic"]
    _run(cmd)


def cmd_eval(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir(args.run_dir)
    _eval(run_dir=run_dir, deterministic=args.deterministic)


def _diag(run_dir: Path) -> None:
    print("\n=== DIAG: deterministic=True (often flat for SAC) ===")
    _eval(run_dir=run_dir, deterministic=True)

    print("\n=== DIAG: deterministic=False (samples actions; should trade if policy not dead) ===")
    _eval(run_dir=run_dir, deterministic=False)

    print("\n[NOTE] If deterministic=True is flat but deterministic=False trades -> mean policy ~0 (common).")
    print("[NOTE] If both are flat -> either reward/cost makes flat optimal, or a logging/mismatch bug remains.")


def cmd_status(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir(args.run_dir)
    pause_file = run_dir / "PAUSE"
    paused_txt = run_dir / "PAUSED.txt"
    print(f"run_dir: {run_dir}")
    print(f"PAUSE file: {'YES' if pause_file.exists() else 'NO'}")
    if paused_txt.exists():
        print("PAUSED.txt:")
        print(paused_txt.read_text().strip())

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("checkpoint_*.zip"))
        if ckpts:
            print(f"latest_checkpoint: {ckpts[-1].name}")
        else:
            print("latest_checkpoint: none")
    else:
        print("checkpoints/: missing")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hc", description="HC_IA_RETAIL single entrypoint")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    t = sub.add_parser("train", help="Start a new training run (creates a new run_dir)")
    t.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    t.add_argument("--timesteps", type=int, default=None)
    t.add_argument("--data_path", type=str, default=None)
    t.add_argument("--eval_freq", type=int, default=None)
    t.add_argument("--n_eval_episodes", type=int, default=None)
    t.add_argument("--checkpoint_freq", type=int, default=None)
    t.add_argument("--chunk", type=int, default=None)

    t.add_argument("--auto-eval", action="store_true", help="Run eval_oos after training on latest run")
    t.add_argument("--diag", action="store_true", help="Run eval twice (deterministic vs stochastic) after training")
    t.add_argument("--deterministic-eval", action="store_true", help="When auto-eval, use deterministic policy")
    t.set_defaults(func=cmd_train)

    # resume
    r = sub.add_parser("resume", help="Resume an existing run_dir (requires train.py to support --resume)")
    r.add_argument("run_dir", type=str, help="Path to run dir (e.g., runs/run_YYYY...)")
    r.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    r.add_argument("--timesteps", type=int, default=None)
    r.add_argument("--eval_freq", type=int, default=None)
    r.add_argument("--n_eval_episodes", type=int, default=None)
    r.add_argument("--checkpoint_freq", type=int, default=None)
    r.add_argument("--chunk", type=int, default=None)

    r.add_argument("--auto-eval", action="store_true")
    r.add_argument("--diag", action="store_true")
    r.add_argument("--deterministic-eval", action="store_true")
    r.set_defaults(func=cmd_resume)

    # pause
    pa = sub.add_parser("pause", help="Request pause: creates run_dir/PAUSE (training must watch it)")
    pa.add_argument("run_dir", type=str, help="Path to run dir")
    pa.set_defaults(func=cmd_pause)

    # eval
    e = sub.add_parser("eval", help="Run eval_oos on a run_dir (default: latest)")
    e.add_argument("--run_dir", type=str, default=None)
    e.add_argument("--deterministic", action="store_true")
    e.set_defaults(func=cmd_eval)

    # diag
    d = sub.add_parser("diag", help="Run eval twice on a run_dir: deterministic vs stochastic")
    d.add_argument("--run_dir", type=str, default=None)
    d.set_defaults(func=lambda a: _diag(_resolve_run_dir(a.run_dir)))

    # status
    s = sub.add_parser("status", help="Show pause/checkpoint status for a run_dir (default: latest)")
    s.add_argument("--run_dir", type=str, default=None)
    s.set_defaults(func=cmd_status)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()