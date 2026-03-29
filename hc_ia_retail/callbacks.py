from __future__ import annotations

import signal
import time
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

class RunControlCallback(BaseCallback):
    def __init__(self, run_dir: str, check_freq: int = 1000):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.check_freq = int(check_freq)
        self.pause_file = self.run_dir / "PAUSE"
        self._stop_requested = False

    def _on_training_start(self) -> None:
        def _handler(sig, frame):
            self._stop_requested = True
        signal.signal(signal.SIGINT, _handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, _handler)  # kill

    def _save_state(self, reason: str):
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ts = int(self.num_timesteps)
        model_path = ckpt_dir / f"checkpoint_{ts}_{reason}.zip"
        self.model.save(str(model_path))

        # replay buffer (SAC)
        try:
            rb_path = ckpt_dir / "replay_buffer.pkl"
            self.model.save_replay_buffer(str(rb_path))
        except Exception:
            pass

        # vecnormalize
        try:
            vec_path = self.run_dir / "vecnormalize.pkl"
            self.training_env.save(str(vec_path))
        except Exception:
            pass

        (self.run_dir / "PAUSED.txt").write_text(
            f"paused_at_timesteps={ts}\nreason={reason}\ntime={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        # consume pause file to avoid immediate repause on resume
        try:
            if self.pause_file.exists():
                self.pause_file.unlink()
        except Exception:
            pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        if self.pause_file.exists():
            self._save_state("pause_file")
            return False

        if self._stop_requested:
            self._save_state("sigint")
            return False

        return True