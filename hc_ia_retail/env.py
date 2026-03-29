# HC_IA_RETAIL/env.py

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .config import env_cfg


class RetailTradingEnv(gym.Env):
    """BTCUSDT PERP-style continuous trading environment.

    Core timing convention (explicit + auditable):
    - At time index t == self.idx, the agent outputs action a_t in [-1, 1].
    - Raw target position is pos_raw = L * a_t, where L = env_cfg.max_leverage.
    - The simulator holds an executed position pos_exec, which may follow pos_raw with inertia.
    - PnL for the interval t -> t+1 is computed on the position applied during the interval: pos_interval.
    - Transaction costs are charged immediately on executed turnover.
    - Funding is applied on pos_interval.
    - Equity updates multiplicatively: equity *= exp(pnl_log_net).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        max_steps: int | None,
        forward_col: str = "log_ret_1_fwd",
        reward_mode: str | None = None,
        *,
        fee_bps: float | None = None,
        spread_bps: float | None = None,
        lambda_turnover: float | None = None,
        funding_mode: str | None = None,
        max_leverage: float | None = None,
        dd_mode: str | None = None,
        lambda_dd: float | None = None,
        dual_ascent: bool | None = None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.max_steps = max_steps
        self.forward_col = str(forward_col)

        # --- Funding column contract (single source of truth) ---
        # If include_funding_in_obs=True, we require that the funding feature is part of `feature_cols`.
        # The environment is allowed to append it explicitly (contract requested by tests).
        self._funding_col: str | None = None
        # True => value is already per-1h (do not rescale again)
        self._funding_is_hourly: bool = False

        include_funding_in_obs = bool(getattr(env_cfg, "include_funding_in_obs", False))
        if include_funding_in_obs:
            # Prefer already-hourly funding features when available
            if "funding_rate_1h_scaled" in self.df.columns:
                self._funding_col = "funding_rate_1h_scaled"
                self._funding_is_hourly = True
            elif "funding_rate_1h" in self.df.columns:
                self._funding_col = "funding_rate_1h"
                self._funding_is_hourly = True
            elif "funding_rate" in self.df.columns:
                self._funding_col = "funding_rate"
                self._funding_is_hourly = False
            else:
                self._funding_col = None
                self._funding_is_hourly = False
                print(
                    "[RetailTradingEnv] include_funding_in_obs=True but no funding column found in df. "
                    "Proceeding without funding in observation (expected 'funding_rate_1h_scaled' or 'funding_rate_1h')."
                )

            # Ensure funding is present in feature_cols when switch is ON (test contract)
            if self._funding_col is not None and self._funding_col not in self.feature_cols:
                self.feature_cols = list(self.feature_cols) + [self._funding_col]

        cfg_reward_mode = getattr(env_cfg, "reward_mode", "dsr")
        self.reward_mode = str(reward_mode if reward_mode is not None else cfg_reward_mode).lower()
        if self.reward_mode not in {"dsr", "pnl"}:
            raise ValueError(f"Invalid reward_mode={self.reward_mode!r}. Expected 'dsr' or 'pnl'.")

        cfg_funding_mode = str(getattr(env_cfg, "funding_mode", "none")).lower()
        self.funding_mode = str((funding_mode if funding_mode is not None else cfg_funding_mode)).lower()
        if self.funding_mode not in {"none", "binance_8h", "constant"}:
            raise ValueError(
                f"Invalid funding_mode={self.funding_mode!r}. Expected 'none', 'binance_8h', or 'constant'."
            )
        self._funding_missing_warned = False

        self.max_leverage = float(max_leverage if max_leverage is not None else getattr(env_cfg, "max_leverage", 1.0))
        self.fee_bps = float(fee_bps if fee_bps is not None else getattr(env_cfg, "fee_bps", 0.0))
        self.spread_bps = float(spread_bps if spread_bps is not None else getattr(env_cfg, "spread_bps", 0.0))
        self.lambda_turnover = float(
            lambda_turnover if lambda_turnover is not None else getattr(env_cfg, "lambda_turnover", 0.0)
        )
        self.dd_mode = str(dd_mode if dd_mode is not None else getattr(env_cfg, "dd_mode", "none"))
        self.lambda_dd = float(lambda_dd if lambda_dd is not None else getattr(env_cfg, "lambda_dd", 0.0))

        # Keep lagrangians in code but v1 config keeps them OFF.
        self.turnover_lagrangian = bool(getattr(env_cfg, "turnover_lagrangian", False))
        self.turnover_tau = float(getattr(env_cfg, "turnover_tau", 0.010))
        self.turnover_lambda = float(getattr(env_cfg, "turnover_lambda_init", 0.0))
        self.turnover_lambda_lr = float(getattr(env_cfg, "turnover_lambda_lr", 0.50))
        self.turnover_lambda_max = float(getattr(env_cfg, "turnover_lambda_max", 10.0))
        self.turnover_dual_ascent = bool(getattr(env_cfg, "turnover_dual_ascent", True))
        if dual_ascent is not None:
            self.turnover_dual_ascent = bool(dual_ascent)
        self.turnover_dual_update_every = int(getattr(env_cfg, "turnover_dual_update_every", 256) or 0)

        self._dual_sum_abs_dpos = 0.0
        self._dual_steps_k = 0
        self._dual_updates = 0
        self._ep_sum_abs_dpos = 0.0
        self._ep_steps = 0

        self.action_lagrangian = bool(getattr(env_cfg, "action_lagrangian", False))
        self.action_tau = float(getattr(env_cfg, "action_tau", 0.05))
        self.action_lambda = float(getattr(env_cfg, "action_lambda_init", 0.0))
        self.action_lambda_lr = float(getattr(env_cfg, "action_lambda_lr", 0.05))
        self.action_lambda_max = float(getattr(env_cfg, "action_lambda_max", 5.0))
        self.action_dual_ascent = bool(getattr(env_cfg, "action_dual_ascent", True))
        if dual_ascent is not None:
            self.action_dual_ascent = bool(dual_ascent)
        self.action_dual_update_every = int(getattr(env_cfg, "action_dual_update_every", 256) or 0)

        self._dual_sum_action_hinge = 0.0
        self._dual_steps_action_k = 0
        self._dual_action_updates = 0
        self._ep_sum_action_hinge = 0.0
        self._ep_action_steps = 0

        self.exposure_lagrangian = bool(getattr(env_cfg, "exposure_lagrangian", False))
        self.exposure_pmax = float(getattr(env_cfg, "exposure_pmax", 1.00))
        self.exposure_lambda = float(getattr(env_cfg, "exposure_lambda_init", 0.0))
        self.exposure_lambda_lr = float(getattr(env_cfg, "exposure_lambda_lr", 0.05))
        self.exposure_lambda_max = float(getattr(env_cfg, "exposure_lambda_max", 10.0))
        self.exposure_dual_ascent = bool(getattr(env_cfg, "exposure_dual_ascent", True))
        if dual_ascent is not None:
            self.exposure_dual_ascent = bool(dual_ascent)
        self.exposure_dual_update_every = int(getattr(env_cfg, "exposure_dual_update_every", 256) or 0)

        self._dual_sum_exposure_hinge = 0.0
        self._dual_steps_exposure_k = 0
        self._dual_exposure_updates = 0
        self._ep_sum_exposure_hinge = 0.0
        self._ep_exposure_steps = 0

        # Execution model
        self.exec_mode = str(getattr(env_cfg, "execution_model", "instant")).lower()
        if self.exec_mode not in {"instant", "ewma"}:
            raise ValueError(f"Invalid execution_model={self.exec_mode!r}. Expected 'instant' or 'ewma'.")

        beta_cfg = getattr(env_cfg, "execution_beta", None)
        hl_cfg = getattr(env_cfg, "execution_half_life_h", 12.0)
        if beta_cfg is not None:
            try:
                self.exec_beta = float(beta_cfg)
            except Exception:
                self.exec_beta = 1.0
        else:
            try:
                H = float(hl_cfg)
            except Exception:
                H = 12.0
            if np.isfinite(H) and H > 0.0:
                self.exec_beta = float(1.0 - (2.0 ** (-1.0 / H)))
            else:
                self.exec_beta = 1.0

        if not np.isfinite(self.exec_beta) or self.exec_beta <= 0.0:
            self.exec_beta = 1.0
        self.exec_beta = float(np.clip(self.exec_beta, 1e-6, 1.0))

        if self.forward_col not in self.df.columns:
            # V1 feature-dataset fallback:
            # if the dataset contains backward 1-step log returns (`log_ret_1`),
            # then the forward return at t is exactly `log_ret_1` shifted by -1.
            if "log_ret_1" in self.df.columns:
                self.df = self.df.copy()
                self.df[self.forward_col] = pd.to_numeric(self.df["log_ret_1"], errors="coerce").shift(-1)
            else:
                raise ValueError(f"Missing required forward return column: {self.forward_col}")

        if self.forward_col in self.df.columns:
            self.df[self.forward_col] = pd.to_numeric(self.df[self.forward_col], errors="coerce")
            before_len = len(self.df)
            self.df = self.df.dropna(subset=[self.forward_col]).reset_index(drop=True)
            dropped = before_len - len(self.df)
            if dropped > 0:
                print(
                    f"[RetailTradingEnv] Dropped {dropped} trailing rows with missing {self.forward_col} "
                    f"after forward-return alignment."
                )

        self.action_mode = str(getattr(env_cfg, "action_mode", "direct")).lower()
        if self.action_mode not in {"direct", "residual"}:
            raise ValueError(f"Invalid action_mode={self.action_mode!r}. Expected 'direct' or 'residual'.")
        self.residual_delta_max = float(getattr(env_cfg, "residual_delta_max", 0.25))
        if not np.isfinite(self.residual_delta_max) or self.residual_delta_max <= 0.0:
            raise ValueError(
                f"Invalid residual_delta_max={self.residual_delta_max!r}. Expected a strictly positive finite float."
            )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation contract: market/regime features are provided via `feature_cols`,
        # while state variables are controlled explicitly by config switches.
        # IMPORTANT: observation shape must be fixed at init time for SB3/Gym compatibility.
        self._obs_include_position = bool(getattr(env_cfg, "include_position_in_obs", True))
        self._obs_include_equity_norm = bool(getattr(env_cfg, "include_equity_norm_in_obs", True))
        self._obs_include_dd = bool(getattr(env_cfg, "include_drawdown_in_obs", False))
        self._obs_include_bench_pos = bool(getattr(env_cfg, "include_bench_pos_in_obs", False))

        self._freq_per_year = int(getattr(env_cfg, "freq_per_year", 6 * 365))

        extra = 0
        extra += 1 if self._obs_include_position else 0
        extra += 1 if self._obs_include_equity_norm else 0
        extra += 1 if self._obs_include_dd else 0
        extra += 1 if self._obs_include_bench_pos else 0

        self.n_obs = len(self.feature_cols) + extra
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32)

        self._reset_internal()

    def _reset_internal(self, initial_position: float | None = None):
        if self.max_steps is not None:
            max_start = max(len(self.df) - self.max_steps - 1, 0)
            random_start = bool(getattr(env_cfg, "random_start", False))
            if random_start and max_start > 0 and hasattr(self, "np_random"):
                start = int(self.np_random.integers(0, max_start + 1))
            else:
                start = 0
            self.idx = start
        else:
            self.idx = 0

        self.start_idx = int(self.idx)
        self.steps = 0

        self._ep_sum_abs_dpos = 0.0
        self._ep_steps = 0
        self._dual_sum_abs_dpos = 0.0
        self._dual_steps_k = 0
        self._dual_updates = 0

        self._ep_sum_action_hinge = 0.0
        self._ep_action_steps = 0
        self._dual_sum_action_hinge = 0.0
        self._dual_steps_action_k = 0
        self._dual_action_updates = 0

        self._ep_sum_exposure_hinge = 0.0
        self._ep_exposure_steps = 0
        self._dual_sum_exposure_hinge = 0.0
        self._dual_steps_exposure_k = 0
        self._dual_exposure_updates = 0

        self.position = 0.0 if initial_position is None else float(initial_position)

        self.bench_position = 0.0
        self.bench_equity = 1.0

        self.prev_action = 0.0

        self.equity = float(env_cfg.initial_cash)
        self.equity_peak = float(self.equity)
        self.dd_prev = 0.0
        self.dd_streak = 0
        # Keep a stable drawdown state for observation/reward shaping
        self.dd_prev = 0.0

        self.A = 0.0
        self.B = 1e-6

    def _obs(self) -> np.ndarray:
        row = self.df.iloc[self.idx]
        feats = row[self.feature_cols].to_numpy(dtype=np.float32)
        equity_norm = np.float32(self.equity / float(env_cfg.initial_cash))

        parts: list[np.ndarray] = [feats]

        if self._obs_include_position:
            parts.append(np.array([self.position], dtype=np.float32))

        if self._obs_include_equity_norm:
            parts.append(np.array([equity_norm], dtype=np.float32))

        if self._obs_include_dd:
            parts.append(np.array([self.dd_prev], dtype=np.float32))

        if self._obs_include_bench_pos:
            bench_mode = str(getattr(env_cfg, "bench_mode", "none")).lower()
            if bench_mode != "none":
                bt = float(self._bench_target_pos(row, freq_per_year=self._freq_per_year))
            else:
                bt = 0.0
            parts.append(np.array([bt], dtype=np.float32))

        obs = np.concatenate(parts, axis=0).astype(np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        init_pos = None
        if isinstance(options, dict) and "initial_position" in options:
            init_pos = float(options["initial_position"])

        # Observation shape must remain fixed after init.
        # We allow options keys only if they do NOT change the configured shape.
        if isinstance(options, dict):
            if "include_position_in_obs" in options and bool(options["include_position_in_obs"]) != self._obs_include_position:
                raise ValueError("include_position_in_obs cannot be changed at reset; it changes observation shape. Configure via env_cfg.")
            if "include_equity_norm_in_obs" in options and bool(options["include_equity_norm_in_obs"]) != self._obs_include_equity_norm:
                raise ValueError("include_equity_norm_in_obs cannot be changed at reset; it changes observation shape. Configure via env_cfg.")
            if "include_drawdown_in_obs" in options and bool(options["include_drawdown_in_obs"]) != self._obs_include_dd:
                raise ValueError("include_drawdown_in_obs cannot be changed at reset; it changes observation shape. Configure via env_cfg.")
            if "include_bench_pos_in_obs" in options and bool(options["include_bench_pos_in_obs"]) != self._obs_include_bench_pos:
                raise ValueError("include_bench_pos_in_obs cannot be changed at reset; it changes observation shape. Configure via env_cfg.")

        self._reset_internal(initial_position=init_pos)
        return self._obs(), {}

    def _get_timestamp(self, row: pd.Series):
        if "timestamp" in row.index:
            return row["timestamp"]
        if "time" in row.index:
            return row["time"]
        if "datetime" in row.index:
            return row["datetime"]
        return None

    def _timestamp_iso_utc(self, ts: any) -> str | None:
        if ts is None:
            return None
        try:
            if hasattr(ts, "to_pydatetime"):
                dt = ts.to_pydatetime()
            else:
                dt = pd.to_datetime(ts, utc=True, errors="coerce").to_pydatetime()
            if dt is None:
                return None
            return dt.replace(tzinfo=None).isoformat() + "Z"
        except Exception:
            try:
                s = str(ts)
                return s if s else None
            except Exception:
                return None

    def _funding_rate_1h(self, row: pd.Series) -> float:
        """Return the funding rate aligned with the dataset feature.

        Contract:
        - If the dataset contains 'funding_rate_1h_scaled' or 'funding_rate_1h', we must return it *as-is*
          (already per-1h). No rescaling here.
        - Only if neither exists and 'funding_rate' exists, return that raw value (typically 8h); the step
          function may rescale to per-step if desired.
        """
        if self.funding_mode == "none":
            return 0.0

        if self.funding_mode == "constant":
            return float(getattr(env_cfg, "funding_rate_constant", 0.0))

        # binance_8h: prefer hourly-aligned columns if present
        if "funding_rate_1h_scaled" in row.index:
            val = row["funding_rate_1h_scaled"]
        elif "funding_rate_1h" in row.index:
            val = row["funding_rate_1h"]
        elif "funding_rate" in row.index:
            val = row["funding_rate"]
        else:
            if not self._funding_missing_warned:
                print(
                    "[RetailTradingEnv] funding_mode='binance_8h' but no funding column found in df. "
                    "Using funding_rate_used=0.0. (Expected 'funding_rate_1h_scaled', 'funding_rate_1h' or 'funding_rate'.)"
                )
                self._funding_missing_warned = True
            return 0.0

        try:
            f = float(val)
        except Exception:
            return 0.0

        if not np.isfinite(f):
            return 0.0
        return f

    def _get_vol_1h_estimate(self, row: pd.Series) -> float:
        prefs = getattr(env_cfg, "vol_col_preference", ("sigma_1h", "vol_roll", "vol_24", "vol", "sigma"))
        for c in prefs:
            if c in row.index:
                try:
                    v = float(row[c])
                except Exception:
                    continue
                if np.isfinite(v) and v > 0.0:
                    return v

        for c in ("sigma_1h", "vol_roll", "vol_24", "vol_24h", "sigma_24", "vol", "sigma"):
            if c in row.index:
                try:
                    v = float(row[c])
                except Exception:
                    continue
                if np.isfinite(v) and v > 0.0:
                    return v

        return 0.0

    def _bench_target_pos(self, row: pd.Series, *, freq_per_year: int) -> float:
        mode = str(getattr(env_cfg, "bench_mode", "none")).lower()
        L = float(self.max_leverage)

        if mode == "bh":
            return float(np.clip(L, 0.0, L))

        if mode == "bh_vol_target":
            target_vol_ann = float(getattr(env_cfg, "bench_target_vol_annual", 0.20))
            sigma_1h = float(self._get_vol_1h_estimate(row))
            if sigma_1h <= 0.0 or not np.isfinite(sigma_1h):
                return float(np.clip(L, 0.0, L))

            vol_ann = float(sigma_1h * np.sqrt(max(1.0, float(freq_per_year))))
            if vol_ann <= 0.0 or not np.isfinite(vol_ann):
                return float(np.clip(L, 0.0, L))

            lev = float(target_vol_ann / vol_ann)
            return float(np.clip(lev, 0.0, L))

        return 0.0

    def _step_once(self, action):
        t_index = int(self.idx)
        row = self.df.iloc[t_index]
        t_timestamp = self._get_timestamp(row)

        # Forward return used for t -> t+1
        if "close" in self.df.columns and (t_index + 1) < len(self.df):
            p0 = float(self.df.iloc[t_index]["close"])
            p1 = float(self.df.iloc[t_index + 1]["close"])
            if p0 > 0.0 and np.isfinite(p0) and np.isfinite(p1) and p1 > 0.0:
                ret_fwd_used = float(np.log(p1 / p0))
            else:
                ret_fwd_used = float(row[self.forward_col])
        else:
            ret_fwd_used = float(row[self.forward_col])

        # Action
        a_t = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        a_t = float(np.clip(a_t, -1.0, 1.0))

        prev_a = float(getattr(self, "prev_action", 0.0))
        da = abs(a_t - prev_a)
        tau_a = float(getattr(env_cfg, "action_tau", 0.05))
        hinge_da = max(0.0, da - tau_a)
        c_da = float(getattr(env_cfg, "action_churn_cost_bps", 0.0))
        pen_da = c_da * hinge_da
        self.prev_action = a_t

        action_raw = float(a_t)
        action_clipped = float(a_t)
        prev_action = float(prev_a)
        d_action = float(a_t - prev_a)
        abs_d_action = float(abs(d_action))

        action_hinge = 0.0
        if self.action_lagrangian:
            action_hinge = float(hinge_da)


        action_smooth_pen = 0.0
        action_smooth_pen_bps = 0.0
        if bool(getattr(env_cfg, "action_smooth_penalty", False)):
            beta_a = float(getattr(env_cfg, "action_smooth_beta", 0.0))
            if beta_a > 0.0 and np.isfinite(beta_a):
                # Quadratic penalty on action changes, defined in log-return units,
                # then converted into bps to stay consistent with reward_final_bps.
                action_smooth_pen = float(-beta_a * (d_action ** 2))
                action_smooth_pen_bps = float(10000.0 * action_smooth_pen)

        L = float(self.max_leverage)

        # Positions: benchmark-aware target vs executed position
        pos_prev = float(self.position)
        bench_base_pos = float(self._bench_target_pos(row, freq_per_year=self._freq_per_year))

        if self.action_mode == "residual":
            delta_t = float(self.residual_delta_max * a_t)
            pos_raw_unclipped = float(bench_base_pos + delta_t)
            pos_raw = float(np.clip(pos_raw_unclipped, -L, L))
        else:
            delta_t = float("nan")
            pos_raw_unclipped = float(L * a_t)
            pos_raw = float(pos_raw_unclipped)

        if self.exec_mode == "ewma":
            beta = float(self.exec_beta)
            pos_exec_pre_cap = float((1.0 - beta) * pos_prev + beta * pos_raw)
        else:
            beta = 1.0
            pos_exec_pre_cap = float(pos_raw)

        # v1: NO HARD CAP (exposure_pmax=1.0 anyway)
        pmax = float(getattr(env_cfg, "exposure_pmax", 1.0))
        pos_exec = float(pos_exec_pre_cap)

        # Turnover + transaction costs
        dpos_exec = float(pos_exec - pos_prev)
        turnover = float(abs(dpos_exec))
        fee = float((self.fee_bps / 1e4) * turnover)
        spread = float((self.spread_bps / 1e4) * turnover)
        cost = float(fee + spread)

        # Position held during the interval t -> t+1
        use_exec_for_pnl = bool(getattr(env_cfg, "pnl_on_pos_exec", False))
        pos_interval = float(pos_exec) if use_exec_for_pnl else float(pos_prev)

        pos_target = float(pos_raw)
        pos_new = float(pos_exec)  # alias for audit/tests

        # Exposure hinge (diagnostic only)
        exposure_hinge = max(0.0, abs(pos_exec_pre_cap) - pmax)
        exposure_lambda = float(getattr(self, "exposure_lambda", 0.0))
        reward_penalty_lagrangian_exposure = exposure_lambda * exposure_hinge
        exposure_pmax_effective = float(pmax)

        # --------- STEP UNITS FIX (4h bars) ----------
        # Convert 1h quantities -> per-step quantities.
        freq_per_year = int(getattr(env_cfg, "freq_per_year", 6 * 365))
        hours_per_step = float((24.0 * 365.0) / max(1.0, float(freq_per_year)))

        # Funding: strict dataset alignment.
        # - If hourly feature exists ('funding_rate_1h_scaled' or 'funding_rate_1h'), funding_rate_used must equal it (no extra scaling).
        # - Else if only 'funding_rate' exists (typically 8h), rescale to per-step.
        funding_rate_raw = float(self._funding_rate_1h(row))
        if getattr(self, "_funding_is_hourly", False):
            funding_rate_used = float(funding_rate_raw)  # exact match to dataset feature
        else:
            # raw is assumed to be 8h when coming from 'funding_rate'
            funding_rate_used = float(funding_rate_raw * (hours_per_step / 8.0))

        # Vol: sigma_1h is per-1h std; scale to step: sigma_step = sigma_1h * sqrt(hours_per_step)
        sigma_1h = float(self._get_vol_1h_estimate(row))
        sigma_step = (
            float(sigma_1h * np.sqrt(max(0.0, hours_per_step)))
            if (sigma_1h > 0.0 and np.isfinite(sigma_1h))
            else 0.0
        )
        # --------------------------------------------

        # Funding applied on pos_interval (declared even if zero)
        funding_cost = float(pos_interval * funding_rate_used)

        if self.funding_mode == "none":
            funding_mode_str = "disabled"
            funding_rate_used = 0.0
            funding_cost = 0.0
        else:
            if not np.isfinite(funding_rate_used):
                funding_rate_used = 0.0
            if abs(float(funding_rate_used)) <= 0.0:
                funding_mode_str = "zeroed"
            else:
                funding_mode_str = "enabled"
            funding_cost = float(pos_interval * float(funding_rate_used))

        # PnL in log space (net)
        pnl_log_gross = float(pos_interval * ret_fwd_used)
        pnl_log_net = float(pnl_log_gross - cost - funding_cost)

        # --------- Benchmark PnL (same frictions & timing) ----------
        bench_mode = str(getattr(env_cfg, "bench_mode", "none")).lower()
        alpha_relative = bool(getattr(env_cfg, "alpha_relative_reward", False))

        bench_pos_prev = float(getattr(self, "bench_position", 0.0))
        bench_pos_target = float(self._bench_target_pos(row, freq_per_year=freq_per_year)) if bench_mode != "none" else 0.0

        if self.exec_mode == "ewma":
            beta_b = float(self.exec_beta)
            bench_pos_exec_pre_cap = float((1.0 - beta_b) * bench_pos_prev + beta_b * bench_pos_target)
        else:
            bench_pos_exec_pre_cap = float(bench_pos_target)

        bench_apply_pmax_cap = bool(getattr(env_cfg, "bench_apply_pmax_cap", False))
        if bench_apply_pmax_cap and pmax < 1.0:
            bench_pos_exec = float(np.clip(bench_pos_exec_pre_cap, 0.0, pmax))
        else:
            bench_pos_exec = float(bench_pos_exec_pre_cap)

        bench_turnover = float(abs(bench_pos_exec - bench_pos_prev))
        bench_fee = float((self.fee_bps / 1e4) * bench_turnover)
        bench_spread = float((self.spread_bps / 1e4) * bench_turnover)
        bench_cost = float(bench_fee + bench_spread)

        bench_funding_cost = float(bench_pos_exec * funding_rate_used) if funding_mode_str != "disabled" else 0.0
        bench_pnl_log_gross = float(bench_pos_exec * ret_fwd_used)
        bench_pnl_log_net = float(bench_pnl_log_gross - bench_cost - bench_funding_cost)

        self.bench_position = float(bench_pos_exec)
        # -----------------------------------------------------------

        # --------- Reward: pnl_net + drawdown penalty ----------
        # Portfolio-manager v1 contract:
        #   reward_t = pnl_log_net_t - dd_penalty_t
        # where pnl_log_net already includes trading costs + funding.
        # We keep benchmark/risk/inventory diagnostics in info for auditability,
        # but they do NOT enter the reward.

        reward_pnl_net_bps = float(10000.0 * pnl_log_net)

        # Benchmark diagnostics (not used in reward)
        bench_mode = str(getattr(env_cfg, "bench_mode", "none")).lower()
        alpha_relative = bool(getattr(env_cfg, "alpha_relative_reward", False))
        reward_bench_net_bps = 0.0
        bench_kappa = float(getattr(env_cfg, "bench_kappa", 1.0))

        # Project equity and drawdown at t+1 using realized step PnL
        equity_t = float(self.equity)
        equity_tp1 = float(equity_t * np.exp(pnl_log_net))
        equity_peak_t = float(getattr(self, "equity_peak", equity_t))
        equity_peak_tp1 = float(max(equity_peak_t, equity_tp1))
        drawdown_tp1 = float(1.0 - (equity_tp1 / max(equity_peak_tp1, 1e-9)))

        # Drawdown penalty only
        dd_mode = str(getattr(env_cfg, "dd_mode", "none")).lower()
        dd_threshold = float(getattr(env_cfg, "dd_threshold", 0.12))
        dd_pen_power = float(getattr(env_cfg, "dd_pen_power", 2.0))
        lambda_dd = float(getattr(env_cfg, "lambda_dd", 0.0))
        dd_exposure_bps = float(getattr(env_cfg, "dd_exposure_bps", 0.0))

        dd_excess = float(max(0.0, drawdown_tp1 - dd_threshold))
        dd_pen_bps = 0.0
        dd_exposure_pen_bps = 0.0
        if dd_mode == "hinge" and lambda_dd > 0.0 and np.isfinite(lambda_dd):
            p = dd_pen_power if (np.isfinite(dd_pen_power) and dd_pen_power > 0.0) else 2.0
            dd_pen_bps = float(lambda_dd * (dd_excess ** p) * abs(pos_interval))

            if dd_exposure_bps > 0.0 and np.isfinite(dd_exposure_bps):
                dd_exposure_pen_bps = float(dd_exposure_bps * dd_excess * abs(pos_interval))

        # Diagnostics kept explicit but switched off in reward
        risk_eta = float(getattr(env_cfg, "risk_eta", 0.0))
        risk_pen_mv_bps = 0.0
        inventory_pen_bps = 0.0
        reward_core_bps = float(reward_pnl_net_bps)
        # Action smoothing penalty is part of the economic objective when enabled.
        reward_action_smooth_bps = float(action_smooth_pen_bps)

        # Final reward in bps/step: pure pnl_net minus drawdown penalty terms plus action smoothing penalty
        reward_final_bps = float(
            reward_pnl_net_bps
            - dd_pen_bps
            - dd_exposure_pen_bps
            + reward_action_smooth_bps
        )

        # --- Reward contract (v2): env returns float in log-return units ---
        reward_final = float(reward_final_bps / 10000.0)
        reward_raw = float(reward_final)  # no clipping in this environment
        is_clipped = False
        # ----------------------------------------------------------------

        # Equity update uses NET pnl (use projected values computed above)
        self.equity = float(equity_tp1)
        self.equity_peak = float(max(getattr(self, "equity_peak", equity_t), equity_tp1))

        # Drawdown state
        drawdown = float(np.clip(drawdown_tp1, 0.0, 1.0))
        self.dd_prev = float(drawdown)
        self.dd_streak = 0

        # Apply position change AFTER pnl/costs for this step
        self.position = float(pos_exec)

        # Advance
        self.idx += 1
        self.steps += 1

        terminated = (self.idx >= len(self.df) - 1) or (self.equity <= env_cfg.ruin_threshold * env_cfg.initial_cash)
        truncated = (self.max_steps is not None and self.steps >= self.max_steps)

        equity_norm = float(self.equity / float(env_cfg.initial_cash))

        info = {
            "t_index": t_index,
            "timestamp": self._timestamp_iso_utc(t_timestamp),
            "ret_fwd_used": float(ret_fwd_used),

            "a_t": float(a_t),
            "action_mode": str(self.action_mode),
            "residual_delta_max": float(self.residual_delta_max),
            "bench_base_pos": float(bench_base_pos),
            "delta_t": float(delta_t) if np.isfinite(delta_t) else None,
            "pos_prev": float(pos_prev),
            "pos_raw_unclipped": float(pos_raw_unclipped),
            "pos_raw": float(pos_raw),
            "pos_exec_pre_cap": float(pos_exec_pre_cap),
            "pos_exec": float(pos_exec),
            "pos_interval": float(pos_interval),

            # Aliases for test contract (positions)
            "pos_target": float(pos_target),
            "pos_new": float(pos_new),
            "position": float(pos_new),

            "turnover": float(turnover),
            "fee_bps": float(self.fee_bps),
            "spread_bps": float(self.spread_bps),
            "fee": float(fee),
            "spread": float(spread),
            "cost": float(cost),

            # Step scaling audit
            "freq_per_year": int(freq_per_year),
            "hours_per_step": float(hours_per_step),

            # Funding audit (per-step)
            "funding_mode": str(funding_mode_str),
            "funding_source_mode": str(self.funding_mode),
            "funding_rate_1h": float(funding_rate_raw),
            "funding_rate_used": float(funding_rate_used),
            "funding_cost": float(funding_cost),

            # Benchmark audit
            "alpha_relative_reward": bool(alpha_relative),
            "bench_mode": str(bench_mode),
            "bench_kappa": float(bench_kappa),
            "bench_pos_target": float(bench_pos_target),
            "bench_pos_base_for_action": float(bench_base_pos),
            "bench_pos_exec": float(bench_pos_exec),
            "bench_turnover": float(bench_turnover),
            "bench_cost": float(bench_cost),
            "bench_funding_cost": float(bench_funding_cost),
            "bench_pnl_log_net": float(bench_pnl_log_net),

            # Vol/risk audit
            "sigma_1h_est": float(sigma_1h),
            "sigma_step_est": float(sigma_step),
            "risk_eta": float(risk_eta),
            "risk_pen_mv_bps": float(risk_pen_mv_bps),

            # Drawdown shaping audit
            "dd_mode": str(dd_mode),
            "dd_threshold": float(dd_threshold),
            "dd_excess": float(dd_excess),
            "dd_pen_bps": float(dd_pen_bps),
            "dd_exposure_pen_bps": float(dd_exposure_pen_bps),

            # Inventory/time-in-market audit
            "inventory_cost_bps": float(getattr(env_cfg, "inventory_cost_bps", 0.0)),
            "inventory_pen_bps": float(inventory_pen_bps),

            # Reward audit
            "reward_pnl_net_bps": float(reward_pnl_net_bps),
            "reward_bench_net_bps": float(reward_bench_net_bps),
            "reward_core_bps": float(reward_core_bps),
            "reward_action_smooth_bps": float(reward_action_smooth_bps),
            "reward_final_bps": float(reward_final_bps),
            "reward_raw": float(reward_raw),
            "reward_final": float(reward_final),
            "reward": float(reward_final),
            "is_clipped": bool(is_clipped),
            "action_smooth_pen": float(action_smooth_pen),
            "action_smooth_pen_bps": float(action_smooth_pen_bps),

            # Equity
            "equity": float(self.equity),
            "equity_norm": float(equity_norm),
            "drawdown": float(drawdown),

            # Observation contract audit
            "obs_include_position": bool(self._obs_include_position),
            "obs_include_equity_norm": bool(self._obs_include_equity_norm),
            "obs_include_drawdown": bool(self._obs_include_dd),
            "obs_include_bench_pos": bool(self._obs_include_bench_pos),
            "obs_feature_dim": int(len(self.feature_cols)),
            "obs_dim_total": int(self.n_obs),

            # PnL logs (explicit, required by tests)
            "pnl_log_net": float(pnl_log_net),
            "pnl_log_gross": float(pnl_log_gross),
        }

        return self._obs(), float(reward_final), terminated, truncated, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._step_once(action)
        return obs, reward, terminated, truncated, info