# HC_IA_RETAIL/config.py
from dataclasses import dataclass
from pathlib import Path

# BASE_DIR pointe sur la racine du repo (HC_IA_RETAIL), pas sur le package hc_ia_retail/
BASE_DIR = Path(__file__).resolve().parents[1]

# Dossiers standard du projet
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"
MODELS_DIR = BASE_DIR / "models"


@dataclass
class DataConfig:
    # --- Instrument / market identity (audit-grade) ---
    instrument_name: str = "BTCUSDT_PERP"
    market: str = "binance_futures"  # Binance Futures (USDT-M)
    interval: str = "4h"

    # Price convention (declared explicitly for audit)
    # For the V1 training dataset, market features are precomputed and the reward
    # contract is driven by forward returns rather than raw close-only ingestion.
    price_mode: str = "feature_dataset"

    # Default dataset path (prebuilt V1 SAC training dataset)
    data_path: Path = DATA_DIR / "train_v1_sac_dataset_4h.parquet"

    # Backward-compatible alias (some modules may still reference csv_path)
    csv_path: Path = DATA_DIR / "train_v1_sac_dataset_4h.parquet"

    time_col: str = "timestamp"
    ohlcv_cols: tuple[str, ...] = ("open", "high", "low", "close", "volume")

    # Time resolution (used for annualization in eval): 4h bars
    freq_per_year: int = 6 * 365

    train_frac: float = 0.80
    val_frac: float = 0.10  # sur le reste: val puis test
    window_size: int = 64   # mémoire (en bars; 64 bars en 4h ≈ 10.7 jours)


@dataclass
class FeatureConfig:
    # features “core” (propres, stationnaires)
    use_log_returns: bool = True
    vol_window: int = 24
    rsi_window: int = 14
    z_window: int = 100
    ma_fast: int = 24
    ma_slow: int = 24 * 7


@dataclass
class RegimeDataConfig:
    enabled: bool = True
    rl_dataset_root: Path = BASE_DIR.parent / "hc-regime-engine" / "artifacts" / "rl_dataset"
    rl_dataset_run_id: str | None = None
    use_latest_rl_dataset: bool = True

    # observation construction mode
    regime_feature_mode: str = "market_plus_regime"  # market_only | regime_only | market_plus_regime

    # regime feature policy switches
    include_z_hat_filter: bool = False

    # strict merge contract
    strict_timestamp_merge: bool = True
    allow_inner_join_only: bool = True


@dataclass
class EnvConfig:
    # Action convention (PERP): agent outputs a in [-1, 1]
    # Environment maps to target exposure: pos_target = max_leverage * a
    max_leverage: float = 1.0
    short_factor: float = 1.0

    # Action interpretation
    # - "direct": agent action maps directly to portfolio exposure in [-max_leverage, +max_leverage]
    # - "residual": agent action maps to a bounded overlay around a benchmark/base position
    action_mode: str = "direct"  # "direct" | "residual"
    residual_delta_max: float = 0.25

    # --- Execution model (R5) ---
    execution_model: str = "instant"  # "instant" | "ewma"
    execution_beta: float | None = None
    execution_half_life_h: float = 12.0

    # --- Funding (PERP) ---
    funding_mode: str = "binance_8h"  # "none" | "binance_8h" | "constant"
    funding_path: Path = DATA_DIR / "BTCUSDT_PERP_funding_8h.parquet"
    funding_scale_to_1h: bool = True

    # --- Observation switches ---
    include_funding_in_obs: bool = True

    # --- Costs model ---
    cost_model: str = "turnover_linear"
    fee_bps: float = 2.0
    spread_bps: float = 2.0

    # --- PnL timing convention ---
    pnl_on_pos_exec: bool = True

    # --- Reward units ---
    # On travaille en bps/step
    reward_scale_bps: float = 10000.0

    # ---------- IMPORTANT: align units with 4h bars ----------
    # Used by benchmark sizing & annualization in-env
    # (fixes bug where env defaulted to 24*365)
    freq_per_year: int = 6 * 365
    # ---------------------------------------------------------

    # --- Disable old shaping / constraints for v1 ---
    exposure_floor_bonus: bool = False
    exposure_floor_pmin: float = 0.20
    exposure_floor_gamma_bps: float = 0.10

    lambda_turnover: float = 0.00

    turnover_lagrangian: bool = False
    turnover_tau: float = 0.020
    turnover_lambda_init: float = 0.0
    turnover_lambda_lr: float = 0.10
    turnover_lambda_max: float = 5.0
    turnover_dual_ascent: bool = True
    turnover_dual_update_every: int = 256

    action_smooth_penalty: bool = False
    action_smooth_beta: float = 0.00005

    action_lagrangian: bool = False
    action_tau: float = 0.15
    action_lambda_init: float = 0.0
    action_lambda_lr: float = 0.05
    action_lambda_max: float = 10.0
    action_dual_ascent: bool = True
    action_dual_update_every: int = 256

    exposure_lagrangian: bool = False
    exposure_pmax: float = 1.00
    exposure_lambda_init: float = 0.0
    exposure_lambda_lr: float = 0.05
    exposure_lambda_max: float = 10.0
    exposure_dual_ascent: bool = True
    exposure_dual_update_every: int = 256

    # Reward contract v1: direct net PnL.
    # In direct mode, the agent controls full exposure and benchmark terms are disabled.
    alpha_relative_reward: bool = False
    bench_mode: str = "none"  # "none" | "bh" | "bh_vol_target"
    bench_kappa: float = 1.0
    bench_target_vol_annual: float = 0.20
    bench_apply_pmax_cap: bool = True  # parity, harmless with pmax=1.0

    tracking_error_penalty: bool = False
    te_eta: float = 0.0

    # Mean-variance risk aversion (bps/step after scaling)
    risk_eta: float = 0.0  # vR0: disable MV risk term to avoid drowning alpha signal

    vol_col_preference: tuple[str, ...] = ("sigma_1h", "vol_roll", "vol_24", "vol_24h", "sigma_24", "vol", "sigma")

    # Reward mode: pnl direct (stable & aligned with alpha-relative)
    reward_mode: str = "pnl"
    simple_pnl_reward: bool = False

    # DSR (unused in v1)
    sharpe_alpha: float = 0.002
    dsr_warmup_steps: int = 512
    dsr_clip: float = 5.0

    # Drawdown shaping diagnostic mode: disabled to test PM signal under real frictions only.
    # We keep real costs/funding in the reward, but remove hand-shaped drawdown penalties.
    lambda_dd: float = 0.0
    dd_mode: str = "none"          # "none" | "hinge"
    dd_threshold: float = 0.06
    dd_delay_steps: int = 0
    dd_pen_power: float = 2.0

    # Extra penalty for holding exposure while in drawdown: disabled in diagnostic mode.
    dd_exposure_bps: float = 0.0

    # Inventory / time-in-market penalty remains disabled in diagnostic mode.
    inventory_cost_bps: float = 0.00

    # Action churn penalty disabled in diagnostic mode.
    action_churn_cost_bps: float = 0.0

    ruin_threshold: float = 0.15
    initial_cash: float = 1.0

    # IMPORTANT: explicit observation contract switches
    include_position_in_obs: bool = True
    include_equity_norm_in_obs: bool = True
    include_drawdown_in_obs: bool = True
    include_bench_pos_in_obs: bool = False

    max_steps_train: int = 4096


@dataclass
class SACConfig:
    seed: int = 42
    total_timesteps: int = 200_000

    learning_rate: float = 2e-4
    buffer_size: int = 300_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005

    train_freq: int = 1
    gradient_steps: int = 1

    ent_coef: str | float = "auto"
    target_entropy: float = -1.0


@dataclass
class ModelConfig:
    gru_hidden: int = 64
    gru_layers: int = 1
    mlp_latent: int = 128


@dataclass
class PathsConfig:
    out_dir: Path = RUNS_DIR
    models_dir: Path = MODELS_DIR


data_cfg = DataConfig()
feat_cfg = FeatureConfig()
regime_cfg = RegimeDataConfig()
env_cfg = EnvConfig()
sac_cfg = SACConfig()
model_cfg = ModelConfig()
paths_cfg = PathsConfig()