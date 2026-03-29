from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ObservationContract:
    mode: str
    market_feature_columns: List[str]
    regime_feature_columns: List[str]
    env_state_columns: List[str]
    final_agent_feature_columns: List[str]


VALID_OBSERVATION_MODES = {
    "market_only",
    "regime_only",
    "market_plus_regime",
}


def validate_observation_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in VALID_OBSERVATION_MODES:
        raise ValueError(
            f"Invalid observation mode={mode!r}. Expected one of {sorted(VALID_OBSERVATION_MODES)}"
        )
    return m


def build_env_state_columns(
    *,
    include_position_in_obs: bool,
    include_equity_norm_in_obs: bool,
    include_drawdown_in_obs: bool,
    include_bench_pos_in_obs: bool,
) -> List[str]:
    cols: List[str] = []
    if include_position_in_obs:
        cols.append("position")
    if include_equity_norm_in_obs:
        cols.append("equity_norm")
    if include_drawdown_in_obs:
        cols.append("drawdown")
    if include_bench_pos_in_obs:
        cols.append("bench_pos")
    return cols


def build_observation_contract(
    *,
    mode: str,
    market_feature_columns: Sequence[str],
    regime_feature_columns: Sequence[str],
    include_position_in_obs: bool = True,
    include_equity_norm_in_obs: bool = True,
    include_drawdown_in_obs: bool = True,
    include_bench_pos_in_obs: bool = False,
) -> ObservationContract:
    m = validate_observation_mode(mode)

    market_cols = [str(c) for c in market_feature_columns]
    regime_cols = [str(c) for c in regime_feature_columns]

    env_state_cols = build_env_state_columns(
        include_position_in_obs=include_position_in_obs,
        include_equity_norm_in_obs=include_equity_norm_in_obs,
        include_drawdown_in_obs=include_drawdown_in_obs,
        include_bench_pos_in_obs=include_bench_pos_in_obs,
    )

    if m == "market_only":
        final_agent_cols = list(market_cols)
    elif m == "regime_only":
        final_agent_cols = list(regime_cols)
    else:
        final_agent_cols = list(market_cols) + list(regime_cols)

    if len(final_agent_cols) != len(set(final_agent_cols)):
        dupes = []
        seen = set()
        for c in final_agent_cols:
            if c in seen and c not in dupes:
                dupes.append(c)
            seen.add(c)
        raise ValueError(f"Duplicate feature columns in final_agent_feature_columns: {dupes}")

    return ObservationContract(
        mode=m,
        market_feature_columns=market_cols,
        regime_feature_columns=regime_cols,
        env_state_columns=env_state_cols,
        final_agent_feature_columns=final_agent_cols,
    )