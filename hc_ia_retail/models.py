# HC_IA_RETAIL/models.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GRUWindowExtractor(BaseFeaturesExtractor):
    """
    Reçoit obs vectorisée (window * n_features_per_step), reshape -> (B, T, F) puis GRU.
    Note: la "récurrence" est dans la fenêtre, pas stateful cross-episode. Stable et efficace avec SAC.
    """
    def __init__(self, observation_space, window_size: int, per_step_dim: int, gru_hidden: int, gru_layers: int, out_dim: int):
        super().__init__(observation_space, features_dim=out_dim)
        self.window_size = window_size
        self.per_step_dim = per_step_dim

        self.gru = nn.GRU(
            input_size=per_step_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # obs: (B, window*per_step_dim)
        b = obs.shape[0]
        x = obs.view(b, self.window_size, self.per_step_dim)
        y, _ = self.gru(x)
        last = y[:, -1, :]
        return self.mlp(last)