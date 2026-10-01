"""Dueling MLP: Q(s,a)=V(s)+A(s,a)-mean_a A(s,a)."""

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.value = nn.Linear(hidden_size, 1)
        self.advantage = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.features(state)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=1, keepdim=True)
