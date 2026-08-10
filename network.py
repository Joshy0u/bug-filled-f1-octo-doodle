"""
network.py
Neural network architecture and action space conversion for Actor-Critic
"""

import torch
import torch.nn as nn
from config import EnvConfig


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int = EnvConfig.input_dim):
        super(ActorCritic, self).__init__()

        # Actor (driver)
        self.actor_base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(128, 2)  # [Mean steering, Mean speed]

        # Critic (race engineer)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # State Value (V)s
        )

    def forward(self, state):
        state_value = self.critic(state)
        x = self.actor_base(state)
        raw_actions = torch.tanh(self.actor_head(x))
        return raw_actions, state_value


def scale_action(raw_action: torch.Tensor) -> torch.Tensor:
    """
    Scales raw continuous model outputs [-1,1] to physical vehichle bounds:
    Steering: [-0.4, 0.4] raw
    speed: [1.0, 4.0] m/s
    """
    steering = raw_action[..., 0] * 0.4
    speed = (raw_action[..., 1] + 1.0) * 1.5 + 1.0
    return torch.stack([steering, speed], dim=-1)
