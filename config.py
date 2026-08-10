"""
config.py
Holds configuration constants and hyperparameters for F1TENTH RL.
No dependencies on local files.
"""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    map_file: str = "./maps/sakhir_centerline.csv"
    map_prefix: str = "./maps/sakhir"

    num_agents: int = 4
    input_dim: int = 1080
    max_steps: int = 1000


@dataclass
class Hyperparameters:
    lr: float = 0.0003
    gamma: float = 0.99
    update_every: int = 64
    num_episodes: int = 200
    clip_grad_norm: float = 0.5

    # Action exploration noise:
    init_action_std: tuple = (0.4, 0.2)
    min_action_std: tuple = (0.15, 0.10)
    std_decay: float = 0.998


@dataclass
class rewardParams:
    crash_penalty: float = -10.0
    step_cost: float = -0.05
    progress_multiplier: float = 12.0
    backwards_multiplier: float = -4.0
    alignment_multiplier: float = 0.3
    speed_multiplier: float = 0.5
