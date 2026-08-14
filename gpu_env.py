import torch
import numpy as np
import os
from config import EnvConfig


class PyTorchGPUEnv:
    def __init__(self, num_agents=1024, device="cuda"):
        self.num_agents = num_agents
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dt = 0.05  # Time step (20 Hz)
        self.wheelbase = 0.33  # F1TENTH vehicle scale (meters)

        # 1. Load Centerline Map onto GPU VRAM
        if os.path.exists(EnvConfig.map_file):
            raw_data = np.loadtxt(EnvConfig.map_file, delimiter=",", skiprows=1)
            # Centerline coordinates (N, 2)
            self.centerline = torch.tensor(
                raw_data[:, :2], dtype=torch.float32, device=self.device
            )
        else:
            raise FileNotFoundError(f"Map file {EnvConfig.map_file} not found!")

        # 2. Allocate Vehicle State Tensors on CUDA (Shape: [1024])
        self.x = torch.zeros(self.num_agents, device=self.device)
        self.y = torch.zeros(self.num_agents, device=self.device)
        self.yaw = torch.zeros(self.num_agents, device=self.device)
        self.v_x = torch.zeros(self.num_agents, device=self.device)
        self.steer = torch.zeros(self.num_agents, device=self.device)

        self.dones = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(self.num_agents, dtype=torch.int32, device=self.device)

        self.reset()

    def reset(self, mask=None):
        """
        Resets agents. If a boolean mask is provided, resets ONLY crashed agents
        without interrupting active ones!
        """
        if mask is None:
            mask = torch.ones(self.num_agents, dtype=torch.bool, device=self.device)

        # Start positions near origin or first waypoint
        start_wp = self.centerline[0]

        # Reset masked agents
        if mask.any():
            # Generate noise using rand_like on a sliced tensor to avoid signature mismatch
            noise_x = (torch.rand_like(self.x[mask]) - 0.5) * 0.2
            noise_y = (torch.rand_like(self.y[mask]) - 0.5) * 0.2

            self.x[mask] = start_wp[0] + noise_x
            self.y[mask] = start_wp[1] + noise_y
            self.yaw[mask] = 0.0
            self.v_x[mask] = 0.0
            self.steer[mask] = 0.0
            self.dones[mask] = False
            self.steps[mask] = 0

        return self._get_obs()

    def step(self, actions):
        """
        actions: Tensor of shape (1024, 2) -> [target_steering, target_acceleration]
        ALL vector operations execute in parallel across CUDA cores!
        """
        target_steer = actions[:, 0]
        accel = actions[:, 1]

        # 1. Kinematic State Update (Fully Vectorized on GPU)
        self.steer = torch.clamp(target_steer, -0.4, 0.4)
        self.v_x = torch.clamp(self.v_x + accel * self.dt, 0.0, 8.0)

        # Kinematic Bicycle Yaw & Coordinate Integration
        self.yaw += (self.v_x / self.wheelbase) * torch.tan(self.steer) * self.dt
        self.x += self.v_x * torch.cos(self.yaw) * self.dt
        self.y += self.v_x * torch.sin(self.yaw) * self.dt

        self.steps += 1

        # 2. Track Distance Calculations (Batched Euclidean Distances)
        # Shape: (1024, 2)
        pos = torch.stack([self.x, self.y], dim=1)

        # Compute distance matrix between ALL 1024 agents and ALL map waypoints
        # pos[:, None, :] shape (1024, 1, 2) - centerline[None, :, :] shape (1, N_wp, 2)
        dists_to_waypoints = torch.norm(
            pos[:, None, :] - self.centerline[None, :, :], dim=2
        )

        min_dists, closest_wp_indices = torch.min(dists_to_waypoints, dim=1)

        # 3. Collision Checks & Penalties (Track boundary check: e.g., > 1.5m off centerline)
        off_track = min_dists > 1.1
        max_steps_reached = self.steps >= EnvConfig.max_steps

        self.dones = off_track | max_steps_reached

        # 4. Batched GPU Rewards
        # Forward speed incentive along track
        rewards = self.v_x * 0.5 - (min_dists * 1.0)

        # Inverse/Kinetic Crash Penalty or Flat Crash Penalty
        rewards = torch.where(
            self.dones, torch.tensor(-20.0, device=self.device), rewards
        )

        # Auto-reset crashed agents seamlessly on GPU without halting active ones
        if self.dones.any():
            self.reset(mask=self.dones)

        return self._get_obs(), rewards, self.dones, {}

    def _get_obs(self):
        """
        Returns observation tensor directly in GPU VRAM (Shape: [1024, 4])
        [x_rel, y_rel, v_x, steer]
        """
        return torch.stack([self.x, self.y, self.v_x, self.steer], dim=1)

    def get_telemetry(self):
        """Returns real-time training metrics across all active GPU agents."""
        return {
            "mean_speed": self.v_x[~self.dones].mean().item(),
            "max_speed": self.v_x.max().item(),
            "active_agents": (~self.dones).sum().item(),
            "mean_steps_alive": self.steps.float().mean().item(),
        }
