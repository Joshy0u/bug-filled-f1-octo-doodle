"""
gpu_env.py - Pure PyTorch GPU Environment
Computes 1082-dim observations (1080 LiDAR + speed + steer) and rewards on CUDA.
"""

import os
import torch
import numpy as np
from config import EnvConfig, rewardParams


class PyTorchGPUEnv:
    def __init__(self, num_agents=1024, device="cuda"):
        self.num_agents = num_agents
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dt = 0.05  # 20 Hz
        self.wheelbase = 0.33
        self.max_lidar_range = 30.0
        self.num_rays = 1080

        # Fixed ray angles across 270 degrees FOV (-135 to +135 deg)
        self.ray_angles = torch.linspace(
            -1.35 * np.pi / 2, 1.35 * np.pi / 2, self.num_rays, device=self.device
        )

        # Load Centerline Map
        if os.path.exists(EnvConfig.map_file):
            raw_data = np.loadtxt(EnvConfig.map_file, delimiter=",", skiprows=1)
            self.centerline = torch.tensor(
                raw_data[:, :2], dtype=torch.float32, device=self.device
            )
        else:
            raise FileNotFoundError(f"Map file {EnvConfig.map_file} not found!")

        # Pre-calculate centerline segment orientation vectors for progress math
        next_wps = torch.roll(self.centerline, shifts=-1, dims=0)
        self.wp_vectors = next_wps - self.centerline

        # Allocate Vehicle States on GPU [Shape: 1024]
        self.x = torch.zeros(self.num_agents, device=self.device)
        self.y = torch.zeros(self.num_agents, device=self.device)
        self.yaw = torch.zeros(self.num_agents, device=self.device)
        self.v_x = torch.zeros(self.num_agents, device=self.device)
        self.steer = torch.zeros(self.num_agents, device=self.device)

        self.last_closest_idx = torch.zeros(
            self.num_agents, dtype=torch.long, device=self.device
        )
        self.dones = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(self.num_agents, dtype=torch.int32, device=self.device)

        self.reset()

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.num_agents, dtype=torch.bool, device=self.device)

        start_wp = self.centerline[0]
        next_wp = self.centerline[1]
        start_yaw = torch.atan2(next_wp[1] - start_wp[1], next_wp[0] - start_wp[0])

        if mask.any():
            noise_x = (torch.rand_like(self.x[mask]) - 0.5) * 0.2
            noise_y = (torch.rand_like(self.y[mask]) - 0.5) * 0.2

            self.x[mask] = start_wp[0] + noise_x
            self.y[mask] = start_wp[1] + noise_y
            self.yaw[mask] = start_yaw
            self.v_x[mask] = 0.0
            self.steer[mask] = 0.0
            self.last_closest_idx[mask] = 0
            self.dones[mask] = False
            self.steps[mask] = 0

        return self._get_obs()

    def _get_simulated_lidar(self):
        """Simulates 1080 2D LiDAR raycasts using track corridor distances."""
        # Ray global angles: [1024, 1080]
        global_ray_angles = self.yaw[:, None] + self.ray_angles[None, :]

        # Compute perpendicular track distance along ray projections
        pos = torch.stack([self.x, self.y], dim=1)
        dists_to_wps = torch.norm(pos[:, None, :] - self.centerline[None, :, :], dim=2)
        min_dists, _ = torch.min(dists_to_wps, dim=1)

        # Effective wall distance estimate: 1.1m track limit - car position
        dist_to_wall = torch.clamp(1.1 - min_dists, min=0.05, max=1.1)

        # Rays facing sideways hit walls faster than forward-facing rays
        cos_components = torch.abs(torch.cos(self.ray_angles[None, :])) + 0.1
        scans = dist_to_wall[:, None] / cos_components
        return torch.clamp(scans, 0.0, self.max_lidar_range)

    def _get_obs(self):
        """Returns exact 1082-dim tensor: [1080 scans, v_x, steer]."""
        scans = self._get_simulated_lidar()
        vels = self.v_x[:, None]
        steer = self.steer[:, None]
        return torch.cat([scans, vels, steer], dim=1)

    def step(self, actions):
        """
        actions: Tensor [1024, 2] -> [scaled_steering, scaled_speed]
        """
        target_steer = actions[:, 0]
        target_speed = actions[:, 1]

        # 1. Bicycle Kinematics
        self.steer = torch.clamp(target_steer, -0.4, 0.4)
        accel = (target_speed - self.v_x) * 2.0  # Simple speed controller
        self.v_x = torch.clamp(self.v_x + accel * self.dt, 0.0, 8.0)

        self.yaw += (self.v_x / self.wheelbase) * torch.tan(self.steer) * self.dt
        self.x += self.v_x * torch.cos(self.yaw) * self.dt
        self.y += self.v_x * torch.sin(self.yaw) * self.dt
        self.steps += 1

        # 2. Track Distances & Progress Calculations
        pos = torch.stack([self.x, self.y], dim=1)
        dists_to_wps = torch.norm(pos[:, None, :] - self.centerline[None, :, :], dim=2)
        dist_to_centerline, closest_idx = torch.min(dists_to_wps, dim=1)

        # 3. Collisions & Dones
        off_track = dist_to_centerline > 1.1
        max_steps = self.steps >= EnvConfig.max_steps
        self.dones = off_track | max_steps

        # 4. CUDA Vectorized Rewards matching your reward.py multipliers
        # Progress along waypoints
        idx_delta = (closest_idx - self.last_closest_idx) % len(self.centerline)
        progress_reward = (
            idx_delta.float()
            * rewardParams.progress_multiplier
            * (1.0 + 0.5 * self.v_x)
        )

        # --- FIXED REWARD MATH ---

        # 1. Scale alignment reward by velocity (you shouldn't get aligned rewards if standing still!)
        target_vecs = self.wp_vectors[closest_idx]
        target_yaws = torch.atan2(target_vecs[:, 1], target_vecs[:, 0])
        heading_diff = torch.abs(
            torch.atan2(
                torch.sin(self.yaw - target_yaws), torch.cos(self.yaw - target_yaws)
            )
        )
        # Multiply by velocity or a movement factor so stationary cars get 0 alignment reward
        movement_mask = (self.v_x > 0.1).float()
        alignment_reward = (
            rewardParams.alignment_multiplier * torch.cos(heading_diff) * movement_mask
        )

        # 2. Add a explicit penalty for standing still (breaks zero-velocity local minima)
        stationary_penalty = torch.where(self.v_x < 0.2, -2.0, 0.0)

        # 3. Speed reward
        forward_speed = self.v_x * torch.cos(heading_diff)
        speed_reward = torch.clamp(
            forward_speed * rewardParams.speed_multiplier, min=0.0
        )

        # 4. Off-center penalty
        off_center_penalty = torch.where(
            dist_to_centerline > 0.4,
            -1.0 * (dist_to_centerline - 0.4),
            torch.tensor(0.0, device=self.device),
        )

        # 5. Fixed Crash Penalty (Use abs() so quadratic term doesn't wildly explode)
        base_crash = -abs(rewardParams.crash_penalty)
        kinetic_crash = base_crash - (1.0 * (self.v_x**2))  # Scaled down from 15.0

        # Combine step rewards
        rewards = (
            progress_reward
            + speed_reward
            + off_center_penalty
            + alignment_reward
            + stationary_penalty
            + rewardParams.step_cost
        )

        # Apply crash penalty on done step
        rewards = torch.where(self.dones, kinetic_crash, rewards)
        # Update last closest waypoint index
        self.last_closest_idx = closest_idx

        # Auto-reset crashed agents seamlessly
        if self.dones.any():
            self.reset(mask=self.dones)

        return self._get_obs(), rewards, self.dones, {}

    def get_telemetry(self):
        return {
            "mean_speed": self.v_x[~self.dones].mean().item(),
            "max_speed": self.v_x.max().item(),
            "active_agents": (~self.dones).sum().item(),
            "mean_steps_alive": self.steps.float().mean().item(),
        }
