"""
render_eval.py - Stage 2 Evaluation
Loads GPU-trained model weights into f110_gym for real-time visual inspection.
"""

import os
import torch
import numpy as np
import gym
import f110_gym

from config import EnvConfig
from network import ActorCritic, scale_action
from env_setup import make_f110_env


def main():
    device = torch.device("cpu")

    # 1. Instantiate network & load GPU-trained weights
    model = ActorCritic(input_dim=EnvConfig.input_dim).to(device)
    if os.path.exists("f1_actor_critic.pt"):
        model.load_state_dict(
            torch.load("f1_actor_critic.pt", map_location=device, weights_only=True)
        )
        print("Successfully loaded trained 'f1_actor_critic.pt' weights!")
    else:
        print("Warning: 'f1_actor_critic.pt' not found. Running with random weights.")

    model.eval()

    # 2. Initialize f110_gym environment with GUI
    map_path = os.path.abspath(EnvConfig.map_prefix)
    env = make_f110_env(map_path, ".png")

    default_poses = np.zeros((EnvConfig.num_agents, 3))
    obs, _, done, _ = env.reset(poses=default_poses)
    current_steer = np.zeros((EnvConfig.num_agents, 1), dtype=np.float32)

    print("\nRunning Visual Evaluation in f110_gym... Press Ctrl+C to exit.")

    while True:
        # Format shapes precisely for N agents
        scans = np.array(obs["scans"], dtype=np.float32)  # Shape: (4, 1080)
        vels = np.array(obs["linear_vels_x"], dtype=np.float32).reshape(
            -1, 1
        )  # Shape: (4, 1)
        steer = current_steer.reshape(-1, 1)  # Shape: (4, 1)

        # Stack into 1082-dim vector per agent -> Shape: (4, 1082)
        state_array = np.hstack([scans, vels, steer])
        state_tensor = torch.FloatTensor(state_array).to(device)

        with torch.no_grad():
            raw_actions, _ = model(state_tensor)
            scaled_actions = scale_action(raw_actions).cpu().numpy()

        current_steer = scaled_actions[:, 0:1]

        # Step standard f110_gym environment
        obs, reward, done_signal, info = env.step(scaled_actions)
        env.render(mode="human")

        if np.any(done_signal):
            print("Car crashed or finished lap. Resetting...")
            obs, _, done, _ = env.reset(poses=default_poses)


if __name__ == "__main__":
    main()
