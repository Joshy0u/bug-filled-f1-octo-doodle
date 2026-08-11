import gym
import f110_gym
import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.distributions import Normal
from env_setup import make_f110_env, configure_agents_and_ghostmode


from config import EnvConfig, Hyperparameters
from network import ActorCritic, scale_action
from reward import calc_reward, CENTERLINE
from ppo_agent import update_model

"""
MULTI-AGENT COLLISION & EPISODE LOGIC (f110_gym)
------------------------------------------------
1. DO NOT collapse `gym_dones` into a single scalar/boolean. f110_gym returns 
   a global boolean wrapper that trips as soon as ANY single car collides.
   
2. Retrieve true per-agent collision states directly from the simulator:
      collisions = np.array(env.unwrapped.sim.collisions, dtype=bool)

3. Maintain an independent `all_dones` boolean vector (size = num_agents):
   - When `collisions[i] == True`, set `all_dones[i] = True`.
   - Freeze Agent `i` action to [0.0, 0.0] and reward to 0.0 for subsequent steps.
   - Continue the episode until `np.all(all_dones) == True`.
"""


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 1: instantiate the Model and Optimizer
    model = ActorCritic().to(device)
    if os.path.exists("f1_actor_critic.pt"):
        model.load_state_dict(
            torch.load("f1_actor_critic.pt", map_location=device, weights_only=True)
        )
        print("Loaded existing model weights from 'f1_actor_critic.pt'.")

    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # load staring position and setup env+map
    centerline = CENTERLINE
    if centerline is None:
        centerline = np.loadtxt(EnvConfig.map_file, delimiter=",", skiprows=1)
    # start_x, start_y = centerline[0, 0], centerline[0, 1]
    # start_yaw = np.arctan2(centerline[1, 1] - start_y, centerline[1, 0] - start_x)

    # Initialize environment with GUI rendering
    map_path = os.path.abspath(EnvConfig.map_prefix)
    env = make_f110_env(map_path, ".png")
    # Action standard deviation for Gaussian exploration
    # currently its a static value range, but this means it wont rely on the model weights
    # for example if a car takes a good turn that step, the next step would be completely random, so over time we need to decrease
    # the use of action_std randomness, to allow the model to learn and exploit good actions.

    action_std = torch.tensor(
        Hyperparameters.init_action_std
    )  # [steering_std, speed_std]
    min_action_std = torch.tensor(
        Hyperparameters.min_action_std
    ).to(  # to force more steering choices
        device
    )  # Minimum std for exploration

    recent_steps = []
    last_metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "total_loss": 0.0,
        "entropy_loss": 0.0,
        "mean_val": 0.0,
    }

    # this is where the the main core loop goes:
    for episode in range(1, Hyperparameters.num_episodes + 1):
        # 1. Generate staggered starting poses along the centerline for 4 car
        start_poses = []
        for i in range(EnvConfig.num_agents):
            idx = (i * 3) % len(centerline)
            next_idx = (idx + 1) % len(centerline)
            x, y = centerline[idx, 0], centerline[idx, 1]
            nextX, nextY = centerline[next_idx, 0], centerline[next_idx, 1]
            yaw = np.arctan2(nextY - y, nextX - x)
            start_poses.append([x, y, yaw])

        obs, reward, done_signal, _ = env.reset(poses=np.array(start_poses))
        configure_agents_and_ghostmode(env)
        states, log_probs, entropies = [], [], []
        rewards, next_states, dones, values = [], [], [], []

        episode_rewards = np.zeros(EnvConfig.num_agents)
        step = 0
        all_dones = np.zeros(EnvConfig.num_agents, dtype=bool)

        while not np.all(all_dones) and step < EnvConfig.max_steps:
            step += 1

            # STEP A: Feed multi-agent LiDAR scan array(Shape: 4, 1080) to PyTorch
            scan_array = np.array(obs["scans"], dtype=np.float32)
            state_tensor = torch.FloatTensor(scan_array).to(device)

            # Batched forward pass for all 4 agents
            action_means, state_values = model(state_tensor)

            dist = Normal(action_means, action_std)
            raw_actions = dist.sample()
            raw_actions_clamped = torch.clamp(raw_actions, -1.0, 1.0)

            log_prob = dist.log_prob(raw_actions_clamped).sum(dim=-1)
            step_entropy = dist.entropy().sum(dim=-1)

            # Scale actions for environment (Shape: 4,2)
            scaled_actions = scale_action(raw_actions_clamped)
            env_actions = scaled_actions.cpu().numpy()

            # Zero out actions for agents that already crashed so they stop moving
            for i in range(EnvConfig.num_agents):
                if all_dones[i]:
                    env_actions[i] = [0.0, 0.0]

            # STEP B: Step the simulator
            next_obs, _, gym_dones, info = env.step(env_actions)
            env.render(mode="human")
            if step <= 2:
                configure_agents_and_ghostmode(env)

            # STEP C: Calculate custom rewards per agent.
            # Robustly convert gym_dones to a 1D array of shape (NUM_AGENTS)
            # ^^ this comment above might have been the problem, one true value abruptly ends all the agents.

            sim_obj = getattr(env.unwrapped, "sim", None)
            if sim_obj is not None and hasattr(sim_obj, "collisions"):
                raw_collisions = np.array(sim_obj.collisions, dtype=bool)
            elif isinstance(info, dict) and "collisions" in info:
                raw_collisions = np.array(info["collisions"], dtype=bool)
            else:
                raw_collisions = np.atleast_1d(gym_dones)
                if len(raw_collisions) < EnvConfig.num_agents:
                    raw_collisions = np.full(EnvConfig.num_agents, bool(gym_dones))

            for i in range(EnvConfig.num_agents):
                if raw_collisions[i]:
                    all_dones[i] = True

            # ---Reward Calculation per agent---
            step_rewards_list = []
            for i in range(EnvConfig.num_agents):
                # Extract single agent observation slice
                # 1. If agent was already dead before this step, give 0 reward and keep them frozen.
                if all_dones[i] and not raw_collisions[i]:
                    step_rewards_list.append(0.0)
                    continue

                # Check if this specific agent crashed ON THIS STEP.
                # 2. Extract single agent observation dictionary.
                agent_obs = {
                    "poses_x": next_obs["poses_x"][i],
                    "poses_y": next_obs["poses_y"][i],
                    "poses_theta": next_obs["poses_theta"][i],
                    "linear_vels_x": next_obs["linear_vels_x"][i],
                    "scans": next_obs["scans"][i],
                }

                # 3. Pass agent's specific collisions status into reward function.
                agent_is_dead = all_dones[i]  # true on the step it collides.
                r = calc_reward(agent_obs, done=agent_is_dead, agent_id=i)
                step_rewards_list.append(r)

            step_rewards = torch.tensor(
                step_rewards_list, dtype=torch.float32, device=device
            )
            step_dones = torch.tensor(all_dones, dtype=torch.bool, device=device)

            next_scan_array = np.array(next_obs["scans"], dtype=np.float32)
            next_state_tensor = torch.FloatTensor(next_scan_array).to(device)

            # buffer updates
            states.append(state_tensor)
            log_probs.append(log_prob)
            entropies.append(step_entropy)
            rewards.append(step_rewards)
            next_states.append(next_state_tensor)
            dones.append(step_dones)
            values.append(state_values)

            episode_rewards += np.array(step_rewards_list)
            obs = next_obs

            # STEP E: Learning step
            if len(states) >= Hyperparameters.update_every or np.all(all_dones):
                last_metrics = update_model(
                    model,
                    optimizer,
                    states,
                    log_probs,
                    rewards,
                    next_states,
                    dones,
                    values,
                    entropies,
                    Hyperparameters.gamma,
                )
                states, log_probs, entropies = [], [], []
                rewards, next_states, dones, values = [], [], [], []
        recent_steps.append(step)
        print(
            f"Episode {episode:02d}/{Hyperparameters.num_episodes} | Max Steps: {step:03d} | Mean Agent Score: {np.mean(episode_rewards):+.2f}"
        )

        if episode % 10 == 0:
            print("-" * 40)
            print(f"--- DIAGNOSTICS AT EPISODE {episode:03d} ---")
            print(f"  Avg Max Steps (last 10): {np.mean(recent_steps[-10:]):.1f}")
            print(f"  Critic Loss:         {last_metrics['critic_loss']:.4f}")
            print(f"  Actor Loss:          {last_metrics['actor_loss']:.4f}")
            print(f"  Total Loss:          {last_metrics['total_loss']:.4f}")
            print(f"  Entropy Loss:        {last_metrics['entropy_loss']:.4f}")
            print(f"  Avg Predicted Val:   {last_metrics['mean_val']:.4f}")
            print(f"  Current Action Std:  {action_std.cpu().numpy().round(3)}")
            print("-" * 40)

        # Decay exploration noise per episode
        action_std = torch.max(action_std * 0.998, min_action_std)

    torch.save(model.state_dict(), "f1_actor_critic.pt")
    print("\nTraining complete. Model saved as 'f1_actor_critic.pt'.")
    env.close()


if __name__ == "__main__":
    main()
