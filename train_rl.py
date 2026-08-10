import gym
import f110_gym
import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.distributions import Normal

from config import EnvConfig, Hyperparameters
from network import ActorCritic, scale_action
from reward import calc_reward, CENTERLINE
from ppo_agent import update_model


# NOTE: so entropy works, however the noise masks the actor critic model, so thats problematic
# we have gotten to an ALRIGHT stopping point, but the car is frozen beacuse the policy is completely frozen
# we need to find a balance between exploration and exploitation if possible
# i am also considering prioritizing crashing a little bit, as a simulated annealing approach, so that the car can travel farther distances
# lastly(and probably most doable in the immediate moment), is to clean up this code, not only making it readable, but also ensuring it still works after refactor.
# consider trying the f1tenth ros2 labs or whatever they are called, this could be because i didnt do that first, so consider that.


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
    env = gym.make(
        "f110-v0", map=map_path, map_ext=".png", num_agents=EnvConfig.num_agents
    )

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

            # STEP C: Calculate custom rewards per agent.
            # Robustly convert gym_dones to a 1D array of shape (NUM_AGENTS)

            if isinstance(gym_dones, (bool, np.bool_)):
                # If gym returned a single boolean, duplicate it for all agents
                gym_dones_arr = np.full(EnvConfig.num_agents, gym_dones, dtype=bool)
            elif isinstance(gym_dones, dict):
                gym_dones_arr = np.array(
                    [
                        gym_dones.get(f"agent_{i}", gym_dones.get(i, False))
                        for i in range(EnvConfig.num_agents)
                    ],
                    dtype=bool,
                )
            else:
                gym_dones_arr = np.atleast_1d(np.array(gym_dones, dtype=bool))

            step_rewards_list = []
            for i in range(EnvConfig.num_agents):
                # Extract single agent observation slice
                # IF agent was already dead before this step, give 0 reward and move on
                if all_dones[i]:
                    step_rewards_list.append(0.0)
                    continue

                # Check if this specific agent crashed ON THIS STEP.
                agent_done = gym_dones_arr[i]
                agent_obs = {
                    "poses_x": next_obs["poses_x"][i],
                    "poses_y": next_obs["poses_y"][i],
                    "poses_theta": next_obs["poses_theta"][i],
                    "linear_vels_x": next_obs["linear_vels_x"][i],
                    "scans": next_obs["scans"][i],
                }
                r = calc_reward(agent_obs, done=agent_done, agent_id=i)
                step_rewards_list.append(r)

            all_dones = np.logical_or(all_dones, gym_dones_arr)

            step_rewards = torch.tensor(
                step_rewards_list, dtype=torch.float32, device=device
            )
            step_dones = torch.tensor(gym_dones, dtype=torch.bool, device=device)

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
