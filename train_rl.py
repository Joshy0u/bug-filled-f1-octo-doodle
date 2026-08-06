import gym
import f110_gym
import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.distributions import Normal

from model import ActorCritic
from reward import calc_reward


# NOTE: so entropy works, however the noise masks the actor critic model, so thats problematic
# we have gotten to an ALRIGHT stopping point, but the car is frozen beacuse the policy is completely frozen
# we need to find a balance between exploration and exploitation if possible
# i am also considering prioritizing crashing a little bit, as a simulated annealing approach, so that the car can travel farther distances
# lastly(and probably most doable in the immediate moment), is to clean up this code, not only making it readable, but also ensuring it still works after refactor.
# consider trying the f1tenth ros2 labs or whatever they are called, this could be because i didnt do that first, so consider that.


def scale_action(raw_action):
    """
    we are changing this so that it goes slower, and is able to turn in the physics engine.
    Scales raw continuous model outputs [-1,1] to physical vehicle bounds:
    steering: [-0,4, 0.4] rad
    speed: [1.0, 7.0] m/s
    """
    steering = raw_action[..., 0] * 0.4
    speed = (raw_action[..., 1] + 1.0) * 1.5 + 1.0
    return steering, speed


def update_model(
    model,
    optimizer,
    states,
    log_probs,
    rewards,
    next_states,
    dones,
    values,
    entropies,
    gamma=0.99,
):
    """
    Update the Actor-Critic model using the collected experience.
    Performs N-step Advantage Actor-Critic (A2C) update, over mini-batch.

    Parameters:
        model (ActorCritic): The Actor-Critic model to be updated.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model.
        states (torch.Tensor): Tensor of states.
        log_probs (torch.Tensor): Log probabilities of the actions taken.
        rewards (torch.Tensor): Tensor of rewards received.
        next_states (torch.Tensor): Tensor of next states.
        dones (torch.Tensor): Tensor indicating if the episode is done.
        values (torch.Tensor): Tensor of state values predicted by the critic.
        gamma (float): Discount factor for future rewards.

    Returns:
        None
    """
    returns = []

    # Bootstrap value from the last state if not done
    with torch.no_grad():
        if dones[-1]:
            next_value = 0.0
        else:
            _, next_val = model(next_states[-1])
            next_value = next_val.item()

    # N-step return calculation
    R = next_value
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            R = 0.0
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float32).to(states[0].device)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)
    values = torch.squeeze(torch.stack(values), dim=1)

    # Calculate advantages
    advantages = returns - values

    # advantage normalization
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # losses
    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = (
        (returns - values).pow(2).mean()
    )  # MSE between Discounted Return and Critic Value
    entropy_loss = entropies.mean()

    # Decreased entropy loss weight slightly to prevent high noise overrides
    total_loss = actor_loss + 0.5 * critic_loss - 0.02 * entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=0.5
    )  # prevents exploding gradients
    optimizer.step()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "total_loss": total_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "mean_val": values.mean().item(),
    }


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
    gamma = 0.99  # discount factor for future rewards (to prioritize immediate rewards over distant ones)
    UPDATE_EVERY = 16

    # load staring position and setup env+map
    centerline = np.loadtxt("./maps/sakhir_centerline.csv", delimiter=",", skiprows=1)
    start_x, start_y = centerline[0, 0], centerline[0, 1]
    start_yaw = np.arctan2(centerline[1, 1] - start_y, centerline[1, 0] - start_x)

    # Initialize environment with GUI rendering
    map_path = os.path.abspath("./maps/sakhir")
    env = gym.make("f110-v0", map=map_path, map_ext=".png", num_agents=1)

    # Action standard deviation for Gaussian exploration
    # currently its a static value range, but this means it wont rely on the model weights
    # for example if a car takes a good turn that step, the next step would be completely random, so over time we need to decrease
    # the use of action_std randomness, to allow the model to learn and exploit good actions.

    action_std = torch.tensor([0.6, 0.3]).to(device)  # [steering_std, speed_std]
    min_action_std = torch.tensor([0.35, 0.15]).to(  # to force more steering choices
        device
    )  # Minimum std for exploration
    num_episodes = 300

    recent_steps = []
    last_metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "total_loss": 0.0}

    # this is where the the main core loop goes:
    for episode in range(1, num_episodes + 1):
        obs, reward, done, _ = env.reset(
            poses=np.array([[start_x, start_y, start_yaw]])
        )
        calc_reward(
            obs, done=True
        )  # Reset the last closest index for centerline reward

        # batch buffers for A2C update
        states, log_probs, entropies = [], [], []
        rewards, next_states, dones, values = [], [], [], []

        episode_reward = 0.0
        step = 0

        while not done:
            step += 1

            # STEP A: Feed LiDAR array -> numPY float32 -> PyTorch GPU Tensor
            scan_array = np.array(obs["scans"], dtype=np.float32)
            state_tensor = torch.FloatTensor(scan_array).to(device)

            # query ACtor-Critic model.
            action_mean, state_value = model(state_tensor)

            # add exploration noise so the car tests different steering angles
            # Use a Normal distribution to sample the noise
            dist = Normal(action_mean, action_std)
            raw_action = dist.sample()
            raw_action_clamped = torch.clamp(raw_action, -1.0, 1.0)

            log_prob = dist.log_prob(raw_action).sum(
                dim=-1
            )  # Log probability of the sampled action
            # calculate entropy for this step and store it.
            step_entropy = dist.entropy().sum(dim=-1)

            steering, speed = scale_action(raw_action_clamped)
            action_env = np.array([[steering.item(), speed.item()]])

            # STEP B: Step the Simulator
            next_obs, _, done, info = env.step(action_env)

            # Draw pyGame scene
            # env.render(mode="human")
            # time.sleep(0.01)  # slow down the rendering for visualization

            # STEP C: Calculate Custom Reward
            step_reward = calc_reward(next_obs, done)

            # STEP D: LEARNING STEP (Backpropagation)
            next_scan_array = np.array(next_obs["scans"], dtype=np.float32)
            next_state_tensor = torch.FloatTensor(next_scan_array).to(device)

            # store transition in batch memory
            states.append(state_tensor)
            log_probs.append(log_prob)
            entropies.append(step_entropy)
            rewards.append(step_reward)
            next_states.append(next_state_tensor)
            dones.append(done)
            values.append(state_value)

            episode_reward += step_reward

            # advance state:
            obs = next_obs
            if len(states) >= UPDATE_EVERY or done:
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
                    gamma,
                )

                # clear batch memory
                states, log_probs, entropies = [], [], []
                rewards, next_states, dones, values = [], [], [], []

        # Decrease action standard deviation over time (not inside while loop, otherwise it would decrease way too fast, per every step before episode ends)
        action_std = torch.max(action_std * 0.98, min_action_std)
        recent_steps.append(step)

        print(
            f"Episode {episode:02d}/{num_episodes} | Steps survived: {step:03d} | Total Score: {episode_reward:+.2f}"
        )
        # Diagnostics block every 10 episodes
        if episode % 10 == 0:
            print("-" * 40)
            print(f"--- DIAGNOSTICS AT EPISODE {episode:03d} ---")
            print(f"  Avg Steps (last 10): {np.mean(recent_steps[-10:]):.1f}")
            print(f"  Critic Loss:        {last_metrics['critic_loss']:.4f}")
            print(f"  Actor Loss:         {last_metrics['actor_loss']:.4f}")
            print(f"  Total Loss:         {last_metrics['total_loss']:.4f}")
            print(f"  Entropy Loss:       {last_metrics['entropy_loss']:.4f}")
            print(f"  Avg Predicted Val:  {last_metrics['mean_val']:.4f}")
            print(f"  Current Action Std: {action_std.cpu().numpy().round(3)}")
            print("-" * 40)

    torch.save(model.state_dict(), "f1_actor_critic.pt")
    print("\nTraining complete. Model saved as 'f1_actor_critic.pt'.")
    env.close


if __name__ == "__main__":
    main()
