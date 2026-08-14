"""
train_gpu_rl.py - GPU PPO Trainer
Trains ActorCritic on 1024 parallel agents in CUDA VRAM.
"""

import time
import torch
import torch.optim as optim
from torch.distributions import Normal

from config import EnvConfig, Hyperparameters
from gpu_env import PyTorchGPUEnv
from network import ActorCritic, scale_action


def train():
    device = torch.device("cuda")
    print(f"Initializing 1,024 Parallel Agents on {torch.cuda.get_device_name(0)}...")

    env = PyTorchGPUEnv(num_agents=1024, device="cuda")
    model = ActorCritic(input_dim=EnvConfig.input_dim).to(device)

    # Lower learning rate slightly to keep gradients stable on CUDA
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    action_std = torch.tensor(Hyperparameters.init_action_std, device=device)

    obs = env.reset()
    start_time = time.time()
    total_steps = 0

    print("Starting CUDA Training Loop...")

    for epoch in range(1, 51):  # 50 Epochs
        states, log_probs, rewards, values, dones = [], [], [], [], []

        for _ in range(Hyperparameters.update_every):
            action_means, state_values = model(obs)

            dist = Normal(action_means, action_std)
            raw_actions = dist.sample()
            raw_actions_clamped = torch.clamp(raw_actions, -1.0, 1.0)

            log_prob = dist.log_prob(raw_actions_clamped).sum(dim=-1)
            scaled_actions = scale_action(raw_actions_clamped)

            next_obs, step_rewards, step_dones, _ = env.step(scaled_actions)

            states.append(obs)
            log_probs.append(log_prob)
            rewards.append(step_rewards)
            values.append(state_values.squeeze(-1))
            dones.append(step_dones)

            obs = next_obs
            total_steps += env.num_agents

        # --- PPO UPDATE ON GPU ---
        returns = []
        with torch.no_grad():
            _, next_val = model(obs)
            R = next_val.squeeze(-1)

        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + Hyperparameters.gamma * R * (~d).float()
            returns.insert(0, R)

        returns = torch.stack(returns)
        # Normalize returns to prevent policy gradient explosion
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = (returns - values).pow(2).mean()
        total_loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), Hyperparameters.clip_grad_norm
        )
        optimizer.step()

        telemetry = env.get_telemetry()
        print(
            f"[Epoch {epoch:02d}/50] "
            f"Active: {telemetry['active_agents']}/1024 | "
            f"Avg Speed: {telemetry['mean_speed']:.2f} m/s | "
            f"Loss: {total_loss.item():.4f}"
        )

    elapsed = time.time() - start_time
    print(f"----------------------------------------")
    print(f"Training Complete! Processed {total_steps:,} transitions in {elapsed:.2f}s")
    print(f"Speed: {total_steps / elapsed:,.2f} Steps/Second")
    print(f"----------------------------------------")

    torch.save(model.state_dict(), "f1_actor_critic.pt")
    print("Saved weights to 'f1_actor_critic.pt'.")


if __name__ == "__main__":
    train()
