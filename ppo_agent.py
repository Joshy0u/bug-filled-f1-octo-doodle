"""
ppo_agent.py
Handles A2C/PPO update logic (batches) and gradient updates.
"""

import torch
from config import Hyperparameters


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
    gamma=Hyperparameters.gamma,
) -> dict:
    returns = []

    with torch.no_grad():
        _, next_val = model(next_states[-1])
        next_value = next_val.squeeze(-1) * (1.0 - dones[-1].float())

    R = next_value
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1.0 - d.float())
        returns.insert(0, R)

    returns = torch.stack(returns)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)
    values = torch.stack(values).squeeze(-1)

    advantages = returns - values
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = (returns - values).pow(2).mean()
    entropy_loss = entropies.mean()

    total_loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=Hyperparameters.clip_grad_norm
    )
    optimizer.step()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "total_loss": total_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "mean_val": values.mean().item(),
    }
