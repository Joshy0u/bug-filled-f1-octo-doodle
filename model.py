import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, input_dim=1080):
        super(ActorCritic, self).__init__()

        # Actor (Driver), takes 1080-dimensional LiDAR input and outputs steering and speed
        self.actor_base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(128, 2)  # Outputs: [Mean Steering, Mean Speed]

        # Critic (Race Engineer)
        # Takes 1080 LiDar Rays -> outputs a single Value score (state eval) V^pi(s)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # the single value score V^pi(s)
        )

    def forward(self, state):
        # Critic evalutes how good/dangerous the state is
        state_value = self.critic(state)

        # Actor calculates mean action
        x = self.actor_base(state)
        raw_actions = torch.tanh(self.actor_head(x))

        #dimensional check
        if raw_actions.dim() > 1:
            raw_steering = raw_actions[..., 0]
            raw_speed = raw_actions[..., 1]
        else: 
            raw_steering = raw_actions[0]
            raw_speed = raw_actions[1]


        # Scale raw actions to vehicle limits
        # Steering -0.4 rad to +0.4 rad |  Speed 1 m/s to 7 m/s
        # The raw actions are in the range [-1, 1], so we need to scale them to the desired range.
        steering = raw_steering * 0.4
        speed = (raw_speed+1.0) * 3.0 + 1.0

        action = torch.stack([steering, speed], dim=-1)

        return action, state_value