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
            nn.Linear(128, 1),  # the single value score V^pi(s)
        )

    def forward(self, state):
        # Critic evalutes how good/dangerous the state is
        state_value = self.critic(state)

        # Actor calculates mean action
        x = self.actor_base(state)
        raw_actions = torch.tanh(self.actor_head(x))

        # i removed dimensional check here, but if it complains, then reminder to do the dimensional check [...,1] and [...,0] respectively
        return raw_actions, state_value

