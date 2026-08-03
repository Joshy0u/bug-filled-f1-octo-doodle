from pickle import TRUE

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

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 1: instantiate the Model and Optimizer
    model = ActorCritic().to(device)
    if os.path.exists("f1_actor_critic.pt"):
        model.load_state_dict(torch.load("f1_actor_critic.pt", map_location=device))
        print("Loaded existing model weights from 'f1_actor_critic.pt'.")

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    gamma = 0.99 # discount factor for future rewards (to prioritize immediate rewards over distant ones)

    # load staring position and setup env+map
    centerline = np.loadtxt("./maps/sakhir_centerline.csv", delimiter=",", skiprows=1)
    start_x, start_y = centerline[0, 0], centerline[0, 1]
    start_yaw = np.arctan2(centerline[1, 1] - start_y, centerline[1, 0] - start_x)

    # Initialize environment with GUI rendering
    map_path = os.path.abspath("./maps/sakhir")
    env = gym.make(
        'f110_gym:f110-v0', 
        map=map_path, 
        map_ext=".png", 
        num_agents=1
    )

    #Action standard deviation for Gaussian exploration
    #currently its a static value range, but this means it wont rely on the model weights
    #for example if a car takes a good turn that step, the next step would be completely random, so over time we need to decrease
    #the use of action_std randomness, to allow the model to learn and exploit good actions.
    
    action_std = torch.tensor([0.2, 0.5]).to(device)  # [steering_std, speed_std]
    min_action_std = torch.tensor([0.05, 0.1]).to(device)  # Minimum std for exploration
    num_episodes = 200

    # this is where the the main core loop goes:
    for episode in range(1, num_episodes+1):
        obs, reward, done, _ = env.reset(np.array([[start_x, start_y, start_yaw]]))
        calc_reward(obs, done=True)  # Reset the last closest index for centerline reward
        episode_reward = 0.0
        step = 0

        while not done:
            step += 1

            # STEP A: Feed LiDAR array -> numPY float32 -> PyTorch GPU Tensor
            scan_array = np.array(obs['scans'], dtype=np.float32)
            state_tensor = torch.FloatTensor(scan_array).to(device)

            # query ACtor-Critic model. 
            action_mean, state_value = model(state_tensor)

            # add exploration noise so the car tests different steering angles
            # Use a Normal distribution to sample the noise
            dist = Normal(action_mean, action_std)
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(dim=-1)  # Log probability of the sampled action

            steering = torch.clamp(raw_action[..., 0], -0.4, 0.4)
            speed = torch.clamp( raw_action[..., 1], 3.0, 7.0)

            action_env = np.array([[steering.item(), speed.item()]])

            # STEP B: Step the Simulator
            next_obs, _, done, info = env.step(action_env)

            # Draw pyGame scene
            #env.render(mode="human")
            #time.sleep(0.01)  # slow down the rendering for visualization

            # STEP C: Calculate Custom Reward
            step_reward = calc_reward(next_obs, done)
            episode_reward += step_reward

            # STEP D: LEARNING STEP (Backpropagation)
            next_scan_array = np.array(next_obs['scans'], dtype=np.float32)
            next_state_tensor = torch.FloatTensor(next_scan_array).to(device)
            with torch.no_grad():
                next_value = torch.zeros(1).to(device) if done else model(next_state_tensor)[1]

            # Calculate Advantage ( Actual reward - Predicted reward)
            td_target = step_reward + gamma * next_value
            advantage = (td_target - state_value).detach()

            # Backpropagation on GPU (calculate Actor and Critic loss)
            actor_loss = -log_prob * advantage # log prob * advantage
            critic_loss = (td_target - state_value).pow(2) # MSE(Target, Predicted value)
            total_loss = actor_loss + 0.5 * critic_loss  # weight critic loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # prevents exploding gradients
            optimizer.step()

            #advance state:
            obs = next_obs

        # Decrease action standard deviation over time (not inside while loop, otherwise it would decrease way too fast, per every step before episode ends)
        action_std = torch.max(action_std * 0.995, min_action_std)

        print(f"Episode {episode:02d}/{num_episodes} | Steps survived: {step:03d} | Total Score: {episode_reward:+.2f}")

    torch.save(model.state_dict(), "f1_actor_critic.pt")
    print("\nTraining complete. Model saved as 'f1_actor_critic.pt'.")
    env.close

    
if __name__ == "__main__":
    main()