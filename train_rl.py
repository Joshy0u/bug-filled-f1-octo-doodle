import gym 
import f110_gym
import numpy as np
import os
import time

from model import ActorCritic


def main():
    # Load starting position
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


    obs, reward, done, _ = env.reset(np.array([[start_x, start_y, start_yaw]]))
    print("\n--- LIGHTS OUT AND AWAY WE GO (Visual started) ---")

    for step in range(1, 1000):
        # Action: Steer right (+0.25 rad) at 7 m/s
        action = np.array([[0.0, 10.0]]) # (steering angle, speed meters/sec)
        obs, reward, done, info = env.step(action)

        # Draw the scene
        env.render(mode='human')
        time.sleep(0.02)

        if done:
            print(f"\n💥 CRASH DETECTED at Step {step}!")
            time.sleep(1.0)
            break

    env.close()

if __name__ == "__main__":
    main()