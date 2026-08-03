import numpy as np

# NOTE: IF YOU WANT THE CAR TO GO FASTER, (I.E CHANGE THE MAX SPEED), YOU ALSO NEED TO CHANGE THE WEIGHTS ACCORDINGLY. (rn its tuned to 0.7+ reward for 7 m/s)

# V1, NOT WORRIED ABOUT RACING LINE OR OPTIMAL TIME JUST YET, ONLY TO FINISH A LAP. 

# need 3 things: progress/speed reward
# wall penalty
# collision penalty

def calc_reward(obs, done):
    """
    Calculates a custom reward signal based on vehicle telemetry and sensor data

    parameters:
        obs(dict): Observation dictionary returned by env.step()
        done(bool): Bool. Whether the car has collided/failed.
    """
    # CRASH PENALTY
    if done:
        return -1.0

    # PROGRESS REWARD
    speed = max(0.0, obs['linear_vels_x'][0])
    min_wall_dist = float(np.min(obs['scans']))

    # Scale speed reward to max ~0.7 
    speed_reward = (speed / 7.0) * 0.7  # Assuming max speed is 10 m/s

    # Scale wall penalty to max ~0.3
    wall_penalty = 0.0
    if min_wall_dist < 0.4:
        wall_penalty = -0.3 * (1.0 - (min_wall_dist / 0.4))

    # total stays cleanly within [-1, 1] range
    total_reward = np.clip(speed_reward + wall_penalty, -1.0, 1.0)

    return float(total_reward)