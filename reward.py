import os

import numpy as np

# NOTE: IF YOU WANT THE CAR TO GO FASTER, (I.E CHANGE THE MAX SPEED), YOU ALSO NEED TO CHANGE THE WEIGHTS ACCORDINGLY. (rn its tuned to 0.7+ reward for 7 m/s)

# V1, NOT WORRIED ABOUT RACING LINE OR OPTIMAL TIME JUST YET, ONLY TO FINISH A LAP.

# need 3 things: progress/speed reward
# wall penalty
# collision penalty


# NEW VARIANT: WITH CENTERLINE, REWARDING OFF CENTERLINE
map_file = "./maps/sakhir_centerline.csv"
if os.path.exists(map_file):
    data = np.loadtxt(map_file, delimiter=",", skiprows=1)
    CENTERLINE = data[:, :2]
else:
    print(
        f"Centerline map file '{map_file}' not found. Centerline reward will be disabled."
    )
    CENTERLINE = None

_last_closest_index = (
    None  # Initialize the last closest index for centerline reward during episode
)


def calc_reward(obs, done):
    """
    Calculates a custom reward signal based on vehicle telemetry and sensor data

    parameters:
        obs(dict): Observation dictionary returned by env.step()
        done(bool): Bool. Whether the car has collided/failed.
    """

    global _last_closest_index

    # 1: CRASH PENALTY(fix)
    if done:
        _last_closest_index = None  # Reset for next episode
        return -10.0

    # fallback to simple reward if CSV missing
    if CENTERLINE is None:
        print("Centerline data not available. Using speed-based reward only.")
        speed = max(0.0, obs["linear_vels_x"][0])
        return float(
            np.clip((speed / 7.0) * 0.7, -1.0, 1.0)
        )  # Normalize speed to [-1, 1] range

    # 2: GET VEHICLE POSITIONS AND SENSORS
    car_x, car_y = obs["poses_x"][0], obs["poses_y"][0]
    speed = max(0.0, obs["linear_vels_x"][0])  # Ensure speed is non-negative
    min_wall_distance = float(
        np.min(obs["scans"])
    )  # minimum distance to wall from LiDAR scan

    # 3: CENTERLINE CALCULATIONS
    car_pos = np.array([car_x, car_y])
    distances = np.linalg.norm(CENTERLINE - car_pos, axis=1)
    closest_index = int(np.argmin(distances))
    dist_to_centerline = distances[closest_index]

    # Initialize closest index on first step after reset to avoid teleport rewards.
    if _last_closest_index is None:
        _last_closest_index = closest_index
        return 0.0

    # 4: REWARD CALCULATIONS
    # Calculate index delta handling track loop wrap-around
    idx_delta = (closest_index - _last_closest_index) % len(CENTERLINE)
    if idx_delta > len(CENTERLINE) // 2:
        idx_delta -= len(CENTERLINE)  # Wrap around for negative delta

    _last_closest_index = closest_index

    # Give positive reward for moving forward along waypoints, penalize for moving backward
    progress_reward = 0.0
    if idx_delta > 0:
        progress_reward = min(0.4, idx_delta * 0.1)
    elif idx_delta < 0:
        progress_reward = -0.5  # for driving backwards

    # 5: SPEED REWARD
    speed_reward = (speed / 7.0) * 0.3

    # 6: OFF-CENTER PENALTY Width is 1.1m each side. , penalize if car approaches (>0.5m) off-center.
    off_center_penalty = 0.0
    if dist_to_centerline > 0.4:
        off_center_penalty = -0.3 * (
            (dist_to_centerline - 0.4) / 0.6
        )  # Linear penalty up to 1.1m

    # 7: EMERGENCY WALL PENALTY
    wall_penalty = 0.0
    if min_wall_distance < 0.25:
        wall_penalty = -0.4 * (1.0 - (min_wall_distance / 0.4))

    # COMBINE ALL TERMS
    total_reward = progress_reward + speed_reward + off_center_penalty + wall_penalty
    total_reward = float(np.clip(total_reward, -2.0, 1.0))

    return total_reward
