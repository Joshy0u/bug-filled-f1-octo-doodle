import os

import numpy as np
from config import EnvConfig, rewardParams

# NOTE: IF YOU WANT THE CAR TO GO FASTER, (I.E CHANGE THE MAX SPEED), YOU ALSO NEED TO CHANGE THE WEIGHTS ACCORDINGLY. (rn its tuned to 0.7+ reward for 7 m/s)

# V1, NOT WORRIED ABOUT RACING LINE OR OPTIMAL TIME JUST YET, ONLY TO FINISH A LAP.


# NEW VARIANT: WITH CENTERLINE, REWARDING OFF CENTERLINE
if os.path.exists(EnvConfig.map_file):
    data = np.loadtxt(EnvConfig.map_file, delimiter=",", skiprows=1)
    CENTERLINE = data[:, :2]
else:
    print(
        f"Centerline map file '{EnvConfig.map_file}' not found. Centerline reward will be disabled."
    )
    CENTERLINE = None

# Track last closest index per agent: {agent_id: index}
_last_closest_indices = {}


def calc_reward(obs, done, agent_id=0):
    """
    Calculates a custom reward signal based on vehicle telemetry and sensor data

    parameters:
        obs(dict): Observation dictionary returned by env.step()
        done(bool): Bool. Whether the car has collided/failed.
    """

    global _last_closest_indices

    # Helper function to extract float safely whether input is scalar or array/list
    def get_scalar(val):
        if isinstance(val, (list, np.ndarray)):
            return float(val[0])
        return float(val)

    if isinstance(done, (list, np.ndarray)):
        is_done = bool(done[agent_id]) if len(done) > agent_id else bool(done[0])
    else:
        is_done = bool(done)

    # Speed extraction happens here.
    speed = max(0.0, get_scalar(obs["linear_vels_x"]))  # Ensure speed is non-negative

    # 1: CRASH PENALTY:
    if is_done:
        _last_closest_indices.pop(agent_id, None)  # reset only this agent's state.
        base_crash_penalty = rewardParams.crash_penalty
        kinetic_penalty = -15.0 * (speed**2)
        return float(base_crash_penalty + kinetic_penalty)

    # fallback to simple reward if CSV missing
    if CENTERLINE is None:
        print("Centerline data not available. Using speed-based reward only.")
        speed = max(0.0, get_scalar(obs["linear_vels_x"]))
        return float(
            np.clip((speed / 4.0) * 0.7, -1.0, 1.0)
        )  # Normalize speed to [-1, 1] range

    # 2: GET VEHICLE POSITIONS AND SENSORS, AND VEHICLE ORIENTATION.
    car_x = get_scalar(obs["poses_x"])
    car_y = get_scalar(obs["poses_y"])
    car_yaw = get_scalar(obs["poses_theta"])
    speed = max(0.0, get_scalar(obs["linear_vels_x"]))  # Ensure speed is non-negative

    scan_data = np.array(obs["scans"], dtype=np.float32)
    min_wall_distance = float(
        np.min(scan_data)
    )  # minimum distance to wall from LiDAR scan

    # 3: CENTERLINE CALCULATIONS
    car_pos = np.array([car_x, car_y])
    distances = np.linalg.norm(CENTERLINE - car_pos, axis=1)
    closest_index = int(np.argmin(distances))
    dist_to_centerline = distances[closest_index]

    # Initialize closest index on first step after reset to avoid teleport rewards.
    if agent_id not in _last_closest_indices or _last_closest_indices[agent_id] is None:
        _last_closest_indices[agent_id] = closest_index
        return 0.0
    last_idx = _last_closest_indices[agent_id]

    # 4: REWARD CALCULATIONS
    # Calculate index delta handling track loop wrap-around
    idx_delta = (closest_index - last_idx) % len(CENTERLINE)
    if idx_delta > len(CENTERLINE) // 2:
        idx_delta -= len(CENTERLINE)  # Wrap around for negative delta

    curr_wp = CENTERLINE[closest_index]  # wp = waypoint.
    last_wp = CENTERLINE[last_idx]
    meters_advanced = float(np.linalg.norm(curr_wp - last_wp))
    _last_closest_indices[agent_id] = closest_index

    # Give positive reward for moving forward along waypoints, penalize for moving backward
    progress_reward = 0.0
    if idx_delta > 0:
        progress_reward = (
            meters_advanced * rewardParams.progress_multiplier * (1.0 + 0.5 * speed)
        )
    elif idx_delta < 0:
        progress_reward = (
            rewardParams.backwards_multiplier * meters_advanced
        )  # for driving backwards

    # NEW TRACK ALIGNMENT (Prevents wall swerving)
    # #calculate vector next to centerline point
    next_idx = (closest_index + 1) % len(CENTERLINE)
    target_vec = CENTERLINE[next_idx] - CENTERLINE[closest_index]
    target_yaw = np.arctan2(target_vec[1], target_vec[0])

    heading_diff = np.abs(
        np.arctan2(np.sin(car_yaw - target_yaw), np.cos(car_yaw - target_yaw))
    )
    alignment_reward = rewardParams.alignment_multiplier * np.cos(
        heading_diff
    )  # +0.3 if aligned, negative if sideways.

    # 5: SPEED REWARD
    # Reward velocity aligned with track
    forward_speed = speed * np.cos(heading_diff)
    speed_reward = max(0.0, (forward_speed * rewardParams.speed_multiplier))

    # 6: OFF-CENTER PENALTY Width is 1.1m each side. , penalize if car approaches (>0.5m) off-center.
    # ALSO INCLUDES STEP PENALTY (so it doesnt look to just stay alive)
    wall_penalty = 0.0
    if min_wall_distance < 0.6:
        wall_penalty = -5.0 * ((0.6 - min_wall_distance) ** 2)

    off_center_penalty = 0.0
    if dist_to_centerline > 0.4:
        off_center_penalty = -1.0 * (dist_to_centerline - 0.4)

    # COMBINE ALL TERMS
    total_reward = (
        progress_reward
        + speed_reward
        + off_center_penalty
        + wall_penalty
        + alignment_reward
        + rewardParams.step_cost
    )
    total_reward = float(np.clip(total_reward, -50.0, 50.0))

    return total_reward
