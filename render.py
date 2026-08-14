import gym
import f110_gym
import numpy as np
import torch
import pyglet
from pyglet import gl

# Import your custom modules
from config import EnvConfig
from network import ActorCritic, scale_action


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load trained policy model
    model = ActorCritic().to(device)
    try:
        model.load_state_dict(torch.load("f1_actor_critic.pt", map_location=device))
        print("Successfully loaded f1_actor_critic.pt")
    except FileNotFoundError:
        print("Model file 'f1_actor_critic.pt' not found. Running with random weights.")
    model.eval()

    num_visual_agents = 8  # Number of parallel ghost cars

    # 2. Spin up independent single-agent gym instances
    envs = [
        gym.make("f110-v0", map="./maps/sakhir", map_ext=".png", num_agents=1)
        for _ in range(num_visual_agents)
    ]

    # Set identical starting pose for all ghosts [x, y, yaw]
    start_pose = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    obs_list = [env.reset(poses=start_pose) for env in envs]

    active_mask = [True] * num_visual_agents
    steer_angles = [
        np.zeros((1, 1), dtype=np.float32) for _ in range(num_visual_agents)
    ]

    # 3. Setup Master Renderer on Env 0
    main_env = envs[0]
    main_env.render(mode="human")  # Initializes Pyglet window

    # Color palette for ghost cars (RGB normalized 0.0 to 1.0)
    car_colors = [
        (0.9, 0.2, 0.2),  # Red
        (0.2, 0.8, 0.2),  # Green
        (0.2, 0.4, 0.9),  # Blue
        (0.9, 0.8, 0.1),  # Yellow
        (0.8, 0.2, 0.8),  # Magenta
        (0.1, 0.8, 0.8),  # Cyan
        (0.9, 0.5, 0.1),  # Orange
        (0.5, 0.5, 0.9),  # Purple
    ]

    print("Starting ghost evaluation loop...")

    while any(active_mask):
        # --- A. STEP ALL ENVIRONMENTS INDEPENDENTLY ---
        for i in range(num_visual_agents):
            if not active_mask[i]:
                continue

            # Extract observation signals
            scan = np.array(obs_list[i]["scans"], dtype=np.float32).flatten()
            vel = np.array(obs_list[i]["linear_vels_x"], dtype=np.float32).flatten()

            current_steer = np.array([steer_angles[i, 0]], dtype=np.float32)

            # Construct model state input
            state_arr = np.concatenate([scan, vel, current_steer])
            state_tensor = torch.from_numpy(state_arr).unsqueeze(0).to(device)

            # Infer action (add small noise per agent if you want trajectory variance)
            with torch.no_grad():
                raw_action, _ = model(state_tensor)
                scaled_action = scale_action(raw_action).cpu().numpy()

            steer_angles[i, 0] = float(scaled_action[0, 0])

            # Step physical dynamics in env i
            next_obs, _, done, _ = envs[i].step(scaled_action)
            obs_list[i] = next_obs

            if done:
                active_mask[i] = False
                print(f"Ghost Agent {i} crashed / reached limit.")

        # --- B. CUSTOM OVERLAY RENDER ---
        # 1. Render primary window background and track
        main_env.render(mode="human")

        # 2. Draw all active ghost cars onto main window using Pyglet
        unwrapped_env = main_env.unwrapped
        renderer = getattr(unwrapped_env, "current_renderer", None) or getattr(
            unwrapped_env, "renderer", None
        )
        if renderer is not None and hasattr(renderer, "window"):
            renderer.window.dispatch_events()

            # Access underlying Pyglet coordinate transforms
            for i in range(1, num_visual_agents):
                if active_mask[i]:
                    x = obs_list[i]["poses_x"][0]
                    y = obs_list[i]["poses_y"][0]
                    yaw = obs_list[i]["poses_theta"][0]
                    color = car_colors[i % len(car_colors)]

                    # Draw custom bounding box quad for ghost car i
                    draw_ghost_car_box(renderer, x, y, yaw, color)

    # Clean up all open environments
    for env in envs:
        env.close()


def draw_ghost_car_box(renderer, x, y, yaw, color, length=0.33, width=0.15):
    """Helper function to project additional car geometries into the active Pyglet canvas."""
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)

    # Half dimensions
    hl, hw = length / 2.0, width / 2.0

    # Local corner offsets
    corners = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])

    # Rotate & Translate to track space
    world_corners = []
    for cx, cy in corners:
        wx = x + (cx * cos_a - cy * sin_a)
        wy = y + (cx * sin_a + cy * cos_a)
        world_corners.extend([wx, wy])

    # Convert to pixel coordinates using map scale
    pix_corners = []
    for i in range(0, len(world_corners), 2):
        px, py = renderer.cars[0].to_pixel(world_corners[i], world_corners[i + 1])
        pix_corners.extend([px, py])

    # Draw colored rectangle outline/fill
    r, g, b = [int(c * 255) for c in color]
    pyglet.graphics.draw(4, gl.GL_QUADS, ("v2f", pix_corners), ("c3b", (r, g, b) * 4))


if __name__ == "__main__":
    main()
