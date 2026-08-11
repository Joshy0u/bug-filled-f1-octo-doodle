import gym
import f110_gym
import numpy as np
from config import EnvConfig


def configure_agents_and_ghostmode(env):
    """
    Forces ghost mode (no car-vs-car collisions) and sets distinct agent colors.
    Must be callable anytime after env.reset() or env.render().
    """
    unwrapped = env.unwrapped

    # 1. DISABLE INTER-CAR COLLISIONS IN SIMULATOR
    if hasattr(unwrapped, "sim"):
        # Explicit global toggle
        unwrapped.sim.check_car_collisions = False

        # Direct override on individual C++ / Python racecars
        if hasattr(unwrapped.sim, "agents"):
            for agent in unwrapped.sim.agents:
                # Turn off all inter-car collision flags supported by f110_gym variants
                if hasattr(agent, "check_other_car_collisions"):
                    agent.check_other_car_collisions = False
                if hasattr(agent, "check_car_collisions"):
                    agent.check_car_collisions = False

    # 2. DEFINE DISTINCT RGB COLORS (0-255 scale)
    agent_colors = [
        [255, 50, 50],  # Agent 0: Red
        [0, 220, 255],  # Agent 1: Cyan
        [255, 215, 0],  # Agent 2: Yellow
        [220, 50, 255],  # Agent 3: Magenta
    ]

    # Apply colors to simulation agent data objects
    if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "agents"):
        for i, agent in enumerate(unwrapped.sim.agents):
            if i < len(agent_colors):
                agent.color = agent_colors[i]

    # Apply colors to Pyglet GUI Renderer window objects
    renderer = getattr(unwrapped, "renderer", None)
    if renderer is not None:
        # Check standard car shapes batch
        if hasattr(renderer, "cars"):
            for i, car in enumerate(renderer.cars):
                if i < len(agent_colors):
                    # Pyglet color update
                    if hasattr(car, "color"):
                        car.color = agent_colors[i]
                    elif hasattr(car, "vertices"):
                        # Fallback for direct shape color assignment
                        car.color = agent_colors[i]


def make_f110_env(map_path, map_ext):
    """
    Factory function to instantiate and prepare the F1Tenth environment.
    """
    env = gym.make(
        "f110-v0",
        map=map_path,
        map_ext=map_ext,
        num_agents=EnvConfig.num_agents,
    )
    return env
