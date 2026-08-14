import time
import torch
from gpu_env import PyTorchGPUEnv
import matplotlib.pyplot as plt


def run_benchmark():
    num_agents = 1024
    num_steps = 1000

    print(f"Initalizing GPU Environment with {num_agents} agents on CUDA...")
    env = PyTorchGPUEnv(num_agents=num_agents)

    # pre-allocate random dummy action tensor on GPU
    dummy_actions = (torch.rand(num_agents, 2, device=env.device) * 2.0) - 1.0

    traj_x, traj_y = [], []

    torch.cuda.synchronize()
    start_time = time.time()

    for step in range(num_steps):
        obs, rewards, dones, _ = env.step(dummy_actions)

        # Record Agent 0 position for trajectory trace.
        traj_x.append(env.x[0].item())
        traj_y.append(env.y[0].item())

        if step % 100 == 0:
            telemetry = env.get_telemetry()
            print(
                f"[Step {step:04d}] "
                f"Active: {telemetry['active_agents']}/{env.num_agents} | "
                f"Avg Speed: {telemetry['mean_speed']:.2f} m/s | "
                f"Max Speed: {telemetry['max_speed']:.2f} m/s | "
                f"Avg Survival Steps: {telemetry['mean_steps_alive']:.1f}"
            )

    torch.cuda.synchronize()  # Wait for GPU computation to complete

    elapsed_time = time.time() - start_time
    total_samples = num_agents * num_steps
    sps = total_samples / elapsed_time

    print(f"----------------------------------------")
    print(
        f"Completed {total_samples:,} environment steps in {elapsed_time:.2f} seconds!"
    )
    print(f"THROUGHPUT: {sps:,.2f} Steps/Second (SPS)")
    print(f"----------------------------------------")

    # ------------------------------------------------------device
    # VISUALIZATION WITH MATPLOTLIB.
    #

    print("generating visual track plot")
    centerline = env.centerline.cpu().numpy()

    plt.figure(figsize=(10, 6))

    # 1. Plot track Centerline
    plt.plot(
        centerline[:, 0],
        centerline[:, 1],
        "k--",
        label="Track Centerline",
        alpha=0.6,
    )

    all_x = env.x.cpu().numpy()
    all_y = env.y.cpu().numpy()
    plt.scatter(
        all_x,
        all_y,
        c="blue",
        s=10,
        alpha=0.5,
        label="All 1024 Agents(final)",
        zorder=3,
    )

    # 3. Plot full path trajector for Agent 0
    plt.plot(traj_x, traj_y, "r-", linewidth=1.5, label="Agent 0 path", zorder=4)

    plt.scatter(traj_x[0], traj_y[0], c="green", s=80, label="start", zorder=5)

    plt.title("F1TENTH GPU Multi-agent env visual eval")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.savefig("track_visual.png", dpi=150)
    print("saved visual to track_visual.png")


if __name__ == "__main__":
    run_benchmark()
