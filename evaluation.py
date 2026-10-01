"""Policy-independent episode evaluation and visualization."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment.uav_env import UAVEnv


@dataclass
class EpisodeResult:
    reward: float
    outcome: str
    on_time: bool
    arrival_error: float | None
    path_length: float
    steps: int
    trajectory: list[tuple[float, float]]
    intruder_trajectories: list[list[tuple[float, float]]]
    goal: tuple[float, float]
    obstacles: list


def run_episode(env: UAVEnv, agent, *, seed: int = 0, scenario_kind: str = "random",
                density_per_km2: float | None = None) -> EpisodeResult:
    state, _ = env.reset(seed=seed, scenario_kind=scenario_kind, density_per_km2=density_per_km2)
    total_reward = 0.0
    while True:
        action = agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return EpisodeResult(total_reward, info["outcome"], info["on_time"],
                                 info["arrival_error"], info["path_length"], env.steps,
                                 list(env.trajectory), [list(track) for track in env.intruder_trajectories],
                                 env.scenario.goal, list(env.scenario.obstacles))


def summarize(results: list[EpisodeResult]) -> dict[str, float | int | None]:
    if not results:
        raise ValueError("Need at least one episode")
    successes = [r for r in results if r.outcome == "success"]
    return {
        "episodes": len(results),
        "average_reward": float(np.mean([r.reward for r in results])),
        "success_rate": len(successes) / len(results),
        "collision_rate": sum(r.outcome == "collision" for r in results) / len(results),
        "on_time_rate": sum(r.on_time for r in results) / len(results),
        "average_path_length": float(np.mean([r.path_length for r in results])),
        "average_success_path_length": float(np.mean([r.path_length for r in successes])) if successes else None,
    }


def plot_trajectory(result: EpisodeResult, path: str | Path, title: str = "UAV trajectory") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    own = np.asarray(result.trajectory)
    ax.plot(own[:, 0], own[:, 1], color="tab:blue", linewidth=2, label="Own UAV")
    ax.scatter(own[0, 0], own[0, 1], color="tab:blue", marker="o", s=40)
    for index, track in enumerate(result.intruder_trajectories):
        points = np.asarray(track)
        if np.min(np.linalg.norm(points - own[0], axis=1)) > 180:
            continue
        ax.plot(points[:, 0], points[:, 1], color="tab:orange", alpha=0.45,
                label="Intruder" if index == 0 else None)
        ax.scatter(points[0, 0], points[0, 1], color="tab:orange", s=12, alpha=0.6)
    for index, obstacle in enumerate(result.obstacles):
        if np.linalg.norm(np.asarray(obstacle.position) - own[0]) > 180:
            continue
        ax.add_patch(plt.Circle(obstacle.position, obstacle.radius, color="gray", alpha=0.35,
                                label="Static obstacle" if index == 0 else None))
    ax.scatter(*result.goal, color="tab:green", marker="*", s=180, label="Goal")
    ax.set_xlim(min(own[:, 0].min(), result.goal[0]) - 60, max(own[:, 0].max(), result.goal[0]) + 60)
    ax.set_ylim(min(own[:, 1].min(), result.goal[1]) - 60, max(own[:, 1].max(), result.goal[1]) + 60)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title} — {result.outcome}")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rewards(rewards: list[float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(rewards) + 1), rewards, alpha=0.3, label="Episode reward")
    if rewards:
        window = min(50, len(rewards))
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(rewards) + 1), moving, color="tab:blue", label=f"{window}-episode mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
