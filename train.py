"""Train the paper-configuration D3QN baseline."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from agents.d3qn_agent import D3QNAgent
from config import AgentConfig, EnvConfig
from environment.uav_env import UAVEnv
from evaluation import plot_rewards, run_episode, summarize


def checkpoint_key(metrics: dict) -> tuple[float, float, float, float]:
    """Rank held-out policies by the actual flight task, then timing and safety."""
    return (metrics["success_rate"], metrics["on_time_rate"],
            -metrics["collision_rate"], metrics["average_reward"])


def validate(agent: D3QNAgent, env_config: EnvConfig, episodes_per_scenario: int) -> dict:
    env = UAVEnv(env_config)
    results = []
    agent.set_eval_mode(True)
    try:
        for kind in ("simple", "complex", "random"):
            for i in range(episodes_per_scenario):
                results.append(run_episode(env, agent, seed=50_000 + i,
                                           scenario_kind=kind,
                                           density_per_km2=15.0 if kind == "random" else None))
    finally:
        agent.set_eval_mode(False)
    return summarize(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--density", type=float, default=40.0, help="Intruders per km²")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=3, help="Per scenario at checkpoint selection")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_interval < 1 or args.eval_episodes < 1:
        parser.error("episodes, eval-interval and eval-episodes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    config = replace(AgentConfig(), batch_size=args.batch_size)
    env_config = EnvConfig()
    env = UAVEnv(env_config)
    agent = D3QNAgent(config=config, seed=args.seed)
    records = []
    validation_records = []
    best_key = None
    best_episode = None
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode, density_per_km2=args.density)
        total = 0.0
        losses = []
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            loss = agent.observe(state, action, reward, next_state, terminated or truncated)
            if loss is not None:
                losses.append(loss)
            total += reward
            state = next_state
            if terminated or truncated:
                break
        agent.end_episode()
        records.append({"episode": episode + 1, "reward": total, "loss": float(np.mean(losses)) if losses else "",
                        "outcome": info["outcome"], "collision": int(info["outcome"] == "collision"),
                        "success": int(info["outcome"] == "success"), "on_time": int(info["on_time"]),
                        "path_length": info["path_length"], "epsilon": agent.epsilon})
        if (episode + 1) % args.eval_interval == 0 or episode + 1 == args.episodes:
            validation = validate(agent, env_config, args.eval_episodes)
            validation_records.append({"episode": episode + 1, **validation})
            key = checkpoint_key(validation)
            if best_key is None or key > best_key:
                best_key = key
                best_episode = episode + 1
                agent.save(args.output_dir / "best.pt")
        if (episode + 1) % 50 == 0 or episode + 1 == args.episodes:
            recent = records[-50:]
            print(f"Episode {episode + 1}/{args.episodes}: reward={total:.2f}, "
                  f"success={np.mean([r['success'] for r in recent]):.3f}, "
                  f"collision={np.mean([r['collision'] for r in recent]):.3f}, epsilon={agent.epsilon:.3f}")
    agent.save(args.output_dir / "last.pt")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (args.output_dir / "validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_records[0]))
        writer.writeheader()
        writer.writerows(validation_records)
    plot_rewards([r["reward"] for r in records], args.output_dir / "reward_curve.png")
    summary = {"episodes": args.episodes, "seed": args.seed, "density_per_km2": args.density,
               "success_rate": float(np.mean([r["success"] for r in records])),
               "collision_rate": float(np.mean([r["collision"] for r in records])),
               "on_time_rate": float(np.mean([r["on_time"] for r in records])),
               "average_path_length": float(np.mean([r["path_length"] for r in records])),
               "best_validation_episode": best_episode,
               "best_validation_metrics": next(r for r in validation_records if r["episode"] == best_episode),
               "agent_config": asdict(config), "environment_config": asdict(env_config)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
