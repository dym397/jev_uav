"""Evaluate one trained checkpoint in simple, complex and random scenes."""

import argparse
import csv
import json
from pathlib import Path

from agents.d3qn_agent import D3QNAgent
from environment.uav_env import UAVEnv
from evaluation import plot_trajectory, run_episode, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline/best.pt"))
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per scenario")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    agent = D3QNAgent.load(args.checkpoint)
    agent.set_eval_mode(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ("simple", "complex", "random"):
        results = [run_episode(UAVEnv(), agent, seed=args.seed + i, scenario_kind=kind,
                               density_per_km2=15.0 if kind == "random" else None)
                   for i in range(args.episodes)]
        metrics = summarize(results)
        rows.append({"scenario": kind, **metrics})
        plot_trajectory(results[0], args.output_dir / f"trajectory_{kind}.png", kind.title())
        print(f"{kind}: success={metrics['success_rate']:.3f}, collision={metrics['collision_rate']:.3f}, "
              f"on_time={metrics['on_time_rate']:.3f}, path={metrics['average_path_length']:.1f} m")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
