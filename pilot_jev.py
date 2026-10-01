"""Fixed-scenario feasibility pilot for native NanoJev and D3QN."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from agents.jev_agent import NanoJevAgent, NativeNanoJevClient
from config import EnvConfig
from environment.obstacle import Intruder
from environment.scenario import Scenario
from environment.uav_env import UAVEnv
from evaluation import plot_trajectory, run_episode


def pilot_scenarios() -> dict[str, Scenario]:
    base = dict(start=(900.0, 1000.0), heading=0.0, speed=6.0,
                goal=(1000.0, 1000.0), due_time=100.0 / 6.0, world_size=2000.0)
    crossing = Intruder(950.0, 1040.0, 0.0, -4.0)
    return {
        "no_intruder": Scenario(**base),
        "one_crossing": Scenario(**base, intruders=[crossing]),
        "two_crossing": Scenario(**base, intruders=[crossing, Intruder(975.0, 950.0, 0.0, 4.0)]),
    }


def run_pilot(agent, scenes: dict[str, Scenario], *, output_dir: Path, max_steps: int = 30) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scenarios": {}}
    for name, scenario in scenes.items():
        before_invalid = getattr(agent, "invalid_actions", 0)

        class TimedPolicy:
            def __init__(self):
                self.latencies_ms = []
                self.actions = []

            def select_action(self, state):
                start = perf_counter()
                action = agent.select_action(state)
                self.latencies_ms.append((perf_counter() - start) * 1000)
                self.actions.append(int(action))
                return action

        timed = TimedPolicy()
        result = run_episode(UAVEnv(EnvConfig(max_steps=max_steps), scenario=scenario), timed)
        latencies = timed.latencies_ms
        report["scenarios"][name] = {
            "outcome": result.outcome,
            "collision": result.outcome == "collision",
            "on_time": result.on_time,
            "steps": result.steps,
            "reward": result.reward,
            "path_length_m": result.path_length,
            "invalid_actions": getattr(agent, "invalid_actions", 0) - before_invalid,
            "actions": timed.actions,
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        }
        plot_trajectory(result, output_dir / f"trajectory_{name}.png", name.replace("_", " ").title())
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("external/NanoJev/checkpoints/NanoJev-unified"))
    parser.add_argument("--source-dir", type=Path, default=Path("external/NanoJev"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--d3qn-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jev_pilot"))
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    scenes = pilot_scenarios()
    client = NativeNanoJevClient(args.checkpoint_dir, source_dir=args.source_dir,
                                device=args.device, precision=args.precision)
    jev = NanoJevAgent(client)
    report = run_pilot(jev, scenes, output_dir=args.output_dir / "nanojev", max_steps=args.max_steps)
    print(json.dumps({"checkpoint_dir": str(args.checkpoint_dir), **report}, indent=2))
    if args.d3qn_checkpoint is not None:
        from agents.d3qn_agent import D3QNAgent

        d3qn = D3QNAgent.load(args.d3qn_checkpoint)
        d3qn.set_eval_mode(True)
        baseline = run_pilot(d3qn, scenes, output_dir=args.output_dir / "d3qn", max_steps=args.max_steps)
        print(json.dumps({"d3qn_checkpoint": str(args.d3qn_checkpoint), **baseline}, indent=2))


if __name__ == "__main__":
    main()
