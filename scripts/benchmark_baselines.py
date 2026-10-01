"""Evaluate fixed maintain and frozen D3QN on the open-model development scenes."""

import argparse
import json
from collections import Counter
from pathlib import Path

from agents.d3qn_agent import D3QNAgent
from config import Action, EnvConfig
from environment.uav_env import UAVEnv
from evaluation import run_episode
from uav_eval.runner import scenario_set


class MaintainPolicy:
    def select_action(self, state):
        return Action.MAINTAIN.value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3qn-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    d3qn = D3QNAgent.load(args.d3qn_checkpoint, device="cpu")
    d3qn.set_eval_mode(True)
    scenes = scenario_set("development")
    report = {"split": "development", "max_steps": 30, "scene_names": list(scenes),
              "d3qn_checkpoint": str(args.d3qn_checkpoint), "policies": {}}
    for name, policy in (("maintain", MaintainPolicy()), ("d3qn", d3qn)):
        rows = {}
        for scene_name, scenario in scenes.items():
            result = run_episode(UAVEnv(EnvConfig(max_steps=30), scenario=scenario), policy)
            rows[scene_name] = {
                "outcome": result.outcome,
                "on_time": result.on_time,
                "steps": result.steps,
                "reward": result.reward,
            }
        report["policies"][name] = {
            "outcomes": dict(Counter(row["outcome"] for row in rows.values())),
            "on_time": sum(row["on_time"] for row in rows.values()),
            "scenarios": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: value["outcomes"] for name, value in report["policies"].items()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
