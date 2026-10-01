"""Paired zero-shot test of labeled, verbose and compact semantic state text."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from agents.jev_agent import NanoJevAgent, NativeNanoJevClient, build_choice_request
from config import EnvConfig
from environment.scenario import make_scenario
from environment.uav_env import UAVEnv
from pilot_jev import pilot_scenarios, run_pilot


def paired_scenarios(complex_seeds) -> dict:
    scenes = pilot_scenarios()
    for seed in complex_seeds:
        scenes[f"complex_{seed}"] = make_scenario("complex", seed)
    return scenes


def summarize_report(report: dict) -> dict:
    rows = list(report["scenarios"].values())
    return {
        "episodes": len(rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "on_time": sum(bool(row["on_time"]) for row in rows),
        "decisions": sum(len(row["actions"]) for row in rows),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "mean_reward": fmean(row["reward"] for row in rows),
        "mean_path_length_m": fmean(row["path_length_m"] for row in rows),
        "scenario_latency_p50_range_ms": [
            min(row["latency_p50_ms"] for row in rows),
            max(row["latency_p50_ms"] for row in rows),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("external/NanoJev/checkpoints/NanoJev-unified"))
    parser.add_argument("--source-dir", type=Path, default=Path("external/NanoJev"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--complex-seeds", type=int, default=8,
                        help="Number of complex scenarios, starting at seed 100")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_ablation"))
    args = parser.parse_args()
    if args.max_steps <= 0 or args.complex_seeds < 0:
        parser.error("max-steps must be positive and complex-seeds nonnegative")

    seeds = list(range(100, 100 + args.complex_seeds))
    scenes = paired_scenarios(seeds)
    client = NativeNanoJevClient(args.checkpoint_dir, source_dir=args.source_dir,
                                device=args.device, precision=args.precision)
    reports = {}
    for style in ("labeled", "semantic", "compact"):
        agent = NanoJevAgent(client, state_style=style)
        reports[style] = run_pilot(agent, scenes, output_dir=args.output_dir / style,
                                   max_steps=args.max_steps)

    initial_state = UAVEnv(EnvConfig(max_steps=args.max_steps),
                           scenario=scenes["one_crossing"]).reset()[0]
    inputs = {}
    for style in ("labeled", "semantic", "compact"):
        request = build_choice_request(initial_state, state_style=style)
        state_text = request["states"][0]["state"]
        inputs[style] = {
            "example_state": state_text,
            "state_characters": len(state_text),
            "state_tokens": len(client.predictor.tokenizer.encode(
                state_text, add_special_tokens=False)),
        }

    comparison = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "model_revision": "unified-games-v1",
        "complex_seeds": seeds,
        "max_steps": args.max_steps,
        "controlled_variables": ["checkpoint", "question instructions", "nine actions",
                                 "fourteen numeric observation values", "initial scenarios"],
        "changed_variable": "state text serialization",
        "input_examples": inputs,
        "summary": {style: summarize_report(report) for style, report in reports.items()},
        "paired_scenarios": {
            name: {
                style: {
                    "outcome": reports[style]["scenarios"][name]["outcome"],
                    "first_action": reports[style]["scenarios"][name]["actions"][0],
                    "steps": reports[style]["scenarios"][name]["steps"],
                } for style in reports
            } for name in scenes
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(comparison, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
