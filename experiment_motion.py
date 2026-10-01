"""Test whether NanoJev uses measured intruder motion and its expression."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from agents.jev_agent import (NanoJevAgent, NativeNanoJevClient,
                              build_choice_request, parse_choice_response)
from config import EnvConfig
from environment.uav_env import UAVEnv
from experiment_semantic import paired_scenarios, summarize_report
from pilot_jev import run_pilot


STYLES = ("labeled", "flat_contacts", "structured_contacts")


def motion_scenarios(seeds) -> dict:
    scenes = paired_scenarios(seeds)
    moving_away = deepcopy(scenes["one_crossing"])
    moving_away.intruders[0].vy = 4.0
    return {name: scene for name, scene in scenes.items() if name != "one_crossing"} | {
        "one_crossing": scenes["one_crossing"],
        "one_moving_away": moving_away,
    }


def first_observation(scene, max_steps: int):
    return UAVEnv(EnvConfig(max_steps=max_steps), scenario=scene).reset()[0]


def distribution(client, observation, style: str) -> dict:
    response = client(build_choice_request(observation, state_style=style))
    parsed = parse_choice_response(response)
    if parsed is None:
        raise RuntimeError(f"Invalid NanoJev response in {style} diagnostic")
    answer = response["states"][0]["answers"]["action"]
    return {"choice": parsed[0], "probabilities": answer["probabilities"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("external/NanoJev/checkpoints/NanoJev-unified"))
    parser.add_argument("--source-dir", type=Path, default=Path("external/NanoJev"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--complex-seeds", type=int, default=8,
                        help="Number of newly seeded complex scenes, starting at 200")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/motion_ablation"))
    args = parser.parse_args()
    if args.max_steps <= 0 or args.complex_seeds < 0:
        parser.error("max-steps must be positive and complex-seeds nonnegative")

    seeds = list(range(200, 200 + args.complex_seeds))
    scenes = motion_scenarios(seeds)
    approaching = first_observation(scenes["one_crossing"], args.max_steps)
    moving_away = first_observation(scenes["one_moving_away"], args.max_steps)
    if not np.array_equal(approaching.numeric, moving_away.numeric):
        raise RuntimeError("Counterfactual pair must share the exact initial 14-value observation")

    client = NativeNanoJevClient(args.checkpoint_dir, source_dir=args.source_dir,
                                device=args.device, precision=args.precision)
    diagnostics = {}
    input_examples = {}
    for style in STYLES:
        request = build_choice_request(approaching, state_style=style)
        text = request["states"][0]["state"]
        input_examples[style] = {
            "approaching_state": text,
            "state_tokens": len(client.predictor.tokenizer.encode(text, add_special_tokens=False)),
        }
        a = distribution(client, approaching, style)
        b = distribution(client, moving_away, style)
        total_variation = 0.5 * sum(
            abs(a["probabilities"][str(i)] - b["probabilities"][str(i)])
            for i in range(9)
        )
        diagnostics[style] = {
            "approaching": a,
            "moving_away": b,
            "argmax_changed": a["choice"] != b["choice"],
            "total_variation_distance": total_variation,
        }

    reports = {}
    for style in STYLES:
        agent = NanoJevAgent(client, state_style=style)
        reports[style] = run_pilot(agent, scenes, output_dir=args.output_dir / style,
                                   max_steps=args.max_steps)

    comparison = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "model_revision": "unified-games-v1",
        "complex_seeds": seeds,
        "max_steps": args.max_steps,
        "observation_assumption": "Perfect current intruder position and velocity within 100 m; no future state",
        "controlled_variables": ["checkpoint", "question instructions", "nine actions",
                                 "environment", "initial scenarios"],
        "variants": {
            "labeled": "14-value paper vector only",
            "flat_contacts": "paper vector plus detected relative position and velocity fields",
            "structured_contacts": "same facts as flat_contacts, organized per intruder",
        },
        "input_examples": input_examples,
        "counterfactual_first_step": diagnostics,
        "summary": {style: summarize_report(report) for style, report in reports.items()},
        "paired_scenarios": {
            name: {
                style: {
                    "outcome": reports[style]["scenarios"][name]["outcome"],
                    "first_action": reports[style]["scenarios"][name]["actions"][0],
                    "steps": reports[style]["scenarios"][name]["steps"],
                } for style in STYLES
            } for name in scenes
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "counterfactual_first_step": diagnostics,
        "summary": comparison["summary"],
        "output": str(args.output_dir / "comparison.json"),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
