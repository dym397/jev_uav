"""Paired attribution study: Harness-only versus safety-filtered NanoJev choice."""

import argparse
import json
from collections import Counter
from pathlib import Path

from agents.jev_agent import NativeNanoJevClient
from agents.jev_harness import HarnessNanoJevAgent, HarnessOnlyAgent, SafeChoiceNanoJevAgent
from config import EnvConfig
from experiment_semantic import paired_scenarios, summarize_report
from pilot_jev import run_pilot


def run_traced_policy(agent, scenes: dict, *, output_dir: Path, max_steps: int = 30) -> tuple[dict, dict]:
    """Evaluate one policy on each fresh scenario and retain its decision trace."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scenarios": {}}
    diagnostics: dict[str, list[dict]] = {}
    for name, scenario in scenes.items():
        agent.reset()
        one = run_pilot(agent, {name: scenario}, output_dir=output_dir, max_steps=max_steps)
        report["scenarios"][name] = one["scenarios"][name]
        diagnostics[name] = list(agent.step_diagnostics)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report, diagnostics


def summarize_traces(report: dict, diagnostics: dict, *, missing_source: str) -> dict:
    """Combine episode outcomes with model authority and fallback counts."""
    rows = [step for scene in diagnostics.values() for step in scene]
    result = summarize_report(report)
    result["decision_sources"] = dict(Counter(
        step.get("decision_source", missing_source) for step in rows
    ))
    result["model_chose_final"] = sum(
        step.get("jev_raw_choice") is not None
        and step["jev_raw_choice"] == step["chosen_action"]
        for step in rows
    )
    result["model_calls"] = sum(step.get("jev_raw_choice") is not None for step in rows)
    result["offered_action_counts"] = dict(sorted(Counter(
        len(step.get("offered_actions", [])) for step in rows
    ).items()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("external/NanoJev/checkpoints/NanoJev-unified"))
    parser.add_argument("--source-dir", type=Path, default=Path("external/NanoJev"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--complex-seeds", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/harness_followup"))
    args = parser.parse_args()
    if args.max_steps <= 0 or args.complex_seeds < 0:
        parser.error("max-steps must be positive and complex-seeds nonnegative")

    seeds = list(range(100, 100 + args.complex_seeds))
    scenes = paired_scenarios(seeds)
    config = EnvConfig(max_steps=args.max_steps)
    reports = {}
    traces = {}
    missing_sources = {
        "harness_only": "utility_top",
        "safe_choice_jev": "nanojev_safe",
        "safe_top3_jev": "nanojev_safe_top3",
        "top3_jev_no_veto": "nanojev_top3",
        "original_jev_harness": "utility_veto",
    }

    reports["harness_only"], traces["harness_only"] = run_traced_policy(
        HarnessOnlyAgent(config), scenes,
        output_dir=args.output_dir / "harness_only", max_steps=args.max_steps,
    )
    print("harness_only:", summarize_traces(
        reports["harness_only"], traces["harness_only"],
        missing_source=missing_sources["harness_only"]),
          flush=True)

    client = NativeNanoJevClient(
        args.checkpoint_dir, source_dir=args.source_dir,
        device=args.device, precision=args.precision,
    )
    policies = {
        "safe_choice_jev": SafeChoiceNanoJevAgent(client, config=config),
        "safe_top3_jev": SafeChoiceNanoJevAgent(client, config=config, max_candidates=3),
        "top3_jev_no_veto": HarnessNanoJevAgent(
            client, config=config, top_k=3, shielded=False, include_diagnostics=True,
        ),
        "original_jev_harness": HarnessNanoJevAgent(
            client, config=config, top_k=3, shielded=True, include_diagnostics=True,
        ),
    }
    for name, agent in policies.items():
        reports[name], traces[name] = run_traced_policy(
            agent, scenes, output_dir=args.output_dir / name, max_steps=args.max_steps,
        )
        print(f"{name}:", summarize_traces(
            reports[name], traces[name], missing_source=missing_sources[name]), flush=True)

    checkpoint_config = json.loads((args.checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    comparison = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_schema_version": checkpoint_config.get("schema_version"),
        "max_steps": args.max_steps,
        "complex_seeds": seeds,
        "observation_boundary": "current and remembered 14-value observations only",
        "policies": {
            "harness_only": "argmax Harness utility; no NanoJev call",
            "safe_choice_jev": "offer all candidates passing existing safety thresholds; NanoJev choice is final",
            "safe_top3_jev": "offer up to three highest-utility safety-passing candidates; NanoJev choice is final",
            "top3_jev_no_veto": "original top-3 request; execute NanoJev choice without utility veto",
            "original_jev_harness": "top 3 by utility; retain the original 0.05 utility veto",
        },
        "summary": {
            name: summarize_traces(
                reports[name], traces[name], missing_source=missing_sources[name]
            ) for name in reports
        },
        "paired_scenarios": {
            scene: {
                name: {
                    "outcome": reports[name]["scenarios"][scene]["outcome"],
                    "on_time": reports[name]["scenarios"][scene]["on_time"],
                    "steps": reports[name]["scenarios"][scene]["steps"],
                    "reward": reports[name]["scenarios"][scene]["reward"],
                }
                for name in reports
            }
            for scene in scenes
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
