"""Paired evaluation of UAV-JevHarness against zero-shot NanoJev and D3QN on the 14-value state."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from agents.jev_agent import NanoJevAgent, NativeNanoJevClient, build_choice_request
from agents.jev_harness import HarnessNanoJevAgent, UAVJevHarness
from config import EnvConfig
from environment.uav_env import UAVEnv
from experiment_semantic import paired_scenarios, summarize_report
from pilot_jev import run_pilot


def run_harness_pilot(
    agent: HarnessNanoJevAgent,
    scenes: dict,
    *,
    output_dir: Path,
    max_steps: int = 30,
) -> dict:
    """Run paired scenarios while recording step-level multi-question Jev diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_by_scene: dict[str, list[dict]] = {}

    class ResettingWrapper:
        def __init__(self, inner: HarnessNanoJevAgent):
            self.inner = inner
            self.current_scene = None

        @property
        def invalid_actions(self) -> int:
            return self.inner.invalid_actions

        def select_action(self, state) -> int:
            return self.inner.select_action(state)

    wrapper = ResettingWrapper(agent)
    report = {"scenarios": {}}
    for name, scenario in scenes.items():
        agent.reset()
        single_report = run_pilot(wrapper, {name: scenario}, output_dir=output_dir, max_steps=max_steps)
        row = single_report["scenarios"][name]
        steps_log = list(agent.step_diagnostics)
        diagnostics_by_scene[name] = steps_log
        if steps_log:
            threat_scores = [d["jev_threat_score"] for d in steps_log if d["jev_threat_score"] is not None]
            safety_probs = [d["jev_safety_p_true"] for d in steps_log if d["jev_safety_p_true"] is not None]
            row["mean_jev_threat_score"] = round(fmean(threat_scores), 4) if threat_scores else None
            row["max_jev_threat_score"] = round(max(threat_scores), 4) if threat_scores else None
            row["mean_jev_safety_p_true"] = round(fmean(safety_probs), 4) if safety_probs else None
            row["min_predicted_clearance_m"] = min(d["min_clr_m"] for d in steps_log)
        report["scenarios"][name] = row

    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics_by_scene, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("external/NanoJev/checkpoints/NanoJev-unified"),
    )
    parser.add_argument("--source-dir", type=Path, default=Path("external/NanoJev"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--complex-seeds", type=int, default=8)
    parser.add_argument(
        "--d3qn-checkpoint",
        type=Path,
        default=Path("outputs/baseline_1000/model.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/harness_ablation"))
    args = parser.parse_args()
    if args.max_steps <= 0 or args.complex_seeds < 0:
        parser.error("max-steps must be positive and complex-seeds nonnegative")

    seeds = list(range(100, 100 + args.complex_seeds))
    scenes = paired_scenarios(seeds)
    client = NativeNanoJevClient(
        args.checkpoint_dir,
        source_dir=args.source_dir,
        device=args.device,
        precision=args.precision,
    )

    reports: dict[str, dict] = {}
    cached_d3qn = Path("outputs/semantic_ablation/d3qn/summary.json")
    if cached_d3qn.is_file():
        reports["d3qn"] = json.loads(cached_d3qn.read_text(encoding="utf-8"))
        out_d3qn = args.output_dir / "d3qn"
        out_d3qn.mkdir(parents=True, exist_ok=True)
        (out_d3qn / "summary.json").write_text(
            json.dumps(reports["d3qn"], indent=2, allow_nan=False), encoding="utf-8"
        )
    elif args.d3qn_checkpoint is not None and args.d3qn_checkpoint.is_file():
        from agents.d3qn_agent import D3QNAgent

        d3qn = D3QNAgent.load(args.d3qn_checkpoint)
        d3qn.set_eval_mode(True)
        reports["d3qn"] = run_pilot(
            d3qn, scenes, output_dir=args.output_dir / "d3qn", max_steps=args.max_steps
        )

    for style in ("labeled", "compact"):
        cached = Path("outputs/semantic_ablation") / style / "summary.json"
        if cached.is_file():
            reports[style] = json.loads(cached.read_text(encoding="utf-8"))
            out_style = args.output_dir / style
            out_style.mkdir(parents=True, exist_ok=True)
            (out_style / "summary.json").write_text(
                json.dumps(reports[style], indent=2, allow_nan=False), encoding="utf-8"
            )
        else:
            agent = NanoJevAgent(client, state_style=style)
            reports[style] = run_pilot(
                agent, scenes, output_dir=args.output_dir / style, max_steps=args.max_steps
            )

    unshielded_agent = HarnessNanoJevAgent(
        client,
        config=EnvConfig(max_steps=args.max_steps),
        top_k=9,
        shielded=False,
        include_diagnostics=True,
    )
    reports["harness_unshielded"] = run_harness_pilot(
        unshielded_agent,
        scenes,
        output_dir=args.output_dir / "harness_unshielded",
        max_steps=args.max_steps,
    )

    harness_agent = HarnessNanoJevAgent(
        client,
        config=EnvConfig(max_steps=args.max_steps),
        top_k=3,
        shielded=True,
        include_diagnostics=True,
    )
    reports["jev_harness"] = run_harness_pilot(
        harness_agent,
        scenes,
        output_dir=args.output_dir / "jev_harness",
        max_steps=args.max_steps,
    )

    initial_state = UAVEnv(
        EnvConfig(max_steps=args.max_steps), scenario=scenes["one_crossing"]
    ).reset()[0]
    sample_harness = UAVJevHarness(EnvConfig(max_steps=args.max_steps))
    sample_analysis = sample_harness.analyze(initial_state)
    sample_request = sample_harness.build_request(sample_analysis, top_k=3, include_diagnostics=True)
    harness_state_text = sample_request["states"][0]["state"]
    labeled_state_text = build_choice_request(initial_state, state_style="labeled")["states"][0]["state"]

    comparison = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "model_revision": "unified-games-v1",
        "complex_seeds": seeds,
        "max_steps": args.max_steps,
        "strict_input_contract": "14 numeric observation values only (state.numeric); 9 discrete actions",
        "input_examples": {
            "labeled": {
                "example_state": labeled_state_text,
                "state_characters": len(labeled_state_text),
                "state_tokens": len(
                    client.predictor.tokenizer.encode(labeled_state_text, add_special_tokens=False)
                ),
            },
            "jev_harness": {
                "example_state": harness_state_text,
                "questions_asked": list(sample_request["states"][0]["questions"].keys()),
                "offered_candidates": sample_request["states"][0]["questions"]["action"]["criteria"],
                "state_characters": len(harness_state_text),
                "state_tokens": len(
                    client.predictor.tokenizer.encode(harness_state_text, add_special_tokens=False)
                ),
            },
        },
        "summary": {mode: summarize_report(rep) for mode, rep in reports.items()},
        "paired_scenarios": {
            name: {
                mode: {
                    "outcome": reports[mode]["scenarios"][name]["outcome"],
                    "on_time": reports[mode]["scenarios"][name]["on_time"],
                    "first_action": reports[mode]["scenarios"][name]["actions"][0],
                    "steps": reports[mode]["scenarios"][name]["steps"],
                    "reward": round(reports[mode]["scenarios"][name]["reward"], 2),
                }
                for mode in reports
            }
            for name in scenes
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
