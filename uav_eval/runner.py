"""Paired closed-loop UAV evaluation with per-decision provenance."""

import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from time import perf_counter

import numpy as np

from config import Action, EnvConfig
from environment.collision import point_to_segment_distance
from environment.scenario import make_scenario
from environment.uav_env import UAVEnv
from pilot_jev import pilot_scenarios
from .prompts import build_prompt


def scenario_set(split: str) -> dict:
    if split == "smoke":
        return pilot_scenarios()
    starts = {"development": (300, 320), "heldout": (1000, 1100)}
    if split not in starts:
        raise ValueError(f"Unknown scenario split: {split}")
    begin, end = starts[split]
    return {f"complex_{seed}": make_scenario("complex", seed) for seed in range(begin, end)}


def swept_separation(own_before, own_after, intruder_before, intruder_after) -> float:
    rx = own_before[0] - intruder_before[0]
    ry = own_before[1] - intruder_before[1]
    vx = (own_after[0] - own_before[0]) - (intruder_after[0] - intruder_before[0])
    vy = (own_after[1] - own_before[1]) - (intruder_after[1] - intruder_before[1])
    denominator = vx * vx + vy * vy
    fraction = min(max(-(rx * vx + ry * vy) / denominator, 0.0), 1.0) if denominator else 0.0
    return math.hypot(rx + fraction * vx, ry + fraction * vy)


def _valid_probabilities(raw) -> dict[int, float]:
    if not isinstance(raw, dict) or set(raw) != set(range(9)):
        raise ValueError("Backend must return all nine action probabilities")
    values = {}
    for key, value in raw.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Backend returned an invalid probability")
        values[key] = float(value)
    if not math.isclose(math.fsum(values.values()), 1.0, rel_tol=0, abs_tol=1e-3):
        raise ValueError("Backend probabilities must sum to one")
    return values


def _episode(backend, spec, scene_name, scenario, max_steps, decisions_file) -> dict:
    env = UAVEnv(EnvConfig(max_steps=max_steps), scenario=scenario)
    state, _ = env.reset()
    latencies = []
    actions = []
    invalid = 0
    total_reward = 0.0
    minimum_separation = None
    while True:
        prompt = build_prompt(state, spec)
        started = perf_counter()
        error = None
        probabilities = None
        try:
            probabilities = _valid_probabilities(backend.predict(prompt))
            action = max(range(9), key=lambda a: (probabilities[a], -a))
        except Exception as failure:
            invalid += 1
            error = f"{type(failure).__name__}: {failure}"
            action = Action.MAINTAIN.value
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        actions.append(action)
        own_before = env.position
        intruders_before = [intruder.position for intruder in env.scenario.intruders]
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        clearances = [
            swept_separation(own_before, env.position, before, intruder.position)
            for before, intruder in zip(intruders_before, env.scenario.intruders)
        ]
        clearances.extend(
            point_to_segment_distance(obstacle.position, own_before, env.position) - obstacle.radius
            for obstacle in env.scenario.obstacles
        )
        if clearances:
            step_clearance = min(clearances)
            minimum_separation = (step_clearance if minimum_separation is None else
                                  min(minimum_separation, step_clearance))
        decisions_file.write(json.dumps({
            "model_id": backend.model_id,
            "state_style": spec.state_style,
            "question_style": spec.question_style,
            "scenario": scene_name,
            "step": env.steps - 1,
            "prompt": prompt,
            "probabilities": {str(a): p for a, p in probabilities.items()} if probabilities else None,
            "action": action,
            "latency_ms": latency_ms,
            "invalid": error is not None,
            "error": error,
            "outcome_after_action": info["outcome"],
            "step_minimum_separation_m": min(clearances) if clearances else None,
        }, ensure_ascii=False, allow_nan=False) + "\n")
        if terminated or truncated:
            return {
                "outcome": info["outcome"],
                "on_time": info["on_time"],
                "arrival_error_s": info["arrival_error"],
                "reward": total_reward,
                "path_length_m": info["path_length"],
                "steps": env.steps,
                "actions": actions,
                "invalid_decisions": invalid,
                "minimum_separation_m": minimum_separation,
                "latency_p50_ms": float(np.percentile(latencies, 50)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
            }


def run_sweep(backend, specs, scenarios, output_dir, *, max_steps: int) -> dict:
    if max_steps < 1 or not specs or not scenarios:
        raise ValueError("Need positive max_steps, prompt specs, and scenarios")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "decisions.jsonl"
    summary_path = output_dir / "summary.json"
    if decisions_path.exists() or summary_path.exists():
        raise FileExistsError(f"Evaluation output already exists in {output_dir}")
    report = {
        "model_id": backend.model_id,
        "max_steps": max_steps,
        "scenario_names": list(scenarios),
        "arms": [],
    }
    with decisions_path.open("w", encoding="utf-8") as decisions_file:
        for spec in specs:
            rows = {
                name: _episode(backend, spec, name, scenario, max_steps, decisions_file)
                for name, scenario in scenarios.items()
            }
            outcomes = Counter(row["outcome"] for row in rows.values())
            report["arms"].append({
                "state_style": spec.state_style,
                "question_style": spec.question_style,
                "scenarios": rows,
                "outcomes": dict(outcomes),
                "on_time": sum(row["on_time"] for row in rows.values()),
                "decisions": sum(row["steps"] for row in rows.values()),
                "invalid_decisions": sum(row["invalid_decisions"] for row in rows.values()),
                "mean_reward": fmean(row["reward"] for row in rows.values()),
            })
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                            encoding="utf-8")
    return report
