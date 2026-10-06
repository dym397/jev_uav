"""E5: closed-loop episodes on fresh complex-scenario seeds, paired across policies.

Every policy flies the same scenario seeds. Model policies see the prompt built from
the current observation only; baselines are D3QN (14 values), the nine constant
actions and a uniform-random policy.
"""

import json
import math
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

from config import EnvConfig
from environment.collision import point_to_segment_distance
from environment.scenario import make_scenario
from environment.uav_env import UAVEnv
from .prompts import PromptSpec, build_prompt
from .runner import _valid_probabilities, swept_separation

E5_SEEDS = range(5000, 5100)  # unused by training, development (300-319) and heldout (1000-1099)
MAX_STEPS = 30
OUTCOMES = ("success", "collision", "timeout", "out_of_bounds")


class ModelPolicy:
    """Argmax of the backend's nine-action distribution; invalid answers fall back to MAINTAIN."""

    def __init__(self, backend, state_style: str, question_style: str = "balanced", max_invalid: int | None = None):
        self.backend = backend
        self.spec = PromptSpec(state_style, question_style)
        self.invalid = 0
        self.max_invalid = max_invalid
        self.latencies_ms = []

    def select_action(self, state) -> int:
        started = perf_counter()
        try:
            probabilities = _valid_probabilities(self.backend.predict(build_prompt(state, self.spec)))
            action = max(range(9), key=lambda a: (probabilities[a], -a))
        except Exception as failure:
            self.invalid += 1
            if self.max_invalid is not None and self.invalid > self.max_invalid:
                # A paid API out of credit fails every call; stop rather than fly whole episodes on the fallback.
                raise RuntimeError(f"more than {self.max_invalid} invalid decisions; last: {failure!r}") from failure
            action = 4
        self.latencies_ms.append((perf_counter() - started) * 1000)
        return action


class ConstantPolicy:
    def __init__(self, action: int):
        self.action = action

    def select_action(self, state) -> int:
        return self.action


class RandomPolicy:
    def __init__(self, seed: int):
        self.random = random.Random(seed)

    def select_action(self, state) -> int:
        return self.random.randrange(9)


def rollout(policy, seed: int, *, max_steps: int = MAX_STEPS) -> dict:
    env = UAVEnv(EnvConfig(max_steps=max_steps), scenario=make_scenario("complex", seed))
    state, _ = env.reset()
    actions = []
    minimum = math.inf
    while True:
        action = int(policy.select_action(state))
        actions.append(action)
        own_before = env.position
        intruders_before = [intruder.position for intruder in env.scenario.intruders]
        state, _, terminated, truncated, info = env.step(action)
        for before, intruder in zip(intruders_before, env.scenario.intruders):
            minimum = min(minimum, swept_separation(own_before, env.position, before, intruder.position))
        for obstacle in env.scenario.obstacles:
            minimum = min(minimum, point_to_segment_distance(obstacle.position, own_before, env.position)
                          - obstacle.radius)
        if terminated or truncated:
            return {"seed": seed, "outcome": info["outcome"], "on_time": bool(info["on_time"]),
                    "steps": env.steps, "path_length_m": info["path_length"],
                    "minimum_separation_m": minimum if math.isfinite(minimum) else None,
                    "actions": actions}


def oracle_feasible(seed: int, *, beam: int = 64, max_steps: int = MAX_STEPS) -> dict:
    """Beam search with the simulator itself (future intruder motion known).

    Not a policy: it bounds what is achievable on each seed. A found plan proves the
    seed is solvable; "not found" only means the beam missed one.
    """
    import copy

    root = UAVEnv(EnvConfig(max_steps=max_steps), scenario=make_scenario("complex", seed))
    root.reset()
    frontier = [(root, [])]
    best = {"seed": seed, "solvable": False, "on_time": False, "plan": None}
    for _ in range(max_steps):
        children = {}
        for env, plan in frontier:
            for action in range(9):
                child = copy.deepcopy(env)
                _, _, terminated, truncated, info = child.step(action)
                if info["outcome"] == "success":
                    if not best["solvable"] or (info["on_time"] and not best["on_time"]):
                        best.update(solvable=True, on_time=bool(info["on_time"]), plan=plan + [action])
                    if info["on_time"]:
                        return best
                    continue
                if terminated or truncated:
                    continue
                key = (round(child.position[0]), round(child.position[1]),
                       round(child.speed), round(child.heading, 1))
                goal = child.scenario.goal
                score = math.hypot(goal[0] - child.position[0], goal[1] - child.position[1])
                if key not in children or score < children[key][0]:
                    children[key] = (score, child, plan + [action])
        if not children:
            break
        ranked = sorted(children.values(), key=lambda item: item[0])[:beam]
        frontier = [(env, plan) for _, env, plan in ranked]
    return best


def run_policy(name: str, policy, seeds, output_path: Path, **tags) -> None:
    """Append one JSONL record per episode; seeds already recorded for this name are skipped."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["policy"] == name:
                done.add(row["seed"])
    with output_path.open("a", encoding="utf-8") as handle:
        for seed in seeds:
            if seed in done:
                continue
            invalid_before = getattr(policy, "invalid", 0)
            latency_before = len(getattr(policy, "latencies_ms", ()))
            row = {"policy": name, **tags, **rollout(policy, seed)}
            if hasattr(policy, "invalid"):
                row["invalid_decisions"] = policy.invalid - invalid_before
                step_latencies = sorted(policy.latencies_ms[latency_before:])
                row["latency_p50_ms"] = step_latencies[len(step_latencies) // 2]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(rows: list[dict], reference: str = "d3qn") -> dict:
    by_policy = {}
    for row in rows:
        by_policy.setdefault(row["policy"], {})[row["seed"]] = row
    ref = by_policy.get(reference, {})
    summary = {}
    for name, episodes in by_policy.items():
        n = len(episodes)
        counts = Counter(row["outcome"] for row in episodes.values())
        separations = sorted(row["minimum_separation_m"] for row in episodes.values()
                             if row["minimum_separation_m"] is not None)
        entry = {
            "episodes": n,
            **{f"{outcome}_rate": counts[outcome] / n for outcome in OUTCOMES},
            "success_ci95": wilson(counts["success"], n),
            "collision_ci95": wilson(counts["collision"], n),
            "on_time_rate": sum(row["on_time"] for row in episodes.values()) / n,
            "median_minimum_separation_m": separations[len(separations) // 2] if separations else None,
            "mean_steps": sum(row["steps"] for row in episodes.values()) / n,
            "invalid_decisions": sum(row.get("invalid_decisions", 0) for row in episodes.values()),
        }
        shared = sorted(set(episodes) & set(ref))
        if ref and name != reference and shared:
            # Discordant pairs on success, for a paired (McNemar-style) comparison.
            mine = {s: episodes[s]["outcome"] == "success" for s in shared}
            theirs = {s: ref[s]["outcome"] == "success" for s in shared}
            entry["vs_reference"] = {
                "shared_seeds": len(shared),
                "only_this_succeeds": sum(mine[s] and not theirs[s] for s in shared),
                "only_reference_succeeds": sum(theirs[s] and not mine[s] for s in shared),
            }
        summary[name] = entry
    return summary


def read_rows(paths) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line)
    return rows
