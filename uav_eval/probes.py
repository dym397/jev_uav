"""E1 single-step probes with simulator-derived labels.

The simulator is used offline to label each probe: which first actions keep a safe
separation over several steps, and whether holding course leads to a conflict.
Future positions, separations and labels never enter a model prompt; prompts are
built only from the probe's current ``Observation``.
"""

import json
import math
import random

from config import Action, EnvConfig
from environment.scenario import Scenario
from environment.obstacle import Intruder
from environment.uav_env import UAVEnv
from .runner import swept_separation


HORIZON_STEPS = 8
SAFE_MARGIN_M = 12.0          # best reachable separation >= this: safe
UNSAFE_MARGIN_M = 10.0        # < collision distance: unsafe; in between: ambiguous
CONFLICT_DISTANCE_M = 10.0
GOOD_PROGRESS_M = 5.0
APPROACH_DEADBAND_MPS = 0.2
SIDE_DEADBAND_M = 3.0
CONTINUATIONS = (Action.MAINTAIN, Action.DECELERATE, Action.TURN_RIGHT, Action.TURN_LEFT)
FAMILIES = ("head_on", "crossing_left", "crossing_right", "overtaking", "diverging", "receding")
OWN_START = (900.0, 1000.0)
OWN_SPEED = 6.0
GOAL = (1000.0, 1000.0)
DUE_TIME = 100.0 / 6.0
MIRROR = {0: 6, 1: 7, 2: 8, 3: 3, 4: 4, 5: 5, 6: 0, 7: 1, 8: 2}


def mirror_action(action: int) -> int:
    """Left/right reflection of an action index (yaw sign flips, acceleration kept)."""
    return MIRROR[int(action)]


def scenario_for(intruder: dict) -> Scenario:
    """Open airspace, heading +x, no static obstacles, far from the world bounds."""
    return Scenario(OWN_START, 0.0, OWN_SPEED, GOAL, DUE_TIME,
                    [Intruder(intruder["x"], intruder["y"], intruder["vx"], intruder["vy"])], [])


def observation_for(intruder: dict):
    state, _ = UAVEnv(EnvConfig(max_steps=HORIZON_STEPS + 1), scenario=scenario_for(intruder)).reset()
    return state


def _rollout(intruder: dict, first: int, then: int) -> tuple[float, float]:
    """(minimum swept separation, goal-distance reduction) over the horizon."""
    env = UAVEnv(EnvConfig(max_steps=HORIZON_STEPS + 1), scenario=scenario_for(intruder))
    env.reset()
    start_goal = math.dist(env.position, GOAL)
    minimum = math.inf
    for step in range(HORIZON_STEPS):
        own_before = env.position
        others = [i.position for i in env.scenario.intruders]
        _, _, terminated, truncated, _ = env.step(first if step == 0 else then)
        for before, now in zip(others, env.scenario.intruders):
            minimum = min(minimum, swept_separation(own_before, env.position, before, now.position))
        if terminated or truncated:
            break
    return minimum, start_goal - math.dist(env.position, GOAL)


def _rollout_min_separation(intruder: dict, first: int, then: int) -> float:
    return _rollout(intruder, first, then)[0]


def label(intruder: dict) -> dict:
    """Ground truth for perception, risk and action questions."""
    state = observation_for(intruder)
    if len(state.contacts) != 1:
        raise ValueError("Probe intruder must be the single detected contact")
    contact = state.contacts[0]
    rng = math.hypot(contact.forward_m, contact.left_m)
    closing = (contact.forward_m * contact.relative_forward_mps +
               contact.left_m * contact.relative_left_mps) / rng
    # Primary label: the action held for the whole horizon. "Recoverable" (any simple
    # continuation after the first step) is nearly always safe here, because braking to
    # the minimum speed on the next step rescues most first actions.
    rollouts = {a.value: _rollout(intruder, a.value, a.value) for a in Action}
    committed = {a: r[0] for a, r in rollouts.items()}
    progress = {a: r[1] for a, r in rollouts.items()}
    best = {a.value: max(_rollout_min_separation(intruder, a.value, c.value) for c in CONTINUATIONS)
            for a in Action}
    safe = sorted(a for a, d in committed.items() if d >= SAFE_MARGIN_M)
    unsafe = sorted(a for a, d in committed.items() if d < UNSAFE_MARGIN_M)
    # "Good": safe and within GOOD_PROGRESS_M of the best goal progress among safe actions,
    # so that simply stopping does not count as a good decision.
    top = max((progress[a] for a in safe), default=None)
    good = sorted(a for a in safe if progress[a] >= top - GOOD_PROGRESS_M) if safe else []
    hold = committed[Action.MAINTAIN.value]
    return {
        "approach": None if abs(closing) < APPROACH_DEADBAND_MPS else ("approaching" if closing < 0 else "receding"),
        "side": None if abs(contact.left_m) < SIDE_DEADBAND_M else ("left" if contact.left_m > 0 else "right"),
        "conflict_if_hold": hold < CONFLICT_DISTANCE_M,
        "hold_min_separation_m": hold,
        "committed_separation_m": {str(a): d for a, d in committed.items()},
        "recoverable_separation_m": {str(a): d for a, d in best.items()},
        "recoverable_safe_actions": sorted(a for a, d in best.items() if d >= SAFE_MARGIN_M),
        "goal_progress_m": {str(a): p for a, p in progress.items()},
        "safe_actions": safe,
        "good_actions": good,
        "unsafe_actions": unsafe,
        # Only a proper, unambiguous split separates good and bad action choices.
        "decision_eligible": bool(safe) and bool(unsafe) and len(safe) + len(unsafe) == 9,
        "paper14": [float(v) for v in state.numeric],
    }


def _sample_intruder(family: str, rng: random.Random) -> dict:
    speed = rng.uniform(2.0, 6.0)
    tcpa = rng.uniform(4.0, 10.0)
    miss = rng.uniform(-12.0, 12.0)
    heading = {
        "head_on": math.pi + rng.uniform(-0.25, 0.25),
        "crossing_left": -math.pi / 2 + rng.uniform(-0.6, 0.6),   # comes from the left, moves right
        "crossing_right": math.pi / 2 + rng.uniform(-0.6, 0.6),
        "overtaking": rng.uniform(-0.3, 0.3),
        "diverging": rng.uniform(-math.pi, math.pi),
        "receding": 0.0,
    }[family]
    if family == "overtaking":
        speed = rng.uniform(1.0, 3.0)   # slower traffic ahead that own UAV catches up with
    vx, vy = speed * math.cos(heading), speed * math.sin(heading)
    rvx, rvy = vx - OWN_SPEED, vy
    norm = math.hypot(rvx, rvy)
    if family == "receding":
        # Behind or abeam and moving away: range is opening, whatever own UAV does next step.
        bearing = rng.uniform(math.pi / 2, 3 * math.pi / 2)
        rng_m = rng.uniform(25.0, 80.0)
        away = bearing + rng.uniform(-0.5, 0.5)
        return {"x": OWN_START[0] + rng_m * math.cos(bearing), "y": OWN_START[1] + rng_m * math.sin(bearing),
                "vx": speed * math.cos(away), "vy": speed * math.sin(away)}
    if family == "diverging":
        # Place it off-axis; it may or may not close, and the label records which.
        bearing = rng.uniform(-math.pi, math.pi)
        rng_m = rng.uniform(25.0, 80.0)
        return {"x": OWN_START[0] + rng_m * math.cos(bearing), "y": OWN_START[1] + rng_m * math.sin(bearing),
                "vx": vx, "vy": vy}
    # Intruder position such that, holding course, relative CPA occurs at tcpa with the given miss.
    px, py = -rvy / norm * miss, rvx / norm * miss
    return {"x": OWN_START[0] + px - rvx * tcpa, "y": OWN_START[1] + py - rvy * tcpa, "vx": vx, "vy": vy}


def _valid_start(intruder: dict) -> bool:
    distance = math.hypot(intruder["x"] - OWN_START[0], intruder["y"] - OWN_START[1])
    return 20.0 <= distance <= 95.0


def counterfactual(intruder: dict) -> dict:
    """Same position, reversed velocity: identical paper-14 vector, different motion."""
    return {**intruder, "vx": -intruder["vx"], "vy": -intruder["vy"]}


def mirrored(intruder: dict) -> dict:
    """Reflect across own UAV's track (the line y = OWN_START[1])."""
    return {**intruder, "y": 2 * OWN_START[1] - intruder["y"], "vy": -intruder["vy"]}


def generate(seed: int, per_family: int = 24, eligible_share: float = 0.6, max_tries: int = 3000) -> list[dict]:
    """Base probes per family (conflict families favour decision-eligible cases), each
    with its counterfactual and mirror. Diverging traffic is mostly conflict-free and
    serves the perception questions."""
    rng = random.Random(seed)
    probes = []
    for family in FAMILIES:
        want_eligible = 0 if family in ("diverging", "receding") else round(per_family * eligible_share)
        eligible, other, tries = [], [], 0
        while len(eligible) + len(other) < per_family and tries < max_tries:
            tries += 1
            intruder = _sample_intruder(family, rng)
            if not _valid_start(intruder):
                continue
            truth = label(intruder)
            if truth["decision_eligible"] and len(eligible) < want_eligible:
                eligible.append((intruder, truth))
            elif not truth["decision_eligible"] and len(other) < per_family - want_eligible:
                other.append((intruder, truth))
            elif tries > max_tries // 2 and len(eligible) + len(other) < per_family:
                (eligible if truth["decision_eligible"] else other).append((intruder, truth))
        for index, (intruder, truth) in enumerate(eligible + other):
            base_id = f"{family}_{index:03d}"
            probes.append({"id": base_id, "base_id": base_id, "family": family, "variant": "base",
                           "intruder": intruder, "truth": truth})
            flipped = counterfactual(intruder)
            probes.append({"id": base_id + "_cf", "base_id": base_id, "family": family,
                           "variant": "counterfactual", "intruder": flipped, "truth": label(flipped)})
            mirror = mirrored(intruder)
            probes.append({"id": base_id + "_mirror", "base_id": base_id, "family": family,
                           "variant": "mirror", "intruder": mirror, "truth": label(mirror)})
    return probes


def write_probes(path, probes) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for probe in probes:
            handle.write(json.dumps(probe, ensure_ascii=False, allow_nan=False) + "\n")


def read_probes(path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
