"""Adapter for NanoJev's native, non-generative choice interface."""

import importlib.util
import math
from pathlib import Path
from time import perf_counter
from typing import Callable

from agents.jev_harness import HarnessNanoJevAgent, UAVJevHarness, parse_harness_response
from config import ACTION_SPACE, Action


STATE_STYLES = ("labeled", "semantic", "compact", "flat_contacts", "structured_contacts")


QUESTION = (
    "Choose one UAV control for the next one-second step. Keep at least 10 m "
    "from intruders and reach the goal near its scheduled time. The state has "
    "heading in radians, speed in m/s, goal distance divided by planned leg "
    "length, goal bearing counterclockwise from heading in radians, remaining "
    "ETA divided by scheduled leg time, and nine nearest-intruder sector "
    "distances divided by 100 m. Sector 0 is forward; sectors rotate "
    "counterclockwise by 40 degrees. A sector value of 1 means no intruder "
    "was detected within 100 m."
)


def semantic_state_to_text(state) -> str:
    """Explain the paper vector without reading the observation's physical metadata."""
    values = state.numeric
    if values.shape != (14,):
        raise ValueError("Semantic observation requires exactly 14 values")
    heading, speed, goal_distance, bearing, eta, *sectors = map(float, values)
    pieces = [
        f"Own UAV heading is {heading:.6g} rad; speed is {speed:.6g} m/s.",
        f"Goal distance is {goal_distance:.6g} times the planned leg length; "
        f"goal bearing is {bearing:.6g} rad counterclockwise from current heading; "
        f"remaining ETA is {eta:.6g} times the scheduled leg time.",
    ]
    for index, ratio in enumerate(sectors):
        direction = f"sector {index}, centered {index * 40} degrees counterclockwise from heading"
        if ratio >= 1.0:
            pieces.append(f"{direction}: no intruder closer than 100 m (ratio {ratio:.6g}).")
        else:
            pieces.append(f"{direction}: nearest intruder about {ratio * 100:.6g} m "
                          f"away (ratio {ratio:.6g}).")
    return " ".join(pieces)


def compact_state_to_text(state) -> str:
    """Keep the full labeled vector and add a short, deterministic interpretation."""
    values = state.numeric
    if values.shape != (14,):
        raise ValueError("Compact observation requires exactly 14 values")
    goal_distance, bearing, eta = map(float, values[2:5])
    occupied = [(index, float(ratio)) for index, ratio in enumerate(values[5:])
                if float(ratio) < 1.0]
    if occupied:
        intruders = "; ".join(
            f"sector {index} ({index * 40} degrees counterclockwise): "
            f"nearest intruder about {ratio * 100:.6g} m away"
            for index, ratio in occupied
        )
        intruders += "; all other sectors have no intruder closer than 100 m."
    else:
        intruders = "No sector has an intruder closer than 100 m."
    return (
        f"{state.state_to_text()}\n"
        f"Interpretation: {goal_distance:.6g} planned-leg lengths remain to goal; "
        f"goal bearing is {bearing:.6g} rad counterclockwise; "
        f"remaining ETA is {eta:.6g} times scheduled leg time. {intruders}"
    )


def flat_contacts_state_to_text(state) -> str:
    """Append measured relative contact vectors as flat key/value fields."""
    fields = [state.state_to_text(), f"detected_contacts={len(state.contacts)}"]
    for index, contact in enumerate(state.contacts):
        fields.extend((
            f"contact_{index}_forward_m={contact.forward_m:.6g}",
            f"contact_{index}_left_m={contact.left_m:.6g}",
            f"contact_{index}_relative_forward_mps={contact.relative_forward_mps:.6g}",
            f"contact_{index}_relative_left_mps={contact.relative_left_mps:.6g}",
        ))
    return "; ".join(fields)


def structured_contacts_state_to_text(state) -> str:
    """Express the same contact vectors as actor-centered position and motion."""
    lines = [state.state_to_text(),
             f"Detected intruders within 100 m: {len(state.contacts)}."]
    for index, contact in enumerate(state.contacts):
        forward_place = "ahead" if contact.forward_m >= 0 else "behind"
        lateral_place = "left" if contact.left_m >= 0 else "right"
        forward_motion = "front" if contact.relative_forward_mps >= 0 else "rear"
        lateral_motion = "left" if contact.relative_left_mps >= 0 else "right"
        lines.append(
            f"Contact {index}: {abs(contact.forward_m):.6g} m {forward_place}, "
            f"{abs(contact.left_m):.6g} m {lateral_place}; relative motion "
            f"{abs(contact.relative_forward_mps):.6g} m/s toward {forward_motion}, "
            f"{abs(contact.relative_left_mps):.6g} m/s toward {lateral_motion}."
        )
    return "\n".join(lines)


def build_choice_request(state, *, state_style: str = "labeled") -> dict:
    """Hold the question and choices fixed while varying observed state facts."""
    if state_style not in STATE_STYLES:
        raise ValueError(f"Unknown state style: {state_style}")
    state_text = {
        "labeled": state.state_to_text,
        "semantic": lambda: semantic_state_to_text(state),
        "compact": lambda: compact_state_to_text(state),
        "flat_contacts": lambda: flat_contacts_state_to_text(state),
        "structured_contacts": lambda: structured_contacts_state_to_text(state),
    }[state_style]()
    return {"states": [{
        "id": "uav",
        "state": state_text,
        "questions": {"action": {
            "type": "choice",
            "instructions": QUESTION,
            "criteria": {str(index): name for index, name in ACTION_SPACE.items()},
        }},
    }]}


def parse_choice_response(response: dict) -> tuple[int, float] | None:
    """Accept only a complete normalized nine-action choice distribution."""
    try:
        rows = response["states"]
        if len(rows) != 1 or rows[0]["id"] != "uav":
            return None
        answer = rows[0]["answers"]["action"]
        probabilities = answer["probabilities"]
        choice = answer["choice"]
        if answer["type"] != "choice" or type(choice) is not str:
            return None
        if set(probabilities) != {str(i) for i in ACTION_SPACE} or choice not in probabilities:
            return None
        values = [probabilities[str(i)] for i in ACTION_SPACE]
        if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            return None
        if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-5):
            return None
        if values[int(choice)] != max(values):
            return None
        return int(choice), float(probabilities[choice])
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


class NanoJevAgent:
    """Drop-in select_action(state) policy; probabilities are uncalibrated."""

    def __init__(self, evaluate: Callable[[dict], dict], *, state_style: str = "labeled"):
        if state_style not in STATE_STYLES:
            raise ValueError(f"Unknown state style: {state_style}")
        self.evaluate = evaluate
        self.state_style = state_style
        self.latencies_ms: list[float] = []
        self.invalid_actions = 0
        self.last_probability: float | None = None

    def select_action(self, state) -> int:
        start = perf_counter()
        response = self.evaluate(build_choice_request(state, state_style=self.state_style))
        self.latencies_ms.append((perf_counter() - start) * 1000)
        parsed = parse_choice_response(response)
        if parsed is None:
            self.invalid_actions += 1
            self.last_probability = None
            return Action.MAINTAIN.value
        action, self.last_probability = parsed
        return action


class NativeNanoJevClient:
    """Load TianyuCodings/NanoJev once and reuse its DecisionPredictor."""

    def __init__(self, checkpoint_dir: Path, *, source_dir: Path = Path("external/NanoJev"),
                 device: str = "cuda:0", precision: str = "bf16"):
        script = Path(source_dir).resolve() / "scripts" / "predict_toy_decisions.py"
        if not script.is_file():
            raise FileNotFoundError(f"NanoJev native predictor missing: {script}")
        spec = importlib.util.spec_from_file_location("nanojev_native_predictor", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load NanoJev predictor: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.predictor = module.DecisionPredictor(
            checkpoint_dir, device_name=device, precision=precision,
            disable_native_triton=True,
        )

    def __call__(self, request: dict) -> dict:
        return self.predictor.predict(request)
