"""Perspective prompt styles and in-process decision backends."""

import re
from dataclasses import replace

import pytest

from environment.observation import IntruderContact
from environment.scenario import make_scenario
from environment.uav_env import UAVEnv
from uav_eval.backends import JevK5Backend, LayaBackend
from uav_eval.prompts import PromptSpec, build_prompt


def observation_with(contacts):
    state, _ = UAVEnv(scenario=make_scenario("complex", 300)).reset()
    return replace(state, contacts=tuple(contacts))


CONTACTS = [
    IntruderContact(forward_m=30.0, left_m=-17.320508, relative_forward_mps=-8.0, relative_left_mps=0.0),
    IntruderContact(forward_m=0.0, left_m=40.0, relative_forward_mps=1.5, relative_left_mps=-2.0),
]


def numbers(text):
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def test_first_and_third_person_carry_identical_facts():
    observation = observation_with(CONTACTS)
    first = build_prompt(observation, PromptSpec("first_person_polar", "balanced"))
    third = build_prompt(observation, PromptSpec("third_person_polar", "balanced"))
    assert numbers(first["state"]) == numbers(third["state"])
    assert first["questions"]["action"]["criteria"] == third["questions"]["action"]["criteria"]
    assert "I am flying" in first["state"] and "You are this UAV" in first["questions"]["action"]["instructions"]
    assert "The UAV is flying" in third["state"] and " I " not in third["state"]


def test_polar_bearing_and_motion_words_follow_body_axes():
    state = build_prompt(observation_with(CONTACTS), PromptSpec("first_person_polar", "balanced"))["state"]
    assert "Intruder 1 is 34.641 m from me, 30 deg to the right of the nose" in state
    assert "8 m/s backward" in state
    assert "Intruder 2 is 40 m from me, 90 deg to the left of the nose" in state
    assert "2 m/s to the right" in state


def test_polar_reports_empty_airspace():
    state = build_prompt(observation_with([]), PromptSpec("third_person_polar", "safety_first"))["state"]
    assert "No intruder is detected within 100 m of the UAV." in state


def test_semantic_styles_use_words_only_and_match_across_perspectives():
    observation = observation_with(CONTACTS)
    first = build_prompt(observation, PromptSpec("first_person_semantic", "balanced"))
    third = build_prompt(observation, PromptSpec("third_person_semantic", "balanced"))
    assert numbers(first["state"]) == numbers(third["state"]) == []
    assert "You are this UAV" in first["questions"]["action"]["instructions"]
    neutral = (first["state"].replace("I am", "The UAV is").replace("would arrive", "")
               .replace("from me", "from the UAV").replace("My goal", "Its goal").replace(" I ", " "))
    assert neutral == third["state"].replace("would arrive", "").replace(" it ", " ")


def test_semantic_words_follow_current_geometry():
    state = build_prompt(observation_with(CONTACTS), PromptSpec("first_person_semantic", "balanced"))["state"]
    assert ("Intruder A is close, ahead and slightly to the right. It is closing in fast, and seen from me "
            "its direction is quickly drifting to the right.") in state
    assert ("Intruder B is close, off to the left. It is closing in slowly, and seen from me "
            "its direction is slowly drifting to the right.") in state


def test_semantic_reports_empty_airspace():
    state = build_prompt(observation_with([]), PromptSpec("third_person_semantic", "balanced"))["state"]
    assert "No intruder is detected near the UAV." in state


def distribution(winner):
    probs = {str(i): 0.01 for i in range(9)}
    probs[str(winner)] = 0.92
    return probs


class FakeJevK5:
    def __init__(self):
        self.calls = []

    def decide(self, state, question):
        self.calls.append((state, question))
        return {"type": "choice", "choice": "7", "probabilities": distribution(7)}


class FakeLaya:
    def predict(self, state, questions):
        assert set(questions) == {"action"}
        return {"answers": {"action": {"type": "choice", "choice": "2", "probabilities": distribution(2)}}}


def test_in_process_backends_normalize_to_nine_actions():
    payload = build_prompt(observation_with(CONTACTS), PromptSpec("first_person_polar", "balanced"))
    fake = FakeJevK5()
    jevk5 = JevK5Backend(fake, model_id="JevK5-9B").predict(payload)
    assert max(jevk5, key=jevk5.get) == 7
    assert fake.calls == [(payload["state"], payload["questions"]["action"])]
    laya_probs = LayaBackend(FakeLaya(), model_id="laya").predict(payload)
    assert max(laya_probs, key=laya_probs.get) == 2


def test_laya_truncation_is_an_error():
    class Truncating(FakeLaya):
        def predict(self, state, questions):
            return {**super().predict(state, questions), "usage": {"truncated": True}}

    payload = build_prompt(observation_with(CONTACTS), PromptSpec("paper14", "balanced"))
    with pytest.raises(ValueError, match="truncated"):
        LayaBackend(Truncating(), model_id="laya").predict(payload)


def test_backend_rejects_partial_distribution():
    class Broken(FakeJevK5):
        def decide(self, state, question):
            return {"type": "choice", "choice": "0", "probabilities": {"0": 1.0}}

    payload = build_prompt(observation_with(CONTACTS), PromptSpec("paper14", "balanced"))
    with pytest.raises(ValueError):
        JevK5Backend(Broken(), model_id="x").predict(payload)
