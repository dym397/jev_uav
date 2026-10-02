"""Prompt-matrix styles: equal-information rewrites of the same observation."""

import re
from dataclasses import replace
from math import degrees

import pytest

from environment.observation import IntruderContact, observe
from config import EnvConfig
from environment.obstacle import Intruder
from uav_eval.prompts import STATE_STYLES, PromptSpec, build_prompt


def numbers(text):
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]


def state(observation, style):
    return build_prompt(observation, PromptSpec(style, "balanced"))["state"]


@pytest.fixture
def observation():
    # Own UAV at (900, 1000) heading +x at 6 m/s; one intruder 30 m ahead, 20 m right.
    intruders = [Intruder(x=930.0, y=980.0, vx=-4.0, vy=1.0)]
    return observe((900.0, 1000.0), 6.0, 0.0, (1000.0, 1000.0), (900.0, 1000.0),
                   100 / 6, 0.0, intruders, EnvConfig())


def test_every_style_renders(observation):
    for style in STATE_STYLES:
        assert state(observation, style)


def test_paper14_prose_carries_the_paper_values(observation):
    text = state(observation, "paper14_prose")
    values = [float(v) for v in observation.numeric]
    found = numbers(text)
    for value in values[1:3] + [values[4]] + values[5:]:
        assert any(abs(n - value) < 1e-3 for n in found), value
    assert "forward" not in text and "m/s forward" not in text  # no relative velocity added


def test_paper14_semantic_has_no_digits_and_names_the_occupied_sector(observation):
    text = state(observation, "paper14_semantic")
    assert not re.search(r"\d", text)
    sector = [i for i, v in enumerate(observation.numeric[5:]) if v < 1]
    assert sector == [8]
    assert "close ahead to the right" in text and "heading roughly east" in text


def test_paper14_rewrites_ignore_velocity(observation):
    moved = replace(observation, contacts=tuple(
        replace(c, relative_left_mps=c.relative_left_mps + 3) for c in observation.contacts))
    for style in ("paper14", "paper14_prose", "paper14_semantic"):
        assert state(observation, style) == state(moved, style)


def test_derived_numbers_match_the_semantic_facts(observation):
    text = state(observation, "first_person_derived")
    contact = observation.contacts[0]
    r = (contact.forward_m ** 2 + contact.left_m ** 2) ** 0.5
    range_rate = (contact.forward_m * contact.relative_forward_mps
                  + contact.left_m * contact.relative_left_mps) / r
    assert f"shrinking by {abs(range_rate):.3f}".rstrip("0") in text
    assert "arrive in 16.667 s" in text and "16.667 s remain" in text


def test_clock_positions():
    from uav_eval.prompts import _clock
    assert [_clock(a) for a in (0, 90, -90, 180, -30, 45)] == [12, 9, 3, 6, 1, 10]


def test_world_frame_recovers_absolute_state(observation):
    text = state(observation, "first_person_world")
    assert "Intruder 1 is at (930, 980) m with velocity (-4, 1) m/s." in text
    turned = observe((900.0, 1000.0), 6.0, 1.0, (1000.0, 1000.0), (900.0, 1000.0),
                     100 / 6, 0.0, [Intruder(x=930.0, y=980.0, vx=-4.0, vy=1.0)], EnvConfig())
    assert "Intruder 1 is at (930, 980) m with velocity (-4, 1) m/s." in state(turned, "first_person_world")


def test_list_and_polar_share_numbers(observation):
    assert numbers(state(observation, "first_person_list")) == numbers(state(observation, "first_person_polar"))
    assert "intruder 1:" in state(observation, "first_person_list")


def test_new_first_person_styles_get_first_person_instructions(observation):
    for style in ("first_person_derived", "first_person_clock", "first_person_world", "first_person_list"):
        prompt = build_prompt(observation, PromptSpec(style, "balanced"))
        assert "You are this UAV" in prompt["questions"]["action"]["instructions"]
    assert "You are this UAV" not in build_prompt(
        observation, PromptSpec("paper14_prose", "balanced"))["questions"]["action"]["instructions"]
