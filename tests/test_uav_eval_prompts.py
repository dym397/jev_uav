"""Behavioral checks for decision-model UAV prompts."""

from dataclasses import replace

from config import ACTION_SPACE
from environment.observation import IntruderContact
from environment.uav_env import UAVEnv
from pilot_jev import pilot_scenarios
from uav_eval.prompts import PromptSpec, build_prompt


def crossing_observation():
    return UAVEnv(scenario=pilot_scenarios()["one_crossing"]).reset()[0]


def test_rich_styles_expose_current_mission_and_contact_motion():
    obs = crossing_observation()
    fields = build_prompt(obs, PromptSpec("kinematic_fields", "balanced"))["state"]
    prose = build_prompt(obs, PromptSpec("kinematic_prose", "balanced"))["state"]
    for expected in ("900", "1000", "16.667", "50", "40", "-6", "-4"):
        assert expected in fields
        assert expected in prose
    assert "contact_0" in fields
    assert "Contact 0" in prose
    assert "future" not in fields.lower()


def test_paper14_remains_independent_of_rich_metadata():
    obs = crossing_observation()
    different = replace(obs, own_position=(-50.0, 80.0), due_time=999.0,
                        contacts=(IntruderContact(1, 2, 3, 4),))
    paper = PromptSpec("paper14", "balanced")
    rich = PromptSpec("kinematic_fields", "balanced")
    assert build_prompt(obs, paper)["state"] == build_prompt(different, paper)["state"]
    assert build_prompt(obs, rich)["state"] != build_prompt(different, rich)["state"]


def test_rich_styles_distinguish_approaching_and_receding_contacts():
    obs = crossing_observation()
    contact = obs.contacts[0]
    opposite = replace(obs, contacts=(replace(contact, relative_left_mps=4.0),))
    for style in ("kinematic_fields", "kinematic_prose", "cpa_brief"):
        spec = PromptSpec(style, "balanced")
        assert build_prompt(obs, spec)["state"] != build_prompt(opposite, spec)["state"]


def test_cpa_uses_clamped_constant_velocity_geometry():
    obs = crossing_observation()
    text = build_prompt(obs, PromptSpec("cpa_brief", "balanced"))["state"]
    assert "cpa_horizon_s=10" in text
    assert "contact_0_tcpa_s=8.846" in text
    assert "contact_0_dcpa_m=5.547" in text
    stationary = replace(obs, contacts=(IntruderContact(30, 40, 0, 0),))
    text = build_prompt(stationary, PromptSpec("cpa_brief", "balanced"))["state"]
    assert "contact_0_tcpa_s=0" in text
    assert "contact_0_dcpa_m=50" in text
    receding = replace(obs, contacts=(IntruderContact(50, 0, 2, 0),))
    text = build_prompt(receding, PromptSpec("cpa_brief", "balanced"))["state"]
    assert "contact_0_tcpa_s=0" in text
    assert "contact_0_dcpa_m=50" in text


def test_no_contact_is_explicit_and_candidates_match_actual_controls():
    obs = UAVEnv(scenario=pilot_scenarios()["no_intruder"]).reset()[0]
    prompt = build_prompt(obs, PromptSpec("kinematic_fields", "safety_first"))
    assert "contacts_within_100m=0" in prompt["state"]
    question = prompt["questions"]["action"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {str(i) for i in ACTION_SPACE}
    assert "yaw -6 deg/s" in question["criteria"]["0"]
    assert "acceleration -3 m/s^2" in question["criteria"]["0"]
    assert "yaw +6 deg/s" in question["criteria"]["8"]
    assert "acceleration +3 m/s^2" in question["criteria"]["8"]
    assert "current heading and speed" in question["instructions"]
    assert "Safety first" in question["instructions"]
