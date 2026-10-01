import json

import numpy as np

from agents.jev_agent import NanoJevAgent, build_choice_request, parse_choice_response
from config import EnvConfig
from environment.observation import Observation
from environment.uav_env import UAVEnv
from experiment_semantic import paired_scenarios, summarize_report
from pilot_jev import pilot_scenarios, run_pilot


def make_observation(*, position=(900.0, 1000.0), due=16.7):
    return Observation(
        numeric=np.array([0.0, 6.0, 1.0, 0.0, 1.0, .4, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        own_position=position, own_speed=6.0, own_heading=0.0,
        goal=(1000.0, 1000.0), current_time=0.0, eta=16.7,
        due_time=due, nearest_distance=40.0,
    )


def test_text_is_determined_only_by_paper_vector():
    a = make_observation()
    b = make_observation(position=(20.0, 30.0), due=999.0)
    assert a.state_to_text() == b.state_to_text()
    assert "900" not in a.state_to_text()
    assert "16.7" not in a.state_to_text()
    assert "sector_0" in a.state_to_text()


def test_native_choice_request_uses_exactly_the_paper_vector():
    observation = make_observation()
    request = build_choice_request(observation)
    row = request["states"][0]
    assert row["state"] == observation.state_to_text()
    assert row["questions"]["action"]["type"] == "choice"
    assert set(row["questions"]["action"]["criteria"]) == {str(i) for i in range(9)}


def test_semantic_request_describes_only_the_same_fourteen_values():
    first = build_choice_request(make_observation(), state_style="semantic")["states"][0]
    changed_metadata = build_choice_request(
        make_observation(position=(20.0, 30.0), due=999.0),
        state_style="semantic",
    )["states"][0]
    baseline = build_choice_request(make_observation())["states"][0]
    assert first["state"] == changed_metadata["state"]
    assert first["questions"] == baseline["questions"]
    assert "900" not in first["state"]
    assert "999" not in first["state"]
    assert "sector 0" in first["state"]
    assert "40 m" in first["state"]
    assert "sector 1" in first["state"]
    assert "no intruder closer than 100 m" in first["state"]


def test_compact_semantics_augment_the_labeled_vector_without_hidden_metadata():
    a = build_choice_request(make_observation(), state_style="compact")["states"][0]
    b = build_choice_request(make_observation(position=(20.0, 30.0), due=999.0),
                             state_style="compact")["states"][0]
    labeled = build_choice_request(make_observation())["states"][0]
    assert a["state"] == b["state"]
    assert a["state"].startswith(labeled["state"])
    assert a["questions"] == labeled["questions"]
    assert "sector 0" in a["state"] and "40 m" in a["state"]
    assert "900" not in a["state"]


def test_motion_input_styles_distinguish_same_position_opposite_velocity():
    crossing = pilot_scenarios()["one_crossing"]
    moving_away = pilot_scenarios()["one_crossing"]
    moving_away.intruders[0].vy = 4.0
    first = UAVEnv(EnvConfig(), scenario=crossing).reset()[0]
    second = UAVEnv(EnvConfig(), scenario=moving_away).reset()[0]
    assert build_choice_request(first)["states"][0]["state"] == \
        build_choice_request(second)["states"][0]["state"]
    flat = build_choice_request(first, state_style="flat_contacts")["states"][0]
    structured = build_choice_request(first, state_style="structured_contacts")["states"][0]
    assert flat["questions"] == structured["questions"]
    assert flat["questions"] == build_choice_request(first)["states"][0]["questions"]
    assert flat["state"] != build_choice_request(second, state_style="flat_contacts")["states"][0]["state"]
    assert structured["state"] != build_choice_request(second, state_style="structured_contacts")["states"][0]["state"]
    assert "contact_0_forward_m=50" in flat["state"]
    assert "contact_0_relative_left_mps=-4" in flat["state"]
    assert "50 m ahead" in structured["state"]
    assert "4 m/s toward right" in structured["state"]


def test_agent_uses_selected_state_style():
    requests = []

    def evaluate(request):
        requests.append(request)
        return {"states": []}

    NanoJevAgent(evaluate, state_style="semantic").select_action(make_observation())
    assert "sector 0" in requests[0]["states"][0]["state"]


def test_semantic_experiment_reuses_fixed_and_seeded_scenes():
    scenes = paired_scenarios((41, 42))
    assert list(scenes) == ["no_intruder", "one_crossing", "two_crossing",
                            "complex_41", "complex_42"]
    assert scenes["complex_41"] == paired_scenarios((41,))["complex_41"]


def test_semantic_summary_counts_outcomes_and_invalid_decisions():
    report = {"scenarios": {
        "a": {"outcome": "success", "on_time": True, "actions": [4, 5],
              "invalid_actions": 0, "reward": 2.0, "path_length_m": 9.0,
              "latency_p50_ms": 1.0},
        "b": {"outcome": "collision", "on_time": False, "actions": [4],
              "invalid_actions": 1, "reward": -4.0, "path_length_m": 3.0,
              "latency_p50_ms": 2.0},
    }}
    summary = summarize_report(report)
    assert summary["episodes"] == 2
    assert summary["outcomes"] == {"success": 1, "collision": 1}
    assert summary["on_time"] == 1
    assert summary["decisions"] == 3
    assert summary["invalid_actions"] == 1


def test_native_choice_response_requires_full_distribution():
    probabilities = {str(i): 0.01 for i in range(9)}
    probabilities["7"] = 0.92
    response = {"states": [{"id": "uav", "answers": {"action": {
        "type": "choice", "choice": "7", "probabilities": probabilities,
    }}}]}
    assert parse_choice_response(response) == (7, 0.92)
    response["states"][0]["answers"]["action"]["probabilities"]["7"] = 0.2
    assert parse_choice_response(response) is None


def test_jev_agent_tracks_invalid_output_without_extra_state():
    requests = []

    def evaluate(request):
        requests.append(request)
        return {"states": []}

    agent = NanoJevAgent(evaluate)
    assert agent.select_action(make_observation()) == 4
    assert agent.invalid_actions == 1
    assert len(agent.latencies_ms) == 1
    assert "900" not in requests[0]["states"][0]["state"]


def test_fixed_pilot_scenes_and_agent_only_action_interface(tmp_path):
    class AlwaysMaintain:
        def select_action(self, state):
            return 4

    scenes = pilot_scenarios()
    assert list(scenes) == ["no_intruder", "one_crossing", "two_crossing"]
    assert [len(scene.intruders) for scene in scenes.values()] == [0, 1, 2]
    report = run_pilot(AlwaysMaintain(), scenes, output_dir=tmp_path, max_steps=30)
    assert set(report["scenarios"]) == set(scenes)
    assert json.loads((tmp_path / "summary.json").read_text())["scenarios"] == report["scenarios"]


def test_harness_strictly_uses_only_fourteen_numeric_values():
    from agents.jev_harness import UAVJevHarness
    from environment.observation import IntruderContact

    h1 = UAVJevHarness()
    h2 = UAVJevHarness()
    obs_a = make_observation(position=(900.0, 1000.0), due=16.7)
    obs_b = Observation(
        numeric=obs_a.numeric.copy(),
        own_position=(-500.0, 4200.0),
        own_speed=99.0,
        own_heading=2.5,
        goal=(-100.0, -200.0),
        current_time=77.0,
        eta=888.0,
        due_time=999.0,
        nearest_distance=1.2,
        contacts=(IntruderContact(10.0, 5.0, -3.0, 2.0),),
    )
    req_a = h1.build_request(h1.analyze(obs_a), top_k=3, include_diagnostics=True)
    req_b = h2.build_request(h2.analyze(obs_b), top_k=3, include_diagnostics=True)
    assert req_a == req_b
    state_text = req_a["states"][0]["state"]
    assert "900" not in state_text
    assert "999" not in state_text
    assert set(req_a["states"][0]["questions"]) == {"action", "threat_level", "safety_margin"}
    assert len(req_a["states"][0]["questions"]["action"]["criteria"]) == 3


def test_harness_agent_parses_multi_question_response_and_tracks_diagnostics():
    from agents.jev_harness import HarnessNanoJevAgent

    def fake_predictor(request):
        criteria = request["states"][0]["questions"]["action"]["criteria"]
        keys = list(criteria.keys())
        probs = {k: 0.1 for k in keys}
        probs[keys[0]] = 1.0 - 0.1 * (len(keys) - 1)
        return {
            "states": [{
                "id": "uav",
                "answers": {
                    "action": {
                        "type": "choice",
                        "choice": keys[0],
                        "probabilities": probs,
                    },
                    "threat_level": {
                        "type": "score",
                        "score": 1.25,
                        "level": 1,
                    },
                    "safety_margin": {
                        "type": "boolean",
                        "p_true": 0.88,
                        "value": True,
                    },
                },
            }]
        }

    agent = HarnessNanoJevAgent(fake_predictor, top_k=3, shielded=True)
    action = agent.select_action(make_observation())
    assert action in range(9)
    assert agent.invalid_actions == 0
    assert agent.last_diagnostics is not None
    assert agent.last_diagnostics["jev_threat_score"] == 1.25
    assert agent.last_diagnostics["jev_safety_p_true"] == 0.88

