"""Behavioral checks for the two JevHarness attribution policies."""

import json

import numpy as np

from config import EnvConfig
from environment.observation import Observation
from pilot_jev import pilot_scenarios


def observation(*, speed=6.0, sector_0=1.0, sector_1=1.0):
    values = np.array(
        [0.0, speed, 1.0, 0.0, 1.0, sector_0, sector_1, 1, 1, 1, 1, 1, 1, 1],
        dtype=np.float32,
    )
    return Observation(values, (0.0, 0.0), speed, 0.0, (100.0, 0.0),
                       0.0, 0.0, 100.0 / 6.0, None)


def choice_response(request, chosen):
    choices = request["states"][0]["questions"]["action"]["criteria"]
    probabilities = {key: float(key == str(chosen)) for key in choices}
    return {"states": [{"id": "uav", "answers": {"action": {
        "type": "choice", "choice": str(chosen), "probabilities": probabilities,
    }}}]}


def test_harness_only_executes_top_kinematic_action_without_model():
    from agents.jev_harness import HarnessOnlyAgent

    agent = HarnessOnlyAgent(EnvConfig(max_steps=30))
    assert agent.select_action(observation()) == 4
    assert agent.step_diagnostics[0]["decision_source"] == "utility_top"


def test_safe_choice_gives_jev_authority_among_all_safe_actions():
    from agents.jev_harness import SafeChoiceNanoJevAgent

    requests = []

    def evaluate(request):
        requests.append(request)
        return choice_response(request, chosen=0)

    agent = SafeChoiceNanoJevAgent(evaluate, config=EnvConfig(max_steps=30))
    # Sector 0 at 50 m leaves actions 0, 3 and 6 safe; action 3 has
    # the highest Harness utility, so choosing 0 proves there is no veto.
    assert agent.select_action(observation(sector_0=0.5)) == 0
    assert list(requests[0]["states"][0]["questions"]["action"]["criteria"]) == ["0", "3", "6"]
    assert "top_candidates=" not in requests[0]["states"][0]["state"]
    assert agent.step_diagnostics[0]["decision_source"] == "nanojev"
    assert agent.step_diagnostics[0]["safe_candidate_count"] == 3


def test_safe_choice_can_limit_to_three_safe_options_without_veto():
    from agents.jev_harness import SafeChoiceNanoJevAgent

    requests = []

    def evaluate(request):
        requests.append(request)
        return choice_response(request, chosen=1)

    agent = SafeChoiceNanoJevAgent(
        evaluate, config=EnvConfig(max_steps=30), max_candidates=3,
    )
    # In clear air all nine actions pass the safety test. The model must
    # receive only the three highest-ranked safe actions and may choose 1,
    # even though Harness utility ranks action 4 first.
    assert agent.select_action(observation()) == 1
    offered = requests[0]["states"][0]["questions"]["action"]["criteria"]
    assert list(offered) == ["1", "4", "5"]
    assert agent.step_diagnostics[0]["safe_candidate_count"] == 9
    assert agent.step_diagnostics[0]["decision_source"] == "nanojev"


def test_safe_choice_skips_model_when_no_safe_action_exists():
    from agents.jev_harness import SafeChoiceNanoJevAgent

    def unexpected_call(_request):
        raise AssertionError("NanoJev must not be asked to choose an unsafe action")

    agent = SafeChoiceNanoJevAgent(unexpected_call, config=EnvConfig(max_steps=30))
    assert agent.select_action(observation(sector_0=0.2)) == 7
    assert agent.step_diagnostics[0]["decision_source"] == "no_safe_fallback"
    assert agent.step_diagnostics[0]["safe_candidate_count"] == 0


def test_safe_choice_skips_model_when_exactly_one_safe_action_exists():
    from agents.jev_harness import SafeChoiceNanoJevAgent

    def unexpected_call(_request):
        raise AssertionError("A sole safe action needs no model choice")

    agent = SafeChoiceNanoJevAgent(unexpected_call, config=EnvConfig(max_steps=30))
    assert agent.select_action(observation(speed=10.0, sector_1=0.5)) == 0
    assert agent.step_diagnostics[0]["decision_source"] == "sole_safe"
    assert agent.step_diagnostics[0]["safe_candidate_count"] == 1


def test_followup_runner_saves_paired_episode_and_step_records(tmp_path):
    from agents.jev_harness import HarnessOnlyAgent
    from experiment_harness_followup import run_traced_policy

    agent = HarnessOnlyAgent(EnvConfig(max_steps=30))
    scenes = {"no_intruder": pilot_scenarios()["no_intruder"]}
    report, diagnostics = run_traced_policy(agent, scenes, output_dir=tmp_path, max_steps=30)
    assert report["scenarios"]["no_intruder"]["outcome"] == "success"
    assert len(diagnostics["no_intruder"]) == report["scenarios"]["no_intruder"]["steps"]
    assert json.loads((tmp_path / "summary.json").read_text())["scenarios"] == report["scenarios"]
    assert json.loads((tmp_path / "diagnostics.json").read_text()) == diagnostics


def test_trace_summary_labels_unshielded_reference_without_claiming_a_shield():
    from experiment_harness_followup import summarize_traces

    report = {"scenarios": {"scene": {
        "outcome": "success", "on_time": True, "actions": [4],
        "invalid_actions": 0, "reward": 2.0, "path_length_m": 6.0,
        "latency_p50_ms": 1.0,
    }}}
    diagnostics = {"scene": [{
        "chosen_action": 4, "jev_raw_choice": 4, "offered_actions": [4, 5, 1],
    }]}
    summary = summarize_traces(report, diagnostics, missing_source="nanojev_top3")
    assert summary["decision_sources"] == {"nanojev_top3": 1}
