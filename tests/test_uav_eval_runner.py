"""Closed-loop behavior and provenance for the cross-model runner."""

import json

from environment.obstacle import Intruder
from environment.scenario import Scenario
from uav_eval.prompts import PromptSpec
from uav_eval.runner import run_sweep, scenario_set, swept_separation


class MaintainBackend:
    model_id = "fixed-maintain"

    def predict(self, payload):
        return {i: (1.0 if i == 4 else 0.0) for i in range(9)}


def test_split_seeds_are_fixed_and_disjoint():
    assert list(scenario_set("smoke")) == ["no_intruder", "one_crossing", "two_crossing"]
    assert list(scenario_set("development"))[0] == "complex_300"
    assert list(scenario_set("development"))[-1] == "complex_319"
    assert list(scenario_set("heldout"))[0] == "complex_1000"
    assert list(scenario_set("heldout"))[-1] == "complex_1099"


def test_swept_separation_detects_between_frame_near_miss():
    assert swept_separation((0, 0), (10, 0), (5, 5), (5, -5)) == 0.0
    assert swept_separation((0, 0), (10, 0), (5, 5), (5, 5)) == 5.0


def test_runner_pairs_scenarios_and_logs_every_model_input(tmp_path):
    specs = [PromptSpec("paper14", "balanced"),
             PromptSpec("kinematic_fields", "balanced")]
    scenarios = scenario_set("smoke")
    result = run_sweep(MaintainBackend(), specs, scenarios, tmp_path, max_steps=2)
    assert result["model_id"] == "fixed-maintain"
    assert result["max_steps"] == 2
    assert len(result["arms"]) == 2
    assert [set(arm["scenarios"]) for arm in result["arms"]] == [set(scenarios), set(scenarios)]
    rows = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    assert all(row["model_id"] == "fixed-maintain" and row["action"] == 4 for row in rows)
    assert rows[0]["prompt"]["questions"]["action"]["criteria"]["4"].startswith("Maintain")
    assert rows[0]["prompt"]["state"] != rows[6]["prompt"]["state"]
    assert json.loads((tmp_path / "summary.json").read_text()) == result


def test_runner_records_invalid_response_and_fixed_fallback(tmp_path):
    class BrokenBackend:
        model_id = "broken"

        def predict(self, payload):
            raise ValueError("bad distribution")

    scenario = {"empty": scenario_set("smoke")["no_intruder"]}
    report = run_sweep(BrokenBackend(), [PromptSpec("kinematic_fields", "balanced")],
                       scenario, tmp_path, max_steps=1)
    row = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert row["action"] == 4
    assert row["invalid"] is True
    assert "bad distribution" in row["error"]
    assert report["arms"][0]["invalid_decisions"] == 1


def test_runner_counts_collision_and_timeout_separately(tmp_path):
    base = dict(start=(900.0, 1000.0), heading=0.0, speed=6.0,
                goal=(1000.0, 1000.0), due_time=100.0 / 6.0)
    scenarios = {"clear": Scenario(**base),
                 "hit": Scenario(**base, intruders=[Intruder(903, 1000, 0, 0)])}
    report = run_sweep(MaintainBackend(), [PromptSpec("kinematic_fields", "balanced")],
                       scenarios, tmp_path, max_steps=1)
    arm = report["arms"][0]
    assert arm["outcomes"] == {"timeout": 1, "collision": 1}
    assert arm["scenarios"]["clear"]["minimum_separation_m"] is None
    assert arm["scenarios"]["hit"]["minimum_separation_m"] == 0.0
