"""E5 closed-loop runner: seeds, resume, model fallback, paired summary."""

import json

from config import Action
from run_e5 import main as run_e5_main
from uav_eval.e5 import (E5_SEEDS, ConstantPolicy, ModelPolicy, RandomPolicy, read_rows, rollout,
                         run_policy, summarize, wilson)
from uav_eval.runner import scenario_set


def test_e5_seeds_are_fresh():
    used = {int(name.split("_")[1]) for split in ("development", "heldout") for name in scenario_set(split)}
    assert used.isdisjoint(E5_SEEDS) and len(E5_SEEDS) == 100


def test_rollout_is_deterministic_and_complete():
    first = rollout(ConstantPolicy(Action.MAINTAIN.value), 5000)
    assert first == rollout(ConstantPolicy(Action.MAINTAIN.value), 5000)
    assert first["outcome"] in ("success", "collision", "timeout", "out_of_bounds")
    assert len(first["actions"]) == first["steps"] and set(first["actions"]) == {4}
    assert rollout(RandomPolicy(1), 5001) == rollout(RandomPolicy(1), 5001)


class FakeBackend:
    model_id = "fake"

    def __init__(self, broken_every=0):
        self.calls = 0
        self.broken_every = broken_every

    def predict(self, prompt):
        self.calls += 1
        assert "Intruder" in prompt["state"] or "No intruder" in prompt["state"]
        if self.broken_every and self.calls % self.broken_every == 0:
            return {0: 1.0}
        return {a: (0.92 if a == 3 else 0.01) for a in range(9)}


def test_model_policy_uses_argmax_and_counts_invalid_answers():
    policy = ModelPolicy(FakeBackend(broken_every=2), "third_person_semantic")
    row = rollout(policy, 5002)
    assert policy.invalid == row["steps"] // 2
    assert all(a == 3 for a in row["actions"][::2]) and all(a == 4 for a in row["actions"][1::2])


def test_run_policy_resumes_and_summary_pairs_with_reference(tmp_path):
    out = tmp_path / "episodes.jsonl"
    run_policy("d3qn", ConstantPolicy(4), range(5000, 5003), out)
    run_policy("fake:x", ModelPolicy(FakeBackend(), "first_person_semantic"), range(5000, 5002), out,
               model_id="fake", state_style="first_person_semantic")
    run_policy("fake:x", ModelPolicy(FakeBackend(), "first_person_semantic"), range(5000, 5003), out)
    rows = read_rows([out])
    assert sorted(r["seed"] for r in rows if r["policy"] == "fake:x") == [5000, 5001, 5002]
    summary = summarize(rows)
    vs = summary["fake:x"]["vs_reference"]
    assert vs["shared_seeds"] == 3
    successes = {p: {r["seed"] for r in rows if r["policy"] == p and r["outcome"] == "success"}
                 for p in ("d3qn", "fake:x")}
    assert vs["only_this_succeeds"] == len(successes["fake:x"] - successes["d3qn"])
    assert vs["only_reference_succeeds"] == len(successes["d3qn"] - successes["fake:x"])


def test_wilson_interval_bounds():
    low, high = wilson(5, 10)
    assert 0.2 < low < 0.5 < high < 0.8 and wilson(0, 10)[0] == 0.0


def test_analyze_writes_table(tmp_path):
    out = tmp_path / "episodes.jsonl"
    run_policy("d3qn", ConstantPolicy(3), range(5000, 5002), out)
    run_policy("random_r0", RandomPolicy(0), range(5000, 5002), out)
    run_e5_main(["analyze", str(out), "--output", str(tmp_path / "table.md")])
    table = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert "| d3qn | 2 |" in table and "random (mean of 1 repeats)" in table
    assert json.loads((tmp_path / "table.json").read_text())["d3qn"]["episodes"] == 2


def test_oracle_plan_replays_to_success():
    from uav_eval.e5 import oracle_feasible

    found = oracle_feasible(5000)
    assert found["solvable"]  # seed 5000 has a known collision-free plan
    plan = list(found["plan"])

    class Replay:
        def select_action(self, state):
            return plan.pop(0)

    assert rollout(Replay(), 5000)["outcome"] == "success"
