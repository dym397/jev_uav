"""E1 probe labels, question construction, resumable runner and analysis."""

import json

import pytest

from uav_eval.e1 import QUESTIONS, analyze, build_question, run_e1
from uav_eval.probes import (MIRROR, counterfactual, generate, label, mirror_action, mirrored,
                             observation_for)


@pytest.fixture(scope="module")
def probes():
    return generate(7, per_family=2)


def test_mirror_action_is_an_involution_that_swaps_turn_sides():
    assert all(mirror_action(mirror_action(a)) == a for a in range(9))
    assert [mirror_action(a) for a in (0, 1, 2)] == [6, 7, 8]
    assert [mirror_action(a) for a in (3, 4, 5)] == [3, 4, 5]
    assert sorted(MIRROR.values()) == list(range(9))


def test_mirror_labels_match_mapped_base_labels(probes):
    for probe in (p for p in probes if p["variant"] == "base"):
        mirror = label(mirrored(probe["intruder"]))
        assert mirror["safe_actions"] == sorted(mirror_action(a) for a in probe["truth"]["safe_actions"])
        assert mirror["conflict_if_hold"] == probe["truth"]["conflict_if_hold"]
        if probe["truth"]["side"] is not None:
            assert mirror["side"] != probe["truth"]["side"]


def test_counterfactual_keeps_paper14_vector_but_reverses_motion(probes):
    for probe in (p for p in probes if p["variant"] == "base"):
        flipped = counterfactual(probe["intruder"])
        assert label(flipped)["paper14"] == probe["truth"]["paper14"]
        base_contact = observation_for(probe["intruder"]).contacts[0]
        cf_contact = observation_for(flipped).contacts[0]
        assert (base_contact.forward_m, base_contact.left_m) == (cf_contact.forward_m, cf_contact.left_m)
        assert base_contact.relative_left_mps == pytest.approx(-cf_contact.relative_left_mps, abs=1e-9)


def test_decision_eligible_means_a_clean_proper_split(probes):
    for truth in (p["truth"] for p in probes if p["truth"]["decision_eligible"]):
        assert truth["safe_actions"] and truth["unsafe_actions"]
        assert sorted(truth["safe_actions"] + truth["unsafe_actions"]) == list(range(9))
        assert set(truth["good_actions"]) <= set(truth["safe_actions"])


def test_prompts_never_contain_labels_or_future(probes):
    probe = next(p for p in probes if p["truth"]["decision_eligible"])
    observation = observation_for(probe["intruder"])
    for style in ("paper14", "kinematic_prose", "first_person_polar", "first_person_semantic"):
        for qid in QUESTIONS:
            state, question = build_question(observation, style, probe["id"], qid)
            text = state + json.dumps(question)
            for leak in ("safe_actions", "decision_eligible", "hold_min_separation", "committed"):
                assert leak not in text
            assert f"{probe['truth']['hold_min_separation_m']:.3f}" not in state


def test_questions_are_deterministic_and_shuffle_only_order(probes):
    observation = observation_for(probes[0]["intruder"])
    _, canonical = build_question(observation, "kinematic_fields", probes[0]["id"], "action")
    _, shuffled = build_question(observation, "kinematic_fields", probes[0]["id"], "action_shuffled")
    assert list(canonical["criteria"]) == [str(i) for i in range(9)]
    assert canonical["criteria"] == shuffled["criteria"]
    assert build_question(observation, "kinematic_fields", probes[0]["id"], "action_shuffled") == \
        build_question(observation, "kinematic_fields", probes[0]["id"], "action_shuffled")
    _, conflict = build_question(observation, "first_person_polar", probes[0]["id"], "conflict")
    assert set(conflict["criteria"]) == {"conflict", "clear"} and "me" in conflict["instructions"]


class Oracle:
    """Answers every question correctly, to validate the scoring end to end."""

    model_id = "oracle"

    def __init__(self, probes):
        self.by_state = {}
        self.probes = probes
        self.calls = 0

    def ask(self, state, question):
        self.calls += 1
        keys = list(question["criteria"])
        truth = self.current["truth"]
        if set(keys) == {"approaching", "receding"}:
            want = truth["approach"] or keys[0]
        elif set(keys) == {"left", "right"}:
            want = truth["side"] or keys[0]
        elif set(keys) == {"conflict", "clear"}:
            want = "conflict" if truth["conflict_if_hold"] else "clear"
        else:
            want = str((truth["good_actions"] or truth["safe_actions"] or [4])[0])
        return {k: (1.0 if k == want else 0.0) for k in keys}


def test_runner_resumes_and_oracle_scores_perfectly(probes, tmp_path, monkeypatch):
    import uav_eval.e1 as e1

    oracle = Oracle(probes)
    original = e1.observation_for

    def tracking(intruder):
        oracle.current = next(p for p in probes if p["intruder"] == intruder)
        return original(intruder)

    monkeypatch.setattr(e1, "observation_for", tracking)
    output = tmp_path / "records.jsonl"
    run_e1(oracle, probes[:3], ["kinematic_fields"], output)
    first_calls = oracle.calls
    run_e1(oracle, probes, ["kinematic_fields"], output)
    assert oracle.calls == len(probes) * len(QUESTIONS)
    assert first_calls == 3 * len(QUESTIONS)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == len(probes) * len(QUESTIONS)

    report = analyze(records, probes)
    row = report["styles"]["kinematic_fields"]
    assert row["invalid"] == 0
    assert row["conflict"]["accuracy"] == 1.0
    assert row["side"]["accuracy"] == 1.0
    assert row["action"]["argmax_safe_rate"] == 1.0
    assert row["action"]["argmax_unsafe_rate"] == 0.0
    assert row["order_invariance"]["argmax_agreement"] == 1.0
    eligible = [p for p in probes if p["truth"]["decision_eligible"]]
    assert report["baselines"]["constant_action"]["4"]["safe_rate"] == pytest.approx(
        sum(4 in p["truth"]["safe_actions"] for p in eligible) / len(eligible))


def test_invalid_answers_are_recorded_not_dropped(probes, tmp_path):
    class Broken:
        model_id = "broken"

        def ask(self, state, question):
            raise ValueError("bad distribution")

    output = run_e1(Broken(), probes[:1], ["paper14"], tmp_path / "r.jsonl")
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == len(QUESTIONS) and all(r["error"] and r["probabilities"] is None for r in records)
    assert analyze(records, probes[:1])["styles"]["paper14"]["invalid"] == len(QUESTIONS)
