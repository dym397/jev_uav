"""E1: layered single-step probes (perception -> risk -> action) and their analysis.

Every question is asked in its own call. Perception and risk questions use neutral
two-option Choice (not Noul), with a per-probe seeded option order. The action question
is asked twice: canonical option order and a seeded shuffle.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import fmean
from time import perf_counter

from .probes import mirror_action, observation_for
from .prompts import FIRST_PERSON_STYLES, PromptSpec, build_prompt


QUESTIONS = ("approach", "side", "conflict", "action", "action_shuffled")
# cpa_brief states code-computed TCPA/DCPA: its risk answers are "assisted", not recognition.
ASSISTED_STYLES = ("cpa_brief",)
# paper14 has no relative velocity: approach and conflict are not identifiable from it.
MOTION_BLIND_STYLES = ("paper14",)


def _seeded(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def _binary(style: str, probe_id: str, qid: str) -> dict:
    first = style in FIRST_PERSON_STYLES
    me, my = ("me", "my") if first else ("the UAV", "the UAV's")
    keep = "I keep my" if first else "the UAV keeps its"
    text, criteria = {
        "approach": (f"Right now, is the detected intruder getting closer to {me} or farther from {me}?",
                     {"approaching": "The distance to the intruder is decreasing.",
                      "receding": "The distance to the intruder is increasing."}),
        "side": (f"On which side of {my} nose is the detected intruder?",
                 {"left": "The intruder is on the left side.",
                  "right": "The intruder is on the right side."}),
        "conflict": (f"If {keep} current heading and speed for the next 8 seconds, will the intruder "
                     f"come within 10 m of {me}?",
                     {"conflict": "Yes, the separation will drop below 10 m.",
                      "clear": "No, the separation will stay at least 10 m."}),
    }[qid]
    keys = list(criteria)
    _seeded(probe_id, qid).shuffle(keys)
    return {"type": "choice", "instructions": text, "criteria": {k: criteria[k] for k in keys}}


def build_question(observation, style: str, probe_id: str, qid: str) -> tuple[str, dict]:
    """(state text, question) for one call. The state text is the same for every qid."""
    payload = build_prompt(observation, PromptSpec(style, "balanced"))
    if qid in ("approach", "side", "conflict"):
        return payload["state"], _binary(style, probe_id, qid)
    question = payload["questions"]["action"]
    if qid == "action_shuffled":
        keys = list(question["criteria"])
        _seeded(probe_id, "action").shuffle(keys)
        question = {**question, "criteria": {k: question["criteria"][k] for k in keys}}
    return payload["state"], question


def run_e1(backend, probes, styles, output_path, *, questions=QUESTIONS) -> Path:
    """Append one JSON line per (style, probe, question); resumes an interrupted run."""
    output_path = Path(output_path)
    done = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            done = {(r["style"], r["probe_id"], r["question"]) for r in map(json.loads, handle)}
    with output_path.open("a", encoding="utf-8") as handle:
        for probe in probes:
            observation = observation_for(probe["intruder"])
            for style in styles:
                for qid in questions:
                    if (style, probe["id"], qid) in done:
                        continue
                    state, question = build_question(observation, style, probe["id"], qid)
                    started = perf_counter()
                    probabilities, error = None, None
                    try:
                        probabilities = backend.ask(state, question)
                    except Exception as failure:  # recorded as an invalid answer
                        error = f"{type(failure).__name__}: {failure}"
                    handle.write(json.dumps({
                        "model_id": backend.model_id, "style": style, "probe_id": probe["id"],
                        "question": qid, "option_order": list(question["criteria"]),
                        "probabilities": probabilities, "error": error,
                        "latency_ms": (perf_counter() - started) * 1000,
                    }, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
    return output_path


def _argmax(probabilities: dict, order: list) -> str:
    # Ties go to the option offered first, as a reader of the option list would break them.
    return max(order, key=lambda k: (probabilities[k], -order.index(k)))


def _tvd(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a[k] - b[k]) for k in a)


def _rate(values) -> float | None:
    values = list(values)
    return fmean(values) if values else None


def analyze(records, probes) -> dict:
    """Per-style metrics plus model-independent baselines computed from the labels."""
    by_probe = {p["id"]: p for p in probes}
    answers = {(r["style"], r["probe_id"], r["question"]): r for r in records}
    styles = sorted({r["style"] for r in records})
    eligible = [p for p in probes if p["truth"]["decision_eligible"]]
    report = {"n_probes": len(probes), "n_decision_eligible": len(eligible), "baselines": {
        "random_safe_rate": _rate(len(p["truth"]["safe_actions"]) / 9 for p in eligible),
        "random_good_rate": _rate(len(p["truth"]["good_actions"]) / 9 for p in eligible),
        "constant_action": {str(a): {
            "safe_rate": _rate(a in p["truth"]["safe_actions"] for p in eligible),
            "good_rate": _rate(a in p["truth"]["good_actions"] for p in eligible),
        } for a in range(9)},
    }, "styles": {}}
    for q, key in (("approach", "approach"), ("side", "side"), ("conflict", "conflict_if_hold")):
        labels = [p["truth"][key] for p in probes if p["truth"][key] is not None]
        report["baselines"][f"{q}_majority_rate"] = max(Counter(labels).values()) / len(labels)

    for style in styles:
        row = {"invalid": sum(1 for r in records if r["style"] == style and r["error"]),
               "calls": sum(1 for r in records if r["style"] == style),
               "latency_p50_ms": _median([r["latency_ms"] for r in records if r["style"] == style]),
               "assisted_risk": style in ASSISTED_STYLES,
               "motion_blind": style in MOTION_BLIND_STYLES}
        for q, key, truth_of in (("approach", "approach", lambda v: v),
                                 ("side", "side", lambda v: v),
                                 ("conflict", "conflict_if_hold", lambda v: "conflict" if v else "clear")):
            pairs = []
            for p in probes:
                r = answers.get((style, p["id"], q))
                if r is None or r["probabilities"] is None or p["truth"][key] is None:
                    continue
                pairs.append((truth_of(p["truth"][key]), r))
            picks = [_argmax(r["probabilities"], r["option_order"]) for _, r in pairs]
            classes = sorted({t for t, _ in pairs})
            row[q] = {
                "n": len(pairs),
                "accuracy": _rate(pick == t for pick, (t, _) in zip(picks, pairs)),
                "balanced_accuracy": _rate(_rate(pick == t for pick, (t, _) in zip(picks, pairs) if t == c)
                                           for c in classes),
                "mean_p_truth": _rate(r["probabilities"][t] for t, r in pairs),
                "modal_answer_share": max(Counter(picks).values()) / len(picks) if picks else None,
                "picked_first_listed_rate": _rate(pick == r["option_order"][0] for pick, (_, r) in zip(picks, pairs)),
            }
        acts = [(p, answers.get((style, p["id"], "action"))) for p in eligible]
        acts = [(p, r) for p, r in acts if r and r["probabilities"]]
        picks = [int(_argmax(r["probabilities"], r["option_order"])) for _, r in acts]
        row["action"] = {
            "n": len(acts),
            "safe_mass": _rate(sum(r["probabilities"][str(a)] for a in p["truth"]["safe_actions"]) for p, r in acts),
            "good_mass": _rate(sum(r["probabilities"][str(a)] for a in p["truth"]["good_actions"]) for p, r in acts),
            "argmax_safe_rate": _rate(a in p["truth"]["safe_actions"] for a, (p, _) in zip(picks, acts)),
            "argmax_good_rate": _rate(a in p["truth"]["good_actions"] for a, (p, _) in zip(picks, acts)),
            "argmax_unsafe_rate": _rate(a in p["truth"]["unsafe_actions"] for a, (p, _) in zip(picks, acts)),
            "modal_action": Counter(picks).most_common(1)[0] if picks else None,
            "modal_action_share": max(Counter(picks).values()) / len(picks) if picks else None,
        }
        order_pairs = []
        for p in probes:
            a, b = answers.get((style, p["id"], "action")), answers.get((style, p["id"], "action_shuffled"))
            if a and b and a["probabilities"] and b["probabilities"]:
                order_pairs.append((a, b))
        row["order_invariance"] = {
            "n": len(order_pairs),
            "argmax_agreement": _rate(_argmax(a["probabilities"], a["option_order"]) ==
                                      _argmax(b["probabilities"], b["option_order"]) for a, b in order_pairs),
            "mean_tvd": _rate(_tvd(a["probabilities"], b["probabilities"]) for a, b in order_pairs),
        }
        mirror_pairs, cf_pairs = [], []
        for p in probes:
            if p["variant"] != "base":
                continue
            base = answers.get((style, p["id"], "action"))
            for suffix, bucket in (("_mirror", mirror_pairs), ("_cf", cf_pairs)):
                other = answers.get((style, p["id"] + suffix, "action"))
                if (base and other and base["probabilities"] and other["probabilities"]
                        and p["id"] + suffix in by_probe):
                    bucket.append((p, by_probe[p["id"] + suffix], base, other))
        row["mirror"] = {
            "n": len(mirror_pairs),
            "argmax_agreement": _rate(
                mirror_action(int(_argmax(b["probabilities"], b["option_order"]))) ==
                int(_argmax(m["probabilities"], m["option_order"])) for _, _, b, m in mirror_pairs),
            "mean_tvd": _rate(_tvd({str(mirror_action(int(k))): v for k, v in b["probabilities"].items()},
                                   m["probabilities"]) for _, _, b, m in mirror_pairs),
        }
        changed = [x for x in cf_pairs if x[0]["truth"]["safe_actions"] != x[1]["truth"]["safe_actions"]]
        row["counterfactual"] = {
            "n": len(cf_pairs),
            "n_safe_set_changed": len(changed),
            "mean_tvd_all": _rate(_tvd(b["probabilities"], c["probabilities"]) for _, _, b, c in cf_pairs),
            "mean_tvd_safe_set_changed": _rate(_tvd(b["probabilities"], c["probabilities"]) for _, _, b, c in changed),
            "argmax_changed_when_safe_set_changed": _rate(
                _argmax(b["probabilities"], b["option_order"]) != _argmax(c["probabilities"], c["option_order"])
                for _, _, b, c in changed),
            "inputs_identical": style in MOTION_BLIND_STYLES,
        }
        report["styles"][style] = row
    return report


def _median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
