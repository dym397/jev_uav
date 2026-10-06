"""E1 perception/decision consistency per (model, style), as JSON.

    python scripts/e1/consistency.py PROBES.jsonl RECORDS.jsonl [RECORDS.jsonl ...] > consistency.json

Every question is its own call on the same state, so a model's conflict and side answers can be set against the
action it picked for that state. Over the decision-eligible probes (safe and unsafe actions both exist):
  safe_rate                      argmax action in the simulator's safe set
  safe_if_conflict_right/_wrong  safe_rate split by whether its own conflict answer was right
  quadrants                      share of (conflict right/wrong) x (action safe/unsafe)
Over all probes, against its own answers (truth not used):
  maintain_if_said_clear / _conflict   P(action = MAINTAIN) given its own conflict answer
  avoid_if_said_conflict               P(turn or decelerate) given it said conflict
  away_if_said_conflict                P(turn away from the side it said) given it said conflict on a side
  action_tvd_by_conflict_answer        TVD of the mean action distribution between its two conflict answers
                                       (0: the action ignores its own risk answer)
Files merge per model (directory name), a later file overriding an earlier one for the same style.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

RIGHT_TURNS, LEFT_TURNS, MAINTAIN = {0, 1, 2}, {6, 7, 8}, 4
AVOID = RIGHT_TURNS | LEFT_TURNS | {3}


def argmax(r):
    order = r["option_order"]
    return max(order, key=lambda k: (r["probabilities"][k], -order.index(k)))


def rate(values):
    values = list(values)
    return fmean(values) if values else None


probes = {p["id"]: p for p in map(json.loads, open(sys.argv[1], encoding="utf-8"))}
answers = defaultdict(dict)   # model -> {(style, probe, question): record}
for path in map(Path, sys.argv[2:]):
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["probabilities"] is not None:
            answers[path.parent.name][(r["style"], r["probe_id"], r["question"])] = r

out = {}
for model, got in answers.items():
    for style in sorted({k[0] for k in got}):
        rows = []
        for pid, p in probes.items():
            act, con, side = (got.get((style, pid, q)) for q in ("action", "conflict", "side"))
            if act and con:
                rows.append((p, int(argmax(act)), argmax(con), argmax(side) if side else None))
        if not rows:
            continue
        truth_conflict = lambda p: "conflict" if p["truth"]["conflict_if_hold"] else "clear"
        eligible = [(p, a, c) for p, a, c, _ in rows if p["truth"]["decision_eligible"]]
        right = [(p, a) for p, a, c in eligible if c == truth_conflict(p)]
        wrong = [(p, a) for p, a, c in eligible if c != truth_conflict(p)]
        safe = lambda p, a: a in p["truth"]["safe_actions"]
        quadrants = {f"{cr}_{sa}": rate(((c == truth_conflict(p)) == (cr == "right")) and (safe(p, a) == (sa == "safe"))
                                         for p, a, c in eligible)
                     for cr in ("right", "wrong") for sa in ("safe", "unsafe")}

        said = {k: [a for _, a, c, _ in rows if c == k] for k in ("conflict", "clear")}
        dist = {k: [rate(a == i for a in v) or 0.0 for i in range(9)] for k, v in said.items()}
        on_side = [(a, s) for _, a, c, s in rows if c == "conflict" and s in ("left", "right")]
        out.setdefault(model, {})[style] = {
            "n_eligible": len(eligible), "n_all": len(rows),
            "safe_rate": rate(safe(p, a) for p, a, _ in eligible),
            "safe_if_conflict_right": rate(safe(p, a) for p, a in right),
            "safe_if_conflict_wrong": rate(safe(p, a) for p, a in wrong),
            "n_conflict_right": len(right), "n_conflict_wrong": len(wrong),
            "quadrants": quadrants,
            "said_conflict_share": len(said["conflict"]) / len(rows),
            "maintain_if_said_clear": rate(a == MAINTAIN for a in said["clear"]),
            "maintain_if_said_conflict": rate(a == MAINTAIN for a in said["conflict"]),
            "avoid_if_said_conflict": rate(a in AVOID for a in said["conflict"]),
            "away_if_said_conflict": rate(a in (RIGHT_TURNS if s == "left" else LEFT_TURNS) for a, s in on_side),
            "toward_if_said_conflict": rate(a in (LEFT_TURNS if s == "left" else RIGHT_TURNS) for a, s in on_side),
            "action_tvd_by_conflict_answer": (0.5 * sum(abs(x - y) for x, y in zip(dist["conflict"], dist["clear"]))
                                              if said["conflict"] and said["clear"] else None),
        }
json.dump(out, sys.stdout, indent=1)
