"""E5 detail: action mix per policy and outcomes restricted to oracle-solvable seeds."""
import json
import sys
from collections import Counter
from pathlib import Path

results = Path(sys.argv[1])
oracle = json.loads((results / "oracle.json").read_text())
solvable = {r["seed"] for r in oracle if r["solvable"]}
print(f"oracle: solvable {len(solvable)}/{len(oracle)}, on time {sum(r['on_time'] for r in oracle)}")
rows = [json.loads(l) for f in ("baselines.jsonl", "JevK5-9B.jsonl") for l in (results / f).read_text().splitlines()]
for name in ("d3qn", "constant_4", "constant_3", "JevK5-9B:third_person_semantic", "JevK5-9B:first_person_semantic"):
    eps = [r for r in rows if r["policy"] == name]
    acts = Counter(a for r in eps for a in r["actions"])
    total = sum(acts.values())
    sub = [r for r in eps if r["seed"] in solvable]
    out = Counter(r["outcome"] for r in sub)
    early = sum(r["outcome"] == "collision" and r["steps"] <= 5 for r in eps)
    print(f"{name}: actions " + " ".join(f"{a}:{n / total:.2f}" for a, n in acts.most_common(4))
          + f" | on solvable seeds {dict(out)} | collisions in first 5 steps {early}")
