"""approach / side / conflict balanced accuracy per (model, style) from the E1 matrix records, as JSON.

    python scripts/e1/state_ba.py RESULTS_DIR PROBES.jsonl > state_ba.json

RESULTS_DIR holds one <model>/records.jsonl per model (a run still in progress is read as far as it got).
Output: {model: {style: {question: {"ba": float|null, "n": int}}}}.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uav_eval.e1 import analyze  # noqa: E402

root, probes_path = Path(sys.argv[1]), sys.argv[2]
probes = [json.loads(line) for line in open(probes_path, encoding="utf-8")]
out = {}
for records_path in sorted(root.glob("*/records.jsonl")):
    records = []
    for line in open(records_path, encoding="utf-8"):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:   # the last line of a run being written
            pass
    report = analyze(records, probes)
    out[records_path.parent.name] = {
        style: {q: {"ba": row[q]["balanced_accuracy"], "n": row[q]["n"]} for q in ("approach", "side", "conflict")}
        for style, row in report["styles"].items()}
json.dump(out, sys.stdout, indent=1)
