"""approach / side / conflict balanced accuracy per (model, style) from E1 records, as JSON.

    python scripts/e1/state_ba.py PROBES.jsonl RECORDS.jsonl [RECORDS.jsonl ...] > state_ba.json

Each records file is <model>/records.jsonl (the directory names the model); files are merged per model, a later file
overriding an earlier one for the same style. A run still in progress is read as far as it got.
Output: {model: {style: {question: {"ba": float|null, "n": int}}}}.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uav_eval.e1 import analyze  # noqa: E402

probes = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
out = {}
for records_path in map(Path, sys.argv[2:]):
    records = []
    for line in open(records_path, encoding="utf-8"):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:   # the last line of a run being written
            pass
    report = analyze(records, probes)
    out.setdefault(records_path.parent.name, {}).update({
        style: {q: {"ba": row[q]["balanced_accuracy"], "n": row[q]["n"]} for q in ("approach", "side", "conflict")}
        for style, row in report["styles"].items()})
json.dump(out, sys.stdout, indent=1)
