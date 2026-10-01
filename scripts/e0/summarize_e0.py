"""Print one row per (model, prompt style, scenario) from an E0 results tree."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for summary in sorted(root.glob("*/summary.json")):
    report = json.loads(summary.read_text())
    for arm in report["arms"]:
        for name, row in arm["scenarios"].items():
            sep = row["minimum_separation_m"]
            print(f"{report['model_id']:<22} {arm['state_style']:<20} {name:<14} "
                  f"{row['outcome']:<10} steps={row['steps']:<3} invalid={row['invalid_decisions']:<3} "
                  f"min_sep={'-' if sep is None else round(sep, 1)!s:<6} "
                  f"p50={row['latency_p50_ms']:.0f}ms actions={''.join(map(str, row['actions']))}")
