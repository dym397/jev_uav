"""Counterfactual probe: does frozen NanoJev notice opposite intruder motion?"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import fmean, median

import numpy as np

from agents.jev_agent import NativeNanoJevClient
from experiment_motion import STYLES, distribution, first_observation
from pilot_jev import pilot_scenarios


def counterfactual_grid() -> dict:
    pairs = {}
    for x in (930.0, 940.0, 950.0, 960.0):
        for y in (1010.0, 1040.0):
            approaching = deepcopy(pilot_scenarios()["one_crossing"])
            moving_away = deepcopy(approaching)
            for scene, vy in ((approaching, -4.0), (moving_away, 4.0)):
                scene.intruders[0].x = x
                scene.intruders[0].y = y
                scene.intruders[0].vy = vy
            pairs[f"x{int(x)}_y{int(y)}"] = (approaching, moving_away)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("external/NanoJev/checkpoints/NanoJev-unified"))
    parser.add_argument("--source-dir", type=Path, default=Path("external/NanoJev"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/motion_ablation/counterfactual_grid.json"))
    args = parser.parse_args()
    client = NativeNanoJevClient(args.checkpoint_dir, source_dir=args.source_dir)
    rows = {}
    for name, (approaching_scene, away_scene) in counterfactual_grid().items():
        approaching = first_observation(approaching_scene, 30)
        away = first_observation(away_scene, 30)
        if not np.array_equal(approaching.numeric, away.numeric):
            raise RuntimeError(f"14-value observation differs in {name}")
        rows[name] = {}
        for style in STYLES:
            a = distribution(client, approaching, style)
            b = distribution(client, away, style)
            tv = 0.5 * sum(abs(a["probabilities"][str(i)] - b["probabilities"][str(i)])
                           for i in range(9))
            rows[name][style] = {
                "approaching_choice": a["choice"],
                "moving_away_choice": b["choice"],
                "argmax_changed": a["choice"] != b["choice"],
                "total_variation_distance": tv,
            }
    summary = {}
    for style in STYLES:
        values = [row[style]["total_variation_distance"] for row in rows.values()]
        summary[style] = {
            "pairs": len(values),
            "argmax_changes": sum(row[style]["argmax_changed"] for row in rows.values()),
            "mean_total_variation": fmean(values),
            "median_total_variation": median(values),
            "max_total_variation": max(values),
        }
    result = {"motion_pair": "same current position; vy=-4 versus +4 m/s",
              "summary": summary, "pairs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
