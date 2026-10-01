# UAV motion information and NanoJev representation experiment

## Design

This is a zero-shot probe of the frozen `C-Tianyu/NanoJev` `unified-games-v1` checkpoint. All arms use the same NanoJev Choice question, nine actions, 2D environment and initial scenarios. The three inputs are:

| Arm | Information and expression |
| --- | --- |
| `labeled` | Original 14-value paper observation only |
| `flat_contacts` | Same 14 values plus each detected intruder's relative forward/left position and velocity as key/value fields |
| `structured_contacts` | Exactly the same numeric facts as `flat_contacts`, grouped into a concise description per intruder |

Contacts are included only inside the existing 100 m detection radius. Position and relative velocity are expressed in own-aircraft body axes. The simulator supplies exact **current** intruder position and velocity; this is a perfect-observation assumption, not a tested noncooperative tracking system. No future trajectory, collision label, expert action, history or UAV fine-tuning is supplied. The D3QN policy still receives only its unchanged 14-value vector.

The counterfactual pair places one intruder at the same location with velocity `(0,-4)` or `(0,+4)` m/s. The initial 14-value observations are exactly equal. On eight geometries, the grid varies intruder x among 930, 940, 950 and 960 m and y among 1010 and 1040 m, holding everything else fixed. Total variation distance compares the two nine-action distributions; these probabilities are not calibrated safety confidence.

## Results

For the original `(950,1040)` pair, all three arms chose action 5 (`Accelerate`) for both motion directions. The distribution difference was 0 for `labeled`, 0.0084 for `flat_contacts`, and 0.0095 for `structured_contacts`.

| Input | Argmax changes in 8 motion pairs | Mean distribution distance |
| --- | ---: | ---: |
| Original 14 values | 0/8 | 0 |
| Added motion, flat fields | 0/8 | 0.0090 |
| Added motion, grouped by intruder | 1/8 | 0.0089 |

The one structured-input action change occurred at `(940,1040)`: `Accelerate` for the approaching intruder and `Maintain` for the moving-away intruder. An action change alone does not establish that the response was safer.

Closed-loop evaluation used the three fixed pilot scenes, the receding-motion counterpart, and eight newly seeded complex scenes (seeds 200–207), with 30 steps maximum per scene:

| Input | Reached goal | Collision | Timeout | Valid decisions | Per-scene decision P50 range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 14 values | 3/12 | 5/12 | 4/12 | 183/183 | 203–229 ms |
| Added motion, flat fields | 2/12 | 5/12 | 5/12 | 196/196 | 223–625 ms |
| Added motion, grouped by intruder | 3/12 | 5/12 | 4/12 | 177/177 | 229–584 ms |
| D3QN (reference) | 8/12 | 4/12 | 0/12 | 110/110 | 0.27–0.76 ms |

The D3QN reference row comes from the separate recorded run in `outputs/motion_ablation/d3qn/summary.json`. This is not a matched training comparison: D3QN was trained for UAV avoidance while this NanoJev checkpoint was trained on games. Valid decisions count legal outputs, not safe actions.

The frozen NanoJev checkpoint showed small probability shifts when motion was supplied, but almost no action-level response to reversed motion and no aggregate closed-loop improvement. These results do not establish whether a UAV-adapted NanoJev model can use relative motion. More prose or raw features alone did not solve the transfer problem in this probe.

## Reproduction

```powershell
.venv\Scripts\python.exe experiment_motion.py --complex-seeds 8 --output-dir outputs\motion_ablation
.venv\Scripts\python.exe experiment_motion_grid.py --output outputs\motion_ablation\counterfactual_grid.json
.venv\Scripts\python.exe -m pytest -q
```

`outputs/motion_ablation/comparison.json` contains full input examples, per-arm summaries and paired scene outcomes. `counterfactual_grid.json` contains the eight first-step probes. Each policy directory contains per-scene metrics and trajectories. Inference timings exclude checkpoint load.
