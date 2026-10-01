# NanoJev state-language ablation

## Question and controls

Does a semantic rendering of the existing 14-value UAV observation change the zero-shot behavior of the frozen `C-Tianyu/NanoJev` `unified-games-v1` checkpoint?

The checkpoint, native Choice instruction, nine action descriptions, environment parameters, and initial scenarios were held fixed. The sole input change was the text serialization of the same 14 numeric values. No intruder velocity, raw position, history, deadline metadata, or UAV training examples were added. The model was loaded once and reused across all arms. Each arm ran 11 deterministic scenarios: the three fixed pilot scenes plus complex scenario seeds 100–107, with a 30-step cap.

| Input style | Description | Initial one-crossing state tokens |
| --- | --- | ---: |
| `labeled` | Field names and numeric values | 92 |
| `semantic` | Full prose for every field and all nine sectors | 346 |
| `compact` | Full labeled vector plus a short interpretation of the goal and occupied sectors | 169 |

## Results

| Frozen NanoJev input | Reached goal | Collision | Timeout | Valid decisions | Per-scenario decision P50 range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Labeled | 3/11 | 5/11 | 3/11 | 156/156 | 201–226 ms |
| Full prose | 0/11 | 6/11 | 5/11 | 203/203 | 399–456 ms |
| Compact interpretation | 5/11 | 5/11 | 1/11 | 119/119 | 243–385 ms |
| D3QN (reference) | 5/11 | 6/11 | 0/11 | 91/91 | 0.26–0.43 ms |

The compact version retained all three successes from the labeled version and added success in `no_intruder` and `complex_100`. It still collided in five scenes. The full-prose version did not reach the goal in any scene. The D3QN reference row comes from the separate recorded run in `outputs/semantic_ablation/d3qn/summary.json`; this is context, not a matched training comparison. Valid decisions count legal outputs, not safe actions.

The result shows that this frozen checkpoint's behavior is sensitive to state wording. It does not establish that semantic input improves safety or generalization. The full-prose slowdown is partly associated with much longer input; the experiment does not isolate length from wording. Eleven selected scenarios are too few for a reliable performance estimate, and these seeds should not be reused as a final holdout after further prompt changes. The checkpoint was trained on games, not UAV avoidance; all NanoJev runs here were zero shot.

## Reproduce

```powershell
.venv\Scripts\python.exe experiment_semantic.py --complex-seeds 8 --output-dir outputs\semantic_ablation
.venv\Scripts\python.exe -m pytest -q
```

The machine used an RTX 3060 Laptop GPU with 6 GB memory. `outputs/semantic_ablation/comparison.json` records input examples, token counts, aggregate results and paired outcomes. Each style directory contains the full per-scenario `summary.json` and trajectory PNGs. Model loading time is excluded from decision latency.
