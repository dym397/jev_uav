# Native NanoJev UAV choice pilot

The adapter uses `TianyuCodings/NanoJev` at commit `76fdfc9ecdca45a9bcef17991a07d3041a87685a` and checkpoint `C-Tianyu/NanoJev`, revision `unified-games-v1`. It passes the same 14 observation values used by D3QN to NanoJev's native `choice` question with the same nine action IDs. No UAV examples were used to train or adapt this checkpoint. The checkpoint's training tasks were Maze, Snake, and shooting games.

Command on this machine (RTX 3060 Laptop GPU, 6 GB):

```powershell
.venv\Scripts\python.exe pilot_jev.py --d3qn-checkpoint outputs\baseline_1000\best.pt --output-dir outputs\nanojev_pilot
```

The three scenarios share a 100 m scheduled leg and a 30 step cap. The fixed D3QN checkpoint came from 1000 training episodes. A maintain-only control reaches the goal with no intruder and collides at step 8 in both crossing scenarios.

| Policy | No intruder | One crossing | Two crossing | Invalid choices | Decision P50 |
| --- | --- | --- | --- | ---: | ---: |
| Native NanoJev, zero shot | timeout, 30 steps | timeout, 30 steps | success, 11 steps | 0/71 | 211–227 ms across scenarios |
| D3QN checkpoint | success, 11 steps | success, 11 steps | success, 11 steps | 0/33 | 0.28–0.56 ms across scenarios |

NanoJev selected repeated right-turn-and-decelerate actions in the empty scene and passed below the goal. The exact action traces, per-scenario P50/P95 latencies, rewards, path lengths and trajectory PNGs are in `outputs/nanojev_pilot/`. These timings include Python request construction and inference; D3QN timings include its `select_action` call. Model load is excluded. The sample is too small for success-rate or collision-rate claims. Zero-shot transfer failure here does not show that NanoJev cannot learn UAV avoidance. The next meaningful test is UAV-specific choice supervision or adaptation with held-out scenario evaluation, while preserving the 14-value observation and nine actions.
