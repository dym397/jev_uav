# UAV-JevHarness 14-Value Ablation

## Question and Strict Input Contract

Can a structured **JevHarness** layer (`UAVJevHarness` in [`agents/jev_harness.py`](../agents/jev_harness.py)) unlock safe, on-time tactical conflict resolution with the frozen `C-Tianyu/NanoJev` (`unified-games-v1`) checkpoint while strictly consuming **only the same 14 numeric observation values** (`state.numeric`) and **9 discrete control actions** (`ACTION_SPACE`) as the D3QN baseline in Li et al. (Drones 2023)?

To ensure a 100% fair apple-to-apple comparison with D3QN:
1. **Strict 14-Value Input Boundary**: `UAVJevHarness` reads only `state.numeric` (`heading_rad`, `speed_mps`, `goal_distance_ratio`, `goal_bearing_ccw_rad`, `eta_remaining_ratio`, `sector_0..8`). Hidden simulator fields (`own_position`, `goal`, `due_time`, `current_time`, `eta`, `nearest_distance`, `contacts`) are never read (verified by unit test `test_harness_strictly_uses_only_fourteen_numeric_values` in [`tests/test_jev_pilot.py`](../tests/test_jev_pilot.py)).
2. **Episode Memory & Angle Unwrapping**: Within an episode, `UAVJevHarness` unwraps `goal_bearing_ccw_rad` from $[0, 2\pi)$ to signed bearing $\beta_g \in (-\pi, \pi]$, integrates own-UAV step displacements from $(v_{t-1}, \psi_{t-1})$, and matches occupied radar sectors across consecutive steps via shifted-position triangle-inequality consistency ($|d_k^{(t)} - d_{\text{shifted}, j}^{(t-1)}| \le 8.0\text{ m}$) to estimate sector closing speeds $c_k$ and intrinsic radial velocities $v_{\text{intr}, \parallel} = (\mathbf{v}_{\text{prev}} \cdot \mathbf{u}_k) - c_k$.
3. **Candidate Projection, Pruning & Multi-Question Inference**:
   - `harness_unshielded`: Appends the 14-value kinematic summary to the state and offers all $K=9$ actions to `NanoJev` without candidate pruning or safety shielding.
   - `jev_harness`: Prunes the 9 actions to the top $K=3$ safe/viable candidates (matching `NanoJev-unified`'s training distribution $K \in \{2, 3, 4\}$), queries `NanoJev` simultaneously on three structured questions in a single batched forward pass (`action` choice, `threat_level` 4-level score, and `safety_margin` boolean), and applies a kinematic safety shield over the Pareto-viable candidates.

## Aggregate Results (11 Paired Scenarios, 30-Step Cap)

| Policy / Mode | Reached Goal | On-Time Arrival | Collisions | Timeouts | Mean Reward | Valid Decisions | P50 Latency Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d3qn` (1000-ep checkpoint) | 5/11 (45.5%) | 5/11 (45.5%) | 6/11 | 0/11 | -13.63 | 91/91 | 0.26–0.43 ms |
| `labeled` (raw 14-value text, $K=9$) | 3/11 (27.3%) | 3/11 (27.3%) | 5/11 | 3/11 | -23.88 | 156/156 | 201–226 ms |
| `compact` (short prose, $K=9$) | 5/11 (45.5%) | 5/11 (45.5%) | 5/11 | 1/11 | -13.75 | 119/119 | 243–385 ms |
| `harness_unshielded` (14D projection text only, $K=9$) | 3/11 (27.3%) | 3/11 (27.3%) | 8/11 | 0/11 | -31.52 | 144/144 | 440–616 ms |
| **`jev_harness` (Full 14D `UAV-JevHarness`, $K=3$)** | **9/11 (81.8%)** | **8/11 (72.7%)** | **2/11** | **0/11** | **+7.31** | **201/201** | **260–536 ms** |

## Scenario-by-Scenario Breakdown

| Scenario | `d3qn` | `labeled` | `compact` | `harness_unshielded` | **`jev_harness`** |
| --- | --- | --- | --- | --- | --- |
| `no_intruder` | success (11, on-time) | timeout (30) | success (11, on-time) | success (16, on-time) | **success (12, on-time)** |
| `one_crossing` | success (11, on-time) | timeout (30) | timeout (30) | collision (8) | **success (22, on-time)** |
| `two_crossing` | success (11, on-time) | success (11, on-time) | success (12, on-time) | collision (10) | **success (24, on-time)** |
| `complex_100` | collision (8) | collision (8) | success (11, on-time) | collision (11) | **success (19, on-time)** |
| `complex_101` | collision (6) | timeout (30) | collision (6) | collision (18) | **success (30, late)** |
| `complex_102` | collision (3) | collision (2) | collision (3) | collision (4) | collision (3, unavoidable) |
| `complex_103` | collision (8) | collision (8) | collision (8) | success (22, on-time) | **success (17, on-time)** |
| `complex_104` | success (11, on-time) | success (11, on-time) | success (11, on-time) | collision (9) | **success (18, on-time)** |
| `complex_105` | collision (5) | collision (9) | collision (8) | collision (7) | collision (8) |
| `complex_106` | success (11, on-time) | success (11, on-time) | success (12, on-time) | success (25, on-time) | **success (26, on-time)** |
| `complex_107` | collision (6) | collision (6) | collision (7) | collision (14) | **success (22, on-time)** |

## Key Findings

1. **Why Prompt-Only (`harness_unshielded`) Fails vs. Full `jev_harness`**:
   - Merely writing kinematic projections into the text while offering all $K=9$ actions (`harness_unshielded`) achieves only 3/11 successes because the frozen `unified-games-v1` checkpoint was trained exclusively on $K \in \{2, 3, 4\}$ game actions (`log_k` out-of-distribution at $K=9$) and carries strong lexical priors from Maze/Snake/ViZDoom.
   - When `jev_harness` prunes candidates to $K=3$ verified maneuvers and applies the 14-value kinematic safety shield, success jumps from **3/11 to 9/11 (81.8%)** (and **9/10 among physically solvable scenes**, since `complex_102` has no collision-free 3-step trajectory under $\pm 6^\circ/\text{s}$ turn limits), outperforming D3QN (**5/11**) by **+36.3 percentage points** under the exact same 14-value input.
2. **Simultaneous Multi-Question Explainability (`choice` + `score` + `boolean`)**:
   - In a single batched GPU forward pass (`outputs/harness_ablation/jev_harness/diagnostics.json`), `jev_harness` outputs normalized candidate probabilities (`action`), a continuous tactical threat score (`threat_level` $\in [0, 3]$), and a safety confidence probability (`safety_margin` $p_{\text{true}} \in [0, 1]$).
   - Because `NanoJev-unified` has not yet been post-trained on UAV trajectories, its raw `threat_level` score (~1.52–1.62) and `safety_margin` probability (~0.61–0.68) are uncalibrated—providing a direct quantitative motivation for **Stage 2: Domain Post-Training (Fine-Tuning `NanoJev` on UAV trajectories)** to align the model's internal choice probabilities and risk/safety heads with physical UAV outcomes.

## Reproduce

```powershell
.venv\Scripts\python.exe experiment_harness.py --complex-seeds 8 --output-dir outputs\harness_ablation
.venv\Scripts\python.exe -m pytest -q
```
