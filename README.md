# ETA-based UAV tactical conflict resolution baseline

Python reproduction of the decision-level 2D environment and D3QN from Li et al., *An ETA-Based Tactical Conflict Resolution Method for Air Logistics Transportation*, Drones 7, 334 (2023). The source is [`paper/1.pdf`](paper/1.pdf).

## Setup and commands

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe train.py --episodes 5000 --output-dir outputs/baseline
.venv\Scripts\python.exe test.py --checkpoint outputs/baseline/best.pt --episodes 100
```

For an execution check, use `--episodes 8 --batch-size 8` for training and `--episodes 3` for testing. A short run does **not** reproduce the paper's reported performance.

Training writes `metrics.csv`, `validation.csv`, `summary.json`, `reward_curve.png`, `best.pt` and `last.pt`. `best.pt` is chosen on fixed-seed deterministic validation episodes (success rate first, then on-time rate, collision rate and reward); `last.pt` is the final training state. Testing writes scenario metrics and 2D trajectory PNGs. Simple and complex scenarios deliberately place moving intruders near the 100 m route; random training uses 40 intruders/km², while random testing uses 15 intruders/km². Reported success means reaching the goal without collision, regardless of arrival time. On-time rate is a separate metric (±10 s).

## Paper mapping

| Code | Paper |
| --- | --- |
| `environment/uav_env.py` | Eq. (5) 2D kinematics, Eq. (16)–(28) reward categories |
| `environment/observation.py` | Eq. (12)–(15), (29): 14 values, nine risk-sector nearest distances, 100 m detection radius |
| `config.py` | Table 1: 3 yaw rates × 3 accelerations = 9 actions; speed 0.1–10 m/s |
| `agents/d3qn_agent.py`, `agents/replay_buffer.py`, `models/d3qn_network.py` | Double Q target, dueling head, replay, 5-step returns, Table 2 parameters |
| `environment/scenario.py` | 2 km × 2 km plane, five static obstacles, 100 m scheduled leg |

The physical environment keeps position and intruder velocity even though the paper's **policy observation** does not expose them directly. `Observation.numeric` is the 14-field layout; `Observation.state_to_text()` serializes only those 14 values, so NanoJev does not receive hidden physical metadata. A replacement agent only needs `select_action(state) -> int`, returning an action ID from `ACTION_SPACE`. Evaluation mode is set on D3QN before test episodes; the environment has no imports from `agents` or `models`.

## NanoJev feasibility pilot

The pilot uses [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) and the `C-Tianyu/NanoJev` checkpoint at revision `unified-games-v1`. This is NanoJev's native **choice** interface: one state and nine action candidates produce a probability distribution in one forward pass. It is not Qwen text generation. The checkpoint was trained on Maze, Snake and shooting games, not UAV avoidance. Its probabilities are not calibrated safety confidence, and this zero-shot test is only an interface and feasibility probe.

Clone NanoJev into `external/NanoJev`, download `best.safetensors`, `config.json`, `tokenizer/*` and `backbone_config/*` into `external/NanoJev/checkpoints/NanoJev-unified`, then install its pinned inference dependencies from `external/NanoJev/requirements-toy.txt` with a CUDA-enabled PyTorch build appropriate for your GPU. The native predictor currently requires CUDA. Run:

```powershell
.venv\Scripts\python.exe pilot_jev.py --d3qn-checkpoint outputs/baseline_1000/best.pt --output-dir outputs/nanojev_pilot
```

The fixed no-intruder, one-crossing and two-crossing scenarios share the same start, goal, due time, observation and nine actions for both agents. `summary.json` and trajectory plots are written under separate `nanojev` and `d3qn` directories. The pilot reports outcomes, action validity and per-decision P50/P95 latency. Three scenarios do not establish a success or collision rate with statistical precision.

See [the measured pilot](docs/nanojev_pilot.md) for checkpoint provenance, the three outcomes, latency and limits of this zero-shot comparison.

To compare different text renderings of the same 14-value observation with the frozen NanoJev checkpoint, run `python experiment_semantic.py --complex-seeds 8`. The paired results and interpretation are in [the state-language ablation](docs/semantic_ablation.md). This changes only the decision agent's state text; the environment and D3QN observation remain unchanged.

To attribute the JevHarness result, run `python experiment_harness_followup.py --complex-seeds 8`. The [Harness attribution experiment](docs/harness_followup.md) compares Harness-only control, safety-filtered NanoJev choice, a three-candidate control without the final utility veto, and the original Harness on paired scenes.

To test whether NanoJev uses additional observed intruder motion, run `python experiment_motion.py --complex-seeds 8` and `python experiment_motion_grid.py`. The [motion representation experiment](docs/motion_representation_experiment.md) compares the original vector, flat relative-motion fields and the same facts grouped by intruder. It records the perfect-current-velocity assumption and counterfactual results.

## Explicit reconstruction assumptions and limits

The publication does not specify several values needed for executable code. These are set centrally and are not claimed as author-provided parameters:

- A simulation step is 1 s; goal radius is 5 m; an episode stops after 120 steps. Intruders use constant velocity and reflect at the 2 km boundary. Collision and goal checks sweep the whole step to avoid missed crossings.
- Nine equal 40° sectors are aligned with own-aircraft heading. The figure illustrates sectors but does not define their exact angular boundaries. Each sector uses only the nearest **intruder**; empty sectors are 1.
- ETA uses Eq. (24) with Eq. (25)'s speed weighting when raw constant-speed arrival lies outside the ±10 s window. The unspecified weight defaults to 0.5 and the missing middle branch uses current speed. The paper's Eq. (14) normalized time field is retained as `(ETA-now)/scheduled_leg_time`.
- Reward components retain the paper's avoidance, ETA and mission structure. The paper omits coefficient values and does not define how per-sector rewards are aggregated. Defaults in `EnvConfig` use the mean occupied-sector penalty and a one-sided ±10 s ETA penalty. The mission shaping rewards *increasing* separation during a threat; this resolves the apparent sign conflict between Eq. (17) and Eq. (28). These choices are reproducible approximations, not an exact numerical reward reproduction.
- The paper describes A* planning of a full 4D route. This baseline begins with one scheduled 100 m leg; static obstacles are kept off its straight planned corridor, matching the role of prior strategic planning. Full-route A* and secondary-conflict studies are not included.
- Network width (128), epsilon schedule and scene randomization are implementation defaults because the paper does not provide them. The published Table 2 values for learning rate, discount, replay size, batch size, five-step returns, online update delay, target update timing, rounds and MSE are used as defaults.

The paper's 99% success findings require its original training setup and 10,000-run tests. This repository supplies a reproducible baseline and reports its own measured results; it does not assert equivalence to those figures.
