# ETA-D3QN Baseline Implementation Plan

**Goal:** Reproduce the paper's 2D tactical conflict-resolution MDP and a replaceable D3QN decision agent.

**Architecture:** `UAVEnv` owns motion, collisions, observations and reward. A typed observation exposes both the paper's 14-value vector and `state_to_text()`. Agents consume that observation through `select_action`; scenario generation, training, evaluation and plotting remain outside the environment.

**Tech stack:** Python 3.12, NumPy, PyTorch, Matplotlib, pytest.

**Source:** `paper/1.pdf`, especially Sections 3, 5 and 6. The paper leaves intruder motion, sector edges, reward coefficients and neural-network widths unspecified; chosen defaults must be recorded in README.

## Tasks

1. Add failing checks for Eq. (5), nine action combinations, sector observation, collision, ETA and reproducible scenarios. Implement configuration and environment until those checks pass.
2. Add failing checks for the dueling head, Double Q target and n-step replay behavior. Implement the D3QN network, buffer and agent until those checks pass.
3. Add train and test commands, metric logging, checkpoints and trajectory/reward plots. Run the full test suite and a short train/evaluate smoke run.
4. Document the exact paper mapping, explicit reconstruction assumptions and how a future `JevAgent.select_action(state)` can replace D3QN.

## Verification

- `python -m pytest -q`
- `python train.py --episodes 8 --batch-size 8 --output-dir outputs/smoke`
- `python test.py --checkpoint outputs/smoke/best.pt --episodes 3 --output-dir outputs/smoke_eval`

The smoke run checks execution and output artifacts, not reproduction of the paper's published performance.
