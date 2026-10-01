# Open Jev UAV Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run versioned Jev-style open models on the fixed UAV simulator under controlled state and question variants.

**Architecture:** A prompt builder derives named state representations from the existing `Observation`; backend adapters normalize model probabilities; a runner logs closed-loop episodes on paired scenarios. The environment and D3QN policy remain unchanged.

**Tech Stack:** Python 3.11+, NumPy, pytest, existing NanoJev predictor, HTTP JSON via Python stdlib, model-specific servers on Linux.

**Spec:** `docs/superpowers/specs/2026-09-30-open-jev-uav-evaluation-design.md`

## Global Constraints

- Keep `UAVEnv` dynamics, nine actions, reward, and terminal conditions unchanged.
- Never include future simulator state, collision outcome, or reward in model input.
- State styles: `paper14`, `kinematic_fields`, `kinematic_prose`, `cpa_brief`; question styles: `balanced`, `safety_first`.
- Treat `kinematic_fields` versus `kinematic_prose` as the same-facts wording comparison. Label `paper14` and `cpa_brief` as different-information comparisons.
- Use fixed pilot scenes for smoke and complex seeds 300–319 for development. Reserve 1000–1099 for final held-out evaluation.
- Keep model assets and run artifacts under `/home/mydisk1/jev_uav` on Linux. Do not persist credentials.

## Review Focus

- No detected intruders: prompt must say zero contacts, with no fabricated risk.
- Opposite contact velocities at identical positions: rich inputs must differ; `paper14` may match.
- CPA when relative velocity is zero or contact is already receding: avoid division by zero and clamp time to the stated horizon.
- Malformed model distributions: log invalidity and use the declared maintain fallback.
- Scenario pairing and result provenance: every arm must use the same initial seed list and record exact prompt/model identifiers.

---

### Task 1: Prompt suite

**Files:** Create `uav_eval/prompts.py`, `uav_eval/__init__.py`; test `tests/test_uav_eval_prompts.py`.

**Interfaces:** `PromptSpec(state_style: str, question_style: str)` and `build_prompt(observation: Observation, spec: PromptSpec) -> dict` producing `{state, questions}`. `action_criteria() -> dict[str,str]` maps IDs `0..8` to explicit one-second yaw/acceleration controls.

- [ ] Write tests for all four state styles, same-facts field/prose content, opposite velocities, zero contacts, CPA edge cases, and nine candidate control descriptions.
- [ ] Run the tests and confirm the expected missing-feature failure.
- [ ] Implement prompt serialization from current observation only; derive CPA from relative contact vectors and velocities.
- [ ] Run prompt tests and the existing suite.

### Task 2: Probability adapters

**Files:** Create `uav_eval/backends.py`; test `tests/test_uav_eval_backends.py`.

**Interfaces:** `normalize_choice(response: dict) -> dict[int,float]`; `NanoBackend` calls the existing native predictor; `SystemOneHTTPBackend` posts the exact `{state, questions, model?}` payload and normalizes `answers.action.probabilities`.

- [ ] Write tests for valid distributions, malformed/missing/non-finite entries, native Nano shape, and an in-process HTTP server response.
- [ ] Run to observe the expected failure.
- [ ] Implement adapters with no model-specific code in the simulator.
- [ ] Run adapter tests and the existing suite.

### Task 3: Paired episode runner

**Files:** Create `uav_eval/runner.py` and `benchmark_open_models.py`; test `tests/test_uav_eval_runner.py`.

**Interfaces:** `run_sweep(backend, specs, scenarios, output_dir, max_steps) -> dict` writes `decisions.jsonl` and `summary.json`; `scenario_set(split) -> dict[str,Scenario]` supplies smoke/development/held-out seeds. CLI selects backend, endpoint/checkpoint, styles, question styles, split, and output directory.

- [ ] Write tests with deterministic backends for scenario pairing, action/fallback, trace provenance, minimum swept separation, and aggregate outcome counts.
- [ ] Run to observe the expected failure.
- [ ] Implement runner and CLI; record per-decision payload, distribution, selected action, latency, and error.
- [ ] Run runner tests and the entire project suite.

### Task 4: Live model screen and research record

**Files:** Create `docs/open_models_uav.md`; output under `outputs/open_models_*` locally or `/home/mydisk1/jev_uav/results` remotely.

- [ ] Run CLI smoke with the existing NanoJev checkpoint, then development seeds for selected prompt arms.
- [ ] Install/run Laya and JevK5 servers in the existing Linux environment or isolated environments; use already downloaded weights.
- [ ] Run the same smoke/development arms against both and verify logged probabilities, response errors, and hardware occupancy.
- [ ] Record exact commands, versions, checkpoint paths, prompt examples, outcomes, latency, limitations, and a model expansion queue.

## Execution notes

The root project has no `.git`; skip plan commits. A server lacking a dependency or compatible endpoint is recorded as pending with its concrete error rather than interpreted as a model failure. Do not run the final held-out split until prompt and adapter choices are frozen.
