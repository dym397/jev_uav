# Open Jev-style models for UAV conflict avoidance

## Goal

Compare publicly available Jev-style decision models on the same UAV conflict-avoidance simulator. Test whether model choice and input design affect collision avoidance, goal completion, arrival timing, and inference cost. Treat all first-round checkpoints as zero-shot: none has been trained on this UAV environment.

## Existing system and constraints

- Keep `UAVEnv`, its nine actions, dynamics, reward, and terminal conditions fixed. The D3QN policy is a reference baseline, not the required input format for decision models.
- Local machine: RTX 3060 6 GB. Accessible Linux host: two RTX 3090 24 GB cards, with large data disks. Keep model caches and results on `/home/mydisk1/jev_uav`; do not copy credentials into code or reports.
- The existing NanoJev checkpoint, Laya checkpoints, and JevK5-9B weights are already on the Linux host. Other public models enter through the same evaluation contract as they are installed. Record models that cannot run, with the precise reason, rather than silently omit them.
- No claim of flight safety, calibration, or superiority follows from a small simulator screen.

## Experimental factors

The model receives a `state`, a question, and nine candidate controls. The model must return a complete probability distribution over the nine action IDs; deterministic argmax executes the control. Boolean-only models may score all nine actions with separate Noul questions, but must be reported as a distinct decision mechanism. No model sees the simulator's future states, reward, actual next outcome, or an oracle collision label.

The benchmark separates *information* from *wording*:

1. `paper14`: the published 14-value RL observation. This is a historical control arm.
2. `kinematic_fields`: current own position, heading, speed, goal, clock/deadline, and detected relative contact positions and velocities, presented as compact fields. This is an idealized current-sensor arm because the simulator supplies exact current contact velocity.
3. `kinematic_prose`: exactly the same kinematic facts and precision as `kinematic_fields`, organized as a natural-language encounter brief.
4. `cpa_brief`: the kinematic facts plus deterministic closest-point-of-approach estimates over a stated horizon. This tests whether supplying computed geometry helps; it is not a pure wording comparison.

The nine candidate descriptions state the actual yaw-rate and acceleration over the next one-second action. Two question phrasings, `balanced` and `safety_first`, vary priorities while leaving the state and candidates fixed. The primary prompt comparison is `kinematic_fields` versus `kinematic_prose` with the same information and question. Other contrasts are explicitly labeled as changes of information or task framing.

## Evaluation protocol

- Reuse the three fixed pilot scenes only for smoke tests and illustrated traces. Earlier `complex_100..107` scenes have informed prompt work and are development data.
- Use new development seeds `300..319`; reserve `1000..1099` as a final held-out set after prompt and adapter choices are frozen. Every model and prompt arm starts from the same scenario seeds. Closed-loop trajectories can diverge after the first action; compare outcomes by paired seed, not by treating later states as identical.
- Load each model once, then evaluate prompt arms in a fixed recorded order. Record model/revision, runtime, precision, hardware, source paths or hashes, prompt ID, scenario seed, every request and probability vector, action, latency, invalid output, episode outcome, reward, timing, and minimum swept separation. Exclude model load time from per-decision latency and report it separately.
- Primary outcome: collision rate. Secondary: goal success, on-time success, timeout/out-of-bounds, minimum separation, reward, decision latency, and invalid response rate. Report counts and uncertainty for final-set rates; do not select prompts using held-out outcomes.
- Preserve the unmodified D3QN, maintain-action, and deterministic harness baselines as context. A model-assisted harness must be reported separately from model-only action selection.

## Software design

- A prompt builder converts an `Observation` to a named representation and produces the same nine candidate IDs for every backend.
- Backend adapters normalize native NanoJev and Jev-compatible Choice output to `{action_id: probability}`. The runner rejects missing, non-finite, negative, or non-normalized distributions and records a visible invalid result; it may use the fixed maintain-action fallback so an episode can finish.
- A runner drives `UAVEnv` without model imports in the environment, writes append-friendly JSONL decisions and a machine-readable summary, and supports a smoke run before longer sweeps.
- A catalog lists candidate models, weight location, supported primitives, modality, runtime, and run state. Adding a model requires an adapter or a verified Jev-compatible endpoint, not a change to the simulator.

## Initial implementation and expansion

Implement and verify the prompt suite and runner with a deterministic fake backend. Then run the existing NanoJev checkpoint as an end-to-end check. Integrate the already downloaded Laya and JevK5 checkpoints next. Expand to other released open weights in hardware-feasible groups, preserving each model's native implementation and exact version. Larger 27B-class and multimodal models receive separate resource checks and modality-appropriate inputs rather than being marked as failed on a text-only nine-action test.
