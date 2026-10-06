"""E5 closed-loop comparison on fresh seeds: model arms, baselines, and the summary table."""

import argparse
import json
from pathlib import Path

from uav_eval.e5 import (E5_SEEDS, ConstantPolicy, ModelPolicy, RandomPolicy, oracle_feasible, read_rows,
                         run_policy, summarize)
from uav_eval.loading import BACKENDS, build_backend
from uav_eval.prompts import STATE_STYLES

RANDOM_REPEATS = 5


def _seeds(args) -> range:
    return range(args.first_seed, args.first_seed + args.seeds)


def run_model(args) -> None:
    backend = build_backend(args.backend, model_id=args.model_id, model_path=args.model_path,
                            endpoint=args.endpoint, request_model=args.request_model,
                            api_key_env=args.api_key_env, sum_tol=args.sum_tolerance,
                            checkpoint_dir=args.checkpoint_dir, source_dir=args.source_dir,
                            device=args.device, precision="bf16")
    for style in args.styles:
        run_policy(f"{backend.model_id}:{style}", ModelPolicy(backend, style, max_invalid=args.max_invalid), _seeds(args),
                   args.output, model_id=backend.model_id, state_style=style)


def run_baselines(args) -> None:
    from agents.d3qn_agent import D3QNAgent

    d3qn = D3QNAgent.load(args.d3qn_checkpoint, device="cpu")
    d3qn.set_eval_mode(True)
    run_policy("d3qn", d3qn, _seeds(args), args.output)
    for action in range(9):
        run_policy(f"constant_{action}", ConstantPolicy(action), _seeds(args), args.output)
    for repeat in range(RANDOM_REPEATS):
        run_policy(f"random_r{repeat}", RandomPolicy(10_000 + repeat), _seeds(args), args.output)


def run_oracle(args) -> None:
    rows = [oracle_feasible(seed) for seed in _seeds(args)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows), encoding="utf-8")
    print(json.dumps({"seeds": len(rows), "solvable": sum(r["solvable"] for r in rows),
                      "solvable_on_time": sum(r["on_time"] for r in rows)}))


def _pct(value) -> str:
    return f"{100 * value:.0f}%"


def analyze(args) -> None:
    summary = summarize(read_rows(args.inputs))
    randoms = [v for k, v in summary.items() if k.startswith("random_r")]
    lines = ["| policy | n | success (95% CI) | collision (95% CI) | timeout | on time | "
             "median min sep (m) | vs D3QN: only this / only D3QN succeeds | invalid |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, row in sorted(summary.items(), key=lambda kv: -kv[1]["success_rate"]):
        if name.startswith("random_r"):
            continue
        vs = row.get("vs_reference")
        paired = f"{vs['only_this_succeeds']} / {vs['only_reference_succeeds']}" if vs else "-"
        lines.append(
            f"| {name} | {row['episodes']} | {_pct(row['success_rate'])} "
            f"({_pct(row['success_ci95'][0])}-{_pct(row['success_ci95'][1])}) | "
            f"{_pct(row['collision_rate'])} ({_pct(row['collision_ci95'][0])}-{_pct(row['collision_ci95'][1])}) | "
            f"{_pct(row['timeout_rate'])} | {_pct(row['on_time_rate'])} | "
            f"{row['median_minimum_separation_m']:.1f} | "
            f"{paired} | "
            f"{row['invalid_decisions']} |")
    if randoms:
        mean = lambda key: sum(r[key] for r in randoms) / len(randoms)
        lines.append(f"| random (mean of {len(randoms)} repeats) | {randoms[0]['episodes']} | "
                     f"{_pct(mean('success_rate'))} | {_pct(mean('collision_rate'))} | "
                     f"{_pct(mean('timeout_rate'))} | {_pct(mean('on_time_rate'))} | - | - | 0 |")
    table = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("model", "baselines", "oracle"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--output", type=Path, required=True, help="episodes.jsonl (appended, resumable)")
        cmd.add_argument("--first-seed", type=int, default=E5_SEEDS.start)
        cmd.add_argument("--seeds", type=int, default=len(E5_SEEDS))
    model = sub.choices["model"]
    model.add_argument("--backend", choices=BACKENDS, required=True)
    model.add_argument("--model-path", type=Path)
    model.add_argument("--model-id")
    model.add_argument("--endpoint", help="systemone: the /v1/systemone URL")
    model.add_argument("--request-model", help="systemone: the request's model name, if not --model-id")
    model.add_argument("--api-key-env", help="systemone: env var holding a bearer token")
    model.add_argument("--sum-tolerance", type=float, default=1e-3,
                     help="systemone: accepted |sum-1| before renormalizing (rounded probabilities)")
    model.add_argument("--checkpoint-dir", type=Path,
                       default=Path("external/NanoJev/checkpoints/NanoJev-unified"))
    model.add_argument("--source-dir", type=Path, default=Path("external/NanoJev"))
    model.add_argument("--device", default="cuda:0")
    model.add_argument("--max-invalid", type=int,
                       help="Abort the run after this many invalid decisions (default: never; they fall back to MAINTAIN)")
    model.add_argument("--styles", nargs="+", choices=STATE_STYLES,
                       default=["third_person_semantic", "first_person_semantic"])
    sub.choices["baselines"].add_argument("--d3qn-checkpoint", type=Path, required=True)
    report = sub.add_parser("analyze")
    report.add_argument("inputs", nargs="+", type=Path)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    {"model": run_model, "baselines": run_baselines, "oracle": run_oracle,
     "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
