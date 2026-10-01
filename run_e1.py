"""E1 layered single-step probes.

    python run_e1.py make-probes --output probes.jsonl
    python run_e1.py run --backend jevk5 --model-path ... --probes probes.jsonl --output-dir DIR
    python run_e1.py analyze --probes probes.jsonl DIR [DIR ...]
"""

import argparse
import json
from pathlib import Path

from uav_eval.e1 import QUESTIONS, analyze, run_e1
from uav_eval.loading import BACKENDS, build_backend
from uav_eval.probes import generate, read_probes, write_probes
from uav_eval.prompts import STATE_STYLES


DEFAULT_STYLES = ["paper14", "kinematic_fields", "kinematic_prose", "first_person_polar", "third_person_polar"]


def _table(reports: dict) -> str:
    """One markdown row per (model, style); baselines from the first report."""
    lines = ["| model | style | approach bal | side bal | conflict bal | safe mass | argmax safe | "
             "argmax good | argmax unsafe | modal action (share) | order agree | mirror agree | "
             "cf TVD (safe set changed) | invalid |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(value):
        return "-" if value is None else f"{value:.2f}"

    for model, report in reports.items():
        for style, row in report["styles"].items():
            modal = row["action"]["modal_action"]
            modal_text = "-" if modal is None else f"{modal[0]} ({row['action']['modal_action_share']:.2f})"
            lines.append(
                f"| {model} | {style}{' (assisted)' if row['assisted_risk'] else ''} | "
                f"{f(row['approach']['balanced_accuracy'])} | {f(row['side']['balanced_accuracy'])} | "
                f"{f(row['conflict']['balanced_accuracy'])} | {f(row['action']['safe_mass'])} | "
                f"{f(row['action']['argmax_safe_rate'])} | {f(row['action']['argmax_good_rate'])} | "
                f"{f(row['action']['argmax_unsafe_rate'])} | "
                f"{modal_text} | "
                f"{f(row['order_invariance']['argmax_agreement'])} | {f(row['mirror']['argmax_agreement'])} | "
                f"{f(row['counterfactual']['mean_tvd_safe_set_changed'])} | {row['invalid']}/{row['calls']} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make-probes")
    make.add_argument("--seed", type=int, default=20260930)
    make.add_argument("--per-family", type=int, default=24)
    make.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--backend", choices=BACKENDS, required=True)
    run.add_argument("--model-id")
    run.add_argument("--model-path", type=Path)
    run.add_argument("--endpoint")
    run.add_argument("--checkpoint-dir", type=Path, default=Path("external/NanoJev/checkpoints/NanoJev-unified"))
    run.add_argument("--source-dir", type=Path, default=Path("external/NanoJev"))
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--probes", type=Path, required=True)
    run.add_argument("--styles", nargs="+", choices=STATE_STYLES, default=DEFAULT_STYLES)
    run.add_argument("--questions", nargs="+", choices=QUESTIONS, default=list(QUESTIONS))
    run.add_argument("--limit", type=int, help="Only the first N probes (smoke runs)")
    run.add_argument("--output-dir", type=Path, required=True)

    report = sub.add_parser("analyze")
    report.add_argument("--probes", type=Path, required=True)
    report.add_argument("runs", type=Path, nargs="+")
    report.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "make-probes":
        probes = generate(args.seed, per_family=args.per_family)
        write_probes(args.output, probes)
        print(json.dumps({"probes": len(probes), "decision_eligible":
                          sum(p["truth"]["decision_eligible"] for p in probes), "output": str(args.output)}))
    elif args.command == "run":
        probes = read_probes(args.probes)[: args.limit]
        backend = build_backend(args.backend, model_id=args.model_id, model_path=args.model_path,
                                endpoint=args.endpoint, checkpoint_dir=args.checkpoint_dir,
                                source_dir=args.source_dir, device=args.device)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run_config.json").write_text(json.dumps({
            "backend": args.backend, "model_id": backend.model_id,
            "model_path": str(args.model_path) if args.model_path else None,
            "probes": str(args.probes), "styles": args.styles, "questions": args.questions,
            "limit": args.limit}, indent=2), encoding="utf-8")
        run_e1(backend, probes, args.styles, args.output_dir / "records.jsonl", questions=args.questions)
        print(json.dumps({"done": str(args.output_dir)}))
    else:
        probes = read_probes(args.probes)
        reports = {}
        for directory in args.runs:
            with (directory / "records.jsonl").open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            model = records[0]["model_id"] if records else directory.name
            reports[model] = analyze(records, probes)
            (directory / "summary.json").write_text(json.dumps(reports[model], indent=2), encoding="utf-8")
        table = _table(reports)
        baselines = next(iter(reports.values()))["baselines"] if reports else {}
        text = table + "\n\nBaselines: " + json.dumps(baselines, indent=1)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
