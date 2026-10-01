"""Run open decision models on paired UAV scenarios."""

import argparse
import json
import platform
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from uav_eval.loading import BACKENDS, build_backend
from uav_eval.prompts import PromptSpec, QUESTION_STYLES, STATE_STYLES
from uav_eval.runner import run_sweep, scenario_set

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--model-path", type=Path,
                        help="Local weights directory for the jevk5 or laya backends")
    parser.add_argument("--endpoint", help="Local Jev-compatible POST /v1/systemone URL")
    parser.add_argument("--model-id", help="Exact served model ID or checkpoint revision")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("external/NanoJev/checkpoints/NanoJev-unified"))
    parser.add_argument("--source-dir", type=Path, default=Path("external/NanoJev"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--split", choices=("smoke", "development", "heldout"), default="smoke")
    parser.add_argument("--state-styles", nargs="+", choices=STATE_STYLES,
                        default=list(STATE_STYLES))
    parser.add_argument("--question-styles", nargs="+", choices=QUESTION_STYLES,
                        default=list(QUESTION_STYLES))
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.backend == "systemone" and not args.endpoint:
        parser.error("--endpoint is required for systemone")
    if args.backend in ("jevk5", "laya") and args.model_path is None:
        parser.error(f"--model-path is required for {args.backend}")

    started = perf_counter()
    backend = build_backend(args.backend, model_id=args.model_id, model_path=args.model_path,
                            endpoint=args.endpoint, checkpoint_dir=args.checkpoint_dir,
                            source_dir=args.source_dir, device=args.device, precision=args.precision)
    model_id = backend.model_id
    load_seconds = perf_counter() - started if args.backend != "systemone" else None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    output_dir = args.output_dir or Path("outputs/open_models") / slug / f"{args.split}_{timestamp}"
    specs = [PromptSpec(state_style, question_style)
             for state_style in args.state_styles for question_style in args.question_styles]
    report = run_sweep(backend, specs, scenario_set(args.split), output_dir,
                       max_steps=args.max_steps)
    metadata = {
        "model_id": model_id,
        "backend": args.backend,
        "split": args.split,
        "state_styles": args.state_styles,
        "question_styles": args.question_styles,
        "max_steps": args.max_steps,
        "checkpoint_dir": str(args.checkpoint_dir) if args.backend == "nano" else None,
        "model_path": str(args.model_path) if args.model_path else None,
        "source_dir": str(args.source_dir) if args.backend == "nano" else None,
        "endpoint": args.endpoint if args.backend == "systemone" else None,
        "load_seconds": load_seconds,
        "timestamp_utc": timestamp,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "model_id": model_id,
                      "arms": [{"state_style": arm["state_style"],
                                "question_style": arm["question_style"],
                                "outcomes": arm["outcomes"],
                                "invalid_decisions": arm["invalid_decisions"]}
                               for arm in report["arms"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
