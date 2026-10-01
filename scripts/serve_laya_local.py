"""Serve one already-downloaded Laya checkpoint on the local loopback interface."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--kind", choices=("english", "typed-decisions"), default="english")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=18731)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error(f"Missing checkpoint directory: {args.checkpoint}")

    from laya import Router
    from laya.serve import create_app
    import uvicorn

    router = Router(
        models={args.kind: str(args.checkpoint.resolve())},
        default=args.kind,
        device=args.device,
    )
    router.load(args.kind)
    uvicorn.run(create_app(router), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
