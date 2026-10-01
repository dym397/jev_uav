"""Construct a decision backend from command-line style options."""

from pathlib import Path


BACKENDS = ("nano", "systemone", "jevk5", "laya", "decider")


def build_backend(kind: str, *, model_id=None, model_path=None, endpoint=None, request_model=None,
                  checkpoint_dir=Path("external/NanoJev/checkpoints/NanoJev-unified"),
                  source_dir=Path("external/NanoJev"), device="cuda:0", precision="bf16"):
    from .backends import DeciderBackend, JevK5Backend, LayaBackend, NanoBackend, SystemOneHTTPBackend

    if kind == "nano":
        from agents.jev_agent import NativeNanoJevClient

        native = NativeNanoJevClient(checkpoint_dir, source_dir=source_dir, device=device, precision=precision)
        return NanoBackend(native, model_id=model_id or "C-Tianyu/NanoJev@unified-games-v1")
    if kind == "jevk5":
        from jevk5 import JevK5

        return JevK5Backend(JevK5(str(model_path), device=device), model_id=model_id or Path(model_path).name)
    if kind == "laya":
        import laya

        return LayaBackend(laya.load(str(model_path)), model_id=model_id or Path(model_path).name)
    if kind == "decider":
        from decider.infer import Decider

        if str(device).startswith("cuda:"):
            import torch

            # decider's CUDA graphs are captured on the current device; on any other GPU every call fails.
            torch.cuda.set_device(device)
        return DeciderBackend(Decider(str(model_path), device=device), model_id=model_id or Path(model_path).name)
    if kind == "systemone":
        return SystemOneHTTPBackend(endpoint, model_id=model_id, request_model=request_model)
    raise ValueError(f"Unknown backend: {kind}")
