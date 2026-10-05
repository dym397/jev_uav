"""simple-jev's own hf-server on a backbone past one card (Qwen/Qwen3.8-27B in bf16, ~52 GB).

    python scripts/simplejev_serve_sharded.py --model <snapshot> [hf_server flags]

hf_server passes its --device straight to `from_pretrained(device_map=...)`; here the Auto loaders it uses get a
uav_eval.sharding map instead (layers over both GPUs, then host memory; UAV_MAX_MEMORY sets the budgets), with the
embeddings, vision tower and LM head on cuda:0, where the server puts its inputs. The prompt policy, scoring and
numerics are the server's own.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transformers  # noqa: E402

from uav_eval.sharding import device_map_for, max_memory  # noqa: E402


def sharded(loader):
    original = loader.from_pretrained

    def from_pretrained(name, *args, **kwargs):
        kwargs["device_map"] = device_map_for(loader, name, max_memory(), dtype=kwargs.get("dtype"),
                                              revision=kwargs.get("revision"))
        model = original(name, *args, **kwargs)
        print(f"[uav sharding] {name}: {dict(Counter(map(str, model.hf_device_map.values())))}", flush=True)
        return model
    return from_pretrained


for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
    loader = getattr(transformers, name)
    loader.from_pretrained = sharded(loader)

import hf_server  # noqa: E402

if __name__ == "__main__":
    hf_server.main()
