"""jebadiah-27b through its own standalone server (jebadiah-serve, the repository's server/) past one card.

    python scripts/jebadiah_serve_sharded.py --model <snapshot> --port N

The server loads the backbone with `device_map=<one device>`; the 27B is ~52 GB in bf16, so its `load_base` is given
a uav_eval.sharding map instead (layers over both GPUs, then host memory; UAV_MAX_MEMORY sets the budgets). The
embeddings, final norm and LM head stay on cuda:0, where the server's Scorer puts the inputs and reads the fp32
candidate logits. Prompt, contract check, temperatures and numerics are the server's own.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jebadiah_server.model_scripts  # noqa: E402,F401 - puts the model scripts on sys.path
import jebadiah_model  # noqa: E402
from jebadiah_server.cli import main  # noqa: E402

from uav_eval.sharding import device_map_for, max_memory  # noqa: E402


def load_base(base, revision=None, attn_implementation="sdpa", dtype=None, device="cuda"):
    cls = jebadiah_model.model_class(base, revision)
    device_map = device_map_for(cls, base, max_memory(), dtype=dtype, revision=revision)
    model = cls.from_pretrained(base, revision=revision, dtype=dtype, device_map=device_map,
                                attn_implementation=attn_implementation)
    model.config.use_cache = False
    print(f"[uav sharding] {base}: {dict(Counter(map(str, model.hf_device_map.values())))}", flush=True)
    return model


jebadiah_model.load_base = load_base

if __name__ == "__main__":
    main()
