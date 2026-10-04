"""pplx-decider-v1-27b's own server (autojev.server, shipped in the repository's source/) past one card.

    AUTOJEV_CHECKPOINT=<snapshot> PORT=N python scripts/pplx_serve_sharded.py

The 27B backbone is ~51 GB in bf16, so uav_eval.sharding places its layers over both GPUs and host memory
(UAV_MAX_MEMORY sets the budgets); the readout head stays on cuda:0. Weights and numerics are unchanged.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(os.environ["AUTOJEV_CHECKPOINT"]) / "source" / "src"))

import autojev.model  # noqa: E402
import autojev.server  # noqa: E402

from uav_eval.sharding import shard  # noqa: E402

shard(autojev.model, autojev.model.DecisionModel, loaders=("Qwen3_5Model",), backbone="backbone")
autojev.server.main()
