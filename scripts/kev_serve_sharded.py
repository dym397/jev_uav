"""kev.serve for checkpoints past one card (Kev-27B, 51 GB bf16): the backbone over both GPUs and host memory.

    KEV_CUDA_GRAPHS=0 KEV_FUSED=0 python scripts/kev_serve_sharded.py --run <checkpoint> --port N

Same arguments as kev.serve; uav_eval.sharding places the layers and UAV_MAX_MEMORY sets the budgets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kev.model  # noqa: E402
import kev.serve  # noqa: E402

from uav_eval.sharding import shard  # noqa: E402

shard(kev.model, kev.model.DecisionModel)
kev.serve.main()
