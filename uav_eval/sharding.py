"""Run a backbone too large for one card across both GPUs (and, past that, host memory) in its trained dtype.

kev and decider load their backbone with `from_pretrained(...)` and then move the whole model with `.to(device)`,
which assumes it fits on one device. `shard(module, DecisionModel)` swaps in a `from_pretrained` that places the
decoder layers over the GPUs and CPU (accelerate moves offloaded layers' weights in per pass; numbers are unchanged),
and makes the wrapper's `.to(device)` move only what the library adds on top of the backbone (heads, buffers).
A LoRA (kev adapters) is refused when layers are offloaded to CPU: the merge does not reach them.

Embeddings, the final norm, rotary tables and the LM head are all pinned to cuda:0, so a library that reads the last
hidden state or `lm_head.weight` directly finds them on the device it was told to use. Budgets come from
UAV_MAX_MEMORY ("0=20GiB,1=22GiB,cpu=100GiB"). CUDA graphs and fused kernels assume one device; turn them off.
"""

import os
import re
from collections import Counter

import torch

DEFAULT_MAX_MEMORY = "0=19GiB,1=22GiB,cpu=100GiB"   # cuda:0 also holds the embeddings, head and activations


def max_memory(spec: str | None = None) -> dict:
    spec = spec or os.environ.get("UAV_MAX_MEMORY") or DEFAULT_MAX_MEMORY
    return {(int(k) if k.strip().isdigit() else k.strip()): v.strip()
            for k, v in (part.split("=", 1) for part in spec.split(","))}


def _bytes(size: str) -> int:
    number, unit = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]i?B)\s*", size).groups()
    return int(float(number) * {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40,
                                "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}[unit])


def device_map_for(cls, name, budget: dict, **kwargs) -> dict:
    """A module -> device map from a weightless skeleton: everything but the decoder layers on cuda:0, then the
    layers in order, filling cuda:0's remaining budget, then cuda:1's, then host memory."""
    from accelerate import init_empty_weights
    from accelerate.utils import compute_module_sizes
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(name, **{k: v for k, v in kwargs.items() if k == "revision"})
    config_class = getattr(cls, "config_class", None)
    if hasattr(config, "text_config") and config_class is not None and not isinstance(config, config_class):
        config = config.text_config   # a text-only class (Qwen3_5ForCausalLM) on a multimodal checkpoint
    with init_empty_weights():   # Auto* classes build from a config; a concrete model class is called on it
        skeleton = (cls.from_config(config, dtype=kwargs.get("dtype")) if hasattr(cls, "from_config")
                    else cls(config))
    sizes = compute_module_sizes(skeleton, dtype=kwargs.get("dtype"))
    layers_name = max((n for n, m in skeleton.named_modules() if isinstance(m, torch.nn.ModuleList) and n.endswith("layers")),
                      key=lambda n: sizes[n])
    device_map = {}
    # The layers' ancestors are split into their children; every child off the path goes to cuda:0 whole.
    prefix = ""
    for part in layers_name.split("."):
        module = skeleton.get_submodule(prefix[:-1]) if prefix else skeleton
        for child, _ in module.named_children():
            if child != part:
                device_map[prefix + child] = 0
        for own, _ in [*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)]:
            device_map[prefix + own] = 0
        prefix += part + "."
    free = {device: _bytes(size) for device, size in budget.items()}
    free[0] -= sum(sizes[key] for key in device_map)
    order = [device for device in (0, 1, "cpu") if device in free]
    slot = 0
    for index in range(len(skeleton.get_submodule(layers_name))):
        key = f"{layers_name}.{index}"
        while free[order[slot]] < sizes[key]:
            if slot == len(order) - 1:
                raise ValueError(f"{name} does not fit {budget}")
            slot += 1
        free[order[slot]] -= sizes[key]
        device_map[key] = order[slot]
    return device_map


def _sharded_from_pretrained(cls, original):
    def load(name, *args, **kwargs):
        kwargs.pop("device_map", None)
        kwargs["device_map"] = device_map_for(cls, name, max_memory(), **kwargs)
        model = original(name, *args, **kwargs)
        placed = getattr(model, "hf_device_map", None) or {"": 0}
        _offloaded.append("cpu" in map(str, placed.values()))
        print(f"[uav sharding] {name}: {dict(Counter(map(str, placed.values())))}", flush=True)
        return model
    return load


_offloaded = []


def _peft_to(self, *args, **kwargs):
    # Measured on kev-4b: a LoRA merged into CPU-offloaded layers is lost (probabilities off by up to 0.2), while
    # the same model over two GPUs matches the single-card run. Full-weight checkpoints offload exactly.
    if any(_offloaded):
        raise RuntimeError("a LoRA on a CPU-offloaded backbone is not applied correctly; raise UAV_MAX_MEMORY's GPU budgets")
    return self   # only ever wrapping the dispatched backbone in this process


def _to_outside(backbone: str):
    def to(self, *args, **kwargs):
        """`.to` for a wrapper whose backbone is already dispatched: move every other child and its own buffers."""
        for child_name, child in self.named_children():
            if child_name != backbone:
                child.to(*args, **kwargs)
        for buffer_name, buffer in list(self.named_buffers(recurse=False)):
            setattr(self, buffer_name, buffer.to(*args, **kwargs))
        return self
    return to


def shard(module, decision_model_cls, loaders=("AutoModel", "AutoModelForCausalLM"), backbone="lm") -> None:
    """Patch a library module that does `<loader>.from_pretrained(...)` then `DecisionModel.to(device)`; `loaders`
    names the classes it loads the backbone through and `backbone` the wrapper's attribute holding it."""
    for attr in loaders:
        auto = getattr(module, attr, None)
        if auto is None:
            continue

        class Sharded:   # a stand-in exposing the one classmethod the library calls
            from_pretrained = staticmethod(_sharded_from_pretrained(auto, auto.from_pretrained))

        setattr(module, attr, Sharded)
    decision_model_cls.to = _to_outside(backbone)
    try:   # a LoRA loaded onto the dispatched backbone is moved the same way (kev: PeftModel.from_pretrained(...).to(device))
        from peft import PeftModel
    except ImportError:
        pass
    else:
        PeftModel.to = _peft_to
    torch.cuda.set_device(0)
