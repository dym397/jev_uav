"""Heatmaps of E1 approach / side / conflict balanced accuracy: rows models, columns state styles.

    python scripts/e1/plot_state_ba.py state_ba.json OUT_DIR

Input is scripts/e1/state_ba.py's JSON, merged over results/e1/{NanoJev,laya,laya-typed-decisions,JevK5-9B}, semantic/*
and matrix/*. Grey cells were not run; a model whose run is still in progress is left out. paper14 styles carry no
relative velocity, so their approach and conflict columns are marked with † (not identifiable from the input).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MOTION_BLIND_STYLES = ("paper14", "paper14_prose", "paper14_semantic")   # uav_eval.e1's; its package needs torch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

data = json.load(open(sys.argv[1], encoding="utf-8"))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

STYLES = ["paper14", "paper14_prose", "paper14_semantic", "kinematic_fields", "kinematic_prose",
          "third_person_polar", "first_person_polar", "first_person_clock", "first_person_world", "first_person_list",
          "first_person_derived", "third_person_semantic", "first_person_semantic"]
MODELS = ["NanoJev", "laya", "laya-typed-decisions", "decider-0.8b", "decider-2b", "Qwen3.5-4B", "Wald-4B",
          "decider-4b", "kev-4b", "CLM-v0.1-8B", "kev-9b", "JevK5-9B", "Open-Jev-9B", "imajev-9b", "JPT-9B",
          "Decision-2.0-Lux-9B", "Winnow-12B", "decider-12b", "rune-26b-a4b", "kev-27b", "Open-Jev-27B",
          "pplx-decider-v1-27b", "decider-35b-a3b", "jev-official"]
LABELS = {"NanoJev": "NanoJev-unified-games-v1"}
models = [m for m in MODELS if m in data] + sorted(set(data) - set(MODELS))
# A run in progress covers the probe families in file order (head_on first), so its partial BA is not comparable.
full = max(c["side"]["n"] for m in models for c in data[m].values())
partial = [m for m in models if min(c["side"]["n"] for c in data[m].values()) < full]
if partial:
    print("skipped, not finished:", *partial)
models = [m for m in models if m not in partial]
styles = [s for s in STYLES if any(s in data[m] for m in models)]

TITLES = {"approach": "接近判断 BA（approaching / receding）",
          "side": "左右判断 BA（left / right）",
          "conflict": "冲突判断 BA（保持航向航速 8 s 内是否进入 10 m）"}

for q, title in TITLES.items():
    full_n = max(data[m][s][q]["n"] for m in models for s in data[m])
    grid = np.full((len(models), len(styles)), np.nan)
    marks = {}
    for i, m in enumerate(models):
        for j, s in enumerate(styles):
            cell = data[m].get(s, {}).get(q)
            if cell and cell["ba"] is not None:
                grid[i, j] = cell["ba"]
                marks[i, j] = ""
    fig, ax = plt.subplots(figsize=(1.0 * len(styles) + 4, 0.42 * len(models) + 2.4))
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("#e6e6e6")   # not run
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    xlabels = [s + (" †" if q != "side" and s in MOTION_BLIND_STYLES else "") for s in styles]
    ax.set_xticks(range(len(styles)), xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(models)), [LABELS.get(m, m) for m in models])
    for (i, j), mark in marks.items():
        ax.text(j, i, f"{grid[i, j]:.2f}{mark}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("输入表达（state style）")
    ax.set_ylabel("模型")
    note = "0.5 = 随机水平；灰色 = 未跑"
    if q != "side":
        note += "；† = 输入无相对速度，不可判别"
    ax.set_title(f"{title}\n{note}（每格 n = {full_n}）", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out / f"e1_ba_{q}.png", dpi=150)
    plt.close(fig)
print("wrote", *(out / f"e1_ba_{q}.png" for q in TITLES))
