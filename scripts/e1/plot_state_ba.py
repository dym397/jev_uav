"""Heatmaps of E1 approach / side / conflict balanced accuracy: rows models, columns state styles.

    python scripts/e1/plot_state_ba.py state_ba.json OUT_DIR

Input is scripts/e1/state_ba.py's JSON. A cell whose n is below the fullest run's n for that question is a run still
in progress and is marked with * (models with an unfinished run are left out entirely); paper14 styles carry no relative velocity, so their approach and conflict columns
are marked with † (not identifiable from the input).
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

STYLES = ["paper14", "paper14_prose", "paper14_semantic", "third_person_polar", "first_person_polar",
          "first_person_clock", "first_person_world", "first_person_list", "first_person_derived"]
MODELS = ["Wald-4B", "decider-4b", "CLM-v0.1-8B", "kev-9b", "JevK5-9B", "Open-Jev-9B", "imajev-9b", "JPT-9B",
          "Decision-2.0-Lux-9B", "Winnow-12B", "decider-12b", "rune-26b-a4b", "kev-27b", "Open-Jev-27B",
          "pplx-decider-v1-27b", "jev-official"]
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
    fig, ax = plt.subplots(figsize=(1.05 * len(styles) + 3.5, 0.45 * len(models) + 2.2))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    xlabels = [s + (" †" if q != "side" and s in MOTION_BLIND_STYLES else "") for s in styles]
    ax.set_xticks(range(len(styles)), xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(models)), models)
    for (i, j), mark in marks.items():
        ax.text(j, i, f"{grid[i, j]:.2f}{mark}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("输入表达（state style）")
    ax.set_ylabel("模型")
    note = "0.5 = 随机水平"
    if q != "side":
        note += "；† = 输入无相对速度，不可判别"
    ax.set_title(f"{title}\n{note}（每格 n = {full_n}）", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out / f"e1_ba_{q}.png", dpi=150)
    plt.close(fig)
print("wrote", *(out / f"e1_ba_{q}.png" for q in TITLES))
