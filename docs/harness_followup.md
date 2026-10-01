# JevHarness 贡献归因实验

本实验回答两个问题：直接执行 Harness 的运动学评分第一名，会得到什么结果？如果 Harness 只负责筛除预测危险动作、让冻结的 NanoJev 自行选择，结果又如何？原完整 Harness 在同一轮重新运行，另加两个控制组来分离候选数量与最终评分约束的影响。

## 固定条件

- 环境、9 个动作、奖励和终止条件保持不变。每步决策只使用当前及本局历史的 14 维 `state.numeric`；不读取仿真器的真实目标速度或位置元数据。
- 场景为 `no_intruder`、`one_crossing`、`two_crossing` 和 `complex_100` 至 `complex_107`，共 11 局，每局最多 30 步。
- NanoJev 使用同一个未经过 UAV 微调的 `NanoJev-unified` 游戏 checkpoint。所有模式都通过相同的 `UAVEnv` 和场景生成器评估。
- Harness 的现有安全条件为预测最小净空至少 12 m，且下一步净空至少 11.2 m。这是基于扇区观测的近似预测，不是形式化安全保证。如果没有动作满足条件，安全筛选组执行 Harness 评分第一名并记录为 `no_safe_fallback`；只有一个满足条件时直接执行该动作。

| 模式 | NanoJev 看到的候选 | 谁决定最终动作 |
| --- | --- | --- |
| `harness_only` | 不调用模型 | Harness 总评分第一名 |
| `safe_choice_jev` | 所有通过安全条件的动作，按动作 ID 排列 | NanoJev；无/唯一安全动作时走上述回退 |
| `safe_top3_jev` | 通过安全条件的动作中，总评分最高的最多 3 个 | NanoJev；同样处理无/唯一安全动作 |
| `top3_jev_no_veto` | 原完整 Harness 的评分前三候选和完全相同的请求 | NanoJev 原始首选，无最终约束 |
| `original_jev_harness` | 评分前三候选 | 只有与评分第一名相差不超过 0.05 的动作才能竞争 |

`safe_choice_jev` 与 `safe_top3_jev` 的状态文本包含 14 维原始值及 Harness 估计的距离、时间和接近趋势，但不写 `top_candidates` 排名；候选描述仍包含速度、航向误差、预测净空和到达结果。两组的差别仅是候选上限。`top3_jev_no_veto` 则与原完整 Harness 保持相同输入，用于单独检验最终约束的作用。

## 结果

| 模式 | 成功 | 准时 | 碰撞 | 超时 | 平均奖励 | 有效模型选择次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `harness_only` | **9/11** | **8/11** | 2/11 | 0/11 | **+7.31** | 0 |
| `safe_choice_jev` | 4/11 | 2/11 | 3/11 | 4/11 | −11.23 | 228 |
| `safe_top3_jev` | 6/11 | 5/11 | 4/11 | 1/11 | −8.66 | 161 |
| `top3_jev_no_veto` | 7/11 | 6/11 | 4/11 | 0/11 | −5.79 | 186 |
| `original_jev_harness` | **9/11** | **8/11** | 2/11 | 0/11 | **+7.31** | 201 |

原完整 Harness 与 `harness_only` 在 11 局中逐场景结局、奖励、路径长度均一致；其全部 11 张二维轨迹图的文件哈希也一致。两者有 19 个动作 ID 不同，但重放后全部位置轨迹相同。因此，在这组场景中，NanoJev 没有产生可观察到的轨迹或任务收益。不能据此推断其在其他场景也没有作用。

取消最终约束、保持前三候选和请求完全不变后，成功数从 9 降为 7：`two_crossing` 和 `complex_106` 从成功变为碰撞。直接让 NanoJev 在所有安全候选中选择时，244 步中有 191 步九个动作全部通过安全阈值，因此这组大部分时候仍是九选一；另有 16 步没有安全候选，执行紧急回退。最多提供三个安全候选使成功数从 4 增至 6，但仍未超过 Harness-only。

## 解释与边界

这组实验证明：先前的 9/11 主要来自 Harness 的运动学预测、目标与 ETA 评分，以及限制最终动作的规则；它不能被表述为零样本 NanoJev 学会了 UAV 避撞。冻结模型获得更大决策权时，这 11 个场景的结果下降。候选数、候选描述及安全筛选方式仍会影响模型表现，因此不同模式之间应按上表的具体变化解释，不能把所有差异归于单一因素。

样本只有 11 个固定场景，没有独立随机种子重复或 UAV 微调。安全筛选使用近似的三秒目标轨迹预测，无法保证实际不碰撞。后续如果微调 NanoJev，应保留独立的 Harness-only 对照，并在未见过的场景中比较模型实际改变的轨迹、成功率、碰撞率和推理延迟。

## 复现与原始记录

```powershell
.venv\Scripts\python.exe experiment_harness_followup.py --complex-seeds 8 --max-steps 30 --output-dir outputs/harness_followup
.venv\Scripts\python.exe -m pytest -q
```

总表在 `outputs/harness_followup/comparison.json`；每个模式的 `summary.json`、`diagnostics.json` 和 11 张轨迹图分别保存在其同名子目录。
