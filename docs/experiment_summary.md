# NanoJev 替换 D3QN 做 UAV 避撞决策：实验总结

## 1. 背景与目标

本项目尝试用 NanoJev 替换 D3QN，作为无人机避撞的决策核心。D3QN 基线复现自 Li et al. (Drones 2023)。

两者使用相同的接口：

- 输入：14 维观测 `state.numeric`
  - `heading_rad`：航向（rad）
  - `speed_mps`：速度（m/s）
  - `goal_distance_ratio`：目标距离 / 计划航段长度
  - `goal_bearing_ccw_rad`：目标相对航向的方位角（逆时针，rad）
  - `eta_remaining_ratio`：剩余 ETA / 计划航段时间
  - `sector_0..8`：9 个扇区内最近入侵机的距离 / 100 m（扇区 0 为正前方，每 40° 逆时针一个扇区，1 表示 100 m 内无入侵机）
- 输出：9 个离散动作

| ID | 动作 | ID | 动作 | ID | 动作 |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | 右转减速 | 3 | 减速 | 6 | 左转减速 |
| 1 | 右转 | 4 | 保持 | 7 | 左转 |
| 2 | 右转加速 | 5 | 加速 | 8 | 左转加速 |

转向角速度 ±6°/s，加速度 ±3 m/s²，碰撞距离 10 m，探测半径 100 m，步长 1 s。

## 2. 共同实验条件

- NanoJev 模型：`C-Tianyu/NanoJev`，revision `unified-games-v1`，代码 commit `76fdfc9ecdca45a9bcef17991a07d3041a87685a`。
- 这个 checkpoint 只在 Maze、Snake 和射击游戏上训练过。所有实验都是零样本，没有使用任何 UAV 数据训练或微调。
- NanoJev 通过原生 `choice` 接口选择动作（非生成式）。
- D3QN 对照：1000 episode 训练得到的 checkpoint（`outputs/baseline_1000/best.pt`）。
- 每局最多 30 步。
- 硬件：RTX 3060 Laptop GPU，6 GB 显存。
- 延迟统计不含模型加载时间。

场景集：

| 名称 | 内容 | 使用的实验 |
| --- | --- | --- |
| 固定场景 | `no_intruder`、`one_crossing`、`two_crossing`，共享 100 m 航段 | 全部 |
| complex 100–107 | 8 个随机生成的复杂场景 | 实验 2、4、5 |
| complex 200–207 | 8 个新种子的复杂场景 | 实验 3 |
| receding 对照 | 与 `one_crossing` 相同位置、入侵机反向运动 | 实验 3 |

## 3. 实验 1：零样本 pilot

详细记录：[nanojev_pilot.md](nanojev_pilot.md)，原始数据：`outputs/nanojev_pilot/`

**问题**：不做任何修改，冻结的 NanoJev 能否直接用 14 维观测做 UAV 避撞？

**设置**：把 D3QN 使用的 14 个数值原样传给 NanoJev 的 `choice` 问题，候选为同样的 9 个动作。只用 3 个固定场景。另设一个只执行"保持"的对照策略：无入侵机时能到达目标，两个交叉场景都在第 8 步碰撞。

| 策略 | 无入侵机 | 一架交叉 | 两架交叉 | 无效选择 | 决策延迟 P50 |
| --- | --- | --- | --- | ---: | ---: |
| NanoJev 零样本 | 超时（30 步） | 超时（30 步） | 成功（11 步） | 0/71 | 211–227 ms |
| D3QN | 成功（11 步） | 成功（11 步） | 成功（11 步） | 0/33 | 0.28–0.56 ms |

**观察**：
- 在空场景里，NanoJev 反复选择"右转减速"，从目标下方飞过。
- NanoJev 的输出始终是合法动作，问题在于选择本身不合理。
- 延迟约为 D3QN 的 400–800 倍。

**结论**：零样本迁移失败。3 个场景太少，不能用来估计成功率。

## 4. 实验 2：状态文本表述消融

详细记录：[semantic_ablation.md](semantic_ablation.md)，原始数据：`outputs/semantic_ablation/`

**问题**：只改变 14 维观测转成文本的方式，能否改变 NanoJev 的零样本表现？

**设置**：checkpoint、问题指令、9 个动作描述、环境参数和初始场景都固定。唯一变量是 14 个数值的文本写法。不增加任何新信息（入侵机速度、位置、历史等）。11 个场景：3 个固定场景 + complex 100–107。

| 写法 | 说明 | `one_crossing` 初始状态 token 数 |
| --- | --- | ---: |
| `labeled` | 字段名 + 数值 | 92 |
| `semantic` | 每个字段和 9 个扇区都用完整自然语言描述 | 346 |
| `compact` | 完整带标签的数值 + 对目标和有入侵机扇区的简短解读 | 169 |

| 输入写法 | 成功 | 碰撞 | 超时 | 有效决策 | 决策延迟 P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `labeled` | 3/11 | 5/11 | 3/11 | 156/156 | 201–226 ms |
| `semantic` | 0/11 | 6/11 | 5/11 | 203/203 | 399–456 ms |
| `compact` | 5/11 | 5/11 | 1/11 | 119/119 | 243–385 ms |
| D3QN（参照） | 5/11 | 6/11 | 0/11 | 91/91 | 0.26–0.43 ms |

**观察**：
- `compact` 保留了 `labeled` 的 3 次成功，另外在 `no_intruder` 和 `complex_100` 上成功。但仍有 5 次碰撞。
- `semantic` 一次都没成功。它的输入长度约为 `labeled` 的 3.8 倍，延迟也约翻倍。实验没有把长度和措辞的影响分开。

**结论**：模型行为对措辞很敏感，但没有一种写法带来可靠的避撞能力。最好的 `compact` 也只和 D3QN 持平，碰撞数没有减少。

## 5. 实验 3：入侵机运动信息消融

详细记录：[motion_representation_experiment.md](motion_representation_experiment.md)，原始数据：`outputs/motion_ablation/`

**问题**：14 维观测里没有入侵机的运动方向。补上这些信息后，NanoJev 能否利用？

**设置**：在 14 维之外，补充 100 m 探测范围内每架入侵机在本机机体坐标系下的相对位置（前/左）和相对速度。这些数据直接取自仿真器，是完美观测假设，不代表真实的非合作跟踪系统。不提供未来轨迹、碰撞标签、专家动作或历史。D3QN 仍只使用原来的 14 维。

| 组别 | 输入 |
| --- | --- |
| `labeled` | 只有原始 14 维 |
| `flat_contacts` | 14 维 + 每架入侵机的位置和速度，按键值对平铺 |
| `structured_contacts` | 与 `flat_contacts` 数值完全相同，按入侵机分组简要描述 |

### 5.1 反事实测试

同一位置放一架入侵机，速度分别为 (0, −4) 和 (0, +4) m/s，一个接近、一个远离。两种情况下初始 14 维观测完全相同。入侵机 x 取 930/940/950/960 m，y 取 1010/1040 m，共 8 组几何。用总变差距离（TVD）比较两个 9 动作概率分布。

| 输入 | 最优动作改变的组数 | 平均 TVD |
| --- | ---: | ---: |
| `labeled` | 0/8 | 0 |
| `flat_contacts` | 0/8 | 0.0090 |
| `structured_contacts` | 1/8 | 0.0089 |

- 原始 (950, 1040) 这一组：三种输入在两个运动方向下都选择"加速"。TVD 分别为 0、0.0084、0.0095。
- 唯一的动作变化在 (940, 1040)，接近时选"加速"，远离时选"保持"。这并不能说明哪个更安全。
- 概率分布只有不到 1% 的变化，模型几乎没有对运动方向作出反应。

### 5.2 闭环测试

12 个场景：3 个固定场景 + receding 对照 + complex 200–207。

| 输入 | 成功 | 碰撞 | 超时 | 有效决策 | 决策延迟 P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `labeled` | 3/12 | 5/12 | 4/12 | 183/183 | 203–229 ms |
| `flat_contacts` | 2/12 | 5/12 | 5/12 | 196/196 | 223–625 ms |
| `structured_contacts` | 3/12 | 5/12 | 4/12 | 177/177 | 229–584 ms |
| D3QN（参照） | 8/12 | 4/12 | 0/12 | 110/110 | 0.27–0.76 ms |

**结论**：补充运动信息没有带来闭环提升，模型基本没有利用这些信息。增加原始特征或更多文字都没有解决迁移问题。注意 D3QN 是专门为 UAV 避撞训练的，这不是公平的训练对比。

## 6. 实验 4：JevHarness 消融

详细记录：[harness_ablation.md](harness_ablation.md)，代码：[jev_harness.py](../agents/jev_harness.py)，原始数据：`outputs/harness_ablation/`

**问题**：在环境和 NanoJev 之间加一层结构化的 Harness，且严格只读取 14 维观测，能否得到安全、准时的避撞决策？

**Harness 做了什么**：
1. 只读 14 维：不读取仿真器的隐藏字段（本机位置、目标、截止时间、入侵机位置等），有单元测试 `test_harness_strictly_uses_only_fourteen_numeric_values` 验证。
2. 本局记忆：把目标方位角展开到 (−π, π]；用上一步的速度和航向积分本机位移；跨步匹配有入侵机的扇区（允许 8 m 误差），估计各扇区的接近速度和入侵机径向速度。
3. 候选投影：对 9 个动作做 3 秒运动学预测，剪到评分最高的 3 个（NanoJev 训练时 K ∈ {2, 3, 4}）。
4. 多问题推理：一次批量前向同时问三个问题，即选动作（`action`）、威胁等级（`threat_level`，0–3）、是否安全（`safety_margin`）。
5. 安全约束：只有与 Harness 评分第一名相差不超过 0.05 的候选才能被选中。

**组别**：

| 组别 | 说明 |
| --- | --- |
| `d3qn` | 1000 episode checkpoint |
| `labeled` | 原始 14 维文本，K = 9 |
| `compact` | 简短解读，K = 9 |
| `harness_unshielded` | 文本里加入 Harness 的运动学投影，但不剪枝、不约束，K = 9 |
| `jev_harness` | 完整 Harness，K = 3 |

### 6.1 汇总结果（11 个场景）

| 组别 | 成功 | 准时 | 碰撞 | 超时 | 平均奖励 | 有效决策 | 决策延迟 P50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d3qn` | 5/11 | 5/11 | 6/11 | 0/11 | −13.63 | 91/91 | 0.26–0.43 ms |
| `labeled` | 3/11 | 3/11 | 5/11 | 3/11 | −23.88 | 156/156 | 201–226 ms |
| `compact` | 5/11 | 5/11 | 5/11 | 1/11 | −13.75 | 119/119 | 243–385 ms |
| `harness_unshielded` | 3/11 | 3/11 | 8/11 | 0/11 | −31.52 | 144/144 | 440–616 ms |
| `jev_harness` | 9/11 | 8/11 | 2/11 | 0/11 | +7.31 | 201/201 | 260–536 ms |

### 6.2 逐场景结果

格式：结局（步数），准时/迟到仅对成功标注。

| 场景 | `d3qn` | `labeled` | `compact` | `harness_unshielded` | `jev_harness` |
| --- | --- | --- | --- | --- | --- |
| `no_intruder` | 成功（11，准时） | 超时（30） | 成功（11，准时） | 成功（16，准时） | 成功（12，准时） |
| `one_crossing` | 成功（11，准时） | 超时（30） | 超时（30） | 碰撞（8） | 成功（22，准时） |
| `two_crossing` | 成功（11，准时） | 成功（11，准时） | 成功（12，准时） | 碰撞（10） | 成功（24，准时） |
| `complex_100` | 碰撞（8） | 碰撞（8） | 成功（11，准时） | 碰撞（11） | 成功（19，准时） |
| `complex_101` | 碰撞（6） | 超时（30） | 碰撞（6） | 碰撞（18） | 成功（30，迟到） |
| `complex_102` | 碰撞（3） | 碰撞（2） | 碰撞（3） | 碰撞（4） | 碰撞（3） |
| `complex_103` | 碰撞（8） | 碰撞（8） | 碰撞（8） | 成功（22，准时） | 成功（17，准时） |
| `complex_104` | 成功（11，准时） | 成功（11，准时） | 成功（11，准时） | 碰撞（9） | 成功（18，准时） |
| `complex_105` | 碰撞（5） | 碰撞（9） | 碰撞（8） | 碰撞（7） | 碰撞（8） |
| `complex_106` | 成功（11，准时） | 成功（11，准时） | 成功（12，准时） | 成功（25，准时） | 成功（26，准时） |
| `complex_107` | 碰撞（6） | 碰撞（6） | 碰撞（7） | 碰撞（14） | 成功（22，准时） |

`complex_102` 在 ±6°/s 转向限制下不存在 3 步内无碰撞的轨迹，所有策略都在前 4 步碰撞。

### 6.3 观察

- 只把运动学投影写进文本（`harness_unshielded`）反而最差：3/11，碰撞 8 次。说明信息写进提示词，模型并不会用。
- 剪枝加约束后跳到 9/11，比 D3QN 多 4 个场景。
- `threat_level` 输出在 1.52–1.62 之间，`safety_margin` 概率在 0.61–0.68 之间，基本不随状态变化，没有校准。
- 当时的解读是 NanoJev + Harness 超过了 D3QN。实验 5 检验了这个解读。

## 7. 实验 5：JevHarness 贡献归因

详细记录：[harness_followup.md](harness_followup.md)，原始数据：`outputs/harness_followup/`

**问题**：实验 4 的 9/11 有多少来自 NanoJev，有多少来自 Harness？

**设置**：同样的 11 个场景和 30 步上限。安全条件为预测最小净空 ≥ 12 m 且下一步净空 ≥ 11.2 m。这是基于扇区观测的近似预测，不是形式化安全保证。没有安全动作时执行 Harness 评分第一名（记为 `no_safe_fallback`）；只有一个安全动作时直接执行。

| 组别 | NanoJev 看到的候选 | 谁决定最终动作 |
| --- | --- | --- |
| `harness_only` | 不调用模型 | Harness 评分第一名 |
| `safe_choice_jev` | 所有安全动作，按 ID 排列 | NanoJev |
| `safe_top3_jev` | 安全动作中评分最高的最多 3 个 | NanoJev |
| `top3_jev_no_veto` | 与完整 Harness 相同的前 3 候选和请求 | NanoJev 首选，无最终约束 |
| `original_jev_harness` | 评分前 3 | 只有与第一名相差 ≤ 0.05 的动作可选 |

`safe_choice_jev` 和 `safe_top3_jev` 的状态文本不写候选排名，两者只差候选上限。

### 7.1 结果

| 组别 | 成功 | 准时 | 碰撞 | 超时 | 平均奖励 | 有效模型选择次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `harness_only` | 9/11 | 8/11 | 2/11 | 0/11 | +7.31 | 0 |
| `safe_choice_jev` | 4/11 | 2/11 | 3/11 | 4/11 | −11.23 | 228 |
| `safe_top3_jev` | 6/11 | 5/11 | 4/11 | 1/11 | −8.66 | 161 |
| `top3_jev_no_veto` | 7/11 | 6/11 | 4/11 | 0/11 | −5.79 | 186 |
| `original_jev_harness` | 9/11 | 8/11 | 2/11 | 0/11 | +7.31 | 201 |

### 7.2 观察

- `original_jev_harness` 与 `harness_only` 在 11 个场景中的结局、奖励、路径长度完全一致，11 张轨迹图的文件哈希也一致。两者有 19 个动作 ID 不同，但重放后位置轨迹相同。
- 去掉最终约束（`top3_jev_no_veto`）后，`two_crossing` 和 `complex_106` 从成功变为碰撞，9 → 7。
- `safe_choice_jev` 中，244 步里有 191 步 9 个动作全部通过安全阈值，所以大部分时候仍是九选一；另有 16 步没有安全候选，走了回退。
- 把候选限制到最多 3 个安全动作，4 → 6，但仍低于 `harness_only`。
- 给 NanoJev 的决策权越大，结果越差：9 → 7 → 6 → 4。

**结论**：实验 4 的 9/11 完全来自 Harness 的运动学预测、目标与 ETA 评分，以及限制最终动作的规则。这组场景里，NanoJev 没有产生任何可观察的轨迹或任务收益。不能表述为"零样本 NanoJev 学会了 UAV 避撞"。

## 8. 总览

| 实验 | 场景数 | 最好的 NanoJev 组 | NanoJev 成功 | D3QN 成功 | 提升来源 |
| --- | ---: | --- | ---: | ---: | --- |
| 1. 零样本 pilot | 3 | 原始 14 维 | 1/3 | 3/3 | — |
| 2. 文本表述 | 11 | `compact` | 5/11 | 5/11 | 措辞变化 |
| 3. 运动信息 | 12 | `labeled` / `structured_contacts` | 3/12 | 8/12 | 无 |
| 4. Harness 消融 | 11 | `jev_harness` | 9/11 | 5/11 | Harness |
| 5. Harness 归因 | 11 | `original_jev_harness` | 9/11 | — | Harness（`harness_only` 同为 9/11） |

## 9. 结论

1. 冻结的 `unified-games-v1` checkpoint 零样本迁移到 UAV 避撞失败。
2. 改变文本写法会显著改变模型的选择（0/11 到 5/11），但没有一种写法带来可靠的避撞。最好的结果只和 D3QN 持平，碰撞数没有减少。
3. 补充入侵机运动信息后，模型对运动方向几乎没有反应（8 组里仅 1 组改变动作，TVD < 0.01），闭环没有提升。
4. JevHarness 的 9/11 完全来自 Harness 本身。`harness_only` 不调用模型就能达到相同结果和相同轨迹。
5. NanoJev 每步延迟 200–600 ms，D3QN 不到 1 ms。
6. NanoJev 的威胁等级和安全裕度输出未校准，基本不随状态变化。

这些实验只检验了冻结的游戏 checkpoint 在零样本条件下的表现。它们不能说明 NanoJev 在 UAV 数据上微调之后是否能学会避撞。

## 10. 局限

- 样本量只有 3–12 个场景，没有独立随机种子重复，不足以支持成功率或碰撞率的统计结论。5/11 与 3/11 这类差别本身不一定可靠。
- complex 100–107 在调整提示词和 Harness 时已反复使用，不能再作为最终测试集。
- 实验 3 的入侵机运动信息是完美观测假设。
- Harness 的安全条件基于扇区观测的 3 秒近似预测，不保证实际不碰撞。
- D3QN 为 UAV 避撞专门训练，NanoJev 没有，两者不是匹配的训练对比。
- 30 步上限较短，`complex_101` 这类需要较长绕行的场景可能受影响。

## 11. 下一步

1. 构建 UAV 监督数据：用 `harness_only` 或 D3QN 生成专家轨迹，转成 NanoJev `choice` 格式，保持 14 维输入和 9 个动作。
2. 在 6 GB 显存限制下微调 NanoJev（可能需要 LoRA 或冻结部分层）。
3. 用新的、未使用过的场景种子作为测试集，每个条件多个随机种子重复。
4. 对照组保留 D3QN 和 `harness_only`，比较成功率、碰撞率、准时率、NanoJev 实际改变的轨迹以及推理延迟。
5. 在 UAV 结果上校准 `threat_level` 和 `safety_margin` 输出。

## 12. 复现命令

```powershell
.venv\Scripts\python.exe pilot_jev.py --d3qn-checkpoint outputs\baseline_1000\best.pt --output-dir outputs\nanojev_pilot
.venv\Scripts\python.exe experiment_semantic.py --complex-seeds 8 --output-dir outputs\semantic_ablation
.venv\Scripts\python.exe experiment_motion.py --complex-seeds 8 --output-dir outputs\motion_ablation
.venv\Scripts\python.exe experiment_motion_grid.py --output outputs\motion_ablation\counterfactual_grid.json
.venv\Scripts\python.exe experiment_harness.py --complex-seeds 8 --output-dir outputs\harness_ablation
.venv\Scripts\python.exe experiment_harness_followup.py --complex-seeds 8 --max-steps 30 --output-dir outputs\harness_followup
.venv\Scripts\python.exe -m pytest -q
```
