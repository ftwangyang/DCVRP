# Early / Uniform / Late 客户披露实验：结果与审计

## 实验设置

- 问题规模：20 个顾客、4 辆车、10 个时间区间。
- 动态率：10%、25%、50%、75%。
- 测试规模：每个动态率 100 个实例。
- 本地测试随机种子：20260821。
- DVNDA 模型：`best_accepted.pt`，epoch 10；在 CUDA GPU 上推理。
- Early 与 Late 采用严格配对设计：同一实例中的坐标、需求、服务时间、动态顾客身份和原始披露次序完全相同，只改变披露区间。
- Early：每个动态顾客相对原 Uniform 披露区间提前 1 个时间区间，最早限制在第 1 个动态披露边界。
- Late：每个动态顾客相对原 Uniform 披露区间推迟 1 个时间区间，最晚限制在第 9 个披露边界。
- Uniform：按要求直接使用论文 Table I 的均值和标准差，没有重新测试，因此不能与本地 Early/Late 进行逐实例配对显著性检验。
- Cost：所有车辆在完整调度周期内实际执行路径的累计行驶距离；被破坏且未执行的预测路径不计入 Cost。
- QoS：被完成服务的顾客数占全部顾客数的百分比。

## 汇总结果

表中本地 Early/Late 结果写为“均值 ± 样本标准差”；Uniform 为论文结果。论文没有提供 QoS 标准差，故 Uniform QoS 仅列均值。

| 动态率 | 披露模式 | Greedy Cost ↓ | DVNDA Cost ↓ | Greedy QoS (%) ↑ | DVNDA QoS (%) ↑ |
|---:|:---|---:|---:|---:|---:|
| 10% | Early | 9.05 ± 1.02 | **8.55 ± 1.09** | 99.65 ± 1.47 | **99.90 ± 0.70** |
| 10% | Uniform | 9.07 ± 1.12 | **8.31 ± 1.22** | 99.90 | **100.00** |
| 10% | Late | 9.15 ± 1.02 | **8.59 ± 1.09** | 99.70 ± 1.39 | **99.85 ± 0.86** |
| 25% | Early | 9.71 ± 1.14 | **9.27 ± 1.14** | 99.75 ± 1.10 | **99.85 ± 0.86** |
| 25% | Uniform | 9.69 ± 1.25 | **8.95 ± 1.30** | 99.90 | **100.00** |
| 25% | Late | 9.76 ± 1.19 | **9.27 ± 1.15** | 99.65 ± 1.28 | **99.85 ± 0.86** |
| 50% | Early | 11.13 ± 1.25 | **10.18 ± 1.13** | 99.75 ± 1.31 | **99.95 ± 0.50** |
| 50% | Uniform | 11.25 ± 1.43 | **10.47 ± 1.53** | 99.90 | **100.00** |
| 50% | Late | 10.99 ± 1.30 | **10.23 ± 1.20** | 99.70 ± 1.19 | **99.90 ± 0.70** |
| 75% | Early | 11.60 ± 1.31 | **11.08 ± 1.31** | 99.85 ± 0.86 | 99.85 ± 0.86 |
| 75% | Uniform | 12.43 ± 1.51 | **11.78 ± 1.44** | 99.45 | **100.00** |
| 75% | Late | **11.25 ± 1.35** | 11.32 ± 1.32 | 99.75 ± 1.10 | **99.90 ± 0.70** |

## 配对统计检验

对本地 Early/Late 结果使用双侧配对 t 检验。`DVNDA − Greedy` 小于 0 表示 DVNDA 的 Cost 更低。

| 动态率 | 模式 | DVNDA − Greedy | p 值 | 解释 |
|---:|:---|---:|---:|:---|
| 10% | Early | −0.494 | <0.001 | DVNDA 显著更低 |
| 10% | Late | −0.554 | <0.001 | DVNDA 显著更低 |
| 25% | Early | −0.448 | <0.001 | DVNDA 显著更低 |
| 25% | Late | −0.485 | <0.001 | DVNDA 显著更低 |
| 50% | Early | −0.944 | <0.001 | DVNDA 显著更低 |
| 50% | Late | −0.764 | <0.001 | DVNDA 显著更低 |
| 75% | Early | −0.514 | <0.001 | DVNDA 显著更低 |
| 75% | Late | +0.070 | 0.575 | 两者没有显著差异 |

Late 相对 Early 的配对 Cost 变化如下：

| 动态率 | Greedy: Late − Early | p 值 | DVNDA: Late − Early | p 值 |
|---:|---:|---:|---:|---:|
| 10% | +0.099 | 0.023 | +0.040 | 0.296 |
| 25% | +0.044 | 0.568 | +0.007 | 0.903 |
| 50% | −0.136 | 0.113 | +0.043 | 0.610 |
| 75% | −0.352 | 0.001 | +0.231 | 0.011 |

## 可以由数据支持的结论

1. DVNDA 在 8 个本地配对条件中的 7 个条件下显著优于 Greedy；在 75%-Late 条件下，DVNDA 与 Greedy 的差值仅为 0.070，且不显著（p=0.575）。因此，严谨表述应为“DVNDA 在绝大多数披露条件下显著优于 Greedy，并在最困难的 75%-Late 条件下与 Greedy 相当”，不能表述为 12 个表格单元中全部严格优于。
2. DVNDA 的 QoS 在全部 Early/Late 条件下保持在 99.85%–99.95%，说明模型对披露时间变化具有较稳定的服务完成率。
3. 对 DVNDA 而言，75% 动态率下 Late 相对 Early 的 Cost 增幅为 0.231，且具有统计显著性（p=0.011）；在较低动态率下该差异很小。这支持“高动态率下时间分布影响更明显”这一较弱而准确的结论。

## 不能由当前 Cost 定义支持的结论

当前结果不支持“客户披露越晚，所有方法的纯路径 Cost 都必然越高”。Greedy 在 50% 和 75% 动态率下的 Late Cost 反而低于 Early，75% 时差异显著。这不是通过继续挑选随机种子就应该消除的误差，而是纯实际行驶距离指标本身不具备单调性：披露时间改变会改变最近邻访问次序；较晚披露也可能减少绕行，因而得到更短的几何路径。若个别任务未完成，只统计已执行距离还会进一步降低 Cost。

因此，不应把预期趋势硬编码到环境、按结果筛选测试种子或反复调参直到满足结论。若论文必须检验“晚披露增加调度难度”，建议预先定义并报告一个额外指标，例如总目标值 `实际距离 + 等待/迟到惩罚 + 未服务惩罚`、平均响应时间或任务迟到率，同时保留当前纯路径 Cost。只有在指标定义修改后重新执行所有三种模式，才能严格检验晚披露带来的综合代价。

## 推荐的论文结果表述

> The temporal distribution of dynamic requests affects both solution quality and service completion. Across the paired Early and Late tests, DVNDA significantly reduces travel cost relative to Greedy in seven of the eight conditions, while remaining statistically comparable under the most dynamic 75%-Late condition. DVNDA maintains a QoS of 99.85%–99.95% across all tested patterns. The influence of revelation timing is most pronounced at the 75% dynamic ratio for DVNDA. Because the reported Cost measures only executed travel distance, later revelation does not necessarily produce a monotonically longer route; it may alter the visiting order and reduce geometric detours. Thus, the experiment demonstrates robust performance under different temporal evolution patterns, but does not support a universal Late > Uniform > Early ordering of pure route distance.

