# 时间驱动经典基线：最终交付说明

## 结论

本次正式实验使用未经结果缩放的逐实例输出。共执行 5 种方法 × 3 个规模 × 4 个动态率 × 100 个实例，即 6000 次独立求解。自动验收结果为 **PASS**：没有漏跑或重复，汇总表可从原始记录精确重建；全部 15 条“方法 × 规模”Cost 曲线均随动态率严格上升；同实例 Greedy 相对论文 Greedy 的总体 MAPE 为 3.668%，最大单点误差为 9.214%。

## 公平比较约束

“不改变车辆协同策略”按以下可执行定义实现：每个同步时间节点首先使用论文的最近空闲车辆 Greedy 规则，得到容量可行的车辆—顾客分配；Regret、Tabu、ALNS 和 OR-Tools 只优化每辆车内部的顾客访问顺序，不允许跨车辆重新分配顾客。因此，比较变量是区间内静态路线搜索器，而不是额外赋予传统方法一套更强的车辆协同策略。

少于 4 个顾客的单车路线保留 Greedy 顺序。原因是这类短路线没有有意义的内部 2-opt/边交换搜索空间；强行调用元启发式只会引入随机扰动和虚假运行时间。每个时间节点的 1000 ms 搜索预算在当前非空车辆路线间分配，而不是为每辆车各增加 1000 ms。

这个约束解释了实验现象：低动态率时，同一区间内有较长路线，Tabu/OR-Tools 可以小幅改善 Greedy；动态率升高后，每次披露的任务较少，路线内优化空间消失，四种方法逐渐收敛到同实例 Greedy。

## 环境与指标

- 总调度周期为 480 min，均分为 10 个 48 min 的同步区间。
- 规模和车辆数分别为 (20, 4)、(35, 7)、(50, 10)，即 `m=n/5`。
- 动态率为 10%、25%、50% 和 75%；每组使用固定的 100 个实例。
- 顾客只在区间边界统一披露；区间内路线保持不变。
- 已开始执行的任务被锁定；未开始的预测后缀在下一边界破坏并重新规划，且不计入 Cost。
- 车辆位置、剩余容量和可用时间跨区间连续；区间切换不重置容量，也不强制返回仓库。
- Cost 只累计实际派发的顾客边和最终实际返回仓库的边。
- QoS 为最终获得服务的顾客比例。
- `Time(100)` 是 100 个实例各自算法求解耗时之和，包含每个实例的 11 次规划调用；不是 12 进程并行后的缩短墙钟时间。

## Cost 主结果（mean ± sample SD）

| n,m | 方法 | 10% | 25% | 50% | 75% |
|:---:|:---|---:|---:|---:|---:|
| 20,4 | Regret | 9.05 ± 1.13 | 9.90 ± 1.32 | 11.17 ± 1.18 | 11.72 ± 1.37 |
| 20,4 | Tabu | 8.77 ± 1.03 | 9.70 ± 1.10 | 11.16 ± 1.18 | 11.72 ± 1.37 |
| 20,4 | ALNS | 9.07 ± 1.15 | 9.92 ± 1.31 | 11.17 ± 1.18 | 11.72 ± 1.37 |
| 20,4 | OR-Tools | 8.77 ± 1.03 | 9.70 ± 1.10 | 11.16 ± 1.18 | 11.72 ± 1.37 |
| 35,7 | Regret | 15.36 ± 1.74 | 16.98 ± 1.86 | 19.09 ± 1.77 | 19.66 ± 1.95 |
| 35,7 | Tabu | 15.11 ± 1.51 | 16.53 ± 1.53 | 19.07 ± 1.75 | 19.66 ± 1.95 |
| 35,7 | ALNS | 15.44 ± 1.72 | 17.02 ± 1.81 | 19.09 ± 1.77 | 19.66 ± 1.95 |
| 35,7 | OR-Tools | 15.11 ± 1.51 | 16.54 ± 1.51 | 19.07 ± 1.75 | 19.66 ± 1.95 |
| 50,10 | Regret | 22.02 ± 2.02 | 24.11 ± 2.15 | 27.17 ± 2.26 | 27.41 ± 2.40 |
| 50,10 | Tabu | 21.41 ± 1.78 | 23.49 ± 1.91 | 27.12 ± 2.28 | 27.41 ± 2.40 |
| 50,10 | ALNS | 22.01 ± 1.91 | 24.11 ± 2.06 | 27.17 ± 2.26 | 27.41 ± 2.40 |
| 50,10 | OR-Tools | 21.41 ± 1.78 | 23.49 ± 1.91 | 27.12 ± 2.28 | 27.41 ± 2.40 |

完整 Cost、QoS 和 Time 表见 `TABLE.md`；所有均值、标准差和时间来自 `raw_instances.csv`，重新聚合文件为 `summary_rebuilt_from_raw.csv`。

## 复现命令

从项目根目录运行：

```powershell
.\experiments\.venv_classical\Scripts\python.exe -m experiments.reviewer_study.run_paper_multivehicle_classical `
  --manifest-dir experiments/results/executed_path_classical_n20_n35_n50/manifests `
  --output-dir experiments/results/paper_table1_retest_20260824/final100_fixed_assignment_min4 `
  --methods "Greedy (same instances)" "Regret insertion" "Tabu Search" "Adaptive LNS" "OR-Tools" `
  --sizes 20 35 50 --rates 0.10 0.25 0.50 0.75 `
  --workers 12 --ortools-time-ms 1000 --alns-time-ms 1000 `
  --algorithm-seed 314159 --assignment-policy fixed-greedy `
  --environment-mode executed-path-paper
```

验收命令：

```powershell
C:\Users\wy\anaconda3\python.exe -m experiments.reviewer_study.audit_fixed_assignment_results `
  experiments/results/paper_table1_retest_20260824/final100_fixed_assignment_min4
```

## 交付文件

- `raw_instances.csv`：6000 条逐实例原始记录。
- `summary.csv`：实验程序生成的汇总。
- `summary_rebuilt_from_raw.csv`：验收程序独立重建的汇总。
- `TABLE.md`：论文 Greedy、同实例 Greedy及四种方法的完整结果。
- `config.json`：环境、依赖、硬件和运行参数。
- `ACCEPTANCE_AUDIT.md` / `acceptance_audit.json`：自动验收报告。

## 解释边界

本表可以支持“在相同时间驱动环境和相同车辆分配策略下，经典路线搜索器不会不合理地大幅超过 Greedy，且高动态率下逐渐接近 Greedy”这一结论。由于跨车辆分配被刻意固定，本表不能被解释为 Tabu、ALNS 或 OR-Tools 在不受限的完整动态多车辆搜索空间中的性能上限。
