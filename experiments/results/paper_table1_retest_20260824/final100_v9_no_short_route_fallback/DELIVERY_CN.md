# 时间驱动经典基线最终交付

## 1. 验收结论

- 完整评测包含 5 个方法、3 个规模、4 个动态率、每组 100 个实例，共 6000 条逐实例原始记录；无重复、无缺组。
- 同实例 Greedy 对论文 Table 1 中 12 个 Greedy Cost 的 MAPE 为 **2.180%**，最大绝对百分比误差为 **4.352%**，满足 MAPE 不超过 3%、最大误差小于 5% 的环境复现门槛。
- 5 个方法在每个规模下的平均 Cost 均随动态率 10%→25%→50%→75% 严格增加，共 15/15 条序列通过。
- 在 75% 动态率下，四种经典方法与同实例 Greedy 的最大差距为 **0.935%**；与论文 Greedy 的最大误差为 **4.663%**。低动态率下 Regret/ALNS 可产生真实的较大改进，未用惩罚、缩放或结果截断将其强行拉回 Greedy。
- 所有路线均调用其名义求解器。`short_route_fallback=null`；单顾客探针证实 Regret、Tabu、ALNS 和 OR-Tools 均实际被调用。

## 2. 实验环境

- 时间跨度 `T=480 min`，同步划分为 10 个 48 min 区间；每个区间边界执行一次完整静态多车重优化，`T=480` 再执行一次终端闭合求解，因此每实例共 11 次规划调用。
- 区间内部路径不调整；顾客只在区间边界披露。边界处保留已经开始的顾客边或返仓边，销毁尚未开始的预测后缀并重新规划。
- Cost 为所有车辆实际承诺执行的顾客边、返仓边与终端闭合边的累计归一化距离；未开始且被销毁的预测路径不计入 Cost。
- 发布代码先在 `[0,100]^2` 生成整数坐标再归一化，因此执行时间使用 `100 min / normalized distance unit`；车辆完成物理返仓后才将容量恢复为 150。
- 多车规模为 `(n,m)=(20,4),(35,7),(50,10)`。车辆—顾客分配统一使用相同的容量可行 Greedy 协同策略，四种方法只替换每辆车内部的静态路径排序，以免比较中改变车辆协同策略。
- 测试实例使用发布生成器的 Bernoulli 动态成员、Poisson 披露机制。12 组实际动态率分别为：n=20：9.20%、23.25%、51.05%、74.15%；n=35：9.97%、25.37%、49.57%、73.86%；n=50：9.96%、24.92%、49.64%、76.12%。
- CPU：Intel64 Family 6 Model 151，24 logical CPUs；Python 3.9.13；ALNS 7.0.0；OR-Tools 9.10.4067。评测用 12 个 CPU worker，完整并行墙钟时间 697.59 s。
- `Time(100)` 为该组 100 个实例从环境开始到结束的逐实例 elapsed time 之和，包含全部 11 次规划调用；不是 12 worker 并行后的缩短墙钟时间。

## 3. 完整结果

### n=20, m=4

| 动态率 | 论文 Greedy Cost | 方法 | Cost（mean±SD） | QoS %（mean±SD） | Time(100) s |
|---:|---:|:---|---:|---:|---:|
| 10% | 9.07 | Regret insertion | 8.76±1.35 | 99.95±0.50 | 0.44 |
| 10% | 9.07 | Tabu Search | 8.86±1.24 | 99.95±0.50 | 317.98 |
| 10% | 9.07 | Adaptive LNS | 8.78±1.35 | 99.95±0.50 | 336.57 |
| 10% | 9.07 | OR-Tools | 8.86±1.24 | 99.95±0.50 | 317.72 |
| 25% | 9.69 | Regret insertion | 9.65±1.24 | 100.00±0.00 | 0.33 |
| 25% | 9.69 | Tabu Search | 9.58±1.27 | 100.00±0.00 | 264.34 |
| 25% | 9.69 | Adaptive LNS | 9.64±1.26 | 100.00±0.00 | 284.57 |
| 25% | 9.69 | OR-Tools | 9.58±1.27 | 100.00±0.00 | 264.07 |
| 50% | 11.25 | Regret insertion | 11.41±1.59 | 100.00±0.00 | 0.18 |
| 50% | 11.25 | Tabu Search | 11.51±1.54 | 100.00±0.00 | 166.51 |
| 50% | 11.25 | Adaptive LNS | 11.45±1.63 | 100.00±0.00 | 186.08 |
| 50% | 11.25 | OR-Tools | 11.51±1.54 | 100.00±0.00 | 165.97 |
| 75% | 12.43 | Regret insertion | 12.99±1.61 | 100.00±0.00 | 0.13 |
| 75% | 12.43 | Tabu Search | 12.89±1.53 | 100.00±0.00 | 55.97 |
| 75% | 12.43 | Adaptive LNS | 13.01±1.61 | 100.00±0.00 | 61.34 |
| 75% | 12.43 | OR-Tools | 12.89±1.53 | 100.00±0.00 | 55.48 |

### n=35, m=7

| 动态率 | 论文 Greedy Cost | 方法 | Cost（mean±SD） | QoS %（mean±SD） | Time(100) s |
|---:|---:|:---|---:|---:|---:|
| 10% | 15.95 | Regret insertion | 14.52±2.10 | 100.00±0.00 | 0.93 |
| 10% | 15.95 | Tabu Search | 14.88±2.05 | 100.00±0.00 | 296.41 |
| 10% | 15.95 | Adaptive LNS | 14.44±2.02 | 100.00±0.00 | 310.53 |
| 10% | 15.95 | OR-Tools | 14.88±2.05 | 100.00±0.00 | 296.17 |
| 25% | 16.96 | Regret insertion | 16.29±2.37 | 100.00±0.00 | 0.61 |
| 25% | 16.96 | Tabu Search | 16.41±2.14 | 100.00±0.00 | 258.99 |
| 25% | 16.96 | Adaptive LNS | 16.33±2.36 | 100.00±0.00 | 271.57 |
| 25% | 16.96 | OR-Tools | 16.41±2.14 | 100.00±0.00 | 258.55 |
| 50% | 19.63 | Regret insertion | 19.59±2.61 | 100.00±0.00 | 0.34 |
| 50% | 19.63 | Tabu Search | 19.38±2.43 | 100.00±0.00 | 162.42 |
| 50% | 19.63 | Adaptive LNS | 19.51±2.59 | 100.00±0.00 | 177.88 |
| 50% | 19.63 | OR-Tools | 19.38±2.43 | 100.00±0.00 | 161.55 |
| 75% | 21.55 | Regret insertion | 22.43±2.50 | 100.00±0.00 | 0.23 |
| 75% | 21.55 | Tabu Search | 22.43±2.50 | 100.00±0.00 | 46.28 |
| 75% | 21.55 | Adaptive LNS | 22.41±2.57 | 100.00±0.00 | 45.04 |
| 75% | 21.55 | OR-Tools | 22.43±2.50 | 100.00±0.00 | 45.67 |

### n=50, m=10

| 动态率 | 论文 Greedy Cost | 方法 | Cost（mean±SD） | QoS %（mean±SD） | Time(100) s |
|---:|---:|:---|---:|---:|---:|
| 10% | 21.41 | Regret insertion | 19.79±2.68 | 100.00±0.00 | 1.89 |
| 10% | 21.41 | Tabu Search | 20.82±2.46 | 100.00±0.00 | 303.79 |
| 10% | 21.41 | Adaptive LNS | 19.78±2.70 | 100.00±0.00 | 310.81 |
| 10% | 21.41 | OR-Tools | 20.84±2.43 | 100.00±0.00 | 303.49 |
| 25% | 22.84 | Regret insertion | 23.07±2.76 | 100.00±0.00 | 1.40 |
| 25% | 22.84 | Tabu Search | 23.34±2.58 | 100.00±0.00 | 253.55 |
| 25% | 22.84 | Adaptive LNS | 23.04±2.79 | 100.00±0.00 | 264.60 |
| 25% | 22.84 | OR-Tools | 23.34±2.58 | 100.00±0.00 | 253.22 |
| 50% | 26.71 | Regret insertion | 26.98±3.05 | 100.00±0.00 | 0.56 |
| 50% | 26.71 | Tabu Search | 27.04±2.60 | 100.00±0.00 | 165.34 |
| 50% | 26.71 | Adaptive LNS | 26.97±2.98 | 100.00±0.00 | 170.99 |
| 50% | 26.71 | OR-Tools | 27.04±2.60 | 100.00±0.00 | 163.92 |
| 75% | 30.19 | Regret insertion | 31.40±3.29 | 100.00±0.00 | 0.30 |
| 75% | 30.19 | Tabu Search | 31.27±3.06 | 100.00±0.00 | 35.83 |
| 75% | 30.19 | Adaptive LNS | 31.41±3.35 | 100.00±0.00 | 34.90 |
| 75% | 30.19 | OR-Tools | 31.27±3.06 | 100.00±0.00 | 35.12 |

## 4. 误差门槛的正确解释

MAPE≤3%、最大误差<5%用于检验“新环境中的同实例 Greedy 是否复现论文环境”，该门槛已经通过。经典优化器的 Cost 不是论文 Greedy 的重复测量：低动态率时其真实优化收益不能被当成复现误差，更不能为了通过门槛而人为降低算法质量。本次保留真实结果；到 75% 动态率时四种方法已经自然收敛到 Greedy 附近。

## 5. 复现与审计命令

```powershell
.\experiments\.venv_classical\Scripts\python.exe -m experiments.reviewer_study.run_paper_multivehicle_classical `
  --manifest-dir experiments/results/paper_table1_retest_20260824/manifests_release_seed1234 `
  --output-dir experiments/results/paper_table1_retest_20260824/final100_v9_no_short_route_fallback `
  --methods "Greedy (same instances)" "Regret insertion" "Tabu Search" "Adaptive LNS" "OR-Tools" `
  --sizes 20 35 50 --rates 0.10 0.25 0.50 0.75 --workers 12 `
  --ortools-time-ms 1000 --alns-time-ms 1000 --algorithm-seed 314159 `
  --assignment-policy fixed-greedy --environment-mode executed-path-paper

C:\Users\wy\anaconda3\python.exe -m experiments.reviewer_study.audit_fixed_assignment_results `
  experiments/results/paper_table1_retest_20260824/final100_v9_no_short_route_fallback

.\experiments\.venv_classical\Scripts\python.exe -m `
  experiments.reviewer_study.verify_no_short_route_fallback `
  --output experiments/results/paper_table1_retest_20260824/final100_v9_no_short_route_fallback/NO_SHORT_ROUTE_AUDIT.json
```

