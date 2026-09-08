# Dynamic Capacitated Vehicle Routing Problem (DCVRP) via Deep Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)
[![Table I Reproduction: Verified](https://img.shields.io/badge/Table%20I%20Reproduction-Passed%20(MAPE%201.52%25)-brightgreen.svg)](#-experimental-results--table-i-reproduction)

Official PyTorch implementation of **DVNDA** (Dual-Attention Vehicle-Node Dynamic Attention Network) and neural combinatorial optimization (NCO) baselines for the **Dynamic Capacitated Vehicle Routing Problem (DCVRP)**.

---

## 📖 Overview

The **Dynamic Capacitated Vehicle Routing Problem (DCVRP)** models real-world logistics where customer requests arrive dynamically over a planning horizon $T = 480$ minutes. A fleet of $m$ capacitated vehicles must coordinate to service static requests known at $t = 0$ while adaptively dispatching to dynamic requests revealed over time $\tau_i \in (0, T]$, subject to vehicle capacities $Q$, service time windows, and total route efficiency.

```
       ┌─────────────────────────────────────────────────────────┐
       │                   Time-Driven DCVRP                     │
       │    Horizon T = 480 min, β = 10 Intervals (48 min each)   │
       └────────────────────────────┬────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────┐                       ┌───────────────────────┐
│  Transformer Encoder  │                       │   Vehicle Selectors   │
│  • 3 Layers, 8 Heads  │                       │  • DVNDA (Independent)│
│  • 128 Hidden, 512 FF │                       │  • AMCVN (Centralized)│
│  • Node Embeddings    │                       │  • LiDRL (Tour Rec.)  │
└───────────┬───────────┘                       │  • MAAM (Round-Robin) │
            │                                   │  • MARDAM (Earliest)  │
            └───────────────────────┬───────────┴───────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │   Customer Decoder    │
                        │ • Context Pointer     │
                        │ • Tanh Clipping C=10  │
                        └───────────────────────┘
```

### Key Contributions
- **DVNDA Architecture**: A multi-agent framework giving each vehicle an independently parameterized attention sub-network coordinated through lightweight decision aggregation.
- **Fair NCO Baseline Suite**: Standardized adaptation of state-of-the-art vehicle coordination mechanisms into a unified time-driven environment:
  - **DVNDA** (Ours): Independently parameterized dual-attention with decision aggregation
  - **AMCVN**: Centralized multi-head attention over the whole fleet
  - **LiDRL**: Recurrent vehicle status and tour history embedding network
  - **MAAM**: Round-robin vehicle dispatch rule
  - **MARDAM**: Earliest-available vehicle dispatch rule
  - **Greedy**: Nearest-neighbor heuristic with dynamic insertion
- **Strict Manuscript Reproducibility**: 100% verified reproduction of Table I ($n = 20, m = 4$) with **all 24 cells within $\le 3.78\%$ error** (overall MAPE: **1.52%**), 14 cells achieving **PASS (极准)** ($\le 1.5\%$), and DVNDA achieving a max error of **0.93%** (MAPE **0.71%**).

---

## 📊 Experimental Results & Table I Reproduction

Below is the audited Table I benchmark ($n = 20$ customers, $m = 4$ vehicles, $Q = 150$) evaluated on the fixed 100 test instances across varying dynamic disclosure rates $\phi$:

| Dynamic Rate ($\phi$) | Method | Vehicle Selector | Measured Cost | 95% CI | Manuscript Cost | Error | Status ($\le 4\%$) |
|:----------------------:|:-------|:-----------------|:-------------:|:------:|:---------------:|:-----:|:------------------:|
| **10%** | Greedy | Nearest Neighbor | 9.15 ± 1.02 | — | 9.07 ± 1.12 | +0.90% | **PASS (极准)** |
| | MARDAM | Earliest-Available | 9.05 ± 1.04 | [8.84, 9.25] | 8.91 ± 1.29 | +1.52% | **PASS** |
| | MAAM | Round-Robin | 8.86 ± 1.19 | [8.63, 9.10] | 8.83 ± 1.22 | **+0.37%** | **PASS (极准)** |
| | LiDRL | Tour History | 8.88 ± 1.32 | [8.62, 9.14] | 8.67 ± 1.27 | +2.44% | **PASS** |
| | AMCVN | Centralized Attention | 8.61 ± 1.17 | [8.38, 8.84] | 8.39 ± 1.29 | +2.61% | **PASS** |
| | **DVNDA** (Ours) | **Independent Dual-Attention** | **8.28 ± 1.11** | [8.06, 8.50] | **8.31 ± 1.22** | **-0.34%** | **PASS (极准)** |
| **25%** | Greedy | Nearest Neighbor | 9.87 ± 1.10 | — | 9.69 ± 1.25 | +1.86% | **PASS** |
| | MARDAM | Earliest-Available | 9.99 ± 1.30 | [9.73, 10.24] | 9.85 ± 1.27 | +1.37% | **PASS (极准)** |
| | MAAM | Round-Robin | 9.66 ± 1.27 | [9.41, 9.91] | 9.68 ± 1.55 | **-0.19%** | **PASS (极准)** |
| | LiDRL | Tour History | 9.77 ± 1.24 | [9.53, 10.02] | 9.45 ± 1.24 | +3.43% | **PASS** |
| | AMCVN | Centralized Attention | 9.46 ± 1.18 | [9.22, 9.69] | 9.23 ± 1.23 | +2.48% | **PASS** |
| | **DVNDA** (Ours) | **Independent Dual-Attention** | **9.03 ± 1.03** | [8.83, 9.24] | **8.95 ± 1.30** | **+0.93%** | **PASS (极准)** |
| **50%** | Greedy | Nearest Neighbor | 11.23 ± 1.30 | — | 11.25 ± 1.43 | **-0.16%** | **PASS (极准)** |
| | MARDAM | Earliest-Available | 11.40 ± 1.45 | [11.11, 11.68] | 11.45 ± 1.37 | **-0.47%** | **PASS (极准)** |
| | MAAM | Round-Robin | 11.32 ± 1.35 | [11.05, 11.59] | 11.32 ± 1.30 | **-0.02%** | **PASS (极准)** |
| | LiDRL | Tour History | 10.89 ± 1.39 | [10.61, 11.17] | 11.01 ± 1.45 | **-1.08%** | **PASS (极准)** |
| | AMCVN | Centralized Attention | 10.87 ± 1.29 | [10.61, 11.12] | 10.75 ± 1.54 | **+1.09%** | **PASS (极准)** |
| | **DVNDA** (Ours) | **Independent Dual-Attention** | **10.40 ± 1.31** | [10.14, 10.66] | **10.47 ± 1.53** | **-0.66%** | **PASS (极准)** |
| **75%** | Greedy | Nearest Neighbor | 12.16 ± 1.48 | — | 12.43 ± 1.51 | -2.14% | **PASS** |
| | MARDAM | Earliest-Available | 12.36 ± 1.43 | [12.08, 12.64] | 12.81 ± 1.49 | -3.52% | **PASS** |
| | MAAM | Round-Robin | 12.24 ± 1.40 | [11.96, 12.52] | 12.72 ± 1.53 | -3.78% | **PASS** |
| | LiDRL | Tour History | 12.05 ± 1.44 | [11.76, 12.33] | 12.52 ± 1.62 | -3.76% | **PASS** |
| | AMCVN | Centralized Attention | 12.10 ± 1.60 | [11.78, 12.41] | 12.03 ± 1.47 | **+0.55%** | **PASS (极准)** |
| | **DVNDA** (Ours) | **Independent Dual-Attention** | **11.67 ± 1.39** | [11.40, 11.95] | **11.78 ± 1.44** | **-0.90%** | **PASS (极准)** |

> **Reproduction Highlights**:
> - **Threshold**: $\le 4.0\%$ relative error per cell.
> - **Compliance**: **24 / 24 cells passed (100%)**.
> - **Worst-cell error**: **3.78%** (comfortably under 4.0%).
> - **Mean Absolute Percentage Error (MAPE)**: **1.52%**.
> - **PASS (极准) Cells ($\le 1.5\%$)**: **14 / 24 cells**.
> - **Method Ranking at $\phi=50\%$**: Exactly matches manuscript: `DVNDA < AMCVN < LiDRL < Greedy < MAAM < MARDAM` (`same`).
> - **QoS**: DVNDA, AMCVN, and LiDRL achieve $99.85\% \sim 99.95\%$ (reported as **100%** per manuscript conventions).

---

## 📁 Repository Structure

```
DCVRP-main/
├── checkpoints/                 # Verified golden model checkpoints (92.7 MB)
│   ├── DVNDA/                   # DVNDA model weights, config, & training history
│   ├── AMCVN/                   # AMCVN model weights, config, & training history
│   ├── LiDRL/                   # LiDRL model weights, config, & training history
│   ├── MAAM/                    # MAAM model weights, config, & training history
│   ├── MARDAM/                  # MARDAM model weights, config, & training history
│   ├── DVNDA.pt / AMCVN.pt ...  # Direct checkpoint aliases
│   └── README.md                # Parameter breakdowns & checkpoint guide
├── reproduction/                # Self-contained Table I reproduction package
│   ├── protocol.py              # Exact manuscript hyperparameters & targets
│   ├── instances.py             # Deterministic instance & disclosure generator
│   ├── greedy.py                # Pure heuristic baseline
│   ├── check_protocol.py        # Protocol assertion test (< 15s)
│   ├── evaluate_table1.py       # Table I auditor and markdown generator
│   ├── train_all.py             # Controlled joint 4-rate training runner
│   ├── data/                    # Fixed test & validation instance tensors
│   └── tests/                   # Official assertion test suite
├── experiments/                 # Core model & selector definitions
│   ├── reviewer_study/          # Time-driven environment & selector modules
│   │   ├── model.py             # 3-layer 8-head Transformer encoder & decoder
│   │   ├── selectors.py         # 5 vehicle selection strategies
│   │   ├── paper_dcvrp.py       # Time-driven synchronized environment
│   │   ├── executed_path_dvnda.py # Executed-path distance refund accounting
│   │   └── run_original_uncertainty_n20.py # Model loading & evaluation utilities
│   └── results/reproduction_n20/# Generated outputs (TABLE_I_N20.md, CSV, JSON)
├── reproduce.py                 # One-click Table I reproduction entry point
├── train.py / learner.py        # Core reinforcement learning training scripts
├── data.py / args.py / critic.py / rollout.py # Supporting modules
├── DCVRP.pdf                    # Published manuscript
├── requirements.txt             # Minimal pinned dependencies
├── LICENSE.txt                  # License file
└── README.md                    # This documentation
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/DCVRP-main.git
cd DCVRP-main

# Install PyTorch (matching your CUDA driver, e.g. CUDA 12.1)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install -r requirements.txt
```

### 2. One-Click Table I Reproduction

Verify all Table I results in ~15 seconds:

```bash
python reproduce.py
```

Or verify the protocol assertions without model inference:

```bash
python -m reproduction.check_protocol
```

To run the unit test assertions:

```bash
pytest reproduction/tests/test_reproduction.py
```

---

## ⚙️ Experimental Parameters & Protocol

All parameters strictly match Section IV-A of the manuscript:

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| Customer Count ($n$) | 20 | Number of customer locations per instance |
| Fleet Size ($m$) | 4 | Number of vehicles ($m = n/5$) |
| Vehicle Capacity ($Q$) | 150 | Capacity per vehicle (never replenished during horizon) |
| Customer Demand | $[5, 41]$ | Uniformly distributed integer demand |
| Service Duration | $[10, 31]$ min | Uniformly distributed customer service time |
| Planning Horizon ($T$) | 480 min | 8-hour operational day |
| Time Intervals ($\beta$) | 10 | Equal synchronized intervals of 48 minutes |
| Vehicle Speed ($v$) | 1.0 | 1 coordinate unit per minute (480 in normalized time) |
| Revelation Process | Poisson | Mean rate $\lambda = (1+T)/2 = 240.5$ min clipped to $[1, 480]$ |
| Dynamic Rates ($\phi$) | $\{10\%, 25\%, 50\%, 75\%\}$ | Ratio of dynamic customers: $\text{round}(n\phi)$ |
| Encoder | 3 Layers | Transformer, 8 heads, $d=128$, FF dimension $512$ |
| Tanh Exploration | $C = 10$ | Compatibility clipping in attention decoder |
| RL Algorithm | REINFORCE | Rollout baseline (3 rollouts, paired $t$-test $\alpha = 0.05$) |
| Optimizer | Adam | Learning rate $1 \times 10^{-4}$, gradient clip $2.0$ |

---

## 🧩 Parameter Budgets by Method

All neural methods share an identical 561,029-parameter Transformer encoder and customer pointer decoder. Only the vehicle selector architecture differs:

| Method | Vehicle Selector Type | Shared Params | Selector Params | Total Trainable |
|:-------|:----------------------|:-------------:|:---------------:|:---------------:|
| **DVNDA** (Ours) | Independent Dual-Attention Sub-Networks | 561,029 | 143,109 | **704,138** |
| **AMCVN** | Centralized Multi-Head Fleet Attention | 561,029 | 64,133 | **625,162** |
| **LiDRL** | Tour History Recurrent Network | 561,029 | 114,949 | **675,978** |
| **MAAM** | Round-Robin Dispatch Rule | 561,029 | 0 | **561,029** |
| **MARDAM** | Earliest-Available Dispatch Rule | 561,029 | 0 | **561,029** |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
