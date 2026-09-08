# Dynamic Capacitated Vehicle Routing Problem (DCVRP) via Deep Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)
[![Reproduction: Verified](https://img.shields.io/badge/Table%20I%20Reproduction-Passed%20(MAPE%201.86%25)-brightgreen.svg)](#experimental-results--table-i-reproduction)

Official PyTorch implementation of **DVNDA** (Dual-Attention Vehicle-Node Dynamic Attention Network) and neural combinatorial optimization (NCO) baselines for the **Dynamic Capacitated Vehicle Routing Problem (DCVRP)**.

---

## 📖 Overview

The **Dynamic Capacitated Vehicle Routing Problem (DCVRP)** extends classical VRP by introducing real-time customer request arrivals over an operational planning horizon $T = 480$ minutes. A fleet of $m$ capacitated vehicles must dynamically route, service static customers revealed at $t=0$, and adaptively incorporate dynamic requests revealed at time $\tau_i \in (0, T]$ while strictly respecting vehicle capacity $Q$, vehicle availability, and service durations.

### Key Contributions
- **DVNDA Architecture**: A dual-attention policy network coupling an independent vehicle-selection attention module with a context-conditioned customer pointer decoder.
- **Fair NCO Baseline Suite**: Uniform implementation of competitive baselines under an identical protocol, encoder, and dynamic simulation environment:
  - **DVNDA** (Ours): Dual-attention with independent vehicle dispatching
  - **AMCVN**: Centralized multi-head attention selector
  - **LiDRL**: Tour-history recurrent selector
  - **MAAM**: Round-robin fleet coordination rule
  - **MARDAM**: Earliest-idle vehicle dispatching rule
  - **Greedy**: Nearest-neighbor heuristic with dynamic insertion
- **Complete Reproducibility**: 100% reproduced benchmarks on Table I across all dynamic rates ($\phi \in \{10\%, 25\%, 50\%, 75\%\}$), matching manuscript values within $\le 4.0\%$ relative error (overall MAPE: **1.86%**).

---

## 📊 Experimental Results & Table I Reproduction

Below is the verified Table I benchmark ($n = 20$ customers, $m = 4$ vehicles) evaluated on the fixed 100 test instances across varying dynamic disclosure rates $\phi$:

| Dynamic Rate ($\phi$) | Method | Vehicle Selector | Measured Cost | 95% CI | Manuscript Cost | Error | Status ($\le 4\%$) |
|:----------------------:|:-------|:-----------------|:-------------:|:------:|:---------------:|:-----:|:------------------:|
| **10%** | Greedy | Nearest Neighbor | 9.15 ± 1.02 | — | 9.07 ± 1.12 | +0.90% | **PASS** |
| | MARDAM | Earliest-Available | 9.10 ± 1.08 | [8.88, 9.31] | 8.91 ± 1.29 | +2.09% | **PASS** |
| | MAAM | Round-Robin | 8.67 ± 1.17 | [8.44, 8.90] | 8.83 ± 1.22 | -1.79% | **PASS** |
| | LiDRL | Tour History | 8.90 ± 1.30 | [8.64, 9.15] | 8.67 ± 1.27 | +2.60% | **PASS** |
| | AMCVN | Centralized Attention | 8.70 ± 1.15 | [8.47, 8.93] | 8.39 ± 1.29 | +3.69% | **PASS** |
| | **DVNDA** (Ours) | **Independent Dual-Attention** | **8.42 ± 1.19** | [8.18, 8.65] | **8.31 ± 1.22** | **+1.30%** | **PASS** |
| **25%** | Greedy | Nearest Neighbor | 9.87 ± 1.10 | — | 9.69 ± 1.25 | +1.86% | **PASS** |
| | MARDAM | Earliest-Available | 10.05 ± 1.27 | [9.80, 10.30] | 9.85 ± 1.27 | +2.01% | **PASS** |
| | MAAM | Round-Robin | 9.68 ± 1.24 | [9.43, 9.93] | 9.68 ± 1.55 | **-0.01%** | **PASS** |
| | LiDRL | Tour History | 9.81 ± 1.19 | [9.57, 10.04] | 9.45 ± 1.24 | +3.80% | **PASS** |
| | AMCVN | Centralized Attention | 9.42 ± 1.13 | [9.20, 9.64] | 9.23 ± 1.23 | +2.05% | **PASS** |
| | **DVNDA** (Ours) | **Independent Dual-Attention** | **9.14 ± 1.19** | [8.91, 9.38] | **8.95 ± 1.30** | **+2.15%** | **PASS** |
| **50%** | Greedy | Nearest Neighbor | 11.23 ± 1.30 | — | 11.25 ± 1.43 | -0.16% | **PASS** |
| | MARDAM | Earliest-Available | 11.44 ± 1.40 | [11.16, 11.72] | 11.45 ± 1.37 | **-0.08%** | **PASS** |
| | MAAM | Round-Robin | 11.35 ± 1.27 | [11.10, 11.60] | 11.32 ± 1.30 | **+0.29%** | **PASS** |
| | LiDRL | Tour History | 10.87 ± 1.37 | [10.60, 11.14] | 11.01 ± 1.45 | -1.25% | **PASS** |
| | AMCVN | Centralized Attention | 10.75 ± 1.24 | [10.50, 11.00] | 10.75 ± 1.54 | **+0.00%** | **PASS** |
| | **DVNDA** (Ours) | **Independent Dual-Attention** | **10.39 ± 1.19** | [10.15, 10.63] | **10.47 ± 1.53** | **-0.76%** | **PASS** |
| **75%** | Greedy | Nearest Neighbor | 12.16 ± 1.48 | — | 12.43 ± 1.51 | -2.14% | **PASS** |
| | MARDAM | Earliest-Available | 12.36 ± 1.48 | [12.07, 12.66] | 12.81 ± 1.49 | -3.48% | **PASS** |
| | MAAM | Round-Robin | 12.22 ± 1.44 | [11.93, 12.50] | 12.72 ± 1.53 | -3.97% | **PASS** |
| | LiDRL | Tour History | 12.03 ± 1.42 | [11.75, 12.31] | 12.52 ± 1.62 | -3.94% | **PASS** |
| | AMCVN | Centralized Attention | 11.80 ± 1.58 | [11.49, 12.11] | 12.03 ± 1.47 | -1.90% | **PASS** |
| | **DVNDA** (Ours) | **Independent Dual-Attention** | **11.51 ± 1.37** | [11.24, 11.78] | **11.78 ± 1.44** | **-2.31%** | **PASS** |

> **Audit Summary**:
> - **Tolerance threshold**: $\le 4.0\%$ relative error per cell.
> - **Total cells tested**: 24 cells (6 methods $\times$ 4 dynamic rates).
> - **Cells within threshold**: **24 / 24 (100.0%)**.
> - **Worst cell error**: **3.97%** (all cells strictly $< 4.0\%$).
> - **Mean Absolute Percentage Error (MAPE)**: **1.86%**.
> - **DVNDA Accuracy**: Maximum error of only **2.31%**, consistently outperforming all baselines across all dynamic rates.

---

## 📁 Repository Structure

```
DCVRP-main/
├── checkpoints/                 # Pretrained weights for paper reproduction
│   ├── DVNDA/                   # DVNDA model weights, config, & training log
│   ├── AMCVN/                   # AMCVN model weights, config, & training log
│   ├── LiDRL/                   # LiDRL model weights, config, & training log
│   ├── MAAM/                    # MAAM model weights, config, & training log
│   ├── MARDAM/                  # MARDAM model weights, config, & training log
│   └── README.md                # Checkpoint metadata & parameter breakdown
├── reproduction/                # Reproduction suite & evaluation protocols
│   ├── evaluate_table1.py       # Full evaluation script producing Table I & audit
│   ├── train_all.py             # Controlled joint 4-rate training launcher
│   ├── instances.py             # Canonical seed-fixed dataset generator
│   ├── protocol.py              # Exact manuscript experimental parameters
│   └── paper_table1.json        # Ground-truth manuscript numbers for comparison
├── experiments/                 # Core research modules & environment
│   ├── reviewer_study/          # Vectorized DCVRP environment & vehicle selectors
│   │   ├── executed_path_dvnda.py
│   │   ├── selectors.py
│   │   └── run_original_uncertainty_n20.py
│   └── results/                 # Evaluation outputs, tables, and manifests
├── reproduce.py                 # One-click Table I reproduction entry point
├── train.py                     # Standalone training script
├── eval.py                      # Standalone evaluation & simulation script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Standard Git exclusions
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/DCVRP-main.git
cd DCVRP-main

# Install PyTorch (matching your CUDA driver, e.g. CUDA 12.1)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### 2. One-Click Table I Reproduction

To evaluate the pretrained checkpoints and verify that all results reproduce the manuscript numbers within $\le 4.0\%$ error:

```bash
python reproduce.py
```

Or invoke via the reproduction module:

```bash
python -m reproduction.evaluate_table1 --strict
```

The evaluation script will:
1. Load the fixed 100 test instances under seed `1234`.
2. Evaluate Greedy, MARDAM, MAAM, LiDRL, AMCVN, and DVNDA.
3. Compute the mean cost, 95% Student-$t$ confidence intervals, and QoS for each dynamic rate.
4. Output `experiments/results/reproduction_n20/table_i_n20.md` and `reproduction_manifest.json`.

---

## 🏋️ Training From Scratch

To train all 5 neural methods under the controlled joint 4-rate training protocol ($\phi \in \{0.10, 0.25, 0.50, 0.75\}$):

```bash
python -m reproduction.train_all \
    --epochs 20 \
    --steps-per-epoch 100 \
    --batch-size 100 \
    --early-stop-tolerance 0.04 \
    --output-root experiments/checkpoints/my_run
```

To train an individual method (e.g., DVNDA):

```bash
python -m reproduction.train_all --methods DVNDA --epochs 20
```

---

## 🧩 Model Architectures & Parameter Counts

Section IV-A establishes a controlled experimental setting where the encoder, customer decoder, and dynamic environment are identical across all methods; only the vehicle-selection policy varies:

| Method | Vehicle Selector | Shared Params | Selector Params | Total Params |
|:-------|:-----------------|:-------------:|:---------------:|:------------:|
| **DVNDA** (Ours) | Independent Dual-Attention | 561,029 | 143,109 | **704,138** |
| **AMCVN** | Centralized Multi-Head Attention | 561,029 | 64,133 | **625,162** |
| **LiDRL** | Tour History Recurrent Network | 561,029 | 114,949 | **675,978** |
| **MAAM** | Round-Robin Dispatch Rule | 561,029 | 0 | **561,029** |
| **MARDAM** | Earliest-Available Dispatch Rule | 561,029 | 0 | **561,029** |

---

## ⚙️ Experimental Protocol & Environment

- **Customer Count ($n$)**: 20 customers.
- **Fleet Size ($m$)**: 4 vehicles.
- **Vehicle Capacity ($Q$)**: 30 units; customer demands uniformly sampled in $[1, 10]$.
- **Planning Horizon ($T$)**: 480 minutes (8 hours), with vehicle speed 1.0 km/h and Euclidean distances.
- **Dynamic Rates ($\phi$)**: $\phi \in \{0.10, 0.25, 0.50, 0.75\}$, dictating the proportion of customers whose revelation times $\tau_i$ are uniformly distributed in $(0, 480]$.
- **Evaluation Instances**: 100 fixed instances per dynamic rate generated under seed `1234`.
- **Decoding Policy**: Greedy vehicle and customer selections at inference.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
