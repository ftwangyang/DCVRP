# DVNDA: Deep Reinforcement Learning with Dual-Attention for Dynamic Capacitated Vehicle Routing Problem

Official implementation of the **Dual-Attention Vehicle Selection Network (DVNDA)** for the **Dynamic Capacitated Vehicle Routing Problem (DCVRP)**.

---

## 📌 Overview

The **Dynamic Capacitated Vehicle Routing Problem (DCVRP)** extends classical VRP by incorporating real-world operational dynamics:
- Customer requests are revealed dynamically over an operational horizon according to a stochastic Poisson disclosure process.
- Operations are synchronized across discrete time intervals ($\beta = 10$).
- Real-time vehicle dispatching and customer sequencing decisions are updated dynamically.

**DVNDA** decouples the multi-vehicle dynamic routing decision into two collaborative attention-based stages:
1. **Vehicle Selection**: Dedicated dual-attention sub-networks for each vehicle that attend to both current fleet states and active spatial-temporal customer demands.
2. **Customer Sequencing**: An attention pointer network with $C=10$ tanh exploration that decodes feasible destinations for the selected vehicle.

---

## 📁 Repository Structure

```
DCVRP-main/
├── models/                  # Neural network architectures
│   ├── __init__.py          # Module exports
│   ├── transformer.py       # Multi-Head Attention & Transformer Encoder
│   ├── selectors.py         # Vehicle selectors (DVNDA, AMCVN, LiDRL, MAAM, MARDAM)
│   └── attention_model.py   # Full AttentionLearner actor network
├── env/                     # DCVRP dynamic environment & data generation
│   ├── __init__.py          # Module exports
│   ├── environment.py       # Time-driven synchronized environment (Eqs. 11-14)
│   └── dataset.py           # Instance generator with Poisson revelation process
├── checkpoints/             # Trained model weights
│   ├── DVNDA.pt             # Proposed DVNDA model
│   ├── AMCVN.pt             # AMCVN baseline model
│   ├── LiDRL.pt             # LiDRL baseline model
│   ├── MAAM.pt              # MAAM baseline model
│   ├── MARDAM.pt            # MARDAM baseline model
│   └── README.md            # Checkpoint metadata & parameter breakdown
├── train.py                 # Training script (REINFORCE with Rollout Baseline)
├── eval.py                  # Evaluation & simulation script
├── DCVRP.pdf                # Published manuscript
├── requirements.txt         # Minimal Python dependencies
├── LICENSE.txt              # MIT License
└── README.md                # This documentation
```

---

## ⚙️ Experimental Parameters & Problem Setting

All parameters strictly follow Section IV-A of the manuscript:

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| Customer Count ($n$) | 20 | Number of customer locations per instance |
| Fleet Size ($m$) | 4 | Number of vehicles |
| Vehicle Capacity ($Q$) | 150 | Capacity per vehicle |
| Customer Demand | $[5, 41]$ | Uniformly distributed integer customer demand |
| Service Duration | $[10, 31]$ min | Uniformly distributed customer service time |
| Planning Horizon ($T$) | 480 min | 8-hour operational day |
| Time Intervals ($\beta$) | 10 | Equal synchronized decision intervals of 48 minutes |
| Vehicle Speed ($v$) | 1.0 | 1 coordinate unit per minute (480 in normalized units) |
| Revelation Process | Poisson | Mean rate $\lambda = (1+T)/2 = 240.5$ min clipped to $[1, 480]$ |
| Dynamic Rates ($\phi$) | $\{10\%, 25\%, 50\%, 75\%\}$ | Ratio of dynamic customers: $\text{round}(n\phi)$ |
| Encoder | 3 Layers | Transformer, 8 heads, $d=128$, FF dimension $512$ |
| Tanh Exploration | $C = 10$ | Compatibility clipping in attention decoder |
| RL Algorithm | REINFORCE | Rollout baseline (paired $t$-test $\alpha = 0.05$) |
| Optimizer | Adam | Learning rate $1 \times 10^{-4}$, gradient clipping $2.0$ |

---

## 🧩 Model Architectures & Parameter Budgets

All neural methods share an identical 561,029-parameter Transformer encoder and customer pointer decoder. Only the vehicle selection strategy varies:

| Method | Vehicle Selector Architecture | Shared Parameters | Selector Parameters | Total Parameters |
|:-------|:------------------------------|:-----------------:|:-------------------:|:----------------:|
| **DVNDA** (Ours) | Independent Dual-Attention Sub-Networks | 561,029 | 143,109 | **704,138** |
| **AMCVN** | Centralized Multi-Head Fleet Attention | 561,029 | 64,133 | **625,162** |
| **LiDRL** | Tour History Recurrent Network | 561,029 | 114,949 | **675,978** |
| **MAAM** | Round-Robin Dispatch Rule | 561,029 | 0 | **561,029** |
| **MARDAM** | Earliest-Available Dispatch Rule | 561,029 | 0 | **561,029** |

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/DCVRP-main.git
cd DCVRP-main

# Install PyTorch (CUDA recommended)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### 2. Evaluation

To evaluate the pretrained **DVNDA** model on 100 evaluation instances across all dynamic rates ($\phi \in \{0.10, 0.25, 0.50, 0.75\}$):

```bash
python eval.py --method DVNDA
```

To evaluate all models:

```bash
python eval.py --method all
```

Expected output format:
```
======================================================================================
Method     | Dynamic Rate | Distance (Mean +/- SD)   | QoS (%)        | Time (s)    
--------------------------------------------------------------------------------------
DVNDA      | phi =   10%    | 8.28 +/- 1.11            | 100% (99.90%)  | 1s (0.28s)  
DVNDA      | phi =   25%    | 9.03 +/- 1.03            | 100% (99.85%)  | 1s (0.29s)  
DVNDA      | phi =   50%    | 10.40 +/- 1.31           | 100% (99.85%)  | 1s (0.28s)  
DVNDA      | phi =   75%    | 11.67 +/- 1.39           | 100% (99.85%)  | 1s (0.26s)  
======================================================================================
```

### 3. Training From Scratch

To train the DVNDA model using REINFORCE with the Rollout Baseline:

```bash
python train.py --method DVNDA --epochs 100 --batch-size 100 --lr 0.0001
```

Checkpoints will be saved automatically to `checkpoints/`.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
