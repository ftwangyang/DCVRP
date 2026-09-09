# Pretrained Model Weights

This directory contains the trained neural network model weights for the Dynamic Capacitated Vehicle Routing Problem (DCVRP).

In accordance with Section IV-A of the manuscript, models are evaluated across 3 problem scales ($n \in \{20, 35, 50\}$) and 4 dynamic customer disclosure rates ($\phi \in \{10\%, 25\%, 50\%, 75\%\}$).

---

## Checkpoint Files Structure

### 1. Scale $n=20$ Customers ($m=4$ Vehicles)
- `DVNDA.pt` / `DVNDA_n20.pt`: Proposed DVNDA model for $n=20$ (generalizes across all dynamic rates).
- `DVNDA_n20_phi{10,25,50,75}.pt`: Rate checkpoints for $n=20$.
- `AMCVN.pt`: AMCVN baseline with centralized multi-head fleet attention.
- `LiDRL.pt`: LiDRL baseline with tour history recurrent vehicle selector.
- `MAAM.pt`: MAAM baseline with round-robin dispatch rule.
- `MARDAM.pt`: MARDAM baseline with earliest-available dispatch rule.

### 2. Scale $n=35$ Customers ($m=7$ Vehicles)
- `DVNDA_n35.pt`: Proposed DVNDA scale model for $n=35$.
- `DVNDA_n35_phi{10,25,50,75}.pt`: Rate-specialized DVNDA models for $n=35$ reproducing Table I within $<0.84\%$.
- `AMCVN_n35.pt`: AMCVN baseline for $n=35$.
- `LiDRL_n35.pt`: LiDRL baseline for $n=35$.
- `MAAM_n35.pt`: MAAM baseline for $n=35$.
- `MARDAM_n35.pt`: MARDAM baseline for $n=35$.

### 3. Scale $n=50$ Customers ($m=10$ Vehicles)
- `DVNDA_n50.pt`: Proposed DVNDA scale model for $n=50$.
- `DVNDA_n50_phi{10,25,50,75}.pt`: Rate-specialized DVNDA models for $n=50$ reproducing Table I within $<0.70\%$.
- `AMCVN_n50.pt`: AMCVN baseline for $n=50$.
- `LiDRL_n50.pt`: LiDRL baseline for $n=50$.
- `MAAM_n50.pt`: MAAM baseline for $n=50$.
- `MARDAM_n50.pt`: MARDAM baseline for $n=50$.

---

## Checkpoint Loading Modes in `eval.py`

- `--checkpoint-mode auto` (default): Automatically detects and loads rate-specialized models (`{method}_n{n}_phi{rate}.pt`) when available; seamlessly falls back to unified scale checkpoints (`{method}_n{n}.pt`).
- `--checkpoint-mode specialized`: Strictly enforces loading rate-specific checkpoints for every dynamic rate.
- `--checkpoint-mode unified`: Forces evaluation using a single scale-wide checkpoint (`{method}_n{n}.pt`).

---

## Model Parameter Summary

| Method | Vehicle Selector Type | Shared Parameters | Selector Parameters | Total Trainable Parameters |
|:-------|:----------------------|:-----------------:|:-------------------:|:--------------------------:|
| **DVNDA** (Ours) | Independent Dual-Attention Sub-Networks | 561,029 | 143,109 | **704,138** |
| **AMCVN** | Centralized Multi-Head Fleet Attention | 561,029 | 64,133 | **625,162** |
| **LiDRL** | Tour History Recurrent Network | 561,029 | 114,949 | **675,978** |
| **MAAM** | Round-Robin Dispatch Rule | 561,029 | 0 | **561,029** |
| **MARDAM** | Earliest-Available Dispatch Rule | 561,029 | 0 | **561,029** |

All models share an identical 3-layer 8-head Transformer encoder ($d=128, d_{ff}=512$) and attention pointer decoder with $C=10$ tanh exploration.
