# Pretrained Model Weights

This directory contains the trained neural network model weights for the Dynamic Capacitated Vehicle Routing Problem (DCVRP).

### 20-Customer Scale ($n=20, m=4$)
- `DVNDA.pt`: Proposed DVNDA model with independent dual-attention sub-networks.
- `AMCVN.pt`: AMCVN baseline with centralized multi-head fleet attention.
- `LiDRL.pt`: LiDRL baseline with tour history recurrent vehicle selector.
- `MAAM.pt`: MAAM baseline with round-robin dispatch rule.
- `MARDAM.pt`: MARDAM baseline with earliest-available dispatch rule.

### 35-Customer Scale ($n=35, m=7$)
- `DVNDA_n35.pt`: Proposed DVNDA model for $n=35$.
- `AMCVN_n35.pt`: AMCVN baseline for $n=35$.
- `LiDRL_n35.pt`: LiDRL baseline for $n=35$.
- `MAAM_n35.pt`: MAAM baseline for $n=35$.
- `MARDAM_n35.pt`: MARDAM baseline for $n=35$.

### 50-Customer Scale ($n=50, m=10$)
- `DVNDA_n50.pt`: Proposed DVNDA model for $n=50$.
- `AMCVN_n50.pt`: AMCVN baseline for $n=50$.
- `LiDRL_n50.pt`: LiDRL baseline for $n=50$.
- `MAAM_n50.pt`: MAAM baseline for $n=50$.
- `MARDAM_n50.pt`: MARDAM baseline for $n=50$.

## Model Parameter Summary

| Method | Vehicle Selector Type | Shared Parameters | Selector Parameters | Total Trainable Parameters |
|:-------|:----------------------|:-----------------:|:-------------------:|:--------------------------:|
| **DVNDA** (Ours) | Independent Dual-Attention Sub-Networks | 561,029 | 143,109 | **704,138** |
| **AMCVN** | Centralized Multi-Head Fleet Attention | 561,029 | 64,133 | **625,162** |
| **LiDRL** | Tour History Recurrent Network | 561,029 | 114,949 | **675,978** |
| **MAAM** | Round-Robin Dispatch Rule | 561,029 | 0 | **561,029** |
| **MARDAM** | Earliest-Available Dispatch Rule | 561,029 | 0 | **561,029** |

All models share an identical 3-layer 8-head Transformer encoder ($d=128, d_{ff}=512$) and attention pointer decoder with $C=10$ tanh exploration.
