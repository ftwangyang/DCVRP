# Pretrained Model Weights

This directory contains the trained neural network model weights for the Dynamic Capacitated Vehicle Routing Problem (DCVRP).

## Files

- `DVNDA.pt`: The proposed DVNDA model with independent dual-attention vehicle selection sub-networks ($n=20, m=4$).
- `DVNDA_n35.pt`: The proposed DVNDA model for the 35-customer scale ($n=35, m=7$).
- `DVNDA_n50.pt`: The proposed DVNDA model for the 50-customer scale ($n=50, m=10$).
- `AMCVN.pt`: AMCVN model with centralized multi-head fleet attention.
- `LiDRL.pt`: LiDRL model with tour history recurrent vehicle selector.
- `MAAM.pt`: MAAM model with round-robin dispatch rule.
- `MARDAM.pt`: MARDAM model with earliest-available dispatch rule.

## Model Parameter Summary

| Method | Vehicle Selector Type | Shared Parameters | Selector Parameters | Total Trainable Parameters |
|:-------|:----------------------|:-----------------:|:-------------------:|:--------------------------:|
| **DVNDA** (Ours) | Independent Dual-Attention Sub-Networks | 561,029 | 143,109 | **704,138** |
| **AMCVN** | Centralized Multi-Head Fleet Attention | 561,029 | 64,133 | **625,162** |
| **LiDRL** | Tour History Recurrent Network | 561,029 | 114,949 | **675,978** |
| **MAAM** | Round-Robin Dispatch Rule | 561,029 | 0 | **561,029** |
| **MARDAM** | Earliest-Available Dispatch Rule | 561,029 | 0 | **561,029** |

All models share an identical 3-layer 8-head Transformer encoder ($d=128, d_{ff}=512$) and attention pointer decoder with $C=10$ tanh exploration.
