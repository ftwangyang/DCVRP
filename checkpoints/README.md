# Pretrained Model Weights

This directory contains the trained neural network model weights for the Dynamic Capacitated Vehicle Routing Problem (DCVRP).

In accordance with Section IV-A of the manuscript, models are evaluated across 3 problem scales ($n \in \{20, 35, 50\}$) and 4 dynamic customer disclosure rates ($\phi \in \{10\%, 25\%, 50\%, 75\%\}$).

---

## Checkpoint Files Structure

In strict accordance with the manuscript design, **exactly one unified model weight file is maintained per problem scale** ($n \in \{20, 35, 50\}$). A single trained model evaluates across all four dynamic customer revelation rates ($\phi \in \{10\%, 25\%, 50\%, 75\%\}$) without any rate-specific specialization or branch loading.

### 1. Scale $n=20$ Customers ($m=4$ Vehicles)
- `DVNDA.pt`: Proposed DVNDA model for $n=20$ (evaluates across all dynamic rates $\phi$).
- `AMCVN.pt`: AMCVN baseline with centralized multi-head fleet attention.
- `LiDRL.pt`: LiDRL baseline with tour history recurrent vehicle selector.
- `MAAM.pt`: MAAM baseline with round-robin dispatch rule.
- `MARDAM.pt`: MARDAM baseline with earliest-available dispatch rule.

### 2. Scale $n=35$ Customers ($m=7$ Vehicles)
- `DVNDA_n35.pt`: Proposed DVNDA model for $n=35$.
- `AMCVN_n35.pt`: AMCVN baseline for $n=35$.
- `LiDRL_n35.pt`: LiDRL baseline for $n=35$.
- `MAAM_n35.pt`: MAAM baseline for $n=35$.
- `MARDAM_n35.pt`: MARDAM baseline for $n=35$.

### 3. Scale $n=50$ Customers ($m=10$ Vehicles)
- `DVNDA_n50.pt`: Proposed DVNDA model for $n=50$.
- `AMCVN_n50.pt`: AMCVN baseline for $n=50$.
- `LiDRL_n50.pt`: LiDRL baseline for $n=50$.
- `MAAM_n50.pt`: MAAM baseline for $n=50$.
- `MARDAM_n50.pt`: MARDAM baseline for $n=50$.

---

## Evaluation Command

To evaluate a single scale or all scales using the unified checkpoints:

```bash
# Evaluate a single scale (e.g., n=20)
python eval.py --method all -n 20 --compare-table1

# Evaluate all scales (n=20, 35, 50) sequentially
python eval.py --method all -n all --compare-table1 --save-dir results
```

---

## Model Parameter Summary

| Method | Vehicle Selector Type | Shared Parameters | Selector Parameters | Total Trainable Parameters |
|:-------|:----------------------|:-----------------:|:-------------------:|:--------------------------:|
| **DVNDA** (Ours) | Distributed Vehicle Networks with Decision Aggregation | 561,029 | 143,109 | **704,138** |
| **AMCVN** | Centralized Multi-Head Fleet Attention | 561,029 | 64,133 | **625,162** |
| **LiDRL** | Tour History Recurrent Network | 561,029 | 114,949 | **675,978** |
| **MAAM** | Round-Robin Dispatch Rule | 561,029 | 0 | **561,029** |
| **MARDAM** | Earliest-Available Dispatch Rule | 561,029 | 0 | **561,029** |

All models share an identical 3-layer 8-head Transformer encoder ($d=128, d_{ff}=512$) and attention pointer decoder with $C=10$ tanh exploration.
