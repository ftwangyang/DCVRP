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

| Method | Vehicle Selector Type | Shared Encoder/Decoder | Selector (n=20, m=4) | Total (n=20) |
|:-------|:----------------------|:----------------------:|:--------------------:|:------------:|
| **DVNDA** (Ours) | Distributed Vehicle Networks with Decision Aggregation | 709,504 | 183,556 | **893,060** |
| **AMCVN** | Centralized vehicle network | 709,504 | 83,841 | **793,345** |
| **LiDRL** | Tour-history vehicle selector | 709,504 | 150,404 | **859,908** |
| **MAAM** | Round-robin dispatch rule | 709,504 | 0 | **709,504** |
| **MARDAM** | Earliest-available dispatch rule | 709,504 | 0 | **709,504** |

DVNDA selector parameters scale with the fleet: **1,030,727** at n=35 (m=7) and **1,168,394** at n=50 (m=10). Shared encoder/decoder size is 709,504 at every scale.

All models share an identical 3-layer 8-head Transformer encoder ($d=128, d_{ff}=512$) and attention pointer decoder with $C=10$ tanh exploration. Vehicle selection follows Eq. 27 (argmax).
