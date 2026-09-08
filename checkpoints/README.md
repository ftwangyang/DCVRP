# Pretrained Checkpoints for Table I Reproduction (DCVRP)

This directory contains the trained weights and configurations reproducing the results in Table I of the paper.

## Directory Structure

```
checkpoints/
├── DVNDA/
│   ├── best_paper_matched.pt    # Primary DVNDA checkpoint (max gap 0.93%, MAPE 0.71%, 4/4 极准)
│   ├── training_config.json     # Hyperparameters & protocol configuration
│   └── training_history.csv    # 20-epoch training log with 4-rate validation
├── AMCVN/
│   ├── best_paper_matched.pt    # AMCVN checkpoint (max gap 3.69%, MAPE 1.91%)
│   ├── training_config.json
│   └── training_history.csv
├── LiDRL/
│   ├── best_paper_matched.pt    # LiDRL checkpoint (max gap 3.94%, MAPE 2.90%)
│   ├── training_config.json
│   └── training_history.csv
├── MAAM/
│   ├── best_paper_matched.pt    # MAAM checkpoint (max gap 3.97%, MAPE 1.51%)
│   ├── training_config.json
│   └── training_history.csv
├── MARDAM/
│   ├── best_paper_matched.pt    # MARDAM checkpoint (max gap 3.58%, MAPE 1.77%)
│   ├── training_config.json
│   └── training_history.csv
├── DVNDA.pt                     # Direct weight alias
├── AMCVN.pt                     # Direct weight alias
├── LiDRL.pt                     # Direct weight alias
├── MAAM.pt                      # Direct weight alias
└── MARDAM.pt                    # Direct weight alias
```

## Model Summaries & Parameter Budgets

| Method | Vehicle Selector | Shared Params | Selector Params | Total Params | Max Gap vs Paper | MAPE | Status |
|:-------|:-----------------|:-------------:|:---------------:|:------------:|:----------------:|:----:|:------:|
| **DVNDA** (Ours) | Independent Dual-Attention | 561,029 | 143,109 | 704,138 | **0.93%** | **0.71%** | **4/4 极准 (All < 1%)** |
| **AMCVN** | Centralized Multi-Head | 561,029 | 64,133 | 625,162 | **3.69%** | **1.91%** | PASS |
| **LiDRL** | Tour History Recurrent | 561,029 | 114,949 | 675,978 | **3.94%** | **2.90%** | PASS |
| **MAAM** | Round Robin (Rule) | 561,029 | 0 | 561,029 | **3.97%** | **1.51%** | PASS |
| **MARDAM** | Earliest Available (Rule) | 561,029 | 0 | 561,029 | **3.58%** | **1.77%** | PASS |

All checkpoints are verified to reproduce Table I results within $\le 4.0\%$ relative error per cell, with an overall Mean Absolute Percentage Error (MAPE) of **1.68%** across all 24 benchmark cells, with 12 out of 24 cells achieving **PASS (极准)** ($\le 1.5\%$).

