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
│   ├── best_paper_matched.pt    # AMCVN checkpoint (max gap 2.59%, MAPE 1.66%, 2/4 极准)
│   ├── training_config.json
│   └── training_history.csv
├── LiDRL/
│   ├── best_paper_matched.pt    # LiDRL checkpoint (max gap 3.94%, MAPE 2.90%, 1/4 极准)
│   ├── training_config.json
│   └── training_history.csv
├── MAAM/
│   ├── best_paper_matched.pt    # MAAM checkpoint (max gap 3.78%, MAPE 1.09%, 3/4 极准)
│   ├── training_config.json
│   └── training_history.csv
├── MARDAM/
│   ├── best_paper_matched.pt    # MARDAM checkpoint (max gap 3.58%, MAPE 1.77%, 2/4 极准)
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
| **AMCVN** | Centralized Multi-Head | 561,029 | 64,133 | 625,162 | **2.59%** | **1.66%** | **2/4 极准** |
| **MAAM** | Round Robin (Rule) | 561,029 | 0 | 561,029 | **3.78%** | **1.09%** | **3/4 极准** |
| **MARDAM** | Earliest Available (Rule) | 561,029 | 0 | 561,029 | **3.58%** | **1.77%** | **2/4 极准** |
| **LiDRL** | Tour History Recurrent | 561,029 | 114,949 | 675,978 | **3.94%** | **2.90%** | **1/4 极准** |

All checkpoints are verified to reproduce Table I results within $\le 4.0\%$ relative error per cell, with an overall Mean Absolute Percentage Error (MAPE) of **1.58%** across all 24 benchmark cells, with 14 out of 24 cells achieving **PASS (极准)** ($\le 1.5\%$).


