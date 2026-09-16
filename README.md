# Distributed Vehicle Network with Decision Aggregation for Dynamic Capacitated Vehicle Routing Problem (DVNDA)

Official PyTorch implementation of **Distributed Vehicle Network with Decision Aggregation (DVNDA)** for the **Dynamic Capacitated Vehicle Routing Problem (DCVRP)**.

---

## 🚀 Quick Start & Reproduction

### 1. Installation

```bash
git clone https://github.com/ftwangyang/DCVRP.git
cd DCVRP

# Install dependencies (PyTorch 2.0+ recommended)
pip install -r requirements.txt
```

### 2. Reproduce Evaluation Results

Pretrained model checkpoints for all methods and problem scales ($n=20, 35, 50$) are provided in [`checkpoints/`](checkpoints/).

To reproduce the benchmark results on any or all scales:

```bash
# 1. Reproduce full benchmark across all scales (n=20, 35, 50) and all 6 methods:
python eval.py --method all -n all --compare-table1

# 2. Evaluate DVNDA (ours) on a specific scale:
python eval.py --method DVNDA -n 20   # 20 customers, 4 vehicles
python eval.py --method DVNDA -n 35   # 35 customers, 7 vehicles
python eval.py --method DVNDA -n 50   # 50 customers, 10 vehicles

# 3. Evaluate any baseline method (Greedy, AMCVN, LiDRL, MAAM, MARDAM):
python eval.py --method Greedy -n 20
python eval.py --method AMCVN -n 20
```

Evaluation outputs (metrics, costs, and comparison tables) are automatically exported to the `results/` directory.

### 3. Training from Scratch

Paper-aligned training follows Algorithm 1 execution (routes are executed to the fixed time boundary and vehicle clocks are synchronized to \(T_{r+1}\)) and Algorithm 2: REINFORCE with a greedy rollout baseline, 3 sampled policy rollouts, Eq. 27 vehicle argmax, and Eq. 14 depot-return cost.

```bash
# Fine-tune DVNDA on n=20 / 35 / 50 from the current checkpoints
python scripts/retrain_dvnda.py --epochs 100 --device cuda

# Or train one scale
python train.py --method DVNDA -n 20 -m 4 --epochs 100 --rollouts 3 --init-from checkpoints/DVNDA.pt --output-dir checkpoints/paper_align
python train.py --method DVNDA -n 35 -m 7 --epochs 100 --rollouts 3 --init-from checkpoints/DVNDA_n35.pt --output-dir checkpoints/paper_align
python train.py --method DVNDA -n 50 -m 10 --epochs 100 --rollouts 3 --init-from checkpoints/DVNDA_n50.pt --output-dir checkpoints/paper_align

# Multi-seed search from scratch; stop a seed at epoch 10 if Table I MAE > 8%
python scripts/multiseed_dvnda.py -n 20 --seeds 42 7 2024 3407 20260821 --from-scratch --early-stop-epoch 10 --early-stop-mae 8
```

---

## 📁 Repository Structure

```
.
├── models/             # Neural network architectures (DVNDA, AMCVN, LiDRL, etc.)
├── env/                # DCVRP dynamic simulation environment & dataset generator
├── checkpoints/        # Pretrained model weights (n=20, 35, 50)
├── results/            # Evaluation outputs (auto-created upon running eval.py)
├── train.py            # Training script (REINFORCE with Rollout Baseline)
├── eval.py             # Evaluation script for benchmark reproduction
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
