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

Evaluation outputs (metrics, costs, and comparison tables) are automatically exported to [`results/`](results/).

### 3. Training from Scratch

To train models from scratch:

```bash
# Train DVNDA on scale n=20
python train.py --method DVNDA -n 20 -m 4 --epochs 100

# Train DVNDA on scale n=35
python train.py --method DVNDA -n 35 -m 7 --epochs 100

# Train DVNDA on scale n=50
python train.py --method DVNDA -n 50 -m 10 --epochs 100
```

---

## 📁 Repository Structure

```
.
├── models/             # Neural network architectures (DVNDA, AMCVN, LiDRL, etc.)
├── env/                # DCVRP dynamic simulation environment & dataset generator
├── checkpoints/        # Pretrained model weights (n=20, 35, 50)
├── results/            # Benchmark reproduction logs (CSV, JSON, Markdown)
├── train.py            # Training script (REINFORCE with Rollout Baseline)
├── eval.py             # Evaluation script for benchmark reproduction
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 📚 Citation

If you find this codebase helpful in your research, please cite:

```bibtex
@article{wang2026distributed,
  title={Distributed Vehicle Network with Decision Aggregation for Dynamic Capacitated Vehicle Routing Problem},
  author={Wang, Yang and Jia, Ya-Hui and Yang, Qiang and Wei, Feng-Feng and Lin, Zhenhong and Chen, Wei-Neng},
  journal={IEEE Transactions},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
