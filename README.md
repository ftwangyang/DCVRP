# Distributed Vehicle Network with Decision Aggregation (DVNDA)


## Installation

```bash
# Clone the repository
git clone https://github.com/ftwangyang/DCVRP.git
cd DCVRP

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Evaluation
Evaluate the pre-trained model (or baseline heuristics) on different dynamic rates ($\phi \in \{0.10, 0.25, 0.50, 0.75\}$). 

**Standard Benchmark (Uniform Topology):**
```bash
python eval.py --method DVNDA --instances 100 --rates 0.1 0.25 0.5 0.75
```

**Real-World Generalization (e.g., Vienna):**
```bash
python eval.py --method DVNDA --distribution real --real-data-path data/vienna_16080.csv
```

**Temporal Generalization (Early Peak):**
```bash
python eval.py --method DVNDA --revelation early_peak
```

### 2. Training
To train the DVNDA model from scratch:
```bash
python train.py --method DVNDA --epochs 100 --batch-size 100
```
*Note: The model automatically evaluates against a Greedy rollout baseline using paired t-tests during training.*

## Datasets
The repository includes extracted real-world normalized datasets (`data/`) mapped exactly to the node-counts specified in the supplementary tests:
- `vienna_16080.csv` (16,080 nodes)
- `london_5629.csv` (5,629 nodes)
- `newyork_4043.csv` (4,043 nodes)
