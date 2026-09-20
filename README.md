# DVNDA for Dynamic Capacitated Vehicle Routing

PyTorch implementation of Distributed Vehicle Network with Decision Aggregation (DVNDA) for the Dynamic Capacitated Vehicle Routing Problem (DCVRP).

The horizon \(T\) is split into ten equal intervals. At the start of interval \(r\), requests with \(a_i \le T_r\) are disclosed (the last interval uses \(T\)). Started work is kept and the unstarted suffix is discarded at the next boundary. Each vehicle makes a single round trip per interval. Dynamic appearance times follow the truncated Poisson PMF in Eq. 34. Evaluation uses greedy decoding.

## Requirements

Python 3.9+ and PyTorch 2.0+.

```bash
pip install -r requirements.txt
```

## Evaluation

```bash
python eval.py --method DVNDA -n 20
python eval.py --method Greedy -n 20
python eval.py --method DVNDA -n 20 --compare-table1
```

## Training

```bash
python train.py --method DVNDA -n 20 -m 4 --epochs 100 --output-dir checkpoints
```

n=20 uses batch size 100 and 1000 steps per epoch; n=35 and 50 use batch size 50 and 500 steps.

## License

MIT. See [LICENSE.txt](LICENSE.txt).
