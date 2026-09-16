"""Greedy + optional neural probe against Table I after the 5-knob protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import DCVRPEnvironment, generate_evaluation_split, set_seed
from env.greedy import run_greedy
from eval import TABLE1_BENCHMARK
from models import AttentionLearner, build_selector


def _gap(measured: float, target: float) -> float:
    return (measured - target) / target * 100.0


def evaluate_greedy(split: dict[float, object], targets: dict) -> list[dict]:
    rows = []
    for rate, dataset in split.items():
        distances, qos = run_greedy(dataset, reveal="continuous")
        mean = float(distances.mean())
        qos_mean = float(qos.mean() * 100.0)
        target = targets[round(float(rate), 2)]["Greedy"][0]
        rows.append(
            {
                "method": "Greedy",
                "rate": float(rate),
                "measured": mean,
                "qos": qos_mean,
                "target": target,
                "gap": _gap(mean, target),
            }
        )
    return rows


@torch.no_grad()
def evaluate_neural(
    split: dict[float, object],
    targets: dict,
    method: str,
    device: torch.device,
    checkpoint: Path | None,
) -> list[dict]:
    selector = build_selector(method, vehicle_count=4)
    model = AttentionLearner(selector).to(device)
    if checkpoint is not None and checkpoint.exists():
        chk = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(chk["model"], strict=True)
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    rows = []
    for rate, dataset in split.items():
        env = DCVRPEnvironment(dataset, nodes=dataset.nodes.to(device), pending_cost=0.0)
        model(env)
        mean = float(env.route_distance().mean().item())
        qos_mean = float(env.qos().mean().item() * 100.0)
        target = targets[round(float(rate), 2)][method][0]
        rows.append(
            {
                "method": method,
                "rate": float(rate),
                "measured": mean,
                "qos": qos_mean,
                "target": target,
                "gap": _gap(mean, target),
            }
        )
    return rows


def print_rows(title: str, rows: list[dict]) -> float:
    mae = sum(abs(r["gap"]) for r in rows) / len(rows)
    print(title)
    for row in rows:
        print(
            f"  phi={row['rate']*100:4.0f}%  {row['measured']:.2f} vs {row['target']:.2f}  "
            f"({row['gap']:+.1f}%)  QoS {row['qos']:.1f}%"
        )
    print(f"  MAE {mae:.2f}%  max {max(abs(r['gap']) for r in rows):.2f}%")
    return mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--customer-count", type=int, default=20)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--neural", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/DVNDA.pt"))
    args = parser.parse_args()

    n = args.customer_count
    m = max(1, round(n / 5))
    set_seed(args.seed)
    split = generate_evaluation_split(
        instances=args.instances,
        customer_count=n,
        vehicle_count=m,
        seed=args.seed,
    )
    first = next(iter(split.values()))
    print(
        f"n={n} instances={args.instances} speed={first.veh_speed:.3f} "
        f"loc_scl={first.location_scale:.1f}"
    )
    targets = TABLE1_BENCHMARK[n]
    greedy_mae = print_rows("Greedy (continuous, all 5 methods share this env)", evaluate_greedy(split, targets))
    if args.neural:
        device = torch.device(args.device)
        print_rows("DVNDA random init", evaluate_neural(split, targets, "DVNDA", device, None))
        if args.checkpoint.exists():
            print_rows(
                f"DVNDA {args.checkpoint.name}",
                evaluate_neural(split, targets, "DVNDA", device, args.checkpoint),
            )
    if greedy_mae > 5.0:
        print("Greedy MAE > 5%: protocol still off Table I, inspect before a long train.")
    else:
        print("Greedy MAE <= 5%: data protocol is in the Table I band.")


if __name__ == "__main__":
    main()
