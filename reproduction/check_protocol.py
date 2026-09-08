"""Verify the reproduction protocol without training anything.

Three checks run here:

1. the instance generator transcribed from ``data.py`` is byte-for-byte equal to
   the released one;
2. the disclosure-time process matches the manuscript's stated mean rate;
3. Greedy -- the only training-free row of Table I -- reproduces the manuscript.

Check 3 is the substantive one.  Because Greedy contains no learned component,
any disagreement is attributable to the generator, the time semantics, the
capacity semantics, or the cost definition, and not to model quality.  The sweep
over generators and reveal rules shows which of those choices Table I is
sensitive to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction import instances, protocol  # noqa: E402
from reproduction.greedy import run_greedy  # noqa: E402

PAPER_GREEDY_COST = {0.10: 9.07, 0.25: 9.69, 0.50: 11.25, 0.75: 12.43}
PAPER_GREEDY_QOS = {0.10: 99.90, 0.25: 99.90, 0.50: 99.90, 0.75: 99.45}


class ReleasedView:
    """Adapt an ``InstanceBatch`` to the attribute names Greedy expects."""

    def __init__(self, batch: instances.InstanceBatch) -> None:
        self.veh_count = batch.vehicle_count
        self.veh_capa = batch.vehicle_capacity
        self.veh_speed = batch.vehicle_speed
        self.nodes = batch.nodes


def check_generator() -> None:
    print("1. generator transcription")
    instances.check_released_equivalence()
    print("   released transcription matches data.py byte-for-byte: OK")


def check_disclosure_process() -> None:
    print("\n2. disclosure-time process")
    rates = instances.disclosure_rates(protocol.CUSTOMER_COUNT)
    mean_rate = float(rates.mean())
    expected = (1.0 + protocol.HORIZON_MINUTES) / 2.0
    print(
        f"   per-slot rates linspace(1, {protocol.HORIZON_MINUTES}, "
        f"{protocol.CUSTOMER_COUNT}); mean {mean_rate:.1f} minutes"
    )
    print(f"   manuscript scalar rate (1+T)/2 = {expected:.1f}: "
          f"{'OK' if abs(mean_rate - expected) < 1e-6 else 'MISMATCH'}")
    split = instances.generate_manuscript_split(protocol.TEST_SEED, 100)
    for rate, data in split.items():
        disclosure = data.nodes[:, 1:, 4]
        dynamic = disclosure > 0
        count = float(dynamic.float().sum(dim=1).mean())
        minutes = float(disclosure[dynamic].mean() * protocol.HORIZON_MINUTES)
        print(
            f"   phi={rate:.2f}: {count:.2f} dynamic customers per instance "
            f"(exactly {round(protocol.CUSTOMER_COUNT * rate)}), "
            f"mean disclosure {minutes:.1f} minutes"
        )


def greedy_row(data, reveal: str) -> tuple[list[float], float]:
    cells, worst = [], 0.0
    for rate in sorted(data):
        instances.seed_all(protocol.DECODE_SEED)
        cost, qos = run_greedy(data[rate], reveal=reveal)
        error = float(cost.mean()) / PAPER_GREEDY_COST[rate] - 1.0
        worst = max(worst, abs(error))
        cells.append((float(cost.mean()), float(cost.std()), error,
                      100.0 * float(qos.mean())))
    return cells, worst


def check_greedy(test_instances: int, seed: int) -> float:
    print("\n3. Greedy row of Table I (no training involved)")
    manuscript = instances.generate_manuscript_split(seed, test_instances)
    released = {
        rate: ReleasedView(batch)
        for rate, batch in instances.generate_released_split(
            seed, test_instances
        ).items()
    }

    header = f"   {'generator / reveal':<34}" + "".join(
        f"{f'phi={r:.2f}':>18}" for r in PAPER_GREEDY_COST
    ) + f"{'worst':>8}"
    print(header)
    print(f"   {'manuscript Table I':<34}" + "".join(
        f"{PAPER_GREEDY_COST[r]:>10.2f}{'':>8}" for r in PAPER_GREEDY_COST
    ))

    best = float("inf")
    for label, data in (("corrected paired", manuscript), ("released", released)):
        for reveal in ("continuous", "next_boundary"):
            cells, worst = greedy_row(data, reveal)
            best = min(best, worst)
            body = "".join(
                f"{mean:>10.2f}{f'({100*err:+.1f}%)':>8}"
                for mean, _, err, _ in cells
            )
            print(f"   {label + ' / ' + reveal:<34}{body}{100*worst:>7.1f}%")

    cells, worst = greedy_row(manuscript, "continuous")
    print("\n   reported configuration (corrected paired generator, continuous reveal):")
    for rate, (mean, sd, error, qos) in zip(sorted(PAPER_GREEDY_COST), cells):
        print(
            f"     phi={rate:.2f}: cost {mean:5.2f} +/- {sd:4.2f} vs "
            f"manuscript {PAPER_GREEDY_COST[rate]:5.2f} ({100*error:+.2f}%); "
            f"QoS {qos:.2f}% vs {PAPER_GREEDY_QOS[rate]:.2f}%"
        )
    return worst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=protocol.TEST_INSTANCES)
    parser.add_argument("--seed", type=int, default=protocol.TEST_SEED)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    check_generator()
    check_disclosure_process()
    worst = check_greedy(args.instances, args.seed)
    print(
        f"\nprotocol check: worst Greedy cell {100*worst:.2f}% "
        f"(tolerance {100*args.tolerance:.0f}%) -> "
        f"{'PASS' if worst <= args.tolerance else 'FAIL'}"
    )
    return 0 if worst <= args.tolerance else 2


if __name__ == "__main__":
    raise SystemExit(main())
