"""Train the five neural methods of Table I under one identical protocol.

Section IV-A states that only the vehicle-selection strategy differs between
methods: the encoder, node decoder, training procedure, and dynamic environment
are shared.  This launcher enforces that by construction -- every method gets
byte-identical arguments except ``--selector`` -- so a reviewer can verify the
controlled comparison from the command lines alone.

MARDAM's earliest-idle rule and MAAM's round-robin rule have no trainable
vehicle-selection parameters; for those two methods the shared customer decoder
is the only thing that learns.  That is a property of the published strategies,
not an implementation shortcut.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction import protocol  # noqa: E402

# Method -> selector name registered in experiments/reviewer_study/selectors.py
SELECTORS = {
    "MARDAM": "earliest_available",
    "MAAM": "round_robin",
    "LiDRL": "lidrl_tour_history",
    "AMCVN": "centralized",
    "DVNDA": "independent",
}

RUNNER = "experiments.reviewer_study.run_original_uncertainty_n20"


def command(
    method: str,
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    vehicle_policy: str,
    checkpoint_root: Path,
    output_root: Path,
    early_stop_tolerance: float | None = None,
    checkpoint_every: int = 1,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        RUNNER,
        "--mode", "train",
        "--device", "cuda",
        "--selector", SELECTORS[method],
        "--epochs", str(epochs),
        "--steps-per-epoch", str(steps_per_epoch),
        "--batch-size", str(batch_size),
        "--learning-rate", str(protocol.LEARNING_RATE),
        "--max-grad-norm", str(protocol.MAX_GRAD_NORM),
        "--train-vehicle-policy", vehicle_policy,
        # One rollout-baseline t-test per epoch, as in Algorithm 2.
        "--baseline-update-interval", str(steps_per_epoch),
        "--validation-size", str(protocol.VALIDATION_INSTANCES),
        "--validation-rollouts", "1",
        "--checkpoint", str(checkpoint_root / method / "best.pt"),
        "--output-dir", str(output_root / method),
        "--checkpoint-every", str(checkpoint_every),
        "--force-train",
    ]
    if early_stop_tolerance is not None:
        cmd.extend(["--early-stop-tolerance", str(early_stop_tolerance)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=list(SELECTORS))
    parser.add_argument("--epochs", type=int, default=protocol.EPOCHS)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=protocol.ITERATIONS_PER_EPOCH,
        help="Paper budget is 1000; reduced budgets must be reported as such.",
    )
    parser.add_argument("--batch-size", type=int, default=protocol.BATCH_SIZE)
    parser.add_argument(
        "--vehicle-policy",
        default="sample_logp",
        choices=("public_argmax", "argmax_logp", "sample_logp"),
    )
    parser.add_argument(
        "--early-stop-tolerance",
        type=float,
        default=protocol.ACCEPTANCE_TOLERANCE,
        help="Stop early when all rates are within this error (default 0.04)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default 1)",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "experiments/checkpoints/repro100",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments/results/repro100",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Launch all methods concurrently on one GPU.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = {
        method: command(
            method,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            vehicle_policy=args.vehicle_policy,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            early_stop_tolerance=args.early_stop_tolerance,
            checkpoint_every=args.checkpoint_every,
        )
        for method in args.methods
    }

    updates = args.epochs * args.steps_per_epoch
    print(
        f"{len(commands)} methods x {args.epochs} epochs x "
        f"{args.steps_per_epoch} steps = {updates} parameter updates each "
        f"(paper budget {protocol.EPOCHS * protocol.ITERATIONS_PER_EPOCH})"
    )
    for method, argv in commands.items():
        print(f"\n{method} ({SELECTORS[method]}):\n  " + " ".join(argv))
    if args.dry_run:
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.parallel:
        processes = {}
        for method, argv in commands.items():
            log = args.output_root / f"{method}_train.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            processes[method] = (
                subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT),
                handle,
            )
            print(f"launched {method} -> {log}")
        failures = []
        for method, (process, handle) in processes.items():
            code = process.wait()
            handle.close()
            print(f"{method} exited with {code}")
            if code != 0:
                failures.append(method)
    else:
        failures = []
        for method, argv in commands.items():
            log = args.output_root / f"{method}_train.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as handle:
                code = subprocess.call(
                    argv, stdout=handle, stderr=subprocess.STDOUT
                )
            print(f"{method} exited with {code} -> {log}")
            if code != 0:
                failures.append(method)

    print(f"\ntotal wall clock: {(time.time() - started) / 3600:.2f} h")
    if failures:
        print("failed: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
