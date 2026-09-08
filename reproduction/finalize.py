"""Wait for training to finish, then produce the audited Table I.

Chaining the two steps means the reported table is always generated from the
checkpoints that training actually selected, with no manual step in between
where a stale checkpoint could be picked up.
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

from reproduction.train_all import SELECTORS  # noqa: E402


def completed(log: Path, epochs: int) -> bool:
    if not log.exists():
        return False
    text = log.read_text(encoding="utf-8", errors="replace")
    return f"epoch={epochs:03d} validation" in text or "best checkpoint" in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO_ROOT / "experiments/results/repro100",
    )
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=14.0)
    parser.add_argument("--interim-every", type=int, default=0,
                        help="If set, evaluate every N polls while waiting.")
    parser.add_argument(
        "--checkpoint-name",
        default="best_last.pt",
        help=(
            "Final-epoch weights by default.  The runner's own 'best.pt' is "
            "selected on validation QoS first, which freezes early for a method "
            "whose QoS peak is never re-attained and so is not comparable "
            "across methods; see README section 5."
        ),
    )
    args = parser.parse_args()

    deadline = time.time() + args.timeout_hours * 3600
    polls = 0
    while time.time() < deadline:
        pending = [
            method
            for method in SELECTORS
            if not completed(args.log_root / f"{method}_train.log", args.epochs)
        ]
        if not pending:
            print("all methods finished training", flush=True)
            break
        polls += 1
        print(
            f"[{time.strftime('%H:%M:%S')}] waiting on: {', '.join(pending)}",
            flush=True,
        )
        if args.interim_every and polls % args.interim_every == 0:
            subprocess.call([
                sys.executable, "-u", "-m", "reproduction.evaluate_table1",
                "--timing-repetitions", "1", "--skip-greedy",
                "--checkpoint-name", args.checkpoint_name,
                "--output", str(REPO_ROOT / "experiments/results/reproduction_n20_interim"),
            ])
        time.sleep(args.poll_seconds)
    else:
        print("timed out waiting for training", flush=True)

    # Both selection rules are reported: the uniform one is the headline table,
    # the runner's own is kept so the difference stays auditable.
    print("\nrunning final audited evaluation", flush=True)
    code = subprocess.call([
        sys.executable, "-u", "-m", "reproduction.evaluate_table1",
        "--measure-memory", "--timing-repetitions", "3",
        "--checkpoint-name", args.checkpoint_name,
    ])
    if args.checkpoint_name != "best.pt":
        print("\nrunning comparison evaluation on the runner's own selection",
              flush=True)
        subprocess.call([
            sys.executable, "-u", "-m", "reproduction.evaluate_table1",
            "--timing-repetitions", "1", "--checkpoint-name", "best.pt",
            "--output",
            str(REPO_ROOT / "experiments/results/reproduction_n20_qos_selected"),
        ])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
