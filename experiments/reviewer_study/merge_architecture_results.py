"""Merge separately scheduled equal-budget architecture batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=Path("experiments/results/architecture"))
    parser.add_argument("--extra", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    primary_raw = args.primary / "raw"
    extra_raw = args.extra / "raw"
    instances = pd.concat([
        pd.read_csv(primary_raw / "architecture_instances.csv"),
        pd.read_csv(extra_raw / "architecture_instances.csv"),
    ], ignore_index=True)
    instances = instances.drop_duplicates(
        ["selector", "train_seed", "scenario", "instance"], keep="last"
    )
    instances.to_csv(primary_raw / "architecture_instances.csv", index=False)

    resources = pd.concat([
        pd.read_csv(primary_raw / "architecture_resources.csv"),
        pd.read_csv(extra_raw / "architecture_resources.csv"),
    ], ignore_index=True)
    resources = resources.drop_duplicates(["selector", "seed"], keep="last")
    resources.to_csv(primary_raw / "architecture_resources.csv", index=False)
    print(instances.groupby("selector").size())


if __name__ == "__main__":
    main()
