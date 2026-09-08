"""Materialize and hash the immutable instance splits.

The synthetic data are a pure function of the recorded seeds, so the archival
release does not strictly need the tensors.  Writing them anyway, with SHA-256
hashes of both the files and the raw node arrays, lets a reader confirm that the
evaluation split they regenerate is the split the reported numbers came from.

Three splits are written:

* ``train``  -- the public training lifecycle generates one 1000 x 100 pool
  and reshuffles/reuses it across 100 epochs, so the complete 100,000-instance
  pool is archived;
* ``validation`` -- used for checkpoint selection only, and may be inspected
  freely during development;
* ``test`` -- evaluated once, after the checkpoint and the decoding rule are
  frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction import instances, protocol  # noqa: E402

HERE = Path(__file__).resolve().parent


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize(dataset) -> dict:
    return {
        "nodes": dataset.nodes.cpu(),
        "vehicle_count": int(dataset.veh_count),
        "vehicle_capacity_normalized": float(dataset.veh_capa),
        "vehicle_speed_normalized": float(dataset.veh_speed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "data")
    parser.add_argument(
        "--train-instances",
        type=int,
        default=100_000,
        help="Sample of the training stream to archive.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol_name": protocol.PROTOCOL_NAME,
        "generator": "reproduction.instances.generate_manuscript_split",
        "seeds": {
            "train": protocol.TRAIN_SEED,
            "validation": protocol.VALIDATION_SEED,
            "test": protocol.TEST_SEED,
        },
        "splits": {},
    }

    plan = (
        ("validation", protocol.VALIDATION_SEED, protocol.VALIDATION_INSTANCES),
        ("test", protocol.TEST_SEED, protocol.TEST_INSTANCES),
        ("train_sample", protocol.TRAIN_SEED, args.train_instances),
    )
    for name, seed, count in plan:
        split = instances.generate_manuscript_split(seed, count)
        payload = {
            "protocol_name": protocol.PROTOCOL_NAME,
            "split": name,
            "seed": seed,
            "instances_per_rate": count,
            "paired_nested_rates": True,
            "rates": {
                f"{rate:.2f}": serialize(dataset)
                for rate, dataset in split.items()
            },
        }
        path = output / f"{name}_n20_m4_seed{seed}.pt"
        torch.save(payload, path)
        manifest["splits"][name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "seed": seed,
            "instances_per_rate": count,
            "size_bytes": path.stat().st_size,
            "file_sha256": file_sha256(path),
            "nodes_sha256": {
                f"{rate:.2f}": tensor_sha256(dataset.nodes)
                for rate, dataset in split.items()
            },
        }
        print(f"{name}: {count} instances per rate -> {path.name}")

    manifest_path = output / "instance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nmanifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
