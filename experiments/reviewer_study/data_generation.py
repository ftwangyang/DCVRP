"""Controlled DCVRP instance generation for reviewer experiments."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

from data import DCVRP_Dataset


ArrivalProfile = Literal[
    "uniform",
    "early",
    "late",
    "two_peak",
    "clustered",
    "nhpp",
]


def _sample_locations(
    rng: np.random.Generator,
    batch_size: int,
    customer_count: int,
    spatial: str,
) -> np.ndarray:
    if spatial == "uniform":
        return rng.uniform(0.0, 1.0, size=(batch_size, customer_count + 1, 2))

    locations = rng.uniform(0.0, 1.0, size=(batch_size, customer_count + 1, 2))
    cluster_count = 4
    for b in range(batch_size):
        centers = rng.uniform(0.1, 0.9, size=(cluster_count, 2))
        if spatial == "cluster":
            assignments = rng.integers(0, cluster_count, size=customer_count)
            points = centers[assignments] + rng.normal(0.0, 0.055, size=(customer_count, 2))
            locations[b, 1:] = np.clip(points, 0.0, 1.0)
        elif spatial == "mixed":
            clustered = rng.random(customer_count) < 0.5
            assignments = rng.integers(0, cluster_count, size=customer_count)
            points = centers[assignments] + rng.normal(0.0, 0.055, size=(customer_count, 2))
            locations[b, 1:][clustered] = np.clip(points[clustered], 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported spatial distribution: {spatial}")
    return locations


def _arrival_values(
    rng: np.random.Generator,
    count: int,
    profile: ArrivalProfile,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.float32)
    if profile == "uniform":
        values = rng.uniform(0.02, 0.98, size=count)
    elif profile == "early":
        values = 0.02 + 0.96 * rng.beta(2.0, 5.0, size=count)
    elif profile == "late":
        values = 0.02 + 0.96 * rng.beta(5.0, 2.0, size=count)
    elif profile == "two_peak":
        choose_late = rng.random(count) < 0.5
        values = rng.normal(0.28, 0.055, size=count)
        values[choose_late] = rng.normal(0.76, 0.055, size=choose_late.sum())
        values = np.clip(values, 0.02, 0.98)
    elif profile == "clustered":
        centers = rng.uniform(0.12, 0.88, size=3)
        values = centers[rng.integers(0, 3, size=count)] + rng.normal(0.0, 0.025, size=count)
        values = np.clip(values, 0.02, 0.98)
    elif profile == "nhpp":
        # Piecewise non-homogeneous process with morning and afternoon peaks.
        grid = np.linspace(0.02, 0.98, 512)
        intensity = (
            0.25
            + 1.3 * np.exp(-0.5 * ((grid - 0.30) / 0.09) ** 2)
            + 1.0 * np.exp(-0.5 * ((grid - 0.73) / 0.11) ** 2)
        )
        probabilities = intensity / intensity.sum()
        values = rng.choice(grid, size=count, replace=True, p=probabilities)
    else:
        raise ValueError(f"Unsupported arrival profile: {profile}")
    return values.astype(np.float32)


def generate_dataset(
    batch_size: int,
    customer_count: int = 20,
    vehicle_count: int = 4,
    dynamic_ratio: float = 0.5,
    arrival_profile: ArrivalProfile = "uniform",
    spatial: str = "uniform",
    seed: int = 1234,
    vehicle_capacity: float = 1.0,
    vehicle_speed: float = 4.8,
) -> DCVRP_Dataset:
    """Generate normalized instances with an exact per-instance dynamic ratio.

    Coordinates, service duration, and disclosure time are normalized to the
    same scale used by the manuscript model: coordinates in [0, 1], planning
    horizon 1.0, capacity 1.0, and speed 4.8 normalized distance units per
    planning horizon.
    """
    if not 0.0 <= dynamic_ratio <= 1.0:
        raise ValueError("dynamic_ratio must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    locations = _sample_locations(rng, batch_size, customer_count, spatial)
    demands = rng.integers(5, 41, size=(batch_size, customer_count, 1)) / 150.0
    durations = rng.integers(10, 31, size=(batch_size, customer_count, 1)) / 480.0
    appearances = np.zeros((batch_size, customer_count, 1), dtype=np.float32)

    dynamic_count = int(round(customer_count * dynamic_ratio))
    for b in range(batch_size):
        dynamic_indices = rng.choice(customer_count, size=dynamic_count, replace=False)
        appearances[b, dynamic_indices, 0] = _arrival_values(
            rng, dynamic_count, arrival_profile
        )

    customers = np.concatenate(
        [locations[:, 1:], demands, durations, appearances], axis=2
    ).astype(np.float32)
    depot = np.zeros((batch_size, 1, DCVRP_Dataset.CUST_FEAT_SIZE), dtype=np.float32)
    depot[:, :, :2] = locations[:, :1]
    nodes = torch.from_numpy(np.concatenate([depot, customers], axis=1))
    return DCVRP_Dataset(
        vehicle_count,
        vehicle_capacity,
        vehicle_speed,
        nodes,
        cust_mask=None,
    )


def mean_revelation_time(dataset: DCVRP_Dataset) -> float:
    values = dataset.nodes[:, 1:, 4]
    dynamic = values[values > 0]
    return float(dynamic.mean()) if dynamic.numel() else math.nan

