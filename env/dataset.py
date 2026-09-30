"""Synthetic DCVRP instances (Section IV-A).

- Coordinates: uniform in the unit square.
- Travel speed: 1; after dividing times by T=480 the normalized speed is 480.
- Demands: integers in [5, 41], divided by Q=150.
- Service times: integers in [10, 31] minutes, divided by T.
- Dynamism: exactly n' = round(n φ) nested dynamic customers (Eq. 10).
- Default arrivals: homogeneous Poisson process on (0, T], so a_i ~ Uniform(0, T].
  Nested extra customers at higher φ therefore appear throughout the horizon,
  and mean route cost increases with dynamism.
- Optional `--revelation poisson`: truncated Poisson PMF from Eq. 34, which
  concentrates around t = (1+T)/2 and flattens the φ trend.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_CUSTOMER_COUNT = 20
DEFAULT_VEHICLE_COUNT = 4
DEFAULT_VEHICLE_CAPACITY = 150.0
DEFAULT_VEHICLE_SPEED = 1.0
DEFAULT_HORIZON = 480.0
DEFAULT_DECISION_INTERVALS = 10
DEFAULT_DYNAMIC_RATES = (0.10, 0.25, 0.50, 0.75)
REVELATION_POISSON = "poisson"
REVELATION_HPP = "hpp"
REVELATION_EARLY_PEAK = "early_peak"
REVELATION_LATE_PEAK = "late_peak"
DEFAULT_REVELATION = REVELATION_HPP
REVELATION_MODES = (REVELATION_POISSON, REVELATION_HPP, REVELATION_EARLY_PEAK, REVELATION_LATE_PEAK)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DCVRPDataset(Dataset):
    def __init__(
        self,
        vehicle_count,
        vehicle_capacity,
        vehicle_speed,
        nodes,
        customer_mask=None,
        location_scale: float | None = None,
        revelation: str = DEFAULT_REVELATION,
    ):
        self.veh_count = vehicle_count
        self.veh_capa = vehicle_capacity
        self.veh_speed = vehicle_speed
        self.nodes = nodes
        self.cust_mask = customer_mask
        self.location_scale = 1.0 if location_scale is None else float(location_scale)
        self.revelation = str(revelation)
        self.batch_size, self.nodes_count, _ = nodes.size()

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        if self.cust_mask is None:
            return self.nodes[index]
        return self.nodes[index], self.cust_mask[index]


def _validate_revelation(revelation: str) -> str:
    mode = str(revelation).strip().lower()
    if mode not in REVELATION_MODES:
        raise ValueError(f"Unknown revelation mode: {revelation}")
    return mode


import os

SPATIAL_DISTRIBUTIONS = ("uniform", "clustered", "mixed", "real")
DEFAULT_DISTRIBUTION = "uniform"

_REAL_DATA_CACHE = {}

def _get_real_data(path: str) -> torch.Tensor:
    if path not in _REAL_DATA_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Real-world data file not found: {path}. Please provide a CSV with coordinates.")
        # Load coordinates (assume comma separated, 2 columns: x, y)
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        tensor_data = torch.from_numpy(data)
        # Auto-normalize to [0, 1]^2 as required by the algorithm
        min_vals = tensor_data.min(dim=0, keepdim=True).values
        max_vals = tensor_data.max(dim=0, keepdim=True).values
        tensor_data = (tensor_data - min_vals) / (max_vals - min_vals + 1e-8)
        _REAL_DATA_CACHE[path] = tensor_data
    return _REAL_DATA_CACHE[path]


def _generate_spatial_coordinates(batch_size: int, num_nodes: int, distribution: str, real_data_path: str | None = None) -> torch.Tensor:
    dist = str(distribution).lower()
    if dist == "uniform":
        return torch.rand(batch_size, num_nodes, 2)
    elif dist == "clustered":
        num_clusters = 4
        centers = torch.rand(batch_size, num_clusters, 2)
        assignments = torch.randint(0, num_clusters, (batch_size, num_nodes))
        node_centers = torch.gather(centers, 1, assignments.unsqueeze(-1).expand(-1, -1, 2))
        noise = torch.randn(batch_size, num_nodes, 2) * 0.07
        return torch.clamp(node_centers + noise, 0.0, 1.0)
    elif dist == "mixed":
        num_clusters = 4
        centers = torch.rand(batch_size, num_clusters, 2)
        assignments = torch.randint(0, num_clusters, (batch_size, num_nodes))
        node_centers = torch.gather(centers, 1, assignments.unsqueeze(-1).expand(-1, -1, 2))
        noise = torch.randn(batch_size, num_nodes, 2) * 0.07
        clustered_coords = torch.clamp(node_centers + noise, 0.0, 1.0)
        uniform_coords = torch.rand(batch_size, num_nodes, 2)
        mask = torch.rand(batch_size, num_nodes, 1) < 0.5
        return torch.where(mask, clustered_coords, uniform_coords)
    elif dist == "real":
        if not real_data_path:
            raise ValueError("real_data_path must be provided for 'real' distribution")
        real_coords = _get_real_data(real_data_path)
        # Uniformly sample nodes from the real city graph
        indices = torch.randint(0, len(real_coords), (batch_size, num_nodes))
        return real_coords[indices]
    else:
        raise ValueError(f"Unknown spatial distribution: {distribution}")


def _base_draws(
    batch_size: int,
    customer_count: int,
    horizon: float,
    revelation: str,
    distribution: str = DEFAULT_DISTRIBUTION,
    real_data_path: str | None = None,
):
    coordinates = _generate_spatial_coordinates(batch_size, customer_count + 1, distribution, real_data_path)
    demands = torch.randint(5, 42, (batch_size, customer_count, 1)).float()
    service = torch.randint(10, 32, (batch_size, customer_count, 1)).float()
    if revelation == REVELATION_POISSON:
        # Eq. 34: P(a_i = t) ∝ λ^t e^{-λ} / t! on {1, ..., T}.
        horizon_int = int(horizon)
        lam = (1.0 + float(horizon)) / 2.0
        times = torch.arange(1, horizon_int + 1, dtype=torch.float32)
        log_unnorm = times.mul(math.log(lam)).sub_(lam).sub_(torch.lgamma(times + 1.0))
        index = torch.distributions.Categorical(logits=log_unnorm).sample(
            (batch_size, customer_count)
        )
        disclosure = (index + 1).unsqueeze(-1).float()
    elif revelation == REVELATION_EARLY_PEAK:
        # Uniform in [0, 0.3T]
        disclosure = torch.rand(batch_size, customer_count, 1) * (0.3 * float(horizon))
    elif revelation == REVELATION_LATE_PEAK:
        # Uniform in [0.6T, 0.9T]
        disclosure = (0.6 + torch.rand(batch_size, customer_count, 1) * 0.3) * float(horizon)
    else:
        # Homogeneous Poisson process: a_i ~ Uniform(0, T].
        disclosure = (1.0 - torch.rand(batch_size, customer_count, 1)) * float(horizon)
    order = torch.argsort(torch.rand(batch_size, customer_count), dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(customer_count).view(1, -1).expand(batch_size, -1),
    )
    return coordinates, demands, service, disclosure, ranks


def _pack(
    coordinates,
    demands,
    service,
    release,
    vehicle_count,
    vehicle_capacity,
    vehicle_speed,
    horizon,
    revelation,
):
    customers = torch.cat(
        [
            coordinates[:, 1:, :],
            demands / float(vehicle_capacity),
            service / float(horizon),
            release / float(horizon),
        ],
        dim=2,
    )
    depot = torch.zeros(coordinates.size(0), 1, 5)
    depot[:, :, :2] = coordinates[:, :1, :]
    return DCVRPDataset(
        vehicle_count,
        1.0,
        float(vehicle_speed) * float(horizon),
        torch.cat([depot, customers], dim=1),
        location_scale=1.0,
        revelation=revelation,
    )


def generate_dataset(
    batch_size: int,
    dynamic_rate: float = 0.5,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    vehicle_count: int = DEFAULT_VEHICLE_COUNT,
    vehicle_capacity: float = DEFAULT_VEHICLE_CAPACITY,
    vehicle_speed: float = DEFAULT_VEHICLE_SPEED,
    horizon: float = DEFAULT_HORIZON,
    seed: int | None = None,
    revelation: str = DEFAULT_REVELATION,
    distribution: str = DEFAULT_DISTRIBUTION,
    real_data_path: str | None = None,
) -> DCVRPDataset:
    if seed is not None:
        set_seed(seed)
    revelation = _validate_revelation(revelation)
    coordinates, demands, service, disclosure, ranks = _base_draws(
        batch_size, customer_count, horizon, revelation, distribution, real_data_path
    )
    dynamic_count = int(round(customer_count * float(dynamic_rate)))
    release = torch.where(
        (ranks < dynamic_count).unsqueeze(-1),
        disclosure,
        torch.zeros_like(disclosure),
    )
    return _pack(
        coordinates,
        demands,
        service,
        release,
        vehicle_count,
        vehicle_capacity,
        vehicle_speed,
        horizon,
        revelation,
    )


def generate_evaluation_split(
    instances: int = 100,
    dynamic_rates: Sequence[float] = DEFAULT_DYNAMIC_RATES,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    vehicle_count: int = DEFAULT_VEHICLE_COUNT,
    vehicle_capacity: float = DEFAULT_VEHICLE_CAPACITY,
    vehicle_speed: float = DEFAULT_VEHICLE_SPEED,
    horizon: float = DEFAULT_HORIZON,
    seed: int = 20260821,
    revelation: str = DEFAULT_REVELATION,
    distribution: str = DEFAULT_DISTRIBUTION,
    real_data_path: str | None = None,
) -> dict[float, DCVRPDataset]:
    """Shared instances; larger phi adds the same nested dynamic set."""
    set_seed(seed)
    revelation = _validate_revelation(revelation)
    coordinates, demands, service, disclosure, ranks = _base_draws(
        instances, customer_count, horizon, revelation, distribution, real_data_path
    )
    split = {}
    for rate in sorted(float(value) for value in dynamic_rates):
        dynamic_count = int(round(customer_count * rate))
        release = torch.where(
            (ranks < dynamic_count).unsqueeze(-1),
            disclosure,
            torch.zeros_like(disclosure),
        )
        split[rate] = _pack(
            coordinates,
            demands,
            service,
            release,
            vehicle_count,
            vehicle_capacity,
            vehicle_speed,
            horizon,
            revelation,
        )
    return split
