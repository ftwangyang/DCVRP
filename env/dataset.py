"""Synthetic DCVRP instances (Section IV-A).

- Coordinates: uniform in the unit square.
- Travel speed: 1; after dividing times by T=480 the normalized speed is 480.
- Demands: integers in [5, 41], divided by Q=150.
- Service times: integers in [10, 31] minutes, divided by T.
- Dynamism: exactly n' = round(n φ) nested dynamic customers (Eq. 10).
- Default arrivals: truncated Poisson PMF on {1, ..., T} (Eq. 34).
- Optional `--revelation hpp`: homogeneous Poisson process, a_i ~ Uniform(0, T].
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
DEFAULT_REVELATION = REVELATION_POISSON
REVELATION_MODES = (REVELATION_POISSON, REVELATION_HPP)


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


def _base_draws(
    batch_size: int,
    customer_count: int,
    horizon: float,
    revelation: str,
):
    coordinates = torch.rand(batch_size, customer_count + 1, 2)
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
) -> DCVRPDataset:
    if seed is not None:
        set_seed(seed)
    revelation = _validate_revelation(revelation)
    coordinates, demands, service, disclosure, ranks = _base_draws(
        batch_size, customer_count, horizon, revelation
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
) -> dict[float, DCVRPDataset]:
    """Shared instances; larger phi adds the same nested dynamic set."""
    set_seed(seed)
    revelation = _validate_revelation(revelation)
    coordinates, demands, service, disclosure, ranks = _base_draws(
        instances, customer_count, horizon, revelation
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
