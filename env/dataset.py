"""Synthetic DCVRP instances following Section IV-A, with homogeneous Poisson arrivals.

- Coordinates: continuous uniform in the [0, 1] unit square.
- Traveling speed: 1; after dividing times by T=480 the normalized speed is 480.
- Demands: integers in [5, 41], divided by Q=150.
- Service times: integers in [10, 31] minutes, divided by T.
- Dynamism: exact n'=round(n phi) nested customers (Eq. 10).
- Revelation: homogeneous Poisson process. Conditioned on a fixed dynamic
  count, arrival times are i.i.d. Uniform(0, T]. Independent Poisson(240.5)
  draws clump around interval 5 and turn phi=75% into a second static VRP.
"""

from __future__ import annotations

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
    ):
        self.veh_count = vehicle_count
        self.veh_capa = vehicle_capacity
        self.veh_speed = vehicle_speed
        self.nodes = nodes
        self.cust_mask = customer_mask
        self.location_scale = 1.0 if location_scale is None else float(location_scale)
        self.batch_size, self.nodes_count, _ = nodes.size()

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        if self.cust_mask is None:
            return self.nodes[index]
        return self.nodes[index], self.cust_mask[index]


def _base_draws(batch_size: int, customer_count: int, horizon: float):
    coordinates = torch.rand(batch_size, customer_count + 1, 2)
    demands = torch.randint(5, 42, (batch_size, customer_count, 1)).float()
    service = torch.randint(10, 32, (batch_size, customer_count, 1)).float()
    # Homogeneous Poisson process on (0, T]: given N=n', times ~ Uniform(0, T].
    # torch.rand is [0, 1), so (1-u)*T is (0, T].
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
) -> DCVRPDataset:
    if seed is not None:
        set_seed(seed)
    coordinates, demands, service, disclosure, ranks = _base_draws(
        batch_size, customer_count, horizon
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
) -> dict[float, DCVRPDataset]:
    """Shared instances; larger phi adds the same nested dynamic set."""
    set_seed(seed)
    coordinates, demands, service, disclosure, ranks = _base_draws(
        instances, customer_count, horizon
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
        )
    return split
