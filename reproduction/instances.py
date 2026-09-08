"""Instance generation for the n=20 DCVRP reproduction.

Two generators are provided, because the released ``data.py`` and the prose of
Section IV-A do not describe the same distribution:

``"manuscript"`` (legacy function name; the corrected reported protocol)
    Continuous coordinates in the unit square, exactly ``round(n*phi)`` dynamic
    customers, and independent disclosure draws using the released per-slot
    Poisson rates.  This intentionally combines the manuscript's spatial/count
    semantics with the only executable revelation-rate vector in the release.

``"released"``
    A transcription of ``DCVRP_Dataset.generate`` plus ``normalize``: integer
    coordinates in ``[0,100]`` min-max normalized over the batch, Bernoulli
    dynamic membership, and one shared vector of Poisson draws per batch.
    ``check_released_equivalence`` proves the transcription byte-for-byte, so a
    reader can verify it instead of trusting it.

Reporting both, with the measured difference between them, is what lets a
reader tell which choices Table I is and is not sensitive to.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch

from . import protocol


@dataclass(frozen=True)
class InstanceBatch:
    """Normalized instances plus the physical quantities needed to audit them."""

    nodes: torch.Tensor           # (B, n+1, 5) normalized model input
    vehicle_count: int
    vehicle_capacity: float       # normalized, always 1.0
    vehicle_speed: float          # normalized coordinate units per unit time
    dynamic_mask: torch.Tensor    # (B, n) bool, True where a customer is dynamic
    disclosure_minutes: torch.Tensor  # (B, n) physical minutes, 0 when static
    location_scale: float         # divisor used to normalize the coordinates

    def __len__(self) -> int:
        return self.nodes.size(0)

    @property
    def realized_dynamic_rate(self) -> float:
        return float(self.dynamic_mask.float().mean())

    def sha256(self) -> str:
        array = self.nodes.detach().cpu().contiguous().numpy()
        return hashlib.sha256(array.tobytes()).hexdigest()


def seed_all(seed: int) -> None:
    """Seed every stream the generator and the models draw from."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disclosure_rates(customer_count: int) -> torch.Tensor:
    """Per-slot Poisson rates ``lambda_i``.

    The released generator spreads the rates linearly from 1 to the horizon
    across the ordered customer slots.  Their mean is exactly ``(1 + T) / 2``,
    the scalar rate quoted in the manuscript.
    """

    return torch.linspace(
        1.0, float(protocol.HORIZON_MINUTES), steps=customer_count
    )


def generate_released(
    batch_size: int,
    dynamic_rate: float,
    *,
    customer_count: int = protocol.CUSTOMER_COUNT,
    vehicle_count: int = protocol.VEHICLE_COUNT,
    disclosure_sharing: str = "released",
) -> InstanceBatch:
    """Generate one normalized batch at a single degree of dynamism.

    ``disclosure_sharing`` selects the disclosure-time sampling:

    ``"released"``
        One vector of ``customer_count`` Poisson draws is shared by every
        instance in the batch.  This is what ``data.py`` does, because it calls
        ``create_logistics_distribution`` without forwarding ``batch_size``.

    ``"independent"``
        A fresh Poisson draw per instance and slot.  Statistically preferable
        and reported as a sensitivity check.
    """

    if disclosure_sharing not in ("released", "independent"):
        raise ValueError(f"unknown disclosure_sharing: {disclosure_sharing}")

    size = (batch_size, customer_count, 1)

    locations = torch.randint(
        *protocol.LOCATION_RANGE,
        (batch_size, customer_count + 1, 2),
        dtype=torch.float,
    )
    demands = torch.randint(*protocol.DEMAND_RANGE, size, dtype=torch.float)
    durations = torch.randint(
        *protocol.SERVICE_DURATION_RANGE, size, dtype=torch.float
    )

    is_dynamic = torch.empty(size).bernoulli_(dynamic_rate)

    rates = disclosure_rates(customer_count).view(1, customer_count, 1)
    draw_batch = 1 if disclosure_sharing == "released" else batch_size
    poisson_values = torch.poisson(rates.expand(draw_batch, -1, -1)).clamp_(
        min=1.0, max=float(protocol.HORIZON_MINUTES)
    )
    disclosure = is_dynamic * poisson_values

    customers = torch.cat((locations[:, 1:], demands, durations, disclosure), 2)
    depot = torch.zeros((batch_size, 1, 5))
    depot[:, :, :2] = locations[:, 0:1]
    nodes = torch.cat((depot, customers), 1)

    # ---- normalization, transcribed from DCVRP_Dataset.normalize ------------
    location_offset = nodes[:, :, :2].min().item()
    location_scale = nodes[:, :, :2].max().item() - location_offset
    time_scale = float(protocol.HORIZON_MINUTES)

    nodes[:, :, :2] -= location_offset
    nodes[:, :, :2] /= location_scale
    nodes[:, :, 2] /= protocol.VEHICLE_CAPACITY
    nodes[:, :, 3:] /= time_scale

    return InstanceBatch(
        nodes=nodes,
        vehicle_count=vehicle_count,
        vehicle_capacity=1.0,
        vehicle_speed=protocol.VEHICLE_SPEED * time_scale / location_scale,
        dynamic_mask=is_dynamic.squeeze(-1).bool(),
        disclosure_minutes=disclosure.squeeze(-1),
        location_scale=location_scale,
    )


def generate_manuscript_split(
    seed: int,
    instances: int,
    *,
    dynamic_rates=protocol.DYNAMIC_RATES,
    customer_count: int = protocol.CUSTOMER_COUNT,
    vehicle_count: int = protocol.VEHICLE_COUNT,
) -> dict[float, object]:
    """Build the reported evaluation split.

    The dynamic-rate variants are *paired*: coordinates, demands, service
    durations, and disclosure draws are shared, and the dynamic sets are nested,
    so raising ``phi`` only converts further requests from static to dynamic.
    Comparing methods across rates on common random numbers isolates the effect
    of the dynamic rate from instance difficulty.
    """

    seed_all(seed)
    rates = tuple(sorted(float(rate) for rate in dynamic_rates))
    if not rates:
        raise ValueError("dynamic_rates cannot be empty")
    if rates[0] < 0.0 or rates[-1] > 1.0:
        raise ValueError("dynamic rates must lie in [0, 1]")

    # Keep the physical instances and every disclosure draw common across the
    # paired rate variants.  Crucially, use the released per-customer-slot
    # rates here.  The former implementation delegated to
    # ``generate_paired_paper_datasets``, whose scalar Poisson(240.5) process
    # contradicted this package's protocol.py, README and disclosure audit.
    coordinates = torch.rand(instances, customer_count + 1, 2)
    demands = torch.randint(
        5, 42, (instances, customer_count, 1), dtype=torch.int64
    ).float()
    service_minutes = torch.randint(
        10, 32, (instances, customer_count, 1), dtype=torch.int64
    ).float()
    slot_rates = disclosure_rates(customer_count).view(1, customer_count, 1)
    disclosure_minutes = torch.poisson(
        slot_rates.expand(instances, -1, -1)
    ).clamp_(min=1.0, max=float(protocol.HORIZON_MINUTES))
    dynamic_order = torch.argsort(torch.rand(instances, customer_count), dim=1)
    ranks = torch.empty_like(dynamic_order)
    ranks.scatter_(
        1,
        dynamic_order,
        torch.arange(customer_count).view(1, -1).expand(instances, -1),
    )

    from data import DCVRP_Dataset

    paired: dict[float, DCVRP_Dataset] = {}
    for rate in rates:
        dynamic_count = int(round(customer_count * rate))
        dynamic = ranks < dynamic_count
        release = torch.where(
            dynamic.unsqueeze(-1),
            disclosure_minutes,
            torch.zeros_like(disclosure_minutes),
        )
        customer_features = torch.cat(
            [
                coordinates[:, 1:, :],
                demands / protocol.VEHICLE_CAPACITY,
                service_minutes / protocol.HORIZON_MINUTES,
                release / protocol.HORIZON_MINUTES,
            ],
            dim=2,
        )
        depot = torch.zeros(instances, 1, 5)
        depot[:, :, :2] = coordinates[:, :1, :]
        dataset = DCVRP_Dataset(
            vehicle_count,
            veh_capa=1.0,
            veh_speed=protocol.VEHICLE_SPEED * protocol.HORIZON_MINUTES,
            nodes=torch.cat([depot, customer_features], dim=1),
            cust_mask=None,
        )
        dataset.paper_dynamic_rates = torch.full((instances,), rate)
        dataset.paper_dynamic_counts = torch.full(
            (instances,), dynamic_count, dtype=torch.int64
        )
        paired[rate] = dataset
    return paired


def generate_released_split(
    seed: int,
    instances: int,
    *,
    dynamic_rates=protocol.DYNAMIC_RATES,
    disclosure_sharing: str = "released",
) -> dict[float, InstanceBatch]:
    """Build a released-generator split: an independent batch per dynamic rate.

    The seed is reset once before the split and the rates are visited in
    ascending order, so the split is a pure function of ``(seed, instances,
    rates, disclosure_sharing)``.
    """

    seed_all(seed)
    return {
        float(rate): generate_released(
            instances, float(rate), disclosure_sharing=disclosure_sharing
        )
        for rate in sorted(float(rate) for rate in dynamic_rates)
    }


def check_released_equivalence(
    batch_size: int = 64, dynamic_rate: float = 0.5
) -> None:
    """Assert this transcription matches the released generator exactly."""

    from data import DCVRP_Dataset

    seed_all(999)
    released = DCVRP_Dataset.generate(
        batch_size,
        cust_count=protocol.CUSTOMER_COUNT,
        veh_count=protocol.VEHICLE_COUNT,
        veh_capa=protocol.VEHICLE_CAPACITY,
        veh_speed=protocol.VEHICLE_SPEED,
        min_cust_count=None,
        cust_loc_range=protocol.LOCATION_RANGE,
        cust_dem_range=protocol.DEMAND_RANGE,
        horizon=protocol.HORIZON_MINUTES,
        cust_dur_range=protocol.SERVICE_DURATION_RANGE,
        dod=dynamic_rate,
    )
    released.normalize()

    seed_all(999)
    ours = generate_released(
        batch_size, dynamic_rate, disclosure_sharing="released"
    )

    if not torch.equal(released.nodes, ours.nodes):
        difference = (released.nodes - ours.nodes).abs().max().item()
        raise AssertionError(
            f"transcription differs from data.py (max abs {difference:g})"
        )
    if abs(released.veh_speed - ours.vehicle_speed) > 1e-9:
        raise AssertionError(
            f"speed differs: {released.veh_speed} vs {ours.vehicle_speed}"
        )
