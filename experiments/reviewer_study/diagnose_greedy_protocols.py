"""Compare the two manuscript-adjacent Greedy execution protocols.

This diagnostic does not fit parameters or alter costs.  It evaluates the
same immutable manifests with (1) the periodic-static executed-path simulator
and (2) the manuscript baseline's event-driven nearest-idle rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import DCVRP_Dataset

from .executed_path_multivehicle_env import ExecutedPathFleet
from .paper_dcvrp import PAPER_HORIZON_MINUTES, PAPER_VEHICLE_CAPACITY
from .paper_greedy import run_paper_greedy
from .paper_multivehicle_env import PaperInstance, nearest_idle_routes, paper_vehicle_count


PAPER_GREEDY = {
    (20, 0.10): 9.07,
    (20, 0.25): 9.69,
    (20, 0.50): 11.25,
    (20, 0.75): 12.43,
    (35, 0.10): 15.95,
    (35, 0.25): 16.96,
    (35, 0.50): 19.63,
    (35, 0.75): 21.55,
    (50, 0.10): 21.41,
    (50, 0.25): 22.84,
    (50, 0.50): 26.71,
    (50, 0.75): 30.19,
}


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            data["coordinates"],
            data["demands"],
            data["service_minutes"],
            data["disclosure_minutes"],
        )


def event_dataset(
    coordinates: np.ndarray,
    demands: np.ndarray,
    service: np.ndarray,
    disclosure: np.ndarray,
    vehicle_count: int,
) -> DCVRP_Dataset:
    batch, customer_count = demands.shape
    nodes = torch.zeros(batch, customer_count + 1, DCVRP_Dataset.CUST_FEAT_SIZE)
    nodes[:, :, :2] = torch.as_tensor(coordinates, dtype=torch.float32)
    nodes[:, 1:, 2] = torch.as_tensor(
        demands / PAPER_VEHICLE_CAPACITY, dtype=torch.float32
    )
    nodes[:, 1:, 3] = torch.as_tensor(
        service / PAPER_HORIZON_MINUTES, dtype=torch.float32
    )
    nodes[:, 1:, 4] = torch.as_tensor(
        disclosure / PAPER_HORIZON_MINUTES, dtype=torch.float32
    )
    return DCVRP_Dataset(
        vehicle_count,
        veh_capa=1.0,
        veh_speed=PAPER_HORIZON_MINUTES,
        nodes=nodes,
        cust_mask=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for size in (20, 35, 50):
        vehicle_count = paper_vehicle_count(size)
        for rate in (0.10, 0.25, 0.50, 0.75):
            path = args.manifest_dir / f"release_n{size}_r{int(100 * rate):02d}.npz"
            coordinates, demands, service, disclosure = load_arrays(path)

            periodic_costs = []
            periodic_qos = []
            planned_suffix_costs = []
            fleet = ExecutedPathFleet(vehicle_count)
            alternative_fleets = {
                "released-time-final-only": ExecutedPathFleet(
                    vehicle_count, travel_minutes_per_normalized_unit=100.0
                ),
                "released-time-return-completed": ExecutedPathFleet(
                    vehicle_count,
                    travel_minutes_per_normalized_unit=100.0,
                    completed_route_return="complete-before-boundary",
                ),
                "released-time-return-started": ExecutedPathFleet(
                    vehicle_count,
                    travel_minutes_per_normalized_unit=100.0,
                    completed_route_return="commit-if-started",
                ),
            }
            alternative_results = {
                name: ([], []) for name in alternative_fleets
            }
            for index in range(coordinates.shape[0]):
                planned_customer_distance = 0.0

                def recording_planner(state):
                    nonlocal planned_customer_distance
                    plan = nearest_idle_routes(state)
                    for vehicle, route in enumerate(plan.routes):
                        previous = state.vehicle_positions[vehicle]
                        for customer in route:
                            destination = state.customer_coordinates[customer]
                            planned_customer_distance += float(
                                np.linalg.norm(previous - destination)
                            )
                            previous = destination
                    return plan

                result = fleet.run(
                    PaperInstance(
                        coordinates[index],
                        demands[index],
                        service[index],
                        disclosure[index],
                    ),
                    recording_planner,
                )
                periodic_costs.append(result.cost)
                periodic_qos.append(result.qos_percent)
                final_closure_distance = sum(
                    event.distance
                    for event in result.events
                    if event.customer is None
                )
                planned_suffix_costs.append(
                    planned_customer_distance + final_closure_distance
                )
                instance = PaperInstance(
                    coordinates[index],
                    demands[index],
                    service[index],
                    disclosure[index],
                )
                for name, alternative_fleet in alternative_fleets.items():
                    alternative = alternative_fleet.run(
                        instance, nearest_idle_routes
                    )
                    alternative_results[name][0].append(alternative.cost)
                    alternative_results[name][1].append(alternative.qos_percent)

            dataset = event_dataset(
                coordinates, demands, service, disclosure, vehicle_count
            )
            event_cost, event_qos, _ = run_paper_greedy(dataset, torch.device("cpu"))
            protocol_results = [
                ("periodic-static", np.asarray(periodic_costs), np.asarray(periodic_qos)),
                (
                    "periodic-planned-customer",
                    np.asarray(planned_suffix_costs),
                    np.asarray(periodic_qos),
                ),
                ("event-driven", event_cost.numpy(), 100.0 * event_qos.numpy()),
            ]
            protocol_results.extend(
                (name, np.asarray(values[0]), np.asarray(values[1]))
                for name, values in alternative_results.items()
            )
            for protocol, costs, qos in protocol_results:
                target = PAPER_GREEDY[(size, rate)]
                mean = float(np.mean(costs))
                rows.append(
                    {
                        "protocol": protocol,
                        "customer_count": size,
                        "vehicle_count": vehicle_count,
                        "dynamic_rate": rate,
                        "instances": int(len(costs)),
                        "cost_mean": mean,
                        "cost_sd": float(np.std(costs, ddof=1)),
                        "qos_mean_percent": float(np.mean(qos)),
                        "paper_greedy": target,
                        "ape_percent": 100.0 * abs(mean - target) / target,
                    }
                )
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = frame.groupby("protocol")["ape_percent"].agg(["mean", "max"])
    print(frame.to_string(index=False))
    print(summary.to_string())


if __name__ == "__main__":
    main()
