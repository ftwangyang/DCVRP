"""Regression tests for continuous interval-boundary state transitions."""

from __future__ import annotations

import unittest

import torch

from data import DCVRP_Dataset
from experiments.reviewer_study.environment import ExperimentalEnvironment


class _FirstVehicleSelector:
    def select(self, vehicles, customers, vehicle_done, customer_mask, greedy):
        batch = vehicles.size(0)
        index = torch.zeros((batch, 1), dtype=torch.long, device=vehicles.device)
        scores = torch.zeros((batch, vehicles.size(1)), device=vehicles.device)
        logp = torch.zeros((batch, 1), device=vehicles.device)
        return index, scores, logp


def _environment() -> ExperimentalEnvironment:
    nodes = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.2, 0.1, 0.0]]])
    data = DCVRP_Dataset(1, 1.0, 2.0, nodes)
    environment = ExperimentalEnvironment(data, segment_count=4, horizon=1.0)
    environment.selector = _FirstVehicleSelector()
    environment.reset()
    return environment


class BoundaryTransitionTest(unittest.TestCase):
    def test_enroute_position_and_distance_are_interpolated(self):
        environment = _environment()
        environment.route_logs[0][0] = [
            {
                "customer": 1,
                "arrival": 0.5,
                "departure": 0.6,
                "distance": 1.0,
                "origin": [0.0, 0.0],
                "start_time": 0.0,
            },
            {
                "customer": 0,
                "arrival": 1.1,
                "departure": 1.1,
                "distance": 1.0,
                "origin": [1.0, 0.0],
                "start_time": 0.6,
            },
        ]
        environment.served[0, 1] = True
        environment.committed[0, 1] = True
        environment.total_distance[0] = 2.0
        environment.vehicle_distance[0, 0] = 2.0

        environment._segment_transition(0.25)

        self.assertAlmostEqual(float(environment.vehicles[0, 0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(environment.total_distance[0]), 0.5, places=6)
        self.assertAlmostEqual(float(environment.executed_partial_distance[0, 0]), 0.5, places=6)
        self.assertFalse(bool(environment.served[0, 1]))
        self.assertEqual(environment.route_logs[0][0], [])

    def test_in_service_customer_remains_committed_then_completes(self):
        environment = _environment()
        environment.route_logs[0][0] = [
            {
                "customer": 1,
                "arrival": 0.2,
                "departure": 0.4,
                "distance": 0.4,
                "origin": [0.0, 0.0],
                "start_time": 0.0,
            },
            {
                "customer": 0,
                "arrival": 0.9,
                "departure": 0.9,
                "distance": 1.0,
                "origin": [1.0, 0.0],
                "start_time": 0.4,
            },
        ]
        environment.served[0, 1] = True
        environment.committed[0, 1] = True
        environment.total_distance[0] = 1.4
        environment.vehicle_distance[0, 0] = 1.4

        environment._segment_transition(0.3)
        self.assertAlmostEqual(float(environment.vehicles[0, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(environment.vehicles[0, 0, 3]), 0.4, places=6)
        self.assertTrue(bool(environment.committed[0, 1]))
        self.assertFalse(bool(environment.completed[0, 1]))
        self.assertEqual(len(environment.route_logs[0][0]), 1)

        environment._segment_transition(0.5)
        self.assertTrue(bool(environment.completed[0, 1]))
        self.assertFalse(bool(environment.committed[0, 1]))


if __name__ == "__main__":
    unittest.main()
