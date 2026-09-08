"""Reproducibility checks for generated instances and selector learning."""

from __future__ import annotations

import unittest

import torch

from experiments.reviewer_study.data_generation import generate_dataset
from experiments.reviewer_study.protocol import StudyConfig, _episode, build_model, set_seed
from experiments.reviewer_study.selectors import IndependentSelector, SharedSelector
from experiments.reviewer_study.run_original_uncertainty_n20 import (
    build_model as build_n20_model,
)


class GenerationTest(unittest.TestCase):
    def test_dynamic_ratio_is_exact_per_instance(self):
        dataset = generate_dataset(
            12, customer_count=20, vehicle_count=4,
            dynamic_ratio=0.75, seed=17,
        )
        counts = (dataset.nodes[:, 1:, 4] > 0).sum(1)
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 15)))

    def test_early_and_late_profiles_have_expected_order(self):
        early = generate_dataset(
            64, customer_count=20, dynamic_ratio=1.0,
            arrival_profile="early", seed=18,
        ).nodes[:, 1:, 4].mean()
        late = generate_dataset(
            64, customer_count=20, dynamic_ratio=1.0,
            arrival_profile="late", seed=18,
        ).nodes[:, 1:, 4].mean()
        self.assertLess(float(early), float(late))


class GradientTest(unittest.TestCase):
    def test_mardam_uses_vehicle_to_customer_cross_attention(self):
        torch.manual_seed(18)
        model = build_n20_model(
            torch.device("cpu"),
            "earliest_available",
            "mardam",
        )
        customers = torch.rand(2, 9, 5)
        customer_mask = torch.zeros(2, 9, dtype=torch.bool)
        vehicles = torch.rand(2, 3, 4)
        fleet_customer_mask = torch.zeros(2, 3, 9, dtype=torch.bool)
        vehicle_index = torch.tensor([[0], [2]])
        model._encode_customers(customers, customer_mask)
        representation = model._represent_vehicle(
            vehicles,
            vehicle_index,
            fleet_customer_mask,
        )
        self.assertEqual(representation.shape, (2, 1, 128))
        self.assertFalse(hasattr(model, "vehicle_embedding"))
        self.assertEqual(model.fleet_attention.query_size, 4)
        self.assertEqual(model.fleet_attention.key_size, 128)

    def test_lidrl_selector_uses_tour_history_and_receives_gradient(self):
        device = torch.device("cpu")
        model = build_n20_model(device, "lidrl_tour_history", "dvnda")
        selector = model.selector
        vehicles = torch.rand(3, 4, 4)
        customers = torch.rand(3, 21, 5)
        vehicle_done = torch.zeros(3, 4, dtype=torch.bool)
        customer_mask = torch.zeros(3, 4, 21, dtype=torch.bool)
        scores = selector.scores(
            vehicles, customers, vehicle_done, customer_mask
        )
        self.assertEqual(scores.shape, (3, 4))
        scores.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in selector.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertGreater(
            float(torch.sqrt(sum(gradient.square().sum() for gradient in gradients))),
            0.0,
        )

    def test_independent_selector_receives_policy_gradient(self):
        device = torch.device("cpu")
        config = StudyConfig(
            customer_count=8,
            vehicle_count=2,
            model_size=32,
            layer_count=1,
            head_count=4,
            ff_size=64,
            selector_size=16,
            batch_size=4,
            test_size=4,
        )
        set_seed(19)
        model = build_model("independent", config, device)
        _, _, cost, log_probability = _episode(
            model, config, device, data_seed=20, greedy=False
        )
        standardized = (cost - cost.mean()) / cost.std().clamp_min(1.0e-6)
        (standardized.detach() * log_probability).mean().backward()
        gradients = [
            parameter.grad for parameter in model.selector.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        norm = torch.sqrt(sum((gradient ** 2).sum() for gradient in gradients))
        self.assertGreater(float(norm), 0.0)

    def test_vectorized_shared_selector_matches_vehicle_loop(self):
        torch.manual_seed(23)
        selector = SharedSelector(6).eval()
        vehicles = torch.rand(3, 6, 4)
        customers = torch.rand(3, 21, 5)
        vehicle_done = torch.zeros(3, 6, dtype=torch.bool)
        customer_mask = torch.zeros(3, 21, dtype=torch.bool)
        vectorized = selector.scores(
            vehicles, customers, vehicle_done, customer_mask
        )
        loop = torch.stack([
            selector.network(
                vehicles[:, index:index + 1],
                vehicles,
                customers,
                vehicle_done,
                customer_mask,
            ).squeeze(1)
            for index in range(6)
        ], dim=1)
        self.assertTrue(torch.allclose(vectorized, loop, atol=1.0e-6))

    def test_vectorized_independent_selector_matches_vehicle_loop(self):
        torch.manual_seed(24)
        selector = IndependentSelector(6).eval()
        vehicles = torch.rand(3, 6, 4)
        customers = torch.rand(3, 21, 5)
        vehicle_done = torch.zeros(3, 6, dtype=torch.bool)
        customer_mask = torch.zeros(3, 21, dtype=torch.bool)
        loop = torch.stack([
            network(
                vehicles[:, index:index + 1],
                vehicles,
                customers,
                vehicle_done,
                customer_mask,
            ).squeeze(1)
            for index, network in enumerate(selector.vehicle_networks)
        ], dim=1)
        vectorized = selector.scores(
            vehicles, customers, vehicle_done, customer_mask
        )
        self.assertTrue(torch.allclose(vectorized, loop, atol=1.0e-6))

    def test_episode_cost_matches_reported_executed_cost(self):
        device = torch.device("cpu")
        config = StudyConfig(
            customer_count=8, vehicle_count=2, model_size=32,
            layer_count=1, head_count=4, ff_size=64,
            selector_size=16, batch_size=4, test_size=4,
        )
        set_seed(21)
        model = build_model("independent", config, device)
        _, environment, cost, _ = _episode(
            model, config, device, data_seed=22, greedy=False
        )
        metrics = environment.metrics()
        expected = torch.tensor(
            metrics["distance"] + config.pending_cost * metrics["unserved"],
            dtype=cost.dtype,
        )
        self.assertTrue(torch.allclose(cost.detach().cpu(), expected, atol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
