"""AttentionLearner: Deep Reinforcement Learning Actor for DCVRP.

Combines a 3-layer Transformer encoder for dynamic customer embeddings,
pointer attention decoder for customer selection with tanh clipping (C=10),
and a modular vehicle selector (DVNDA Distributed Vehicle Networks with Decision Aggregation by default).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .selectors import BaseSelector
from .transformer import MultiHeadAttention, TransformerEncoder


class AttentionLearner(nn.Module):
    """Full neural actor network for DCVRP."""

    def __init__(
        self,
        selector: BaseSelector,
        customer_feature_size: int = 5,
        vehicle_state_size: int = 4,
        model_size: int = 128,
        layer_count: int = 3,
        head_count: int = 8,
        ff_size: int = 512,
        tanh_exploration: float = 10.0,
    ):
        super().__init__()
        self.selector = selector
        self.model_size = model_size
        self.inverse_sqrt_dimension = model_size ** -0.5
        self.tanh_exploration = tanh_exploration
        self.greedy = False
        self.vehicle_greedy: bool | None = None
        self.include_vehicle_log_probability = True

        self.depot_embedding = nn.Linear(customer_feature_size, model_size)
        self.customer_embedding = nn.Linear(customer_feature_size, model_size)
        self.customer_encoder = TransformerEncoder(
            layer_count, head_count, model_size, ff_size
        )
        self.vehicle_embedding = nn.Linear(vehicle_state_size, model_size)
        self.fleet_attention = MultiHeadAttention(head_count, model_size)
        self.vehicle_attention = MultiHeadAttention(head_count, model_size)
        self.customer_project = nn.Linear(model_size, model_size)

    def _encode_customers(self, customers: torch.Tensor, mask: torch.Tensor | None = None):
        embedding = torch.cat(
            [
                self.depot_embedding(customers[:, 0:1, :]),
                self.customer_embedding(customers[:, 1:, :]),
            ],
            dim=1,
        )
        if mask is not None:
            embedding = embedding.masked_fill(mask.unsqueeze(2), 0.0)
        self.encoded_customers = self.customer_encoder(embedding, mask)
        self.fleet_attention.precompute(self.encoded_customers)
        self.customer_representation = self.customer_project(self.encoded_customers)
        if mask is not None:
            self.customer_representation = self.customer_representation.masked_fill(
                mask.unsqueeze(2), 0.0
            )

    def _represent_vehicle(
        self,
        vehicles: torch.Tensor,
        vehicle_index: torch.Tensor,
        customer_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vehicle_embedding = self.vehicle_embedding(vehicles)
        fleet_representation = self.fleet_attention(vehicle_embedding)
        query = fleet_representation.gather(
            1,
            vehicle_index.unsqueeze(2).expand(-1, -1, self.model_size),
        )
        return self.vehicle_attention(
            query, fleet_representation, fleet_representation
        )

    def _customer_log_probability(self, environment) -> torch.Tensor:
        vehicle = self._represent_vehicle(
            environment.vehicles,
            environment.cur_veh_idx,
            environment.mask,
        )
        compatibility = vehicle.matmul(
            self.customer_representation.transpose(1, 2)
        )
        compatibility *= self.inverse_sqrt_dimension
        compatibility = self.tanh_exploration * compatibility.tanh()
        compatibility = compatibility.masked_fill(
            environment.cur_veh_mask, -torch.inf
        )
        return compatibility.log_softmax(dim=2).squeeze(1)

    def forward(self, environment):
        reset_selector = getattr(self.selector, "reset_state", None)
        set_environment = getattr(
            self.selector, "set_operational_environment", None
        )
        clear_environment = getattr(
            self.selector, "clear_operational_environment", None
        )
        if reset_selector is not None:
            reset_selector()
        if set_environment is not None:
            set_environment(environment)
        try:
            environment.selector = self.selector
            environment.policy_greedy = (
                self.greedy if self.vehicle_greedy is None else self.vehicle_greedy
            )
            environment.reset()
            self._encode_customers(environment.nodes, environment.cust_mask)
            actions, action_log_probabilities, rewards = [], [], []
            while not environment.done:
                if environment.new_customers:
                    self._encode_customers(environment.nodes, environment.cust_mask)
                    environment.new_customers = False
                customer_log_probability = self._customer_log_probability(environment)
                if self.greedy:
                    customer_index = customer_log_probability.argmax(
                        dim=1, keepdim=True
                    )
                else:
                    customer_index = customer_log_probability.exp().multinomial(1)
                customer_selected_logp = customer_log_probability.gather(
                    1, customer_index
                )
                joint_logp = customer_selected_logp
                if self.include_vehicle_log_probability:
                    joint_logp = joint_logp + environment.cur_vehicle_logp
                actions.append(
                    (environment.cur_veh_idx.clone(), customer_index.clone())
                )
                action_log_probabilities.append(joint_logp)
                rewards.append(environment.step(customer_index))
            return actions, action_log_probabilities, rewards
        finally:
            if clear_environment is not None:
                clear_environment()
