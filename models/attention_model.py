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
        # Eq. 27 is argmax, so vehicle selection is always greedy. The flag is
        # retained so checkpoints / trainers can set it without AttributeError.
        self.vehicle_greedy: bool | None = True
        # Eq. 33 uses log p_theta(pi|S). Vehicle scores still contribute the
        # log-probability of the argmax vehicle so the selector receives gradient.
        self.include_vehicle_log_probability = True

        # 1. Feature Embeddings
        self.depot_embedding = nn.Linear(customer_feature_size, model_size)
        self.customer_embedding = nn.Linear(customer_feature_size, model_size)

        # 2. 3-Layer Transformer Customer Encoder (Eqs. 16-20)
        self.customer_encoder = TransformerEncoder(
            layer_count, head_count, model_size, ff_size
        )

        # 3. Node Selection Decoder (Eqs. 28-31)
        # Context vector H_c^T combines graph embedding \bar{h}_G and last node embedding h_{last} (Eq. 28) -> query_size = 2 * model_size
        self.decoder_attention = MultiHeadAttention(
            head_count=head_count,
            query_size=2 * model_size,
            key_size=model_size,
            value_size=model_size,
        )
        # Compatibility pointer projections (Eq. 30)
        self.project_node_queries = nn.Linear(model_size, model_size, bias=False)
        self.project_node_keys = nn.Linear(model_size, model_size, bias=False)

    def _encode_customers(self, customers: torch.Tensor, mask: torch.Tensor | None = None):
        """Compute customer node embeddings via Transformer encoder."""
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
        self.projected_keys = self.project_node_keys(self.encoded_customers)

    def _customer_log_probability(
        self, environment, last_visited_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Compute customer action probabilities according to Eqs. 28-31."""
        B = environment.minibatch_size
        k_star = environment.cur_veh_idx  # (B, 1)

        # 1. Last visited node embedding h_{last, k*}^{T-1} (Eq. 28)
        last_node_idx = last_visited_nodes.gather(1, k_star)  # (B, 1)
        h_last = self.encoded_customers.gather(
            1, last_node_idx.unsqueeze(2).expand(-1, -1, self.model_size)
        )  # (B, 1, d)

        # 2. Graph embedding \bar{h}_{G, k*} from vehicle k*'s perspective (Eq. 28)
        visible = (~environment.cur_veh_mask).float()  # (B, 1, N)
        denom = visible.sum(dim=2, keepdim=True).clamp_min(1.0)
        h_G = (self.encoded_customers * visible.transpose(1, 2)).sum(
            dim=1, keepdim=True
        ) / denom  # (B, 1, d)

        # 3. Form combined context vector H_c^T = [\bar{h}_{G, k*}, h_{last, k*}^{T-1}] (Eq. 28)
        H_c = torch.cat([h_G, h_last], dim=-1)  # (B, 1, 2d)

        # 4. Decoder Multi-Head Attention \hat{H}_c^T (Eq. 29)
        hat_H_c = self.decoder_attention(
            H_c,
            self.encoded_customers,
            self.encoded_customers,
            mask=environment.cur_veh_mask,
        )  # (B, 1, d)

        # 5. Compatibility scores with tanh exploration C=10 (Eq. 30)
        q = self.project_node_queries(hat_H_c)  # (B, 1, d)
        compatibility = (
            q @ self.projected_keys.transpose(1, 2)
        ) * self.inverse_sqrt_dimension  # (B, 1, N)
        compatibility = self.tanh_exploration * compatibility.tanh()
        compatibility = compatibility.masked_fill(
            environment.cur_veh_mask, -torch.inf
        )

        # 6. Probability distribution via softmax over candidate nodes (Eq. 31)
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
            environment.policy_greedy = True  # Eq. 27 argmax, independent of node sampling
            environment.reset()
            self._encode_customers(environment.nodes, environment.cust_mask)

            # Track last visited node for each vehicle (initialized to depot node 0)
            B, V = environment.minibatch_size, environment.veh_count
            last_visited_nodes = torch.zeros(
                B, V, dtype=torch.long, device=environment.device
            )

            actions, action_log_probabilities, rewards = [], [], []
            while not environment.done:
                if environment.interval_advanced:
                    last_visited_nodes = environment.last_node.clone()
                    environment.interval_advanced = False
                if environment.new_customers:
                    self._encode_customers(environment.nodes, environment.cust_mask)
                    environment.new_customers = False

                customer_log_probability = self._customer_log_probability(
                    environment, last_visited_nodes
                )
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

                # Update last visited node for the acting vehicle
                last_visited_nodes.scatter_(
                    1, environment.cur_veh_idx, customer_index
                )

                actions.append(
                    (environment.cur_veh_idx.clone(), customer_index.clone())
                )
                action_log_probabilities.append(joint_logp)
                rewards.append(environment.step(customer_index))

            return actions, action_log_probabilities, rewards
        finally:
            if clear_environment is not None:
                clear_environment()
