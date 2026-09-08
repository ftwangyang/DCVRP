"""Trainable attention learner with explicit vehicle-policy gradients."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from learner import MultiHeadAttention, TransformerEncoder

from .selectors import BaseSelector, RandomSelector, build_selector


class ExperimentalAttentionLearner(nn.Module):
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
        # The manuscript/public evaluator samples customer choices while the
        # vehicle layer follows Eq. (27)'s argmax rule.  Keep the two decoding
        # decisions independently configurable; None preserves the historical
        # behavior in which both were controlled by ``self.greedy``.
        self.vehicle_greedy: bool | None = None
        # The released implementation applies REINFORCE only to the sampled
        # customer decisions: Eq. (27)'s vehicle argmax is not part of the
        # sequence log-probability.  Keep this switch explicit so the public
        # protocol and trainable selector ablations cannot be mixed silently.
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

    def _encode_customers(self, customers, mask=None):
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

    def _represent_vehicle(self, vehicles, vehicle_index, customer_mask=None):
        vehicle_embedding = self.vehicle_embedding(vehicles)
        fleet_representation = self.fleet_attention(vehicle_embedding)
        query = fleet_representation.gather(
            1,
            vehicle_index.unsqueeze(2).expand(-1, -1, self.model_size),
        )
        return self.vehicle_attention(
            query, fleet_representation, fleet_representation
        )

    def _customer_log_probability(self, environment):
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
                    customer_index = customer_log_probability.argmax(dim=1, keepdim=True)
                else:
                    customer_index = customer_log_probability.exp().multinomial(1)
                customer_selected_logp = customer_log_probability.gather(
                    1, customer_index
                )
                joint_logp = customer_selected_logp
                if self.include_vehicle_log_probability:
                    joint_logp = joint_logp + environment.cur_vehicle_logp
                actions.append((environment.cur_veh_idx.clone(), customer_index.clone()))
                action_log_probabilities.append(joint_logp)
                rewards.append(environment.step(customer_index))
            return actions, action_log_probabilities, rewards
        finally:
            if clear_environment is not None:
                clear_environment()


class PaperDescriptionAttentionLearner(ExperimentalAttentionLearner):
    """Literal implementation of manuscript Eqs. (16)--(31).

    The distributed vehicle scorer is inherited unchanged.  The customer
    decoder is replaced by the stated graph-embedding/last-node context:

    * Eq. (20): mean embedding of currently available customers;
    * Eq. (28): concatenate that mean with the selected vehicle's last node;
    * Eq. (29): multi-head attention from the context to node embeddings;
    * Eqs. (30)--(31): projected compatibility, C=10 tanh, and softmax.

    This class is kept separate from ``ExperimentalAttentionLearner`` so older
    enhanced checkpoints remain loadable under their original architecture.
    """

    def __init__(
        self,
        *args,
        model_size: int = 128,
        head_count: int = 8,
        **kwargs,
    ):
        super().__init__(
            *args,
            model_size=model_size,
            head_count=head_count,
            **kwargs,
        )
        # Remove the released-code proxy decoder.  It uses a selected-vehicle
        # representation but not Eq. (28)'s explicit graph/last-node context.
        del self.vehicle_embedding
        del self.fleet_attention
        del self.vehicle_attention
        del self.customer_project
        self.decoder_attention = MultiHeadAttention(
            head_count,
            2 * model_size,
            key_size=model_size,
            value_size=model_size,
        )
        self.compatibility_query = nn.Linear(model_size, model_size, bias=False)
        self.compatibility_key = nn.Linear(model_size, model_size, bias=False)

    def _encode_customers(self, customers, mask=None):
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
        if mask is not None:
            self.encoded_customers = self.encoded_customers.masked_fill(
                mask.unsqueeze(2), 0.0
            )

    def _available_graph_embedding(self, environment):
        # C_available contains revealed, unassigned customers only; the depot
        # is not part of the mean in Eq. (20).
        available = ~(environment.cust_mask | environment.served)
        available = available.clone()
        available[:, 0] = False
        weights = available.to(self.encoded_customers.dtype).unsqueeze(2)
        count = weights.sum(dim=1).clamp_min(1.0)
        return (self.encoded_customers * weights).sum(dim=1) / count

    def _last_node_embedding(self, environment):
        # Vehicle positions in the interval environment always coincide with
        # the depot or the destination of the last dispatched leg.  Matching
        # coordinates therefore recovers h_last,k* without adding an
        # undocumented state feature to s_k.
        selected_position = environment.cur_veh[:, :, :2]
        squared_distance = (
            environment.nodes[:, :, :2] - selected_position
        ).square().sum(dim=2)
        last_index = squared_distance.argmin(dim=1, keepdim=True)
        return self.encoded_customers.gather(
            1,
            last_index.unsqueeze(2).expand(-1, -1, self.model_size),
        ).squeeze(1)

    def _customer_log_probability(self, environment):
        graph_embedding = self._available_graph_embedding(environment)
        last_embedding = self._last_node_embedding(environment)
        context = torch.cat([graph_embedding, last_embedding], dim=1).unsqueeze(1)
        attended = self.decoder_attention(
            context,
            self.encoded_customers,
            self.encoded_customers,
            mask=environment.cur_veh_mask,
        )
        query = self.compatibility_query(attended)
        keys = self.compatibility_key(self.encoded_customers)
        compatibility = query.matmul(keys.transpose(1, 2))
        compatibility *= self.inverse_sqrt_dimension
        compatibility = self.tanh_exploration * compatibility.tanh()
        compatibility = compatibility.masked_fill(
            environment.cur_veh_mask, -torch.inf
        )
        return compatibility.log_softmax(dim=2).squeeze(1)


class MARDAMAttentionLearner(ExperimentalAttentionLearner):
    """MARDAM-style decoder adapted to the unified DCVRP environment.

    Every vehicle queries the encoded customer set before the active vehicle
    attends to the fleet. This is the public MARDAM architecture; it is not the
    vehicle self-attention proxy used by the earlier reviewer-study scripts.
    """

    def __init__(
        self,
        *args,
        vehicle_state_size: int = 4,
        model_size: int = 128,
        head_count: int = 8,
        **kwargs,
    ):
        super().__init__(
            *args,
            vehicle_state_size=vehicle_state_size,
            model_size=model_size,
            head_count=head_count,
            **kwargs,
        )
        # The public architecture uses raw vehicle states as queries, so the
        # proxy-only vehicle embedding must not remain in the parameter count.
        del self.vehicle_embedding
        self.fleet_attention = MultiHeadAttention(
            head_count,
            vehicle_state_size,
            model_size,
        )
        self.vehicle_attention = MultiHeadAttention(head_count, model_size)

    def _represent_vehicle(self, vehicles, vehicle_index, customer_mask=None):
        if customer_mask is None:
            raise ValueError("MARDAM fleet-to-customer attention requires a mask")
        fleet_representation = self.fleet_attention(
            vehicles,
            mask=customer_mask,
        )
        query = fleet_representation.gather(
            1,
            vehicle_index.unsqueeze(2).expand(-1, -1, self.model_size),
        )
        return self.vehicle_attention(
            query,
            fleet_representation,
            fleet_representation,
        )


class OperationalAttentionLearner(ExperimentalAttentionLearner):
    """DVNDA-MO customer decoder with validation-calibrated priorities.

    This class deliberately has no additional trainable tensors, so an
    original independent-DVNDA checkpoint loads exactly. Its operational
    calibration is reported as a separate method and must not be presented as
    the original DVNDA policy.
    """

    def __init__(
        self,
        *args,
        customer_spatial_dispersion_weight: float = 15.0,
        customer_service_weight: float = 150.0,
        customer_age_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.customer_spatial_dispersion_weight = float(
            customer_spatial_dispersion_weight
        )
        self.customer_service_weight = float(customer_service_weight)
        self.customer_age_weight = float(customer_age_weight)

    def _customer_log_probability(self, environment):
        neural_log_probability = super()._customer_log_probability(environment)
        current_time = environment.cur_veh[:, :, 3]
        effective_disclosure = (
            torch.ceil(
                environment.nodes[:, :, 4]
                * float(environment.segment_count)
                - 1.0e-7
            )
            / float(environment.segment_count)
        )
        age = (current_time - effective_disclosure).clamp_min(0.0)
        distance = torch.linalg.vector_norm(
            environment.cur_veh[:, :, :2] - environment.nodes[:, :, :2],
            dim=2,
        )
        service = environment.nodes[:, :, 3]
        calibration = (
            self.customer_age_weight * age
            + self.customer_spatial_dispersion_weight * distance
            - self.customer_service_weight * service
        )
        calibration[:, 0] = 0.0
        logits = neural_log_probability + calibration
        return logits.masked_fill(
            environment.cur_veh_mask.squeeze(1), -torch.inf
        ).log_softmax(dim=1)

    def forward(self, environment):
        setter = getattr(self.selector, "set_operational_environment", None)
        clearer = getattr(self.selector, "clear_operational_environment", None)
        if setter is not None:
            setter(environment)
        try:
            return super().forward(environment)
        finally:
            if clearer is not None:
                clearer()


__all__ = [
    "ExperimentalAttentionLearner",
    "PaperDescriptionAttentionLearner",
    "MARDAMAttentionLearner",
    "OperationalAttentionLearner",
    "build_selector",
]


class JointPairHead(nn.Module):
    def __init__(self, vehicle_state_size: int, model_size: int, head_count: int):
        super().__init__()
        self.vehicle_embedding = nn.Linear(vehicle_state_size, model_size)
        self.fleet_attention = MultiHeadAttention(head_count, model_size)
        self.inverse_sqrt_dimension = model_size ** -0.5

    def forward(self, vehicles, customer_representation):
        vehicle = self.vehicle_embedding(vehicles)
        vehicle = self.fleet_attention(vehicle, vehicle, vehicle)
        return (
            vehicle.matmul(customer_representation.transpose(1, 2))
            * self.inverse_sqrt_dimension
        )


class JointPairLearner(ExperimentalAttentionLearner):
    """Jointly score all feasible vehicle-customer pairs."""

    def __init__(self, vehicle_count: int, *args, **kwargs):
        bootstrap = RandomSelector(vehicle_count)
        super().__init__(bootstrap, *args, **kwargs)
        head_count = kwargs.get("head_count", 8)
        vehicle_state_size = kwargs.get("vehicle_state_size", 4)
        self.bootstrap_selector = bootstrap
        self.selector = JointPairHead(
            vehicle_state_size, self.model_size, head_count
        )

    def forward(self, environment):
        reset_selector = getattr(self.bootstrap_selector, "reset_state", None)
        if reset_selector is not None:
            reset_selector()
        environment.selector = self.bootstrap_selector
        environment.policy_greedy = True
        environment.reset()
        self._encode_customers(environment.nodes, environment.cust_mask)
        actions, log_probabilities, rewards = [], [], []
        while not environment.done:
            if environment.new_customers:
                self._encode_customers(environment.nodes, environment.cust_mask)
                environment.new_customers = False
            pair_scores = self.selector(
                environment.vehicles, self.customer_representation
            )
            pair_mask = environment.mask | environment.veh_done[:, :, None]
            all_done = environment.veh_done.all(dim=1)
            pair_mask[all_done, 0, 0] = False
            pair_scores = pair_scores.masked_fill(pair_mask, -torch.inf)
            flat_scores = pair_scores.flatten(1)
            distribution = Categorical(logits=flat_scores)
            if self.greedy:
                pair_index = flat_scores.argmax(dim=1)
            else:
                pair_index = distribution.sample()
            vehicle_index = (pair_index // environment.nodes_count).unsqueeze(1)
            customer_index = (pair_index % environment.nodes_count).unsqueeze(1)
            environment.cur_veh_idx = vehicle_index
            environment.cur_veh = environment.vehicles.gather(
                1,
                vehicle_index[:, :, None].expand(-1, -1, environment.VEH_STATE_SIZE),
            )
            environment.cur_veh_mask = environment.mask.gather(
                1,
                vehicle_index[:, :, None].expand(-1, -1, environment.nodes_count),
            )
            actions.append((vehicle_index.clone(), customer_index.clone()))
            log_probabilities.append(
                distribution.log_prob(pair_index).unsqueeze(1)
            )
            rewards.append(environment.step(customer_index))
        return actions, log_probabilities, rewards


__all__.append("JointPairLearner")
