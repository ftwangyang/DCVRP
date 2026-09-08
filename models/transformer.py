"""Transformer encoder and attention building blocks for DCVRP."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_prod_attention(queries, keys, values, mask=None):
    """Scaled dot-product attention."""
    weights = queries.matmul(keys.transpose(-1, -2))
    weights *= keys.size(-1) ** -0.5
    if mask is not None:
        if mask.dim() == weights.dim() - 1:
            mask = mask.unsqueeze(-2).expand_as(weights)
        weights = weights.masked_fill(mask, -float("inf"))
    weights = F.softmax(weights, dim=-1)
    return weights.matmul(values)


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with optional precomputation."""

    def __init__(
        self,
        head_count: int,
        query_size: int,
        key_size: int | None = None,
        value_size: int | None = None,
        key_size_per_head: int | None = None,
        value_size_per_head: int | None = None,
    ):
        super().__init__()
        self.head_count = head_count
        self.query_size = query_size
        self.key_size = query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size
        self.key_size_per_head = (
            self.key_size // self.head_count
            if key_size_per_head is None
            else key_size_per_head
        )
        self.value_size_per_head = (
            self.value_size // self.head_count
            if value_size_per_head is None
            else value_size_per_head
        )
        self._inv_sqrt_d = self.key_size_per_head ** -0.5
        self.query_project = nn.Linear(
            self.query_size, self.head_count * self.key_size_per_head, bias=False
        )
        self.key_project = nn.Linear(
            self.key_size, self.head_count * self.key_size_per_head, bias=False
        )
        self.value_project = nn.Linear(
            self.value_size, self.head_count * self.value_size_per_head, bias=False
        )
        self.recombine = nn.Linear(
            self.head_count * self.value_size_per_head, self.value_size, bias=False
        )
        self._k_proj = None
        self._v_proj = None
        self.init_parameters()

    def init_parameters(self):
        nn.init.uniform_(self.query_project.weight, -self._inv_sqrt_d, self._inv_sqrt_d)
        nn.init.uniform_(self.key_project.weight, -self._inv_sqrt_d, self._inv_sqrt_d)
        inv_sq_dv = self.value_size_per_head ** -0.5
        nn.init.uniform_(self.value_project.weight, -inv_sq_dv, inv_sq_dv)

    def precompute(self, keys, values=None):
        values = keys if values is None else values
        l_kv = keys.size(-2)
        self._k_proj = (
            self.key_project(keys)
            .view(-1, l_kv, self.head_count, self.key_size_per_head)
            .permute(0, 2, 3, 1)
        )
        self._v_proj = (
            self.value_project(values)
            .view(-1, l_kv, self.head_count, self.value_size_per_head)
            .permute(0, 2, 1, 3)
        )

    def forward(self, queries, keys=None, values=None, mask=None):
        *size, l_q, _ = queries.size()
        q_proj = (
            self.query_project(queries)
            .view(-1, l_q, self.head_count, self.key_size_per_head)
            .permute(0, 2, 1, 3)
        )

        if keys is None:
            if self._k_proj is None:
                l_kv = l_q
                k_proj = (
                    self.key_project(queries)
                    .view(-1, l_kv, self.head_count, self.key_size_per_head)
                    .permute(0, 2, 3, 1)
                )
            else:
                l_kv = self._k_proj.size(-1)
                k_proj = self._k_proj
        else:
            l_kv = keys.size(-2)
            k_proj = (
                self.key_project(keys)
                .view(-1, l_kv, self.head_count, self.key_size_per_head)
                .permute(0, 2, 3, 1)
            )

        if values is None:
            if self._v_proj is None:
                v_proj = (
                    self.value_project(queries)
                    .view(-1, l_kv, self.head_count, self.value_size_per_head)
                    .permute(0, 2, 1, 3)
                )
            else:
                v_proj = self._v_proj
        else:
            v_proj = (
                self.value_project(values)
                .view(-1, l_kv, self.head_count, self.value_size_per_head)
                .permute(0, 2, 1, 3)
            )

        weights = q_proj.matmul(k_proj) * self._inv_sqrt_d
        if mask is not None:
            if mask.numel() * self.head_count == weights.numel():
                m = mask.view(-1, 1, l_q, l_kv).expand_as(weights)
            else:
                m = mask.view(-1, 1, 1, l_kv).expand_as(weights)
            weights = weights.masked_fill(m, -float("inf"))
        weights = F.softmax(weights, dim=-1)

        att = (
            weights.matmul(v_proj)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(*size, l_q, self.head_count * self.value_size_per_head)
        )
        return self.recombine(att)


class TransformerEncoderLayer(nn.Module):
    """Single layer of Transformer encoder with BatchNorm and feed-forward."""

    def __init__(self, head_count: int, model_size: int, ff_size: int):
        super().__init__()
        self.mha = MultiHeadAttention(head_count, model_size)
        self.bn1 = nn.BatchNorm1d(model_size)
        self.ff1 = nn.Linear(model_size, ff_size)
        self.ff2 = nn.Linear(ff_size, model_size)
        self.bn2 = nn.BatchNorm1d(model_size)

    def forward(self, h_in, mask=None):
        att = self.mha(h_in, mask=mask)
        att = self.bn1((h_in + att).permute(0, 2, 1)).permute(0, 2, 1)
        h_out = F.relu(self.ff1(att))
        h_out = self.ff2(h_out)
        h_out = self.bn2((att + h_out).permute(0, 2, 1)).permute(0, 2, 1)
        if mask is not None:
            h_out[mask] = 0
        return h_out


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers."""

    def __init__(self, layer_count: int, head_count: int, model_size: int, ff_size: int):
        super().__init__()
        for l in range(layer_count):
            self.add_module(
                str(l), TransformerEncoderLayer(head_count, model_size, ff_size)
            )

    def forward(self, inputs, mask=None):
        h = inputs
        for child in self.children():
            h = child(h, mask)
        return h


class VehicleSelectionNetwork(nn.Module):
    """Dual-attention sub-network for vehicle selection (DVNDA).

    Applies fleet attention across vehicle states and customer attention
    across customer features, followed by a 2-layer MLP decision head.
    """

    def __init__(
        self,
        veh_state_size: int = 4,
        cust_feat_size: int = 5,
        model_size: int = 64,
        head_count: int = 4,
    ):
        super().__init__()
        self.model_size = model_size
        self.veh_embedding = nn.Linear(veh_state_size, model_size)
        self.cust_embedding = nn.Linear(cust_feat_size, model_size)
        self.fleet_attention = MultiHeadAttention(head_count, model_size)
        self.cust_attention = MultiHeadAttention(head_count, model_size)
        self.decision_layer1 = nn.Linear(model_size * 3, model_size)
        self.decision_layer2 = nn.Linear(model_size, 1)

    def forward(self, my_state, all_vehicles, customers, veh_mask, cust_mask):
        all_veh_emb = self.veh_embedding(all_vehicles)
        cust_emb = self.cust_embedding(customers)
        my_emb = self.veh_embedding(my_state)
        fleet_ctx = self.fleet_attention(
            my_emb, all_veh_emb, all_veh_emb, mask=veh_mask.unsqueeze(1)
        )
        cust_ctx = self.cust_attention(
            my_emb, cust_emb, cust_emb, mask=cust_mask.unsqueeze(1)
        )
        combined = torch.cat([my_emb, fleet_ctx, cust_ctx], dim=2)
        h = F.relu(self.decision_layer1(combined))
        return self.decision_layer2(h).squeeze(2)
