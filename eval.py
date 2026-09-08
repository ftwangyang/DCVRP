import torch
import random
import numpy as np
import multiprocessing as mp
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class DCVRP_Environment:

    VEH_STATE_SIZE = 4  # x, y, remaining_capacity, current_time
    CUST_FEAT_SIZE = 5  # x, y, demand, service_duration, appearance_time

    def __init__(self, data, nodes=None, pending_cost=5, segment_count=10, horizon=480):
        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()
        self.pending_cost = pending_cost if pending_cost is not None else 5
        self.device = self.nodes.device
        self.batch_indices = torch.arange(self.minibatch_size, device=self.device)
        self.horizon = horizon
        self.segment_count = segment_count
        self.segment_duration = horizon / segment_count
        self.current_segment = 0
        self.route_logs = None
        self.returned_to_depot = None
        self.dvnda = None
        self.new_customers = False

    def reset(self):
        B, V = self.minibatch_size, self.veh_count
        self.vehicles = self.nodes.new_zeros((B, V, self.VEH_STATE_SIZE))
        self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2]
        self.vehicles[:, :, 2] = self.veh_capa
        self.vehicles[:, :, 3] = 0.0
        self.veh_done = torch.zeros((B, V), dtype=torch.bool, device=self.device)
        self.returned_to_depot = torch.zeros((B, V), dtype=torch.bool, device=self.device)
        self.done = False
        self.cust_mask = (self.nodes[:, :, 4] > 0)
        self.total_cust_mask = self.cust_mask[:, None, :].expand(-1, V, -1).clone()
        self.served = torch.zeros((B, self.nodes_count), dtype=torch.bool, device=self.device)
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self.route_logs = [[[] for _ in range(V)] for _ in range(B)]
        self.current_segment = 0
        self.new_customers = False
        self._rebuild_mask()
        self._update_cur_veh()

    def _rebuild_mask(self):
        self.mask = self.total_cust_mask | self.served[:, None, :]
        cap_mask = (self.vehicles[:, :, 2].unsqueeze(-1) < self.nodes[:, None, :, 2])
        self.mask = self.mask | cap_mask
        self.mask = self.mask | self.veh_done[:, :, None]
        self.mask[:, :, 0] = False

    def _update_mask_after_action(self, cust_idx):
        new_served = self.served.clone()
        new_served.scatter_(1, cust_idx, cust_idx > 0)
        self.served = new_served
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1
        self._rebuild_mask()

    def _update_vehicles(self, dest):
        dist = torch.norm(self.cur_veh[:, 0, :2] - dest[:, 0, :2], dim=1, keepdim=True)
        travel_time = dist / self.veh_speed
        arrival_time = self.cur_veh[:, :, 3] + travel_time
        new_cur_veh = self.cur_veh.clone()
        new_cur_veh[:, :, :2] = dest[:, :, :2]
        new_cur_veh[:, :, 2] = (new_cur_veh[:, :, 2] - dest[:, :, 2]).clamp(min=0.0)
        new_cur_veh[:, :, 3] = arrival_time + dest[:, :, 3]
        updated = self.vehicles.clone()
        updated.scatter_(
            1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
            new_cur_veh
        )
        self.vehicles = updated
        self.cur_veh = new_cur_veh
        return dist, arrival_time

    def _log_route_step(self, batch_idx, veh_idx, cust_idx, arrival_time, departure_time):
        self.route_logs[batch_idx][veh_idx].append({
            "customer": int(cust_idx),
            "arrival": float(arrival_time),
            "departure": float(departure_time)
        })

    def _update_done(self, cust_idx):
        is_depot = (cust_idx == 0)
        new_ret = self.returned_to_depot.clone()
        new_ret.scatter_(1, self.cur_veh_idx, is_depot)
        self.returned_to_depot = new_ret
        new_done = self.veh_done.clone()
        new_done.scatter_(1, self.cur_veh_idx, is_depot)
        self.veh_done = new_done
        if self.current_segment >= self.segment_count - 1:
            self.done = self.veh_done.all(dim=1).all().item()

    def _update_cur_veh(self):
        if self.dvnda is None:
            raise RuntimeError("dvnda not set — call AttentionLearner.forward first")
        if next(self.dvnda.parameters()).device != self.device:
            self.dvnda = self.dvnda.to(self.device)
        self.cur_veh_idx, _ = self.dvnda(
            self.vehicles, self.nodes, self.veh_done, self.cust_mask
        )
        idx_exp = self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE)
        self.cur_veh = self.vehicles.gather(1, idx_exp)
        idx_exp_n = self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count)
        self.cur_veh_mask = self.mask.gather(1, idx_exp_n)

    def _check_segment_transition(self):
        if self.current_segment >= self.segment_count - 1:
            return
        if not self.returned_to_depot.all().item():
            return
        next_seg_time = (self.current_segment + 1) * self.segment_duration
        self._segment_transition(next_seg_time)

    def _segment_transition(self, seg_time):
        B, V = self.minibatch_size, self.veh_count
        for b in range(B):
            for v in range(V):
                route = self.route_logs[b][v]

                if len(route) == 0:
                    self.vehicles[b, v, :2] = self.nodes[b, 0, :2]
                    self.vehicles[b, v, 2] = float(self.veh_capa)
                    self.vehicles[b, v, 3] = float(seg_time)
                    continue
                cut_idx = None
                for idx, evt in enumerate(route):
                    if evt["arrival"] >= seg_time:
                        cut_idx = idx
                        break
                if cut_idx is not None:
                    removed = route[cut_idx:]
                    route[:] = route[:cut_idx]
                    for evt in removed:
                        cust = evt["customer"]
                        if cust != 0:
                            self.served[b, cust] = False
                if len(route) == 0:
                    self.vehicles[b, v, :2] = self.nodes[b, 0, :2]
                    self.vehicles[b, v, 2] = float(self.veh_capa)
                    self.vehicles[b, v, 3] = float(seg_time)
                    continue
                last = route[-1]
                last_cust = last["customer"]
                delivered = 0.0
                for evt in route:
                    if evt["customer"] != 0:
                        delivered += float(self.nodes[b, evt["customer"], 2])
                if last_cust == 0:
                    self.vehicles[b, v, :2] = self.nodes[b, 0, :2]
                    self.vehicles[b, v, 2] = max(float(self.veh_capa) - delivered, 0.0)
                    self.vehicles[b, v, 3] = float(max(last["departure"], seg_time))
                else:
                    cust_pos = self.nodes[b, last_cust, :2]
                    self.vehicles[b, v, :2] = cust_pos
                    self.vehicles[b, v, 2] = max(float(self.veh_capa) - delivered, 0.0)
                    self.vehicles[b, v, 3] = float(seg_time)

        reveal = (self.nodes[:, :, 4] <= seg_time) & (~self.served)
        old_cust_mask = self.cust_mask.clone()
        self.cust_mask = self.cust_mask & ~reveal
        reveal_exp = reveal[:, None, :].expand(-1, V, -1)
        self.total_cust_mask = self.total_cust_mask & ~reveal_exp
        self.new_customers = (old_cust_mask != self.cust_mask).any().item()
        self.current_segment += 1
        self.pending_customers = (~self.served).float().sum(-1, keepdim=True) - 1

        if self.current_segment >= self.segment_count:
            self.done = True
        else:
            self.done = False
            self.veh_done[:] = False
            self.returned_to_depot[:] = False
            self._rebuild_mask()

    def step(self, cust_idx):
        cust_idx = cust_idx.to(self.device)
        dest = self.nodes.gather(
            1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE)
        )
        prev_time = self.cur_veh[:, :, 3].clone()
        prev_pos = self.cur_veh[:, :, :2].clone()
        dist, arrival_time = self._update_vehicles(dest)
        for b in range(self.minibatch_size):
            veh = int(self.cur_veh_idx[b, 0].item())
            cust = int(cust_idx[b, 0].item())
            if cust != 0:
                arr_t = float(arrival_time[b, 0].item())
                dep_t = float(self.vehicles[b, veh, 3].item())
            else:
                arr_t = float(
                    prev_time[b, 0].item()
                    + torch.norm(prev_pos[b, 0] - self.nodes[b, 0, :2]).item() / self.veh_speed
                )
                dep_t = float(self.vehicles[b, veh, 3].item())
            self._log_route_step(b, veh, cust, arr_t, dep_t)

        self._update_done(cust_idx)
        self._update_mask_after_action(cust_idx)
        reward = -dist

        if self.done and self.current_segment >= self.segment_count - 1:
            pending = (~self.served).float().sum(-1, keepdim=True) - 1
            reward = reward - self.pending_cost * pending

        self._check_segment_transition()

        if not self.done:
            self._update_cur_veh()

        return reward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, head_count, model_size, ff_size):
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
    def __init__(self, layer_count, head_count, model_size, ff_size):
        super().__init__()
        for l in range(layer_count):
            self.add_module(str(l), TransformerEncoderLayer(head_count, model_size, ff_size))

    def forward(self, inputs, mask=None):
        h = inputs
        for child in self.children():
            h = child(h, mask)
        return h

def scaled_dot_prod_attention(queries, keys, values, mask=None):
    weights = queries.matmul(keys.transpose(-1, -2))
    weights *= keys.size(-1) ** -0.5
    if mask is not None:
        if mask.dim() == weights.dim() - 1:
            mask = mask.unsqueeze(-2).expand_as(weights)
        weights[mask] = -float('inf')
    weights = F.softmax(weights, dim=-1)
    return weights.matmul(values)

class MultiHeadAttention(nn.Module):
    def __init__(self, head_count, query_size, key_size=None, value_size=None,
                 key_size_per_head=None, value_size_per_head=None):
        super().__init__()
        self.head_count = head_count
        self.query_size = query_size
        self.key_size = query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size
        self.key_size_per_head = (self.key_size // self.head_count
                                  if key_size_per_head is None else key_size_per_head)
        self.value_size_per_head = (self.value_size // self.head_count
                                    if value_size_per_head is None else value_size_per_head)
        self._inv_sqrt_d = self.key_size_per_head ** -0.5
        self.query_project = nn.Linear(self.query_size,
                                       self.head_count * self.key_size_per_head, bias=False)
        self.key_project = nn.Linear(self.key_size,
                                     self.head_count * self.key_size_per_head, bias=False)
        self.value_project = nn.Linear(self.value_size,
                                       self.head_count * self.value_size_per_head, bias=False)
        self.recombine = nn.Linear(self.head_count * self.value_size_per_head,
                                   self.value_size, bias=False)
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
        self._k_proj = self.key_project(keys).view(
            -1, l_kv, self.head_count, self.key_size_per_head).permute(0, 2, 3, 1)
        self._v_proj = self.value_project(values).view(
            -1, l_kv, self.head_count, self.value_size_per_head).permute(0, 2, 1, 3)

    def forward(self, queries, keys=None, values=None, mask=None):
        *size, l_q, _ = queries.size()
        q_proj = self.query_project(queries).view(
            -1, l_q, self.head_count, self.key_size_per_head).permute(0, 2, 1, 3)

        if keys is None:
            if self._k_proj is None:
                l_kv = l_q
                k_proj = self.key_project(queries).view(
                    -1, l_kv, self.head_count, self.key_size_per_head).permute(0, 2, 3, 1)
            else:
                l_kv = self._k_proj.size(-1)
                k_proj = self._k_proj
        else:
            l_kv = keys.size(-2)
            k_proj = self.key_project(keys).view(
                -1, l_kv, self.head_count, self.key_size_per_head).permute(0, 2, 3, 1)

        if values is None:
            if self._v_proj is None:
                v_proj = self.value_project(queries).view(
                    -1, l_kv, self.head_count, self.value_size_per_head).permute(0, 2, 1, 3)
            else:
                v_proj = self._v_proj
        else:
            v_proj = self.value_project(values).view(
                -1, l_kv, self.head_count, self.value_size_per_head).permute(0, 2, 1, 3)

        weights = q_proj.matmul(k_proj) * self._inv_sqrt_d
        if mask is not None:
            if mask.numel() * self.head_count == weights.numel():
                m = mask.view(-1, 1, l_q, l_kv).expand_as(weights)
            else:
                m = mask.view(-1, 1, 1, l_kv).expand_as(weights)
            weights[m] = -float('inf')
        weights = F.softmax(weights, dim=-1)

        att = weights.matmul(v_proj).permute(0, 2, 1, 3).contiguous().view(
            *size, l_q, self.head_count * self.value_size_per_head)
        return self.recombine(att)

class VehicleSelectionNetwork(nn.Module):
    def __init__(self, veh_state_size, cust_feat_size, model_size=64, head_count=4):
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
            my_emb, all_veh_emb, all_veh_emb, mask=veh_mask.unsqueeze(1))
        cust_ctx = self.cust_attention(
            my_emb, cust_emb, cust_emb, mask=cust_mask.unsqueeze(1))
        combined = torch.cat([my_emb, fleet_ctx, cust_ctx], dim=2)
        h = F.relu(self.decision_layer1(combined))
        return self.decision_layer2(h).squeeze(2)

class MLPAggregator(nn.Module):
    """Lightweight MLP that re-scores vehicles using score + embedding + global context."""
    def __init__(self, model_size=64):
        super().__init__()
        input_size = 1 + model_size + model_size
        self.fc1 = nn.Linear(input_size, model_size)
        self.fc2 = nn.Linear(model_size, 1)

    def forward(self, scores, veh_emb, veh_done):
        valid_mask = (~veh_done).float().unsqueeze(-1)
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_ctx = (veh_emb * valid_mask).sum(dim=1, keepdim=True) / valid_count
        global_ctx = global_ctx.expand_as(veh_emb)
        score_feat = scores.unsqueeze(-1)
        combined = torch.cat([score_feat, veh_emb, global_ctx], dim=-1)
        h = F.relu(self.fc1(combined))
        new_scores = self.fc2(h).squeeze(-1)
        return new_scores


class AttentionAggregator(nn.Module):
    """Attention-based aggregation: learnable query attends to vehicle representations."""
    def __init__(self, model_size=64, head_count=2):
        super().__init__()
        self.model_size = model_size
        self.kv_project = nn.Linear(1 + model_size, model_size)
        self.query = nn.Parameter(torch.randn(1, 1, model_size) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_size, num_heads=head_count, batch_first=True,
        )
        self.output_project = nn.Linear(model_size, 1)
        self.vehicle_project = nn.Linear(model_size, model_size)

    def forward(self, scores, veh_emb, veh_done):
        B, V, D = veh_emb.shape
        score_feat = scores.unsqueeze(-1)
        kv_input = torch.cat([score_feat, veh_emb], dim=-1)
        kv = self.kv_project(kv_input)
        vehicle_repr = self.vehicle_project(kv)
        query = self.query.expand(B, -1, -1)
        attn_out, _ = self.attention(query, kv, kv, key_padding_mask=veh_done)
        logits = (vehicle_repr * attn_out.expand_as(vehicle_repr)).sum(dim=-1)
        return logits


class DVNDA(nn.Module):
    def __init__(self, veh_count, veh_state_size, cust_feat_size,
                 model_size=64, head_count=4, aggregator_type="argmax"):
        super().__init__()
        self.veh_count = veh_count
        self.model_size = model_size
        self.aggregator_type = aggregator_type
        self.veh_embedding_agg = nn.Linear(veh_state_size, model_size)
        self.vehicle_networks = nn.ModuleList([
            VehicleSelectionNetwork(veh_state_size, cust_feat_size, model_size, head_count)
            for _ in range(veh_count)
        ])
        if aggregator_type == "mlp":
            self.aggregator = MLPAggregator(model_size)
        elif aggregator_type == "attention":
            self.aggregator = AttentionAggregator(model_size, head_count=2)
        else:
            self.aggregator = None

    def forward(self, vehicles, customers, veh_done, cust_mask):
        B = vehicles.size(0)
        device = vehicles.device
        scores = torch.empty((B, self.veh_count), device=device)

        for i, net in enumerate(self.vehicle_networks):
            my_state = vehicles[:, i:i + 1, :]
            scores[:, i] = net(my_state, vehicles, customers,
                               veh_done, cust_mask).squeeze(1)

        if self.aggregator_type == "argmax":
            scores.masked_fill_(veh_done, -float('inf'))
            _, veh_idx = scores.max(dim=1, keepdim=True)
            return veh_idx, scores
        else:
            veh_emb = self.veh_embedding_agg(vehicles)
            agg_logits = self.aggregator(scores.detach(), veh_emb, veh_done)
            agg_logits.masked_fill_(veh_done, -float('inf'))
            _, veh_idx = agg_logits.max(dim=1, keepdim=True)
            return veh_idx, agg_logits

class AttentionLearner(nn.Module):
    def __init__(self, cust_feat_size, veh_state_size, model_size=128,
                 layer_count=3, head_count=8, ff_size=512,
                 tanh_xplor=10, greedy=False, veh_count=4,
                 aggregator_type="argmax"):
        super().__init__()
        self.model_size = model_size
        self.inv_sqrt_d = model_size ** -0.5
        self.tanh_xplor = tanh_xplor
        self.greedy = greedy
        self.depot_embedding = nn.Linear(cust_feat_size, model_size)
        self.cust_embedding = nn.Linear(cust_feat_size, model_size)
        self.cust_encoder = TransformerEncoder(layer_count, head_count, model_size, ff_size)
        self.veh_embedding = nn.Linear(veh_state_size, model_size)
        self.fleet_attention = MultiHeadAttention(head_count, model_size)
        self.veh_attention = MultiHeadAttention(head_count, model_size)
        self.cust_project = nn.Linear(model_size, model_size)
        self.dvnda = DVNDA(
            veh_count=veh_count,
            veh_state_size=veh_state_size,
            cust_feat_size=cust_feat_size,
            model_size=64, head_count=4,
            aggregator_type=aggregator_type,
        )
    def _encode_customers(self, customers, mask=None):
        cust_emb = torch.cat((
            self.depot_embedding(customers[:, 0:1, :]),
            self.cust_embedding(customers[:, 1:, :])
        ), dim=1)
        if mask is not None:
            mask = mask.to(cust_emb.device)
            cust_emb[mask] = 0
        self.cust_enc = self.cust_encoder(cust_emb, mask)
        self.fleet_attention.precompute(self.cust_enc)
        self.cust_repr = self.cust_project(self.cust_enc)
        if mask is not None:
            self.cust_repr[mask] = 0

    def _repr_vehicle(self, vehicles, veh_idx, veh_done):
        if vehicles.device != veh_idx.device:
            veh_idx = veh_idx.to(vehicles.device)
        veh_emb = self.veh_embedding(vehicles)
        fleet_repr = self.fleet_attention(veh_emb)
        veh_query = fleet_repr.gather(
            1, veh_idx.unsqueeze(2).expand(-1, -1, self.model_size))
        return self.veh_attention(veh_query, fleet_repr, fleet_repr)

    def _score_customers(self, veh_repr):
        compat = veh_repr.matmul(self.cust_repr.transpose(1, 2))
        compat *= self.inv_sqrt_d
        if self.tanh_xplor is not None:
            compat = self.tanh_xplor * compat.tanh()
        return compat

    def _get_logp(self, compat, veh_mask):
        veh_mask = veh_mask.to(compat.device)
        compat[veh_mask] = -float('inf')
        return compat.log_softmax(dim=2).squeeze(1)

    def step(self, dyna):
        veh_repr = self._repr_vehicle(dyna.vehicles, dyna.cur_veh_idx, dyna.veh_done)
        compat = self._score_customers(veh_repr)
        logp = self._get_logp(compat, dyna.cur_veh_mask)
        if self.greedy:
            cust_idx = logp.argmax(dim=1, keepdim=True)
        else:
            cust_idx = logp.exp().multinomial(1)
        return cust_idx, logp.gather(1, cust_idx)

    def to(self, device):
        super().to(device)
        if self.dvnda is not None:
            self.dvnda = self.dvnda.to(device)
        return self

    def forward(self, dyna):
        model_device = self.depot_embedding.weight.device
        if model_device != dyna.nodes.device:
            self.to(dyna.nodes.device)
        if self.dvnda is not None:
            self.dvnda = self.dvnda.to(dyna.nodes.device)
        dyna.dvnda = self.dvnda
        dyna.reset()
        self._encode_customers(dyna.nodes, dyna.cust_mask)
        actions, logps, rewards = [], [], []
        while not dyna.done:
            if dyna.new_customers:
                self._encode_customers(dyna.nodes, dyna.cust_mask)
                dyna.new_customers = False
            cust_idx, logp = self.step(dyna)
            actions.append((dyna.cur_veh_idx.clone(), cust_idx))
            logps.append(logp)
            reward = dyna.step(cust_idx)
            rewards.append(reward)
        return actions, logps, rewards

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    seed = 4321
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        from tqdm import tqdm
        TQDM_ENABLED = True
    except ImportError:
        class tqdm:
            def __init__(self, iterable, total=-1, desc=""):
                self.iterable = iterable
                self.total = total if total > 0 else len(iterable)
                self.desc = desc

            def __iter__(self):
                print("\r{}  0% ...".format(self.desc), end='', flush=True)
                for i, elem in enumerate(self.iterable):
                    yield elem
                    print("\r{} {: 4.0%} ...".format(self.desc, (i + 1) / self.total), end='', flush=True)
                print(" Done!")

        TQDM_ENABLED = False


    n = 50
    print(f"\nProcessing n={n}")

    data = torch.load("./data/dcvrp_n{}m{}/norm_data.pyth".format(n, 10))


    batch_size = 100 if torch.cuda.is_available() else 64
    loader = DataLoader(data,
                        batch_size=batch_size,
                        pin_memory=True if torch.cuda.is_available() else False,
                        num_workers=0)

    learner = AttentionLearner(5, 4)
    learner = learner.to(device)

    chkpt = torch.load("./output/50/chkpt_ep100.pyth".format(n, n // 5),
                       map_location=device)
    learner.load_state_dict(chkpt["model"])

    learner.eval()
    learner.greedy = True

    costs = []
    qos = []

    with torch.no_grad():
        for b, batch in enumerate(tqdm(loader, desc=f"Processing n={n}")):
            if isinstance(batch, (list, tuple)):
                batch = [item.to(device, non_blocking=True) if torch.is_tensor(item) else item for item in batch]
            elif torch.is_tensor(batch):
                batch = batch.to(device, non_blocking=True)
            elif hasattr(batch, 'to'):
                batch = batch.to(device, non_blocking=True)

            env = DCVRP_Environment(data, batch, pending_cost=0)

            _, _, rewards = learner(env)

            cost_batch = -torch.stack(rewards).sum(0).squeeze(-1)
            costs.append(cost_batch.cpu())

            pending = (env.served ^ True).float().sum(-1) - 1
            qos_batch = 1 - pending / (env.nodes_count - 1)
            qos.append(qos_batch.cpu())

            if torch.cuda.is_available() and b % 10 == 0:
                torch.cuda.empty_cache()

    costs = torch.cat(costs, 0)
    qos = torch.cat(qos, 0)

    print("{:5.2f} +- {:5.2f} (qos={:.2%})".format(costs.mean(), costs.std(), qos.mean()))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


