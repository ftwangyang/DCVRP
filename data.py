import torch
from torch.utils.data import Dataset
import numpy as np

def create_logistics_distribution(horizon, cust_count, batch_size=1):
    lambda_values = torch.linspace(1, horizon, steps=cust_count)
    lambda_values = lambda_values.unsqueeze(0).unsqueeze(-1)
    lambda_values = lambda_values.expand(batch_size, -1, -1)
    poisson_vals = torch.poisson(lambda_values)
    poisson_vals = torch.clamp(poisson_vals, min=1, max=horizon)
    return poisson_vals

class DCVRP_Dataset(Dataset):
    CUST_FEAT_SIZE = 5

    @classmethod
    def generate(cls,
                 batch_size=1,
                 cust_count=50,
                 veh_count=10,
                 veh_capa=150,
                 veh_speed=1,
                 min_cust_count=None,
                 cust_loc_range=(0, 101),
                 cust_dem_range=(5, 41),
                 horizon=480,
                 cust_dur_range=(10, 31),
                 dod=(0.1,0.25,0.5,0.75),
                 distribution={'data_type': 'uniform'}
               ):
        size = (batch_size, cust_count, 1)

        data_type = distribution.get('data_type', 'uniform')

        if data_type == 'uniform':
            locs = torch.randint(*cust_loc_range, (batch_size, cust_count + 1, 2), dtype=torch.float)
        elif data_type == 'cluster':
            n_cluster = distribution.get('n_cluster', 5)
            lower = distribution.get('lower', 0)
            upper = distribution.get('upper', 100)
            std = distribution.get('std', 5.0)
            center = np.random.rand(batch_size, n_cluster * 2)
            center = lower + (upper - lower) * center
            center = center.reshape(batch_size, n_cluster, 2)
            locs = torch.zeros(batch_size, cust_count + 1, 2)
            for j in range(batch_size):
                depot_idx = np.random.choice(cust_count + 1)
                locs[j, depot_idx] = torch.FloatTensor(1, 2).uniform_(lower, upper)
                for i in range(n_cluster):
                    if i < n_cluster - 1:
                        num = (cust_count) // n_cluster
                    else:
                        num = cust_count - (cust_count // n_cluster) * i
                    coords = torch.randn(num, 2) * std + torch.FloatTensor(center[j, i])
                    coords = torch.clamp(coords, lower, upper)
                    start_idx = i * (cust_count // n_cluster)
                    end_idx = start_idx + num
                    locs[j, start_idx + 1:end_idx + 1] = coords
        elif data_type == 'mixed':
            n_cluster_mix = distribution.get('n_cluster_mix', 5)
            lower = distribution.get('lower', 0)
            upper = distribution.get('upper', 100)
            std = distribution.get('std', 5.0)
            center = np.random.rand(batch_size, n_cluster_mix * 2)
            center = lower + (upper - lower) * center
            center = center.reshape(batch_size, n_cluster_mix, 2)
            locs = torch.zeros(batch_size, cust_count + 1, 2)

            for j in range(batch_size):

                depot = torch.FloatTensor(1, 2).uniform_(lower, upper)
                locs[j, 0] = depot

                num_customers = cust_count
                half_customers = num_customers // 2

                coords = torch.FloatTensor(num_customers, 2).uniform_(lower, upper)

                mutate_idx = np.random.choice(num_customers, half_customers, replace=False)


                base = half_customers // n_cluster_mix
                remainder = half_customers % n_cluster_mix

                cluster_sizes = [base] * n_cluster_mix
                for r in range(remainder):
                    cluster_sizes[r] += 1

                # 分配
                start = 0
                for i, num in enumerate(cluster_sizes):
                    if num == 0:
                        continue
                    idx = mutate_idx[start:start + num]
                    cluster_coords = torch.randn(num, 2) * std + torch.FloatTensor(center[j, i])
                    cluster_coords = torch.clamp(cluster_coords, lower, upper)
                    coords[idx] = cluster_coords
                    start += num

                locs[j, 1:] = coords

        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        dems = torch.randint(*cust_dem_range, size, dtype=torch.float)

        durs = torch.randint(*cust_dur_range, size, dtype=torch.float)

        if isinstance(dod, float):
            is_dyn = torch.empty(size).bernoulli_(dod)
        elif isinstance(dod, (list, tuple)) and len(dod) == 1:
            is_dyn = torch.empty(size).bernoulli_(dod[0])
        else:
            ratio = torch.tensor(dod)[torch.randint(0, len(dod), (batch_size,), dtype=torch.int64)]
            is_dyn = ratio[:, None, None].expand(*size).bernoulli()

        poisson_vals = create_logistics_distribution(horizon, cust_count)

        aprs = is_dyn * poisson_vals

        customers = torch.cat((locs[:, 1:], dems, durs, aprs), 2)


        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:, :, :2] = locs[:, 0:1]

        nodes = torch.cat((depot_node, customers), 1)


        if min_cust_count is None:
            cust_mask = None
        else:
            counts = torch.randint(min_cust_count + 1, cust_count + 2, (batch_size, 1), dtype=torch.int64)
            cust_mask = torch.arange(cust_count + 1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0


        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        return dataset

    def __init__(self, veh_count, veh_capa, veh_speed, nodes, cust_mask=None):
        self.veh_count = veh_count
        self.veh_capa = veh_capa
        self.veh_speed = veh_speed

        self.nodes = nodes
        self.batch_size, self.nodes_count, d = nodes.size()
        if d != self.CUST_FEAT_SIZE:
            raise ValueError(f"Expected {self.CUST_FEAT_SIZE} customer features per node, got {d}")
        self.cust_mask = cust_mask

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        if self.cust_mask is None:
            return self.nodes[idx]
        else:
            return self.nodes[idx], self.cust_mask[idx]

    def nodes_gen(self):
        if self.cust_mask is None:
            yield from self.nodes
        else:
            yield from (n[~m] for n, m in zip(self.nodes, self.cust_mask))

    def normalize(self):
        loc_scl = self.nodes[:, :, :2].max().item()
        loc_off = self.nodes[:, :, :2].min().item()
        loc_scl -= loc_off
        t_scl = float(480.0)

        self.nodes[:, :, :2] -= loc_off
        self.nodes[:, :, :2] /= loc_scl
        self.nodes[:, :, 2] /= self.veh_capa
        self.nodes[:, :, 3:] /= t_scl

        self.veh_capa = 1
        self.veh_speed *= t_scl / loc_scl

        return loc_scl, t_scl

    def save(self, fpath):
        torch.save({
            "veh_count": self.veh_count,
            "veh_capa": self.veh_capa,
            "veh_speed": self.veh_speed,
            "nodes": self.nodes,
            "cust_mask": self.cust_mask
        }, fpath)

    @classmethod
    def load(cls, fpath):
        data = torch.load(fpath)
        return cls(**data)
