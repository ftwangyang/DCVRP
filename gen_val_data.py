from data import DCVRP_Dataset
import numpy as np
import torch
import os
BATCH_SIZE = 100
SEED = 1234
torch.manual_seed(SEED)
def save_normalized_data(data, out_dir, filename="norm_data.pyth"):
    data.normalize()
    os.makedirs(out_dir, exist_ok=True)
    torch.save(data, os.path.join(out_dir, filename))

if __name__ == "__main__":
    for n, m in ((20,4), (35,7),(50, 10)):
        np.random.seed(1234)
        out_dir = "data/dcvrp_n{}m{}".format(n, m)
        data = DCVRP_Dataset.generate(BATCH_SIZE, n, m)
        save_normalized_data(data, out_dir)

