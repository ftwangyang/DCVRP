import copy
import torch

class Baseline:
    def __init__(self, learner, use_cumul_reward = False):
        self.learner = learner
        self.use_cumul = use_cumul_reward

    def __call__(self, vrp_dynamics):
        if self.use_cumul:
            actions, logps, rewards = self.learner(vrp_dynamics)
            rewards = torch.stack(rewards).sum(dim = 0)
            bl_vals = self.eval(vrp_dynamics)
        else:
            self.learner._encode_customers(vrp_dynamics.nodes, vrp_dynamics.cust_mask)
            vrp_dynamics.reset()
            actions, logps, rewards, bl_vals = [], [], [], []
            while not vrp_dynamics.done:
                veh_repr = self.learner._repr_vehicle(
                        vrp_dynamics.vehicles,
                        vrp_dynamics.cur_veh_idx,
                        vrp_dynamics.mask)
                compat = self.learner._score_customers(veh_repr)
                logp = self.learner._get_logp(compat, vrp_dynamics.cur_veh_mask)
                cust_idx = logp.exp().multinomial(1)
                bl_vals.append( self.eval_step(vrp_dynamics, compat, cust_idx) )
                actions.append( (vrp_dynamics.cur_veh_idx, cust_idx) )
                logps.append( logp.gather(1, cust_idx) )
                r = vrp_dynamics.step(cust_idx)
                rewards.append(r)
        self.update(rewards, bl_vals)
        return actions, logps, rewards, bl_vals

    def eval(self, vrp_dynamics):
        raise NotImplementedError()

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx):
        raise NotImplementedError()

    def update(self, rewards, bl_vals):
        pass

    def parameters(self):
        return []

    def state_dict(self, destination = None):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def to(self, device):
        pass

try:
    from tqdm import tqdm
    TQDM_ENABLED = True
except ImportError:
    class tqdm:
        def __init__(self, iterable, total = -1, desc = ""):
            self.iterable = iterable
            self.total = total if total > 0 else len(iterable)
            self.desc = desc
        def __iter__(self):
            print("\r{}  0% ...".format(self.desc), end = '', flush = True)
            for i,elem in enumerate(self.iterable):
                yield elem
                print("\r{} {: 4.0%} ...".format(self.desc, (i+1) / self.total), end = '', flush = True)
            print(" Done!")
    TQDM_ENABLED = False

try:
    import matplotlib
    from matplotlib import pyplot
    matplotlib.rcParams["backend"] = "Agg"
    MPL_ENABLED = True
except ImportError:
    matplotlib = None
    pyplot = None
    MPL_ENABLED = False

try:
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2
    ORTOOLS_ENABLED = True
except ImportError:
    pywrapcp = None
    routing_enums_pb2 = None
    ORTOOLS_ENABLED = False

import os as _os
for cand in [
        "./bin/LKH",
        _os.path.join(_os.environ.get("HOME", "~"), "LKH-3.0.5/LKH"),
        "/usr/local/bin/LKH",
        "/usr/bin/LKH"
        ]:
    if _os.path.isfile(cand) and _os.access(cand, _os.X_OK):
        LKH_ENABLED = True
        LKH_BIN = cand
        break
else:
    LKH_ENABLED = False
    LKH_BIN = None

try:
    from scipy.stats import ttest_rel
    SCIPY_ENABLED = True
except ImportError:
    ttest_rel = None
    SCIPY_ENABLED = False



class RolloutBaseline(Baseline):
    def __init__(self, learner, rollout_count = 1, update_threshold = 0.05):
        super().__init__(learner, True)

        if not SCIPY_ENABLED:
            raise RuntimeError("Cannot use rollout baseline without scipy.stats.ttest_rel")

        self.learner = learner
        self.policy = copy.deepcopy(learner)
        self.policy.eval()
        self.count = rollout_count
        self.thresh = update_threshold

    def eval(self, dyna):
        val = []
        with torch.no_grad():
            for it in range(self.count):
                _,_,rewards = self.policy(dyna)
                val.append( torch.stack(rewards).sum(dim = 0) )
        return torch.stack(val).mean(dim = 0)


    def update(self, rewards, bl_vals):

        if (rewards - bl_vals).mean().item() > 0:

            t, p = ttest_rel(rewards.cpu().numpy(), bl_vals.cpu().numpy())

            if p > 1 - self.thresh:
                self.policy.load_state_dict(self.learner.state_dict())

    def to(self, device):
        self.policy.to(device = device)
