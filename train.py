import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from itertools import chain, repeat, zip_longest
from data import DCVRP_Dataset
from learner import AttentionLearner, DCVRP_Environment
from rollout import RolloutBaseline
from critic import CriticBaseline
from args import parse_args, write_config_file
from tqdm import tqdm
import os.path

def save_checkpoint(args, ep, learner, optim, baseline = None, lr_sched = None):
    checkpoint = {
            "ep": ep,
            "model": learner.state_dict(),
            "optim": optim.state_dict()
            }
    if args.rate_decay is not None:
        checkpoint["lr_sched"] = lr_sched.state_dict()
    if args.baseline_type == "critic":
        checkpoint["critic"] = baseline.state_dict()
    torch.save(checkpoint, os.path.join(args.output_dir, "chkpt_ep{}.pyth".format(ep+1)))

def load_checkpoint(args, learner, optim, baseline = None, lr_sched = None):
    checkpoint = torch.load(args.resume_state)
    learner.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    if args.rate_decay is not None:
        lr_sched.load_state_dict(checkpoint["lr_sched"])
    if args.baseline_type == "critic":
        baseline.load_state_dict(checkpoint["critic"])
    return checkpoint["ep"]

def set_random_seed(seed):

    if seed is not None:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")

def reinforce_loss(logprobs, rewards, baseline=None, weights=None,
                   discount=1.0, reduction='mean'):

    if weights is None:
        weights = repeat(1.0)

    if isinstance(rewards, torch.Tensor):
        if baseline is None:
            baseline = torch.zeros_like(rewards)

        loss = torch.stack([-logp * w for logp, w in zip(logprobs, weights)]).sum(dim=0)
        loss *= (rewards - baseline.detach())

        if baseline.requires_grad:
            loss += F.smooth_l1_loss(baseline, rewards)
    else:
        if baseline is None:
            baseline = repeat(torch.zeros_like(rewards[0]))

        cumul = torch.zeros_like(rewards[0])
        vals = []
        for r in reversed(rewards):
            cumul = r + discount * cumul
            vals.append(cumul)
        vals.reverse()

        loss = []
        bl_loss = []
        for val, logp, bl, w in zip(vals, logprobs, baseline, weights):
            loss.append(-logp * (val - bl.detach()) * w)
            if bl.requires_grad:
                bl_loss.append(F.smooth_l1_loss(bl, val))

        loss = torch.stack(loss).sum(dim=0)
        if bl_loss:
            loss += torch.stack(bl_loss).sum(dim=0)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:  # reduction == 'mean'
        return loss.mean()

def train_epoch(args, data, Environment, env_params, bl_wrapped_learner,
                optim, device, ep):

    bl_wrapped_learner.learner.train()
    loader = DataLoader(data, args.batch_size, True)

    ep_loss = ep_prob = ep_val = ep_bl = ep_norm = 0

    desc = f"Ep.#{ep + 1:>3d}/{args.epoch_count:<3d}"
    with tqdm(loader, desc=desc) as progress:
        for minibatch in progress:
            if data.cust_mask is None:
                custs, mask = minibatch.to(device), None
            else:
                custs, mask = minibatch[0].to(device), minibatch[1].to(device)

            dyna = Environment(data, custs, mask, *env_params)
            actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
            loss = reinforce_loss(logps, rewards, bl_vals)

            prob = torch.stack(logps).sum(0).exp().mean()
            val = rewards.mean()
            bl = bl_vals[0].mean() if bl_vals is not None else 0

            optim.zero_grad()
            loss.backward()

            grad_norm = 0
            if args.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(
                    chain.from_iterable(grp["params"] for grp in optim.param_groups),
                    args.max_grad_norm
                )
            optim.step()

            postfix = f"l={loss:.4g} p={prob:9.4g} val={val:6.4g} bl={bl:6.4g} |g|={grad_norm:.4g}"
            progress.set_postfix_str(postfix)

            ep_loss += loss.item()
            ep_prob += prob.item()
            ep_val += val.item()
            ep_bl += bl.item() if isinstance(bl, torch.Tensor) else bl
            ep_norm += grad_norm

    return tuple(stat / args.iter_count for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm))

def test_epoch(args, test_env, learner, ref_costs):

    learner.eval()

    if args.problem_type[0] == "DCVRP":
        costs = test_env.nodes.new_zeros(test_env.minibatch_size)
        for _ in range(100):
            _, _, rewards = learner(test_env)
            costs -= torch.stack(rewards).sum(0).squeeze(-1)
        costs = costs / 100
    else:
        _, _, rs = learner(test_env)
        costs = -torch.stack(rs).sum(dim=0).squeeze(-1)

    mean = costs.mean()
    std = costs.std()
    gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()

    print(f"Cost on test dataset: {mean:5.2f} +- {std:5.2f} ({gap:.2%})")
    return mean.item(), std.item(), gap.item()

def print_val_summary(val_history, epoch=None):

    if not val_history:
        return

    current_info = f" (interrupted at epoch {epoch + 1})" if epoch is not None else ""
    print(f"\n{'=' * 80}")
    print(f"TRAINING {'INTERRUPTED' if epoch is not None else 'COMPLETED'}!")
    print(f"Final val: {val_history[-1]:.6f}{current_info}")

    print(f"Complete val history:")
    for i, val in enumerate(val_history):
        marker = ""
        if i == 0:
            marker = " (Initial)"
        elif i == len(val_history) - 1:
            marker = " (Final)" if epoch is None else " (Last)"
        print(f"  Epoch {i + 1}: {val:.6f}{marker}")

    if len(val_history) > 1:
        total_improvement = val_history[-1] - val_history[0]
        best_val = max(val_history)
        worst_val = min(val_history)
        best_epoch = val_history.index(best_val) + 1
        worst_epoch = val_history.index(worst_val) + 1

        print(f"Total improvement: {total_improvement:+.6f}")
        print(f"Best val: {best_val:.6f} (Epoch {best_epoch})")
        print(f"Worst val: {worst_val:.6f} (Epoch {worst_epoch})")
    print(f"{'=' * 80}")

def export_train_test_stats(args, start_ep, train_stats, test_stats):
    fpath = os.path.join(args.output_dir, "loss_gap.csv")
    with open(fpath, 'a') as f:
        f.write( (' '.join("{: >16}" for _ in range(9)) + '\n').format(
            "#EP", "#LOSS", "#PROB", "#VAL", "#BL", "#NORM", "#TEST_MU", "#TEST_STD", "#TEST_GAP"
            ))
        for ep, (tr,te) in enumerate( zip_longest(train_stats, test_stats, fillvalue=float('nan')), start = start_ep):
            f.write( ("{: >16d}" + ' '.join("{: >16.3g}" for _ in range(8)) + '\n').format(
                ep, *tr, *te))

def save_val_history(args, val_history, interrupted_epoch=None):
    if not val_history:
        return

    val_history_path = os.path.join(args.output_dir, "val_history.txt")
    with open(val_history_path, 'w') as f:
        title = "Val History (Interrupted Training)" if interrupted_epoch is not None else "Val History Summary"
        f.write(f"{title}\n")
        f.write("=" * 50 + "\n")

        for i, val in enumerate(val_history):
            f.write(f"Epoch {i + 1}: {val:.6f}\n")

        if interrupted_epoch is not None:
            f.write(f"\nTraining was interrupted at epoch {interrupted_epoch + 1}\n")
        elif len(val_history) > 1:
            f.write("\nSummary Statistics:\n")
            f.write(f"Total improvement: {val_history[-1] - val_history[0]:+.6f}\n")
            f.write(f"Best val: {max(val_history):.6f} (Epoch {val_history.index(max(val_history)) + 1})\n")
            f.write(f"Worst val: {min(val_history):.6f} (Epoch {val_history.index(min(val_history)) + 1})\n")
            f.write(f"Average val: {sum(val_history) / len(val_history):.6f}\n")

    print(f"Val history saved to {val_history_path}")

def main(args):
    set_random_seed(args.rng_seed)
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    verbose_print = print if args.verbose else lambda *a, **k: None

    gen_params = [
        args.customers_count, args.vehicles_count, args.veh_capa,
        args.veh_speed, args.min_cust_count, args.loc_range, args.dem_range
    ]

    verbose_print(f"Generating {args.iter_count * args.batch_size} {args.problem_type.upper()} training samples...",
                  end=" ", flush=True)
    train_data = DCVRP_Dataset.generate(args.iter_count * args.batch_size, *gen_params)
    train_data.normalize()
    verbose_print("Done.")

    verbose_print(f"Generating {args.test_batch_size} {args.problem_type.upper()} test samples...",
                  end=" ", flush=True)
    test_data = DCVRP_Dataset.generate(args.test_batch_size, *gen_params)
    test_data.normalize()
    verbose_print("Done.")

    env_params = [args.pending_cost]
    test_env = DCVRP_Environment(test_data, None, None, *env_params)
    test_env.nodes = test_env.nodes.to(device)

    verbose_print("Initializing attention model...", end=" ", flush=True)
    learner = AttentionLearner(
        DCVRP_Dataset.CUST_FEAT_SIZE,
        DCVRP_Environment.VEH_STATE_SIZE,
        args.model_size,
        args.layer_count,
        args.head_count,
        args.ff_size,
        args.tanh_xplor,
        veh_count=args.vehicles_count,
        aggregator_type=args.aggregator_type,
    )
    learner.to(device)
    verbose_print("Done.")

    verbose_print(f"Initializing '{args.baseline_type}' baseline...", end=" ", flush=True)
    baseline_map = {
        "rollout": lambda: RolloutBaseline(learner, args.rollout_count, args.rollout_threshold),
        "critic": lambda: CriticBaseline(learner, args.customers_count,
                                         args.critic_use_qval, args.loss_use_cumul)
    }

    if args.baseline_type == "rollout":
        args.loss_use_cumul = True

    baseline = baseline_map[args.baseline_type]()
    baseline.to(device)
    verbose_print("Done.")

    verbose_print("Initializing Adam optimizer...", end=" ", flush=True)
    optim = Adam(learner.parameters(), args.learning_rate)
    lr_sched = None
    if args.rate_decay is not None:
        lr_sched = LambdaLR(optim, lambda ep: args.learning_rate * args.rate_decay ** ep)
    verbose_print("Done.")

    verbose_print("Creating output dir...", end=" ", flush=True)
    if args.output_dir is None:
        timestamp = time.strftime("%y%m%d-%H%M")
        args.output_dir = f"./output/{args.problem_type.upper()}n{args.customers_count}m{args.vehicles_count}_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print(f"'{args.output_dir}' created.")

    start_ep = 0 if args.resume_state is None else load_checkpoint(
        args, learner, optim, baseline, lr_sched)

    verbose_print("Running...")
    train_stats = []
    test_stats = []
    val_history = []

    try:
        for ep in range(start_ep, args.epoch_count):

            ep_stats = train_epoch(args, train_data, DCVRP_Environment,
                                   env_params, baseline, optim, device, ep)
            ep_loss, ep_prob, ep_val, ep_bl, ep_norm = ep_stats
            train_stats.append(ep_stats)
            val_history.append(ep_val)

            print(f"\nEpoch {ep + 1}/{args.epoch_count} - Val: {ep_val:.6f}")

            if lr_sched is not None:
                lr_sched.step()
            if args.pend_cost_growth is not None:
                env_params[0] *= args.pend_cost_growth

            if (ep + 1) % args.checkpoint_period == 0:
                save_checkpoint(args, ep, learner, optim, baseline, lr_sched)

        print_val_summary(val_history)
        save_val_history(args, val_history)

    except KeyboardInterrupt:
        print_val_summary(val_history, ep)
        save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
        save_val_history(args, val_history, ep)

    finally:
        export_train_test_stats(args, start_ep, train_stats, test_stats)

if __name__ == "__main__":
    main(parse_args())