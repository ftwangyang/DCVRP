"""DVNDA-only, sequential seed experiment with an explicit validation gate.

This is a reduced-budget experiment, not a claim of exact paper reproduction.
Root model/environment are preserved in the run's source snapshot. In particular,
the argmax vehicle score gradient remains a disclosed estimator limitation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scipy.stats import ttest_rel
from env.dataset import DCVRPDataset, set_seed
from env.environment import DCVRPEnvironment
from models import AttentionLearner, build_selector

RATES = (.10, .25, .50, .75)
TARGETS = (8.31, 8.95, 10.47, 11.78)


def dataset(size, rate, generator):
    nodes = torch.zeros(size, 21, 5)
    nodes[:, :, :2] = torch.rand(size, 21, 2, generator=generator)
    nodes[:, 1:, 2] = torch.randint(5, 42, (size, 20), generator=generator) / 150
    nodes[:, 1:, 3] = torch.randint(10, 32, (size, 20), generator=generator) / 480
    releases = torch.poisson(torch.full((size, 20), 240.5), generator=generator).clamp(1, 480)
    order = torch.rand(size, 20, generator=generator).argsort(1)
    dynamic = torch.zeros(size, 20, dtype=torch.bool)
    dynamic.scatter_(1, order[:, :round(20 * rate)], True)
    nodes[:, 1:, 4] = releases * dynamic / 480
    return DCVRPDataset(4, 1., 480., nodes)


def split(size, seed):
    generator = torch.Generator().manual_seed(seed)
    return [dataset(size, rate, generator) for rate in RATES]


def gate_fails(epoch, max_gap, gate_epoch=10, threshold=8.):
    return epoch >= gate_epoch and (not math.isfinite(max_gap) or max_gap > threshold)


def rollout(model, data, device):
    env = DCVRPEnvironment(data, nodes=data.nodes.to(device), pending_cost=5.)
    _, logps, rewards = model(env)
    returns = torch.stack(rewards).sum(0)
    pending = (~env.served[:, 1:]).sum(1)
    expected = -(env.route_distance() + 5 * pending)
    if not torch.allclose(returns.flatten(), expected, atol=2e-4, rtol=1e-5):
        raise RuntimeError('Reward and closed-route cost disagree')
    return returns, logps, env


@torch.no_grad()
def evaluate(model, data_split, device):
    model.eval()
    model.greedy = True
    rows, costs = [], []
    for rate, target, data in zip(RATES, TARGETS, data_split):
        returns, _, env = rollout(model, data, device)
        distance = env.route_distance()
        measured = distance.mean().item()
        rows.append(dict(rate=rate, distance=measured, sd=distance.std().item(),
                         qos=env.qos().mean().item() * 100,
                         gap_percent=100 * (measured / target - 1)))
        costs.append(-returns.flatten().cpu())
    costs = torch.cat(costs)
    gaps = [abs(row['gap_percent']) for row in rows]
    return dict(rows=rows, objective=costs.mean().item(),
                mape=sum(gaps)/4, max_gap=max(gaps)), costs


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def train_one(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    set_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 71009)
    model = AttentionLearner(build_selector('DVNDA', vehicle_count=4)).to(args.device)
    baseline = copy.deepcopy(model).eval()
    baseline.greedy = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    validation = split(args.val_size, 3194813)
    best = float('inf')
    status = dict(state='running', seed=args.seed, method='DVNDA', steps=args.steps,
                  max_epochs=args.epochs, gate_epoch=args.gate_epoch, threshold=8.,
                  gate_metric='maximum absolute percentage gap over four validation rates',
                  validation_seed=3194813, holdout_seed=73927431,
                  protocol='unit square; constant Poisson240.5; speed480; penalty5',
                  checkpoint_selection='minimum validation distance plus unserved penalty',
                  limitation='argmax vehicle scores included in policy gradient; estimator not proven unbiased')
    write_json(output / 'status.json', status)
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        model.greedy = False
        for step in range(1, args.steps + 1):
            data = dataset(args.batch, RATES[(step-1) % 4], generator)
            with torch.no_grad():
                baseline_returns, _, _ = rollout(baseline, data, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for _ in range(3):
                returns, logps, _ = rollout(model, data, args.device)
                loss = -(torch.stack(logps).sum(0) * (returns-baseline_returns).detach()).mean()/3
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                loss_value += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 10 == 0:
                print(f'epoch={epoch} step={step}/{args.steps} loss={loss_value:.4f}', flush=True)
        metrics, candidate_costs = evaluate(model, validation, args.device)
        _, baseline_costs = evaluate(baseline, validation, args.device)
        if candidate_costs.mean() < baseline_costs.mean() and ttest_rel(candidate_costs, baseline_costs, alternative='less').pvalue < .05:
            baseline.load_state_dict(model.state_dict())
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          baseline=baseline.state_dict(), epoch=epoch, seed=args.seed,
                          metrics=metrics, data_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        torch.save(checkpoint, output / 'last.pt')
        if metrics['objective'] < best:
            best = metrics['objective']
            torch.save(checkpoint, output / 'best.pt')
        record = dict(epoch=epoch, updates=epoch*args.steps, seconds=time.monotonic()-start, **metrics)
        with (output/'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        status.update(epoch=epoch, metrics=metrics)
        write_json(output/'status.json', status)
        print(json.dumps(record), flush=True)
        if gate_fails(epoch, metrics['max_gap'], args.gate_epoch):
            status['state'] = 'stopped_by_8_percent_gate'
            break
    else:
        status['state'] = 'budget_complete'
    selected = torch.load(output/'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(selected['model'])
    holdout, _ = evaluate(model, split(args.val_size, 73927431), args.device)
    write_json(output/'holdout.json', dict(selected_epoch=selected['epoch'], **holdout))
    status['holdout'] = holdout
    write_json(output/'status.json', status)


def launch(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for folder in ('env', 'models'):
        shutil.copytree(ROOT/folder, snapshot/folder, ignore=shutil.ignore_patterns('__pycache__'))
    (snapshot/'scripts').mkdir()
    shutil.copy2(__file__, snapshot/'scripts'/Path(__file__).name)
    manifest = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in snapshot.rglob('*') if p.is_file()}
    write_json(output/'manifest.json', manifest)
    results = []
    write_json(output/'status.json', dict(state='running', seeds=args.seeds, results=results))
    for seed in args.seeds:
        run = output/f'seed_{seed}'
        command = [sys.executable, '-u', str(snapshot/'scripts'/Path(__file__).name),
                   '--worker', '--seed', str(seed), '--output', str(run), '--device', args.device,
                   '--steps', str(args.steps), '--epochs', str(args.epochs),
                   '--gate-epoch', str(args.gate_epoch), '--batch', str(args.batch),
                   '--val-size', str(args.val_size)]
        with (output/f'seed_{seed}.log').open('w', encoding='utf-8') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=snapshot)
            write_json(output/'status.json', dict(state='running', seeds=args.seeds,
                       current_seed=seed, child_pid=child.pid, results=results))
            code = child.wait()
        results.append(dict(seed=seed, exit_code=code,
                       result=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else None))
        write_json(output/'status.json', dict(state='running', seeds=args.seeds, results=results))
    write_json(output/'status.json', dict(state='complete' if all(r['exit_code']==0 for r in results) else 'failed', results=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1234, 2026])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gate-epoch', type=int, default=10)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--val-size', type=int, default=100)
    args = parser.parse_args()
    if min(args.steps, args.epochs, args.gate_epoch, args.batch, args.val_size) < 1:
        parser.error('Budgets must be positive')
    (train_one if args.worker else launch)(args)


if __name__ == '__main__':
    main()
