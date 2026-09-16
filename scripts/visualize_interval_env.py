"""Visualize keep/destroy cuts and the final Gantt for one DCVRP instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import DCVRPEnvironment, generate_dataset, set_seed
from models import AttentionLearner, build_selector


def run_episode(n: int, phi: float, seed: int, device: str, checkpoint=None):
    set_seed(seed)
    data = generate_dataset(
        batch_size=1,
        customer_count=n,
        vehicle_count=max(1, round(n / 5)),
        dynamic_rate=phi,
        seed=seed,
    )
    selector = build_selector("DVNDA", vehicle_count=data.veh_count)
    model = AttentionLearner(selector).to(device)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model'], strict=True)
    model.eval()
    model.greedy = True
    model.vehicle_greedy = True
    env = DCVRPEnvironment(data, nodes=data.nodes.to(device), pending_cost=5.0, record_trace=True)
    with torch.no_grad():
        _, _, rewards = model(env)
    reward = float(torch.stack(rewards).sum())
    expected = -float(env.route_distance()[0]) - 5 * int((~env.served[0, 1:]).sum())
    assert abs(reward-expected) < 2e-4
    return data, env


def plot_routes(data, env, out: Path):
    nodes = data.nodes[0].cpu()
    logs = env.interval_logs
    cols = min(5, max(1, len(logs)))
    rows = max(1, (len(logs) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    depot = nodes[0, :2].numpy()
    for idx, log in enumerate(logs):
        ax = axes[idx // cols][idx % cols]
        ax.scatter(depot[0], depot[1], c="black", marker="s", s=30, zorder=3)
        ax.scatter(nodes[1:, 0], nodes[1:, 1], c="lightgray", s=12)
        for cid, point in enumerate(nodes):
            ax.text(float(point[0])+.01, float(point[1])+.01, str(cid), fontsize=6)
        for event in log['events']:
            origin, dest = event['origin'], event['destination']
            destroyed_event = event['decision'] == 'destroyed'
            ax.plot([origin[0], dest[0]], [origin[1], dest[1]],
                    color='0.75' if destroyed_event else plt.cm.tab10.colors[event['vehicle']],
                    linestyle='--' if destroyed_event else '-', lw=.8 if destroyed_event else 1.5)
        kept = log["kept"]
        destroyed = log["destroyed"]
        revealed = log["revealed"]
        if kept:
            ax.scatter(nodes[kept, 0], nodes[kept, 1], c="tab:green", s=22, label="kept")
        if destroyed:
            ax.scatter(nodes[destroyed, 0], nodes[destroyed, 1], c="tab:red", s=22, label="destroyed")
        if revealed:
            ax.scatter(nodes[revealed, 0], nodes[revealed, 1], c="tab:blue", s=22, label="new")
        xy = torch.tensor(log["vehicle_xy"])
        ax.scatter(xy[:, 0], xy[:, 1], c="gold", marker="^", s=28, zorder=4)
        ax.set_title(f"Interval {idx+1}: {idx*48}-{(idx+1)*48} min\nKeep {len(kept)} / release {len(destroyed)} / new {len(revealed)}", fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
    for j in range(len(logs), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle('Solid lines: committed legs (vehicle colors) | dashed: removed plan | red nodes: removed | blue: next interval disclosures\nYellow triangles: committed destination at its availability time, not interpolated physical position', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_gantt(env, out: Path):
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.tab10.colors
    for log in env.interval_logs:
        t1 = log["boundary"] * 480
        ax.axvline(t1, color="0.8", lw=0.8)
        for event in log['events']:
            if event['decision'] == 'destroyed':
                continue
            v = event['vehicle']
            start, arrival, finish = [event[k]*480 for k in ('start','arrival','finish')]
            ax.barh(v, arrival-start, left=start, height=.6, color=colors[v], alpha=.4)
            ax.barh(v, finish-arrival, left=arrival, height=.6, color=colors[v])
            if finish-start > 1:
                ax.text((start+finish)/2, v, str(event['customer']), ha='center', va='center', fontsize=7)
    ax.set_xlabel('Time (minutes); pale = travel, solid = service; labels = customer (0 = depot)')
    ax.set_yticks(range(env.veh_count), [f'Vehicle {v+1}' for v in range(env.veh_count)])
    ax.set_xlim(0, 480)
    ax.set_title('Actual committed travel and service schedule')
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def verify_trace(data, env):
    """Reconstruct full tours from committed events, independent of cost accumulator."""
    nodes = data.nodes[0].cpu()
    positions = nodes[0, :2].repeat(env.veh_count, 1)
    times = [0.] * env.veh_count
    caps = [1.] * env.veh_count
    seen = set()
    distance = 0.
    routes = [[0] for _ in range(env.veh_count)]
    assert len(env.interval_logs) == 10
    for log in env.interval_logs:
        depot_counts = [0]*env.veh_count
        for e in log['events']:
            if e['decision'] == 'destroyed':
                continue
            v, c = e['vehicle'], e['customer']
            assert e['start'] + 1e-5 >= times[v]
            assert torch.allclose(positions[v], torch.tensor(e['origin']), atol=1e-5)
            assert torch.allclose(nodes[c, :2], torch.tensor(e['destination']), atol=1e-5)
            leg = float(torch.norm(nodes[c, :2] - positions[v]))
            assert abs(e['arrival'] - e['start'] - leg/data.veh_speed) < 1e-5
            assert abs(e['finish'] - e['arrival'] - float(nodes[c, 3])) < 1e-5
            if c:
                assert c not in seen
                seen.add(c)
                assert float(nodes[c, 4]) <= log['interval'] * env.segment_duration + 1e-6
                caps[v] -= float(nodes[c, 2])
                assert caps[v] >= -1e-5
            elif leg > 1e-8:
                depot_counts[v] += 1
            assert abs(e['capacity'] - caps[v]) < 1e-5
            assert depot_counts[v] <= 1
            distance += leg
            positions[v] = nodes[c, :2]
            times[v] = e['finish']
            if c or leg > 1e-8:
                routes[v].append(c)
    assert all(t <= env.horizon + 1e-5 for t in times)
    assert torch.allclose(positions, nodes[0, :2].expand_as(positions), atol=1e-5)
    assert abs(distance - float(env.route_distance()[0])) < 2e-4
    assert seen == set(env.served[0, 1:].nonzero().flatten().add(1).tolist())
    return dict(passed=True, independently_reconstructed_distance=distance,
                served=len(seen), routes=routes, remaining_capacities=caps)


def plot_final(data, verification, out):
    fig, ax = plt.subplots(figsize=(8, 7))
    nodes = data.nodes[0].cpu()
    ax.scatter(nodes[1:, 0], nodes[1:, 1], c='0.65', s=30)
    ax.scatter(nodes[0, 0], nodes[0, 1], c='black', marker='s', s=65)
    for c, p in enumerate(nodes):
        ax.text(float(p[0])+.01, float(p[1])+.01, str(c), fontsize=9)
    for v, route in enumerate(verification['routes']):
        color = plt.cm.tab10.colors[v]
        for a,b in zip(route, route[1:]):
            ax.annotate('', xy=nodes[b, :2], xytext=nodes[a, :2],
                        arrowprops=dict(arrowstyle='->', color=color, alpha=.65))
        ax.plot([], [], color=color, label=f'Vehicle {v+1}: '+ ' > '.join(map(str, route)))
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.08), fontsize=7)
    ax.set_aspect('equal')
    ax.set_title(f"Committed routes | distance={verification['independently_reconstructed_distance']:.3f} | served={verification['served']}")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--phi", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--audit-seeds', type=int, default=0,
                        help='Independently audit this many seeds at each of four dynamic rates')
    parser.add_argument("--out", type=Path, default=Path("results/interval_trace"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    data, env = run_episode(args.n, args.phi, args.seed, args.device, args.checkpoint)
    verification = verify_trace(data, env)
    verification['checkpoint'] = str(args.checkpoint) if args.checkpoint else 'untrained model'
    if args.checkpoint:
        verification['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.audit_seeds:
        audits = []
        for rate in (.1, .25, .5, .75):
            for seed in range(args.audit_seeds):
                audit = verify_trace(*run_episode(args.n, rate, seed, args.device, args.checkpoint))
                audits.append(dict(rate=rate, seed=seed, **audit))
        verification['additional_audits'] = audits
    verification['source_hashes'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('env/environment.py', 'env/dataset.py', 'train.py',
                     'models/attention_model.py', 'models/selectors.py', 'models/transformer.py')}
    (args.out/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    (args.out / "interval_logs.json").write_text(
        json.dumps(env.interval_logs, indent=2), encoding="utf-8"
    )
    plot_routes(data, env, args.out / "interval_routes.png")
    plot_gantt(env, args.out / "interval_gantt.png")
    plot_final(data, verification, args.out/'final_routes.png')
    print(f"served={int(env.served[0, 1:].sum())}/{args.n}")
    print(f"distance={float(env.route_distance()[0]):.3f} qos={float(env.qos()[0]):.3f}")
    print(f"intervals={len(env.interval_logs)} wrote {args.out}")
    for log in env.interval_logs:
        print(
            f"  r={log['interval']} keep={log['kept']} "
            f"destroy={log['destroyed']} new={log['revealed']}"
        )


if __name__ == "__main__":
    main()
