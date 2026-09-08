"""Evaluate trained DVNDA selectors on directed multi-city road networks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from .protocol import load_model
from .road_data import CITIES, build_graph, download_osm_json, generate_road_dataset, graph_metadata
from .road_environment import RoadExperimentalEnvironment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("experiments/checkpoints/architecture"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/real_roads"))
    parser.add_argument("--cache-root", type=Path, default=Path("experiments/data/osm"))
    parser.add_argument("--selector", default="independent")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    parser.add_argument("--cities", nargs="+", default=list(CITIES))
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--radius-m", type=int, default=1200)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    metadata = []
    city_data = {}
    city_selections = {}
    for city in args.cities:
        print(f"OSM city={city}", flush=True)
        payload = download_osm_json(city, args.cache_root, radius_m=args.radius_m)
        graph = build_graph(payload)
        metadata.append(graph_metadata(city, graph, payload))
        city_data[city] = generate_road_dataset(
            graph,
            args.instances,
            customer_count=20,
            vehicle_count=4,
            dynamic_ratio=0.5,
            seed=20_265_000 + list(CITIES).index(city),
        )
        city_selections[city] = [
            [int(node) for node in selection]
            for selection in city_data[city][3]
        ]

    for train_seed in args.seeds:
        checkpoint = args.checkpoint_root / f"{args.selector}_seed{train_seed}.pt"
        model, config, selector_name, _ = load_model(checkpoint, device)
        model.eval()
        model.greedy = True
        for city, (dataset, distance, travel, _) in city_data.items():
            conditions = [
                ("normal", 0.0, 0.0),
                ("peak_congestion", 0.10, 0.45),
            ]
            for condition, travel_noise, congestion in conditions:
                environment = RoadExperimentalEnvironment(
                    dataset,
                    nodes=dataset.nodes.to(device),
                    distance_matrix=distance.to(device),
                    travel_time_matrix=travel.to(device),
                    pending_cost=config.pending_cost,
                    segment_count=config.segment_count,
                    horizon=1.0,
                    travel_noise=travel_noise,
                    congestion=congestion,
                    stochastic_seed=20_266_000 + train_seed,
                )
                started = time.perf_counter()
                with torch.no_grad():
                    _, _, rewards = model(environment)
                elapsed = time.perf_counter() - started
                frame = pd.DataFrame(environment.metrics())
                frame["cost_with_penalty"] = -torch.stack(rewards).sum(0).squeeze(-1).cpu().numpy()
                frame["penalty"] = frame["cost_with_penalty"] - frame["distance"]
                frame["wall_time_ms_per_instance"] = elapsed * 1000.0 / len(frame)
                frame["city"] = city
                frame["condition"] = condition
                frame["travel_noise"] = travel_noise
                frame["congestion"] = congestion
                frame["train_seed"] = train_seed
                frame["selector"] = selector_name
                frame["instance"] = range(len(frame))
                frames.append(frame)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(raw_dir / "real_road_instances.csv", index=False)
    (raw_dir / "road_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (raw_dir / "road_sampled_node_ids.json").write_text(
        json.dumps(city_selections, indent=2), encoding="utf-8"
    )
    print(raw.groupby(["city", "condition"])[[
        "cost_with_penalty", "distance", "qos", "response_time_min"
    ]].mean())


if __name__ == "__main__":
    main()
