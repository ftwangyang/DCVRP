"""Training, evaluation, and source-data export for reviewer experiments."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from .data_generation import generate_dataset, mean_revelation_time
from .environment import ExperimentalEnvironment
from .model import ExperimentalAttentionLearner, JointPairLearner
from .selectors import build_selector


@dataclass(frozen=True)
class StudyConfig:
    customer_count: int = 20
    vehicle_count: int = 4
    dynamic_ratio: float = 0.5
    arrival_profile: str = "uniform"
    spatial: str = "uniform"
    segment_count: int = 10
    pending_cost: float = 5.0
    model_size: int = 64
    layer_count: int = 2
    head_count: int = 4
    ff_size: int = 128
    selector_size: int = 32
    selector_heads: int = 4
    tanh_exploration: float = 10.0
    learning_rate: float = 1.0e-4
    epochs: int = 5
    steps_per_epoch: int = 12
    batch_size: int = 16
    test_size: int = 100
    validation_size: int = 24
    gradient_clip: float = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(selector_name: str, config: StudyConfig, device: torch.device):
    if selector_name.lower() == "joint_pair":
        model = JointPairLearner(
            config.vehicle_count,
            model_size=config.model_size,
            layer_count=config.layer_count,
            head_count=config.head_count,
            ff_size=config.ff_size,
            tanh_exploration=config.tanh_exploration,
        )
        return model.to(device)
    selector = build_selector(
        selector_name,
        config.vehicle_count,
        selector_size=config.selector_size,
        selector_heads=config.selector_heads,
    )
    model = ExperimentalAttentionLearner(
        selector,
        model_size=config.model_size,
        layer_count=config.layer_count,
        head_count=config.head_count,
        ff_size=config.ff_size,
        tanh_exploration=config.tanh_exploration,
    )
    return model.to(device)


def _episode(
    model: ExperimentalAttentionLearner,
    config: StudyConfig,
    device: torch.device,
    data_seed: int,
    greedy: bool,
    travel_noise: float = 0.0,
    service_noise: float = 0.0,
    congestion: float = 0.0,
    batch_size_override: int | None = None,
):
    batch_size = (
        batch_size_override
        if batch_size_override is not None
        else (config.test_size if greedy else config.batch_size)
    )
    dataset = generate_dataset(
        batch_size,
        customer_count=config.customer_count,
        vehicle_count=config.vehicle_count,
        dynamic_ratio=config.dynamic_ratio,
        arrival_profile=config.arrival_profile,
        spatial=config.spatial,
        seed=data_seed,
    )
    environment = ExperimentalEnvironment(
        dataset,
        nodes=dataset.nodes.to(device),
        pending_cost=config.pending_cost,
        segment_count=config.segment_count,
        horizon=1.0,
        travel_noise=travel_noise,
        service_noise=service_noise,
        congestion=congestion,
        stochastic_seed=data_seed + 91,
    )
    model.greedy = greedy
    actions, log_probabilities, rewards = model(environment)
    cost = -torch.stack(rewards).sum(0).squeeze(-1)
    log_probability = torch.stack(log_probabilities).sum(0).squeeze(-1)
    return dataset, environment, cost, log_probability


def train_one(
    selector_name: str,
    seed: int,
    config: StudyConfig,
    checkpoint_dir: Path,
    history_dir: Path,
    device: torch.device,
) -> tuple[ExperimentalAttentionLearner, pd.DataFrame, dict]:
    set_seed(seed)
    model = build_model(selector_name, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    history_rows = []
    training_started = time.perf_counter()
    peak_memory = 0
    final_selector_grad_norm = 0.0
    best_validation_cost = float("inf")
    best_epoch = 0
    best_state = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(config.epochs):
        epoch_costs = []
        epoch_losses = []
        epoch_started = time.perf_counter()
        for step in range(config.steps_per_epoch):
            data_seed = seed * 100_000 + epoch * config.steps_per_epoch + step
            _, _, cost, log_probability = _episode(
                model, config, device, data_seed=data_seed, greedy=False
            )
            standardized_cost = (cost - cost.mean()) / cost.std().clamp_min(1.0e-6)
            loss = (standardized_cost.detach() * log_probability).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), config.gradient_clip)
            selector_gradients = [
                parameter.grad
                for parameter in model.selector.parameters()
                if parameter.grad is not None
            ]
            final_selector_grad_norm = float(
                torch.sqrt(sum((gradient.detach() ** 2).sum() for gradient in selector_gradients))
            ) if selector_gradients else 0.0
            optimizer.step()
            epoch_costs.append(float(cost.mean().detach().cpu()))
            epoch_losses.append(float(loss.detach().cpu()))

        epoch_seconds = time.perf_counter() - epoch_started
        model.eval()
        with torch.no_grad():
            _, validation_environment, validation_cost, _ = _episode(
                model,
                config,
                device,
                data_seed=88_000_000 + seed,
                greedy=True,
                batch_size_override=config.validation_size,
            )
        validation_metrics = validation_environment.metrics()
        validation_cost_mean = float(validation_cost.mean().cpu())
        if validation_cost_mean < best_validation_cost:
            best_validation_cost = validation_cost_mean
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        history_rows.append({
            "selector": selector_name,
            "seed": seed,
            "epoch": epoch + 1,
            "train_cost_mean": float(np.mean(epoch_costs)),
            "train_loss_mean": float(np.mean(epoch_losses)),
            "epoch_seconds": epoch_seconds,
            "selector_grad_norm": final_selector_grad_norm,
            "validation_cost_mean": validation_cost_mean,
            "validation_distance_mean": float(validation_metrics["distance"].mean()),
            "validation_qos_mean": float(validation_metrics["qos"].mean()),
        })

    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    model.load_state_dict(best_state)
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    checkpoint_path = checkpoint_dir / f"{selector_name}_seed{seed}.pt"
    torch.save({
        "selector": selector_name,
        "seed": seed,
        "config": asdict(config),
        "model": model.state_dict(),
    }, checkpoint_path)
    history = pd.DataFrame(history_rows)
    history.to_csv(history_dir / f"{selector_name}_seed{seed}.csv", index=False)
    resources = {
        "selector": selector_name,
        "seed": seed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selector_parameter_count": sum(p.numel() for p in model.selector.parameters()),
        "training_seconds": training_seconds,
        "peak_memory_bytes": peak_memory,
        "final_selector_grad_norm": final_selector_grad_norm,
        "best_epoch": best_epoch,
        "best_validation_cost": best_validation_cost,
        "checkpoint": str(checkpoint_path),
    }
    return model, history, resources


def evaluate_one(
    model: ExperimentalAttentionLearner,
    selector_name: str,
    train_seed: int,
    config: StudyConfig,
    device: torch.device,
    test_seed: int = 20260821,
    scenario: str = "in_distribution",
    travel_noise: float = 0.0,
    service_noise: float = 0.0,
    congestion: float = 0.0,
) -> pd.DataFrame:
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        dataset, environment, cost, _ = _episode(
            model,
            config,
            device,
            data_seed=test_seed,
            greedy=True,
            travel_noise=travel_noise,
            service_noise=service_noise,
            congestion=congestion,
        )
    elapsed = time.perf_counter() - started
    metrics = environment.metrics()
    rows = pd.DataFrame(metrics)
    rows.insert(0, "instance", np.arange(len(rows), dtype=int))
    rows.insert(0, "scenario", scenario)
    rows.insert(0, "train_seed", train_seed)
    rows.insert(0, "selector", selector_name)
    rows["cost_with_penalty"] = cost.detach().cpu().numpy()
    rows["penalty"] = rows["cost_with_penalty"] - rows["distance"]
    rows["wall_time_ms_per_instance"] = 1000.0 * elapsed / len(rows)
    rows["dynamic_ratio"] = config.dynamic_ratio
    rows["mean_revelation_time"] = mean_revelation_time(dataset)
    rows["arrival_profile"] = config.arrival_profile
    rows["spatial"] = config.spatial
    rows["segment_count"] = config.segment_count
    rows["travel_noise"] = travel_noise
    rows["service_noise"] = service_noise
    rows["congestion"] = congestion
    return rows


def load_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[ExperimentalAttentionLearner, StudyConfig, str, int]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = StudyConfig(**payload["config"])
    model = build_model(payload["selector"], config, device)
    model.load_state_dict(payload["model"])
    return model, config, payload["selector"], int(payload["seed"])


def save_config(config: StudyConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
