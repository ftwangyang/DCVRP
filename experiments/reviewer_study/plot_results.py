"""Generate publication figures from reviewer-study source CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .statistics import seed_level_summary


PALETTE = {
    "shared": "#484878",
    "shared_wide": "#7884B4",
    "shared_id": "#B4C0E4",
    "centralized": "#A8A8A8",
    "attention_aggregation": "#5D9E73",
    "independent_narrow": "#E4CCD8",
    "independent": "#B64342",
    "joint_pair": "#42949E",
}

DISPLAY_NAMES = {
    "shared": "Shared",
    "shared_wide": "Shared-wide",
    "shared_id": "Shared+ID",
    "centralized": "Centralized",
    "attention_aggregation": "Attention agg.",
    "independent_narrow": "Indep.-narrow",
    "independent": "DVNDA",
    "joint_pair": "Joint pair",
}

CLASSICAL_NAMES = {
    "nearest": "Nearest",
    "regret": "Regret",
    "local_search": "Local search",
    "tabu": "Tabu",
    "alns": "ALNS",
    "ortools": "OR-Tools",
    "DVNDA": "DVNDA",
}


def apply_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False


def panel_label(axis, label):
    axis.text(
        -0.16, 1.05, label, transform=axis.transAxes,
        fontsize=9, fontweight="bold", ha="left", va="bottom",
    )


def save_figure(figure, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _seed_means(data, groups, metric):
    return data.groupby(groups + ["train_seed"], as_index=False)[metric].mean()


def architecture_figure(results_root: Path, figure_root: Path):
    raw = pd.read_csv(results_root / "architecture" / "raw" / "architecture_instances.csv")
    resources = pd.read_csv(results_root / "architecture" / "raw" / "architecture_resources.csv")
    order = [name for name in PALETTE if name in set(raw["selector"])]
    colors = [PALETTE[name] for name in order]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    for axis, metric, ylabel in [
        (axes[0, 0], "cost_with_penalty", "Penalized cost"),
        (axes[0, 1], "qos", "Customers completed by horizon"),
    ]:
        seed = _seed_means(raw, ["selector"], metric)
        values = [seed.loc[seed.selector == name, metric].to_numpy() for name in order]
        means = np.array([value.mean() for value in values])
        errors = np.array([
            1.96 * value.std(ddof=1) / np.sqrt(len(value)) if len(value) > 1 else 0
            for value in values
        ])
        axis.bar(np.arange(len(order)), means, yerr=errors, color=colors,
                 edgecolor="#272727", linewidth=0.6, capsize=2)
        axis.set_xticks(np.arange(len(order)))
        axis.set_xticklabels(
            [DISPLAY_NAMES.get(name, name) for name in order],
            rotation=28, ha="right",
        )
        axis.set_ylabel(ylabel)
        if metric == "qos":
            axis.set_ylim(0, 1.02)

    seed_response = _seed_means(raw, ["selector"], "response_time_min")
    response_values = [
        seed_response.loc[seed_response.selector == name, "response_time_min"].to_numpy()
        for name in order
    ]
    axes[1, 0].boxplot(
                       response_values,
                       labels=[DISPLAY_NAMES.get(name, name) for name in order],
                       showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "#E0E0F0", "edgecolor": "#484878"},
                       medianprops={"color": "#B64342"})
    axes[1, 0].tick_params(axis="x", rotation=28)
    axes[1, 0].set_ylabel("Response time (min)")

    resource_mean = resources.groupby("selector", as_index=False).agg(
        params=("selector_parameter_count", "mean"),
        train_s=("training_seconds", "mean"),
    )
    for name, color in zip(order, colors):
        row = resource_mean[resource_mean.selector == name]
        axes[1, 1].scatter(row.params, row.train_s, s=35, color=color,
                           edgecolor="#272727", linewidth=0.5,
                           label=DISPLAY_NAMES.get(name, name))
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Selector parameters")
    axes[1, 1].set_ylabel("Training time (s)")
    axes[1, 1].legend(fontsize=6, ncol=2)
    for label, axis in zip("abcd", axes.flat):
        panel_label(axis, label)
    fig.suptitle("Controlled vehicle-selector ablations", fontsize=9, y=1.01)
    fig.tight_layout(pad=1.2)
    save_figure(fig, figure_root / "figure_architecture_ablation")


def robustness_figure(results_root: Path, figure_root: Path):
    raw = pd.read_csv(results_root / "robustness" / "raw" / "robustness_instances.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))

    interval = raw[raw.scenario.str.startswith("interval_")].copy()
    interval["segments"] = interval.scenario.str.replace("interval_", "", regex=False).astype(int)
    interval_seed = interval.groupby(["segments", "train_seed"], as_index=False).agg(
        cost=("cost_with_penalty", "mean"), idle=("idle_boundary_wait_min", "mean")
    )
    for metric, axis, color, ylabel in [
        ("cost", axes[0, 0], "#0F4D92", "Penalized cost"),
        ("idle", axes[0, 1], "#B64342", "Fleet boundary wait (min)"),
    ]:
        for seed, frame in interval_seed.groupby("train_seed"):
            axis.plot(frame.segments, frame[metric], color=color, alpha=0.25, lw=0.8)
        mean = interval_seed.groupby("segments")[metric].mean()
        axis.plot(mean.index, mean.values, color=color, marker="o", lw=1.8)
        axis.set_xlabel("Number of planning intervals")
        axis.set_ylabel(ylabel)

    schedule_order = ["schedule_fixed", "schedule_event", "schedule_hybrid"]
    schedule = raw[raw.scenario.isin(schedule_order)]
    schedule_seed = schedule.groupby(["scenario", "train_seed"], as_index=False).agg(
        cost=("cost_with_penalty", "mean"), idle=("idle_boundary_wait_min", "mean")
    )
    schedule_estimates = [
        _mean_ci(schedule_seed.loc[schedule_seed.scenario == value, "cost"])
        for value in schedule_order
    ]
    axes[0, 2].bar(
        np.arange(3), [value[0] for value in schedule_estimates],
        yerr=[value[1] for value in schedule_estimates], color="#B4C0E4",
        edgecolor="#484878", linewidth=0.6, capsize=2,
    )
    axes[0, 2].set_xticks(np.arange(3))
    axes[0, 2].set_xticklabels(["Fixed", "Event", "Hybrid"], rotation=20)
    axes[0, 2].set_ylabel("Penalized cost")

    arrival_order = ["early", "uniform", "late", "two_peak", "clustered", "nhpp"]
    arrival = raw[raw.scenario.str.startswith("arrival_")].copy()
    arrival["profile"] = arrival.scenario.str.replace("arrival_", "", regex=False)
    arrival_seed = arrival.groupby(["profile", "train_seed"], as_index=False)["cost_with_penalty"].mean()
    means = [arrival_seed.loc[arrival_seed.profile == p, "cost_with_penalty"].mean() for p in arrival_order]
    errors = [
        arrival_seed.loc[arrival_seed.profile == p, "cost_with_penalty"].std(ddof=1)
        for p in arrival_order
    ]
    axes[1, 0].bar(np.arange(len(arrival_order)), means, yerr=errors,
                   color="#B4C0E4", edgecolor="#484878", linewidth=0.6, capsize=2)
    axes[1, 0].set_xticks(np.arange(len(arrival_order)))
    axes[1, 0].set_xticklabels(arrival_order, rotation=28, ha="right")
    axes[1, 0].set_ylabel("Penalized cost")

    dynamic = raw[raw.scenario.str.startswith("dynamic_")].copy()
    dynamic["ratio"] = dynamic.scenario.str.replace("dynamic_", "", regex=False).astype(float)
    dynamic_seed = dynamic.groupby(["ratio", "train_seed"], as_index=False).agg(
        cost=("cost_with_penalty", "mean"), qos=("qos", "mean")
    )
    cost_mean = dynamic_seed.groupby("ratio").cost.mean()
    axes[1, 1].plot(cost_mean.index, cost_mean.values, marker="o", color="#B64342")
    axes[1, 1].set_xlabel("Dynamic customer ratio")
    axes[1, 1].set_ylabel("Penalized cost")
    qos_axis = axes[1, 1].twinx()
    qos_mean = dynamic_seed.groupby("ratio").qos.mean()
    qos_axis.plot(qos_mean.index, qos_mean.values, marker="s", color="#0F4D92")
    qos_axis.set_ylabel("Completed QoS", color="#0F4D92")
    qos_axis.set_ylim(0, 1.02)
    qos_axis.spines["top"].set_visible(False)

    uncertainty = raw[raw.scenario.str.startswith("uncertainty_")].copy()
    uncertainty["condition"] = uncertainty.scenario.str.replace("uncertainty_", "", regex=False)
    uncertainty_seed = uncertainty.groupby(["condition", "train_seed"], as_index=False).agg(
        cost=("cost_with_penalty", "mean"), response=("response_time_min", "mean")
    )
    for condition, frame in uncertainty_seed.groupby("condition"):
        axes[1, 2].scatter(frame.cost.mean(), frame.response.mean(), s=30, label=condition)
    axes[1, 2].set_xlabel("Penalized cost")
    axes[1, 2].set_ylabel("Response time (min)")
    axes[1, 2].legend(fontsize=5.2)
    for label, axis in zip("abcdef", axes.flat):
        panel_label(axis, label)
    fig.suptitle("Dynamic scheduling and operational robustness", fontsize=9, y=1.01)
    fig.tight_layout(pad=1.2)
    save_figure(fig, figure_root / "figure_dynamic_robustness")


def scaling_figure(results_root: Path, figure_root: Path):
    raw = pd.read_csv(results_root / "scaling" / "raw" / "selector_scaling.csv")
    summary = raw.groupby(["selector", "vehicle_count"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colors = {"shared": "#7884B4", "centralized": "#A8A8A8", "independent": "#B64342"}
    for selector, frame in summary.groupby("selector"):
        axes[0].plot(frame.vehicle_count, frame.parameter_count, marker="o", lw=1.6,
                     color=colors.get(selector), label=selector)
        axes[1].plot(frame.vehicle_count, frame.latency_mean_ms, marker="o", lw=1.6,
                     color=colors.get(selector), label=selector)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Fleet size")
    axes[0].set_ylabel("Selector parameters")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Fleet size")
    axes[1].set_ylabel("Decision latency (ms)")
    axes[1].legend(fontsize=6)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout(pad=1.2)
    save_figure(fig, figure_root / "figure_scalability")


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = values.mean() if len(values) else np.nan
    if len(values) < 2:
        return mean, 0.0
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, half


def classical_figure(results_root: Path, figure_root: Path):
    raw = pd.read_csv(results_root / "classical" / "raw" / "classical_instances.csv")
    production_path = results_root / "production" / "raw" / "production_instances.csv"
    production = pd.read_csv(production_path) if production_path.exists() else None
    optimality = pd.read_csv(
        results_root / "optimality" / "raw" / "optimality_gap_instances.csv"
    )
    order = ["nearest", "regret", "local_search", "tabu", "alns", "ortools"]
    order = [method for method in order if method in set(raw.method)]
    display_order = order + (["DVNDA"] if production is not None else [])
    colors = list(plt.cm.Blues(np.linspace(0.38, 0.88, len(order))))
    if production is not None:
        colors.append(PALETTE["independent"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for axis, metric, ylabel in [
        (axes[0, 0], "cost_with_penalty", "Penalized cost"),
        (axes[0, 1], "qos", "Customers completed by horizon"),
        (axes[1, 0], "planning_time_ms", "Planning time per instance (ms)"),
    ]:
        estimates = [_mean_ci(raw.loc[raw.method == method, metric]) for method in order]
        if production is not None:
            production_metric = (
                "wall_time_ms_per_instance"
                if metric == "planning_time_ms" else metric
            )
            seed_values = production.groupby("train_seed")[production_metric].mean()
            estimates.append(_mean_ci(seed_values))
        axis.bar(
            np.arange(len(display_order)),
            [value[0] for value in estimates],
            yerr=[value[1] for value in estimates],
            color=colors,
            edgecolor="#272727",
            linewidth=0.6,
            capsize=2,
        )
        axis.set_xticks(np.arange(len(display_order)))
        axis.set_xticklabels(
            [CLASSICAL_NAMES.get(name, name) for name in display_order],
            rotation=28, ha="right",
        )
        axis.set_ylabel(ylabel)
        if metric == "qos":
            axis.set_ylim(0, 1.02)
        if metric == "planning_time_ms":
            axis.set_yscale("log")

    gap_order = [method for method in order if method in set(optimality.method)]
    estimates = [
        _mean_ci(optimality.loc[optimality.method == method, "optimality_gap_percent"])
        for method in gap_order
    ]
    axes[1, 1].bar(
        np.arange(len(gap_order)),
        [value[0] for value in estimates],
        yerr=[value[1] for value in estimates],
        color="#B4C0E4",
        edgecolor="#484878",
        linewidth=0.6,
        capsize=2,
    )
    axes[1, 1].set_xticks(np.arange(len(gap_order)))
    axes[1, 1].set_xticklabels(
        [CLASSICAL_NAMES.get(name, name) for name in gap_order],
        rotation=28, ha="right",
    )
    axes[1, 1].set_ylabel("Static n=8 optimality gap (%)")
    for label, axis in zip("abcd", axes.flat):
        panel_label(axis, label)
    fig.suptitle("Rolling-horizon classical baselines and exact references", fontsize=9, y=1.01)
    fig.tight_layout(pad=1.2)
    save_figure(fig, figure_root / "figure_classical_baselines")


def generalization_figure(results_root: Path, figure_root: Path):
    aggregation = pd.read_csv(
        results_root / "aggregation_ordering" / "raw"
        / "aggregation_ordering_instances.csv"
    )
    roads = pd.read_csv(results_root / "real_roads" / "raw" / "real_road_instances.csv")
    transfer_path = results_root / "transfer" / "raw" / "transfer_instances.csv"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))

    scenario_order = [
        "aggregation_argmax", "aggregation_softmax", "aggregation_random",
        "vehicle_id_permuted",
    ]
    scenario_order = [s for s in scenario_order if s in set(aggregation.scenario)]
    seed = aggregation.groupby(["scenario", "train_seed"], as_index=False).agg(
        cost=("cost_with_penalty", "mean"), qos=("qos", "mean")
    )
    labels = [
        "Argmax", "Softmax", "Random vehicle", "ID permutation"
    ][:len(scenario_order)]
    for axis, metric, ylabel in [
        (axes[0, 0], "cost", "Penalized cost"),
        (axes[0, 1], "qos", "Customers completed by horizon"),
    ]:
        estimates = [_mean_ci(seed.loc[seed.scenario == s, metric]) for s in scenario_order]
        axis.bar(
            np.arange(len(scenario_order)), [v[0] for v in estimates],
            yerr=[v[1] for v in estimates], color="#B4C0E4",
            edgecolor="#484878", linewidth=0.6, capsize=2,
        )
        axis.set_xticks(np.arange(len(scenario_order)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        if metric == "qos":
            axis.set_ylim(0, 1.02)

    road_seed = roads.groupby(["city", "condition", "train_seed"], as_index=False).agg(
        cost=("cost_with_penalty", "mean"), qos=("qos", "mean")
    )
    markers = {"normal": "o", "peak_congestion": "^"}
    for (city, condition), frame in road_seed.groupby(["city", "condition"]):
        axes[1, 0].scatter(
            frame.cost.mean(), frame.qos.mean(), s=42,
            marker=markers.get(condition, "o"), label=f"{city}, {condition}",
            edgecolor="#272727", linewidth=0.5,
        )
    axes[1, 0].set_xlabel("Penalized cost")
    axes[1, 0].set_ylabel("Customers completed by horizon")
    axes[1, 0].legend(fontsize=6)

    if transfer_path.exists():
        transfer = pd.read_csv(transfer_path)
        transfer_seed = transfer.groupby(
            ["selector", "target_vehicle_count", "stage", "train_seed"],
            as_index=False,
        ).qos.mean()
        styles = {"zero_shot": "--", "fine_tuned": "-"}
        for (selector, stage), frame in transfer_seed.groupby(["selector", "stage"]):
            summary = frame.groupby("target_vehicle_count").qos.mean()
            axes[1, 1].plot(
                summary.index, summary.values, marker="o",
                linestyle=styles.get(stage, "-"),
                color=PALETTE.get(selector, "#484878"),
                label=f"{selector}, {stage.replace('_', ' ')}",
            )
        axes[1, 1].set_xlabel("Target fleet size")
        axes[1, 1].set_ylabel("Customers completed by horizon")
        axes[1, 1].set_ylim(0, 1.02)
        axes[1, 1].legend(fontsize=5.5)
    else:
        axes[1, 1].axis("off")

    for label, axis in zip("abcd", axes.flat):
        panel_label(axis, label)
    fig.suptitle("Aggregation, ordering, road-network, and fleet transfer tests", fontsize=9, y=1.01)
    fig.tight_layout(pad=1.2)
    save_figure(fig, figure_root / "figure_generalization")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--figure-root", type=Path, default=Path("experiments/figures"))
    return parser.parse_args()


def main():
    args = parse_args()
    apply_style()
    if (args.results_root / "architecture" / "raw" / "architecture_instances.csv").exists():
        architecture_figure(args.results_root, args.figure_root)
    if (args.results_root / "robustness" / "raw" / "robustness_instances.csv").exists():
        robustness_figure(args.results_root, args.figure_root)
    if (args.results_root / "scaling" / "raw" / "selector_scaling.csv").exists():
        scaling_figure(args.results_root, args.figure_root)
    if (
        (args.results_root / "classical" / "raw" / "classical_instances.csv").exists()
        and (args.results_root / "optimality" / "raw" / "optimality_gap_instances.csv").exists()
    ):
        classical_figure(args.results_root, args.figure_root)
    if (
        (args.results_root / "aggregation_ordering" / "raw" / "aggregation_ordering_instances.csv").exists()
        and (args.results_root / "real_roads" / "raw" / "real_road_instances.csv").exists()
    ):
        generalization_figure(args.results_root, args.figure_root)


if __name__ == "__main__":
    main()
