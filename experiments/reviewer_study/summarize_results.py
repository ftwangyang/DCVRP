"""Create reviewer-facing CSV and LaTeX tables from raw source data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from .statistics import (
    _holm_adjust,
    format_mean_ci,
    instance_level_summary,
    paired_architecture_comparisons,
    seed_level_summary,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    return parser.parse_args()


def _format_metric_row(row: pd.Series) -> str:
    """Format physically bounded metrics without impossible CI endpoints."""
    low = max(0.0, float(row["ci95_low"]))
    high = float(row["ci95_high"])
    if row["metric"] == "qos":
        high = min(1.0, high)
    return format_mean_ci(
        row["mean"], low, high,
        digits=3 if row["metric"] == "qos" else 2,
    )


def _wide_table(summary: pd.DataFrame, index: str, metrics: list[str]) -> pd.DataFrame:
    frame = summary[summary["metric"].isin(metrics)].copy()
    frame["display"] = frame.apply(_format_metric_row, axis=1)
    return frame.pivot(index=index, columns="metric", values="display").reset_index()


def _write_table(frame: pd.DataFrame, summary_root: Path, name: str) -> None:
    frame.to_csv(summary_root / f"table_{name}.csv", index=False)
    (summary_root / f"table_{name}.tex").write_text(
        _to_latex(frame), encoding="utf-8"
    )


def _latex_value(value) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, float):
        text = f"{value:.3f}"
    else:
        text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _to_latex(frame: pd.DataFrame) -> str:
    columns = "l" + "r" * max(len(frame.columns) - 1, 0)
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(_latex_value(column) for column in frame.columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_value(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main():
    args = parse_args()
    summary_root = args.results_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    architecture_file = args.results_root / "architecture" / "raw" / "architecture_instances.csv"
    if architecture_file.exists():
        architecture = pd.read_csv(architecture_file)
        summary = seed_level_summary(architecture, ["selector"])
        summary.to_csv(summary_root / "architecture_summary_long.csv", index=False)
        table = _wide_table(
            summary,
            "selector",
            ["cost_with_penalty", "distance", "qos", "response_time_min"],
        )
        _write_table(table, summary_root, "architecture")
        comparisons = paired_architecture_comparisons(architecture)
        comparisons.to_csv(summary_root / "architecture_paired_tests.csv", index=False)

    robustness_file = args.results_root / "robustness" / "raw" / "robustness_instances.csv"
    if robustness_file.exists():
        robustness = pd.read_csv(robustness_file)
        summary = seed_level_summary(robustness, ["scenario"])
        summary.to_csv(summary_root / "robustness_summary_long.csv", index=False)
        table = _wide_table(
            summary,
            "scenario",
            ["cost_with_penalty", "distance", "qos", "response_time_min"],
        )
        _write_table(table, summary_root, "robustness")

    scaling_file = args.results_root / "scaling" / "raw" / "selector_scaling.csv"
    if scaling_file.exists():
        scaling = pd.read_csv(scaling_file)
        grouped = scaling.groupby(["selector", "vehicle_count"], as_index=False).agg(
            parameter_count=("parameter_count", "mean"),
            latency_mean_ms=("latency_mean_ms", "mean"),
            latency_sd_ms=("latency_mean_ms", "std"),
            peak_memory_mb=("peak_memory_mb", "mean"),
        )
        _write_table(grouped, summary_root, "scaling")

    aggregation_file = (
        args.results_root / "aggregation_ordering" / "raw"
        / "aggregation_ordering_instances.csv"
    )
    if aggregation_file.exists():
        aggregation = pd.read_csv(aggregation_file)
        summary = seed_level_summary(aggregation, ["scenario"])
        summary.to_csv(summary_root / "aggregation_summary_long.csv", index=False)
        _write_table(
            _wide_table(
                summary,
                "scenario",
                ["cost_with_penalty", "distance", "qos", "response_time_min"],
            ),
            summary_root,
            "aggregation_ordering",
        )
        per_instance = aggregation.groupby(
            ["scenario", "train_seed", "instance"], as_index=False
        ).cost_with_penalty.mean()
        pivot = per_instance.pivot_table(
            index=["train_seed", "instance"], columns="scenario",
            values="cost_with_penalty",
        )
        paired_rows = []
        reference = "aggregation_argmax"
        for scenario in pivot.columns:
            if scenario == reference:
                continue
            difference = (pivot[scenario] - pivot[reference]).dropna()
            seed_difference = difference.groupby(level="train_seed").mean()
            test = stats.ttest_1samp(seed_difference, 0.0) if len(seed_difference) > 1 else None
            paired_rows.append({
                "reference": reference,
                "scenario": scenario,
                "seed_count": len(seed_difference),
                "mean_cost_difference": float(seed_difference.mean()),
                "paired_t_p": float(test.pvalue) if test is not None else float("nan"),
            })
        paired_frame = pd.DataFrame(paired_rows)
        if len(paired_frame):
            paired_frame["holm_p"] = _holm_adjust(
                paired_frame["paired_t_p"].to_numpy(float)
            )
        paired_frame.to_csv(
            summary_root / "aggregation_paired_tests.csv", index=False
        )

    road_file = args.results_root / "real_roads" / "raw" / "real_road_instances.csv"
    if road_file.exists():
        roads = pd.read_csv(road_file)
        summary = seed_level_summary(roads, ["city", "condition"])
        summary.to_csv(summary_root / "real_roads_summary_long.csv", index=False)
        _write_table(
            summary[summary["metric"].isin([
                "cost_with_penalty", "distance", "qos", "response_time_min"
            ])].assign(
                display=lambda frame: frame.apply(_format_metric_row, axis=1)
            ).pivot(
                index=["city", "condition"], columns="metric", values="display"
            ).reset_index(),
            summary_root,
            "real_roads",
        )

    transfer_file = args.results_root / "transfer" / "raw" / "transfer_instances.csv"
    if transfer_file.exists():
        transfer = pd.read_csv(transfer_file)
        summary = seed_level_summary(
            transfer, ["selector", "target_vehicle_count", "stage"]
        )
        summary.to_csv(summary_root / "transfer_summary_long.csv", index=False)
        frame = summary[summary["metric"].isin([
            "cost_with_penalty", "qos", "wall_time_ms_per_instance"
        ])].copy()
        frame["display"] = frame.apply(_format_metric_row, axis=1)
        table = frame.pivot(
            index=["selector", "target_vehicle_count", "stage"],
            columns="metric", values="display",
        ).reset_index()
        _write_table(table, summary_root, "fleet_transfer")

    classical_file = args.results_root / "classical" / "raw" / "classical_instances.csv"
    if classical_file.exists():
        classical = pd.read_csv(classical_file)
        metrics = [
            "cost_with_penalty", "distance", "qos", "response_time_min",
            "planning_time_ms", "vehicle_utilization", "unserved",
        ]
        summary = instance_level_summary(classical, ["method"], metrics)
        summary.to_csv(summary_root / "classical_summary_long.csv", index=False)
        _write_table(
            _wide_table(
                summary,
                "method",
                ["cost_with_penalty", "distance", "qos", "planning_time_ms"],
            ),
            summary_root,
            "classical_baselines",
        )

        production_file = (
            args.results_root / "production" / "raw" / "production_instances.csv"
        )
        if production_file.exists():
            production = pd.read_csv(production_file)
            production_summary = seed_level_summary(
                production,
                ["selector"],
                [
                    "cost_with_penalty", "distance", "qos",
                    "response_time_min", "wall_time_ms_per_instance",
                ],
            )
            classical_summary = instance_level_summary(
                classical,
                ["method"],
                [
                    "cost_with_penalty", "distance", "qos",
                    "response_time_min", "planning_time_ms",
                ],
            )
            classical_summary["method"] = classical_summary["method"].astype(str)
            production_summary = production_summary.copy()
            production_summary["method"] = "DVNDA"
            production_summary["metric"] = production_summary["metric"].replace(
                {"wall_time_ms_per_instance": "planning_time_ms"}
            )
            combined = pd.concat(
                [
                    classical_summary[[
                        "method", "metric", "mean", "ci95_low", "ci95_high"
                    ]],
                    production_summary[[
                        "method", "metric", "mean", "ci95_low", "ci95_high"
                    ]],
                ],
                ignore_index=True,
            )
            _write_table(
                _wide_table(
                    combined,
                    "method",
                    [
                        "cost_with_penalty", "distance", "qos",
                        "response_time_min", "planning_time_ms",
                    ],
                ),
                summary_root,
                "neural_vs_classical",
            )

    optimality_file = (
        args.results_root / "optimality" / "raw" / "optimality_gap_instances.csv"
    )
    if optimality_file.exists():
        optimality = pd.read_csv(optimality_file)
        summary = instance_level_summary(
            optimality, ["method"], ["optimality_gap_percent", "method_seconds"]
        )
        summary.to_csv(summary_root / "optimality_summary_long.csv", index=False)
        _write_table(
            _wide_table(
                summary, "method", ["optimality_gap_percent", "method_seconds"]
            ),
            summary_root,
            "optimality_gap",
        )


if __name__ == "__main__":
    main()
