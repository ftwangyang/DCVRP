"""Hierarchical summaries and paired statistical comparisons."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = [
    "cost_with_penalty",
    "distance",
    "qos",
    "response_time_min",
    "completion_time_min",
    "mean_completion_time_min",
    "unserved",
    "committed_not_completed",
    "late",
    "vehicle_utilization",
    "route_balance_cv",
    "idle_boundary_wait_min",
    "replanning_events",
    "decision_epochs",
    "wall_time_ms_per_instance",
]


def seed_level_summary(
    data: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    metrics = metrics or [metric for metric in PRIMARY_METRICS if metric in data]
    seed_columns = group_columns + ["train_seed"]
    seed_means = data.groupby(seed_columns, dropna=False)[metrics].mean().reset_index()
    rows = []
    for keys, frame in seed_means.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metrics:
            values = frame[metric].dropna().to_numpy(float)
            count = len(values)
            mean = float(values.mean()) if count else math.nan
            standard_deviation = float(values.std(ddof=1)) if count > 1 else math.nan
            if count > 1:
                half_width = float(
                    stats.t.ppf(0.975, count - 1)
                    * standard_deviation
                    / math.sqrt(count)
                )
            else:
                half_width = math.nan
            rows.append({
                **base,
                "metric": metric,
                "seed_count": count,
                "mean": mean,
                "sd_between_seeds": standard_deviation,
                "ci95_low": mean - half_width if count > 1 else math.nan,
                "ci95_high": mean + half_width if count > 1 else math.nan,
            })
    return pd.DataFrame(rows)


def instance_level_summary(
    data: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """Mean and t-based CI when independent test instances are the replicates."""
    rows = []
    for keys, frame in data.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metrics:
            values = frame[metric].dropna().to_numpy(float)
            count = len(values)
            mean = float(values.mean()) if count else math.nan
            sd = float(values.std(ddof=1)) if count > 1 else math.nan
            half_width = (
                float(stats.t.ppf(0.975, count - 1) * sd / math.sqrt(count))
                if count > 1 else math.nan
            )
            rows.append({
                **base,
                "metric": metric,
                "instance_count": count,
                "mean": mean,
                "sd_between_instances": sd,
                "ci95_low": mean - half_width if count > 1 else math.nan,
                "ci95_high": mean + half_width if count > 1 else math.nan,
            })
    return pd.DataFrame(rows)


def paired_architecture_comparisons(
    data: pd.DataFrame,
    reference: str = "shared",
    metric: str = "cost_with_penalty",
) -> pd.DataFrame:
    paired = data.pivot_table(
        index=["train_seed", "instance"],
        columns="selector",
        values=metric,
    )
    if reference not in paired:
        raise ValueError(f"Reference selector {reference!r} is unavailable")
    rows = []
    for selector in paired.columns:
        if selector == reference:
            continue
        frame = paired[[reference, selector]].dropna()
        difference = frame[selector] - frame[reference]
        seed_difference = difference.groupby(level="train_seed").mean()
        if len(seed_difference) > 1:
            test = stats.ttest_1samp(seed_difference, 0.0)
            p_value = float(test.pvalue)
            sd = float(seed_difference.std(ddof=1))
            effect_size = float(seed_difference.mean() / sd) if sd > 0 else math.nan
        else:
            p_value = math.nan
            effect_size = math.nan
        rows.append({
            "reference": reference,
            "selector": selector,
            "metric": metric,
            "paired_instances": len(frame),
            "seed_count": len(seed_difference),
            "mean_difference": float(seed_difference.mean()),
            "relative_difference_percent": float(
                100.0 * seed_difference.mean() / frame[reference].mean()
            ),
            "paired_t_p": p_value,
            "standardized_seed_effect": effect_size,
        })
    result = pd.DataFrame(rows)
    if len(result):
        result["holm_p"] = _holm_adjust(result["paired_t_p"].to_numpy(float))
    return result


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p_values[valid])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def format_mean_ci(mean: float, low: float, high: float, digits: int = 2) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(low) or not np.isfinite(high):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
