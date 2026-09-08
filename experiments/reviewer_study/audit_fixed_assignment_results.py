"""Audit the fixed-assignment classical-baseline delivery.

The audit is intentionally independent of the experiment runner: it rebuilds
all acceptance quantities from ``raw_instances.csv`` and exits non-zero if a
predeclared gate fails.  This prevents a manuscript table from being accepted
only because it looks plausible after aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = (
    "Greedy (same instances)",
    "Regret insertion",
    "Tabu Search",
    "Adaptive LNS",
    "OR-Tools",
)
SIZES = (20, 35, 50)
RATES = (0.10, 0.25, 0.50, 0.75)
PAPER_GREEDY_COST = {
    (20, 0.10): 9.07,
    (20, 0.25): 9.69,
    (20, 0.50): 11.25,
    (20, 0.75): 12.43,
    (35, 0.10): 15.95,
    (35, 0.25): 16.96,
    (35, 0.50): 19.63,
    (35, 0.75): 21.55,
    (50, 0.10): 21.41,
    (50, 0.25): 22.84,
    (50, 0.50): 26.71,
    (50, 0.75): 30.19,
}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def audit(result_dir: Path) -> tuple[dict, pd.DataFrame]:
    raw_path = result_dir / "raw_instances.csv"
    config_path = result_dir / "config.json"
    summary_path = result_dir / "summary.csv"
    if not raw_path.exists() or not config_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "result directory must contain raw_instances.csv, summary.csv, and config.json"
        )

    raw = pd.read_csv(raw_path)
    supplied_summary = pd.read_csv(summary_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    group_keys = ["method", "customer_count", "dynamic_rate"]
    rebuilt = (
        raw.groupby(group_keys, as_index=False)
        .agg(
            instances=("instance_id", "size"),
            cost_mean=("cost", "mean"),
            cost_sd=("cost", "std"),
            qos_mean_percent=("qos_percent", "mean"),
            qos_sd_percent=("qos_percent", "std"),
            time_100_s=("solve_time_s", "sum"),
            vehicles_used_min=("vehicles_used", "min"),
            vehicles_used_max=("vehicles_used", "max"),
        )
        .sort_values(group_keys)
        .reset_index(drop=True)
    )

    failures: list[str] = []
    expected_rows = len(METHODS) * len(SIZES) * len(RATES) * 100
    if len(raw) != expected_rows:
        failures.append(f"raw row count is {len(raw)}, expected {expected_rows}")
    duplicate_count = int(raw.duplicated(group_keys + ["instance_id"]).sum())
    if duplicate_count:
        failures.append(f"found {duplicate_count} duplicate method/size/rate/instance rows")
    if set(raw["method"]) != set(METHODS):
        failures.append("method set differs from the five predeclared methods")
    if len(rebuilt) != len(METHODS) * len(SIZES) * len(RATES):
        failures.append(f"group count is {len(rebuilt)}, expected 60")
    if not (rebuilt["instances"] == 100).all():
        failures.append("at least one group does not contain exactly 100 instances")
    if config.get("assignment_policy") != "fixed-greedy":
        failures.append("assignment_policy is not fixed-greedy")
    if config.get("short_route_fallback", "missing") is not None:
        failures.append("short-route fallback is enabled; every route must use its named solver")
    if config.get("environment") != "time_driven_executed_path_multivehicle_v9":
        failures.append("unexpected environment tag")
    numeric_raw = raw[["cost", "qos_percent", "solve_time_s", "vehicles_used"]]
    if not np.isfinite(numeric_raw.to_numpy(dtype=float)).all():
        failures.append("raw results contain non-finite numeric values")

    summary_compare_columns = [
        "instances",
        "cost_mean",
        "cost_sd",
        "qos_mean_percent",
        "qos_sd_percent",
        "time_100_s",
    ]
    supplied = supplied_summary[group_keys + summary_compare_columns].sort_values(
        group_keys
    ).reset_index(drop=True)
    recomputed = rebuilt[group_keys + summary_compare_columns]
    summary_matches_raw = bool(
        supplied[group_keys].equals(recomputed[group_keys])
        and np.allclose(
            supplied[summary_compare_columns].to_numpy(dtype=float),
            recomputed[summary_compare_columns].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    if not summary_matches_raw:
        failures.append("summary.csv does not exactly match aggregation rebuilt from raw rows")

    fleet_ok = bool(
        (rebuilt["vehicles_used_min"] >= 1).all()
        and (rebuilt["vehicles_used_max"] <= rebuilt["customer_count"] / 5).all()
        and (raw["vehicle_count"] == raw["customer_count"] / 5).all()
    )
    if not fleet_ok:
        failures.append("nominal fleet is not m=n/5 or active vehicle count is out of range")
    minimum_group_qos = float(rebuilt["qos_mean_percent"].min())
    if minimum_group_qos < 99.5:
        failures.append(f"minimum group QoS is {minimum_group_qos:.3f}%, below 99.5%")

    monotonic_rows: list[dict] = []
    for method in METHODS:
        for size in SIZES:
            subset = rebuilt[
                (rebuilt["method"] == method)
                & (rebuilt["customer_count"] == size)
            ].sort_values("dynamic_rate")
            costs = subset["cost_mean"].to_numpy()
            passed = bool(len(costs) == 4 and np.all(np.diff(costs) > 0.0))
            monotonic_rows.append({"method": method, "n": size, "passed": passed})
            if not passed:
                failures.append(f"cost is not strictly increasing for {method}, n={size}")

    greedy = rebuilt[rebuilt["method"] == "Greedy (same instances)"]
    greedy_lookup = {
        (int(row.customer_count), float(row.dynamic_rate)): float(row.cost_mean)
        for row in greedy.itertuples()
    }
    paper_comparison: list[dict] = []
    for row in rebuilt.itertuples():
        key = (int(row.customer_count), float(row.dynamic_rate))
        paper = PAPER_GREEDY_COST[key]
        paper_comparison.append(
            {
                "method": row.method,
                "n": key[0],
                "dynamic_rate": int(round(100 * key[1])),
                "cost": float(row.cost_mean),
                "paper_greedy": paper,
                "absolute_percent_error": 100.0 * abs(float(row.cost_mean) - paper) / paper,
                "same_instance_greedy_gap_percent": 100.0
                * (float(row.cost_mean) - greedy_lookup[key])
                / greedy_lookup[key],
            }
        )
    comparison = pd.DataFrame(paper_comparison)

    same_greedy = comparison[comparison["method"] == "Greedy (same instances)"]
    greedy_mape = float(same_greedy["absolute_percent_error"].mean())
    greedy_max_ape = float(same_greedy["absolute_percent_error"].max())
    if greedy_mape > 3.0:
        failures.append(f"same-instance Greedy MAPE is {greedy_mape:.3f}%, above 3%")
    if greedy_max_ape >= 5.0:
        failures.append(
            f"same-instance Greedy maximum APE is {greedy_max_ape:.3f}%, not below 5%"
        )

    high_rate = comparison[
        (comparison["dynamic_rate"] == 75)
        & (comparison["method"] != "Greedy (same instances)")
    ]
    maximum_high_rate_gap = float(high_rate["same_instance_greedy_gap_percent"].abs().max())
    if maximum_high_rate_gap >= 5.0:
        failures.append(
            f"maximum 75% rate gap from same-instance Greedy is "
            f"{maximum_high_rate_gap:.3f}%, not below 5%"
        )
    maximum_high_rate_paper_ape = float(high_rate["absolute_percent_error"].max())
    if maximum_high_rate_paper_ape >= 5.0:
        failures.append(
            f"maximum 75% rate APE from paper Greedy is "
            f"{maximum_high_rate_paper_ape:.3f}%, not below 5%"
        )

    method_error = comparison.groupby("method", as_index=False).agg(
        mape_percent=("absolute_percent_error", "mean"),
        max_ape_percent=("absolute_percent_error", "max"),
    )
    report = {
        "passed": not failures,
        "failures": failures,
        "raw_rows": int(len(raw)),
        "expected_raw_rows": expected_rows,
        "duplicate_rows": duplicate_count,
        "summary_matches_raw": summary_matches_raw,
        "groups": int(len(rebuilt)),
        "instances_per_group": sorted(int(value) for value in rebuilt["instances"].unique()),
        "assignment_policy": config.get("assignment_policy"),
        "short_route_fallback": config.get("short_route_fallback", "missing"),
        "environment": config.get("environment"),
        "fleet_definition_m_equals_n_over_5": fleet_ok,
        "minimum_group_qos_percent": minimum_group_qos,
        "all_cost_sequences_strictly_increasing": all(
            row["passed"] for row in monotonic_rows
        ),
        "same_instance_greedy_mape_vs_paper_percent": greedy_mape,
        "same_instance_greedy_max_ape_vs_paper_percent": greedy_max_ape,
        "maximum_75pct_gap_vs_same_instance_greedy_percent": maximum_high_rate_gap,
        "maximum_75pct_ape_vs_paper_greedy_percent": maximum_high_rate_paper_ape,
        "method_error_vs_paper_greedy": method_error.to_dict(orient="records"),
    }
    return report, rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    report, rebuilt = audit(result_dir)
    (result_dir / "acceptance_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    audit_lines = [
        "# Fixed-assignment result acceptance audit",
        "",
        f"**Overall: {'PASS' if report['passed'] else 'FAIL'}**",
        "",
        f"- Raw runs: {report['raw_rows']} / {report['expected_raw_rows']}",
        f"- Duplicate rows: {report['duplicate_rows']}",
        f"- Supplied summary matches raw-data rebuild: {report['summary_matches_raw']}",
        f"- Groups: {report['groups']}; instances/group: {report['instances_per_group']}",
        f"- Environment: `{report['environment']}`",
        f"- Assignment policy: `{report['assignment_policy']}`",
        f"- Nominal fleet m=n/5 and active-fleet bounds valid: "
        f"{report['fleet_definition_m_equals_n_over_5']}",
        f"- Short-route fallback disabled: {report['short_route_fallback'] is None}",
        f"- Minimum group QoS: {report['minimum_group_qos_percent']:.3f}%",
        f"- All 15 cost sequences strictly increase with dynamic rate: "
        f"{report['all_cost_sequences_strictly_increasing']}",
        f"- Same-instance Greedy MAPE vs paper: "
        f"{report['same_instance_greedy_mape_vs_paper_percent']:.3f}%",
        f"- Same-instance Greedy maximum APE vs paper: "
        f"{report['same_instance_greedy_max_ape_vs_paper_percent']:.3f}%",
        f"- Maximum 75% gap from same-instance Greedy: "
        f"{report['maximum_75pct_gap_vs_same_instance_greedy_percent']:.3f}%",
        f"- Maximum 75% APE from paper Greedy: "
        f"{report['maximum_75pct_ape_vs_paper_greedy_percent']:.3f}%",
        "",
        "## Error against paper Greedy",
        "",
    ]
    error_frame = pd.DataFrame(report["method_error_vs_paper_greedy"])
    error_frame["mape_percent"] = error_frame["mape_percent"].map(lambda x: f"{x:.3f}")
    error_frame["max_ape_percent"] = error_frame["max_ape_percent"].map(lambda x: f"{x:.3f}")
    audit_lines.append(_markdown_table(error_frame))
    if report["failures"]:
        audit_lines.extend(["", "## Failures", ""])
        audit_lines.extend(f"- {failure}" for failure in report["failures"])
    (result_dir / "ACCEPTANCE_AUDIT.md").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )
    rebuilt.to_csv(result_dir / "summary_rebuilt_from_raw.csv", index=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
