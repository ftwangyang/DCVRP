"""Build a concise reviewer-facing Markdown report from verified result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/REVIEWER_EXPERIMENT_REPORT.md"))
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No verified rows available._"
    text = frame.fillna("--").astype(str)
    header = "| " + " | ".join(text.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in text.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def add_table(lines: list[str], title: str, path: Path) -> None:
    lines.extend([f"## {title}", ""])
    if path.exists():
        lines.extend([markdown_table(pd.read_csv(path)), ""])
    else:
        lines.extend(["_Not completed; no result file was found._", ""])


def add_evidence_status(lines: list[str], results_root: Path) -> None:
    """Add claim-level conclusions so tables cannot be read selectively."""
    architecture = pd.read_csv(
        results_root / "architecture" / "raw" / "architecture_instances.csv"
    )
    production = pd.read_csv(
        results_root / "production" / "raw" / "production_instances.csv"
    )
    classical = pd.read_csv(
        results_root / "classical" / "raw" / "classical_instances.csv"
    )
    transfer = pd.read_csv(
        results_root / "transfer" / "raw" / "transfer_instances.csv"
    )
    paired = pd.read_csv(
        results_root / "summary" / "architecture_paired_tests.csv"
    )
    architecture_means = architecture.groupby("selector").cost_with_penalty.mean()
    best_selector = architecture_means.idxmin()
    best_cost = architecture_means.min()
    independent_cost = architecture_means["independent"]
    shared_cost = architecture_means["shared"]
    independence_test = paired[paired["selector"] == "independent"].iloc[0]
    dvnda_cost = production.cost_with_penalty.mean()
    classical_means = classical.groupby("method").cost_with_penalty.mean()
    best_classical = classical_means.idxmin()
    best_classical_cost = classical_means.min()
    relative_excess = 100.0 * (dvnda_cost / best_classical_cost - 1.0)
    transfer_means = transfer.groupby(
        ["selector", "target_vehicle_count", "stage"]
    ).cost_with_penalty.mean()
    independent_zero = transfer_means.loc[("independent", 100, "zero_shot")]
    independent_tuned = transfer_means.loc[("independent", 100, "fine_tuned")]
    shared_zero = transfer_means.loc[("shared", 100, "zero_shot")]
    shared_tuned = transfer_means.loc[("shared", 100, "fine_tuned")]
    lines.extend([
        "## Evidence status",
        "",
        "**Status: not yet safe to preserve the manuscript's original superiority claim.** "
        "The experimental requests have been implemented, but the resulting evidence "
        "requires a claim/method revision before a point-by-point response can state that "
        "all reviewer concerns are resolved.",
        "",
        f"- The lowest mean ablation cost was obtained by `{best_selector}` "
        f"({best_cost:.2f}), not the independent DVNDA selector ({independent_cost:.2f}).",
        f"- Independent versus shared weights changed mean cost from {shared_cost:.2f} "
        f"to {independent_cost:.2f}; the paired seed-level test was not significant "
        f"after Holm correction (adjusted p={independence_test.holm_p:.3f}).",
        f"- The continued DVNDA checkpoint had mean cost {dvnda_cost:.2f}, whereas the "
        f"best rolling-horizon baseline (`{best_classical}`) achieved "
        f"{best_classical_cost:.2f}; DVNDA was {relative_excess:.1f}% higher under the "
        "common executed-distance-plus-unserved-penalty objective.",
        f"- At m=100, one-step fine-tuning changed independent cost "
        f"{independent_zero:.2f} to {independent_tuned:.2f} and shared cost "
        f"{shared_zero:.2f} to {shared_tuned:.2f}; transfer is therefore not reliably "
        "improved by short fine-tuning.",
        "- No manuscript checkpoint, original train/test split, or reported raw result "
        "file was present in the repository. These are clean local reruns, not a "
        "reproduction of the manuscript's published table values.",
        "",
    ])


def add_integrity_checks(lines: list[str], results_root: Path) -> None:
    official = {
        "architecture": results_root / "architecture" / "raw" / "architecture_instances.csv",
        "production": results_root / "production" / "raw" / "production_instances.csv",
        "aggregation_ordering": results_root / "aggregation_ordering" / "raw" / "aggregation_ordering_instances.csv",
        "robustness": results_root / "robustness" / "raw" / "robustness_instances.csv",
        "classical": results_root / "classical" / "raw" / "classical_instances.csv",
        "optimality": results_root / "optimality" / "raw" / "optimality_gap_instances.csv",
        "real_roads": results_root / "real_roads" / "raw" / "real_road_instances.csv",
        "transfer": results_root / "transfer" / "raw" / "transfer_instances.csv",
    }
    records = []
    for name, path in official.items():
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include="number")
        required = [
            column for column in ["cost_with_penalty", "distance", "qos"]
            if column in numeric
        ]
        missing = int(numeric[required].isna().sum().sum()) if required else 0
        identity_error = "--"
        if {"cost_with_penalty", "distance", "unserved"}.issubset(frame.columns):
            error = (
                frame.cost_with_penalty - frame.distance - 5.0 * frame.unserved
            ).abs().max()
            identity_error = f"{error:.2e}"
        records.append({
            "dataset": name,
            "rows": len(frame),
            "required_numeric_NA": missing,
            "max_cost_identity_error": identity_error,
        })
    lines.extend([
        "## Raw-data integrity checks",
        "",
        markdown_table(pd.DataFrame(records)),
        "",
        "The cost identity is `penalized cost = executed distance + 5 x unserved`; "
        "small nonzero residuals are CSV floating-point round-off.",
        "",
    ])


def main():
    args = parse_args()
    summary = args.results_root / "summary"
    lines = [
        "# Reviewer-requested experiment report",
        "",
        "All values in this report are generated from per-instance CSV source data. "
        "The original repository files are not modified; compatibility repairs and "
        "reviewer studies live under `experiments/`.",
        "",
        "## Evidence map",
        "",
        "| Reviewer concern | Implemented evidence |",
        "| --- | --- |",
        "| Independent vs shared weights and parameter count | shared, shared-wide, shared+ID, centralized, independent-narrow, independent |",
        "| Sequential vehicle/customer choice | direct joint-pair decoder |",
        "| Aggregation and genuine coordination | argmax, softmax, learned attention aggregation, random-vehicle negative control |",
        "| Vehicle-ID overfitting and ordering | subnet-to-vehicle permutation test |",
        "| Fixed-interval latency | 6/8/10/12/14 intervals plus fixed/event/hybrid scheduling |",
        "| Long-lag requests and dynamism | 10--90% dynamic ratios, six temporal arrival processes, and tanh coefficient C in {2,5,10,20} |",
        "| Travel/service uncertainty | travel noise, service noise, joint noise, and peak congestion |",
        "| Strong transportation baselines | nearest, regret insertion, local search, tabu, ALNS, OR-Tools |",
        "| Optimality reference | exact CVRP MIP on n=8, m=2 instances |",
        "| Actual road topology | directed OpenStreetMap networks for Vienna, London, and New York |",
        "| Fleet scalability | parameters, memory, latency to m=100 and m=10 transfer/fine-tuning |",
        "| Limited metrics | distance, penalized cost, completed QoS, response/completion time, unserved/late/committed counts, utilization, balance, idle wait, decision epochs, latency |",
        "",
    ]
    add_evidence_status(lines, args.results_root)
    environment_path = args.results_root / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        lines.extend([
            "## Execution environment",
            "",
            markdown_table(pd.DataFrame([environment])),
            "",
        ])
    add_table(lines, "Architecture ablation (mean and 95% CI across training seeds)", summary / "table_architecture.csv")
    add_table(lines, "Aggregation and vehicle-ordering tests", summary / "table_aggregation_ordering.csv")
    add_table(lines, "Dynamic scheduling and uncertainty", summary / "table_robustness.csv")
    add_table(lines, "Main neural method vs rolling-horizon classical baselines", summary / "table_neural_vs_classical.csv")
    add_table(lines, "Classical baseline details", summary / "table_classical_baselines.csv")
    add_table(lines, "Small-instance exact optimality gaps", summary / "table_optimality_gap.csv")
    add_table(lines, "Directed real-road networks", summary / "table_real_roads.csv")
    add_table(lines, "Selector scaling to 100 vehicles", summary / "table_scaling.csv")
    add_table(lines, "Fleet-size transfer and fine-tuning", summary / "table_fleet_transfer.csv")

    metadata_path = args.results_root / "real_roads" / "raw" / "road_metadata.json"
    lines.extend(["## Road-network provenance", ""])
    if metadata_path.exists():
        metadata = pd.DataFrame(json.loads(metadata_path.read_text(encoding="utf-8")))
        lines.extend([markdown_table(metadata), ""])
    else:
        lines.extend(["_Not completed; no road metadata was found._", ""])

    add_integrity_checks(lines, args.results_root)

    lines.extend([
        "## Interpretation guardrails",
        "",
        "- QoS means physically completed service by the normalized 480-minute horizon; assignment alone is not counted.",
        "- Penalized evaluation cost uses the same definition for every method: executed distance plus five times the number of customers not completed by the horizon.",
        "- Confidence intervals for neural models use independent training seeds as the top-level replicate; repeated stochastic rollouts are not treated as independent training runs.",
        "- The equal-budget architecture study supports attribution between variants. The separately continued main-model training is used for performance comparisons with optimized classical baselines.",
        "- Exact MIP gaps are static small-instance references and are not presented as dynamic online optimality bounds.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
