"""Audit the independent Tabu and OR-Tools paired reruns."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paper_multivehicle_solvers as solvers


KEYS = ["customer_count", "dynamic_rate", "instance_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--vienna-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-random-dir", type=Path)
    parser.add_argument("--old-vienna-dir", type=Path)
    return parser.parse_args()


def pair_results(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(result_dir / "raw_instances.csv")
    summary = pd.read_csv(result_dir / "summary.csv")
    tabu = raw[raw.method.eq("Tabu Search")].set_index(KEYS)
    ortools = raw[raw.method.eq("OR-Tools")].set_index(KEYS)
    paired = tabu[["cost", "qos_percent"]].join(
        ortools[["cost", "qos_percent"]], lsuffix="_tabu", rsuffix="_ortools"
    )
    paired["cost_delta_tabu_minus_ortools"] = (
        paired.cost_tabu - paired.cost_ortools
    )
    paired["same_cost"] = np.isclose(
        paired.cost_tabu, paired.cost_ortools, rtol=0.0, atol=1.0e-12
    )
    paired["same_cost_and_qos"] = paired.same_cost & np.isclose(
        paired.qos_percent_tabu,
        paired.qos_percent_ortools,
        rtol=0.0,
        atol=1.0e-12,
    )
    return paired, summary


def overall(paired: pd.DataFrame) -> dict:
    delta = paired.cost_delta_tabu_minus_ortools
    return {
        "paired_instances": int(len(paired)),
        "same_cost_instances": int(paired.same_cost.sum()),
        "same_cost_percent": float(100.0 * paired.same_cost.mean()),
        "same_cost_and_qos_percent": float(
            100.0 * paired.same_cost_and_qos.mean()
        ),
        "tabu_lower_cost_instances": int((delta < -1.0e-12).sum()),
        "ortools_lower_cost_instances": int((delta > 1.0e-12).sum()),
        "mean_absolute_paired_cost_difference": float(delta.abs().mean()),
        "maximum_absolute_paired_cost_difference": float(delta.abs().max()),
    }


def cell_table(
    dataset: str, paired: pd.DataFrame, summary: pd.DataFrame
) -> list[dict]:
    rows = []
    for (size, rate), group in paired.groupby(level=[0, 1], sort=True):
        tabu = summary[
            summary.method.eq("Tabu Search")
            & summary.customer_count.eq(size)
            & np.isclose(summary.dynamic_rate, rate)
        ].iloc[0]
        ortools = summary[
            summary.method.eq("OR-Tools")
            & summary.customer_count.eq(size)
            & np.isclose(summary.dynamic_rate, rate)
        ].iloc[0]
        delta = group.cost_delta_tabu_minus_ortools
        rows.append(
            {
                "dataset": dataset,
                "customer_count": int(size),
                "vehicle_count": int(tabu.vehicle_count),
                "dynamic_rate": float(rate),
                "tabu_cost_mean": float(tabu.cost_mean),
                "tabu_cost_sd": float(tabu.cost_sd),
                "ortools_cost_mean": float(ortools.cost_mean),
                "ortools_cost_sd": float(ortools.cost_sd),
                "tabu_minus_ortools_cost": float(
                    tabu.cost_mean - ortools.cost_mean
                ),
                "tabu_qos_mean_percent": float(tabu.qos_mean_percent),
                "ortools_qos_mean_percent": float(ortools.qos_mean_percent),
                "tabu_time_100_s": float(tabu.time_100_s),
                "ortools_time_100_s": float(ortools.time_100_s),
                "same_cost_percent": float(100.0 * group.same_cost.mean()),
                "tabu_lower_cost_instances": int((delta < -1.0e-12).sum()),
                "ortools_lower_cost_instances": int((delta > 1.0e-12).sum()),
            }
        )
    return rows


def old_same_percent(result_dir: Path | None) -> float | None:
    if result_dir is None:
        return None
    paired, _ = pair_results(result_dir)
    return float(100.0 * paired.same_cost.mean())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "Random": args.random_dir.resolve(),
        "Vienna": args.vienna_dir.resolve(),
    }
    paired_by_dataset = {}
    summary_by_dataset = {}
    audit = {}
    rows = []
    for name, result_dir in datasets.items():
        paired, summary = pair_results(result_dir)
        paired_by_dataset[name] = paired
        summary_by_dataset[name] = summary
        audit[name] = overall(paired)
        rows.extend(cell_table(name, paired, summary))

    cell_frame = pd.DataFrame(rows).sort_values(
        ["dataset", "customer_count", "dynamic_rate"]
    )
    cell_frame.to_csv(output_dir / "comparison_by_cell.csv", index=False)
    audit["before_fix_same_cost_percent"] = {
        "Random": old_same_percent(args.old_random_dir),
        "Vienna": old_same_percent(args.old_vienna_dir),
    }
    audit["checks"] = {
        "2400_paired_instances": all(
            value["paired_instances"] == 1_200
            for key, value in audit.items()
            if key in datasets
        ),
        "tabu_source_does_not_call_ortools": (
            "ortools_routes(" not in inspect.getsource(solvers.standalone_tabu_routes)
        ),
        "both_datasets_have_distinct_results": all(
            audit[name]["same_cost_percent"] < 100.0 for name in datasets
        ),
    }
    audit["passed"] = all(audit["checks"].values())
    (output_dir / "SEPARATION_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Tabu Search 与 OR-Tools 独立实现复测",
        "",
        "Tabu Search 现采用独立 Python 禁忌搜索，不再调用 OR-Tools；其邻域包括 swap、relocate 和 2-opt。OR-Tools 保持 PATH_CHEAPEST_ARC 初始解与 GUIDED_LOCAL_SEARCH。",
        "",
    ]
    for name in ("Random", "Vienna"):
        stats = audit[name]
        old = audit["before_fix_same_cost_percent"][name]
        lines.extend(
            [
                f"## {name}",
                "",
                (
                    f"逐实例 Cost 完全相同率：修正前 {old:.2f}%，修正后 "
                    f"{stats['same_cost_percent']:.2f}%；Tabu 更低 "
                    f"{stats['tabu_lower_cost_instances']} 例，OR-Tools 更低 "
                    f"{stats['ortools_lower_cost_instances']} 例。"
                ),
                "",
                "| n/m | 动态率 | Tabu Cost | OR-Tools Cost | 差值 | Tabu QoS | OR-Tools QoS | Tabu Time(100) | OR-Tools Time(100) | 同Cost实例 |",
                "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected = cell_frame[cell_frame.dataset.eq(name)]
        for row in selected.itertuples(index=False):
            lines.append(
                f"| {row.customer_count}/{row.vehicle_count} | "
                f"{100 * row.dynamic_rate:.0f}% | "
                f"{row.tabu_cost_mean:.3f}±{row.tabu_cost_sd:.3f} | "
                f"{row.ortools_cost_mean:.3f}±{row.ortools_cost_sd:.3f} | "
                f"{row.tabu_minus_ortools_cost:+.3f} | "
                f"{row.tabu_qos_mean_percent:.2f}% | "
                f"{row.ortools_qos_mean_percent:.2f}% | "
                f"{row.tabu_time_100_s:.2f}s | "
                f"{row.ortools_time_100_s:.2f}s | "
                f"{row.same_cost_percent:.0f}% |"
            )
        lines.append("")
    lines.extend(
        [
            "## 解释",
            "",
            "高动态率下部分单元仍会完全相同，因为固定车辆分配后，每次重规划的逐车路线通常只有一至两个客户，此时不存在或几乎不存在可改变的访问顺序。QoS 也可能一致，因为两种方法共享车辆分配、容量约束和披露过程，仅优化车内访问顺序。",
        ]
    )
    (output_dir / "REPORT_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise SystemExit("Tabu/OR-Tools separation audit failed")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
