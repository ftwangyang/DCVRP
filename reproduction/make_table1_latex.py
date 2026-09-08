"""Emit Table I as LaTeX from the measured CSV produced by evaluate_table1.py.

Two layouts are written:

``TABLE_I_N20_measured.tex``
    A drop-in replacement for Table I in the manuscript's own style (booktabs,
    ``cost $\\pm$ sd``, percentage QoS), with the best cost per dynamic rate in
    bold.  Use this if the reproduced numbers are adopted as the reported ones.

``TABLE_I_N20_comparison.tex``
    An appendix table placing the measured value next to the currently reported
    one with the signed relative error, for a reproducibility appendix or a
    response letter that keeps the original Table I.

Nothing here recomputes or adjusts a measurement; it only formats
``table_i_n20_measured.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "experiments/results/reproduction_n20"
# Table I's row order, best method last, as in the manuscript.
ROW_ORDER = ("Greedy", "MARDAM", "MAAM", "LiDRL", "AMCVN", "DVNDA")


def order_rows(frame: pd.DataFrame) -> list[str]:
    present = set(frame.method)
    return [method for method in ROW_ORDER if method in present]


def measured_table(frame: pd.DataFrame, *, with_time: bool) -> str:
    methods = order_rows(frame)
    rates = sorted(frame.dynamic_rate.unique())
    columns = "cll" + ("ccc" if with_time else "cc")
    header = "Dynamic rate & Method & Cost $\\downarrow$ & QoS $\\uparrow$"
    if with_time:
        header += " & Time (s) $\\downarrow$"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Performance on dynamic CVRP instances with $n=20$ customers "
        "and $m=4$ vehicles, averaged over 100 held-out test instances. "
        "Best cost per dynamic rate in bold.}",
        "\\label{tab:main_results_n20}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        header + " \\\\",
    ]
    for rate in rates:
        block = frame[frame.dynamic_rate == rate].set_index("method")
        best = block.cost_mean.idxmin()
        lines.append("\\midrule")
        for position, method in enumerate(methods):
            row = block.loc[method]
            cost = f"{row.cost_mean:.2f} $\\pm$ {row.cost_sd:.2f}"
            if method == best:
                cost = f"\\textbf{{{cost}}}"
            label = f"{100 * rate:.0f}\\%" if position == 0 else ""
            cells = [label, method, cost, f"{row.qos_percent:.2f}\\%"]
            if with_time:
                cells.append(f"{row.inference_seconds:.2f}")
            lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def comparison_table(frame: pd.DataFrame) -> str:
    methods = order_rows(frame)
    rates = sorted(frame.dynamic_rate.unique())
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Independent re-run of Table I under the fully specified "
        "protocol of the reproducibility package. Reported values are those of "
        "Table I; measured values come from a re-training of all five neural "
        "methods and a re-run of Greedy on the fixed test split. A negative "
        "error means the re-run is cheaper than the reported value.}",
        "\\label{tab:reproduction_n20}",
        "\\begin{tabular}{cllccc}",
        "\\toprule",
        "Dynamic rate & Method & Reported cost & Measured cost & "
        "Rel. error & Measured QoS \\\\",
    ]
    for rate in rates:
        block = frame[frame.dynamic_rate == rate].set_index("method")
        lines.append("\\midrule")
        for position, method in enumerate(methods):
            row = block.loc[method]
            label = f"{100 * rate:.0f}\\%" if position == 0 else ""
            lines.append(
                " & ".join(
                    [
                        label,
                        method,
                        f"{row.paper_cost_mean:.2f} $\\pm$ {row.paper_cost_sd:.2f}",
                        f"{row.cost_mean:.2f} $\\pm$ {row.cost_sd:.2f}",
                        f"{100 * row.cost_relative_error:+.1f}\\%",
                        f"{row.qos_percent:.2f}\\%",
                    ]
                )
                + " \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def ordering_report(frame: pd.DataFrame) -> dict:
    """Whether the measured ranking still supports the manuscript's claim."""

    report = {}
    for rate in sorted(frame.dynamic_rate.unique()):
        block = frame[frame.dynamic_rate == rate]
        measured = list(block.sort_values("cost_mean").method)
        reported = list(block.sort_values("paper_cost_mean").method)
        report[f"{rate:.2f}"] = {
            "measured_order": measured,
            "reported_order": reported,
            "identical": measured == reported,
            "dvnda_is_best_measured": measured[0] == "DVNDA",
            "dvnda_is_best_reported": reported[0] == "DVNDA",
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--with-time",
        action="store_true",
        help="Include the inference-time column in the replacement table.",
    )
    args = parser.parse_args()
    output = args.output or args.input

    csv = args.input / "table_i_n20_measured.csv"
    if not csv.exists():
        print(f"missing {csv}; run reproduction.evaluate_table1 first")
        return 1
    frame = pd.read_csv(csv)
    output.mkdir(parents=True, exist_ok=True)

    (output / "TABLE_I_N20_measured.tex").write_text(
        measured_table(frame, with_time=args.with_time), encoding="utf-8"
    )
    (output / "TABLE_I_N20_comparison.tex").write_text(
        comparison_table(frame), encoding="utf-8"
    )
    report = ordering_report(frame)
    (output / "ordering_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"wrote TABLE_I_N20_measured.tex and TABLE_I_N20_comparison.tex -> {output}")
    for rate, entry in report.items():
        state = "same order" if entry["identical"] else "order differs"
        best = entry["measured_order"][0]
        print(f"  phi={float(rate):.2f}: {state}; measured best = {best}")
    claim = all(entry["dvnda_is_best_measured"] for entry in report.values())
    print(
        "manuscript claim (DVNDA best at every dynamic rate): "
        + ("HOLDS" if claim else "DOES NOT HOLD")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
