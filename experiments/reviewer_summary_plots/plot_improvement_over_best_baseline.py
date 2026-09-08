"""Plot Tables I, II, and IV as percentage-improvement forest plots.

The values below are the reported cost mean and standard deviation for DVNDA
and the lowest-mean non-DVNDA method in each table cell.  With only marginal
summary statistics available, 95% confidence intervals are approximated by an
independent-sample delta method.  Replace ``improvement_ci`` with a paired
bootstrap when per-instance paired costs are released.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Editable SVG text. The serif override below matches the manuscript's figures.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "0.75",
        "legend.framealpha": 1.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.7,
        "ytick.minor.size": 1.7,
    }
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)
N_INSTANCES = 100
DYNAMIC_RATES = (10, 25, 50, 75)

# A single manuscript-blue hue is used throughout.  Every point uses the same
# marker; the horizontal interval and the zero-reference line carry the result.
SERIES_STYLE = {
    "n = 20": {"color": "#2F5597", "hatch": ""},
    "n = 35": {"color": "#5B9BD5", "hatch": "///"},
    "n = 50": {"color": "#A9C4DF", "hatch": "..."},
    "Clustered": {"color": "#2F5597", "hatch": ""},
    "Mixed": {"color": "#8FB9DD", "hatch": "///"},
}


def row(
    table: str,
    rate: int,
    series: str,
    baseline: str,
    baseline_mean: float,
    baseline_sd: float,
    dvnda_mean: float,
    dvnda_sd: float,
) -> dict:
    return {
        "table": table,
        "dynamic_rate_percent": rate,
        "series": series,
        "best_baseline": baseline,
        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "dvnda_mean": dvnda_mean,
        "dvnda_sd": dvnda_sd,
        "n": N_INSTANCES,
    }


ROWS = [
    # Table I: synthetic instances.
    row("I", 10, "n = 20", "OR-Tools", 8.15, 1.19, 8.31, 1.22),
    row("I", 25, "n = 20", "OR-Tools", 9.14, 1.27, 8.95, 1.30),
    row("I", 50, "n = 20", "AMCVN", 10.75, 1.54, 10.47, 1.53),
    row("I", 75, "n = 20", "AMCVN", 12.03, 1.47, 11.78, 1.44),
    row("I", 10, "n = 35", "ALNS", 14.54, 2.02, 14.94, 2.00),
    row("I", 25, "n = 35", "OR-Tools", 16.41, 2.14, 16.14, 2.14),
    row("I", 50, "n = 35", "AMCVN", 19.04, 2.46, 18.90, 2.24),
    row("I", 75, "n = 35", "AMCVN", 21.19, 2.62, 20.98, 2.26),
    row("I", 10, "n = 50", "OR-Tools", 18.38, 2.32, 18.89, 2.27),
    row("I", 25, "n = 50", "AMCVN", 21.56, 2.51, 21.21, 2.66),
    row("I", 50, "n = 50", "AMCVN", 25.49, 2.89, 25.31, 2.81),
    row("I", 75, "n = 50", "AMCVN", 29.33, 3.05, 29.00, 2.69),
    # Table II: Vienna road-network instances.
    row("II", 10, "n = 20", "OR-Tools", 4.87, 0.93, 5.02, 0.62),
    row("II", 25, "n = 20", "AMCVN", 5.82, 0.87, 5.52, 0.78),
    row("II", 50, "n = 20", "AMCVN", 6.81, 0.96, 6.41, 0.87),
    row("II", 75, "n = 20", "AMCVN", 7.54, 0.98, 7.21, 0.94),
    row("II", 10, "n = 35", "OR-Tools", 8.25, 1.38, 8.52, 1.01),
    row("II", 25, "n = 35", "OR-Tools", 9.84, 1.37, 9.64, 1.19),
    row("II", 50, "n = 35", "AMCVN", 11.83, 1.31, 11.72, 1.32),
    row("II", 75, "n = 35", "AMCVN", 13.26, 1.46, 13.07, 1.66),
    row("II", 10, "n = 50", "OR-Tools", 11.35, 1.94, 11.60, 1.09),
    row("II", 25, "n = 50", "AMCVN", 13.25, 1.47, 12.85, 1.38),
    row("II", 50, "n = 50", "AMCVN", 15.73, 1.43, 15.18, 1.57),
    row("II", 75, "n = 50", "AMCVN", 17.96, 1.82, 17.71, 1.55),
    # Table IV: cross-distribution generalization, n=20.
    row("IV", 10, "Clustered", "AMCVN", 9.44, 1.68, 9.04, 1.70),
    row("IV", 25, "Clustered", "AMCVN", 10.07, 1.73, 9.52, 1.72),
    row("IV", 50, "Clustered", "AMCVN", 10.53, 1.93, 9.99, 1.71),
    row("IV", 75, "Clustered", "AMCVN", 11.08, 1.54, 10.18, 1.64),
    row("IV", 10, "Mixed", "AMCVN", 8.95, 1.15, 7.88, 1.10),
    row("IV", 25, "Mixed", "AMCVN", 8.80, 1.37, 8.58, 1.17),
    row("IV", 50, "Mixed", "AMCVN", 10.41, 1.54, 9.97, 1.36),
    row("IV", 75, "Mixed", "AMCVN", 11.75, 1.47, 11.38, 1.49),
]


def improvement_ci(record: dict) -> tuple[float, float, float, float]:
    """Return improvement, SE, lower CI, and upper CI in percentage points."""

    baseline = record["baseline_mean"]
    dvnda = record["dvnda_mean"]
    improvement = 100.0 * (baseline - dvnda) / baseline
    se_baseline = record["baseline_sd"] / math.sqrt(record["n"])
    se_dvnda = record["dvnda_sd"] / math.sqrt(record["n"])
    # I = 100 * (1 - D/B); independent-sample delta method.
    se = 100.0 * math.sqrt(
        (se_dvnda / baseline) ** 2
        + (dvnda * se_baseline / baseline**2) ** 2
    )
    critical = NormalDist().inv_cdf(0.975)
    return (
        improvement,
        se,
        improvement - critical * se,
        improvement + critical * se,
    )


for record in ROWS:
    estimate, se, low, high = improvement_ci(record)
    record.update(
        {
            "improvement_percent": estimate,
            "se_percent": se,
            "ci95_low_percent": low,
            "ci95_high_percent": high,
            "point_estimate_dvnda_better": estimate > 0.0,
            "ci95_supports_dvnda_better": low > 0.0,
            "ci_method": "independent-sample delta method from reported mean and SD",
        }
    )


def write_source_data() -> None:
    path = OUTPUT / "summary_improvement_source_data.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)


def plot_table(
    table: str,
    series_order: tuple[str, ...],
    filename: str,
    x_limits: tuple[float, float],
) -> None:
    records = []
    labels = []
    for series in series_order:
        values = sorted(
            (
                record
                for record in ROWS
                if record["table"] == table and record["series"] == series
            ),
            key=lambda record: record["dynamic_rate_percent"],
        )
        records.extend(values)
        for record in values:
            rate = record["dynamic_rate_percent"]
            if series.startswith("n ="):
                size = series.split("=")[1].strip()
                labels.append(f"n={size} | {rate}%")
            else:
                labels.append(f"{series} | {rate}%")

    estimates = np.array([record["improvement_percent"] for record in records])
    lows = np.array([record["ci95_low_percent"] for record in records])
    highs = np.array([record["ci95_high_percent"] for record in records])
    y_positions = np.arange(len(records))[::-1]
    figure_height = 3.70 if len(records) == 12 else 3.05
    fig, ax = plt.subplots(figsize=(3.50, figure_height))

    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
    ax.errorbar(
        estimates,
        y_positions,
        xerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        color="#2F5597",
        markerfacecolor="#2F5597",
        markeredgecolor="#2F5597",
        markersize=4.2,
        ecolor="#425A7A",
        elinewidth=1.0,
        capsize=2.3,
        capthick=0.9,
        zorder=2,
    )

    group_size = len(DYNAMIC_RATES)
    for group_index in range(1, len(series_order)):
        boundary = len(records) - group_index * group_size - 0.5
        ax.axhline(boundary, color="#D9DDE3", linewidth=0.55, zorder=0)

    ax.set_xlim(*x_limits)
    ax.set_ylim(-0.8, len(records) - 0.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("DVNDA improvement over best baseline (%)")
    ax.minorticks_off()
    ax.tick_params(
        axis="x",
        which="major",
        direction="in",
        top=False,
        bottom=True,
        length=3.2,
        width=0.75,
        labelsize=7.8,
    )
    ax.tick_params(
        axis="y",
        which="major",
        left=False,
        right=False,
        length=0,
        labelsize=7.6,
        pad=3.0,
    )
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.text(
        0.0,
        1.018,
        r"$\leftarrow$ Baseline lower cost",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )
    ax.text(
        1.0,
        1.018,
        r"DVNDA lower cost $\rightarrow$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )
    ax.text(
        0.5,
        1.062,
        "95% confidence intervals",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#333333",
    )
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.14, top=0.89)
    base = OUTPUT / filename
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    write_source_data()
    plot_table(
        "I",
        ("n = 20", "n = 35", "n = 50"),
        "main_table_I_improvement",
        (-7.5, 7.5),
    )
    plot_table(
        "II",
        ("n = 20", "n = 35", "n = 50"),
        "supp_table_II_improvement",
        (-8.0, 10.0),
    )
    plot_table(
        "IV",
        ("Clustered", "Mixed"),
        "supp_table_IV_improvement",
        (-2.5, 16.0),
    )
    for table in ("I", "II", "IV"):
        print(f"Table {table}")
        for record in (row for row in ROWS if row["table"] == table):
            print(
                f"  {record['series']:9s} {record['dynamic_rate_percent']:>2d}% "
                f"{record['improvement_percent']:+.2f}% "
                f"[{record['ci95_low_percent']:+.2f}, "
                f"{record['ci95_high_percent']:+.2f}] "
                f"vs {record['best_baseline']}"
            )


if __name__ == "__main__":
    main()
