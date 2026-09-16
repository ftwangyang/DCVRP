"""Hard acceptance gate for the four n=20 DVNDA Table-I cells."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_RATES = (0.10, 0.25, 0.50, 0.75)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--max-absolute-gap", type=float, default=5.0)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    rows = {
        round(float(row["phi"]), 2): row
        for row in source_rows
        if row["method"] == "DVNDA" and int(row["scale"]) == 20
    }
    missing = [rate for rate in EXPECTED_RATES if rate not in rows]
    checks = []
    for rate in EXPECTED_RATES:
        if rate not in rows:
            continue
        row = rows[rate]
        signed_gap = float(row["cost_gap_percent"])
        qos_percent = float(row["measured_qos_percent"])
        cost_ok = abs(signed_gap) <= args.max_absolute_gap
        qos_ok = qos_percent + 1e-6 >= 100.0
        checks.append(
            {
                "phi": rate,
                "measured_cost": float(row["measured_cost_mean"]),
                "table1_cost": float(row["table1_cost_mean"]),
                "signed_gap_percent": signed_gap,
                "absolute_gap_percent": abs(signed_gap),
                "qos_percent": qos_percent,
                "passed": cost_ok and qos_ok,
            }
        )

    passed = not missing and len(checks) == 4 and all(item["passed"] for item in checks)
    payload = {
        "criterion": (
            f"all four absolute cost gaps <= {args.max_absolute_gap:.2f}% "
            "and QoS 100%"
        ),
        "passed": passed,
        "missing_rates": missing,
        "checks": checks,
    }
    output = args.csv_path.with_name("acceptance_gate.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Acceptance report: {output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
