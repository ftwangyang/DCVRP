"""Table I acceptance is per dynamic rate, not mean MAE."""

from eval import TABLE1_CELL_GAP_LIMIT, table1_gap_report


N20_DVNDA = {
    0.10: 8.31,
    0.25: 8.95,
    0.50: 10.47,
    0.75: 11.78,
}


def test_perfect_match_passes():
    report = table1_gap_report(20, "DVNDA", dict(N20_DVNDA), cell_limit=5.0)
    assert report["all_within"]
    assert report["max_abs"] == 0.0
    assert TABLE1_CELL_GAP_LIMIT == 5.0


def test_mae_below_5_does_not_pass_if_one_rate_fails():
    measured = dict(N20_DVNDA)
    measured[0.75] = 11.78 * 1.06  # +6% on one cell
    report = table1_gap_report(20, "DVNDA", measured, cell_limit=5.0)
    assert report["mae"] < 5.0
    assert report["max_abs"] > 5.0
    assert report["all_within"] is False


def test_qos_below_100_fails_even_if_costs_match():
    qos = {0.10: 100.0, 0.25: 100.0, 0.50: 100.0, 0.75: 83.0}
    report = table1_gap_report(
        20,
        "DVNDA",
        dict(N20_DVNDA),
        cell_limit=5.0,
        qos_by_rate=qos,
        min_qos_percent=100.0,
    )
    assert report["qos_ok"] is False
    assert report["all_within"] is False


def test_all_rates_inside_band_pass():
    measured = {
        0.10: 8.31 * 1.04,
        0.25: 8.95 * 0.96,
        0.50: 10.47 * 1.049,
        0.75: 11.78 * 0.951,
    }
    qos = {rate: 100.0 for rate in measured}
    report = table1_gap_report(
        20, "DVNDA", measured, cell_limit=5.0, qos_by_rate=qos, min_qos_percent=100.0
    )
    assert report["all_within"]
    assert report["max_abs"] <= 5.0
