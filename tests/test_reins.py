import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reins import attribution_by_year, car, drawdowns, ols, perf_table  # noqa: E402
from reins.analysis import MARKET  # noqa: E402


def synth(n=3000, seed=0, beta=1.2, rate_beta=2.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    m = 0.0002 + 0.01 * rng.standard_normal(n)
    dy = 0.0005 * rng.standard_normal(n)
    y = 0.0001 + beta * m + rate_beta * dy + 0.008 * rng.standard_normal(n)
    return pd.DataFrame({"Y": y, MARKET: m, "dy10": dy}, index=idx)


def test_ols_matches_lstsq_and_recovers_betas():
    r = synth(20_000)
    b, se, _ = ols(r["Y"].to_numpy(), r[[MARKET, "dy10"]].to_numpy())
    X = np.column_stack([np.ones(len(r)), r[[MARKET, "dy10"]]])
    assert b == pytest.approx(np.linalg.lstsq(X, r["Y"], rcond=None)[0], rel=1e-10)
    assert abs(b[1] - 1.2) < 3 * se[1] and abs(b[2] - 2.0) < 3 * se[2]


def test_newey_west_equals_white_without_lags_and_is_finite():
    r = synth(5000)
    b0, se0, _ = ols(r["Y"].to_numpy(), r[[MARKET]].to_numpy(), hac_lags=0)
    _, se10, _ = ols(r["Y"].to_numpy(), r[[MARKET]].to_numpy(), hac_lags=10)
    assert np.all(np.isfinite(se10)) and se10[1] == pytest.approx(se0[1], rel=0.15)


def test_attribution_parts_sum_to_total():
    r = synth().rename(columns={"Y": "SREN.SW"})
    a = attribution_by_year(r, "SREN.SW")
    assert (a["market"] + a["rates"] + a["stock-specific"] - a["total"]).abs().max() < 1e-14


def test_car_recovers_injected_abnormal_return():
    r = synth(1500, seed=4)
    d0 = 1000
    r.iloc[d0:d0 + 6, 0] += -0.06 / 6
    c = car(r, "Y", d0, (0, 5))
    assert c["car"] == pytest.approx(-0.06, abs=3 * 0.008 * np.sqrt(6))
    assert c["t"] < -1.96


def test_car_t_statistic_has_correct_size_on_null_data():
    rej = 0
    for s in range(400):
        r = synth(600, seed=100 + s)
        rej += abs(car(r, "Y", 400, (0, 5))["t"]) > 1.96
    assert 0.02 < rej / 400 < 0.09


def test_perf_and_drawdown_on_known_path():
    idx = pd.bdate_range("2020-01-01", periods=4)
    lvl = pd.Series([1.0, 0.5, 0.75, 1.0], index=idx)
    lr = np.log(lvl).diff().fillna(0.0)
    dd = drawdowns(lr, top=1)[0]
    assert dd["depth_%"] == pytest.approx(-50.0) and str(dd["recovered"]) == "2020-01-06"
    p = perf_table(pd.DataFrame({"A": lr}), ["A"])
    assert float(p.loc["A", "TSR_total_%"]) == pytest.approx(0.0, abs=1e-12)
    assert float(p.loc["A", "max_drawdown_%"]) == pytest.approx(-50.0)
