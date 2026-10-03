"""Checks on the real data in data/: no look-ahead anywhere in the pipeline, and frozen results."""
import numpy as np
import pandas as pd
import pytest

import swing_backtest as sb

SYMS = ["BTC", "ETH", "BNB"]
SEL = "A70_MA50_F1"


@pytest.fixture(scope="module")
def raw():
    return {s: sb.load(f"data/{s}USDT_klines_4h.csv", f"data/{s}USDT_funding.csv")[0] for s in SYMS}


@pytest.mark.parametrize("sym", SYMS)
def test_truncating_the_future_never_changes_the_past(raw, sym):
    """Strongest look-ahead test: delete everything after a cut, recompute indicators and trades
    from scratch. Every trade that had already closed must come out identical."""
    _, sig, trend, ff = sb.config(SEL)
    full = sb.run(sb.indicators(raw[sym].copy()), sig, trend, ff)
    for cut in (4000, 7777, 11000, 14001):
        k = sb.indicators(raw[sym].iloc[:cut].copy())
        part = sb.run(k, sig, trend, ff)
        done = full[full.exit_time <= k.t.iloc[-1]]
        # a trade the full run closes on the very last bar may still be open in the truncated run
        assert len(done) - len(part) in (0, 1)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), done.iloc[:len(part)].reset_index(drop=True))


@pytest.mark.parametrize("sym", SYMS)
def test_every_indicator_and_signal_is_causal(raw, sym):
    cut = 9000
    full = sb.indicators(raw[sym].copy())
    part = sb.indicators(raw[sym].iloc[:cut].copy())
    cols = ["atr", "ema50", "last_fund", "sigB"] + [f"trend{n}" for n in sb.DAILY_MAS] + [f"sigA{n}" for n in sb.BREAKOUTS]
    pd.testing.assert_frame_equal(full.loc[:cut - 1, cols], part[cols])


def test_daily_filter_only_sees_finished_days(raw):
    k = sb.indicators(raw["BTC"].copy())
    day = k.t.dt.floor("D")
    assert (k.groupby(day).trend50.nunique() == 1).all()  # constant within a UTC day
    daily = k.groupby(day).close.last()
    ma50 = daily.rolling(50).mean()
    one = pd.Timedelta("1D")
    for d in pd.to_datetime(["2020-06-15", "2021-11-10", "2023-03-01", "2025-08-20"]):
        y = d - one
        expect = daily[y] > ma50[y] and ma50[y] > ma50[y - one]
        assert bool(k.loc[day == d, "trend50"].iloc[0]) == bool(expect)


def test_daily_filter_is_set_on_the_unfinished_last_day(raw):
    # regression: bars of a day that had not finished yet used to get trend = False
    full = sb.indicators(raw["BTC"].copy())
    mid_day = full.index[(full.index > 10000) & full.trend50 & (full.t.dt.hour == 8)]
    cut = mid_day[0] + 1                                   # data ends on the 08:00 bar of a day with the filter on
    part = sb.indicators(raw["BTC"].iloc[:cut].copy())
    assert (part.t.dt.floor("D") == part.t.dt.floor("D").iloc[-1]).sum() == 3
    assert part.trend50.iloc[-1]


@pytest.mark.parametrize("sym", SYMS)
def test_funding_joined_once_per_settlement(raw, sym):
    k = raw[sym]
    f = pd.read_csv(f"data/{sym}USDT_funding.csv")
    assert (f.fundingTime % 3_600_000 != 0).mean() > 0.3   # the raw stamps really are jittered
    assert k.fund.notna().sum() == len(f)                   # yet nothing is lost in the join
    assert k.fund.sum() == pytest.approx(f.fundingRate.sum())
    assert abs(k.fund.notna().mean() - 0.5) < 0.01          # every other 4h bar, not every bar


def test_frozen_results(raw):
    """The numbers quoted in the README, for the data snapshot shipped in data/."""
    _, sig, trend, ff = sb.config(SEL)
    tr = {s: sb.run(sb.indicators(raw[s].copy()), sig, trend, ff) for s in SYMS}
    is_, oos = sb.split(tr["BTC"])
    assert (len(is_), len(oos)) == (37, 38)
    assert is_.r.mean() == pytest.approx(1.084, abs=1e-3)
    assert oos.r.mean() == pytest.approx(0.059, abs=1e-3)
    allr = np.concatenate([t.r.values for t in tr.values()])
    assert len(allr) == 229 and allr.sum() == pytest.approx(90.15, abs=0.01)


def test_selection_rule_on_the_real_grid(raw):
    k = sb.indicators(raw["BTC"].copy())
    rows = [dict(config=n, **sb.stats(sb.split(sb.run(k, s, t, f))[0])) for n, s, t, f in sb.configs()]
    assert sb.select(pd.DataFrame(rows).set_index("config")) == SEL
