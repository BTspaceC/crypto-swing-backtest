"""Engine behaviour on hand-built bars: each test pins down one rule or one classic backtest bug."""
import numpy as np
import pandas as pd
import pytest

import swing_backtest as sb

ATR = 5.0                      # constant, so 1R = 10 and the trail sits 15 below the high
SLIP, FEE = sb.P.slip, sb.P.fee
FLAT = (100, 101, 99, 100)


def bars(rows, signals=(0,), fund=None):
    """rows: (open, high, low, close). Bar 0 is the signal bar unless `signals` says otherwise."""
    k = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    k["t"] = pd.date_range("2024-01-01", periods=len(k), freq="4h")
    k["atr"] = ATR
    k["fund"] = np.nan if fund is None else fund
    k["last_fund"] = k["fund"].ffill()
    k["sig"] = k.index.isin(list(signals))
    k["trend"] = True
    return k


def test_fill_at_next_open_and_initial_stop():
    k = bars([FLAT, (100, 102, 99, 101), (101, 101, 89, 90), FLAT])
    (t,) = sb.run(k, "sig", "trend", False).itertuples()
    entry = 100 * (1 + SLIP)
    assert t.entry_time == k.t[1] and t.entry == pytest.approx(entry)
    assert t.R_px == pytest.approx(2 * ATR)
    assert t.reason == "stop" and t.exit_time == k.t[2]
    assert t.exit == pytest.approx((entry - 10) * (1 - SLIP))
    assert t.r == pytest.approx((t.exit - entry - FEE * (entry + t.exit)) / 10)
    assert t.r < -1  # a full stop always costs slightly more than 1R


def test_stop_can_trigger_on_the_entry_bar():
    k = bars([FLAT, (100, 100, 85, 88), FLAT])
    (t,) = sb.run(k, "sig", "trend", False).itertuples()
    assert t.bars == 1 and t.reason == "stop"


def test_trailing_stop_ignores_the_current_bars_high():
    # Bar 2 spikes to 130 and dips to 95. A stop raised by that same bar's high (130 - 15 = 115)
    # would be "hit" by its low: that is the look-ahead bug. The honest stop is still near 90.
    k = bars([FLAT, (100, 101, 99, 100), (100, 130, 95, 120), (120, 121, 114, 118), FLAT])
    (t,) = sb.run(k, "sig", "trend", False).itertuples()
    assert t.exit_time == k.t[3]          # stopped one bar later, by the trail set from bar 2's high
    assert t.reason == "trail"
    assert t.exit == pytest.approx(115 * (1 - SLIP))


def test_stop_never_moves_down():
    # ATR jumps after entry, so high - 3*ATR falls far below the initial stop; the stop must stay put
    k = bars([FLAT, (100, 104, 99, 103), (103, 104, 91, 95), (95, 96, 89.5, 92), FLAT])
    k.loc[1:, "atr"] = 20.0
    (t,) = sb.run(k, "sig", "trend", False).itertuples()
    assert t.reason == "stop" and t.exit_time == k.t[3]
    assert t.exit == pytest.approx((100 * (1 + SLIP) - 10) * (1 - SLIP))


def test_gap_through_the_stop_fills_at_the_open():
    k = bars([FLAT, (100, 101, 99, 100), (80, 82, 78, 81), FLAT])
    (t,) = sb.run(k, "sig", "trend", False).itertuples()
    assert t.exit == pytest.approx(80 * (1 - SLIP))
    assert t.r < -1.9


def test_time_stop_fires_at_bar_30_when_never_up_1r():
    k = bars([FLAT] * 40)
    t = sb.run(k, "sig", "trend", False).iloc[0]
    assert t.reason == "time" and t.bars == 30
    assert t.exit == pytest.approx(100 * (1 - SLIP))


def test_time_stop_is_checked_once_not_every_bar_after():
    # reaches +1R early, then sags below entry and sits there past bar 30: must stay open
    k = bars([FLAT, (100, 111, 99, 108)] + [(97, 98, 96.5, 97)] * 45)
    tr = sb.run(k, "sig", "trend", False)
    assert len(tr) == 0
    assert tr.attrs["open"]["bars"] == 46 and tr.attrs["open"]["mfe"] > 1


def test_funding_charged_once_per_settlement_and_not_at_entry():
    fund = [np.nan, 0.001, np.nan, 0.001, np.nan, 0.001, np.nan, np.nan]
    k = bars([FLAT, FLAT, FLAT, FLAT, FLAT, (100, 100, 85, 86), FLAT, FLAT], fund=fund)
    (t,) = sb.run(k, "sig", "trend", False).itertuples()
    # entry at bar 1 (its own settlement is not paid); the settlements at bars 3 and 5 are
    assert t.fund_r == pytest.approx(2 * 0.001 * 100 / 10)


def test_funding_filter_uses_latest_settled_rate_at_entry():
    hot = bars([FLAT] * 6, fund=[0.0001, 0.0005, np.nan, np.nan, np.nan, np.nan])
    blocked = sb.run(hot, "sig", "trend", True)
    assert len(blocked) == 0 and blocked.attrs["open"] is None
    cool = bars([FLAT] * 6, fund=[0.0005, 0.0001, np.nan, np.nan, np.nan, np.nan])
    assert sb.run(cool, "sig", "trend", True).attrs["open"] is not None
    assert sb.run(hot, "sig", "trend", False).attrs["open"] is not None  # filter off: trade taken


def test_one_position_at_a_time():
    rng = np.random.default_rng(1)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    rows = [(p, p * 1.01, p * 0.99, p * (1 + rng.normal(0, 0.004))) for p in px]
    k = bars(rows, signals=range(600))
    k["atr"] = 1.5
    tr = sb.run(k, "sig", "trend", False)
    assert len(tr) > 20
    assert (tr.entry_time.values[1:] > tr.exit_time.values[:-1]).all()


def test_entry_delay_shifts_the_fill_by_one_bar():
    k = bars([FLAT, (100, 101, 99, 100), (105, 106, 104, 105)] + [FLAT] * 3)
    on_time = sb.run(k, "sig", "trend", False).attrs["open"]
    late = sb.run(k, "sig", "trend", False, sb.Params(entry_delay=1)).attrs["open"]
    assert pd.Timestamp(on_time["entry_time"]) == k.t[1]
    assert pd.Timestamp(late["entry_time"]) == k.t[2] and late["entry"] == pytest.approx(105 * (1 + SLIP))


def test_mtm_equity_ends_at_closed_total_and_sees_drawdown_inside_a_trade():
    k = bars([FLAT, (100, 101, 99, 100), (100, 100, 92, 93), (93, 125, 93, 124), (124, 125, 100, 101), FLAT])
    tr = sb.run(k, "sig", "trend", False)
    assert len(tr) == 1 and tr.r.iloc[0] > 0
    eq = sb.mtm_equity(k, tr)
    assert eq.close.iloc[-1] == pytest.approx(tr.r.sum())
    assert sb.stats(tr)["maxDD"] == 0                      # a winning trade shows no closed-trade drawdown
    assert sb.max_drawdown(eq.close, eq.low) > 0.7         # but it was 0.8R under water on the way


def test_selection_rule_requires_positive_neighbours():
    names = [c[0] for c in sb.configs()]
    grid = pd.DataFrame({"avgR": 0.1}, index=names)
    grid.loc["A55_MA50_F0", "avgR"] = 2.0      # best, but one neighbour is negative
    grid.loc["A40_MA50_F0", "avgR"] = -0.1
    grid.loc["A70_MA100_F1", "avgR"] = 1.0     # runner-up with positive neighbours
    assert sb.select(grid) == "A70_MA100_F1"
    grid.loc["A40_MA50_F0", "avgR"] = 0.1
    assert sb.select(grid) == "A55_MA50_F0"
