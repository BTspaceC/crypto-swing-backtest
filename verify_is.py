"""Re-derive every in-sample trade of one config from scratch, independently of the engine.

Klines are truncated before IS_END first, so this never touches out-of-sample data.
Usage: python verify_is.py BTCUSDT A70_MA50_F1
"""
import sys

import pandas as pd

import swing_backtest as sb

sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
cfg = sys.argv[2] if len(sys.argv) > 2 else "A70_MA50_F1"
name, sig, trend, ff = sb.config(cfg)
nb, ma = int(sig[4:]) if sig != "sigB" else None, int(trend[5:])

k, _ = sb.load(f"data/{sym}_klines_4h.csv", f"data/{sym}_funding.csv")
k = sb.indicators(k)
k = k[k.t < sb.IS_END].reset_index(drop=True)
f = pd.read_csv(f"data/{sym}_funding.csv")
f["t"] = pd.to_datetime(f.fundingTime, unit="ms").dt.round("h")

tr = sb.run(k, sig, trend, ff)
for _, x in tr.iterrows():
    e = k.index[k.t == x.entry_time][0]
    j = k.index[k.t == x.exit_time][0]
    i = e - 1
    # daily filter rebuilt from bars strictly before the signal bar's UTC day
    d = k[k.t < k.t[i].floor("D")].groupby(k.t.dt.floor("D")).close.last()
    ma50 = d.rolling(50).mean()
    assert d.iloc[-1] > d.rolling(ma).mean().iloc[-1] and ma50.iloc[-1] > ma50.iloc[-2]
    if nb:
        assert k.close[i] > k.high[i - nb:i].max()
    # funding recomputed from the raw file
    paid = f[(f.t > x.entry_time) & (f.t <= x.exit_time)].merge(k[["t", "open"]], on="t")
    assert abs((paid.fundingRate * paid.open).sum() / x.R_px - x.fund_r) < 1e-9
    if ff:
        assert f[f.t <= x.entry_time].fundingRate.iloc[-1] < sb.P.fund_cap
    # stop path replayed bar by bar
    stop, hh = x.entry - x.R_px, float("-inf")
    for q in range(e, j + 1):
        if q > e:
            stop = max(stop, hh - sb.P.trail_atr * k.atr[q - 1])
        if k.low[q] <= stop:
            assert q == j and x.reason in ("stop", "trail")
            break
        hh = max(hh, k.high[q])
    else:
        assert x.reason == "time" and x.bars == sb.P.time_bars and x.mfe < 1
    # anything held past the time-stop bar must have reached +1R by then
    if x.bars > sb.P.time_bars:
        assert (k.high[e:e + sb.P.time_bars].max() - x.entry) / x.R_px >= 1

assert (tr.entry_time.values[1:] >= tr.exit_time.values[:-1]).all(), "overlapping positions"
print(f"{sym} {cfg}: {len(tr)} in-sample trades re-derived, all checks passed")
