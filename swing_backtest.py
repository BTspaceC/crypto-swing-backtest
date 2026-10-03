"""4h swing backtest, long only: indicators, engine, statistics and the selection rule.

    python swing_backtest.py KLINES.csv FUNDING.csv --tag out/BTC                # in-sample grid only
    python swing_backtest.py KLINES.csv FUNDING.csv --tag out/BTC --oos A70_MA50_F1   # + OOS, ONE config

Conventions (each is a choice the written rules left open):
  * bar i closes -> signal; fill at the open of bar i+1, plus slippage
  * 1R = 2 * ATR14 (Wilder) of the last bar before entry; initial stop = fill - 1R, live on the entry bar
  * trailing stop for bar j = max(previous stop, HH(entry..j-1) - 3 * ATR[j-1]); never uses bar j itself
  * stop exit price = min(open[j], stop), minus slippage (a gap through the stop fills at the open)
  * time stop: checked once, at the close of the 30th bar held (entry bar = bar 1)
  * funding: charged at each settlement t with entry_time < t <= open time of the last bar held,
    amount = rate * that bar's open, per unit
  * funding filter: the latest settlement at or before the entry time must be below the cap
  * daily filter: 4h -> UTC daily bars; close > MA_n and MA50 above the previous day's MA50;
    bars of day D only see day D-1
"""
import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

IS_END = pd.Timestamp("2024-01-01")  # trades are split by ENTRY time
BREAKOUTS, DAILY_MAS = (40, 55, 70), (50, 100, 200)


@dataclass(frozen=True)
class Params:
    fee: float = 0.0005        # per side
    slip: float = 0.0002       # per side
    atr_n: int = 14
    init_atr: float = 2.0      # initial stop distance = 1R
    trail_atr: float = 3.0
    time_bars: int = 30
    fund_cap: float = 0.0003   # per 8h
    entry_delay: int = 0       # extra bars between signal and fill (execution-lag sensitivity)


P = Params()


# ----------------------------------------------------------------------------- data
def load(kpath, fpath):
    """Klines joined with funding. Funding stays NaN on bars without a settlement at their open."""
    k = pd.read_csv(kpath)
    k["t"] = pd.to_datetime(k["open_time"], unit="ms")
    k = k.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    assert (k["t"].diff().dropna() == pd.Timedelta("4h")).all(), "gap in 4h klines"

    f = pd.read_csv(fpath)
    # API stamps carry a few ms of jitter (…00:00:00.001) -> round to the hour before joining
    f["t"] = pd.to_datetime(f["fundingTime"], unit="ms").dt.round("h")
    f = f.sort_values("t").drop_duplicates("t")
    k = k.merge(f[["t", "fundingRate"]].rename(columns={"fundingRate": "fund"}), on="t", how="left")

    matched = k["fund"].notna().sum()
    assert matched == f["t"].between(k["t"].iloc[0], k["t"].iloc[-1]).sum(), "funding rows lost in join"
    share = k.loc[k["t"] >= f["t"].iloc[0], "fund"].notna().mean()
    assert 0.45 < share < 0.55, f"funding coverage {share:.3f}, expected ~0.5"
    return k, share


def load_symbol(sym, folder="data"):
    k, _ = load(f"{folder}/{sym}USDT_klines_4h.csv", f"{folder}/{sym}USDT_funding.csv")
    return indicators(k)


def indicators(k, p=P):
    h, l, c = k["high"], k["low"], k["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    k["atr"] = tr.ewm(alpha=1 / p.atr_n, adjust=False, min_periods=p.atr_n).mean()  # Wilder
    k["ema50"] = c.ewm(span=50, adjust=False, min_periods=50).mean()

    # daily bars built from the 4h bars; only complete days (6 bars)
    g = k.groupby(k["t"].dt.floor("D"))
    d = pd.DataFrame({"close": g["close"].last(), "n": g["close"].size()})
    d = d[d["n"] == 6]
    assert (d.index.to_series().diff().dropna() == pd.Timedelta("1D")).all()
    ma50 = d["close"].rolling(50).mean()
    day = k["t"].dt.floor("D")
    for n in DAILY_MAS:
        ok = (d["close"] > d["close"].rolling(n).mean()) & (ma50 > ma50.shift(1))
        ok.index = ok.index + pd.Timedelta("1D")  # day D uses day D-1's finished bar
        k[f"trend{n}"] = day.map(ok).fillna(False).astype(bool).values

    for n in BREAKOUTS:
        k[f"sigA{n}"] = c > h.shift(1).rolling(n).max()

    band = k["ema50"] + 0.5 * k["atr"]
    touched = (l <= band).astype(float).rolling(6).max() == 1
    k["sigB"] = touched & (c > h.shift(1)) & (c > k["ema50"]) & (k["ema50"] > k["ema50"].shift(6))

    # last settled rate known at the OPEN of each bar (filter only — never summed)
    k["last_fund"] = k["fund"].ffill()
    return k


# ----------------------------------------------------------------------------- engine
def run(k, sig_col, trend_col, fund_filter, p=P, signals=None):
    """One position at a time. Returns closed trades; a position still open at the end of the
    data is not a trade but is described in `.attrs["open"]`.

    `signals` (bool array) replaces `sig_col & trend_col` — used by the random-entry benchmark.
    """
    o, h, l, c = (k[x].values for x in ("open", "high", "low", "close"))
    atr, fund, last_fund = k["atr"].values, k["fund"].values, k["last_fund"].values
    sig = (k[sig_col] & k[trend_col]).values if signals is None else signals
    t = k["t"].values
    n = len(k)
    trades, open_pos = [], None
    i = 0
    while i < n - 1 - p.entry_delay:
        e = i + 1 + p.entry_delay
        if not sig[i] or np.isnan(atr[e - 1]):
            i += 1
            continue
        if fund_filter and not (last_fund[e] < p.fund_cap):  # NaN (no settlement yet) -> skip
            i += 1
            continue
        entry = o[e] * (1 + p.slip)
        R = p.init_atr * atr[e - 1]
        stop = entry - R
        hh, ll = -np.inf, np.inf
        fund_cost = 0.0
        exit_px, reason, j = None, None, e
        while j < n:
            if j > e:
                stop = max(stop, hh - p.trail_atr * atr[j - 1])  # hh = highs of e..j-1 only
                if not np.isnan(fund[j]):
                    fund_cost += fund[j] * o[j]
            if l[j] <= stop:
                exit_px = min(o[j], stop) * (1 - p.slip)
                reason = "stop" if stop <= entry - R + 1e-12 else "trail"
                ll = min(ll, min(o[j], stop))
                if o[j] > stop:
                    hh = max(hh, o[j])
                break
            hh, ll = max(hh, h[j]), min(ll, l[j])
            if j - e + 1 == p.time_bars and hh - entry < R:
                exit_px, reason = c[j] * (1 - p.slip), "time"
                break
            j += 1
        if exit_px is None:  # still open at the end of the data
            open_pos = dict(signal_time=t[i], entry_time=t[e], bars=n - e, entry=entry, R_px=R,
                            stop_next_bar=max(stop, hh - p.trail_atr * atr[n - 1]),
                            unrealised_r=(c[n - 1] - entry) / R, mfe=(hh - entry) / R)
            break
        fees = p.fee * (entry + exit_px)
        trades.append(dict(
            signal_time=t[i], entry_time=t[e], exit_time=t[j], bars=j - e + 1, reason=reason,
            entry=entry, exit=exit_px, R_px=R,
            r=(exit_px - entry - fees - fund_cost) / R,
            gross_r=(exit_px - entry) / R, fee_r=fees / R, fund_r=fund_cost / R,
            mfe=(hh - entry) / R, mae=(ll - entry) / R,
            stop_pct=R / entry,  # stop distance as a fraction of price; size = risk / stop_pct
        ))
        i = j  # flat at the close of bar j -> its signal may be taken
    out = pd.DataFrame(trades)
    out.attrs["open"] = open_pos
    return out


def mtm_equity(k, tr):
    """Bar-by-bar equity in R, marking open positions to each bar's close (and to the low for
    the worst intrabar point). Costs are booked when the trade closes."""
    t = k["t"].values
    close, low = k["close"].values, k["low"].values
    eq_close = np.zeros(len(k))
    eq_low = np.zeros(len(k))
    realised = np.zeros(len(k))
    for x in tr.itertuples():
        e, j = np.searchsorted(t, np.datetime64(x.entry_time)), np.searchsorted(t, np.datetime64(x.exit_time))
        eq_close[e:j] += (close[e:j] - x.entry) / x.R_px
        eq_low[e:j] += (low[e:j] - x.entry) / x.R_px
        realised[j] += x.r
    base = np.cumsum(realised)
    return pd.DataFrame({"close": base + eq_close, "low": base + eq_low}, index=k["t"])


def max_drawdown(eq, worst=None):
    """Largest fall from a running peak of `eq`; if `worst` is given, the trough is read from it."""
    peak = np.maximum.accumulate(np.concatenate([[0.0], np.asarray(eq)]))[1:]
    return float((peak - np.asarray(eq if worst is None else worst)).max())


# ----------------------------------------------------------------------------- stats
def stats(tr):
    if len(tr) == 0:
        return dict(n=0)
    r = tr["r"].values
    streak = longest = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        longest = max(longest, streak)
    gp, gl = r[r > 0].sum(), -r[r <= 0].sum()
    top = np.sort(r)[::-1][: max(1, int(np.ceil(len(r) * 0.1)))]
    return dict(
        n=len(r), win=(r > 0).mean(), avgR=r.mean(), medR=np.median(r),
        PF=gp / gl if gl > 0 else np.inf, totR=r.sum(), maxDD=max_drawdown(np.cumsum(r)), maxLS=longest,
        bars=tr["bars"].mean(), fundR=tr["fund_r"].mean(),
        top10_gross=top.sum() / gp if gp > 0 else np.nan,   # share of gross profit
        top10_net=top.sum() / r.sum() if r.sum() > 0 else np.nan,  # share of net profit
    )


def configs():
    """(name, signal column, trend column, funding filter) for all 24 configurations."""
    out = []
    for ff in (0, 1):
        for ma in DAILY_MAS:
            for nb in BREAKOUTS:
                out.append((f"A{nb}_MA{ma}_F{ff}", f"sigA{nb}", f"trend{ma}", bool(ff)))
            out.append((f"B_MA{ma}_F{ff}", "sigB", f"trend{ma}", bool(ff)))
    return out


def config(name):
    return next(c for c in configs() if c[0] == name)


def select(grid):
    """The pre-registered selection rule, applied to an IS grid only.

    Highest IS avgR, subject to: the adjacent breakout lengths (40-55, 55-70) under the same
    MA filter and funding switch also have avgR > 0. Pullback entry B has no neighbours, so
    for B the condition is vacuous.
    """
    adj = {40: (55,), 55: (40, 70), 70: (55,)}
    ok = []
    for name in grid.index:
        if name.startswith("A"):
            nb, rest = int(name[1:3]), name[3:]
            if not all(grid.loc[f"A{m}{rest}", "avgR"] > 0 for m in adj[nb]):
                continue
        ok.append(name)
    return grid.loc[ok, "avgR"].idxmax()


def split(tr):
    if len(tr) == 0:
        return tr, tr
    return tr[tr["entry_time"] < IS_END], tr[tr["entry_time"] >= IS_END]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("klines")
    ap.add_argument("funding")
    ap.add_argument("--tag", default="out/run", help="output prefix, e.g. out/BTC")
    ap.add_argument("--oos", metavar="CONFIG", help="print IS and OOS for this one config instead of the IS grid")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.tag) or ".", exist_ok=True)
    k, share = load(a.klines, a.funding)
    k = indicators(k)
    print(f"bars {len(k)}  {k.t.iloc[0]} -> {k.t.iloc[-1]}   funding coverage of bars: {share:.4f}")

    pd.set_option("display.width", 250)
    if a.oos is None:
        rows = []
        for name, s, tcol, ff in configs():
            is_, _ = split(run(k, s, tcol, ff))  # OOS half is discarded unseen
            rows.append(dict(config=name, **stats(is_)))
        df = pd.DataFrame(rows).set_index("config")
        df.to_csv(f"{a.tag}_IS_grid.csv")
        print(df.round(3).to_string())
    else:
        name, s, tcol, ff = config(a.oos)
        tr = run(k, s, tcol, ff)
        tr.to_csv(f"{a.tag}_{name}_trades.csv", index=False)
        is_, oos = split(tr)
        df = pd.DataFrame([dict(seg="IS", **stats(is_)), dict(seg="OOS", **stats(oos)),
                           dict(seg="ALL", **stats(tr))]).set_index("seg")
        df.to_csv(f"{a.tag}_{name}_IS_OOS.csv")
        print(name)
        print(df.round(3).T.to_string())
        if tr.attrs["open"]:
            print("open position at end of data (not counted):", tr.attrs["open"])


if __name__ == "__main__":
    main()
