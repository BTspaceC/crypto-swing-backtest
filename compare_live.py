"""Compare a log of real trades with what the rules would have done over the same period.

    python compare_live.py live_trades.csv

The log is a CSV with one row per closed trade (times in UTC):

    symbol,entry_time,entry_price,exit_time,exit_price,qty,fees,funding
    BTC,2026-08-19 16:00,61250.5,2026-08-23 05:10,64010.0,0.004,0.25,0.03

`fees` and `funding` are totals paid for the trade in quote currency and may be left empty.
A live trade is matched to the backtest trade in the same coin whose entry is within one 4h bar.

The point is to measure execution, not the strategy: how far real fills are from the assumed
ones, which signals were skipped, and which trades had no signal behind them.
"""
import os
import sys

import pandas as pd

import swing_backtest as sb

BAR = pd.Timedelta("4h")
UNIVERSE = ["BTC", "ETH", "BNB"]


def main(path):
    live = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    live[["fees", "funding"]] = live.reindex(columns=["fees", "funding"]).fillna(0.0)
    name = sb.select(pd.read_csv("out/BTC_IS_grid.csv", index_col=0))
    _, sig, trend, ff = sb.config(name)
    start, end = live.entry_time.min() - BAR, live.exit_time.max()

    rows, missed = [], []
    for sym in sorted(set(UNIVERSE) | set(live.symbol)):  # coins never traded live can still have missed signals
        lv = live[live.symbol == sym]
        folder = "data/live" if os.path.exists(f"data/live/{sym}USDT_klines_4h.csv") else "data"
        bt = sb.run(sb.load_symbol(sym, folder), sig, trend, ff)
        bt = bt[(bt.entry_time >= start) & (bt.entry_time <= end)]
        used = set()
        for x in lv.itertuples():
            gap = (bt.entry_time - x.entry_time).abs()
            hit = gap.idxmin() if len(bt) and gap.min() <= BAR and gap.idxmin() not in used else None
            if hit is None:
                rows.append(dict(symbol=sym, entry_time=x.entry_time, match="无对应信号"))
                continue
            used.add(hit)
            b = bt.loc[hit]
            ref_in, ref_out = b.entry / (1 + sb.P.slip), b.exit / (1 - sb.P.slip)  # prices before assumed slippage
            live_r = (x.exit_price - x.entry_price - (x.fees + x.funding) / x.qty) / b.R_px
            rows.append(dict(
                symbol=sym, entry_time=x.entry_time, match="已匹配",
                entry_slip=x.entry_price / ref_in - 1, exit_slip=1 - x.exit_price / ref_out,
                exit_lag_h=(x.exit_time - b.exit_time) / pd.Timedelta("1h"),
                live_r=live_r, backtest_r=b.r, diff_r=live_r - b.r, reason=b.reason))
        for b in bt.drop(index=list(used)).itertuples():
            missed.append(dict(symbol=sym, entry_time=b.entry_time, backtest_r=b.r))

    res = pd.DataFrame(rows).sort_values("entry_time")
    pd.set_option("display.width", 220)
    print(f"配置 {name}，对照区间 {start:%Y-%m-%d} 至 {end:%Y-%m-%d}\n")
    show = res.copy()
    for c in ("entry_slip", "exit_slip"):
        if c in show:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.3%}")
    for c in ("exit_lag_h", "live_r", "backtest_r", "diff_r"):
        if c in show:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
    print(show.fillna("").to_string(index=False))

    ok = res[res.match == "已匹配"]
    print(f"\n实盘 {len(live)} 笔：匹配到信号 {len(ok)} 笔，无对应信号 {len(res) - len(ok)} 笔；"
          f"规则在此期间另有 {len(missed)} 笔你没有做")
    if len(ok):
        print(f"入场滑点 平均 {ok.entry_slip.mean():+.3%}，出场滑点 平均 {ok.exit_slip.mean():+.3%}"
              f"（回测假设单边 {sb.P.slip:.2%}；正数表示比参考价更差）")
        print(f"单笔结果 实盘平均 {ok.live_r.mean():+.2f}R，回测平均 {ok.backtest_r.mean():+.2f}R，"
              f"差 {ok.diff_r.mean():+.2f}R")
    if missed:
        m = pd.DataFrame(missed)
        print(f"\n漏掉的信号（合计 {m.backtest_r.sum():+.2f}R）：")
        m["backtest_r"] = m["backtest_r"].map("{:+.2f}".format)
        print(m.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "live_trades_example.csv")
