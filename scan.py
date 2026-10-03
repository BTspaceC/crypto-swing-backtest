"""Where does each coin stand right now under the selected configuration?

    python scan.py --refresh                  # download the latest bars into data/live/, then scan
    python scan.py                            # scan whatever is already on disk
    python scan.py --risk 20 --equity 1000    # also size a position: 20 lost if the stop is hit

For every coin it reports the state after the last CLOSED 4h bar: whether the rules hold a
position (and where its stop sits for the next bar), or whether a new signal just fired, or how
far price is from one. It places no orders and stores nothing about your account.
"""
import argparse
import os

import pandas as pd

import fetch_data
import swing_backtest as sb

BAR = pd.Timedelta("4h")


def describe(sym, k, name, risk, equity):
    _, sig, trend, ff = sb.config(name)
    last = k.iloc[-1]
    close_time = last.t + BAR
    tr = sb.run(k, sig, trend, ff)
    pos = tr.attrs["open"]
    atr, R = last.atr, sb.P.init_atr * last.atr
    print(f"\n{sym}USDT   最后一根已收盘 K 线：{last.t:%Y-%m-%d %H:%M} UTC，收盘价 {last.close:g}")

    age = pd.Timestamp.now("UTC").tz_localize(None) - close_time
    if age > BAR:
        print(f"  注意：数据已过期 {age.total_seconds() / 3600:.0f} 小时，先运行 python scan.py --refresh")

    if pos:
        held = pos["bars"]
        print(f"  状态：持仓中（{pd.Timestamp(pos['entry_time']):%Y-%m-%d %H:%M} 入场，成交价 {pos['entry']:g}，已持 {held} 根）")
        print(f"  浮动盈亏 {pos['unrealised_r']:+.2f}R，最高到过 {pos['mfe']:+.2f}R")
        print(f"  下一根 K 线的止损价：{pos['stop_next_bar']:g}")
        if held < sb.P.time_bars and pos["mfe"] < 1:
            print(f"  时间止损：还剩 {sb.P.time_bars - held} 根，届时若仍未到过 +1R 则按收盘价平仓")
        return

    fund_ok = (not ff) or last.last_fund < sb.P.fund_cap
    fired = bool(last[sig]) and bool(last[trend]) and fund_ok
    level = k.high.iloc[-int(sig[4:]):].max() if sig != "sigB" else None
    print("  状态：空仓")
    print(f"  日线趋势过滤：{'通过' if last[trend] else '未通过'}")
    if ff:
        print(f"  最近一次资金费率：{last.last_fund:.4%}（{'低于' if fund_ok else '不低于'}阈值 {sb.P.fund_cap:.2%}）")
    if fired:
        print(f"  【信号】本根收盘触发，规则在下一根开盘入场")
        print(f"    参考入场价 {last.close:g}，初始止损约 {last.close - R:g}（距离 {R:g}，即 {R / last.close:.2%}）")
        if risk:
            qty = risk / R
            line = f"    每笔风险 {risk:g} → 数量 {qty:.4g}，名义价值约 {qty * last.close:,.0f}"
            if equity:
                line += f"，相当于净值的 {qty * last.close / equity:.1f} 倍"
            print(line)
        print("    实际止损以真实成交价减去这段距离为准。")
    else:
        if level is not None:
            print(f"  下一根要触发突破，收盘需高于 {level:g}（距现价 {level / last.close - 1:+.2%}）")
        if risk:
            qty = risk / R
            line = f"  若此刻入场：止损距离 {R / last.close:.2%}，每笔风险 {risk:g} → 数量 {qty:.4g}，名义价值约 {qty * last.close:,.0f}"
            if equity:
                line += f"（净值的 {qty * last.close / equity:.1f} 倍）"
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("symbols", nargs="*", default=["BTC", "ETH", "BNB"])
    ap.add_argument("--refresh", action="store_true", help="download the latest data into data/live/ first")
    ap.add_argument("--config", help="default: the configuration chosen by the selection rule")
    ap.add_argument("--risk", type=float, help="amount lost if the initial stop is hit")
    ap.add_argument("--equity", type=float, help="account size, only used to express notional as a multiple")
    a = ap.parse_args()

    if a.refresh:
        for s in a.symbols:
            fetch_data.fetch(f"{s}USDT", "data/live")
    name = a.config or sb.select(pd.read_csv("out/BTC_IS_grid.csv", index_col=0))
    print(f"配置 {name}")
    for s in a.symbols:
        folder = "data/live" if os.path.exists(f"data/live/{s}USDT_klines_4h.csv") else "data"
        describe(s, sb.load_symbol(s, folder), name, a.risk, a.equity)
    print("\n以上只是规则在历史数据上的机械输出，不是交易建议。")


if __name__ == "__main__":
    main()
