"""Download Binance USDS-M perpetual 4h klines + funding history via the public REST API.

    python fetch_data.py                      # BTCUSDT ETHUSDT BNBUSDT -> data/
    python fetch_data.py SOLUSDT --out data/live

Writes <out>/<SYMBOL>_klines_4h.csv and <out>/<SYMBOL>_funding.csv. Only closed bars are kept.
(The data.binance.vision monthly archive only starts 2020-01; the API goes back to listing.)

Re-downloading into data/ replaces the frozen research snapshot and will change the
out-of-sample numbers. scan.py keeps its own copy in data/live/ for that reason.
"""
import json
import os
import sys
import time
import urllib.request

import pandas as pd

BASE = "https://fapi.binance.com"
START_MS = 1567296000000  # 2019-09-01 00:00 UTC


def get(path, **params):
    url = BASE + path + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, e, file=sys.stderr)
            time.sleep(2 + attempt * 2)
    raise RuntimeError("failed: " + url)


def fetch_klines(symbol):
    rows, start = [], START_MS
    while True:
        chunk = get("/fapi/v1/klines", symbol=symbol, interval="4h", startTime=start, limit=1500)
        if not chunk:
            break
        rows += chunk
        start = chunk[-1][0] + 1
        if len(chunk) < 1500:
            break
        time.sleep(0.25)
    df = pd.DataFrame(rows).iloc[:, :9]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades"]
    df = df.drop_duplicates("open_time").sort_values("open_time")
    # drop the still-forming last bar
    df = df[df["close_time"] < int(time.time() * 1000)]
    return df


def fetch_funding(symbol):
    rows, start = [], START_MS
    while True:
        chunk = get("/fapi/v1/fundingRate", symbol=symbol, startTime=start, limit=1000)
        if not chunk:
            break
        rows += chunk
        start = chunk[-1]["fundingTime"] + 1
        if len(chunk) < 1000:
            break
        time.sleep(0.6)
    df = pd.DataFrame(rows)[["fundingTime", "fundingRate", "markPrice"]]
    df = df.drop_duplicates("fundingTime").sort_values("fundingTime")
    return df


def fetch(sym, out="data"):
    os.makedirs(out, exist_ok=True)
    k = fetch_klines(sym)
    k.to_csv(f"{out}/{sym}_klines_4h.csv", index=False)
    f = fetch_funding(sym)
    f.to_csv(f"{out}/{sym}_funding.csv", index=False)
    print(sym, "klines", len(k), pd.to_datetime(k.open_time.iloc[0], unit="ms"), "->",
          pd.to_datetime(k.open_time.iloc[-1], unit="ms"), "| funding", len(f))


if __name__ == "__main__":
    args = sys.argv[1:]
    out = "data"
    if "--out" in args:
        out = args[args.index("--out") + 1]
        del args[args.index("--out"):args.index("--out") + 2]
    for sym in args or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
        fetch(sym, out)
