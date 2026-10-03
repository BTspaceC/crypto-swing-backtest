"""Integrity checks on the downloaded klines and funding history."""
import sys

import numpy as np
import pandas as pd

H4 = pd.Timedelta("4h")

for s in sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
    k = pd.read_csv(f"data/{s}_klines_4h.csv")
    f = pd.read_csv(f"data/{s}_funding.csv")
    t = pd.to_datetime(k.open_time, unit="ms")
    gaps = (t.diff().dropna() != H4).sum()
    bad = ((k.high < k[["open", "close", "low"]].max(axis=1)) | (k.low > k[["open", "close", "high"]].min(axis=1))).sum()
    rng = (k.high - k.low) / k.close
    print(f"{s}: {len(k)} bars {t.iloc[0]} -> {t.iloc[-1]} | gaps {gaps} | bad OHLC rows {bad} | NaN {k.isna().sum().sum()}")
    print(f"   widest bar: {t[rng.idxmax()]} range/close = {rng.max():.2f}")

    jitter = (f.fundingTime % 3_600_000 != 0).mean()
    ft = pd.to_datetime(f.fundingTime, unit="ms").dt.round("h")
    stamps = ft.values.astype("datetime64[ms]").astype("int64")
    since = k[k.open_time >= stamps.min()]
    print(f"   funding: {len(f)} settlements | intervals {ft.diff().dropna().value_counts().to_dict()}")
    print(f"   stamps off the hour: {jitter:.1%} | bars with a settlement at the open: "
          f"{np.isin(since.open_time.values, stamps).mean():.4f} (expect ~0.5)")
    print(f"   rate >= 0.03%: {(f.fundingRate >= 0.0003).mean():.1%} of settlements")
