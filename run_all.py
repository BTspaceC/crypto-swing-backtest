"""Reproduce every result in order: IS grids -> selection on BTC IS -> OOS for that one config
-> report -> robustness checks -> tests.

Uses the CSVs already in data/. To refresh them first: python fetch_data.py BTCUSDT (etc.) —
newer data will change the out-of-sample numbers.
"""
import subprocess
import sys

import pandas as pd

import swing_backtest as sb

SYMS = ["BTC", "ETH", "BNB"]


def py(*args):
    print(">", "python", *args, flush=True)
    subprocess.run([sys.executable, *args], check=True)


def files(s):
    return [f"data/{s}USDT_klines_4h.csv", f"data/{s}USDT_funding.csv"]


py("check_data.py")
for s in SYMS:
    py("swing_backtest.py", *files(s), "--tag", f"out/{s}")

# the choice is made here, from BTC in-sample numbers alone, before any OOS trade is printed
sel = sb.select(pd.read_csv("out/BTC_IS_grid.csv", index_col=0))
print(f"\nselected: {sel}\n", flush=True)

for s in SYMS:
    py("verify_is.py", f"{s}USDT", sel)
    py("swing_backtest.py", *files(s), "--tag", f"out/{s}", "--oos", sel)
py("report.py")
py("robustness.py")
py("-m", "pytest")
