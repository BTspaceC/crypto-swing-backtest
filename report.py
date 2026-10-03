"""Everything quoted in the README, computed from the files in out/ and data/.

Run after swing_backtest.py has produced the IS grids and the selected config's trades
(run_all.py does this in the right order).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

import swing_backtest as sb
from style import COLOR, INK, MUTED, SURFACE, SYMS

rng = np.random.default_rng(0)
pd.set_option("display.width", 250)


def ci_iid(r, n=20000):
    m = rng.choice(r, (n, len(r))).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def ci_block(tr, n=10000):
    """Resample whole calendar quarters: trades in different coins cluster in the same rallies."""
    g = [v.r.values for _, v in tr.groupby(tr.entry_time.dt.to_period("Q"))]
    m = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(n)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def drop_best(r, k):
    return np.sort(r)[:-k].mean()


# ------------------------------------------------------------------ selection (BTC in-sample only)
grid = pd.read_csv("out/BTC_IS_grid.csv", index_col=0)
SEL = sb.select(grid)
print(f"selected by rule from BTC in-sample grid: {SEL}\n")
_, SIG, TREND, FF = sb.config(SEL)

trades = {}
for s in SYMS:
    t = pd.read_csv(f"out/{s}_{SEL}_trades.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    t["sym"] = s
    trades[s] = t
pooled = pd.concat(trades.values()).sort_values("exit_time").reset_index(drop=True)
pooled.to_csv(f"out/POOLED_{SEL}_trades.csv", index=False)

# ------------------------------------------------------------------ summary table
rows = []
for s, t in list(trades.items()) + [("POOLED", pooled)]:
    for seg, x in zip(("IS", "OOS", "ALL"), (*sb.split(t), t)):
        lo, hi = ci_block(x) if s == "POOLED" else ci_iid(x.r.values)
        rows.append(dict(sym=s, seg=seg, **sb.stats(x), ci_lo=lo, ci_hi=hi, best=x.r.max(),
                         drop1=drop_best(x.r.values, 1), drop3=drop_best(x.r.values, 3),
                         drop5=drop_best(x.r.values, 5), fee_r=x.fee_r.mean()))
summary = pd.DataFrame(rows).set_index(["sym", "seg"])
summary.to_csv(f"out/summary_{SEL}.csv")
print(summary.round(2).to_string(), "\n")

print("total R by entry year")
by = pooled.groupby([pooled.entry_time.dt.year, "sym"]).r.sum().unstack()[SYMS]
by["total"], by["n"] = by.sum(axis=1), pooled.groupby(pooled.entry_time.dt.year).size()
by["avgR"] = by["total"] / by["n"]
print(by.round(2).to_string(), "\n")

overlap = np.mean([((pooled.sym != x.sym) & (pooled.entry_time <= x.exit_time) & (pooled.exit_time >= x.entry_time)).any()
                   for x in pooled.itertuples()])
top = pooled.nlargest(max(1, round(len(pooled) * 0.1)), "r")
print(f"pooled: {overlap:.0%} of trades overlap a trade in another coin; "
      f"best {len(top)} trades = {top.r.sum():.1f}R, other {len(pooled) - len(top)} = {pooled.r.sum() - top.r.sum():.1f}R\n")

# ------------------------------------------------------------------ signal funnel, time stop, funding filter
for s in SYMS:
    k, _ = sb.load(f"data/{s}USDT_klines_4h.csv", f"data/{s}USDT_funding.csv")
    k = sb.indicators(k)
    a = k[SIG]
    b = a & k[TREND]
    c = b & (k.last_fund.shift(-1) < sb.P.fund_cap)
    t = trades[s]
    print(f"{s} funnel: bars {len(k)} -> breakout {a.sum()} -> +trend {b.sum()} -> +funding {c.sum()} -> trades {len(t)}"
          f" | trend on {k[TREND].mean():.0%} of bars | in market {t.bars.sum() / len(k):.1%} of bars")
    n_time = {}
    for name, sg, tc, ff in sb.configs():
        is_, _ = sb.split(sb.run(k, sg, tc, ff))
        n_time[name] = int((is_.reason == "time").sum())
    print(f"   time-stop exits per config, IS: max {max(n_time.values())}, configs with none {sum(v == 0 for v in n_time.values())}/24")
    g = pd.read_csv(f"out/{s}_IS_grid.csv", index_col=0)
    off, on = g[g.index.str.endswith("F0")], g[g.index.str.endswith("F1")]
    on.index = off.index
    d = pd.DataFrame({"n_off": off.n, "n_on": on.n, "avgR_off": off.avgR, "avgR_on": on.avgR,
                      "d_avgR": on.avgR - off.avgR, "DD_off": off.maxDD, "DD_on": on.maxDD, "d_DD": on.maxDD - off.maxDD})
    d.index = d.index.str.replace("_F0", "")
    print("   funding filter off -> on, IS:")
    print(d.round(2).to_string().replace("\n", "\n   "), "\n")

# ------------------------------------------------------------------ figure 1: cumulative R
fig, ax = plt.subplots(figsize=(11, 5.2))
end = pooled.exit_time.max()
ax.axvspan(sb.IS_END, end + pd.Timedelta("200D"), color="#f1f0eb", lw=0)
for s in SYMS:
    t = trades[s].sort_values("exit_time")
    x = [t.entry_time.iloc[0], *t.exit_time, end]
    y = [0, *t.r.cumsum(), t.r.sum()]
    ax.step(x, y, where="post", color=COLOR[s], lw=1.6)
    ax.annotate(f"{s}  {t.r.sum():+.0f}R", (end, y[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10, color=INK)
x = [pooled.entry_time.min(), *pooled.exit_time, end]
y = [0, *pooled.r.cumsum(), pooled.r.sum()]
ax.step(x, y, where="post", color=INK, lw=2.2)
ax.annotate(f"三币合并  {pooled.r.sum():+.0f}R", (end, y[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=10, color=INK, fontweight="bold")
top_y = ax.get_ylim()[1]
ax.text(sb.IS_END - pd.Timedelta("12D"), top_y * 0.97, "样本内（用于选参数）", ha="right", va="top", fontsize=9.5, color=MUTED)
ax.text(sb.IS_END + pd.Timedelta("12D"), top_y * 0.97, "样本外（选定后才看）", ha="left", va="top", fontsize=9.5, color=MUTED)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlim(pooled.entry_time.min() - pd.Timedelta("30D"), end + pd.Timedelta("200D"))
ax.set_ylabel("累计 R（已扣手续费、滑点、资金费率）")
ax.set_title(f"累计盈亏，以初始风险 R 计 · 配置 {SEL}")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig("out/equity.png", dpi=150)

# ------------------------------------------------------------------ figure 2: R distribution + MAE/MFE per coin
C_IS, C_OOS = "#2a78d6", "#eb6834"
fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
for row, s in enumerate(SYMS):
    is_, oos = sb.split(trades[s])
    a = axes[row, 0]
    bins = np.arange(-1.5, np.ceil(trades[s].r.max()) + 0.5, 0.5)
    a.hist([is_.r, oos.r], bins=bins, stacked=True, color=[C_IS, C_OOS], edgecolor=SURFACE, lw=1.2,
           label=[f"样本内 {len(is_)} 笔", f"样本外 {len(oos)} 笔"])
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_title(f"{s} · 单笔结果分布")
    a.set_xlabel("单笔 R")
    a.set_ylabel("笔数")
    a.grid(axis="x", visible=False)
    a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a.legend(frameon=False, fontsize=9)
    a = axes[row, 1]
    for x, c, lab in ((is_, C_IS, "样本内"), (oos, C_OOS, "样本外")):
        w = x.r > 0
        a.scatter(x.mae[w], x.mfe[w], s=42, c=c, edgecolor=SURFACE, lw=1, label=f"{lab} 盈利")
        a.scatter(x.mae[~w], x.mfe[~w], s=36, facecolors="none", edgecolors=c, lw=1.3, label=f"{lab} 亏损")
    a.axhline(1, color=MUTED, lw=0.8, ls=(0, (2, 3)))
    a.set_title(f"{s} · 最大浮亏 vs 最大浮盈")
    a.set_xlabel("持仓期间最大浮亏 MAE（R）")
    a.set_ylabel("最大浮盈 MFE（R）")
    a.set_ylim(top=trades[s].mfe.max() * 1.35)  # headroom so the legend never covers a point
    a.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", columnspacing=1.2, handletextpad=0.3)
fig.tight_layout(h_pad=2)
fig.savefig("out/detail.png", dpi=150)
print("figures written: out/equity.png, out/detail.png")
