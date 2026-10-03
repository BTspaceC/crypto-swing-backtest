"""Stress tests of the ONE selected configuration. Nothing here is used to choose parameters.

  1. cost and execution-lag sensitivity
  2. mark-to-market drawdown (open positions marked every bar) vs closed-trade drawdown
  3. random-entry benchmark: same trend filter, funding filter, exits and trade frequency,
     but the entry bar is random — does the breakout signal add anything?
  4. position-sizing table: risk per trade vs drawdown odds, by block bootstrap

Writes out/robustness.md and out/random_entry.png.  Usage: python robustness.py [n_sims]
"""
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import swing_backtest as sb
from style import COLOR, INK, MUTED, SURFACE, SYMS

N_SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEL = sb.select(pd.read_csv("out/BTC_IS_grid.csv", index_col=0))
_, SIG, TREND, FF = sb.config(SEL)
rng = np.random.default_rng(7)
data = {s: sb.load_symbol(s) for s in SYMS}
md = [f"# 稳健性检验 · 配置 `{SEL}`", "",
      "本文件由 `robustness.py` 生成。所有检验只针对已选定的这一个配置，不用于重新挑选参数。", ""]


def table(df, fmt="{:+.2f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(fmt.format)
    head = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return [head, sep, *rows, ""]


def pooled(p=sb.P, signals=None):
    out = []
    for s in SYMS:
        t = sb.run(data[s], SIG, TREND, FF, p, None if signals is None else signals[s])
        t["sym"] = s
        out.append(t)
    return pd.concat(out).sort_values("exit_time").reset_index(drop=True)


def seg_avg(tr):
    is_, oos = sb.split(tr)
    return is_.r.mean(), oos.r.mean(), tr.r.mean()


base = pooled()

# ------------------------------------------------------------------ 1. costs and execution lag
rows = {}
for label, p in [
    ("基准：滑点 0.02%，下一根开盘成交", sb.P),
    ("滑点 0.05%", replace(sb.P, slip=0.0005)),
    ("滑点 0.10%", replace(sb.P, slip=0.0010)),
    ("滑点 0.20%", replace(sb.P, slip=0.0020)),
    ("手续费翻倍（单边 0.10%）", replace(sb.P, fee=0.0010)),
    ("晚一根 K 线成交", replace(sb.P, entry_delay=1)),
    ("晚一根 + 滑点 0.10%", replace(sb.P, entry_delay=1, slip=0.0010)),
]:
    t = pooled(p)
    a, b, c = seg_avg(t)
    rows[label] = dict(笔数=len(t), 样本内=a, 样本外=b, 全部=c, 总R=t.r.sum())
cost = pd.DataFrame(rows).T
cost["笔数"] = cost["笔数"].astype(int)
cost.index.name = "情景"
print(cost.round(3).to_string(), "\n")
md += ["## 1. 成本与执行延迟", "", "三币合并的平均 R。", "", *table(cost)]

# ------------------------------------------------------------------ 2. mark-to-market drawdown
rows, eqs = {}, []
for s in SYMS:
    t = base[base.sym == s]
    eq = sb.mtm_equity(data[s], t)
    eqs.append(eq)
    rows[s] = dict(按平仓结算=sb.stats(t)["maxDD"], 逐根收盘价=sb.max_drawdown(eq.close),
                   含盘中最低价=sb.max_drawdown(eq.close, eq.low))
idx = eqs[0].index.union(eqs[1].index).union(eqs[2].index)
tot = sum(e.reindex(idx).ffill().fillna(0) for e in eqs)
rows["三币合并"] = dict(按平仓结算=sb.stats(base)["maxDD"], 逐根收盘价=sb.max_drawdown(tot.close),
                    含盘中最低价=sb.max_drawdown(tot.close, tot.low))
dd = pd.DataFrame(rows).T
dd.index.name = "标的"
print(dd.round(2).to_string(), "\n")
md += ["## 2. 回撤：把持仓中的浮亏也算进去", "",
       "最大回撤，单位 R。「按平仓结算」只在交易结束时记账，会低估持仓途中的回吐。", "",
       *table(dd, "{:.1f}")]

# ------------------------------------------------------------------ 3. random-entry benchmark
elig = {s: (k[TREND] & ((k.last_fund.shift(-1) < sb.P.fund_cap) if FF else True)).values for s, k in data.items()}
actual_n = {s: int((base.sym == s).sum()) for s in SYMS}
prob = {}
for s in SYMS:  # calibrate the per-bar entry probability so the trade count matches
    p_ = actual_n[s] / elig[s].sum()
    for _ in range(4):
        n_ = np.mean([len(sb.run(data[s], SIG, TREND, FF, signals=elig[s] & (rng.random(len(elig[s])) < p_)))
                      for _ in range(12)])
        p_ *= actual_n[s] / n_
    prob[s] = p_

null = []
for _ in range(N_SIMS):
    t = pooled(signals={s: elig[s] & (rng.random(len(elig[s])) < prob[s]) for s in SYMS})
    null.append((*seg_avg(t), len(t)))
null = np.array(null)
act = seg_avg(base)
rows = {}
for i, seg in enumerate(("样本内", "样本外", "全部")):
    x = null[:, i]
    rows[seg] = {"突破入场": act[i], "随机入场均值": x.mean(), "随机入场 5%": np.percentile(x, 5),
                 "随机入场 95%": np.percentile(x, 95), "随机入场不差于突破的比例": (x >= act[i]).mean()}
rnd = pd.DataFrame(rows).T
rnd.index.name = "区间"
print(f"random-entry benchmark, {N_SIMS} sims, mean trades {null[:, 3].mean():.0f} vs actual {len(base)}")
print(rnd.round(3).to_string(), "\n")
share = rnd.pop("随机入场不差于突破的比例").map("{:.1%}".format)
out = rnd.map("{:+.2f}".format)
out["随机入场不差于突破的比例"] = share
md += ["## 3. 随机入场基准", "",
       f"保留日线趋势过滤、费率过滤、全部出场规则和交易频率，只把「突破」换成在符合条件的 K 线上随机入场，"
       f"模拟 {N_SIMS} 次（每次约 {null[:, 3].mean():.0f} 笔，实际 {len(base)} 笔）。三币合并的平均 R：", "",
       *table(out), "![随机入场基准](random_entry.png)", ""]

fig, ax = plt.subplots(figsize=(9, 4.2))
x = null[:, 2]
ax.hist(x, bins=40, color="#b9b8b2", edgecolor=SURFACE, lw=0.8)
ax.axvline(x.mean(), color=MUTED, lw=1.2, ls=(0, (3, 3)))
ax.axvline(act[2], color=COLOR["BTC"], lw=2.2)
top = ax.get_ylim()[1]
ax.text(x.mean(), top * 0.97, f" 随机入场均值 {x.mean():+.2f}R", color=MUTED, fontsize=9.5, va="top")
ax.text(act[2], top * 0.85, f" 突破入场 {act[2]:+.2f}R", color=INK, fontsize=9.5, va="top", fontweight="bold")
ax.set_title(f"同样的过滤器和出场规则，把入场换成随机 · {N_SIMS} 次模拟 · 三币合并全样本")
ax.set_xlabel("平均 R")
ax.set_ylabel("模拟次数")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig("out/random_entry.png", dpi=150)

# ------------------------------------------------------------------ 4. sizing vs drawdown odds
HORIZON, N_BOOT = 100, 5000


def paths(tr):
    """Sequences of HORIZON trade results, built from whole calendar quarters in random order."""
    blocks = [v.r.values for _, v in tr.sort_values("entry_time").groupby(tr.entry_time.dt.to_period("Q"))]
    out = np.empty((N_BOOT, HORIZON))
    for b in range(N_BOOT):
        seq = []
        while len(seq) < HORIZON:
            seq.extend(blocks[rng.integers(len(blocks))])
        out[b] = seq[:HORIZON]
    return out


def sizing(tr):
    seq = paths(tr)
    rows = {}
    for f in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15):
        eq = np.cumprod(1 + f * seq, axis=1)
        peak = np.maximum.accumulate(np.concatenate([np.ones((N_BOOT, 1)), eq], axis=1), axis=1)[:, 1:]
        mdd = (1 - eq / peak).max(axis=1)
        rows[f"{f:.1%}"] = {"回撤超 25% 的概率": (mdd >= 0.25).mean(), "回撤超 50% 的概率": (mdd >= 0.50).mean(),
                            "期末净值中位数": np.median(eq[:, -1]), "期末净值 5% 分位": np.percentile(eq[:, -1], 5)}
    df = pd.DataFrame(rows).T
    df.index.name = "每笔风险占净值"
    return df


def fmt_sizing(df):
    out = pd.DataFrame(index=df.index)
    for c in df.columns[:2]:
        out[c] = df[c].map("{:.0%}".format)
    for c in df.columns[2:]:
        out[c] = df[c].map("{:.2f}×".format)
    return out


md += ["## 4. 每笔风险与回撤概率", "",
       f"固定比例下注，连续做 {HORIZON} 笔，按季度整块重抽样 {N_BOOT} 次。"
       "前提是未来的单笔结果分布与所用样本相同，这个前提本身没有被证实。", ""]
for label, tr in (("用全部 229 笔历史交易", base), ("只用 2024 年以来的交易", sb.split(base)[1])):
    df = sizing(tr)
    print(label)
    print(df.round(3).to_string(), "\n")
    md += [f"**{label}**", "", *table(fmt_sizing(df))]

open("out/robustness.md", "w", encoding="utf-8", newline="\n").write("\n".join(md))
print("written: out/robustness.md, out/random_entry.png")
