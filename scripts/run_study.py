"""Study: TSR vs peers, drivers of the share price, catastrophe event study with placebo validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reins import (EVENTS, PEERS, attribution_by_year, drawdowns, event_study, load, ols, perf_table,  # noqa: E402
                   placebo, rolling_beta)
from reins.analysis import MARKET, SECTOR  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
COL = {"SREN.SW": "tab:red", "MUV2.DE": "tab:blue", "HNR1.DE": "tab:green", "SCR.PA": "tab:orange",
       MARKET: "k", SECTOR: "tab:gray"}


def box(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(1.4)
    ax.tick_params(width=1.4)


r = load(ROOT / "data")
out = {"period": [str(r.index[0].date()), str(r.index[-1].date())], "days": len(r)}

# 1. performance since 2005 (EUR, dividends reinvested) and since 2015
perf = perf_table(r, list(PEERS) + [MARKET])
perf10 = perf_table(r.loc["2016-01-01":], list(PEERS) + [MARKET, SECTOR])
out["performance_since_2005"] = json.loads(perf.to_json(orient="index", date_format="iso"))
out["performance_since_2016"] = json.loads(perf10.to_json(orient="index", date_format="iso"))
out["swiss_re_drawdowns"] = drawdowns(r["SREN.SW"])

# 2. drivers: full-sample betas with Newey-West standard errors
drv = {}
for y in PEERS:
    b, se, e = ols(r[y].to_numpy(), r[[MARKET, "dy10"]].to_numpy(), hac_lags=10)
    tot = r[y].var()
    drv[PEERS[y]] = {"beta_market": b[1], "se_market": se[1], "beta_rate_per_100bp_%": b[2],
                     "se_rate_%": se[2], "R2": 1 - e.var() / tot}
out["drivers"] = drv
att = attribution_by_year(r, "SREN.SW")
out["swiss_re_attribution_by_year_%"] = json.loads((100 * att).round(2).to_json(orient="index"))
out["attribution_max_closure_error"] = float((att["market"] + att["rates"] + att["stock-specific"] - att["total"]).abs().max())
rb = {y: rolling_beta(r, y) for y in PEERS}

# 3. event study
ev5 = event_study(r, window=(0, 5))
ev_pre = event_study(r, window=(-5, 5))
ev20 = event_study(r, window=(0, 20))
for name, ev in [("car_0_5", ev5), ("car_m5_5", ev_pre), ("car_0_20", ev20)]:
    g = ev.groupby("stock")["car_%"]
    out[f"event_{name}"] = {"mean_by_stock_%": g.mean().round(3).to_dict(),
                            "share_negative": g.apply(lambda s: float((s < 0).mean())).to_dict(),
                            "significant_|t|>1.96": int((ev["t"].abs() > 1.96).sum()), "n": len(ev),
                            "pooled_mean_%": float(ev["car_%"].mean()),
                            "pooled_t": float(ev["car_%"].mean() / (ev["car_%"].std() / np.sqrt(len(ev))))}
ev5.to_csv(RES / "event_car_0_5.csv", index=False)
ev_pre.to_csv(RES / "event_car_m5_5.csv", index=False)

# 4. validation of the event-study test on random dates
out["placebo_size"] = placebo(r, n=2000, seed=1)
out["placebo_power_-3%"] = placebo(r, n=2000, seed=2, inject=-0.03)
out["placebo_power_-5%"] = placebo(r, n=2000, seed=3, inject=-0.05)
(RES / "summary.json").write_text(json.dumps(out, indent=1, default=str))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(11, 4.6))
for c in list(PEERS) + [MARKET]:
    x = r[c]
    ax.plot(np.exp(x.cumsum()) * 100, color=COL[c], lw=2 if c == "SREN.SW" else 1.4,
            label=PEERS.get(c, "STOXX Europe 600"))
ax.set_yscale("log"); ax.set_ylabel("total return index (2005 = 100, EUR, log scale)")
ax.set_title("Total shareholder return, dividends reinvested"); ax.legend(frameon=True, ncol=5); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig1_tsr.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.6))
att100 = 100 * att
bottom_p, bottom_n = np.zeros(len(att)), np.zeros(len(att))
for k, c in [("market", "tab:blue"), ("rates", "tab:purple"), ("stock-specific", "tab:gray")]:
    v = att100[k].to_numpy()
    ax.bar(att.index, np.where(v > 0, v, 0), bottom=bottom_p, color=c, edgecolor="k", label=k)
    ax.bar(att.index, np.where(v < 0, v, 0), bottom=bottom_n, color=c, edgecolor="k")
    bottom_p += np.where(v > 0, v, 0); bottom_n += np.where(v < 0, v, 0)
ax.plot(att.index, att100["total"], "D", color="tab:red", ms=7, label="total (log return)")
ax.axhline(0, color="k", lw=1); ax.set_ylabel("% (log return)")
ax.set_title("What moved Swiss Re's share price each year"); ax.legend(frameon=True, ncol=4); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig2_attribution.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for y in PEERS:
    ax[0].plot(rb[y]["beta_market"], color=COL[y], lw=1.8, label=PEERS[y])
    ax[1].plot(rb[y]["beta_rate_per_100bp"] * 100, color=COL[y], lw=1.8, label=PEERS[y])
ax[0].set_title("Rolling 2-year market beta"); ax[0].set_ylabel("beta to STOXX Europe 600")
ax[1].set_title("Rolling 2-year rate sensitivity"); ax[1].set_ylabel("% share return per +100 bp Bund 10y")
ax[1].axhline(0, color="k", lw=1)
for a in ax:
    a.legend(frameon=True, fontsize=9); box(a)
fig.tight_layout(); fig.savefig(FIG / "fig3_rolling_betas.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 5))
piv = ev_pre.pivot_table(index="event", columns="stock", values="car_%")
order = sorted(EVENTS, key=lambda k: EVENTS[k])
piv = piv.reindex(order)
im = ax.imshow(piv.to_numpy(), cmap="jet_r", vmin=-15, vmax=15, aspect="auto")
ax.set_yticks(range(len(piv))); ax.set_yticklabels([f"{e} ({EVENTS[e][:4]})" for e in piv.index])
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.iloc[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9, fontweight="bold")
cb = fig.colorbar(im, ax=ax); cb.set_label("abnormal return, days -5..+5 (%)", fontweight="bold")
ax.set_title("Catastrophe event study: cumulative abnormal returns"); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig4_event_heatmap.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2))
pl = []
for d in [str(EVENTS[k]) for k in order]:
    i = r.index.searchsorted(pd.Timestamp(d))
    if i < 240 or i + 21 > len(r):
        continue
    for y in PEERS:
        w = r.iloc[i - 240:i - 20]
        b, _, _ = ols(w[y].to_numpy(), w[[MARKET]].to_numpy())
        ev = r.iloc[i - 10:i + 21]
        pl.append(np.cumsum(ev[y].to_numpy() - b[0] - b[1] * ev[MARKET].to_numpy()))
pl = 100 * np.array(pl)
days = np.arange(-10, 21)
m, s = pl.mean(0), pl.std(0) / np.sqrt(len(pl))
ax.fill_between(days, m - 1.96 * s, m + 1.96 * s, color="tab:red", alpha=0.25, label="95 % band")
ax.plot(days, m, color="tab:red", lw=2.5, label=f"mean of {len(pl)} event-stock pairs")
ax.axvline(0, color="k", ls="--"); ax.axhline(0, color="k", lw=1)
ax.set_xlabel("trading days from event"); ax.set_ylabel("cumulative abnormal return (%)")
ax.set_title("Average path around catastrophes"); ax.legend(frameon=True); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig5_event_path.png", dpi=160); plt.close(fig)
print(json.dumps({k: out[k] for k in out if k.startswith(("event", "placebo", "drivers", "attribution"))},
                 indent=1, default=str))
print(perf.round(2).to_string()); print(perf10.round(2).to_string()); print(out["swiss_re_drawdowns"])
