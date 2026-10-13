"""Data loading and statistics for the reinsurer study (all returns are daily log returns in EUR)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PEERS = {"SREN.SW": "Swiss Re", "MUV2.DE": "Munich Re", "HNR1.DE": "Hannover Re", "SCR.PA": "SCOR"}
MARKET, SECTOR = "^STOXX", "EXH5.DE"

# Major insured catastrophes; day 0 = first European trading day on or after the date (landfall/occurrence).
EVENTS = {
    "Katrina": "2005-08-29", "Christchurch EQ": "2011-02-22", "Tohoku EQ/tsunami": "2011-03-11",
    "Sandy": "2012-10-29", "Harvey": "2017-08-25", "Irma": "2017-09-10", "Maria": "2017-09-20",
    "Michael": "2018-10-10", "Europe floods": "2021-07-14", "Ida": "2021-08-29", "Ian": "2022-09-28",
    "Turkey-Syria EQ": "2023-02-06", "Helene": "2024-09-26", "Milton": "2024-10-09", "LA wildfires": "2025-01-07",
}


def load(data_dir: Path) -> pd.DataFrame:
    """Daily log returns in EUR for the four reinsurers, the STOXX Europe 600 and the insurance ETF, plus the
    daily change of the 10-year Bund zero rate (decimal). Swiss Re (CHF) is converted with CHF/EUR."""
    px = pd.read_csv(data_dir / "prices_adjusted.csv", index_col=0, parse_dates=True)
    px = px[px.index.dayofweek < 5]
    px["SREN.SW"] = px["SREN.SW"] * px["CHFEUR=X"].ffill()
    cols = list(PEERS) + [MARKET, SECTOR]
    lr = np.log(px[cols]).diff()
    b10 = pd.read_csv(data_dir / "ecb_SR_10Y.csv", index_col=0, parse_dates=True).iloc[:, 0] / 100
    lr["dy10"] = b10.reindex(lr.index).ffill().diff()
    return lr.loc["2004-04-27":].dropna(subset=list(PEERS) + [MARKET, "dy10"], how="any")


def ols(y: np.ndarray, X: np.ndarray, hac_lags: int = 0):
    """OLS with intercept; returns coefficients and (Newey-West if hac_lags>0) standard errors."""
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    if hac_lags == 0:
        s2 = e @ e / (len(y) - X.shape[1])
        V = s2 * XtX_inv
    else:
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for L in range(1, hac_lags + 1):
            w = 1 - L / (hac_lags + 1)
            G = Xe[L:].T @ Xe[:-L]
            S += w * (G + G.T)
        V = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.diag(V)), e


def perf_table(r: pd.DataFrame, cols) -> pd.DataFrame:
    rows = {}
    for c in cols:
        x = r[c].dropna()
        lvl = np.exp(x.cumsum())
        dd = lvl / lvl.cummax() - 1
        ann = x.sum() / (len(x) / 252)
        vol = x.std() * np.sqrt(252)
        rows[c] = {"start": x.index[0].date(), "TSR_total_%": 100 * (lvl.iloc[-1] - 1),
                   "TSR_annual_%": 100 * (np.exp(ann) - 1), "vol_%": 100 * vol, "return/vol": ann / vol,
                   "max_drawdown_%": 100 * dd.min(), "max_dd_date": dd.idxmin().date()}
    return pd.DataFrame(rows).T


def drawdowns(x: pd.Series, top: int = 3) -> list[dict]:
    """Largest peak-to-trough episodes with recovery dates (None if not yet recovered)."""
    lvl = np.exp(x.cumsum())
    out, used = [], pd.Series(False, index=lvl.index)
    peak = lvl.cummax()
    dd = lvl / peak - 1
    for _ in range(top):
        trough = dd[~used].idxmin()
        p = lvl[:trough].idxmax()
        rec = lvl[trough:][lvl[trough:] >= lvl[p]]
        r = rec.index[0] if len(rec) else None
        out.append({"peak": p.date(), "trough": trough.date(), "depth_%": 100 * dd[trough],
                    "recovered": r.date() if r is not None else None})
        used[p:(r or lvl.index[-1])] = True
    return out


def rolling_beta(r: pd.DataFrame, y: str, window: int = 504) -> pd.DataFrame:
    """Rolling 2-year betas of y on the market and on the 10-year Bund yield change."""
    out = []
    for i in range(window, len(r) + 1, 21):
        w = r.iloc[i - window:i]
        b, _, _ = ols(w[y].to_numpy(), w[[MARKET, "dy10"]].to_numpy())
        out.append((w.index[-1], b[1], b[2] / 100))  # rate beta per 1 bp -> per 100 bp below
    return pd.DataFrame(out, columns=["date", "beta_market", "beta_rate_per_100bp"]).set_index("date")


def attribution_by_year(r: pd.DataFrame, y: str) -> pd.DataFrame:
    """Split each calendar year's log return into market, rates and stock-specific parts using full-sample
    betas. Log returns add, so the three parts sum exactly to the total."""
    b, _, _ = ols(r[y].to_numpy(), r[[MARKET, "dy10"]].to_numpy())
    parts = pd.DataFrame({"market": b[1] * r[MARKET], "rates": b[2] * r["dy10"]}, index=r.index)
    parts["stock-specific"] = r[y] - parts["market"] - parts["rates"]
    g = parts.groupby(parts.index.year).sum()
    g["total"] = r[y].groupby(r.index.year).sum()
    return g


def car(r: pd.DataFrame, y: str, day0: int, window=(0, 5), est=(-240, -21)) -> dict:
    """Cumulative abnormal return of y around index position day0 (market model on the estimation window)."""
    a, b = day0 + est[0], day0 + est[1]
    if a < 0 or day0 + window[1] >= len(r):
        return None
    w = r.iloc[a:b + 1]
    beta, _, e = ols(w[y].to_numpy(), w[[MARKET]].to_numpy())
    ev = r.iloc[day0 + window[0]:day0 + window[1] + 1]
    ar = ev[y].to_numpy() - beta[0] - beta[1] * ev[MARKET].to_numpy()
    sd = e.std(ddof=2)
    L = len(ar)
    return {"car": float(ar.sum()), "t": float(ar.sum() / (sd * np.sqrt(L))), "ar": ar}


def event_study(r: pd.DataFrame, events=EVENTS, window=(0, 5)) -> pd.DataFrame:
    rows = []
    for name, d in events.items():
        i = r.index.searchsorted(pd.Timestamp(d))
        for y in PEERS:
            c = car(r, y, i, window)
            if c:
                rows.append({"event": name, "date": r.index[i].date(), "stock": PEERS[y], "car_%": 100 * c["car"],
                             "t": c["t"]})
    return pd.DataFrame(rows)


def placebo(r: pd.DataFrame, n: int = 2000, window=(0, 5), seed: int = 0, inject: float = 0.0,
            exclude: int = 30) -> dict:
    """Random non-event dates: size of the t-test (reject |t|>1.96 ~5 %) and, with an injected abnormal
    return `inject` spread over the window, its power."""
    rng = np.random.default_rng(seed)
    ev_idx = [r.index.searchsorted(pd.Timestamp(d)) for d in EVENTS.values()]
    ok = np.ones(len(r), bool)
    ok[:241] = False
    ok[-window[1] - 1:] = False
    for i in ev_idx:
        ok[max(0, i - exclude):i + exclude] = False
    days = rng.choice(np.flatnonzero(ok), n)
    stocks = rng.choice(list(PEERS), n)
    rej, cars = 0, []
    for d, y in zip(days, stocks):
        rr = r
        if inject:
            rr = r.copy()
            L = window[1] - window[0] + 1
            rr.iloc[d + window[0]:d + window[1] + 1, rr.columns.get_loc(y)] += inject / L
        c = car(rr, y, d, window)
        cars.append(c["car"])
        rej += abs(c["t"]) > 1.96
    return {"n": n, "reject_rate": rej / n, "mean_car_%": 100 * float(np.mean(cars)),
            "sd_car_%": 100 * float(np.std(cars))}
