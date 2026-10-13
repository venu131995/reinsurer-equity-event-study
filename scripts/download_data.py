"""Download prices (Yahoo Finance) and the 10-year Bund zero rate (ECB) into data/."""
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import yfinance as yf

D = Path(__file__).resolve().parents[1] / "data"
D.mkdir(exist_ok=True)
t = ["SREN.SW", "MUV2.DE", "HNR1.DE", "SCR.PA", "^STOXX", "EXH5.DE", "CHFEUR=X", "EURUSD=X", "CHF=X"]
px = yf.download(t, start="2003-01-01", auto_adjust=True, progress=False)["Close"]
# CHF/EUR starts in 2005 on Yahoo: fill earlier days from EUR/USD and USD/CHF (agree to 0.02 % on overlap)
px["CHFEUR=X"] = px["CHFEUR=X"].fillna(1 / (px["EURUSD=X"] * px["CHF=X"]))
px.to_csv(D / "prices_adjusted.csv")
rows = urlopen("https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y?format=csvdata")
(D / "ecb_SR_10Y.csv").write_text("\n".join(",".join(r.split(",")[8:10]) for r in rows.read().decode().splitlines()))
print("data written to", D)
