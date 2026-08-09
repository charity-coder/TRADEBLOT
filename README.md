# FX Signal Dial — Streamlit app + MT5 indicator

## 1. `app.py` (Streamlit — unchanged workflow, more pairs)

Added 7 more pairs to the sidebar dropdown alongside the original 4:
**USDCAD, USDCHF, NZDUSD** (majors), plus crosses **EURGBP, EURJPY, GBPJPY,
AUDJPY, EURCHF, CADJPY, CHFJPY**. Pip size auto-detects any `*JPY` pair
(0.01) vs everything else (0.0001) — no other code changes needed to add
even more pairs later; just add an entry to the `PAIRS` dict with a
`base` price and rough daily `vol`.

Run as before:
```
streamlit run app.py
```

## 2. `CompositeSignalEngine.mq5` (MetaTrader 5 custom indicator)

Python and MT5 are different platforms/languages — Streamlit can't run
*inside* MT5, so this is a native **MQL5 port** of the same composite-score
math (SMA20/50 trend, RSI14 momentum, MACD histogram, Bollinger position),
built as a real MT5 custom indicator you attach to any chart.

**Install:**
1. Open MetaTrader 5 → *File → Open Data Folder* → `MQL5/Indicators/`.
2. Copy `CompositeSignalEngine.mq5` into that folder.
3. Open **MetaEditor** (F4 in MT5), open the file, click **Compile**
   (F7). It should compile with no errors and produce a `.ex5` file next
   to it.
4. Back in MT5, refresh the **Navigator** panel (right-click → Refresh)
   and drag **Composite Signal Engine** from *Indicators → Custom* onto
   any chart.

**What it does:**
- Plots the same −100…+100 composite score in a sub-window, color-coded
  red (SELL zone, below −25), gray (HOLD), green (BUY zone, above +25).
- Dotted level lines at +25 / 0 / −25 mark the same thresholds used in
  the Streamlit backtester.
- Optional `Alert()` pop-up when the score crosses a threshold on bar
  close (toggle with the `InpEnableAlerts` input; `InpAlertOncePerBar`
  prevents repeat alerts on the same bar).
- All periods (SMA/RSI/MACD/Bollinger) and the threshold are exposed as
  inputs, so you can tune it per pair/timeframe without touching code.
- Works on **any symbol MT5 has data for** — forex, indices, crypto,
  whatever your broker lists — since it reads straight from the chart's
  own price feed rather than synthetic data.

This is an educational technical-analysis tool, not financial advice or
a signal guarantee — same disclaimer as the Streamlit app.
