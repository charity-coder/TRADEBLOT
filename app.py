"""FX Signal Dial — Streamlit port of the React "Composite Signal Engine" dashboard."""

import io
import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Tesseract OCR setup — works both on a local Windows machine and on
# Streamlit Community Cloud (Linux, where packages.txt installs
# tesseract-ocr onto the system PATH).
# ---------------------------------------------------------------------------
def find_tesseract():
    # Explicit override always wins, if set.
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd.strip().strip('"')):
        return env_cmd.strip().strip('"')

    # Common Windows install locations.
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    ]
    for path in windows_paths:
        if os.path.isfile(path):
            return path

    # Common Linux locations (Streamlit Cloud, most distros).
    linux_paths = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]
    for path in linux_paths:
        if os.path.isfile(path):
            return path

    # Windows registry, if available.
    try:
        import winreg
        for root, subkey in [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Tesseract-OCR"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Tesseract-OCR"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Tesseract-OCR"),
        ]:
            try:
                with winreg.OpenKey(root, subkey) as key:
                    install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                    candidate = os.path.join(install_dir, "tesseract.exe")
                    if os.path.isfile(candidate):
                        return candidate
            except OSError:
                pass
    except ImportError:
        pass

    # Fall back to whatever's on PATH.
    return shutil.which("tesseract")


TESSERACT_PATH = find_tesseract()

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    try:
        TESSERACT_VERSION = str(pytesseract.get_tesseract_version())
        OCR_AVAILABLE = True
    except Exception:
        OCR_AVAILABLE = False
        TESSERACT_VERSION = "Unable to run Tesseract"
else:
    OCR_AVAILABLE = False
    TESSERACT_VERSION = "Tesseract not found"


# ---------------------------------------------------------------------------
# Design tokens (kept close to the original React version)
# ---------------------------------------------------------------------------
BG = "#0A0E12"
PANEL = "#12181F"
PANEL_2 = "#171F29"
BULL = "#33C9A0"
BEAR = "#E8604C"
GOLD = "#D4A24C"
TEXT = "#E8EDF2"
MUTED = "#7C8A99"
GRIDLINE = "#1B222C"
BORDER = "#232B36"

# ---------------------------------------------------------------------------
# Tradable pairs. "vol" is the rough daily-move scale used only to seed the
# synthetic sample-data generator (it has no effect once you upload your own
# CSV/PDF/image data). Pairs are grouped: majors first, then commonly traded
# crosses, so the sidebar selector reads naturally.
# ---------------------------------------------------------------------------
PAIRS = {
    # Majors
    "EURUSD": {"name": "EUR / USD", "base": 1.0850, "vol": 0.0009},
    "GBPUSD": {"name": "GBP / USD", "base": 1.2680, "vol": 0.0012},
    "USDJPY": {"name": "USD / JPY", "base": 156.20, "vol": 0.14},
    "AUDUSD": {"name": "AUD / USD", "base": 0.6520, "vol": 0.0008},
    "USDCAD": {"name": "USD / CAD", "base": 1.3690, "vol": 0.0010},
    "USDCHF": {"name": "USD / CHF", "base": 0.8830, "vol": 0.0009},
    "NZDUSD": {"name": "NZD / USD", "base": 0.5980, "vol": 0.0009},
    # Common crosses
    "EURGBP": {"name": "EUR / GBP", "base": 0.8560, "vol": 0.0006},
    "EURJPY": {"name": "EUR / JPY", "base": 169.50, "vol": 0.16},
    "GBPJPY": {"name": "GBP / JPY", "base": 198.00, "vol": 0.19},
    "AUDJPY": {"name": "AUD / JPY", "base": 101.90, "vol": 0.13},
    "EURCHF": {"name": "EUR / CHF", "base": 0.9580, "vol": 0.0006},
    "CADJPY": {"name": "CAD / JPY", "base": 114.10, "vol": 0.12},
    "CHFJPY": {"name": "CHF / JPY", "base": 176.90, "vol": 0.15},
}


# ---------------------------------------------------------------------------
# Synthetic OHLC generator (same linear-congruential PRNG as the JS version,
# so results are reproducible from a seed)
# ---------------------------------------------------------------------------
def generate_series(pair: dict, n: int = 260, seed: int = 42) -> pd.DataFrame:
    s = seed

    def rand():
        nonlocal s
        s = (s * 9301 + 49297) % 233280
        return s / 233280

    base, vol0 = pair["base"], pair["vol"]
    price, vol = base, vol0
    rows = []
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=n)

    for i in range(n):
        vol = max(vol0 * 0.4, vol * 0.96 + vol0 * 0.06 + (rand() - 0.5) * vol0 * 0.15)
        drift = np.sin(i / 27) * vol0 * 0.25
        shock = (rand() - 0.5) * 2 * vol
        price = price + drift + shock
        open_ = price - shock * 0.4
        close = price
        high = max(open_, close) + abs(shock) * 0.6 * rand()
        low = min(open_, close) - abs(shock) * 0.6 * rand()
        rows.append(
            {
                "date": (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Indicator math
# ---------------------------------------------------------------------------
def sma(values: pd.Series, period: int) -> pd.Series:
    return values.rolling(period).mean()


def ema(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(span=period, adjust=False).mean()


def rsi(values: pd.Series, period: int = 14) -> pd.Series:
    delta = values.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[avg_loss == 0] = 100
    return out


def macd(values: pd.Series, fast: int = 12, slow: int = 26, signal_p: int = 9):
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    line = ema_fast - ema_slow
    signal = ema(line, signal_p)
    hist = line - signal
    return line, signal, hist


def bollinger(values: pd.Series, period: int = 20, mult: float = 2):
    mid = sma(values, period)
    sd = values.rolling(period).std(ddof=0)
    upper = mid + mult * sd
    lower = mid - mult * sd
    return mid, upper, lower


# ---------------------------------------------------------------------------
# Composite signal engine
# ---------------------------------------------------------------------------
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["sma20"] = sma(close, 20)
    df["sma50"] = sma(close, 50)
    df["rsi"] = rsi(close, 14)
    df["macdLine"], df["macdSignal"], df["macdHist"] = macd(close)
    df["bbMid"], df["bbUpper"], df["bbLower"] = bollinger(close, 20, 2)

    scores = np.zeros(len(df))
    votes = np.zeros(len(df))

    # Trend: SMA20 vs SMA50 crossover
    mask = df["sma20"].notna() & df["sma50"].notna()
    diff = (df["sma20"] - df["sma50"]) / df["sma50"]
    scores[mask] += np.clip(diff[mask] * 200, -1, 1)
    votes[mask] += 1

    # Momentum: RSI
    mask = df["rsi"].notna()
    scores[mask] += np.clip((df["rsi"][mask] - 50) / 25, -1, 1)
    votes[mask] += 1

    # MACD histogram
    mask = df["macdHist"].notna()
    norm = df["macdHist"] / (close.abs() * 0.002 + 1e-9)
    scores[mask] += np.clip(norm[mask], -1, 1)
    votes[mask] += 1

    # Bollinger position (mild mean-reversion bias)
    mask = df["bbUpper"].notna()
    width = df["bbUpper"] - df["bbLower"]
    pos = np.where(width > 0, (close - df["bbMid"]) / (width / 2), 0)
    scores[mask] += np.clip(-pos[mask] * 0.6, -1, 1)
    votes[mask] += 1

    df["composite"] = np.where(votes > 0, (scores / np.where(votes == 0, 1, votes)) * 100, 0)
    return df


def signal_from_score(score):
    if score is None or pd.isna(score):
        return "—", "muted"
    if score > 25:
        return "BUY", "bull"
    if score < -25:
        return "SELL", "bear"
    return "HOLD", "neutral"


def pip_size(pair_id: str) -> float:
    return 0.01 if pair_id.endswith("JPY") else 0.0001


# ---------------------------------------------------------------------------
# File extraction: CSV / PDF (pdfplumber tables + text) / JPG / PNG (OCR)
#
# All paths funnel into a raw table of rows with loose "date"/"open"/
# "high"/"low"/"close" fields, which the user reviews and edits in the UI
# before it's cleaned and handed to the indicator/backtest pipeline.
# ---------------------------------------------------------------------------
DATE_RE = re.compile(
    r"^\d{4}-\d{1,2}-\d{1,2}$|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\d{4}/\d{1,2}/\d{1,2}$"
)


def _to_float(token):
    try:
        return float(str(token).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def parse_ohlc_text(text):
    """Best-effort line-by-line parse of loose tabular text (from OCR or a
    PDF's raw text layer) into date/open/high/low/close rows."""
    rows = []
    for line in text.splitlines():
        tokens = [t for t in re.split(r"[,\t]+|\s{1,}", line.strip()) if t]
        if len(tokens) < 2:
            continue

        date_tok, date_idx = None, None
        for i, t in enumerate(tokens):
            if DATE_RE.match(t):
                date_tok, date_idx = t, i
                break
        if date_tok is None:
            continue

        nums = [n for n in (_to_float(t) for t in tokens[date_idx + 1:]) if n is not None]
        if len(nums) >= 4:
            o, h, l, c = nums[0], nums[1], nums[2], nums[3]
        elif len(nums) == 1:
            o = h = l = c = nums[0]
        else:
            continue
        rows.append({"date": date_tok, "open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])


def _table_to_df(table):
    """Map a pdfplumber-extracted table (list of row-lists) onto OHLC
    columns using its header row, if it has one we recognize."""
    if not table or len(table) < 2:
        return None
    header = [str(c or "").strip().lower() for c in table[0]]
    col_idx = {}
    for name in ["date", "open", "high", "low", "close"]:
        for i, h in enumerate(header):
            if name in h:
                col_idx[name] = i
                break
    if "date" not in col_idx or "close" not in col_idx:
        return None

    rows = []
    for r in table[1:]:
        try:
            date = str(r[col_idx["date"]] or "").strip()
            close = _to_float(r[col_idx["close"]])
            if not date or close is None:
                continue
            open_ = _to_float(r[col_idx["open"]]) if "open" in col_idx else close
            high = _to_float(r[col_idx["high"]]) if "high" in col_idx else None
            low = _to_float(r[col_idx["low"]]) if "low" in col_idx else None
            open_ = open_ if open_ is not None else close
            high = high if high is not None else max(open_, close)
            low = low if low is not None else min(open_, close)
            rows.append({"date": date, "open": open_, "high": high, "low": low, "close": close})
        except IndexError:
            continue
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"]) if rows else None


def extract_from_pdf(file_bytes):
    """Try structured tables first (pdfplumber), then fall back to parsing
    the raw text layer. Returns (DataFrame, method_str)."""
    table_frames, text_chunks = [], []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                df = _table_to_df(table)
                if df is not None and not df.empty:
                    table_frames.append(df)
            text_chunks.append(page.extract_text() or "")

    if table_frames:
        return pd.concat(table_frames, ignore_index=True), "structured table"

    text_df = parse_ohlc_text("\n".join(text_chunks))
    if not text_df.empty:
        return text_df, "text layer"

    return pd.DataFrame(columns=["date", "open", "high", "low", "close"]), "none"


def extract_from_image(file_bytes):
    """OCR a JPG/PNG and parse whatever tabular text comes out."""
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "Tesseract OCR is not installed or could not be found. "
            "Install Tesseract OCR, then restart this Streamlit app. "
            "You can also set TESSERACT_CMD to the full path of tesseract.exe."
        )

    try:
        img = Image.open(io.BytesIO(file_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Upscaling small screenshots helps Tesseract read tight price/date
        # text more reliably.
        if img.width < 1800:
            scale = 1800 / img.width
            img = img.resize((1800, int(img.height * scale)), Image.Resampling.LANCZOS)

        # PSM 6 treats the image as a uniform block of text, which works well
        # for screenshots containing OHLC/date columns.
        text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")

        # If the first pass finds nothing, retry as sparse text.
        if not text.strip():
            text = pytesseract.image_to_string(img, config="--oem 3 --psm 11")
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(f"Tesseract executable could not be run at: {TESSERACT_PATH}") from e
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}") from e

    return parse_ohlc_text(text), "OCR"


def clean_ohlc_df(df):
    """Validate/normalize a user-reviewed extraction into the canonical
    date/open/high/low/close frame the pipeline expects, or None if it
    can't be made usable."""
    if df is None or df.empty or not {"date", "open", "high", "low", "close"}.issubset(df.columns):
        return None
    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["_date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["_date", "close"])
    if out.empty:
        return None
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out[["open", "close"]].max(axis=1))
    out["low"] = out["low"].fillna(out[["open", "close"]].min(axis=1))
    out = out.sort_values("_date")
    out["date"] = out["_date"].dt.strftime("%Y-%m-%d")
    return out[["date", "open", "high", "low", "close"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Backtester: signal entries, intrabar stop-loss / take-profit exits,
# risk-based position sizing (risk a fixed % of equity per trade)
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    curve: pd.DataFrame
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    final_equity: float = 10000.0
    return_pct: float = 0.0
    stopped_out: int = 0
    target_hit: int = 0
    signal_exits: int = 0
    risk_reward: float = None


def backtest(df: pd.DataFrame, pair_id: str, threshold=25, risk_pct=1, sl_pips=30, tp_pips=60) -> BacktestResult:
    pip = pip_size(pair_id)
    sl_dist = sl_pips * pip
    tp_dist = tp_pips * pip

    position = 0
    equity = 10000.0
    entry_price = stop_price = target_price = units = None
    curve_dates, curve_equity = [], []
    wins = losses = 0
    gross_profit = gross_loss = 0.0
    entries = stopped_out = target_hit = signal_exits = 0
    peak = equity
    max_dd = 0.0

    rows = df.to_dict("records")

    def close_trade(exit_price):
        nonlocal position, equity, entry_price, wins, losses, gross_profit, gross_loss
        pnl = (exit_price - entry_price) * position * units
        equity += pnl
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            losses += 1
            gross_loss += -pnl
        position = 0
        entry_price = None

    for i in range(1, len(rows)):
        row = rows[i]

        if position != 0:
            exit_price = None
            reason = None
            if position == 1:
                if row["low"] <= stop_price:
                    exit_price, reason = stop_price, "stop"
                elif row["high"] >= target_price:
                    exit_price, reason = target_price, "target"
            else:
                if row["high"] >= stop_price:
                    exit_price, reason = stop_price, "stop"
                elif row["low"] <= target_price:
                    exit_price, reason = target_price, "target"
            if exit_price is not None:
                close_trade(exit_price)
                if reason == "stop":
                    stopped_out += 1
                else:
                    target_hit += 1

        prev_score = rows[i - 1]["composite"]
        desired_pos = 1 if prev_score > threshold else (-1 if prev_score < -threshold else 0)

        if position != 0 and desired_pos != position:
            close_trade(row["open"])
            signal_exits += 1

        if position == 0 and desired_pos != 0:
            entry_price = row["open"]
            stop_price = entry_price - sl_dist if desired_pos == 1 else entry_price + sl_dist
            target_price = entry_price + tp_dist if desired_pos == 1 else entry_price - tp_dist
            risk_amount = equity * (risk_pct / 100)
            units = risk_amount / sl_dist if sl_dist > 0 else 0
            position = desired_pos
            entries += 1

        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak else 0)
        curve_dates.append(row["date"])
        curve_equity.append(equity)

    total_trades = wins + losses
    return BacktestResult(
        curve=pd.DataFrame({"date": curve_dates, "equity": curve_equity}),
        trades=total_trades,
        win_rate=(wins / total_trades * 100) if total_trades else 0.0,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        max_drawdown=max_dd * 100,
        final_equity=equity,
        return_pct=(equity - 10000) / 10000 * 100,
        stopped_out=stopped_out,
        target_hit=target_hit,
        signal_exits=signal_exits,
        risk_reward=(tp_pips / sl_pips) if sl_pips > 0 else None,
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def signal_dial(score):
    score = 0 if score is None or pd.isna(score) else score
    clamped = max(-100, min(100, score))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clamped,
            number={"suffix": "", "font": {"color": TEXT, "size": 30}},
            gauge={
                "axis": {"range": [-100, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                "bar": {"color": GOLD, "thickness": 0.25},
                "bgcolor": PANEL,
                "steps": [
                    {"range": [-100, -25], "color": BEAR},
                    {"range": [-25, 25], "color": "#2C3844"},
                    {"range": [25, 100], "color": BULL},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor=PANEL,
        font={"color": TEXT},
    )
    return fig


def stat_card(label, value, sub=None, tone=None):
    color = BULL if tone == "bull" else BEAR if tone == "bear" else TEXT
    sub_html = f'<div style="font-size:11px;color:{MUTED};margin-top:2px;">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div style="background:{PANEL};border:1px solid {BORDER};border-radius:12px;padding:14px 16px;">
            <div style="font-size:11px;color:{MUTED};margin-bottom:6px;letter-spacing:0.04em;">{label.upper()}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:19px;font-weight:600;color:{color};">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def price_chart(df: pd.DataFrame, pair_id: str):
    plot_df = df.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["bbUpper"], line=dict(color="#2C3844", width=1), name="BB upper"))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["bbLower"], line=dict(color="#2C3844", width=1),
                              name="BB lower", fill="tonexty", fillcolor="rgba(35,43,54,0.4)"))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["close"], line=dict(color=TEXT, width=1.6), name="Close"))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["sma20"], line=dict(color=BULL, width=1.4), name="SMA 20"))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["sma50"], line=dict(color=GOLD, width=1.4), name="SMA 50"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": MUTED, "size": 11},
        xaxis=dict(gridcolor=GRIDLINE, showgrid=False),
        yaxis=dict(gridcolor=GRIDLINE),
        legend=dict(orientation="h", y=1.12, font={"color": MUTED, "size": 10}),
    )
    return fig


def equity_chart(curve: pd.DataFrame):
    fig = go.Figure()
    fig.add_hline(y=10000, line_dash="dash", line_color="#2C3844")
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["equity"], line=dict(color=GOLD, width=1.8), name="Equity"))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": MUTED, "size": 11},
        xaxis=dict(gridcolor=GRIDLINE, showgrid=False),
        yaxis=dict(gridcolor=GRIDLINE),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FX Signal Dial", layout="wide")

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {BG}; color: {TEXT}; }}
        [data-testid="stSidebar"] {{ background-color: {PANEL_2}; }}
        h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; }}
    </style>
    """,
    unsafe_allow_html=True,
)

if "seed" not in st.session_state:
    st.session_state.seed = 42
if "custom_df" not in st.session_state:
    st.session_state.custom_df = None
if "pending_extraction" not in st.session_state:
    st.session_state.pending_extraction = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0
if "controls_key" not in st.session_state:
    st.session_state.controls_key = 0

st.markdown(f'<div style="color:{MUTED};font-size:12px;letter-spacing:0.14em;">COMPOSITE SIGNAL ENGINE</div>', unsafe_allow_html=True)
st.title("FX Signal Dial")


def _process_uploaded_bytes(file_bytes: bytes, filename: str):
    """Shared extraction pipeline for both the file uploader and the camera
    capture widget — routes by extension/content-type onto CSV / PDF /
    OCR-image handling and stores the result as a pending review."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    try:
        if ext == "csv":
            raw = pd.read_csv(io.BytesIO(file_bytes))
            raw = raw.rename(columns={c: c.strip().lower() for c in raw.columns})
            for col in ["open", "high", "low"]:
                if col not in raw.columns:
                    raw[col] = raw.get("close")
            extracted, method = raw, "CSV"
        elif ext == "pdf":
            extracted, method = extract_from_pdf(file_bytes)
        else:  # jpg / jpeg / png (includes camera captures)
            extracted, method = extract_from_image(file_bytes)

        st.session_state.pending_extraction = {
            "df": extracted,
            "method": method,
            "source": filename,
        }
    except Exception as e:
        st.session_state.pending_extraction = None
        st.error(f"Extraction failed: {e}")


with st.sidebar:
    st.header("Controls")

    if OCR_AVAILABLE:
        st.success("✅ OCR ready")
        st.caption(f"Tesseract {TESSERACT_VERSION} — {TESSERACT_PATH}")
    else:
        st.warning("⚠️ OCR unavailable")
        st.caption(
            "CSV/PDF extraction still works. Install Tesseract for JPG/PNG "
            "and camera OCR, then restart Streamlit."
        )

    pair_id = st.selectbox("Pair", options=list(PAIRS.keys()), format_func=lambda k: PAIRS[k]["name"])

    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("🔄 New sample", use_container_width=True):
            st.session_state.seed += 1
            st.session_state.custom_df = None
    with bc2:
        if st.button("🗑️ Clear all", use_container_width=True, key="clear_all_sidebar"):
            # Wipe uploaded/custom price data, any pending review state, and
            # reset every trade-entry parameter (threshold, risk, SL/TP) back
            # to its default. Bumping these widget-key counters forces fresh
            # Streamlit widget instances, which is the only reliable way to
            # reset a slider/uploader/camera back to default once the user
            # has touched it (session_state values otherwise persist).
            st.session_state.custom_df = None
            st.session_state.pending_extraction = None
            st.session_state.last_upload_fp = None
            st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
            st.session_state.camera_key = st.session_state.get("camera_key", 0) + 1
            st.session_state.controls_key = st.session_state.get("controls_key", 0) + 1
            st.rerun()

    use_camera = st.toggle("📷 Use camera instead of file upload", value=False)

    if use_camera:
        camera_photo = st.camera_input(
            "Snap a photo of a price table",
            help="Point your camera at a printed or on-screen price table — "
                 "the same OCR pipeline used for uploaded images will run on "
                 "the capture, then you can review/fix the rows before use.",
            key=f"camera_{st.session_state.get('camera_key', 0)}",
        )
        if camera_photo is not None:
            fingerprint = f"camera:{len(camera_photo.getvalue())}:{st.session_state.get('camera_key', 0)}"
            if st.session_state.get("last_upload_fp") != fingerprint:
                st.session_state.last_upload_fp = fingerprint
                _process_uploaded_bytes(camera_photo.getvalue(), "camera_capture.jpg")
    else:
        uploaded = st.file_uploader(
            "Upload price data — CSV, PDF, JPG, or PNG",
            type=["csv", "pdf", "jpg", "jpeg", "png"],
            help="CSV needs date/open/high/low/close columns. PDF and images "
                 "are scanned for a price table (via table parsing or OCR) — "
                 "you'll get a chance to review and fix the extracted rows.",
            key=f"uploader_{st.session_state.get('uploader_key', 0)}",
        )
        if uploaded is not None:
            fingerprint = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("last_upload_fp") != fingerprint:
                st.session_state.last_upload_fp = fingerprint
                _process_uploaded_bytes(uploaded.getvalue(), uploaded.name)

    st.divider()
    _ck = st.session_state.get("controls_key", 0)
    threshold = st.slider("Signal threshold ±", 10, 60, 25, key=f"threshold_{_ck}")
    risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0, 1.0, step=0.25, key=f"risk_pct_{_ck}")
    sl_pips = st.slider("Stop-loss (pips)", 5, 100, 30, key=f"sl_pips_{_ck}")
    tp_pips = st.slider("Take-profit (pips)", 5, 200, 60, key=f"tp_pips_{_ck}")

# ---- Review extracted upload before it's used ----
pending = st.session_state.pending_extraction
if pending is not None:
    with st.expander(f"📋 Review data extracted from **{pending['source']}** ({pending['method']})", expanded=True):
        if pending["df"].empty:
            st.warning(
                "No usable price rows were found. For images, try a clearer/cropped screenshot of the "
                "table. For PDFs, a born-digital table works best — scanned PDFs may need to be "
                "uploaded as an image instead. You can also upload a CSV directly."
            )
            if st.button("Dismiss"):
                st.session_state.pending_extraction = None
                st.rerun()
        else:
            st.caption(
                f"Extracted {len(pending['df'])} row(s). Fix any misread values below, then confirm — "
                "rows need a valid date and close price."
            )
            edited = st.data_editor(
                pending["df"], num_rows="dynamic", use_container_width=True, key="extract_editor"
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("✅ Use this data", type="primary"):
                    cleaned = clean_ohlc_df(edited)
                    if cleaned is not None and len(cleaned) > 5:
                        st.session_state.custom_df = cleaned
                        st.session_state.pending_extraction = None
                        st.rerun()
                    else:
                        st.error("Not enough valid rows after cleaning — need at least a handful with date + close.")
            with c2:
                if st.button("Discard"):
                    st.session_state.pending_extraction = None
                    st.rerun()

# ---- Build data ----
if st.session_state.custom_df is not None:
    raw_df = st.session_state.custom_df.reset_index(drop=True)
else:
    raw_df = generate_series(PAIRS[pair_id], 260, st.session_state.seed)

df = build_indicators(raw_df)
last = df.iloc[-1]
sig_label, sig_tone = signal_from_score(last["composite"])
bt = backtest(df, pair_id, threshold, risk_pct, sl_pips, tp_pips)

# ---- Disclaimer ----
st.markdown(
    f"""
    <div style="background:{PANEL};border:1px solid #3A2E1E;border-radius:12px;padding:12px 16px;
                margin-bottom:20px;font-size:13px;color:#B7C0CA;line-height:1.5;">
    ⚠️ This is an educational tool, not financial advice. No indicator can guarantee profit — the sample data is
    synthetically generated unless you upload your own CSV. Past or simulated performance never guarantees future results.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Dial + stats ----
col1, col2 = st.columns([1, 2.2])
with col1:
    st.plotly_chart(signal_dial(last["composite"]), use_container_width=True, config={"displayModeBar": False})
    tone_color = BULL if sig_tone == "bull" else BEAR if sig_tone == "bear" else GOLD if sig_tone == "neutral" else MUTED
    st.markdown(
        f'<div style="text-align:center;"><div style="font-size:26px;font-weight:700;color:{tone_color};">{sig_label}</div>'
        f'<div style="font-family:monospace;font-size:13px;color:{MUTED};">score {last["composite"]:.1f} / 100</div></div>',
        unsafe_allow_html=True,
    )

with col2:
    c1, c2, c3, c4 = st.columns(4)
    decimals = 3 if pair_id.endswith("JPY") else 5
    with c1:
        stat_card("Last close", f'{last["close"]:.{decimals}f}')
    with c2:
        rsi_sub = "overbought" if last["rsi"] > 70 else "oversold" if last["rsi"] < 30 else "neutral"
        stat_card("RSI (14)", f'{last["rsi"]:.1f}' if pd.notna(last["rsi"]) else "—", sub=rsi_sub)
    with c3:
        bullish = last["sma20"] > last["sma50"]
        stat_card("SMA20 vs SMA50", "Bullish" if bullish else "Bearish", tone="bull" if bullish else "bear")
    with c4:
        stat_card("MACD hist", f'{last["macdHist"]:.5f}' if pd.notna(last["macdHist"]) else "—",
                   tone="bull" if last["macdHist"] > 0 else "bear")

st.write("")

# ---- Price chart ----
st.markdown(f'<div style="font-size:13px;color:{MUTED};margin-bottom:10px;">PRICE · SMA20 / SMA50 · BOLLINGER BANDS (last 120 bars)</div>', unsafe_allow_html=True)
st.plotly_chart(price_chart(df, pair_id), use_container_width=True, config={"displayModeBar": False})

# ---- Backtest ----
hdr_col, btn_col = st.columns([5, 1.3])
with hdr_col:
    st.markdown(f'<div style="font-size:13px;color:{MUTED};margin:20px 0 14px;">BACKTEST · RISK-MANAGED ENTRIES</div>', unsafe_allow_html=True)
with btn_col:
    if st.button("🗑️ Clear all trades", use_container_width=True, key="clear_all_terminal"):
        # Same full reset as the sidebar button, exposed here too so it's
        # reachable directly from the signal terminal without opening the
        # sidebar (handy on mobile, where the sidebar is collapsed by default).
        st.session_state.custom_df = None
        st.session_state.pending_extraction = None
        st.session_state.last_upload_fp = None
        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
        st.session_state.camera_key = st.session_state.get("camera_key", 0) + 1
        st.session_state.controls_key = st.session_state.get("controls_key", 0) + 1
        st.rerun()
st.caption(
    f"Reward : risk ratio ≈ **{bt.risk_reward:.2f} : 1**" if bt.risk_reward else "Reward : risk ratio ≈ —"
    + f" · position size is recalculated each trade so a full stop-loss costs exactly {risk_pct:.1f}% of equity at that time."
)

b1, b2, b3, b4, b5 = st.columns(5)
with b1:
    stat_card("Trades", bt.trades)
with b2:
    stat_card("Win rate", f"{bt.win_rate:.1f}%")
with b3:
    stat_card("Profit factor", "∞" if bt.profit_factor == float("inf") else f"{bt.profit_factor:.2f}")
with b4:
    stat_card("Max drawdown", f"{bt.max_drawdown:.1f}%", tone="bear")
with b5:
    stat_card("Return", f"{'+' if bt.return_pct >= 0 else ''}{bt.return_pct:.1f}%", tone="bull" if bt.return_pct >= 0 else "bear")

b6, b7, b8, b9 = st.columns(4)
with b6:
    stat_card("Stopped out", bt.stopped_out, tone="bear")
with b7:
    stat_card("Target hit", bt.target_hit, tone="bull")
with b8:
    stat_card("Signal exits", bt.signal_exits)
with b9:
    stat_card("Final equity", f"${bt.final_equity:,.0f}")

st.write("")
st.plotly_chart(equity_chart(bt.curve), use_container_width=True, config={"displayModeBar": False})

# ---- How it works ----
st.markdown(f'<div style="font-size:13px;color:{MUTED};margin:24px 0 10px;">ℹ️ HOW THE COMPOSITE SCORE IS BUILT</div>', unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns(4)
h1.markdown(f'<span style="color:{BULL};font-weight:600;">Trend</span> — SMA20 vs SMA50 crossover, votes bullish when the fast average leads.', unsafe_allow_html=True)
h2.markdown(f'<span style="color:{GOLD};font-weight:600;">Momentum</span> — RSI(14) distance from the neutral 50 line.', unsafe_allow_html=True)
h3.markdown(f'<span style="color:{BEAR};font-weight:600;">MACD</span> — histogram sign and magnitude relative to price.', unsafe_allow_html=True)
h4.markdown(f'<span style="color:{MUTED};font-weight:600;">Volatility</span> — Bollinger Band position, with a mild mean-reversion bias.', unsafe_allow_html=True)

st.caption(
    "The four votes are averaged into a −100…+100 score. Above the threshold signals BUY, below the negative "
    "threshold signals SELL, in between is HOLD. Each simulated trade sets a stop-loss and take-profit in pips "
    "from entry, and position size is recalculated per trade so that a full stop-loss only ever costs the chosen "
    "risk % of equity at that moment — this is what keeps one bad trade from doing outsized damage, and it's the "
    "same logic professional risk management is built on. It does not, on its own, make a strategy profitable."
)