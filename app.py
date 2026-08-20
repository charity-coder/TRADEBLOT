"""FX Signal Dial — Streamlit port of the React "Composite Signal Engine" dashboard."""

import csv
import io
import json
import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from scipy import ndimage as scipy_ndimage
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

# Encodings to try, in order, when reading an uploaded CSV. utf-8-sig covers
# plain UTF-8 (with or without a BOM); utf-16 auto-detects LE/BE from its
# BOM and covers Excel's "Unicode Text" exports (the common cause of a
# "'utf-8' codec can't decode byte 0xff in position 0" error); cp1252/
# latin-1 are fallbacks for older Windows exports with accented characters.
CSV_ENCODINGS = ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"]


def _to_float(token):
    try:
        return float(str(token).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _decode_csv_bytes(file_bytes: bytes):
    """Decode raw upload bytes to text, trying several encodings in turn.
    UTF-16 (with its 0xFF 0xFE / 0xFE 0xFF BOM) is what MetaTrader/Excel
    "Unicode Text" exports use, and is the usual cause of a
    "'utf-8' codec can't decode byte 0xff in position 0" error.
    Returns (text, encoding_used)."""
    last_err = None
    for enc in CSV_ENCODINGS:
        try:
            return file_bytes.decode(enc), enc
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise ValueError(
        f"Could not decode this file with any known encoding "
        f"({', '.join(CSV_ENCODINGS)}). Last error: {last_err}"
    )


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        # Fall back on whichever candidate delimiter actually appears most.
        counts = {cand: sample.count(cand) for cand in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def _row_looks_like_data(first_row) -> bool:
    """True if a row's first cell parses as a date/timestamp — i.e. this is
    a price bar, not a header label. MetaTrader exports (MT4/MT5) commonly
    ship with no header row at all, just DateTime,Open,High,Low,Close,...
    starting from row one."""
    first_cell = str(first_row[0]).strip()
    try:
        pd.to_datetime(first_cell)
        return True
    except (ValueError, TypeError):
        return False


def _guess_headerless_columns(ncols: int):
    """Best-effort column names for a headerless OHLC export, based on
    column count. Covers the common MetaTrader layouts. Unknown counts get
    generic col0, col1... names — the user can still rename them in the
    review table before confirming."""
    if ncols == 5:
        return ["date", "open", "high", "low", "close"]
    if ncols == 6:
        return ["date", "open", "high", "low", "close", "volume"]
    if ncols == 7:
        # MT5 single-datetime export: DATE,OPEN,HIGH,LOW,CLOSE,TICKVOL,SPREAD
        return ["date", "open", "high", "low", "close", "volume", "spread"]
    if ncols == 8:
        # MT4-style export with separate DATE and TIME columns.
        return ["date", "time", "open", "high", "low", "close", "volume", "spread"]
    return [f"col{i}" for i in range(ncols)]


def _numeric_parse_rate(series: pd.Series) -> float:
    """Fraction of non-null values in a series that parse as a plain float
    using strict, unmodified float() — i.e. NOT the comma-stripping
    _to_float() used elsewhere. This is used only to detect whether a
    column is actually numeric-with-a-dot vs. numeric-with-a-comma
    (European decimal format); _to_float's thousands-separator stripping
    would mask a decimal comma as "valid", defeating that detection."""
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return 0.0

    def _strict_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    return non_null.map(_strict_float).mean()


def read_csv_any_encoding(file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV robustly:
    1. Decode with whichever encoding actually matches the file (handles
       UTF-16 exports, which is what produces the 0xFF-at-byte-0 error).
    2. Sniff the delimiter instead of assuming a comma.
    3. Detect whether there's a header row at all — MetaTrader exports
       often start straight into data — and synthesize sensible column
       names (date/open/high/low/close/...) when there isn't one.
    4. Handle the European "1234,56" decimal-comma convention, which shows
       up on semicolon-delimited exports and would otherwise leave every
       price column as unparseable text.
    5. Drop phantom all-empty trailing columns caused by a trailing
       delimiter on every line, which would otherwise throw off the
       headerless column-count guess.
    Raises a ValueError with diagnostic detail (rather than silently
    returning an empty frame) if nothing usable comes out the other end.
    """
    text, encoding_used = _decode_csv_bytes(file_bytes)

    stripped = text.strip("\ufeff \r\n")
    if not stripped:
        raise ValueError("The file decoded successfully but contains no text at all (0 bytes of content).")

    non_empty_lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(non_empty_lines) < 2:
        raise ValueError(
            f"Only {len(non_empty_lines)} non-empty line(s) found after decoding as {encoding_used} — "
            "need at least a header/first row plus one data row."
        )

    sep = _sniff_delimiter("\n".join(non_empty_lines[:20]))

    preview = pd.read_csv(io.StringIO(stripped), sep=sep, header=None, nrows=1)
    header_present = not _row_looks_like_data(preview.iloc[0].tolist())

    def _load(decimal="."):
        if header_present:
            d = pd.read_csv(io.StringIO(stripped), sep=sep, decimal=decimal)
            d.columns = [str(c).strip().lower() for c in d.columns]
        else:
            d = pd.read_csv(io.StringIO(stripped), sep=sep, header=None, decimal=decimal)
            # Drop phantom trailing columns that are entirely empty/NaN —
            # a trailing delimiter on every line otherwise inflates the
            # column count and throws off the positional name-guessing.
            d = d.dropna(axis=1, how="all")
            d.columns = _guess_headerless_columns(d.shape[1])
            if "time" in d.columns and "date" in d.columns:
                d["date"] = d["date"].astype(str) + " " + d["time"].astype(str)
                d = d.drop(columns=["time"])
        return d

    df = _load(decimal=".")

    # If the price-like columns didn't actually parse as numbers, this is
    # probably a European export using ',' as the decimal separator
    # (common on semicolon-delimited files) — retry with decimal=",".
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if price_cols and all(_numeric_parse_rate(df[c]) < 0.5 for c in price_cols):
        df_comma = _load(decimal=",")
        price_cols_comma = [c for c in ["open", "high", "low", "close"] if c in df_comma.columns]
        if price_cols_comma and any(_numeric_parse_rate(df_comma[c]) >= 0.5 for c in price_cols_comma):
            df = df_comma

    if df.empty:
        raise ValueError(
            f"Decoded and parsed the file (encoding={encoding_used}, delimiter={sep!r}, "
            f"header_present={header_present}) but got 0 data rows. The file may only contain "
            f"a header, or every row failed to parse."
        )

    return df


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


# ---------------------------------------------------------------------------
# Chart-vision extraction: derive OHLC bars directly from a candlestick/bar
# chart screenshot (MetaTrader, TradingView, etc.) when there is no printed
# price table anywhere in the image — only the chart itself. This is a
# separate, additive pipeline. It is only ever invoked as a *fallback*, when
# extract_from_image()'s ordinary text OCR above finds zero rows; the
# existing text-table OCR path is completely unchanged and always runs
# first.
#
# Pipeline:
#   1. Find the plot rectangle via connected-component labeling on the dark
#      background (trading-platform plot areas render as one large,
#      contiguous near-black region distinct from toolbars/panels).
#   2. Auto-detect the two candle colors (bullish/bearish) by clustering the
#      saturated pixel colors inside that rectangle.
#   3. Walk the plot area column by column, group pixels into individual
#      candles, and record each one's full wick extent (high/low) and its
#      body extent (open/close side).
#   4. OCR the Y-axis price labels immediately to the right of the plot area
#      and linearly fit pixel-row -> price.
#   5. Map each candle's pixel rows through that calibration to get real
#      prices. If calibration isn't possible, fall back to a normalized
#      0-100 scale and warn the user via diagnostics.
# ---------------------------------------------------------------------------
def _np_from_image(img: Image.Image) -> np.ndarray:
    """RGB uint8 array from a PIL image."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def _detect_plot_area(arr: np.ndarray, dark_threshold: int = 40):
    """Locate the chart's plotting rectangle inside a full screenshot via
    connected-component labeling on a "very dark" pixel mask. Returns
    (row0, row1, col0, col1), or None if nothing plausible was found."""
    luminance = arr.mean(axis=2)
    mask = luminance < dark_threshold
    if not mask.any():
        return None

    labeled, n = scipy_ndimage.label(mask)
    if n == 0:
        return None

    sizes = scipy_ndimage.sum(mask, labeled, range(1, n + 1))
    best_label = int(np.argmax(sizes)) + 1
    rows, cols = np.where(labeled == best_label)
    row0, row1 = int(rows.min()), int(rows.max())
    col0, col1 = int(cols.min()), int(cols.max())

    if (col1 - col0) < 50 or (row1 - row0) < 30:
        return None
    return row0, row1, col0, col1


def _dominant_candle_colors(arr: np.ndarray, region):
    """Cluster the non-background pixel colors inside the plot region into
    a bullish (greener) and a bearish (redder) candle color. Trading
    platforms almost always use exactly two saturated colors for candle
    bodies/wicks alongside muted grid/background colors, so the two most
    common "saturated enough" colors are a reliable, theme-agnostic pick."""
    row0, row1, col0, col1 = region
    crop = arr[row0:row1, col0:col1].reshape(-1, 3).astype(int)

    maxc = crop.max(axis=1)
    minc = crop.min(axis=1)
    saturation = maxc - minc
    luminance = crop.mean(axis=1)
    keep = (luminance > 55) & (saturation > 25)
    candidates = crop[keep]
    if len(candidates) < 50:
        return None

    quantized = (candidates // 12) * 12
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(-counts)
    top = colors[order][:12]

    bull_candidates = [c for c in top if c[1] >= c[0] and c[1] >= c[2]]
    bear_candidates = [c for c in top if c[0] > c[1]]
    if not bull_candidates or not bear_candidates:
        return None
    return tuple(int(v) for v in bull_candidates[0]), tuple(int(v) for v in bear_candidates[0])


def _color_mask(pixels: np.ndarray, color, tol: int) -> np.ndarray:
    diff = np.abs(pixels.astype(int) - np.array(color, dtype=int))
    return diff.sum(axis=2) <= tol


def _extract_candle_columns(arr, region, bull_color, bear_color, color_tol: int = 45):
    """Walk the plot area column by column, classify each candle's color,
    and record the pixel row-extent of its full wick (high/low) plus the
    row-extent of the widest run of matching-colored columns (the body,
    i.e. open/close). Returns a list of dicts in image pixel space."""
    row0, row1, col0, col1 = region
    crop = arr[row0:row1, col0:col1]
    h, w = crop.shape[:2]

    bull_mask = _color_mask(crop, bull_color, color_tol)
    bear_mask = _color_mask(crop, bear_color, color_tol)
    any_mask = bull_mask | bear_mask

    col_has_pixels = any_mask.any(axis=0)
    if not col_has_pixels.any():
        return []

    # Group contiguous columns into candles, merging 1px anti-aliasing gaps.
    spans = []
    start = None
    for x in range(w):
        if col_has_pixels[x] and start is None:
            start = x
        elif not col_has_pixels[x] and start is not None:
            spans.append((start, x - 1))
            start = None
    if start is not None:
        spans.append((start, w - 1))

    merged = []
    for c0, c1 in spans:
        if merged and c0 - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], c1)
        else:
            merged.append((c0, c1))

    candles = []
    for c0, c1 in merged:
        seg_bull = bull_mask[:, c0:c1 + 1]
        seg_bear = bear_mask[:, c0:c1 + 1]
        is_bull = int(seg_bull.sum()) >= int(seg_bear.sum())
        seg = seg_bull if is_bull else seg_bear

        rows_present = np.where(seg.any(axis=1))[0]
        if len(rows_present) == 0:
            continue
        wick_top, wick_bottom = int(rows_present.min()), int(rows_present.max())

        # The body spans nearly the candle's full width; the wick is only
        # 1-3px wide — so rows where the colored run is wide are the body.
        width = c1 - c0 + 1
        row_widths = seg.sum(axis=1)
        body_threshold = max(2, width * 0.5)
        body_rows = np.where(row_widths >= body_threshold)[0]
        if len(body_rows):
            body_top, body_bottom = int(body_rows.min()), int(body_rows.max())
        else:
            body_top, body_bottom = wick_top, wick_bottom

        candles.append({
            "col_start": c0 + col0,
            "col_end": c1 + col0,
            "is_bull": is_bull,
            "wick_top": wick_top + row0,
            "wick_bottom": wick_bottom + row0,
            "body_top": body_top + row0,
            "body_bottom": body_bottom + row0,
        })

    return candles


def _ocr_y_axis_prices(img: Image.Image, region, axis_margin: int = 140):
    """OCR the strip immediately to the right of the plot area (where
    MetaTrader/TradingView print the Y-axis price scale) and pair each
    recognized number with the pixel row its label is centered on. Returns
    a list of (pixel_row, price) tuples sorted by pixel_row."""
    row0, row1, col0, col1 = region
    w, h = img.size
    strip = img.crop((min(col1 + 2, w - 1), max(row0 - 10, 0), min(col1 + axis_margin, w), min(row1 + 10, h)))
    if strip.width < 5 or strip.height < 5:
        return []

    scale = 3 if strip.width < 300 else 1
    if scale != 1:
        strip = strip.resize((strip.width * scale, strip.height * scale), Image.Resampling.LANCZOS)

    data = pytesseract.image_to_data(
        strip,
        config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,",
        output_type=pytesseract.Output.DICT,
    )

    labels = []
    for i, txt in enumerate(data["text"]):
        val = _to_float(txt)
        if val is None or val == 0:
            continue
        y_center = data["top"][i] + data["height"][i] / 2
        labels.append((y_center / scale + max(row0 - 10, 0), val))
    return sorted(labels, key=lambda t: t[0])


def _fit_price_calibration(price_labels):
    """Linear-fit pixel-row -> price from >=2 OCR'd axis labels. Returns
    (slope, intercept) such that price = slope * pixel_row + intercept, or
    None if there aren't enough usable, sufficiently-varied labels."""
    if len(price_labels) < 2:
        return None
    rows = np.array([r for r, _ in price_labels], dtype=float)
    prices = np.array([p for _, p in price_labels], dtype=float)
    if np.ptp(rows) < 3 or np.ptp(prices) == 0:
        return None
    slope, intercept = np.polyfit(rows, prices, 1)
    return float(slope), float(intercept)


def extract_ohlc_from_chart_image(file_bytes):
    """Derive OHLC bars directly from a candlestick/bar chart screenshot —
    no visible price table required. Used only as a fallback when
    extract_from_image()'s text OCR finds zero rows (i.e. the upload is a
    plain chart, not a printed table).

    Returns (DataFrame, method_str, diagnostics_dict). diagnostics_dict is
    always returned, even on failure, so the caller can show the user why
    it did or didn't work.
    """
    diagnostics = {
        "plot_area": None,
        "bull_color": None,
        "bear_color": None,
        "candles_found": 0,
        "price_labels_found": 0,
        "calibration": None,
        "warnings": [],
        "manually_calibrated": False,
    }
    empty = pd.DataFrame(columns=["date", "open", "high", "low", "close"])

    if not OCR_AVAILABLE:
        diagnostics["warnings"].append(
            "Tesseract OCR is unavailable — Y-axis price labels can't be read, so prices "
            "would only be on a normalized 0-100 scale."
        )

    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = _np_from_image(img)

    region = _detect_plot_area(arr)
    if region is None:
        diagnostics["warnings"].append("Could not locate a chart plot area in this image.")
        return empty, "chart vision", diagnostics
    diagnostics["plot_area"] = region

    colors = _dominant_candle_colors(arr, region)
    if colors is None:
        diagnostics["warnings"].append("Could not detect distinct bullish/bearish candle colors.")
        return empty, "chart vision", diagnostics
    bull_color, bear_color = colors
    diagnostics["bull_color"] = bull_color
    diagnostics["bear_color"] = bear_color

    candles = _extract_candle_columns(arr, region, bull_color, bear_color)
    diagnostics["candles_found"] = len(candles)
    if not candles:
        diagnostics["warnings"].append("No candlestick shapes were detected inside the plot area.")
        return empty, "chart vision", diagnostics

    price_labels = []
    if OCR_AVAILABLE:
        try:
            price_labels = _ocr_y_axis_prices(img, region)
        except Exception as e:
            diagnostics["warnings"].append(f"Y-axis OCR failed: {e}")
    diagnostics["price_labels_found"] = len(price_labels)

    calibration = _fit_price_calibration(price_labels)
    diagnostics["calibration"] = calibration
    if calibration is None:
        diagnostics["warnings"].append(
            "Couldn't read enough Y-axis price labels to calibrate real prices — returning a "
            "normalized 0-100 scale instead. Edit the values in the review table, or re-upload "
            "a screenshot that includes the price axis clearly."
        )
        row0, row1, _, _ = region

        def to_price(pixel_row):
            return 100 * (1 - (pixel_row - row0) / max(row1 - row0, 1))
    else:
        slope, intercept = calibration

        def to_price(pixel_row):
            return slope * pixel_row + intercept

    rows = []
    for i, c in enumerate(sorted(candles, key=lambda c: c["col_start"])):
        high = to_price(c["wick_top"])
        low = to_price(c["wick_bottom"])
        price_a = to_price(c["body_top"])
        price_b = to_price(c["body_bottom"])
        # body_top pixel is always the higher price; a bullish candle
        # closes higher than it opens, so close <- body_top, open <-
        # body_bottom (and vice-versa for a bearish candle).
        if c["is_bull"]:
            open_, close = price_b, price_a
        else:
            open_, close = price_a, price_b
        # Real dates aren't recoverable from chart pixels alone (X-axis
        # labels are usually too small/sparse to OCR reliably), so use
        # sequential placeholder dates ending "today" — these parse
        # cleanly through clean_ohlc_df's date validation by default, and
        # the review-table info banner tells the user to edit them if the
        # actual dates/timeframe matter.
        placeholder_date = (
            pd.Timestamp.today().normalize() - pd.Timedelta(days=len(candles) - i)
        ).strftime("%Y-%m-%d")
        rows.append({
            "date": placeholder_date,
            "open": round(open_, 5),
            "high": round(max(high, low, open_, close), 5),
            "low": round(min(high, low, open_, close), 5),
            "close": round(close, 5),
        })

    return pd.DataFrame(rows), "chart vision", diagnostics


def apply_manual_price_calibration(df: pd.DataFrame, price_top: float, price_bottom: float) -> pd.DataFrame:
    """Rescale a chart-vision extraction's normalized 0-100 open/high/low/
    close values into real prices via linear interpolation, given the price
    shown at the top and bottom of the plot area. Used when Y-axis OCR
    couldn't calibrate automatically and the user enters the two reference
    prices by hand instead."""
    out = df.copy()

    def interp(v):
        if pd.isna(v):
            return v
        frac = float(v) / 100.0
        return price_bottom + frac * (price_top - price_bottom)

    for col in ["open", "high", "low", "close"]:
        if col in out.columns:
            out[col] = out[col].apply(interp).round(5)
    return out


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

    # Preserve intraday granularity in the display string. Formatting
    # everything as just "%Y-%m-%d" would collapse e.g. 24 hourly H1 bars
    # per day into one repeated date label on the chart/backtest — so use
    # a date+time format whenever the data actually has sub-daily bars.
    is_intraday = (out["_date"].dt.normalize() != out["_date"]).any()
    date_fmt = "%Y-%m-%d %H:%M" if is_intraday else "%Y-%m-%d"
    out["date"] = out["_date"].dt.strftime(date_fmt)

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
# Voice trade advisor
#
# Turns the indicators/backtest stats the app has already computed into a
# short spoken narrative — read aloud in the browser via the Web Speech
# API's speechSynthesis (built into Chrome/Edge/Safari/Firefox, no API key,
# no extra Python package, no network call). It is NOT a live AI model
# forming independent judgements: it's a deterministic, rule-based summary
# of the same composite-score/RSI/MACD/trend/backtest numbers already shown
# elsewhere on the page, phrased as sentences. This keeps the spoken output
# exactly as trustworthy (and exactly as NOT-financial-advice) as the rest
# of the dashboard.
# ---------------------------------------------------------------------------
def generate_trade_advisory(pair_id: str, last: pd.Series, bt: "BacktestResult", sig_label: str, threshold: int) -> str:
    """Build a short spoken-style narrative summarizing the current signal,
    the indicators behind it, and the backtest track record — entirely
    from data already computed for the on-screen dashboard."""
    pair_name = PAIRS[pair_id]["name"]
    score = 0.0 if pd.isna(last["composite"]) else float(last["composite"])
    parts = []

    parts.append(f"Trade advisory for {pair_name}.")
    parts.append(f"The composite signal is currently {sig_label}, with a score of {score:.0f} out of 100, "
                 f"against a plus or minus {threshold} threshold.")

    if pd.notna(last["rsi"]):
        rsi_val = float(last["rsi"])
        if rsi_val > 70:
            rsi_desc = f"overbought, at {rsi_val:.0f}"
        elif rsi_val < 30:
            rsi_desc = f"oversold, at {rsi_val:.0f}"
        else:
            rsi_desc = f"neutral, at {rsi_val:.0f}"
        parts.append(f"R S I is {rsi_desc}.")

    if pd.notna(last["sma20"]) and pd.notna(last["sma50"]):
        if last["sma20"] > last["sma50"]:
            parts.append("The 20 period moving average is above the 50 period average, a bullish trend bias.")
        else:
            parts.append("The 20 period moving average is below the 50 period average, a bearish trend bias.")

    if pd.notna(last["macdHist"]):
        if last["macdHist"] > 0:
            parts.append("The MACD histogram is positive, adding upward momentum.")
        else:
            parts.append("The MACD histogram is negative, adding downward momentum.")

    if bt.trades > 0:
        parts.append(
            f"Backtesting this data with the current settings produced {bt.trades} trades, "
            f"a {bt.win_rate:.0f} percent win rate, and a maximum drawdown of {bt.max_drawdown:.0f} percent."
        )
    else:
        parts.append("The current settings produced no completed backtest trades on this data.")

    parts.append(
        "Reminder: this is an automated summary of the indicators shown on screen, not financial advice. "
        "No indicator can guarantee future results."
    )
    return " ".join(parts)


def voice_trade_advisor_component(text: str, rate: float, pitch: float, voice_name: str, autoplay: bool):
    """Render Play/Stop controls that speak `text` using the browser's
    built-in speechSynthesis. autoplay=True also speaks immediately on
    render (used when the signal has just changed and auto-read is on)."""
    safe_text = json.dumps(text)
    safe_voice = json.dumps(voice_name or "")
    autoplay_js = "speak();" if autoplay else ""
    html = f"""
    <div style="font-family:sans-serif;">
      <button id="speakBtn" onclick="speak()"
        style="background:{GOLD};color:#12181F;border:none;border-radius:8px;
               padding:8px 14px;font-weight:600;cursor:pointer;margin-right:8px;">
        🔊 Read advisory aloud
      </button>
      <button id="stopBtn" onclick="stopSpeak()"
        style="background:transparent;color:{TEXT};border:1px solid {BORDER};border-radius:8px;
               padding:8px 14px;font-weight:600;cursor:pointer;">
        ⏹ Stop
      </button>
      <span id="voiceStatus" style="margin-left:10px;color:{MUTED};font-size:12px;"></span>
      <script>
        const advisoryText = {safe_text};
        const preferredVoice = {safe_voice};

        function pickVoice() {{
          const voices = window.speechSynthesis.getVoices();
          if (!preferredVoice) return null;
          return voices.find(v => v.name === preferredVoice) || null;
        }}

        function speak() {{
          if (!('speechSynthesis' in window)) {{
            document.getElementById('voiceStatus').innerText = 'Speech synthesis not supported in this browser.';
            return;
          }}
          window.speechSynthesis.cancel();
          const utter = new SpeechSynthesisUtterance(advisoryText);
          utter.rate = {rate};
          utter.pitch = {pitch};
          const v = pickVoice();
          if (v) utter.voice = v;
          utter.onstart = () => {{ document.getElementById('voiceStatus').innerText = 'Speaking…'; }};
          utter.onend = () => {{ document.getElementById('voiceStatus').innerText = ''; }};
          window.speechSynthesis.speak(utter);
        }}

        function stopSpeak() {{
          window.speechSynthesis.cancel();
          document.getElementById('voiceStatus').innerText = '';
        }}

        // Voice lists load asynchronously in some browsers.
        if (window.speechSynthesis) {{
          window.speechSynthesis.onvoiceschanged = () => {{}};
        }}

        {autoplay_js}
      </script>
    </div>
    """
    components.html(html, height=60)


def voice_selector_component(key: str):
    """List the browser's available speechSynthesis voices into a Streamlit
    selectbox-like dropdown rendered in HTML (native <select> can't write
    back to Python session_state directly, so this just displays the
    options for reference — the actual voice used is whatever the browser
    picks by default unless overridden by name in voice_trade_advisor_component)."""
    html = f"""
    <div style="font-family:sans-serif;color:{MUTED};font-size:12px;">
      <div id="voiceList_{key}">Loading available system voices…</div>
      <script>
        function listVoices_{key}() {{
          const voices = window.speechSynthesis.getVoices();
          const el = document.getElementById('voiceList_{key}');
          if (voices.length === 0) {{ el.innerText = 'No system voices detected yet — try clicking play once.'; return; }}
          el.innerText = 'Available voices: ' + voices.map(v => v.name).slice(0, 6).join(', ') +
                         (voices.length > 6 ? ', …' : '');
        }}
        listVoices_{key}();
        if (window.speechSynthesis) {{
          window.speechSynthesis.onvoiceschanged = listVoices_{key};
        }}
      </script>
    </div>
    """
    components.html(html, height=40)


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
if "last_spoken_signal" not in st.session_state:
    st.session_state.last_spoken_signal = None

st.markdown(f'<div style="color:{MUTED};font-size:12px;letter-spacing:0.14em;">COMPOSITE SIGNAL ENGINE</div>', unsafe_allow_html=True)
st.title("FX Signal Dial")


def _process_uploaded_bytes(file_bytes: bytes, filename: str):
    """Shared extraction pipeline for both the file uploader and the camera
    capture widget — routes by extension/content-type onto CSV / PDF /
    OCR-image handling and stores the result as a pending review."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    chart_vision_diagnostics = None
    try:
        if ext == "csv":
            raw = read_csv_any_encoding(file_bytes)
            raw = raw.rename(columns={c: c.strip().lower() for c in raw.columns})
            for col in ["open", "high", "low"]:
                if col not in raw.columns:
                    raw[col] = raw.get("close")
            extracted, method = raw, "CSV"
        elif ext == "pdf":
            extracted, method = extract_from_pdf(file_bytes)
        else:  # jpg / jpeg / png (includes camera captures)
            # Text-table OCR runs first, exactly as before — unchanged.
            extracted, method = extract_from_image(file_bytes)

            # Only if that finds zero rows do we try chart-vision: the
            # upload is probably a plain candlestick/bar chart with no
            # printed price table for OCR to read.
            if extracted.empty:
                cv_df, cv_method, cv_diagnostics = extract_ohlc_from_chart_image(file_bytes)
                chart_vision_diagnostics = cv_diagnostics
                if not cv_df.empty:
                    extracted, method = cv_df, cv_method

        st.session_state.pending_extraction = {
            "df": extracted,
            "method": method,
            "source": filename,
            "chart_vision_diagnostics": chart_vision_diagnostics,
        }
    except Exception as e:
        st.session_state.pending_extraction = None
        if isinstance(e, (UnicodeDecodeError, UnicodeError)) or "codec" in str(e).lower():
            st.error(
                "This file doesn't look like standard UTF-8 text — it may be a UTF-16 export "
                "from Excel, or a non-CSV file saved with a .csv extension. Try re-saving it as "
                "'CSV UTF-8 (Comma delimited)' from Excel, or re-save it from a plain text editor."
            )
        elif ext == "csv":
            st.error(f"Couldn't extract usable rows from **{filename}**: {e}")
        else:
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
            help="CSV needs date/open/high/low/close columns (UTF-8, UTF-16, or "
                 "Windows-1252 encoding are all fine). PDF and images are scanned "
                 "for a price table (via table parsing or OCR) — you'll get a "
                 "chance to review and fix the extracted rows.",
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

    st.divider()
    st.subheader("🔊 Voice trade advisor")
    voice_advisor_enabled = st.toggle("Enable voice advisor", value=False, key=f"voice_enabled_{_ck}")
    if voice_advisor_enabled:
        voice_auto_read = st.checkbox(
            "Auto-read when the signal changes", value=True, key=f"voice_auto_{_ck}",
            help="Speaks automatically whenever BUY/SELL/HOLD flips. Otherwise use the Read aloud button.",
        )
        voice_rate = st.slider("Speech rate", 0.5, 2.0, 1.0, step=0.1, key=f"voice_rate_{_ck}")
        voice_pitch = st.slider("Speech pitch", 0.0, 2.0, 1.0, step=0.1, key=f"voice_pitch_{_ck}")
        voice_name = st.text_input(
            "Voice name (optional)", value="", key=f"voice_name_{_ck}",
            help="Leave blank for your browser's default voice. See available names below.",
        )
        voice_selector_component(key=f"vs_{_ck}")
    else:
        voice_auto_read, voice_rate, voice_pitch, voice_name = False, 1.0, 1.0, ""

# ---- Review extracted upload before it's used ----
pending = st.session_state.pending_extraction
if pending is not None:
    with st.expander(f"📋 Review data extracted from **{pending['source']}** ({pending['method']})", expanded=True):
        cv_diag = pending.get("chart_vision_diagnostics")
        if cv_diag is not None:
            if pending["method"] == "chart vision":
                st.info(
                    "No printed price table was found in this image, so bars were read directly off "
                    "the chart's candlesticks instead (pixel analysis, not OCR text). Dates are "
                    "sequential placeholders, not the chart's real dates — edit them below if the "
                    "actual dates/timeframe matter for your analysis."
                )
            d1, d2, d3 = st.columns(3)
            with d1:
                st.caption(
                    "Plot area: "
                    + (f"rows {cv_diag['plot_area'][0]}–{cv_diag['plot_area'][1]}, "
                       f"cols {cv_diag['plot_area'][2]}–{cv_diag['plot_area'][3]}"
                       if cv_diag["plot_area"] else "not found")
                )
            with d2:
                st.caption(f"Candles detected: {cv_diag['candles_found']}")
            with d3:
                calibrated = cv_diag["calibration"] is not None
                manually_cal = cv_diag.get("manually_calibrated", False)
                if manually_cal:
                    cal_label = "✅ manually calibrated"
                elif calibrated:
                    cal_label = f"✅ from {cv_diag['price_labels_found']} axis label(s)"
                else:
                    cal_label = "⚠️ normalized 0-100 scale"
                st.caption(f"Price calibration: {cal_label}")
            for w in cv_diag["warnings"]:
                st.caption(f"⚠️ {w}")

            # Manual calibration fallback: only offered when chart-vision
            # ran, produced rows, and couldn't calibrate against real
            # Y-axis labels on its own. Lets the user type in the price at
            # the top and bottom of the plot area and linearly rescales
            # every open/high/low/close value in pending["df"] from the
            # normalized 0-100 scale into real prices.
            if (
                pending["method"] == "chart vision"
                and cv_diag["calibration"] is None
                and not cv_diag.get("manually_calibrated", False)
                and not pending["df"].empty
            ):
                st.markdown("---")
                st.markdown("**Manual price calibration**")
                st.caption(
                    "OCR couldn't read the Y-axis price labels. Enter the price shown at the top "
                    "and bottom of the chart's plot area and I'll convert the normalized values "
                    "into real prices."
                )
                mc1, mc2, mc3 = st.columns([1, 1, 0.8])
                with mc1:
                    manual_price_top = st.number_input(
                        "Price at top of plot area", value=1.10000, format="%.5f", key="manual_price_top"
                    )
                with mc2:
                    manual_price_bottom = st.number_input(
                        "Price at bottom of plot area", value=1.05000, format="%.5f", key="manual_price_bottom"
                    )
                with mc3:
                    st.write("")
                    st.write("")
                    apply_calibration = st.button("Apply calibration", use_container_width=True)
                if apply_calibration:
                    if manual_price_top <= manual_price_bottom:
                        st.error("Top price must be greater than bottom price.")
                    else:
                        pending["df"] = apply_manual_price_calibration(
                            pending["df"], manual_price_top, manual_price_bottom
                        )
                        cv_diag["manually_calibrated"] = True
                        pending["chart_vision_diagnostics"] = cv_diag
                        st.session_state.pending_extraction = pending
                        st.success(
                            f"Applied calibration: {manual_price_bottom:.5f} – {manual_price_top:.5f}"
                        )
                        st.rerun()

        if pending["df"].empty:
            st.warning(
                "No usable price rows were found. For images, try a clearer/cropped screenshot of the "
                "table, or a clearer chart screenshot that includes the Y-axis price labels. For PDFs, "
                "a born-digital table works best — scanned PDFs may need to be uploaded as an image "
                "instead. You can also upload a CSV directly."
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

# ---- Voice trade advisor ----
if voice_advisor_enabled:
    advisory_text = generate_trade_advisory(pair_id, last, bt, sig_label, threshold)
    should_autoplay = voice_auto_read and st.session_state.last_spoken_signal != sig_label
    with st.container():
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {BORDER};border-radius:12px;'
            f'padding:14px 16px;margin-bottom:16px;">'
            f'<div style="font-size:11px;color:{MUTED};margin-bottom:8px;letter-spacing:0.04em;">🔊 VOICE TRADE ADVISOR</div>'
            f'<div style="font-size:13px;color:{TEXT};line-height:1.5;margin-bottom:10px;">{advisory_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        voice_trade_advisor_component(advisory_text, voice_rate, voice_pitch, voice_name, should_autoplay)
    if should_autoplay:
        st.session_state.last_spoken_signal = sig_label

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
