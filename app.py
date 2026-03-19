"""
BreakoutRadar — Live NSE Backend
=================================
SETUP (one time):
    pip install nsepy nsetools pandas flask flask-cors

RUN:
    python app.py

TEST:
    http://localhost:3000/api/screener
    http://localhost:3000/api/quote/GUJALKA
    http://localhost:3000/api/health
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import date, timedelta
import pandas as pd
import time
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  WATCHLIST — add/remove symbols here
# ──────────────────────────────────────────────
WATCHLIST = [
    "GUJALKA",    "DEEPAKNTR",  "ATUL",       "FLUOROCHEM",
    "BALAMINES",  "ALKYLAMINE", "NOCIL",      "NEOGEN",
    "VINATIORGA", "NAVINFLUO",  "SUPRIYA",    "DCMSHRIRAM",
    "HFCL",       "JINDALSAW",  "KPRMILL",    "TATAELXSI",
    "SOLARA",     "FINEORG",    "TATACHEM",   "PIDILITIND",
    "AAVAS",      "CERA",       "SAFARI",     "GPPL",
]

# Sector map (extend as needed)
SECTOR_MAP = {
    "GUJALKA":    "Chemicals",  "DEEPAKNTR":  "Chemicals",
    "ATUL":       "Chemicals",  "FLUOROCHEM": "Chemicals",
    "BALAMINES":  "Chemicals",  "ALKYLAMINE": "Chemicals",
    "NOCIL":      "Chemicals",  "NEOGEN":     "Chemicals",
    "VINATIORGA": "Chemicals",  "NAVINFLUO":  "Chemicals",
    "DCMSHRIRAM": "Chemicals",  "FINEORG":    "Chemicals",
    "TATACHEM":   "Chemicals",  "SOLARA":     "Pharma",
    "SUPRIYA":    "Pharma",     "PIDILITIND": "Chemicals",
    "HFCL":       "Telecom",    "JINDALSAW":  "Metals",
    "KPRMILL":    "Textiles",   "TATAELXSI":  "IT",
    "AAVAS":      "Finance",    "CERA":       "Ceramics",
    "SAFARI":     "Consumer",   "GPPL":       "Ports",
}

# ──────────────────────────────────────────────
#  SCORING ENGINE
# ──────────────────────────────────────────────
def compute_score(df: pd.DataFrame, ltp: float) -> dict:
    """
    Breakout score (0–100) modelled on GUJALKA pre-burst pattern.
    4 signals: Volume Spike + Consolidation + Resistance Proximity + RSI Momentum
    """
    score   = 0
    signals = []
    details = {}

    if len(df) < 21:
        return {"score": 0, "signals": [], "details": {}, "volx": 0, "rsi": 0}

    # ── 1. Volume Spike vs 20-day average ──────────────────
    avg_vol   = df["Volume"].rolling(20).mean().iloc[-2]   # yesterday's avg (exclude today)
    today_vol = df["Volume"].iloc[-1]
    volx      = round(today_vol / avg_vol, 1) if avg_vol > 0 else 0

    if volx >= 5:   score += 30; signals.append("VOL")
    elif volx >= 3: score += 22; signals.append("VOL")
    elif volx >= 2: score += 12; signals.append("VOL")
    details["vol_spike_x"] = volx
    details["avg_vol_20d"] = int(avg_vol)
    details["today_vol"]   = int(today_vol)

    # ── 2. Consolidation — tight range last 10 sessions ────
    recent   = df.tail(10)
    hi_10    = recent["High"].max()
    lo_10    = recent["Low"].min()
    rng_pct  = round((hi_10 - lo_10) / lo_10 * 100, 2) if lo_10 > 0 else 99

    if rng_pct < 3:   score += 25; signals.append("CONS")
    elif rng_pct < 5: score += 15; signals.append("CONS")
    details["range_10d_pct"] = rng_pct

    # ── 3. Resistance proximity — near 52-week high ────────
    high_52w  = df["High"].tail(252).max()
    pct_below = round((high_52w - ltp) / high_52w * 100, 2) if high_52w > 0 else 99

    if pct_below <= 1:   score += 25; signals.append("52W")
    elif pct_below <= 3: score += 20; signals.append("RES")
    elif pct_below <= 5: score += 12; signals.append("RES")
    details["52w_high"]     = round(high_52w, 2)
    details["pct_below_52w"] = pct_below

    # ── 4. RSI — momentum zone (not overbought) ────────────
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rs       = gain / loss.replace(0, 0.0001)
    rsi_ser  = 100 - (100 / (1 + rs))
    rsi_val  = round(float(rsi_ser.iloc[-1]), 1)

    if 50 <= rsi_val <= 65:   score += 20; signals.append("MOM")
    elif 45 <= rsi_val < 50:  score += 10; signals.append("MOM")
    details["rsi"] = rsi_val

    # ── 5. Bonus: price above 20-EMA (uptrend confirmation) ─
    ema20 = df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
    if ltp > ema20:
        score += 5
    details["ema20"] = round(ema20, 2)
    details["above_ema20"] = ltp > ema20

    score = min(score, 100)

    # Spark data — last 10 closes for mini chart
    spark = [round(float(v), 2) for v in df["Close"].tail(10).tolist()]

    return {
        "score":   score,
        "signals": signals,
        "details": details,
        "volx":    volx,
        "rsi":     rsi_val,
        "spark":   spark,
        "res":     round(high_52w, 2),
    }


# ──────────────────────────────────────────────
#  DATA FETCHER  (nsepy + nsetools)
# ──────────────────────────────────────────────
def fetch_stock_data(symbol: str) -> dict | None:
    """Fetch historical OHLCV + live quote for one symbol."""
    try:
        from nsepy   import get_history
        from nsetools import Nse

        nse   = Nse()
        end   = date.today()
        start = end - timedelta(days=400)   # ~400 days to cover 252 trading days

        df = get_history(symbol=symbol, start=start, end=end)
        if df is None or len(df) < 21:
            log.warning(f"{symbol}: insufficient history ({len(df) if df is not None else 0} rows)")
            return None

        # Live quote
        quote  = nse.get_quote(symbol) or {}
        ltp    = float(quote.get("lastPrice", str(df["Close"].iloc[-1])).replace(",", ""))
        chg    = float(quote.get("change",    "0").replace(",", ""))
        chg_pct= float(quote.get("pChange",   "0").replace(",", ""))

        scored = compute_score(df, ltp)

        return {
            "sym":    symbol,
            "name":   quote.get("companyName", symbol),
            "price":  ltp,
            "chg":    round(chg_pct, 2),
            "sector": SECTOR_MAP.get(symbol, "Other"),
            **scored,
        }

    except Exception as e:
        log.error(f"{symbol}: {e}")
        return None


# ──────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": str(date.today())})


@app.route("/api/screener")
def screener():
    """
    Query params:
        min_score  (int,   default 60)
        sector     (str,   default all)
        min_vol    (float, default 2.0)
    """
    min_score = int(request.args.get("min_score", 60))
    sector    = request.args.get("sector", "all").lower()
    min_vol   = float(request.args.get("min_vol", 2.0))

    results = []
    for sym in WATCHLIST:
        data = fetch_stock_data(sym)
        time.sleep(0.8)   # polite delay — avoids NSE rate-limiting

        if data is None:
            continue
        if data["score"] < min_score:
            continue
        if data["volx"] < min_vol:
            continue
        if sector != "all" and data["sector"].lower() != sector:
            continue

        results.append(data)
        log.info(f"✓ {sym:12s}  score={data['score']:3d}  vol={data['volx']}x  rsi={data['rsi']}")

    results.sort(key=lambda x: x["score"], reverse=True)
    log.info(f"Screener done — {len(results)} hits from {len(WATCHLIST)} symbols")
    return jsonify(results)


@app.route("/api/quote/<symbol>")
def single_quote(symbol: str):
    """Quick lookup for a single symbol."""
    data = fetch_stock_data(symbol.upper())
    if data:
        return jsonify(data)
    return jsonify({"error": f"Could not fetch data for {symbol}"}), 404


@app.route("/api/watchlist")
def watchlist():
    return jsonify({"symbols": WATCHLIST, "count": len(WATCHLIST)})


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════╗
║       BreakoutRadar  —  NSE Backend          ║
╠══════════════════════════════════════════════╣
║  http://localhost:3000/api/health            ║
║  http://localhost:3000/api/screener          ║
║  http://localhost:3000/api/quote/GUJALKA     ║
║  http://localhost:3000/api/watchlist         ║
╚══════════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=3000, debug=True)
