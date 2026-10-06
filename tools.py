"""The tools the harness can run, and the JSON that describes them to the model.

Design rule: the model never does arithmetic. Every number a user sees
(prices, implied vols, Greeks, P&L) comes from code in this file or pricing.py.
The model's job is to pick the right tool, fill in the arguments, and explain
the result in plain English.
"""

import json
import re
import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import requests

from pricing import bs_greeks, bs_price, implied_vol

# --- Market data (Cboe delayed quotes: free, no API key, ~15 minutes delayed) ---

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
# Cash-settled indexes live under an underscore prefix on Cboe (SPX -> _SPX).
INDEX_SYMBOLS = {"SPX", "XSP", "NDX", "RUT", "VIX", "DJX", "OEX", "XEO"}
NEW_YORK = ZoneInfo("America/New_York")
RISK_FREE_RATE = 0.04  # Assumption: ~4% short-term US rate. Shown in every result.
CACHE_SECONDS = 120  # A SPY chain is ~6 MB; reuse it across tool calls in one conversation.
OCC_SYMBOL = re.compile(r"^(?P<root>[A-Z]+)(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

_chain_cache: dict[str, tuple[float, dict]] = {}


class ToolError(Exception):
    """An error whose message is written for the model to read and act on."""


def _fetch_chain(ticker: str) -> dict:
    """Download and parse every listed option for a ticker (cached for a couple of minutes)."""
    symbol = ticker.strip().upper().lstrip("$^")
    if not re.fullmatch(r"[A-Z.]{1,6}", symbol):
        raise ToolError(f"'{ticker}' is not a valid ticker. Use a US symbol like 'SPY', 'AAPL', or 'SPX'.")

    cached = _chain_cache.get(symbol)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]

    url_symbol = f"_{symbol}" if symbol in INDEX_SYMBOLS else symbol
    try:
        resp = requests.get(CBOE_URL.format(symbol=url_symbol), timeout=20,
                            headers={"User-Agent": "vol-desk-agent/0.1 (Columbia IEOR 4570 class project)"})
    except requests.RequestException as e:
        raise ToolError(f"Could not reach the Cboe market data service ({type(e).__name__}). "
                        "Tell the user live data is unavailable right now; price_option and "
                        "scenario_pnl still work if the user supplies the spot price and volatility.")
    if resp.status_code in (403, 404):
        raise ToolError(f"Cboe has no listed options for '{symbol}'. Check the ticker is a US-listed "
                        "stock, ETF, or index with options (e.g. SPY, QQQ, AAPL, TSLA, SPX).")
    if resp.status_code != 200:
        raise ToolError(f"Cboe market data returned HTTP {resp.status_code}. Try again in a minute.")

    raw = resp.json()
    chain = _parse_chain(raw)
    _chain_cache[symbol] = (time.time(), chain)
    return chain


def _parse_chain(raw: dict) -> dict:
    """Turn Cboe's JSON into a list of plain dicts we can filter and sort."""
    data = raw["data"]
    as_of = datetime.strptime(raw["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    options = []
    for o in data.get("options", []):
        m = OCC_SYMBOL.match(o["option"])
        if not m:
            continue
        options.append({
            "expiry": datetime.strptime(m["yymmdd"], "%y%m%d").date(),
            "type": "call" if m["cp"] == "C" else "put",
            "strike": int(m["strike"]) / 1000,
            "bid": o.get("bid") or 0.0,
            "ask": o.get("ask") or 0.0,
            "last": o.get("last_trade_price") or 0.0,
            "open_interest": o.get("open_interest") or 0,
            "volume": o.get("volume") or 0,
            "cboe_iv": o.get("iv") or 0.0,
        })
    if not options:
        raise ToolError(f"Cboe returned no options for '{data.get('symbol')}'.")
    return {"symbol": data["symbol"], "spot": data["current_price"], "as_of": as_of, "options": options}


def _years_to_expiry(expiry: date, as_of: datetime) -> float:
    """Options stop trading at 4:00pm New York time on expiration day."""
    close = datetime(expiry.year, expiry.month, expiry.day, 16, 0, tzinfo=NEW_YORK)
    return max((close - as_of).total_seconds(), 0.0) / (365 * 24 * 3600)


def _pick_expiration(chain: dict, expiration: str | None) -> date:
    """Use the requested expiration, or default to the one closest to 30 days out."""
    expiries = sorted({o["expiry"] for o in chain["options"]})
    today = chain["as_of"].astimezone(NEW_YORK).date()
    expiries = [e for e in expiries if e >= today]
    if expiration is None:
        return min(expiries, key=lambda e: abs((e - today).days - 30))
    try:
        wanted = date.fromisoformat(expiration)
    except ValueError:
        raise ToolError(f"Expiration '{expiration}' must be a date in YYYY-MM-DD format, e.g. '{expiries[0]}'.")
    if wanted not in expiries:
        nearest = sorted(expiries, key=lambda e: abs((e - wanted).days))[:3]
        raise ToolError(f"{chain['symbol']} has no options expiring {wanted}. "
                        f"Nearest listed expirations: {[str(e) for e in nearest]}.")
    return wanted


def _mid(o: dict) -> float | None:
    """Fair-ish market price: halfway between bid and ask, or the last trade if there's no bid."""
    if o["bid"] > 0 and o["ask"] > 0:
        return (o["bid"] + o["ask"]) / 2
    return o["last"] or None


def _iv_pct(o: dict, spot: float, t: float) -> float | None:
    """Our own implied vol (in %) for one option, backed out of its mid price."""
    price = _mid(o)
    if price is None:
        return None
    iv = implied_vol(price, spot, o["strike"], t, RISK_FREE_RATE, o["type"])
    return round(iv * 100, 2) if iv else None


def _err(message: str) -> str:
    return json.dumps({"error": message})


# --- Tool 1: the option chain ---


def get_option_chain(ticker: str, expiration: str | None = None, num_strikes: int = 10) -> str:
    """Live call and put quotes for the strikes closest to the current stock price."""
    try:
        chain = _fetch_chain(ticker)
        expiry = _pick_expiration(chain, expiration)
    except ToolError as e:
        return _err(str(e))
    num_strikes = max(2, min(int(num_strikes), 30))
    spot, t = chain["spot"], _years_to_expiry(expiry, chain["as_of"])

    by_strike: dict[float, dict] = {}
    for o in chain["options"]:
        if o["expiry"] == expiry:
            by_strike.setdefault(o["strike"], {})[o["type"]] = o
    nearest = sorted(by_strike, key=lambda k: abs(k - spot))[:num_strikes]

    rows = []
    for k in sorted(nearest):
        row = {"strike": k}
        for side in ("call", "put"):
            o = by_strike[k].get(side)
            if o:
                row[side] = {"bid": o["bid"], "ask": o["ask"], "mid": round(_mid(o) or 0, 3),
                             "implied_vol_pct": _iv_pct(o, spot, t), "open_interest": o["open_interest"]}
        rows.append(row)

    upcoming = sorted({o["expiry"] for o in chain["options"] if o["expiry"] >= expiry})
    return json.dumps({
        "symbol": chain["symbol"],
        "spot": spot,
        "quotes_as_of_utc": chain["as_of"].strftime("%Y-%m-%d %H:%M"),
        "note": "Cboe delayed quotes (~15 min). implied_vol_pct is computed by our Black-Scholes solver from the mid price.",
        "expiration": str(expiry),
        "days_to_expiry": round(t * 365, 2),
        "risk_free_rate_pct": RISK_FREE_RATE * 100,
        "strikes": rows,
        "other_expirations": [str(e) for e in upcoming[1:13]],
    })


# --- Tool 2: the volatility smile ---

MONEYNESS_LEVELS = [0.85, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10, 1.15]


def get_volatility_smile(ticker: str, expiration: str | None = None) -> str:
    """Implied vol across strikes for one expiration, plus skew and the market's expected move."""
    try:
        chain = _fetch_chain(ticker)
        expiry = _pick_expiration(chain, expiration)
    except ToolError as e:
        return _err(str(e))
    spot, t = chain["spot"], _years_to_expiry(expiry, chain["as_of"])
    if t * 365 < 1:
        return _err(f"{expiry} expires in under a day; its implied vols are mostly noise. "
                    "Use a later expiration (or omit expiration to get the ~30-day one).")

    # Use out-of-the-money options only (puts below spot, calls above): they're the
    # most traded, and their prices are almost entirely volatility, not intrinsic value.
    otm = [o for o in chain["options"]
           if o["expiry"] == expiry and o["bid"] > 0
           and ((o["type"] == "put" and o["strike"] < spot) or (o["type"] == "call" and o["strike"] >= spot))]
    curve = [(o["strike"], _iv_pct(o, spot, t), o) for o in otm]
    curve = [(k, iv, o) for k, iv, o in curve if iv]
    if len(curve) < 3:
        return _err(f"Not enough liquid options on {chain['symbol']} {expiry} to build a smile. Try another expiration.")

    points = []
    for m in MONEYNESS_LEVELS:
        k, iv, o = min(curve, key=lambda c: abs(c[0] - m * spot))
        if abs(k / spot - m) <= 0.02 and all(p["strike"] != k for p in points):
            points.append({"strike_pct_of_spot": round(k / spot * 100, 1), "strike": k,
                           "uses": o["type"], "implied_vol_pct": iv})

    atm_k, atm_iv, atm_o = min(curve, key=lambda c: abs(c[0] - spot))

    def iv_near(level):
        k, iv, _ = min(curve, key=lambda c: abs(c[0] - level * spot))
        return iv if abs(k / spot - level) <= 0.02 else None

    down, up = iv_near(0.90), iv_near(1.10)
    expected_move_pct = atm_iv * (t ** 0.5)  # 1-standard-deviation move by expiry, in %

    return json.dumps({
        "symbol": chain["symbol"],
        "spot": spot,
        "expiration": str(expiry),
        "days_to_expiry": round(t * 365, 2),
        "atm_implied_vol_pct": atm_iv,
        "atm_implied_vol_reported_by_cboe_pct": round(atm_o["cboe_iv"] * 100, 2) or None,
        "downside_skew_pts": round(down - atm_iv, 2) if down else None,
        "upside_skew_pts": round(up - atm_iv, 2) if up else None,
        "skew_definition": "IV at the 90%-of-spot strike minus ATM IV (downside), and 110% strike minus ATM (upside), in vol points.",
        "expected_move_by_expiry": {"pct": round(expected_move_pct, 2),
                                    "dollars": round(spot * expected_move_pct / 100, 2),
                                    "meaning": "About a 68% chance the price ends within +/- this much by expiry, per the market's implied vol."},
        "smile": points,
    })


# --- Tool 3: price one option ---

OPTION_TYPES = ["call", "put"]


def _check_inputs(option_type, spot, strike, days_to_expiry, volatility_pct):
    if option_type not in OPTION_TYPES:
        raise ToolError(f"option_type must be 'call' or 'put', not '{option_type}'.")
    if spot <= 0 or strike <= 0:
        raise ToolError("spot and strike must be positive dollar amounts.")
    if not 0 < days_to_expiry <= 3650:
        raise ToolError("days_to_expiry must be between 0 and 3650 calendar days (use 0.5 for an option expiring later today).")
    if not 0.5 <= volatility_pct <= 300:
        raise ToolError(f"volatility_pct={volatility_pct} looks wrong. Pass annualized vol in percent: 20 means 20%, not 0.20.")


def price_option(option_type: str, spot: float, strike: float, days_to_expiry: float,
                 volatility_pct: float, risk_free_rate_pct: float = RISK_FREE_RATE * 100) -> str:
    """Black-Scholes price and Greeks for one European option."""
    try:
        _check_inputs(option_type, spot, strike, days_to_expiry, volatility_pct)
    except ToolError as e:
        return _err(str(e))
    t, vol, r = days_to_expiry / 365, volatility_pct / 100, risk_free_rate_pct / 100
    price = bs_price(spot, strike, t, vol, r, option_type)
    g = bs_greeks(spot, strike, t, vol, r, option_type)
    intrinsic = max(spot - strike, 0) if option_type == "call" else max(strike - spot, 0)
    return json.dumps({
        "inputs": {"option_type": option_type, "spot": spot, "strike": strike, "days_to_expiry": days_to_expiry,
                   "volatility_pct": volatility_pct, "risk_free_rate_pct": risk_free_rate_pct},
        "price_per_share": round(price, 4),
        "price_per_contract": round(price * 100, 2),
        "intrinsic_value": round(intrinsic, 4),
        "time_value": round(price - intrinsic, 4),
        "greeks_per_share": {
            "delta": round(g["delta"], 4),
            "gamma": round(g["gamma"], 5),
            "vega_per_vol_pt": round(g["vega"], 4),
            "theta_per_day": round(g["theta"], 4),
        },
        "units": "delta: $ per $1 stock move; gamma: delta change per $1; vega: $ per 1 vol point; theta: $ per calendar day. Multiply by 100 per contract.",
    })


# --- Tool 4: back out implied vol from a price ---


def solve_implied_volatility(option_type: str, option_price: float, spot: float, strike: float,
                             days_to_expiry: float, risk_free_rate_pct: float = RISK_FREE_RATE * 100) -> str:
    """The volatility that makes Black-Scholes match a given option price."""
    try:
        _check_inputs(option_type, spot, strike, days_to_expiry, 20)
    except ToolError as e:
        return _err(str(e))
    if option_price <= 0:
        return _err("option_price must be positive (dollars per share, e.g. 12.35, not per contract).")
    t, r = days_to_expiry / 365, risk_free_rate_pct / 100
    iv = implied_vol(option_price, spot, strike, t, r, option_type)
    if iv is None:
        floor = bs_price(spot, strike, t, 0.001, r, option_type)
        return _err(f"No volatility produces a price of {option_price}. The minimum possible value of this "
                    f"option is about {floor:.2f} (its intrinsic value). Check the price is per share, not per "
                    "contract (divide by 100), and that the strike and option type are right.")
    return json.dumps({"implied_vol_pct": round(iv * 100, 2),
                       "inputs": {"option_type": option_type, "option_price": option_price, "spot": spot,
                                  "strike": strike, "days_to_expiry": days_to_expiry,
                                  "risk_free_rate_pct": risk_free_rate_pct}})


# --- Tool 5: what-if P&L on a position ---


def scenario_pnl(option_type: str, strike: float, days_to_expiry: float, contracts: int,
                 spot: float, volatility_pct: float, spot_change_pct: float = 0.0,
                 vol_change_pts: float = 0.0, days_forward: float = 0.0,
                 risk_free_rate_pct: float = RISK_FREE_RATE * 100) -> str:
    """Reprice a position after a market shock, explain where the P&L came from, and size a delta hedge."""
    try:
        _check_inputs(option_type, spot, strike, days_to_expiry, volatility_pct)
        if contracts == 0:
            raise ToolError("contracts must be non-zero: positive for a long (bought) position, negative for short (sold).")
        if not -90 <= spot_change_pct <= 200:
            raise ToolError("spot_change_pct must be between -90 and 200 (e.g. -5 for a 5% drop).")
        if volatility_pct + vol_change_pts < 0.5:
            raise ToolError(f"Vol can't fall {vol_change_pts} points from {volatility_pct}%. Use a smaller vol_change_pts.")
        if not 0 <= days_forward <= days_to_expiry:
            raise ToolError(f"days_forward must be between 0 and days_to_expiry ({days_to_expiry}).")
    except ToolError as e:
        return _err(str(e))

    r, shares = risk_free_rate_pct / 100, 100 * contracts  # 1 contract = 100 shares
    t0, t1 = days_to_expiry / 365, (days_to_expiry - days_forward) / 365
    vol0, vol1 = volatility_pct / 100, (volatility_pct + vol_change_pts) / 100
    spot1 = spot * (1 + spot_change_pct / 100)
    ds = spot1 - spot

    p0 = bs_price(spot, strike, t0, vol0, r, option_type)
    p1 = bs_price(spot1, strike, t1, vol1, r, option_type)
    g = bs_greeks(spot, strike, t0, vol0, r, option_type)
    option_pnl = (p1 - p0) * shares

    # P&L explain: how much came from each Greek. Whatever is left over is
    # "residual": the part a straight-line Greek approximation misses on big moves.
    explain = {
        "delta": g["delta"] * ds * shares,
        "gamma": 0.5 * g["gamma"] * ds * ds * shares,
        "vega": g["vega"] * vol_change_pts * shares,
        "theta": g["theta"] * days_forward * shares,
    }
    explain["residual"] = option_pnl - sum(explain.values())

    # Delta hedge: trade stock in the opposite direction to cancel the position's delta.
    hedge_shares = -round(g["delta"] * shares)
    hedge_pnl = hedge_shares * ds

    return json.dumps({
        "position": f"{'long' if contracts > 0 else 'short'} {abs(contracts)} x {strike:g} {option_type}, {days_to_expiry:g} days to expiry",
        "scenario": {"spot": f"{spot:g} -> {spot1:.2f} ({spot_change_pct:+g}%)",
                     "volatility_pct": f"{volatility_pct:g} -> {volatility_pct + vol_change_pts:g}",
                     "days_forward": days_forward},
        "option_price_per_share": {"before": round(p0, 4), "after": round(p1, 4)},
        "position_value": {"before": round(p0 * shares, 2), "after": round(p1 * shares, 2)},
        "option_pnl": round(option_pnl, 2),
        "pnl_explain": {k: round(v, 2) for k, v in explain.items()},
        "delta_hedge": {
            "position_delta_shares": round(g["delta"] * shares, 1),
            "hedge_trade": f"{'buy' if hedge_shares > 0 else 'sell'} {abs(hedge_shares)} shares" if hedge_shares else "none needed",
            "hedge_pnl_in_scenario": round(hedge_pnl, 2),
            "hedged_total_pnl": round(option_pnl + hedge_pnl, 2),
            "why": "The hedge cancels the delta P&L, so the hedged result is roughly gamma + vega + theta + residual.",
        },
        "risk_free_rate_pct": risk_free_rate_pct,
    })


# --- What the model sees: the "set notes" in the screenplay ---

_OPTION_TYPE = {"type": "string", "enum": OPTION_TYPES, "description": "'call' (right to buy) or 'put' (right to sell)."}
_SPOT = {"type": "number", "description": "Current price of the underlying stock/ETF/index in dollars, e.g. 779.88."}
_STRIKE = {"type": "number", "description": "Strike price in dollars, e.g. 780."}
_DAYS = {"type": "number", "description": "Calendar days until expiration, e.g. 31. Use days_to_expiry from get_option_chain when available."}
_VOL = {"type": "number", "description": "Annualized implied volatility in PERCENT, e.g. 18.5 for 18.5%. Use the option's implied_vol_pct from get_option_chain when available."}
_RATE = {"type": "number", "description": "Annual risk-free interest rate in percent. Default 4.0. Usually omit."}
_TICKER = {"type": "string", "description": "US ticker with listed options, e.g. 'SPY', 'QQQ', 'AAPL', 'TSLA', or an index like 'SPX'."}
_EXPIRATION = {"type": "string", "description": "Expiration date as YYYY-MM-DD. Omit to use the expiration closest to 30 days out."}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_option_chain",
            "description": (
                "Fetch live (15-min delayed) call and put quotes from Cboe for the strikes nearest the current "
                "price, for one expiration. Returns the spot price, bid/ask/mid, open interest, our implied vol "
                "for each option, days_to_expiry, and other available expirations. Call this first whenever the "
                "user mentions a real ticker or asks about current market prices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": _TICKER,
                    "expiration": _EXPIRATION,
                    "num_strikes": {"type": "integer", "description": "How many strikes nearest the spot price to return (2-30). Default 10."},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_volatility_smile",
            "description": (
                "Build the implied-volatility smile for a ticker and expiration from live Cboe prices: IV at strikes "
                "from 85% to 115% of spot, at-the-money IV, downside and upside skew in vol points, and the "
                "market-implied expected move by expiration. Use for questions about skew, fear, how expensive "
                "options are, or how much the market expects a stock to move."
            ),
            "parameters": {
                "type": "object",
                "properties": {"ticker": _TICKER, "expiration": _EXPIRATION},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "price_option",
            "description": (
                "Price one European option with Black-Scholes and return its Greeks (delta, gamma, vega, theta), "
                "intrinsic value and time value. Use for hypothetical options or 'what is this option worth at X% vol' "
                "questions. Never compute option prices or Greeks yourself; always call this."
            ),
            "parameters": {
                "type": "object",
                "properties": {"option_type": _OPTION_TYPE, "spot": _SPOT, "strike": _STRIKE,
                               "days_to_expiry": _DAYS, "volatility_pct": _VOL, "risk_free_rate_pct": _RATE},
                "required": ["option_type", "spot", "strike", "days_to_expiry", "volatility_pct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_implied_volatility",
            "description": (
                "Back out the implied volatility that makes Black-Scholes match a given option price (the market's "
                "forecast of future movement). Use when the user gives an option price and asks what vol it implies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "option_type": _OPTION_TYPE,
                    "option_price": {"type": "number", "description": "Option price in dollars PER SHARE, e.g. 12.35 (not the per-contract $1,235)."},
                    "spot": _SPOT, "strike": _STRIKE, "days_to_expiry": _DAYS, "risk_free_rate_pct": _RATE,
                },
                "required": ["option_type", "option_price", "spot", "strike", "days_to_expiry"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scenario_pnl",
            "description": (
                "Stress-test an option position: reprice it after a move in the stock, a change in volatility, and/or "
                "time passing. Returns P&L, a P&L explain by Greek (delta, gamma, vega, theta, residual), and the "
                "delta hedge (shares to buy or sell to neutralize small stock moves) with the hedged P&L. Use for any "
                "'what if the stock drops X%' or 'how do I hedge this' question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "option_type": _OPTION_TYPE,
                    "strike": _STRIKE,
                    "days_to_expiry": _DAYS,
                    "contracts": {"type": "integer", "description": "Number of contracts (100 shares each). Positive = long/bought, negative = short/sold."},
                    "spot": _SPOT,
                    "volatility_pct": _VOL,
                    "spot_change_pct": {"type": "number", "description": "Stock move in percent, e.g. -5 for a 5% drop. Default 0."},
                    "vol_change_pts": {"type": "number", "description": "Change in implied vol in vol points, e.g. 3 for 18% -> 21%. Default 0."},
                    "days_forward": {"type": "number", "description": "Days that pass before the shock is measured, e.g. 7. Default 0."},
                    "risk_free_rate_pct": _RATE,
                },
                "required": ["option_type", "strike", "days_to_expiry", "contracts", "spot", "volatility_pct"],
            },
        },
    },
]

# What the harness runs: tool name -> Python function.
TOOL_MAP = {
    "get_option_chain": get_option_chain,
    "get_volatility_smile": get_volatility_smile,
    "price_option": price_option,
    "solve_implied_volatility": solve_implied_volatility,
    "scenario_pnl": scenario_pnl,
}


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return _err(f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}")
    try:
        return TOOL_MAP[name](**args)
    except TypeError as e:
        return _err(f"Bad arguments for {name}: {e}")
    except Exception as e:  # Last line of defense: report, don't crash, don't leak a stack trace.
        return _err(f"{name} failed unexpectedly ({type(e).__name__}). Try different arguments, "
                    "or tell the user this calculation is unavailable.")
