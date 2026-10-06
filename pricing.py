"""Black-Scholes pricing, Greeks, and implied volatility.

Pure math, no network. The tools in tools.py call these functions so the
LLM never does arithmetic itself: the model decides *what* to compute, this
module does the computing.

Conventions (the same ones traders quote in):
- spot, strike, price: dollars
- t_years: time to expiry in years (30 calendar days = 30 / 365)
- vol: annualized volatility as a decimal (20% -> 0.20)
- rate: annualized risk-free rate as a decimal (4% -> 0.04)
"""

from math import erf, exp, log, pi, sqrt


def norm_cdf(x: float) -> float:
    """P(Z <= x) for a standard normal Z."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Height of the standard normal bell curve at x."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(spot, strike, t_years, vol, rate):
    d1 = (log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt(t_years))
    d2 = d1 - vol * sqrt(t_years)
    return d1, d2


def bs_price(spot, strike, t_years, vol, rate, option_type):
    """Black-Scholes price of a European call or put."""
    if t_years <= 0 or vol <= 0:
        # At (or past) expiry an option is worth exactly its intrinsic value.
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    d1, d2 = _d1_d2(spot, strike, t_years, vol, rate)
    if option_type == "call":
        return spot * norm_cdf(d1) - strike * exp(-rate * t_years) * norm_cdf(d2)
    return strike * exp(-rate * t_years) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def bs_greeks(spot, strike, t_years, vol, rate, option_type):
    """Greeks in the units a trader uses.

    delta: $ change in option per $1 move in the stock
    gamma: change in delta per $1 move in the stock
    vega:  $ change in option per 1 vol point (e.g. 20% -> 21%)
    theta: $ change in option per calendar day that passes
    """
    if t_years <= 0 or vol <= 0:
        # At expiry: delta is 1 (or -1) if in the money, else 0; nothing else is left.
        in_money = spot > strike if option_type == "call" else spot < strike
        delta = (1.0 if option_type == "call" else -1.0) if in_money else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    d1, d2 = _d1_d2(spot, strike, t_years, vol, rate)
    pdf = norm_pdf(d1)
    disc = exp(-rate * t_years)

    gamma = pdf / (spot * vol * sqrt(t_years))
    vega = spot * pdf * sqrt(t_years) / 100.0  # per 1 vol point, not per 100
    decay = -spot * pdf * vol / (2.0 * sqrt(t_years))  # time decay, per year

    if option_type == "call":
        delta = norm_cdf(d1)
        theta_year = decay - rate * strike * disc * norm_cdf(d2)
    else:
        delta = norm_cdf(d1) - 1.0
        theta_year = decay + rate * strike * disc * norm_cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta_year / 365.0}


def implied_vol(price, spot, strike, t_years, rate, option_type):
    """The volatility that makes Black-Scholes return `price`.

    Uses bisection: an option's price only goes up as vol goes up, so we can
    keep halving the search range [0.1%, 500%] until we've pinned it down.
    Returns None when no vol can produce the price (e.g. the price is below
    the option's minimum possible value, which happens with stale quotes).
    """
    if t_years <= 0 or price <= 0:
        return None
    low, high = 0.001, 5.0
    if not (bs_price(spot, strike, t_years, low, rate, option_type) <= price
            <= bs_price(spot, strike, t_years, high, rate, option_type)):
        return None
    for _ in range(100):
        mid = 0.5 * (low + high)
        if bs_price(spot, strike, t_years, mid, rate, option_type) < price:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break
    return 0.5 * (low + high)
