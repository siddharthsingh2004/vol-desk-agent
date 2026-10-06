"""Offline checks for the pricing math and tools, using a saved SPY snapshot.

Run from the project folder:  uv run python tests/test_tools.py
"""

import json
import sys
import time
from math import exp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools  # noqa: E402
from pricing import bs_greeks, bs_price, implied_vol  # noqa: E402

T = 1 / 12  # one month


def close(a, b, tol):
    assert abs(a - b) < tol, f"{a} vs {b}"


# Black-Scholes matches the hand-worked example: $500 call, 1 month, 20% vol, 4% rate.
close(bs_price(500, 500, T, 0.20, 0.04, "call"), 12.3468, 1e-3)

# Put-call parity: call - put = spot - PV(strike). Holds for any vol.
c, p = bs_price(500, 480, T, 0.25, 0.04, "call"), bs_price(500, 480, T, 0.25, 0.04, "put")
close(c - p, 500 - 480 * exp(-0.04 * T), 1e-9)

# Implied vol inverts the price.
close(implied_vol(12.3468, 500, 500, T, 0.04, "call"), 0.20, 1e-4)
assert implied_vol(10, 500, 480, T, 0.04, "call") is None  # below intrinsic: impossible

# Greeks agree with bumping the inputs (finite differences).
g = bs_greeks(500, 500, T, 0.20, 0.04, "put")
f = lambda s=500, v=0.20: bs_price(s, 500, T, v, 0.04, "put")
close(g["delta"], (f(500.01) - f(499.99)) / 0.02, 1e-4)
close(g["vega"], (f(v=0.21) - f(v=0.19)) / 2, 1e-3)

# Tools on a real SPY snapshot (Cboe, 2026-10-06), no network needed.
raw = json.loads((Path(__file__).parent / "spy_sample.json").read_text())
tools._chain_cache["SPY"] = (time.time() + 1e9, tools._parse_chain(raw))

smile = json.loads(tools.get_volatility_smile("SPY"))
close(smile["atm_implied_vol_pct"], smile["atm_implied_vol_reported_by_cboe_pct"], 1.0)  # our solver vs Cboe's
assert smile["downside_skew_pts"] > 0, "index puts should carry higher vol than at-the-money"

chain = json.loads(tools.get_option_chain("SPY", num_strikes=4))
assert len(chain["strikes"]) == 4 and chain["expiration"] == "2026-11-06"

# Errors come back as JSON the model can act on, never as exceptions.
assert "error" in json.loads(tools.run_tool("price_option", {"spot": 500}))
assert "error" in json.loads(tools.run_tool("get_option_chain", {"ticker": "SPY", "expiration": "Nov 6"}))
assert "error" in json.loads(tools.run_tool("not_a_tool", {}))

print("All checks passed.")
