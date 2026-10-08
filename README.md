# Vol Desk

Vol Desk is an options and volatility analyst agent. 
Ask it about any US stock, ETF, or index with listed options, and it pulls live quotes, calculates implied volatility, prices options, stress-tests positions, and then 
explains the results in plain English for the user.

## Who it's is For:

This agent is for students learning derivatives
and anyone who wants a fast "what does the market think, and what happens to my position if stock X moves?" answer without opening a pricing spreadsheet.
It's basically your own options assistant, which communicates to you in simple, data-backed language.

#### Important Note

An important part of the project is that the model never does arithmetic. 
Every price, implied volatility, Greek, and profit and loss number comes from a tool. 
Gemini picks the tool, fills in the arguments, and explains the result. Every tool call (name, arguments, result) is returned by `/chat` and drawn in the UI above the answer, for the user's convenience and understanding.

## Tools

| Tool | What it does | Data |
|---|---|---|
| `get_option_chain` | This tool shows the current prices of calls and puts near the stock's price, plus how much movement (volatility) each price implies. | Live from Cboe (free, about 15 min delayed) |
|`get_volatility_smile` |	This tool shows how nervous the market is: whether protection against a drop costs more than betting on a rise, and how much the market expects the stock to move by expiration. |	Live from Cboe |
|`price_option` |	This tool calculates what an option should be worth, and how much its price changes if the stock moves, volatility changes, or a day passes. |	Calculated |
|`solve_implied_volatility` |	This tool takes an option's price and works backward to how much movement the market is expecting. |	Calculated |
|`scenario_pnl` |	This tool answers "what if": how much a position gains or loses if the stock moves or volatility changes, what caused it, and how many shares would offset the risk. |	Calculated |


## How to Use it and Sample Queries

Open https://vol-desk-agent-git-536170830835.us-east1.run.app and sign in with your Columbia account. Type a question or click an example in the sidebar. Each tool the agent calls appears as a card above its answer.

Try these in order, in one chat:

1. **What's the implied vol on SPY right now, and how much does the market expect it to move over the next month?**
2. **How steep is SPY's put skew? Is the market scared of a drop?**
3. **I'm long 5 of the at-the-money SPY calls for that expiration. What happens if SPY drops 3% and vol rises 2 points, and how would I hedge?**


## Files

- `app.py`: FastAPI server, the tool-calling loop (`run_agent`), the session store, and the system prompt
- `tools.py`: the tools, their JSON descriptions, and the Cboe data client
- `pricing.py`: Black-Scholes price, Greeks, and the implied vol solver
- `index.html`: the chat UI, which draws each tool call
- `tests/`: offline checks of the pricing math against a saved SPY snapshot
- `logo.png`: custom hand-drawn logo

## Assumptions and limits

- Quotes are about 15 minutes delayed.
- Black-Scholes treats options as European and ignores dividends. SPY options are American and SPY pays a dividend, so our implied vols differ slightly from Cboe's, which is also reported for comparison.
- The risk-free rate is fixed at 4%.
- Sessions are kept in memory, so they reset when the server restarts.
