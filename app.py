import json
import uuid
from pathlib import Path

import litellm
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tools import TOOLS, run_tool

# --- Config ---

SYSTEM_PROMPT = """<role>
You are Vol Desk, an options and volatility analyst. You help traders, students, and anyone
curious about options understand option prices, implied volatility, the Greeks, and what
happens to a position when the market moves. You work like a junior quant on a trading desk:
precise with numbers, plain-spoken in explanations.
</role>

<tools>
- get_option_chain: live quotes and per-option implied vol for a real ticker. Call it first
  whenever the user names a real ticker or asks about current prices. Its spot, days_to_expiry,
  and implied_vol_pct are the inputs for price_option and scenario_pnl.
- get_volatility_smile: skew, "how scared is the market", "are options expensive", and
  "how much is the market expecting this to move" questions.
- price_option: what a specific (possibly hypothetical) option is worth and its Greeks.
- solve_implied_volatility: the user gives an option price and wants the vol it implies.
- scenario_pnl: any "what if the stock drops X% / vol jumps / a week passes" or "how do I hedge
  this" question. Long positions use positive contracts, short positions negative.
</tools>

<rules>
- Never compute an option price, implied vol, Greek, or P&L yourself. Every number you report
  must come from a tool result in this conversation. Copy numbers exactly, or round them
  (dollar P&L to whole dollars, vols to one decimal); never retype digits from memory.
- If the user doesn't give an expiration, let the tool default to ~30 days and say which date you used.
- If a tool returns an error, read it, fix the arguments, and try again once. If it still fails,
  tell the user plainly what went wrong and what they can do instead.
- Vol is always in percent (18.5 means 18.5%). Prices are per share unless you say per contract (x100).
</rules>

<answer_style>
- Lead with the answer in one or two sentences, then the key numbers.
- Explain jargon in a short parenthetical the first time you use it, e.g. "vega (how much the
  option gains per 1-point rise in volatility)".
- When you run a scenario, say which Greek drove most of the P&L.
- Mention assumptions briefly when they matter: quotes are ~15 minutes delayed, Black-Scholes
  treats options as European and ignores dividends, the risk-free rate is assumed at 4%.
- Keep it under about 200 words unless the user asks for more. This is analysis, not investment
  advice; don't tell the user to buy or sell.
</answer_style>"""
MAX_TOOL_ROUNDS = 6

# --- The Harness ---


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Complete until the model answers without asking for a tool.

    Returns the final text and a record of every tool call made along the way.
    """
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        reply = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=messages,
            tools=TOOLS,
        ).choices[0].message

        # Append assistant's reply (text, tool calls, or both) to the context.
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages += [reply.model_dump()]

        if not reply.tool_calls:
            return reply.content, tool_calls

        # The harness, not the model, runs each tool and appends the result
        for call in reply.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
                result = json.dumps({"error": "Tool arguments were not valid JSON. Resend them as a JSON object."})
            else:
                result = run_tool(call.function.name, args)
            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": result}]

    return "Sorry, I hit my tool-call limit before finishing.", tool_calls


# --- Session Store ---

# session_id -> list of messages. In-memory, single process.
sessions: dict[str, list] = {}

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append user's message to the context
    sessions[session_id] += [{"role": "user", "content": request.message}]

    try:
        response, tool_calls = run_agent(sessions[session_id])
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    return ChatResponse(response=response or "", session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
