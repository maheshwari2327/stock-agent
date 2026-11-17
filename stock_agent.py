"""
Stock Price Information Agent
Built with LangGraph (create_react_agent), Ollama (llama3.2), and yfinance.
"""

import json
import warnings
from typing import Any, Dict
import yfinance as yf

# Suppress minor deprecation warnings for cleaner CLI display
warnings.filterwarnings("ignore")

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# ============================================================================
# SYSTEM PROMPT
# ============================================================================
SYSTEM_PROMPT = """You are a Stock Price Information Agent. Your purpose is to provide current stock market data.

IMPORTANT DISCLAIMER: You provide stock data for informational purposes only. You do not give investment advice, recommendations to buy or sell, or financial guidance.

You have access to the following tool:
- get_stock_price(ticker: str) -> Returns the current price, today's change ($ and %), trading volume, and 52-week high/low for a stock ticker symbol using yfinance (Yahoo Finance data).

## Instructions

When the user asks about a stock:
1. Extract the ticker symbol from the message.
   - If a company name is given (e.g., "Apple"), infer the ticker (e.g., "AAPL").
   - Common mappings: Apple->AAPL, Google->GOOGL, Amazon->AMZN, Microsoft->MSFT, Tesla->TSLA
2. Call get_stock_price with the ticker symbol.
3. Present the data clearly and factually.
4. Always include the disclaimer that this is not financial advice.
5. Never recommend buying, selling, or holding any stock.

## Response Format

📈 **[Company Name] ([TICKER])**

| Metric | Value |
|--------|-------|
| Current Price | $[price] |
| Today's Change | [+/-$change] ([+/-%]) |
| Volume | [volume] |
| 52-Week High | $[high] |
| 52-Week Low | $[low] |

*Data from Yahoo Finance. Not financial advice.*
"""


# ============================================================================
# TOOL DEFINITION
# ============================================================================
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price, today's change, volume, and 52-week range
    for any stock ticker symbol. Uses yfinance — free, no API key required."""
    clean_ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(clean_ticker)
        fast_info = stock.fast_info

        # Extract current price
        last_price = getattr(fast_info, "last_price", None)
        if last_price is None:
            return json.dumps({
                "error": f"No data found for ticker symbol '{clean_ticker}'. Please verify the ticker."
            })

        # Calculate today's change and percentage
        prev_close = getattr(fast_info, "previous_close", last_price)
        change = round(last_price - prev_close, 2)
        change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

        # Volume
        volume = getattr(fast_info, "last_volume", None)
        if volume is not None and isinstance(volume, (int, float)):
            volume_str = f"{int(volume):,}"
        else:
            volume_str = "N/A"

        # 52-week high and low
        high_52 = round(getattr(fast_info, "year_high", 0.0), 2)
        low_52 = round(getattr(fast_info, "year_low", 0.0), 2)

        # Company Name
        company_name = clean_ticker
        try:
            info = stock.info
            company_name = info.get("shortName") or info.get("longName") or clean_ticker
        except Exception:
            pass

        data: Dict[str, Any] = {
            "ticker": clean_ticker,
            "company_name": company_name,
            "current_price": f"{last_price:.2f}",
            "change": f"{'+' if change >= 0 else ''}{change:.2f}",
            "change_pct": f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
            "volume": volume_str,
            "fifty_two_week_high": f"{high_52:.2f}",
            "fifty_two_week_low": f"{low_52:.2f}",
        }
        return json.dumps(data)

    except Exception as e:
        return json.dumps({
            "error": f"Failed to retrieve data for '{clean_ticker}': {str(e)}"
        })


# ============================================================================
# AGENT FACTORY
# ============================================================================
def create_stock_agent(model_name: str = "llama3.2", temperature: float = 0.1):
    """Initializes and returns the LangGraph ReAct stock agent."""
    llm = ChatOllama(model=model_name, temperature=temperature)
    tools = [get_stock_price]
    agent = create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)
    return agent


# ============================================================================
# QUERY RUNNER / CLI
# ============================================================================
def ask_stock_agent(agent, query: str):
    """Executes a query through the ReAct agent and prints the interaction trace."""
    print(f"\nUser: {query}")
    print("-" * 50)
    response = agent.invoke({"messages": [("human", query)]})
    messages = response.get("messages", [])

    for msg in messages:
        msg_type = getattr(msg, "type", "message")
        if msg_type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"Thought / Action -> Calling tool `{tc['name']}` with args: {tc['args']}")
        elif msg_type == "tool":
            print(f"Observation -> Tool returned: {msg.content}")

    # Final assistant response
    if messages:
        final_msg = messages[-1]
        print("\nAgent Response:")
        print(final_msg.content)
        print("-" * 50)
        return final_msg.content
    return ""


if __name__ == "__main__":
    print("Initializing Stock Agent (Ollama llama3.2 + LangGraph)...")
    agent = create_stock_agent()

    # Interactive loop or sample query
    import sys
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        ask_stock_agent(agent, user_query)
    else:
        print("\nRunning demonstration query:")
        ask_stock_agent(agent, "What is Apple's stock price?")
        print("\nType your stock query below (or 'exit' to quit):")
        while True:
            try:
                user_prompt = input("\nEnter stock or question: ").strip()
                if not user_prompt or user_prompt.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break
                ask_stock_agent(agent, user_prompt)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
