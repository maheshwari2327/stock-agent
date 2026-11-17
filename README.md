# 📈 Stock Price Information Agent

An agentic AI assistant built with **LangGraph** (`create_react_agent`), **Ollama** (`llama3.2`), **yfinance**, and **Streamlit** to fetch and present current stock market metrics in a terminal or web interface.

---

## 🛠️ Architecture & Specifications

| Component | Specification |
|-----------|---------------|
| **LLM Model** | Ollama `llama3.2` |
| **Temperature** | `0.1` |
| **Framework** | LangGraph `create_react_agent` |
| **Market Data Library** | `yfinance` (Yahoo Finance - free, no API key) |
| **Web UI** | Streamlit |
| **Pattern** | ReAct (Reasoning + Acting) Loop |

---

## 🔄 ReAct Loop Pattern

```
Human: "What is Apple's stock price?"
  │
  ▼
[Think] Apple's ticker is AAPL. Call get_stock_price("AAPL").
  │
  ▼
[Act] get_stock_price(ticker="AAPL")
  │
  ▼
[Observe] {
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "current_price": "$356.27",
  "change": "+$0.89",
  "change_pct": "+0.25%",
  "volume": "39,985,300",
  "fifty_two_week_high": "$360.83",
  "fifty_two_week_low": "$243.42"
}
  │
  ▼
[Answer] Formatted table with metrics and disclaimer.
```

---

## 🌐 Running in a Web Browser (Streamlit)

Launch the web interface locally:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Web UI Features:
- 💬 **Interactive Chat Interface**: Ask stock questions naturally.
- 🧠 **Live ReAct Reasoning Trace**: Collapsible status widget showing tool actions and observations in real-time.
- ⚡ **Quick Ticker Buttons**: One-click lookup for Apple, Google, Microsoft, Amazon, Tesla, NVIDIA.
- 🔒 **Sticky Financial Disclaimer**: Prominently displayed compliance notice.
- 🚀 **Cached Agent Graph**: High-performance instant re-renders.

---

## 💻 Running via Command Line (CLI)

Interactive CLI mode:
```bash
python stock_agent.py
```

Single direct query:
```bash
python stock_agent.py "What is Google's stock price?"
```

---

## 🚀 Installation & Setup

1. **Ensure Ollama is running with `llama3.2`**:
   ```bash
   ollama pull llama3.2
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
