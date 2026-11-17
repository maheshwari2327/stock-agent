"""
Streamlit Web Application for Stock Price Information Agent
Powered by LangGraph ReAct Agent, Ollama (llama3.2), and yfinance.
"""

import warnings
import streamlit as st
from stock_agent import create_stock_agent

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Stock Price Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        border-left: 4px solid #F59E0B;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing LangGraph Stock Agent & Ollama...")
def load_agent():
    """Initializes and caches the ReAct stock agent."""
    return create_stock_agent(model_name="llama3.2", temperature=0.1)


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Agent Settings")
    st.markdown(
        """
        - **Model:** `Ollama llama3.2`
        - **Framework:** `LangGraph ReAct`
        - **Data Source:** `Yahoo Finance (yfinance)`
        - **Temperature:** `0.1`
        """
    )
    st.divider()

    st.subheader("💡 Quick Tickers")
    quick_stocks = [
        ("Apple", "What is Apple's stock price?"),
        ("Google", "What is Google's stock price?"),
        ("Microsoft", "What is Microsoft's stock price?"),
        ("Amazon", "What is Amazon's stock price?"),
        ("Tesla", "What is Tesla's stock price?"),
        ("NVIDIA", "What is NVIDIA's stock price?"),
    ]

    for name, query in quick_stocks:
        if st.button(f"🔍 {name}", use_container_width=True, key=f"btn_{name}"):
            st.session_state.preset_prompt = query

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main View Header
st.markdown('<div class="main-header">📈 Stock Price Information Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask about any company or ticker symbol. The agent uses LangGraph ReAct reasoning and real-time Yahoo Finance market data.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <strong>Disclaimer:</strong> This agent provides stock data for informational purposes only. It does not provide investment advice, recommendations to buy or sell, or financial guidance.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load Agent
try:
    agent = load_agent()
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Stock Price Information Agent. Ask me for current stock prices, daily changes, volume, and 52-week ranges for any publicly traded company or ticker symbol.",
        }
    ]

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture User Input
prompt_input = st.chat_input("Enter a company name or ticker (e.g. 'What is Apple stock price?')...")

# Handle quick button click
if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
    prompt_input = st.session_state.preset_prompt
    st.session_state.preset_prompt = None

if prompt_input:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate Agent Response
    with st.chat_message("assistant"):
        with st.status("Agent Reasoning & Fetching Market Data...", expanded=True) as status:
            try:
                response = agent.invoke({"messages": [("human", prompt_input)]})
                messages = response.get("messages", [])

                # Log tool calls and observations in the status expander
                for msg in messages:
                    msg_type = getattr(msg, "type", "")
                    if msg_type == "ai" and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            st.write(f"🧠 **Action:** Calling `{tc['name']}` with `{tc['args']}`")
                    elif msg_type == "tool":
                        st.write("📊 **Observation:** Received market data from Yahoo Finance.")

                final_answer = messages[-1].content if messages else "No response generated."
                status.update(label="Complete!", state="complete", expanded=False)
            except Exception as ex:
                status.update(label="Error occurred", state="error", expanded=True)
                final_answer = f"⚠️ An error occurred while retrieving stock data: `{ex}`"

        st.markdown(final_answer)

    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
