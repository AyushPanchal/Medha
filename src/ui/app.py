import streamlit as st
import uuid

from src.langgraph_execs.chatbot_graph import ChatbotGraph
from src.states.chatbot_state import ChatbotState
from langchain_core.runnables import RunnableConfig


# --- Build chatbot graph ---
@st.cache_resource(show_spinner=False)
def get_chatbot_graph():
    return ChatbotGraph().build_graph()


chatbot_graph = get_chatbot_graph()

# --- Streamlit page config ---
st.set_page_config(page_title="💬 Medha - SVNIT Chatbot", page_icon="🤖", layout="centered")
st.title("💬 Medha - SVNIT Chatbot")
st.caption("AI assistant for the Computer Science Department, SVNIT Surat")

# --- Initialize session state for memory ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_state" not in st.session_state:
    st.session_state.chat_state = ChatbotState(question="", context="", messages=[])

if "messages_ui" not in st.session_state:
    st.session_state.messages_ui = [
        {"role": "assistant", "content": "Hello 👋! I’m Medha, your CSE department assistant. How can I help you today?"}
    ]

# --- Display chat messages ---
for msg in st.session_state.messages_ui:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input ---
if prompt := st.chat_input("Type your question here..."):
    # Add user message to UI
    st.session_state.messages_ui.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare state for LangGraph
    state = st.session_state.chat_state
    state.question = prompt
    config = RunnableConfig(configurable={"thread_id": st.session_state.thread_id})

    # Invoke the chatbot graph
    try:
        result_dict = chatbot_graph.invoke(state, config=config)
        state = ChatbotState(**result_dict)
        st.session_state.chat_state = state  # Persist updated state

        # Convert answer to display-friendly format
        answer = state.answer if state.answer else "I'm sorry, I couldn't find an answer."

    except Exception as e:
        answer = f"❌ Backend error: {e}"

    # Display assistant response
    st.session_state.messages_ui.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# --- Optional: Sidebar info ---
with st.sidebar:
    st.header("📘 About Medha")
    st.markdown("""
    **Medha** is an AI chatbot for the **Computer Science Department at SVNIT, Surat**.

    - 🧠 Built with **LangGraph + FAISS + GroqLLM**
    - 💬 Multi-turn memory enabled
    - 🤖 Streamlit frontend with ChatGPT-style UI
    """)
    if st.button("🔄 Reset Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_state = ChatbotState(question="", context="", messages=[])
        st.session_state.messages_ui = [
            {"role": "assistant",
             "content": "Hello 👋! I’m Medha, your CSE department assistant. How can I help you today?"}
        ]
        st.rerun()
