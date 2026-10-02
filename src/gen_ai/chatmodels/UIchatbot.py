from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Mood Chatbot",
    page_icon="🤖",
    layout="centered",
)


# --------------------------------------------------
# AI Model
# --------------------------------------------------

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    temperature=0.6,
)


# --------------------------------------------------
# AI Modes
# --------------------------------------------------

modes = {
    "😡 Angry Mode": (
        "You are an angry AI agent. "
        "You respond aggressively and impatiently, "
        "with a serious tone all the time."
    ),
    "😢 Sad Mode": (
        "You are a sad AI agent. "
        "You respond like someone who is deeply sad, "
        "helpless and depressed about life."
    ),
    "😄 Happy Mode": (
        "You are a very happy AI agent. "
        "You respond with humor, jokes and a cheerful personality."
    ),
}


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🤖 AI Mood Chatbot")
st.caption("Choose an AI personality and start chatting!")


# --------------------------------------------------
# Mode Selection
# --------------------------------------------------

st.subheader("🎭 Choose Your AI Mode")

selected_mode = st.radio(
    "Select a personality:",
    list(modes.keys()),
    horizontal=True,
)


# --------------------------------------------------
# Initialize / Change Conversation
# --------------------------------------------------

if (
    "messages" not in st.session_state
    or st.session_state.get("current_mode") != selected_mode
):

    st.session_state.messages = [
        SystemMessage(content=modes[selected_mode])
    ]

    st.session_state.current_mode = selected_mode


# --------------------------------------------------
# Display Messages
# --------------------------------------------------

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.write(message.content)


# --------------------------------------------------
# User Input
# --------------------------------------------------

prompt = st.chat_input("💬 Type your message...")

if prompt:

    # HumanMessage
    human_message = HumanMessage(content=prompt)

    st.session_state.messages.append(human_message)

    with st.chat_message("user"):
        st.write(prompt)

    # AI Response
    response = llm.invoke(st.session_state.messages)

    # AIMessage
    ai_message = AIMessage(content=response.content)

    st.session_state.messages.append(ai_message)

    with st.chat_message("assistant"):
        st.write(response.content)