import streamlit as st
from ai_agent import preserve_gorq_agent

# Page Config
st.set_page_config(page_title="Gorq AI", page_icon="⚡")
st.title("⚡ Gorq: AI Agent")

# Initialize Agent and Config in Session State
if "agent" not in st.session_state:
    st.session_state.agent = preserve_gorq_agent()
    st.session_state.config = {"configurable": {"thread_id": "streamlit_user_1"}}

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Response
    with st.chat_message("assistant"):
        with st.spinner("Gorq is typing..."):
            try:
                response = st.session_state.agent.invoke(
                    {"messages": [("user", prompt)]},
                    config=st.session_state.config
                )
                final_text = response["messages"][-1].content
                st.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
            except Exception as e:
                st.error(f"Brain freeze! Error: {e}")