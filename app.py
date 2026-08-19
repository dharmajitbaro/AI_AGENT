import streamlit as st
from agent import create_gemini_agent

# 1. Page Configuration
st.set_page_config(page_title="Assistant D", layout="centered")

st.title("ִ𖤐 AI Assistant D")
st.markdown("### An AI agent powered by Google Gemini 2.0 Flash")
st.markdown("---")

# 2. Initialize Agent in Session State
if "agent_executor" not in st.session_state:
    with st.spinner("Initializing Assistant D..."):
        st.session_state.agent_executor = create_gemini_agent()
        st.session_state.config = {"configurable": {"thread_id": "streamlit_session_v1"}}

# 3. Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. User Input & Agent Logic
if prompt := st.chat_input("How may I assist you?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Assistant D is thinking..."):
            try:
                response = st.session_state.agent_executor.invoke(
                    {"messages": [("user", prompt)]},
                    config=st.session_state.config,
                )

                output = response["messages"][-1].content

                st.markdown(output)

                st.session_state.messages.append(
                    {"role": "assistant", "content": output}
                )

            except Exception as e:
                error_msg = f"Something went wrong: {e}"
                st.error(error_msg)
