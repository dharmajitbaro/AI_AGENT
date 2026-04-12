import streamlit as st
from ai_agent import create_gorq_agent  # Changed from preserve_gorq_agent

# 1. Page Configuration
st.set_page_config(page_title="Assistant D", layout="centered")

st.title("ִ𖤐 AI Assistant D")
st.markdown("### An AI agent made with Groq Llama 3.3 API")
st.markdown("---")

# 2. Initialize Agent in Session State
# This ensures the agent and its memory persist across clicks
if "agent_executor" not in st.session_state:
    with st.spinner("Initializing Gorq's brain..."):
        st.session_state.agent_executor = create_gorq_agent()
        # thread_id is what connects the memory to this specific user session
        st.session_state.config = {"configurable": {"thread_id": "streamlit_session_v1"}}

# 3. Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. User Input & Agent Logic
if prompt := st.chat_input("How may i assist you?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Assistant D is thinking..."):
            try:
                # Invoke the agent from ai_agent.py
                response = st.session_state.agent_executor.invoke(
                    {"messages": [("user", prompt)]},
                    config=st.session_state.config
                )
                
                # The last message in the list is the agent's final answer
                output = response["messages"][-1].content
                
                st.markdown(output)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": output})
                
            except Exception as e:
                st.error(f"Something went wrong: {e}")
