import os
import requests
import uuid
import streamlit as st
from dotenv import load_dotenv

# LangChain & LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 1. Load Environment Variables (Local)
load_dotenv()

# 2. Key Management
def get_api_key():
    key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        st.error("Missing GROQ_API_KEY. Please add it to Streamlit Secrets or your .env file.")
        st.stop()
    return key

# 3. Define Tools
# --- LOGIC CHANGE: Wrapping DuckDuckGo in a Tool class ---
# This forces the LLM to provide a clear 'query' string, preventing Error 400
duck_search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="duckduckgo_search",
    description="Search the web for real-time information and news. Input should be a single search query string.",
    func=duck_search.run
)

@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city.
    Use this when user asks about weather, temperature, humidity or wind speed.
    """
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_res = requests.get(geo_url).json()

        if not geo_res.get("results"):
            return f"Sorry, I couldn't find the city '{city}'."

        res = geo_res["results"][0]
        lat, lon = res["latitude"], res["longitude"]
        name, country = res["name"], res.get("country", "")

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
            f"&timezone=auto"
        )
        weather_res = requests.get(weather_url).json()
        curr = weather_res["current"]

        return (f"The current weather in {name}, {country} is {curr['temperature_2m']}°C "
                f"with {curr['relative_humidity_2m']}% humidity.")
    
    except Exception as e:
        return f"I ran into an error getting the weather: {str(e)}"

# 4. Agent Factory Function
def create_gorq_agent():
    api_key = get_api_key()
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key
    )

    memory = MemorySaver()

    # --- LOGIC CHANGE: Enhanced System Message ---
    # We guide the model specifically on how to format tool calls
    system_message = (
        "You are Gorq, a sharp-witted and helpful AI assistant powered by Groq. "
        "When using the search tool, provide a clear search query as a string. "
        "Summarize your findings concisely and maintain a friendly, witty tone."
    )

    agent_executor = create_react_agent(
        model=llm,
        tools=[search_tool, get_weather_data],
        checkpointer=memory,
        prompt=system_message
    )
    
    return agent_executor
