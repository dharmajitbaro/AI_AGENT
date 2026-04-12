import os
import requests
import uuid  # Critical for Python 3.14 type-hint evaluation
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
# Wrapping the search tool to ensure a clean schema for Llama 3.3
duck_search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="duckduckgo_search",
    description="Search the web for real-time info. Input should be a single search query string.",
    func=duck_search.run
)

@tool
def get_weather_data(location_name: str) -> str:
    """
    Fetches real-time weather. 
    Input should be a city name, optionally with state or country (e.g., 'Hamirpur, HP, India').
    """
    try:
        # 1. Geocoding - Search for the BEST match
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=3&language=en&format=json"
        geo_res = requests.get(geo_url).json()

        if not geo_res.get("results"):
            return f"I couldn't find any location matching '{location_name}'."

        # Pick the most relevant result
        res = geo_res["results"][0]
        lat, lon = res["latitude"], res["longitude"]
        # Formulate a clear location name (City, State, Country)
        full_location = f"{res['name']}, {res.get('admin1', '')}, {res.get('country', '')}"

        # 2. Fetch Detailed Weather
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code"
            f"&timezone=auto"
        )
        weather_res = requests.get(weather_url).json()
        curr = weather_res["current"]
        
        # Simple Weather Code Translation
        condition = "Clear" if curr['weather_code'] == 0 else "Partly Cloudy or Rainy"

        return (
            f"Current weather for {full_location}:\n"
            f"- Temperature: {curr['temperature_2m']}°C\n"
            f"- Feels Like: {curr['apparent_temperature']}°C\n"
            f"- Humidity: {curr['relative_humidity_2m']}%\n"
            f"- Conditions: {condition}"
        )
    
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# 4. Agent Factory Function
def create_gorq_agent():
    api_key = get_api_key()
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key
    )

    memory = MemorySaver()

    # System message revised for better tool usage and accuracy
    system_message = (
        "You are Gorq, a sharp-witted AI assistant. "
        "When asked about weather, use the location_name parameter specifically (e.g., 'Delhi, India'). "
        "Always summarize your findings with a touch of wit and confirm the specific location you found."
    )

    agent_executor = create_react_agent(
        model=llm,
        tools=[search_tool, get_weather_data],
        checkpointer=memory,
        prompt=system_message
    )
    
    return agent_executor
