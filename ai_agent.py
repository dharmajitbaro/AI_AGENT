import os
import requests
import streamlit as st
from dotenv import load_dotenv

# LangChain & LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool
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
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city.
    ALWAYS use this tool when the user asks about weather, temperature, humidity or wind speed.
    """
    try:
        # FIX 1: Use the 'params' dictionary so requests safely handles spaces in city names
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1}
        geo_res = requests.get(geo_url, params=geo_params).json()

        if not geo_res.get("results"):
            return f"Sorry, I couldn't find the city '{city}'."

        res = geo_res["results"][0]
        lat, lon = res["latitude"], res["longitude"]
        name, country = res["name"], res.get("country", "Unknown")

        # FIX 1 (cont): Use params for the weather API call as well for clean URL construction
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "auto"
        }
        
        weather_res = requests.get(weather_url, params=weather_params).json()
        curr = weather_res.get("current", {})

        if not curr:
            return f"Could not retrieve current weather conditions for {name}."

        return (f"The current weather in {name}, {country} is {curr.get('temperature_2m')}°C "
                f"with {curr.get('relative_humidity_2m')}% humidity and a wind speed of {curr.get('wind_speed_10m')} km/h.")
    
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

    system_message = (
        "You are Gorq, a sharp-witted and helpful AI assistant. "
        "Be concise, friendly, and occasionally crack a joke. "
        "IMPORTANT: If the user asks about the weather, YOU MUST use the 'get_weather_data' tool. "
        "Do not use DuckDuckGo to search for weather."
    )

    # FIX: Use 'prompt' instead of 'state_modifier' for LangGraph >= 0.2.62
    agent_executor = create_react_agent(
        model=llm,
        tools=[search_tool, get_weather_data],
        checkpointer=memory,
        prompt=system_message 
    )
    
    return agent_executor
