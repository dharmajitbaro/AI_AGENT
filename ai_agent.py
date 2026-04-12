import os
import requests
import streamlit as st
from理论 import load_dotenv # In case you missed the fix earlier, ensure this is just: from dotenv import load_dotenv
from dotenv import load_dotenv

# LangChain & LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 1. Load Environment Variables
load_dotenv()

# 2. Key Management
def get_api_key():
    key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        st.error("Missing GROQ_API_KEY. Please add it to Streamlit Secrets or your .env file.")
        st.stop()
    return key

# 3. Define Tools
# Wrapping the search tool to prevent the 400 Tool Use error
duck_search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="duckduckgo_search",
    description="Search the web for real-time information. Input should be a search query string.",
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

    # Refined instructions to prevent function calling errors
    system_message = (
        "You are Gorq, a sharp-witted and helpful AI assistant. "
        "When you need to search, use the duckduckgo_search tool with a simple string query. "
        "Be concise, friendly, and use a touch of humor."
    )

    agent_executor = create_react_agent(
        model=llm,
        tools=[search_tool, get_weather_data],
        checkpointer=memory,
        prompt=system_message
    )
    
    return agent_executor
