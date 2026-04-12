import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
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
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city.
    Use this when user asks about weather, temperature,
    humidity or wind speed. Always use this tool for
    weather questions, never use web search for weather.
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
            f"&current=temperature_2m,relative_humidity_2m,"
            f"wind_speed_10m,precipitation,weather_code"
            f"&timezone=auto"
        )
        weather_res = requests.get(weather_url).json()
        curr = weather_res["current"]

        return (
            f"Current weather in {name}, {country}:\n"
            f"🌡️ Temperature: {curr['temperature_2m']}°C\n"
            f"💧 Humidity: {curr['relative_humidity_2m']}%\n"
            f"💨 Wind Speed: {curr['wind_speed_10m']} km/h\n"
            f"🌧️ Precipitation: {curr['precipitation']} mm"
        )

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
        "You are a sharp-witted and helpful AI assistant. "
        "Be concise, friendly, and occasionally crack a joke. "
        "IMPORTANT: When user asks about weather, temperature, "
        "humidity or wind speed — ALWAYS use the get_weather_data "
        "tool. Never use web search for weather questions."
    )

    agent_executor = create_react_agent(
        model=llm,
        tools=[search_tool, get_weather_data],
        checkpointer=memory,
        prompt=system_message
    )

    return agent_executor
