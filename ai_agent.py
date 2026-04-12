import os
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo  # built-in, no install needed
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
def get_datetime(dummy: str = "") -> str:
    """
    Returns the current real-time date, time and day.
    Use this when user asks about current time, date, day, month or year.
    Do not pass any argument.
    """
    try:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        return (
            f"Current date and time in India:\n"
            f"📅 Date: {now.strftime('%A, %d %B %Y')}\n"
            f"🕐 Time: {now.strftime('%I:%M %p')}\n"
            f"🌍 Timezone: Asia/Kolkata (IST)"
        )
    except Exception as e:
        return f"Error getting date/time: {str(e)}"

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
        geo_res = requests.get(geo_url, timeout=10).json()

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
        weather_res = requests.get(weather_url, timeout=10).json()
        curr = weather_res["current"]

        return (
            f"Current weather in {name}, {country}:\n"
            f"🌡️ Temperature: {curr['temperature_2m']}°C\n"
            f"💧 Humidity: {curr['relative_humidity_2m']}%\n"
            f"💨 Wind Speed: {curr['wind_speed_10m']} km/h\n"
            f"🌧️ Precipitation: {curr['precipitation']} mm"
        )

    except Exception as e:
        return f"Error getting weather: {str(e)}"

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
    "STRICTLY follow these tool rules:\n"
    "1. Weather questions → ALWAYS use get_weather_data tool. NEVER use web search.\n"
    "2. Time, date, day questions → ALWAYS use get_datetime tool. Call it with no arguments.\n"
    "3. Everything else → use web search tool."
)

    agent_executor = create_react_agent(
        model=llm,
        tools=[search_tool, get_weather_data, get_datetime],
        checkpointer=memory,
        prompt=system_message  # correct parameter
    )

    return agent_executor
