
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 1. Load Environment Variables
load_dotenv()

# 2. Key Management
def get_api_key():
    key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        st.error("Missing GOOGLE_API_KEY. Please add it to Streamlit Secrets or your .env file.")
        st.stop()
    return key

# 3. WMO Weather Code Mapping (standard codes used by Open-Meteo)
WMO_CODES = {
    0: "☀️ Clear sky",
    1: "🌤️ Mainly clear",
    2: "⛅ Partly cloudy",
    3: "☁️ Overcast",
    45: "🌫️ Foggy",
    48: "🌫️ Rime fog",
    51: "🌦️ Light drizzle",
    53: "🌦️ Moderate drizzle",
    55: "🌧️ Dense drizzle",
    56: "🌧️ Freezing light drizzle",
    57: "🌧️ Freezing dense drizzle",
    61: "🌧️ Slight rain",
    63: "🌧️ Moderate rain",
    65: "🌧️ Heavy rain",
    66: "🌧️ Freezing light rain",
    67: "🌧️ Freezing heavy rain",
    71: "🌨️ Slight snow",
    73: "🌨️ Moderate snow",
    75: "❄️ Heavy snow",
    77: "❄️ Snow grains",
    80: "🌦️ Slight showers",
    81: "🌧️ Moderate showers",
    82: "⛈️ Violent showers",
    85: "🌨️ Slight snow showers",
    86: "❄️ Heavy snow showers",
    95: "⛈️ Thunderstorm",
    96: "⛈️ Thunderstorm with slight hail",
    99: "⛈️ Thunderstorm with heavy hail",
}

# 4. Define Tools
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
    humidity, wind speed, or any climate-related question.
    Always use this tool for weather questions, never use web search for weather.
    """
    try:
        # --- Step 1: Geocode the city ---
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search"
            f"?name={city}&count=5&language=en"
        )
        geo_res = requests.get(geo_url, timeout=10).json()

        if not geo_res.get("results"):
            return f"Sorry, I couldn't find the city '{city}'. Please check the spelling."

        # Pick the best match (first result)
        res = geo_res["results"][0]
        lat, lon = res["latitude"], res["longitude"]
        name = res["name"]
        country = res.get("country", "")
        admin = res.get("admin1", "")
        location_str = ", ".join(filter(None, [name, admin, country]))

        # --- Step 2: Fetch current weather ---
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,apparent_temperature,"
            f"relative_humidity_2m,wind_speed_10m,wind_direction_10m,"
            f"wind_gusts_10m,precipitation,weather_code,"
            f"cloud_cover,pressure_msl"
            f"&timezone=auto"
        )
        weather_res = requests.get(weather_url, timeout=10).json()

        if "current" not in weather_res:
            return f"Weather data is temporarily unavailable for {location_str}."

        curr = weather_res["current"]

        # Decode WMO weather code
        code = curr.get("weather_code", -1)
        condition = WMO_CODES.get(code, f"Unknown condition (code {code})")

        # --- Step 3: Build the response ---
        return (
            f"📍 Current weather in {location_str}:\n"
            f"🌤️ Condition: {condition}\n"
            f"🌡️ Temperature: {curr['temperature_2m']}°C "
            f"(Feels like: {curr['apparent_temperature']}°C)\n"
            f"💧 Humidity: {curr['relative_humidity_2m']}%\n"
            f"☁️ Cloud Cover: {curr['cloud_cover']}%\n"
            f"💨 Wind Speed: {curr['wind_speed_10m']} km/h "
            f"(Gusts: {curr['wind_gusts_10m']} km/h)\n"
            f"🌧️ Precipitation: {curr['precipitation']} mm\n"
            f"🔵 Pressure: {curr['pressure_msl']} hPa\n"
            f"\n📊 Coordinates used: {lat}°N, {lon}°E"
        )

    except requests.exceptions.Timeout:
        return "The weather service is taking too long to respond. Please try again."
    except requests.exceptions.ConnectionError:
        return "Could not connect to the weather service. Please check your internet."
    except Exception as e:
        return f"Error getting weather: {str(e)}"

# 5. Agent Factory Function
def create_gemini_agent():
    api_key = get_api_key()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )

    memory = MemorySaver()

        system_message = (
        "You are a sharp-witted and helpful AI assistant named 'Assistant D'. "
        "Be concise, friendly, and occasionally crack a joke. "
        "\n\nSTRICTLY follow these rules:\n"
        "\n## Tool Rules:\n"
        "1. Weather questions → ALWAYS use get_weather_data tool. NEVER use web search for weather.\n"
        "2. Time, date, day questions → ALWAYS use get_datetime tool. Call it with no arguments.\n"
        "3. Factual questions, news, current events → ALWAYS use web search first.\n"
        "4. General conversation (greetings, opinions, jokes) → respond directly.\n"
        "\n## Anti-Hallucination Rules:\n"
        "5. NEVER make up facts, statistics, dates, names, or URLs.\n"
        "6. If a tool returns an error or no results, say 'I couldn't find that information' "
        "— do NOT guess or fabricate an answer.\n"
        "7. If you are unsure about something, clearly say 'I'm not 100% sure about this'.\n"
        "8. When presenting search results, stick to what the search returned. "
        "Do NOT add extra details that weren't in the results.\n"
        "9. When reporting weather, present ALL data from the tool exactly as returned. "
        "Do NOT invent additional weather details.\n"
        "10. NEVER generate fake URLs or links. Only share URLs from search results.\n"
    )

    agent_executor = create_react_agent(
        model=llm,
        tools=[search_tool, get_weather_data, get_datetime],
        checkpointer=memory,
        prompt=system_message,
    )

    return agent_executor
