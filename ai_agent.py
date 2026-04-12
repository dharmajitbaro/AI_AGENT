import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load variables
load_dotenv()

# --- Tools ---
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Fetches current weather data for a given city."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo = requests.get(geo_url).json()
        if not geo.get("results"): return f"City '{city}' not found."

        res = geo["results"][0]
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={res['latitude']}&longitude={res['longitude']}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto"
        )
        w = requests.get(weather_url).json()["current"]
        return f"In {res['name']}, it's {w['temperature_2m']}°C with {w['relative_humidity_2m']}% humidity."
    except Exception as e:
        return f"Weather error: {str(e)}"

# --- Agent Factory ---
def preserve_gorq_agent():
    """Initializes and returns the LangGraph agent."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    memory = MemorySaver()
    system_message = "You are Gorq, a witty AI assistant. Summarize search results with a touch of humor."
    
    agent = create_react_agent(
        model=llm,
        tools=[search_tool, get_weather_data],
        checkpointer=memory,
        prompt=system_message
    )
    return agent