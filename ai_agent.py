            f"💧 Humidity: {curr['relative_humidity_2m']}%\n"
            f"☁️ Cloud Cover: {curr['cloud_cover']}%\n"
            f"💨 Wind Speed: {curr['wind_speed_10m']} km/h "
            f"(Gusts: {curr['wind_gusts_10m']} km/h)\n"
            f"🌧️ Precipitation: {curr['precipitation']} mm\n"
            f"🔵 Pressure: {curr['pressure_msl']} hPa\n"
            f"\n📊 Coordinates used: {lat}°N, {lon}°E"
        )
    except requests.exceptions.Timeout:
        return f"The weather service is taking too long to respond. Please try again."
    except requests.exceptions.ConnectionError:
        return f"Could not connect to the weather service. Please check your internet."
    except Exception as e:
        return f"Error getting weather: {str(e)}"
# 5. Agent Factory Function
def create_gorq_agent():
    api_key = get_api_key()
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key,
        max_retries=2,
    )
    memory = MemorySaver()
    system_message = (
        "You are a sharp-witted and helpful AI assistant named 'Assistant D'. "
        "Be concise, friendly, and occasionally crack a joke. "
        "STRICTLY follow these tool rules:\n"
        "1. Weather questions → ALWAYS use get_weather_data tool. NEVER use web search for weather.\n"
        "2. Time, date, day questions → ALWAYS use get_datetime tool. Call it with no arguments.\n"
        "3. Everything else → use web search tool.\n"
        "4. When reporting weather, present ALL the data returned by the tool clearly. "
        "Do NOT skip any fields."
    )
    agent_executor = create_react_agent(
        model=llm,
        tools=[search_tool, get_weather_data, get_datetime],
        checkpointer=memory,
        prompt=system_message,
    )
    return agent_executor
