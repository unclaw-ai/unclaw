"""Compatibility layer for the skill-owned weather backend."""

from unclaw.skills.weather.tool import (
    GET_WEATHER_DEFINITION,
    WeatherCurrentConditions,
    WeatherDailyForecast,
    WeatherLookupError,
    WeatherLookupRequest,
    WeatherLookupResponse,
    WeatherResolvedLocation,
    get_weather,
    get_weather_async,
    lookup_weather,
    register_weather_tools,
)

__all__ = [
    "GET_WEATHER_DEFINITION",
    "WeatherCurrentConditions",
    "WeatherDailyForecast",
    "WeatherLookupError",
    "WeatherLookupRequest",
    "WeatherLookupResponse",
    "WeatherResolvedLocation",
    "get_weather",
    "get_weather_async",
    "lookup_weather",
    "register_weather_tools",
]
