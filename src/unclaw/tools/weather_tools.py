"""Compatibility layer for the skill-owned weather backend."""

from unclaw.skills.weather.tool import (
    GET_WEATHER_DEFINITION,
    WeatherCurrentConditions,
    WeatherDailyForecast,
    WeatherLookupError,
    WeatherLookupRequest,
    WeatherLookupResponse,
    WeatherRelativeDayAnchor,
    WeatherResolvedLocation,
    WeatherTemporalRequest,
    get_weather,
    get_weather_async,
    ground_weather_tool_result,
    lookup_weather,
    register_weather_tools,
    resolve_weather_temporal_request,
)

__all__ = [
    "GET_WEATHER_DEFINITION",
    "WeatherCurrentConditions",
    "WeatherDailyForecast",
    "WeatherLookupError",
    "WeatherLookupRequest",
    "WeatherLookupResponse",
    "WeatherRelativeDayAnchor",
    "WeatherResolvedLocation",
    "WeatherTemporalRequest",
    "get_weather",
    "get_weather_async",
    "ground_weather_tool_result",
    "lookup_weather",
    "register_weather_tools",
    "resolve_weather_temporal_request",
]
