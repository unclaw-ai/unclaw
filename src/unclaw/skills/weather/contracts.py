"""Typed payload contracts for the weather skill backend."""

from __future__ import annotations

from typing import TypedDict


class WeatherResolvedLocationPayload(TypedDict):
    """Canonical location metadata returned by the weather tool."""

    name: str
    latitude: float
    longitude: float
    timezone: str
    admin1: str | None
    country: str | None


class WeatherCurrentConditionsPayload(TypedDict):
    """Current weather conditions returned by the weather tool."""

    observed_at: str
    weather_code: int
    weather_description: str
    temperature_c: float
    apparent_temperature_c: float
    precipitation_mm: float
    wind_speed_kph: float
    wind_direction_degrees: int


class WeatherDailyForecastPayload(TypedDict):
    """One daily forecast entry returned by the weather tool."""

    date: str
    weather_code: int
    weather_description: str
    temperature_min_c: float
    temperature_max_c: float
    precipitation_probability_max_pct: int | None
    precipitation_sum_mm: float | None
    wind_speed_max_kph: float | None


class WeatherLookupPayload(TypedDict):
    """Formal schema for the dedicated weather tool result payload."""

    provider: str
    location_query: str
    forecast_days: int
    resolved_location: WeatherResolvedLocationPayload
    current: WeatherCurrentConditionsPayload | None
    daily_forecast: list[WeatherDailyForecastPayload]


__all__ = [
    "WeatherCurrentConditionsPayload",
    "WeatherDailyForecastPayload",
    "WeatherLookupPayload",
    "WeatherResolvedLocationPayload",
]
