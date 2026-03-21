from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, urlparse

import pytest

from unclaw.core.executor import create_default_tool_registry
from unclaw.settings import load_settings
from unclaw.skills.weather import tool as weather_tool_module
from unclaw.skills.weather.tool import (
    GET_WEATHER_DEFINITION,
    WeatherLookupError,
    WeatherLookupRequest,
    get_weather,
    get_weather_async,
    lookup_weather,
)
from unclaw.tools.contracts import ToolCall
from unclaw.tools.dispatcher import ToolDispatcher

pytestmark = pytest.mark.unit


def test_weather_tool_is_registered_in_default_tool_registry(make_temp_project) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    registry = create_default_tool_registry(settings)

    assert any(tool.name == GET_WEATHER_DEFINITION.name for tool in registry.list_tools())


def test_lookup_weather_returns_typed_payload_from_dedicated_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            {
                "results": [
                    {
                        "name": "Paris",
                        "latitude": 48.8566,
                        "longitude": 2.3522,
                        "timezone": "Europe/Paris",
                        "admin1": "Ile-de-France",
                        "country": "France",
                    }
                ]
            },
            {
                "current": {
                    "time": "2026-03-21T12:00",
                    "temperature_2m": 12.3,
                    "apparent_temperature": 10.8,
                    "precipitation": 0.4,
                    "weather_code": 63,
                    "wind_speed_10m": 14.2,
                    "wind_direction_10m": 240,
                },
                "daily": {
                    "time": ["2026-03-21", "2026-03-22"],
                    "weather_code": [63, 3],
                    "temperature_2m_max": [14.0, 16.0],
                    "temperature_2m_min": [8.0, 9.0],
                    "precipitation_probability_max": [80, 20],
                    "precipitation_sum": [3.2, 0.1],
                    "wind_speed_10m_max": [26.4, 18.1],
                },
            },
        )
    )

    monkeypatch.setattr(
        "unclaw.skills.weather.tool._fetch_json_document",
        lambda _url: next(responses),
    )

    response = lookup_weather(WeatherLookupRequest(location="Paris"))

    assert response.provider == "open-meteo"
    assert response.resolved_location.display_name() == "Paris, Ile-de-France, France"
    assert response.current is not None
    assert response.current.weather_description == "moderate rain"
    assert response.daily_forecast[0].date == "2026-03-21"
    assert response.daily_forecast[1].weather_description == "overcast"

    payload = response.to_payload()
    assert payload["resolved_location"]["timezone"] == "Europe/Paris"
    assert payload["current"] is not None
    assert payload["current"]["temperature_c"] == pytest.approx(12.3)
    assert payload["daily_forecast"][0]["precipitation_probability_max_pct"] == 80


def test_get_weather_formats_result_and_returns_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "unclaw.skills.weather.tool.lookup_weather",
        lambda request: lookup_weather_result_fixture(request.location),
    )

    result = get_weather(
        ToolCall(tool_name="get_weather", arguments={"location": "Paris"})
    )

    assert result.success is True
    assert "Location: Paris, Ile-de-France, France" in result.output_text
    assert "Current conditions:" in result.output_text
    assert "7-day forecast:" in result.output_text
    assert result.payload is not None
    assert result.payload["location_query"] == "Paris"


def test_get_weather_fails_cleanly_for_unknown_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "unclaw.skills.weather.tool._fetch_json_document",
        lambda _url: {"results": []},
    )

    result = get_weather(
        ToolCall(tool_name="get_weather", arguments={"location": "Atlantis"})
    )

    assert result.success is False
    assert result.error == "Could not resolve weather location 'Atlantis'."


def test_location_variants_normalize_long_form_queries() -> None:
    assert weather_tool_module._iter_location_query_variants(
        "  Marseille ,  Provence-Alpes-Côte d'Azur ,, France  "
    ) == (
        "Marseille, Provence-Alpes-Côte d'Azur, France",
        "Marseille, Provence-Alpes-Cote d'Azur, France",
        "Marseille, Provence-Alpes-Côte d'Azur",
        "Marseille, Provence-Alpes-Cote d'Azur",
        "Marseille, France",
        "Marseille",
    )


def test_lookup_weather_retries_progressively_simpler_location_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_queries: list[str] = []

    def fake_fetch_json_document(url: str) -> dict[str, object]:
        if url.startswith("https://geocoding-api.open-meteo.com/"):
            query = parse_qs(urlparse(url).query)["name"][0]
            attempted_queries.append(query)
            if query != "Marseille, France":
                return {"results": []}
            return {
                "results": [
                    {
                        "name": "Marseille",
                        "latitude": 43.2965,
                        "longitude": 5.3698,
                        "timezone": "Europe/Paris",
                        "admin1": "Provence-Alpes-Côte d'Azur",
                        "country": "France",
                    }
                ]
            }

        return {
            "current": {
                "time": "2026-03-21T12:00",
                "temperature_2m": 18.4,
                "apparent_temperature": 18.0,
                "precipitation": 0.0,
                "weather_code": 1,
                "wind_speed_10m": 11.2,
                "wind_direction_10m": 160,
            },
            "daily": {
                "time": ["2026-03-21"],
                "weather_code": [1],
                "temperature_2m_max": [19.0],
                "temperature_2m_min": [11.0],
                "precipitation_probability_max": [10],
                "precipitation_sum": [0.0],
                "wind_speed_10m_max": [18.1],
            },
        }

    monkeypatch.setattr(
        "unclaw.skills.weather.tool._fetch_json_document",
        fake_fetch_json_document,
    )

    response = lookup_weather(
        WeatherLookupRequest(
            location="  Marseille , Provence-Alpes-Côte d'Azur ,, France  "
        )
    )

    assert response.resolved_location.display_name() == (
        "Marseille, Provence-Alpes-Côte d'Azur, France"
    )
    assert attempted_queries == [
        "Marseille, Provence-Alpes-Côte d'Azur, France",
        "Marseille, Provence-Alpes-Cote d'Azur, France",
        "Marseille, Provence-Alpes-Côte d'Azur",
        "Marseille, Provence-Alpes-Cote d'Azur",
        "Marseille, France",
    ]


def test_get_weather_validates_required_location_argument() -> None:
    registry = create_registry_with_weather_only()
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        ToolCall(tool_name="get_weather", arguments={"location": ""})
    )

    assert result.success is False
    assert result.error == "Argument 'location' must be a non-empty string."


def test_get_weather_async_runs_sync_weather_lookup_in_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    caller_thread = threading.get_ident()
    observed: dict[str, object] = {}

    def fake_get_weather(call: ToolCall):  # type: ignore[no-untyped-def]
        observed["thread"] = threading.get_ident()
        observed["call"] = call
        return lookup_weather_tool_result_fixture()

    monkeypatch.setattr("unclaw.skills.weather.tool.get_weather", fake_get_weather)

    result = asyncio.run(
        get_weather_async(
            ToolCall(tool_name="get_weather", arguments={"location": "Paris"})
        )
    )

    assert result.success is True
    assert observed["call"] == ToolCall(
        tool_name="get_weather",
        arguments={"location": "Paris"},
    )
    assert observed["thread"] != caller_thread


def create_registry_with_weather_only():
    from unclaw.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(GET_WEATHER_DEFINITION, get_weather)
    return registry


def lookup_weather_result_fixture(location: str):
    if location != "Paris":
        raise WeatherLookupError(f"Unexpected location fixture request: {location}")
    return lookup_weather_response_fixture()


def lookup_weather_response_fixture():
    from unclaw.skills.weather.tool import (
        WeatherCurrentConditions,
        WeatherDailyForecast,
        WeatherLookupResponse,
        WeatherResolvedLocation,
    )

    return WeatherLookupResponse(
        provider="open-meteo",
        location_query="Paris",
        forecast_days=7,
        resolved_location=WeatherResolvedLocation(
            name="Paris",
            admin1="Ile-de-France",
            country="France",
            latitude=48.8566,
            longitude=2.3522,
            timezone="Europe/Paris",
        ),
        current=WeatherCurrentConditions(
            observed_at="2026-03-21T12:00",
            weather_code=63,
            weather_description="moderate rain",
            temperature_c=12.3,
            apparent_temperature_c=10.8,
            precipitation_mm=0.4,
            wind_speed_kph=14.2,
            wind_direction_degrees=240,
        ),
        daily_forecast=(
            WeatherDailyForecast(
                date="2026-03-21",
                weather_code=63,
                weather_description="moderate rain",
                temperature_min_c=8.0,
                temperature_max_c=14.0,
                precipitation_probability_max_pct=80,
                precipitation_sum_mm=3.2,
                wind_speed_max_kph=26.4,
            ),
            WeatherDailyForecast(
                date="2026-03-22",
                weather_code=3,
                weather_description="overcast",
                temperature_min_c=9.0,
                temperature_max_c=16.0,
                precipitation_probability_max_pct=20,
                precipitation_sum_mm=0.1,
                wind_speed_max_kph=18.1,
            ),
        ),
    )


def lookup_weather_tool_result_fixture():
    from unclaw.tools.contracts import ToolResult

    response = lookup_weather_response_fixture()
    return ToolResult.ok(
        tool_name="get_weather",
        output_text="fixture weather output",
        payload=response.to_payload(),
    )
