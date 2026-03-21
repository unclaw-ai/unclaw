"""Skill-owned dedicated live weather lookup tool backed by Open-Meteo APIs."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from unclaw.async_utils import run_blocking
from unclaw.skills.weather.contracts import (
    WeatherCurrentConditionsPayload,
    WeatherDailyForecastPayload,
    WeatherLookupPayload,
    WeatherResolvedLocationPayload,
)
from unclaw.tools.contracts import (
    ToolCall,
    ToolArgumentSpec,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_fetch import _DEFAULT_TIMEOUT_SECONDS, _fetch_raw_document
from unclaw.tools.web_safety import _BlockedFetchTargetError

_WEATHER_PROVIDER_NAME = "open-meteo"
_GEOCODING_ENDPOINT = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
_DEFAULT_FORECAST_DAYS = 7
_WHITESPACE_PATTERN = re.compile(r"\s+")


GET_WEATHER_DEFINITION = ToolDefinition(
    name="get_weather",
    description=(
        "Resolve a place name and return current conditions plus a 7-day forecast."
    ),
    permission_level=ToolPermissionLevel.NETWORK,
    arguments={
        "location": ToolArgumentSpec(
            description="City, region, or place name to look up."
        ),
    },
)


@dataclass(frozen=True, slots=True)
class WeatherLookupRequest:
    """Validated request for one dedicated weather lookup."""

    location: str
    forecast_days: int = _DEFAULT_FORECAST_DAYS


@dataclass(frozen=True, slots=True)
class WeatherResolvedLocation:
    """Canonical location resolved through the weather provider geocoder."""

    name: str
    latitude: float
    longitude: float
    timezone: str
    admin1: str | None = None
    country: str | None = None

    def to_payload(self) -> WeatherResolvedLocationPayload:
        return WeatherResolvedLocationPayload(**asdict(self))

    def display_name(self) -> str:
        parts = [self.name]
        if self.admin1:
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class WeatherCurrentConditions:
    """Current observed or model-nowcast conditions."""

    observed_at: str
    weather_code: int
    weather_description: str
    temperature_c: float
    apparent_temperature_c: float
    precipitation_mm: float
    wind_speed_kph: float
    wind_direction_degrees: int

    def to_payload(self) -> WeatherCurrentConditionsPayload:
        return WeatherCurrentConditionsPayload(**asdict(self))


@dataclass(frozen=True, slots=True)
class WeatherDailyForecast:
    """One daily forecast point returned by the weather provider."""

    date: str
    weather_code: int
    weather_description: str
    temperature_min_c: float
    temperature_max_c: float
    precipitation_probability_max_pct: int | None
    precipitation_sum_mm: float | None
    wind_speed_max_kph: float | None

    def to_payload(self) -> WeatherDailyForecastPayload:
        return WeatherDailyForecastPayload(**asdict(self))


@dataclass(frozen=True, slots=True)
class WeatherLookupResponse:
    """Typed weather response returned by the dedicated backend path."""

    provider: str
    location_query: str
    forecast_days: int
    resolved_location: WeatherResolvedLocation
    current: WeatherCurrentConditions | None
    daily_forecast: tuple[WeatherDailyForecast, ...]

    def to_payload(self) -> WeatherLookupPayload:
        return WeatherLookupPayload(
            provider=self.provider,
            location_query=self.location_query,
            forecast_days=self.forecast_days,
            resolved_location=self.resolved_location.to_payload(),
            current=None if self.current is None else self.current.to_payload(),
            daily_forecast=[forecast.to_payload() for forecast in self.daily_forecast],
        )


class WeatherLookupError(ValueError):
    """Raised when a weather lookup cannot be resolved or parsed safely."""


def register_weather_tools(registry: ToolRegistry) -> None:
    """Register the dedicated live weather lookup tool."""

    registry.register(GET_WEATHER_DEFINITION, get_weather)


def get_weather(call: ToolCall) -> ToolResult:
    """Resolve one place and return compact current-weather and forecast data."""
    tool_name = GET_WEATHER_DEFINITION.name

    try:
        request = _read_weather_request(call)
        response = lookup_weather(request)
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))
    except _BlockedFetchTargetError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))
    except HTTPError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"Weather provider returned HTTP error {exc.code} "
                f"for '{call.arguments.get('location', '')}': {exc.reason}"
            ),
        )
    except URLError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"Could not fetch weather for '{call.arguments.get('location', '')}': "
                f"{exc.reason}"
            ),
        )
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not fetch weather for '{call.arguments.get('location', '')}': {exc}",
        )

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=_format_weather_result(response),
        payload=response.to_payload(),
    )


async def get_weather_async(call: ToolCall) -> ToolResult:
    """Expose the blocking dedicated weather tool through an async boundary."""

    return await run_blocking(get_weather, call)


def lookup_weather(request: WeatherLookupRequest) -> WeatherLookupResponse:
    """Run the dedicated weather backend end to end."""
    resolved_location = _resolve_location(request.location)
    forecast_payload = _fetch_json_document(
        _build_forecast_url(resolved_location, forecast_days=request.forecast_days)
    )
    current_conditions = _parse_current_conditions(forecast_payload.get("current"))
    daily_forecast = _parse_daily_forecast(forecast_payload.get("daily"))

    if not daily_forecast:
        raise WeatherLookupError(
            "Weather provider did not return any daily forecast data."
        )

    return WeatherLookupResponse(
        provider=_WEATHER_PROVIDER_NAME,
        location_query=request.location,
        forecast_days=request.forecast_days,
        resolved_location=resolved_location,
        current=current_conditions,
        daily_forecast=daily_forecast,
    )


def _read_weather_request(call: ToolCall) -> WeatherLookupRequest:
    location = call.arguments.get("location")
    if not isinstance(location, str) or not location.strip():
        raise WeatherLookupError("Argument 'location' must be a non-empty string.")
    return WeatherLookupRequest(location=location.strip())


def _resolve_location(location_query: str) -> WeatherResolvedLocation:
    normalized_query = _normalize_location_query(location_query)
    for query_variant in _iter_location_query_variants(normalized_query):
        raw_location = _find_location_result(query_variant)
        if raw_location is None:
            continue
        return _read_resolved_location(raw_location)

    raise WeatherLookupError(
        f"Could not resolve weather location '{normalized_query or location_query}'."
    )


def _find_location_result(location_query: str) -> dict[str, Any] | None:
    payload = _fetch_json_document(_build_geocoding_url(location_query))
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise WeatherLookupError(
            "Weather provider returned an invalid geocoding result."
        )
    if not raw_results:
        return None

    raw_location = raw_results[0]
    if not isinstance(raw_location, dict):
        raise WeatherLookupError(
            "Weather provider returned an invalid geocoding result."
        )
    return raw_location


def _read_resolved_location(raw_location: dict[str, Any]) -> WeatherResolvedLocation:
    return WeatherResolvedLocation(
        name=_read_required_string(raw_location, "name"),
        latitude=_read_required_float(raw_location, "latitude"),
        longitude=_read_required_float(raw_location, "longitude"),
        timezone=_read_required_string(raw_location, "timezone"),
        admin1=_read_optional_string(raw_location, "admin1"),
        country=_read_optional_string(raw_location, "country"),
    )


def _normalize_location_query(location_query: str) -> str:
    parts = _split_location_parts(location_query)
    if not parts:
        return ""
    return ", ".join(parts)


def _split_location_parts(location_query: str) -> tuple[str, ...]:
    return tuple(
        normalized_part
        for raw_part in unicodedata.normalize("NFKC", location_query).split(",")
        if (normalized_part := _normalize_location_part(raw_part))
    )


def _normalize_location_part(location_part: str) -> str:
    normalized = unicodedata.normalize("NFKC", location_part)
    collapsed = _WHITESPACE_PATTERN.sub(" ", normalized)
    return collapsed.strip(" ,")


def _iter_location_query_variants(location_query: str) -> tuple[str, ...]:
    normalized_query = _normalize_location_query(location_query)
    if not normalized_query:
        return ()

    parts = _split_location_parts(normalized_query)
    structural_variants: list[tuple[str, ...]] = [parts]
    if len(parts) > 1:
        for end_index in range(len(parts) - 1, 1, -1):
            structural_variants.append(parts[:end_index])
        if len(parts) >= 3:
            structural_variants.append((parts[0], parts[-1]))
        structural_variants.append((parts[0],))

    variants: list[str] = []
    seen_variants: set[str] = set()
    for variant_parts in structural_variants:
        candidate = ", ".join(variant_parts)
        _append_location_variant(
            variants,
            seen_variants=seen_variants,
            candidate=candidate,
        )
        simplified_ascii = _strip_diacritics(candidate)
        if simplified_ascii != candidate:
            _append_location_variant(
                variants,
                seen_variants=seen_variants,
                candidate=simplified_ascii,
            )

    return tuple(variants)


def _append_location_variant(
    variants: list[str],
    *,
    seen_variants: set[str],
    candidate: str,
) -> None:
    key = candidate.casefold()
    if key in seen_variants:
        return
    seen_variants.add(key)
    variants.append(candidate)


def _strip_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )


def _fetch_json_document(url: str) -> dict[str, Any]:
    document = _fetch_raw_document(
        url,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        allow_private_networks=False,
        accept_header="application/json, text/plain;q=0.1, */*;q=0.01",
    )
    try:
        payload = json.loads(document.decoded_text)
    except json.JSONDecodeError as exc:
        raise WeatherLookupError(
            f"Weather provider returned invalid JSON: {exc.msg}."
        ) from exc

    if not isinstance(payload, dict):
        raise WeatherLookupError(
            "Weather provider returned an invalid JSON document."
        )

    if payload.get("error") is True:
        reason = payload.get("reason")
        if isinstance(reason, str) and reason.strip():
            raise WeatherLookupError(reason.strip())
        raise WeatherLookupError("Weather provider returned an error response.")

    return payload


def _build_geocoding_url(location_query: str) -> str:
    query = urlencode(
        {
            "name": location_query,
            "count": 1,
            "language": "en",
            "format": "json",
        }
    )
    return f"{_GEOCODING_ENDPOINT}?{query}"


def _build_forecast_url(
    location: WeatherResolvedLocation,
    *,
    forecast_days: int,
) -> str:
    query = urlencode(
        {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timezone": location.timezone,
            "forecast_days": forecast_days,
            "current": (
                "temperature_2m,apparent_temperature,precipitation,"
                "weather_code,wind_speed_10m,wind_direction_10m"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,precipitation_sum,"
                "wind_speed_10m_max"
            ),
        }
    )
    return f"{_FORECAST_ENDPOINT}?{query}"


def _parse_current_conditions(
    raw_current: Any,
) -> WeatherCurrentConditions | None:
    if raw_current is None:
        return None
    if not isinstance(raw_current, dict):
        raise WeatherLookupError(
            "Weather provider returned an invalid current-conditions payload."
        )

    return WeatherCurrentConditions(
        observed_at=_read_required_string(raw_current, "time"),
        weather_code=_read_required_int(raw_current, "weather_code"),
        weather_description=_describe_weather_code(
            _read_required_int(raw_current, "weather_code")
        ),
        temperature_c=_read_required_float(raw_current, "temperature_2m"),
        apparent_temperature_c=_read_required_float(
            raw_current,
            "apparent_temperature",
        ),
        precipitation_mm=_read_required_float(raw_current, "precipitation"),
        wind_speed_kph=_read_required_float(raw_current, "wind_speed_10m"),
        wind_direction_degrees=_read_required_int(
            raw_current,
            "wind_direction_10m",
        ),
    )


def _parse_daily_forecast(raw_daily: Any) -> tuple[WeatherDailyForecast, ...]:
    if not isinstance(raw_daily, dict):
        raise WeatherLookupError(
            "Weather provider returned an invalid daily-forecast payload."
        )

    dates = _read_required_list(raw_daily, "time")
    weather_codes = _read_required_list(raw_daily, "weather_code")
    maximums = _read_required_list(raw_daily, "temperature_2m_max")
    minimums = _read_required_list(raw_daily, "temperature_2m_min")
    precipitation_probabilities = _read_required_list(
        raw_daily,
        "precipitation_probability_max",
    )
    precipitation_sums = _read_required_list(raw_daily, "precipitation_sum")
    wind_speed_maxes = _read_required_list(raw_daily, "wind_speed_10m_max")

    expected_length = len(dates)
    series = (
        weather_codes,
        maximums,
        minimums,
        precipitation_probabilities,
        precipitation_sums,
        wind_speed_maxes,
    )
    if any(len(values) != expected_length for values in series):
        raise WeatherLookupError(
            "Weather provider returned mismatched daily forecast series lengths."
        )

    daily_forecast: list[WeatherDailyForecast] = []
    for index, raw_date in enumerate(dates):
        if not isinstance(raw_date, str) or not raw_date.strip():
            raise WeatherLookupError(
                "Weather provider returned an invalid daily forecast date."
            )

        weather_code = _coerce_required_int(weather_codes[index], key="weather_code")
        daily_forecast.append(
            WeatherDailyForecast(
                date=raw_date,
                weather_code=weather_code,
                weather_description=_describe_weather_code(weather_code),
                temperature_min_c=_coerce_required_float(
                    minimums[index],
                    key="temperature_2m_min",
                ),
                temperature_max_c=_coerce_required_float(
                    maximums[index],
                    key="temperature_2m_max",
                ),
                precipitation_probability_max_pct=_coerce_optional_int(
                    precipitation_probabilities[index],
                    key="precipitation_probability_max",
                ),
                precipitation_sum_mm=_coerce_optional_float(
                    precipitation_sums[index],
                    key="precipitation_sum",
                ),
                wind_speed_max_kph=_coerce_optional_float(
                    wind_speed_maxes[index],
                    key="wind_speed_10m_max",
                ),
            )
        )

    return tuple(daily_forecast)


def _read_required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WeatherLookupError(
            f"Weather provider returned an invalid '{key}' value."
        )
    return value.strip()


def _read_optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise WeatherLookupError(
            f"Weather provider returned an invalid '{key}' value."
        )
    normalized = value.strip()
    return normalized or None


def _read_required_float(payload: dict[str, Any], key: str) -> float:
    return _coerce_required_float(payload.get(key), key=key)


def _read_required_int(payload: dict[str, Any], key: str) -> int:
    return _coerce_required_int(payload.get(key), key=key)


def _read_required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise WeatherLookupError(
            f"Weather provider returned an invalid '{key}' series."
        )
    return value


def _coerce_required_float(value: Any, *, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WeatherLookupError(
            f"Weather provider returned an invalid '{key}' value."
        )
    return float(value)


def _coerce_optional_float(value: Any, *, key: str) -> float | None:
    if value is None:
        return None
    return _coerce_required_float(value, key=key)


def _coerce_required_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WeatherLookupError(
            f"Weather provider returned an invalid '{key}' value."
        )
    return int(round(float(value)))


def _coerce_optional_int(value: Any, *, key: str) -> int | None:
    if value is None:
        return None
    return _coerce_required_int(value, key=key)


def _format_weather_result(response: WeatherLookupResponse) -> str:
    lines = [
        f"Provider: {response.provider}",
        f"Location: {response.resolved_location.display_name()}",
        (
            "Coordinates: "
            f"{response.resolved_location.latitude:.4f}, "
            f"{response.resolved_location.longitude:.4f}"
        ),
        f"Timezone: {response.resolved_location.timezone}",
        "",
    ]

    if response.current is not None:
        lines.extend(
            (
                "Current conditions:",
                (
                    f"- {response.current.observed_at}: "
                    f"{response.current.weather_description}; "
                    f"{response.current.temperature_c:.1f} C "
                    f"(feels {response.current.apparent_temperature_c:.1f} C); "
                    f"precipitation {response.current.precipitation_mm:.1f} mm; "
                    f"wind {response.current.wind_speed_kph:.1f} km/h "
                    f"at {response.current.wind_direction_degrees} deg"
                ),
                "",
            )
        )

    lines.append(f"{response.forecast_days}-day forecast:")
    for forecast in response.daily_forecast:
        precipitation_bits: list[str] = []
        if forecast.precipitation_probability_max_pct is not None:
            precipitation_bits.append(
                f"{forecast.precipitation_probability_max_pct}% precip probability"
            )
        if forecast.precipitation_sum_mm is not None:
            precipitation_bits.append(f"{forecast.precipitation_sum_mm:.1f} mm precip")
        precipitation_summary = (
            "; " + ", ".join(precipitation_bits) if precipitation_bits else ""
        )
        wind_summary = (
            ""
            if forecast.wind_speed_max_kph is None
            else f"; wind up to {forecast.wind_speed_max_kph:.1f} km/h"
        )
        lines.append(
            (
                f"- {forecast.date}: {forecast.weather_description}; "
                f"low {forecast.temperature_min_c:.1f} C, "
                f"high {forecast.temperature_max_c:.1f} C"
                f"{precipitation_summary}{wind_summary}"
            )
        )

    return "\n".join(lines)


def _describe_weather_code(weather_code: int) -> str:
    return _WMO_WEATHER_CODE_DESCRIPTIONS.get(weather_code, "unrecognized conditions")


_WMO_WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


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
