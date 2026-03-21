"""Skill-owned dedicated live weather lookup tool backed by Open-Meteo APIs."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from unclaw.async_utils import run_blocking
from unclaw.skills.weather.contracts import (
    WeatherCurrentConditionsPayload,
    WeatherDailyForecastPayload,
    WeatherLookupPayload,
    WeatherRelativeDayAnchorPayload,
    WeatherResolvedLocationPayload,
    WeatherTemporalRequestPayload,
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
_WEATHER_TEMPORAL_PATTERNS: tuple[tuple[str, bool, re.Pattern[str]], ...] = (
    ("today", True, re.compile(r"\btoday\b", flags=re.IGNORECASE)),
    ("today", True, re.compile(r"\baujourd['’]hui\b", flags=re.IGNORECASE)),
    ("tomorrow", True, re.compile(r"\btomorrow\b", flags=re.IGNORECASE)),
    ("tomorrow", True, re.compile(r"\bdemain\b", flags=re.IGNORECASE)),
    (
        "this_weekend",
        False,
        re.compile(r"\bthis\s+weekend\b", flags=re.IGNORECASE),
    ),
    (
        "this_weekend",
        False,
        re.compile(r"\bce(?:\s+|-)?week-?end\b", flags=re.IGNORECASE),
    ),
)
_LEADING_LOCATION_GLUE_PATTERNS = (
    re.compile(r"^(?:for|in|at|on)\s+", flags=re.IGNORECASE),
    re.compile(r"^(?:pour|a|à|au|en|sur)\s+", flags=re.IGNORECASE),
)
_TRAILING_LOCATION_GLUE_PATTERNS = (
    re.compile(r"\s+(?:for|in|at|on)$", flags=re.IGNORECASE),
    re.compile(r"\s+(?:pour|a|à|au|en|sur)$", flags=re.IGNORECASE),
)
_LOCATION_EDGE_TRIM_CHARS = " ,:;!?-"
_UNSUPPORTED_RELATIVE_RANGE_NOTE = (
    "Unsupported relative weather range; no single forecast day selected."
)


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
    temporal_request: WeatherTemporalRequest | None = None


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
class WeatherRelativeDayAnchor:
    """Relative-day anchor that maps a label like today/tomorrow to a forecast."""

    label: str
    forecast: WeatherDailyForecast

    def to_payload(self) -> WeatherRelativeDayAnchorPayload:
        return WeatherRelativeDayAnchorPayload(
            label=self.label,
            date=self.forecast.date,
            forecast=self.forecast.to_payload(),
        )


@dataclass(frozen=True, slots=True)
class WeatherTemporalRequest:
    """Resolved temporal request metadata for a weather question."""

    raw_text: str
    normalized_kind: str
    local_current_date: str
    resolved_target_date: str | None
    supported: bool
    note: str | None = None

    def to_payload(self) -> WeatherTemporalRequestPayload:
        return WeatherTemporalRequestPayload(**asdict(self))


@dataclass(frozen=True, slots=True)
class WeatherLookupResponse:
    """Typed weather response returned by the dedicated backend path."""

    provider: str
    location_query: str
    forecast_days: int
    local_current_date: str
    resolved_location: WeatherResolvedLocation
    current: WeatherCurrentConditions | None
    relative_day_anchors: tuple[WeatherRelativeDayAnchor, ...]
    temporal_request: WeatherTemporalRequest | None
    selected_forecast: WeatherDailyForecast | None
    daily_forecast: tuple[WeatherDailyForecast, ...]

    def to_payload(self) -> WeatherLookupPayload:
        return WeatherLookupPayload(
            provider=self.provider,
            location_query=self.location_query,
            forecast_days=self.forecast_days,
            local_current_date=self.local_current_date,
            resolved_location=self.resolved_location.to_payload(),
            current=None if self.current is None else self.current.to_payload(),
            relative_day_anchors=[
                anchor.to_payload() for anchor in self.relative_day_anchors
            ],
            temporal_request=(
                None
                if self.temporal_request is None
                else self.temporal_request.to_payload()
            ),
            selected_forecast=(
                None
                if self.selected_forecast is None
                else self.selected_forecast.to_payload()
            ),
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
    current_local_date = _resolve_request_local_date(request)
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

    relative_day_anchors = _build_relative_day_anchors(
        daily_forecast,
        current_local_date=current_local_date,
    )
    selected_forecast = _select_forecast_for_temporal_request(
        daily_forecast,
        request.temporal_request,
    )

    return WeatherLookupResponse(
        provider=_WEATHER_PROVIDER_NAME,
        location_query=request.location,
        forecast_days=request.forecast_days,
        local_current_date=current_local_date.isoformat(),
        resolved_location=resolved_location,
        current=current_conditions,
        relative_day_anchors=relative_day_anchors,
        temporal_request=request.temporal_request,
        selected_forecast=selected_forecast,
        daily_forecast=daily_forecast,
    )


def _read_weather_request(call: ToolCall) -> WeatherLookupRequest:
    location = call.arguments.get("location")
    if not isinstance(location, str) or not location.strip():
        raise WeatherLookupError("Argument 'location' must be a non-empty string.")
    raw_location = location.strip()
    current_local_date = date.today()
    temporal_request = resolve_weather_temporal_request(
        raw_location,
        current_local_date=current_local_date,
    )
    normalized_location = (
        _strip_weather_temporal_phrase(raw_location)
        if temporal_request is not None
        else raw_location
    )
    if not normalized_location:
        raise WeatherLookupError("Argument 'location' must include a place name.")
    return WeatherLookupRequest(
        location=normalized_location,
        temporal_request=temporal_request,
    )


def resolve_weather_temporal_request(
    text: str,
    *,
    current_local_date: date | None = None,
) -> WeatherTemporalRequest | None:
    """Resolve supported relative weather phrases against the local Python date."""

    match = _find_weather_temporal_match(text)
    if match is None:
        return None

    current_date = current_local_date or date.today()
    matched_text, normalized_kind, supported = match
    resolved_target_date: str | None = None
    note: str | None = None

    if normalized_kind == "today":
        resolved_target_date = current_date.isoformat()
    elif normalized_kind == "tomorrow":
        resolved_target_date = (current_date + timedelta(days=1)).isoformat()
    else:
        note = _UNSUPPORTED_RELATIVE_RANGE_NOTE

    return WeatherTemporalRequest(
        raw_text=matched_text,
        normalized_kind=normalized_kind,
        local_current_date=current_date.isoformat(),
        resolved_target_date=resolved_target_date,
        supported=supported,
        note=note,
    )


def _find_weather_temporal_match(
    text: str,
) -> tuple[str, str, bool] | None:
    best_match: tuple[int, int, str, str, bool] | None = None
    for normalized_kind, supported, pattern in _WEATHER_TEMPORAL_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        candidate = (
            match.start(),
            -(match.end() - match.start()),
            match.group(0).strip(),
            normalized_kind,
            supported,
        )
        if best_match is None or candidate < best_match:
            best_match = candidate

    if best_match is None:
        return None
    _, _, raw_text, normalized_kind, supported = best_match
    return raw_text, normalized_kind, supported


def _strip_weather_temporal_phrase(text: str) -> str:
    stripped = text
    for _, _, pattern in _WEATHER_TEMPORAL_PATTERNS:
        stripped = pattern.sub(" ", stripped, count=1)
    stripped = _WHITESPACE_PATTERN.sub(" ", stripped).strip(_LOCATION_EDGE_TRIM_CHARS)
    for _ in range(4):
        previous = stripped
        for pattern in _LEADING_LOCATION_GLUE_PATTERNS:
            stripped = pattern.sub("", stripped)
        for pattern in _TRAILING_LOCATION_GLUE_PATTERNS:
            stripped = pattern.sub("", stripped)
        stripped = _WHITESPACE_PATTERN.sub(" ", stripped).strip(_LOCATION_EDGE_TRIM_CHARS)
        if stripped == previous:
            break
    return stripped


def _resolve_request_local_date(request: WeatherLookupRequest) -> date:
    if request.temporal_request is None:
        return date.today()
    return date.fromisoformat(request.temporal_request.local_current_date)


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


def _build_relative_day_anchors(
    daily_forecast: tuple[WeatherDailyForecast, ...],
    *,
    current_local_date: date,
) -> tuple[WeatherRelativeDayAnchor, ...]:
    anchored_labels = (
        ("today", current_local_date.isoformat()),
        ("tomorrow", (current_local_date + timedelta(days=1)).isoformat()),
    )
    anchors: list[WeatherRelativeDayAnchor] = []
    for label, forecast_date in anchored_labels:
        forecast = _find_forecast_by_date(daily_forecast, forecast_date)
        if forecast is None:
            continue
        anchors.append(WeatherRelativeDayAnchor(label=label, forecast=forecast))
    return tuple(anchors)


def _select_forecast_for_temporal_request(
    daily_forecast: tuple[WeatherDailyForecast, ...],
    temporal_request: WeatherTemporalRequest | None,
) -> WeatherDailyForecast | None:
    if temporal_request is None or temporal_request.supported is not True:
        return None
    if temporal_request.resolved_target_date is None:
        return None
    return _find_forecast_by_date(
        daily_forecast,
        temporal_request.resolved_target_date,
    )


def _find_forecast_by_date(
    daily_forecast: tuple[WeatherDailyForecast, ...],
    forecast_date: str,
) -> WeatherDailyForecast | None:
    for forecast in daily_forecast:
        if forecast.date == forecast_date:
            return forecast
    return None


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
        f"Local current date: {response.local_current_date}",
        "",
    ]

    if response.temporal_request is not None:
        lines.extend(_format_temporal_request_lines(response))
        lines.append("")
    elif response.relative_day_anchors:
        lines.extend(_format_relative_day_anchor_lines(response.relative_day_anchors))
        lines.append("")

    if response.current is not None:
        lines.extend(
            (
                "Current conditions:",
                _format_current_conditions(response.current),
                "",
            )
        )

    relative_labels_by_date = {
        anchor.forecast.date: anchor.label for anchor in response.relative_day_anchors
    }
    lines.append(f"{response.forecast_days}-day forecast:")
    for forecast in response.daily_forecast:
        lines.append(
            _format_forecast_summary(
                forecast,
                label=relative_labels_by_date.get(forecast.date),
            )
        )

    return "\n".join(lines)


def _format_temporal_request_lines(
    response: WeatherLookupResponse,
) -> tuple[str, ...]:
    temporal_request = response.temporal_request
    if temporal_request is None:
        return ()

    lines = [
        f"Requested temporal phrase: {temporal_request.raw_text}",
    ]
    if temporal_request.supported is not True:
        lines.append(
            "Temporal grounding: "
            f"{temporal_request.note or _UNSUPPORTED_RELATIVE_RANGE_NOTE}"
        )
        return tuple(lines)

    if temporal_request.resolved_target_date is not None:
        lines.append(f"Resolved target date: {temporal_request.resolved_target_date}")

    if response.selected_forecast is not None:
        lines.extend(
            (
                "Selected forecast day:",
                _format_forecast_summary(
                    response.selected_forecast,
                    label=temporal_request.normalized_kind,
                ),
            )
        )
    else:
        lines.append(
            "Selected forecast day: no matching entry in the returned 7-day forecast."
        )
    return tuple(lines)


def _format_relative_day_anchor_lines(
    relative_day_anchors: tuple[WeatherRelativeDayAnchor, ...],
) -> tuple[str, ...]:
    if not relative_day_anchors:
        return ()
    return (
        "Relative forecast anchors:",
        *(
            _format_forecast_summary(anchor.forecast, label=anchor.label)
            for anchor in relative_day_anchors
        ),
    )


def _format_current_conditions(current: WeatherCurrentConditions) -> str:
    return (
        f"- {current.observed_at}: "
        f"{current.weather_description}; "
        f"{current.temperature_c:.1f} C "
        f"(feels {current.apparent_temperature_c:.1f} C); "
        f"precipitation {current.precipitation_mm:.1f} mm; "
        f"wind {current.wind_speed_kph:.1f} km/h "
        f"at {current.wind_direction_degrees} deg"
    )


def _format_forecast_summary(
    forecast: WeatherDailyForecast,
    *,
    label: str | None = None,
) -> str:
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
    label_suffix = f" ({label})" if label else ""
    return (
        f"- {forecast.date}{label_suffix}: {forecast.weather_description}; "
        f"low {forecast.temperature_min_c:.1f} C, "
        f"high {forecast.temperature_max_c:.1f} C"
        f"{precipitation_summary}{wind_summary}"
    )


def ground_weather_tool_result(
    result: ToolResult,
    *,
    user_input: str,
) -> ToolResult:
    """Append lightweight temporal grounding for the current weather request."""

    if result.tool_name != GET_WEATHER_DEFINITION.name or result.success is not True:
        return result
    if not isinstance(result.payload, dict):
        return result

    payload = dict(result.payload)
    existing_temporal_request = payload.get("temporal_request")
    if isinstance(existing_temporal_request, dict) and existing_temporal_request:
        return result

    current_local_date = _parse_iso_date(payload.get("local_current_date"))
    temporal_request = resolve_weather_temporal_request(
        user_input,
        current_local_date=current_local_date,
    )
    if temporal_request is None:
        return result

    selected_forecast_payload = _find_selected_forecast_payload(
        payload,
        temporal_request=temporal_request,
    )
    grounding_lines = _format_temporal_grounding_lines(
        temporal_request=temporal_request,
        selected_forecast_payload=selected_forecast_payload,
    )
    if not grounding_lines:
        return result

    payload["temporal_request"] = temporal_request.to_payload()
    payload["selected_forecast"] = selected_forecast_payload
    output_text = result.output_text.rstrip()
    if output_text:
        output_text = f"{output_text}\n\n" + "\n".join(grounding_lines)
    else:
        output_text = "\n".join(grounding_lines)
    return ToolResult(
        tool_name=result.tool_name,
        success=result.success,
        output_text=output_text,
        payload=payload,
        error=result.error,
    )


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _find_selected_forecast_payload(
    payload: dict[str, Any],
    *,
    temporal_request: WeatherTemporalRequest,
) -> WeatherDailyForecastPayload | None:
    if temporal_request.supported is not True:
        return None
    target_date = temporal_request.resolved_target_date
    if not isinstance(target_date, str) or not target_date:
        return None

    raw_selected_forecast = payload.get("selected_forecast")
    if isinstance(raw_selected_forecast, dict):
        forecast_date = raw_selected_forecast.get("date")
        if isinstance(forecast_date, str) and forecast_date == target_date:
            return WeatherDailyForecastPayload(**raw_selected_forecast)

    raw_daily_forecast = payload.get("daily_forecast")
    if not isinstance(raw_daily_forecast, list):
        return None
    for raw_entry in raw_daily_forecast:
        if not isinstance(raw_entry, dict):
            continue
        forecast_date = raw_entry.get("date")
        if isinstance(forecast_date, str) and forecast_date == target_date:
            return WeatherDailyForecastPayload(**raw_entry)
    return None


def _format_temporal_grounding_lines(
    *,
    temporal_request: WeatherTemporalRequest,
    selected_forecast_payload: WeatherDailyForecastPayload | None,
) -> tuple[str, ...]:
    lines = [f"Requested temporal phrase: {temporal_request.raw_text}"]
    if temporal_request.supported is not True:
        lines.append(
            "Temporal grounding: "
            f"{temporal_request.note or _UNSUPPORTED_RELATIVE_RANGE_NOTE}"
        )
        return tuple(lines)

    if temporal_request.resolved_target_date is not None:
        lines.append(f"Resolved target date: {temporal_request.resolved_target_date}")

    if selected_forecast_payload is None:
        lines.append(
            "Selected forecast day: no matching entry in the returned 7-day forecast."
        )
        return tuple(lines)

    lines.extend(
        (
            "Selected forecast day:",
            _format_forecast_payload_summary(
                selected_forecast_payload,
                label=temporal_request.normalized_kind,
            ),
        )
    )
    return tuple(lines)


def _format_forecast_payload_summary(
    forecast_payload: WeatherDailyForecastPayload,
    *,
    label: str | None = None,
) -> str:
    return _format_forecast_summary(
        WeatherDailyForecast(
            date=forecast_payload["date"],
            weather_code=forecast_payload["weather_code"],
            weather_description=forecast_payload["weather_description"],
            temperature_min_c=forecast_payload["temperature_min_c"],
            temperature_max_c=forecast_payload["temperature_max_c"],
            precipitation_probability_max_pct=forecast_payload[
                "precipitation_probability_max_pct"
            ],
            precipitation_sum_mm=forecast_payload["precipitation_sum_mm"],
            wind_speed_max_kph=forecast_payload["wind_speed_max_kph"],
        ),
        label=label,
    )


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
