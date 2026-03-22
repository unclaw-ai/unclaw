# Weather

Live weather and short forecasts.

Tool hints: Prefer `get_weather`; use `search_web` only as fallback for official alerts or missing details.

## When to use

Use this skill for current weather conditions, short-range forecasts (up to 7 days), and weather-sensitive planning questions.

## How to use

For current weather or short-forecast questions, call `get_weather` before stating any live weather details.

Use a precise place name for the lookup. Answer from the returned current conditions and 7-day forecast. State any assumption if the user was vague about location.

When `get_weather` returns a selected forecast day or a relative-day anchor like today or tomorrow, answer from that explicit day — do not remap dates yourself.

Use `search_web` only as a fallback for official alerts, longer-range outlooks, or when `get_weather` cannot resolve the requested place or detail.

## Safety

Do not present temperature, precipitation, alerts, or forecast details as certain unless they are grounded by `get_weather` or search results from this conversation.
