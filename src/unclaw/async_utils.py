"""Small helpers for exposing blocking APIs through async boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import ParamSpec, TypeVar, cast

_P = ParamSpec("_P")
_T = TypeVar("_T")
_BLOCKING_EXECUTOR = ThreadPoolExecutor(thread_name_prefix="unclaw-blocking")


def _run_catching(
    func: Callable[_P, _T],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> tuple[bool, _T | BaseException]:
    try:
        return True, func(*args, **kwargs)
    except BaseException as exc:
        return False, exc


async def run_blocking(
    func: Callable[_P, _T],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T:
    """Run one blocking call in a worker thread.

    This keeps the existing synchronous implementation intact while giving
    callers an awaitable boundary when they already run inside an event loop.
    """

    loop = asyncio.get_running_loop()
    succeeded, value = await loop.run_in_executor(
        _BLOCKING_EXECUTOR,
        partial(_run_catching, func, *args, **kwargs),
    )
    if succeeded:
        return cast(_T, value)
    raise cast(BaseException, value)
