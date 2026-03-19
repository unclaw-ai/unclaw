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

    concurrent_future = _BLOCKING_EXECUTOR.submit(
        partial(_run_catching, func, *args, **kwargs)
    )

    try:
        while not concurrent_future.done():
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        concurrent_future.cancel()
        raise

    succeeded, value = concurrent_future.result()
    if succeeded:
        return cast(_T, value)
    raise cast(BaseException, value)
