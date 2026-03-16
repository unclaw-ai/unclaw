"""Small helpers for exposing blocking APIs through async boundaries."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from functools import partial
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


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
    bound_call = partial(func, *args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(executor, bound_call)
