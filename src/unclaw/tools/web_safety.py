"""SSRF protection, IP blocking, hostname validation, and redirect safety."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

_BLOCKED_FETCH_HOSTS = {
    "instance-data",
    "instance-data.ec2.internal",
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.google.internal",
}
_BLOCKED_FETCH_IPS = {"100.100.100.200"}


class _BlockedFetchTargetError(ValueError):
    """Raised when a fetch target is blocked by the safe default policy."""


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Reject redirects that escape the public-network fetch policy."""

    def __init__(self, *, allow_private_networks: bool) -> None:
        super().__init__()
        self._allow_private_networks = allow_private_networks

    def redirect_request(
        self,
        req: Request,
        fp,  # type: ignore[no-untyped-def]
        code: int,
        msg: str,
        headers,
        newurl: str,
    ) -> Request | None:
        _ensure_fetch_target_allowed(
            newurl,
            allow_private_networks=self._allow_private_networks,
        )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _is_supported_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _open_request(
    request: Request,
    *,
    timeout_seconds: float,
    allow_private_networks: bool,
):
    opener = build_opener(
        _SafeRedirectHandler(
            allow_private_networks=allow_private_networks,
        )
    )
    return opener.open(request, timeout=timeout_seconds)


def _ensure_fetch_target_allowed(
    url: str,
    *,
    allow_private_networks: bool,
) -> None:
    if not _is_supported_url(url):
        raise _BlockedFetchTargetError(
            "Only HTTP and HTTPS fetch targets are supported."
        )
    if allow_private_networks:
        return

    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        raise _BlockedFetchTargetError(
            "Could not determine which host to fetch."
        )

    normalized_host = hostname.rstrip(".").lower()
    if _is_blocked_hostname(normalized_host):
        raise _BlockedFetchTargetError(
            _build_blocked_fetch_message(
                target=normalized_host,
                reason="local or metadata-style hosts are blocked by default",
            )
        )

    literal_ip = _parse_ip_address(normalized_host)
    if literal_ip is not None:
        _raise_if_blocked_ip(literal_ip, target=normalized_host)
        return

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    for address in _resolve_host_addresses(normalized_host, port):
        _raise_if_blocked_ip(address, target=normalized_host)


def _is_blocked_hostname(hostname: str) -> bool:
    return hostname in _BLOCKED_FETCH_HOSTS or hostname.endswith(".localhost")


def _parse_ip_address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _resolve_host_addresses(
    hostname: str,
    port: int,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        address_infos = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise _BlockedFetchTargetError(
            f"Could not resolve '{hostname}' while checking safe fetch rules: {exc}."
        ) from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for _family, _socktype, _proto, _canonname, sockaddr in address_infos:
        host = sockaddr[0]
        address = ipaddress.ip_address(host)
        if address not in addresses:
            addresses.append(address)
    return tuple(addresses)


def _raise_if_blocked_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    target: str,
) -> None:
    if not _is_blocked_ip(address):
        return
    raise _BlockedFetchTargetError(
        _build_blocked_fetch_message(
            target=target,
            reason=f"{address.compressed} is on a local or private network",
        )
    )


def _is_blocked_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if address.compressed in _BLOCKED_FETCH_IPS:
        return True
    return any(
        (
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_private,
            address.is_reserved,
            address.is_unspecified,
        )
    )


def _build_blocked_fetch_message(*, target: str, reason: str) -> str:
    return (
        f"Fetching '{target}' is blocked because {reason}. "
        "Only public HTTP and HTTPS targets are allowed by default. "
        "The local owner can relax `security.tools.fetch.allow_private_networks` "
        "in config/app.yaml if needed."
    )
