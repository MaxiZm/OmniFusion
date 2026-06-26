from __future__ import annotations

import hashlib
import html
import ipaddress
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from typing import Callable, Mapping

from omnifusion.fusion.prompts import new_prompt_nonce
from omnifusion.settings import settings


@dataclass(frozen=True)
class WebResponse:
    status: int
    url: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class WebFetchResult:
    url: str
    final_url: str
    mime_type: str
    content_hash: str
    excerpt: str
    truncated: bool
    fenced_content: str
    trace_metadata: dict[str, object]


Transport = Callable[[str, Mapping[str, str]], WebResponse]
Resolver = Callable[[str], list[str]]
Clock = Callable[[], float]


@dataclass(frozen=True)
class _CacheEntry:
    expires_at: float
    result: WebFetchResult


_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_MIME_TYPES = {
    "application/json",
    "application/xhtml+xml",
    "application/xml",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/xml",
}
_METADATA_IPS = {
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("fd00:ec2::254"),
}
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None


def _default_transport(url: str, headers: Mapping[str, str]) -> WebResponse:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    opener = urllib.request.build_opener(_NoRedirectHandler)
    try:
        with opener.open(request, timeout=15) as resp:
            return WebResponse(
                status=resp.status,
                url=resp.geturl(),
                headers=dict(resp.headers.items()),
                body=resp.read(),
            )
    except urllib.error.HTTPError as exc:
        return WebResponse(
            status=exc.code,
            url=exc.geturl(),
            headers=dict(exc.headers.items()),
            body=exc.read(),
        )


def _default_resolver(hostname: str) -> list[str]:
    addr_info = socket.getaddrinfo(hostname, None)
    resolved_ips = []
    for _family, _socktype, _proto, _canonname, sockaddr in addr_info:
        if sockaddr:
            resolved_ips.append(sockaddr[0])
    return resolved_ips


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def _media_type(content_type: str | None) -> str:
    if not content_type:
        return "text/plain"
    return content_type.split(";", 1)[0].strip().lower()


def _clean_html(raw_text: str, mime_type: str) -> str:
    if mime_type not in {"text/html", "application/xhtml+xml"}:
        return raw_text

    without_active_markup = _SCRIPT_STYLE_RE.sub("", raw_text)
    without_tags = _TAG_RE.sub(" ", without_active_markup)
    return html.unescape(without_tags)


def _bounded_text(value: str, max_chars: int) -> str:
    collapsed = re.sub(r"\s+", " ", value).strip()
    return collapsed[:max_chars]


def _ip_from_literal(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(hostname.strip("[]").split("%", 1)[0])
    except ValueError:
        return None


def _validate_ip_allowed(ip_value: str) -> None:
    ip_obj = ipaddress.ip_address(ip_value.strip("[]").split("%", 1)[0])

    if ip_obj in _METADATA_IPS:
        raise ValueError(f"web fetch to private cloud metadata address {ip_value} is blocked")

    if settings.omnifusion_allow_private_egress:
        return

    if (
        ip_obj.is_loopback
        or ip_obj.is_private
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    ):
        raise ValueError(f"web fetch to private or local address {ip_value} is blocked")


def _domain_key(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return (parsed.hostname or "").lower()


def _with_cache_hit(result: WebFetchResult, cache_hit: bool) -> WebFetchResult:
    trace_metadata = dict(result.trace_metadata)
    trace_metadata["cache_hit"] = cache_hit
    return replace(result, trace_metadata=trace_metadata)


class WebFetcher:
    def __init__(
        self,
        *,
        transport: Transport | None = None,
        resolver: Resolver | None = None,
        max_redirects: int = 3,
        max_content_bytes: int = 1_000_000,
        excerpt_chars: int = 2_048,
        nonce: str | None = None,
        cache_ttl_seconds: float | None = None,
        per_domain_interval_seconds: float | None = None,
        now: Clock | None = None,
    ) -> None:
        if max_redirects < 0:
            raise ValueError("max_redirects must be >= 0")
        if max_content_bytes < 1:
            raise ValueError("max_content_bytes must be >= 1")
        if excerpt_chars < 1:
            raise ValueError("excerpt_chars must be >= 1")

        cache_ttl = (
            settings.omnifusion_web_fetch_cache_ttl_seconds
            if cache_ttl_seconds is None
            else cache_ttl_seconds
        )
        domain_interval = (
            settings.omnifusion_web_fetch_per_domain_interval_seconds
            if per_domain_interval_seconds is None
            else per_domain_interval_seconds
        )
        if cache_ttl < 0:
            raise ValueError("cache_ttl_seconds must be >= 0")
        if domain_interval < 0:
            raise ValueError("per_domain_interval_seconds must be >= 0")

        self.transport = transport or _default_transport
        self.resolver = resolver or _default_resolver
        self.max_redirects = max_redirects
        self.max_content_bytes = max_content_bytes
        self.excerpt_chars = excerpt_chars
        self.nonce = nonce
        self.cache_ttl_seconds = cache_ttl
        self.per_domain_interval_seconds = domain_interval
        self.now = now or time.monotonic
        self._cache: dict[str, _CacheEntry] = {}
        self._last_fetch_by_domain: dict[str, float] = {}

    def fetch(self, url: str) -> WebFetchResult:
        initial_url = self._validate_url(url)
        cached = self._read_cache(initial_url)
        if cached is not None:
            return cached

        current_url = initial_url
        headers = {
            "accept": ", ".join(sorted(_ALLOWED_MIME_TYPES)),
            "user-agent": "OmniFusion-WebFetch/0.1",
        }
        domains_checked_this_fetch: set[str] = set()

        for _redirect_count in range(self.max_redirects + 1):
            self._enforce_domain_rate_limit(current_url, domains_checked_this_fetch)
            response = self.transport(current_url, headers)
            response_url = self._validate_url(response.url or current_url)
            normalized_headers = _normalize_headers(response.headers)

            if 300 <= response.status < 400 and normalized_headers.get("location"):
                current_url = self._validate_url(
                    urllib.parse.urljoin(response_url, normalized_headers["location"])
                )
                continue

            result = self._to_result(
                requested_url=initial_url,
                final_url=response_url,
                response=response,
                headers=normalized_headers,
            )
            self._write_cache(initial_url, result)
            return result

        raise ValueError("web fetch exceeded redirect limit")

    def _validate_url(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise ValueError("web fetch URL scheme must be http or https")
        if not parsed.hostname:
            raise ValueError("web fetch URL must include a hostname")

        literal_ip = _ip_from_literal(parsed.hostname)
        if literal_ip is not None:
            _validate_ip_allowed(str(literal_ip))
            return url

        resolved_ips = self.resolver(parsed.hostname)
        if not resolved_ips:
            raise ValueError(f"web fetch hostname '{parsed.hostname}' resolved to no addresses")
        for resolved_ip in set(resolved_ips):
            _validate_ip_allowed(resolved_ip)
        return url

    def _read_cache(self, url: str) -> WebFetchResult | None:
        if self.cache_ttl_seconds <= 0:
            return None
        entry = self._cache.get(url)
        if entry is None:
            return None
        if entry.expires_at <= self.now():
            self._cache.pop(url, None)
            return None
        return _with_cache_hit(entry.result, True)

    def _write_cache(self, url: str, result: WebFetchResult) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        self._cache[url] = _CacheEntry(
            expires_at=self.now() + self.cache_ttl_seconds,
            result=result,
        )

    def _enforce_domain_rate_limit(
        self, url: str, domains_checked_this_fetch: set[str]
    ) -> None:
        if self.per_domain_interval_seconds <= 0:
            return
        domain = _domain_key(url)
        if not domain or domain in domains_checked_this_fetch:
            return
        domains_checked_this_fetch.add(domain)

        now = self.now()
        last_fetch = self._last_fetch_by_domain.get(domain)
        if last_fetch is not None and now - last_fetch < self.per_domain_interval_seconds:
            raise ValueError(f"web fetch rate limit for domain '{domain}'")
        self._last_fetch_by_domain[domain] = now

    def _to_result(
        self,
        *,
        requested_url: str,
        final_url: str,
        response: WebResponse,
        headers: Mapping[str, str],
    ) -> WebFetchResult:
        mime_type = _media_type(headers.get("content-type"))
        if mime_type not in _ALLOWED_MIME_TYPES:
            raise ValueError(f"web fetch MIME type '{mime_type}' is not allowed")

        content_hash = "sha256:" + hashlib.sha256(response.body).hexdigest()
        truncated = len(response.body) > self.max_content_bytes
        bounded_body = response.body[: self.max_content_bytes]
        decoded = bounded_body.decode("utf-8", errors="replace")
        cleaned = _clean_html(decoded, mime_type)
        excerpt = _bounded_text(cleaned, self.excerpt_chars)
        nonce = self.nonce or new_prompt_nonce()

        fenced_content = "\n".join(
            [
                f"--- START OF WEB_FETCH_SOURCE (ID: {nonce}) ---",
                f"Source: {final_url}",
                f"MIME: {mime_type}",
                f"Truncated: {str(truncated).lower()}",
                "",
                excerpt,
                f"--- END OF WEB_FETCH_SOURCE (ID: {nonce}) ---",
            ]
        )
        trace_metadata: dict[str, object] = {
            "url": requested_url,
            "final_url": final_url,
            "status": response.status,
            "mime_type": mime_type,
            "content_hash": content_hash,
            "excerpt": excerpt,
            "truncated": truncated,
            "cache_hit": False,
        }

        return WebFetchResult(
            url=requested_url,
            final_url=final_url,
            mime_type=mime_type,
            content_hash=content_hash,
            excerpt=excerpt,
            truncated=truncated,
            fenced_content=fenced_content,
            trace_metadata=trace_metadata,
        )
