from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from pydantic import SecretStr

from omnifusion.settings import settings


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str = ""


JsonTransport = Callable[[str, Mapping[str, str], bytes | None], Mapping[str, Any]]


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...


def _default_json_transport(
    url: str, headers: Mapping[str, str], body: bytes | None
) -> Mapping[str, Any]:
    method = "POST" if body is not None else "GET"
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _secret_value(value: SecretStr | str | None) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value or ""


def _validate_query(query: str, max_results: int) -> tuple[str, int]:
    normalized = query.strip()
    if not normalized:
        raise ValueError("search query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be >= 1")
    return normalized, min(max_results, 10)


class SearXNGSearchProvider:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: JsonTransport | None = None,
    ) -> None:
        self.base_url = (base_url or settings.omnifusion_searxng_base_url).rstrip("/")
        self.transport = transport or _default_json_transport

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        query, max_results = _validate_query(query, max_results)
        params = urllib.parse.urlencode(
            {"q": query, "format": "json", "language": "en-US"}
        )
        payload = self.transport(
            f"{self.base_url}/search?{params}",
            {"accept": "application/json"},
            None,
        )
        results = []
        for item in list(payload.get("results", []))[:max_results]:
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("content", "")),
                    source="searxng",
                )
            )
        return results


class TavilySearchProvider:
    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        transport: JsonTransport | None = None,
    ) -> None:
        self.api_key = _secret_value(api_key or settings.omnifusion_tavily_api_key)
        self.transport = transport or _default_json_transport

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise ValueError("Tavily search requires OMNIFUSION_TAVILY_API_KEY")
        query, max_results = _validate_query(query, max_results)
        body = json.dumps({"query": query, "max_results": max_results}).encode("utf-8")
        payload = self.transport(
            "https://api.tavily.com/search",
            {
                "accept": "application/json",
                "authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            body,
        )
        results = []
        for item in list(payload.get("results", []))[:max_results]:
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("content", "")),
                    source="tavily",
                )
            )
        return results


class BraveSearchProvider:
    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        transport: JsonTransport | None = None,
    ) -> None:
        self.api_key = _secret_value(api_key or settings.omnifusion_brave_api_key)
        self.transport = transport or _default_json_transport

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise ValueError("Brave search requires OMNIFUSION_BRAVE_API_KEY")
        query, max_results = _validate_query(query, max_results)
        params = urllib.parse.urlencode({"q": query, "count": max_results})
        payload = self.transport(
            f"https://api.search.brave.com/res/v1/web/search?{params}",
            {
                "accept": "application/json",
                "x-subscription-token": self.api_key,
            },
            None,
        )
        results = []
        web = payload.get("web", {})
        if isinstance(web, Mapping):
            for item in list(web.get("results", []))[:max_results]:
                results.append(
                    SearchResult(
                        title=str(item.get("title", "")),
                        url=str(item.get("url", "")),
                        snippet=str(item.get("description", "")),
                        source="brave",
                    )
                )
        return results


class CustomSearchProvider:
    def __init__(self, adapter: SearchProvider | Callable[[str, int], list[SearchResult]]):
        self.adapter = adapter

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if callable(self.adapter):
            return self.adapter(query, max_results)
        return self.adapter.search(query, max_results)


def build_search_provider(
    provider: str | None = None,
    *,
    custom_provider: SearchProvider | Callable[[str, int], list[SearchResult]] | None = None,
) -> SearchProvider:
    provider_name = (provider or settings.omnifusion_web_search_provider).strip().lower()
    if provider_name == "searxng":
        return SearXNGSearchProvider()
    if provider_name == "tavily":
        return TavilySearchProvider()
    if provider_name == "brave":
        return BraveSearchProvider()
    if provider_name == "custom" and custom_provider is not None:
        return CustomSearchProvider(custom_provider)
    raise ValueError(f"Unknown web search provider: {provider_name}")
