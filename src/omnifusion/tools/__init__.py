"""Server-side tools owned by OmniFusion (web_search / web_fetch).

These are the *only* tools OmniFusion owns; client-side tools belong to the client.
They are wired into the fusion panel via fusion.web_grounding when web is enabled.
"""
from .search import (
    BraveSearchProvider,
    CustomSearchProvider,
    SearchProvider,
    SearchResult,
    SearXNGSearchProvider,
    TavilySearchProvider,
    build_search_provider,
)
from .web import WebFetcher, WebFetchResult

__all__ = [
    "BraveSearchProvider",
    "CustomSearchProvider",
    "SearchProvider",
    "SearchResult",
    "SearXNGSearchProvider",
    "TavilySearchProvider",
    "WebFetcher",
    "WebFetchResult",
    "build_search_provider",
]
