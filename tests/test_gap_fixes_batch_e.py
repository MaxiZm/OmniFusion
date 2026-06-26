"""M5 web-fetch gaps (Batch E): full-page retention flag + application/pdf MIME."""
from omnifusion.settings import settings
from omnifusion.tools.web import _ALLOWED_MIME_TYPES, WebFetcher, WebResponse


def _fetcher(**kwargs):
    return WebFetcher(
        resolver=lambda host: ["93.184.216.34"],
        cache_ttl_seconds=0,
        per_domain_interval_seconds=0,
        **kwargs,
    )


def _transport(body: bytes, content_type: str):
    def transport(url, headers):
        return WebResponse(status=200, url=url, headers={"content-type": content_type}, body=body)

    return transport


def test_pdf_is_in_mime_allowlist():
    assert "application/pdf" in _ALLOWED_MIME_TYPES


def test_pdf_fetch_extracts_text(monkeypatch):
    """application/pdf is fetched and text-extracted (pdf-text MIME support)."""
    import omnifusion.tools.web as web_mod

    monkeypatch.setattr(web_mod, "_extract_pdf_text", lambda body: "EXTRACTED PDF TEXT")
    fetcher = _fetcher(transport=_transport(b"%PDF-1.4 ...", "application/pdf"))
    result = fetcher.fetch("https://example.com/doc.pdf")
    assert result.mime_type == "application/pdf"
    assert "EXTRACTED PDF TEXT" in result.excerpt
    assert "EXTRACTED PDF TEXT" in result.fenced_content


def test_full_page_retention_off_by_default():
    fetcher = _fetcher(transport=_transport(b"<p>hello world body</p>", "text/html"))
    result = fetcher.fetch("https://example.com/page")
    assert "full_content" not in result.trace_metadata
    assert result.trace_metadata["excerpt"]  # bounded excerpt still present


def test_full_page_retention_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "omnifusion_web_fetch_store_full_page", True)
    fetcher = _fetcher(transport=_transport(b"<p>hello world body</p>", "text/html"))
    result = fetcher.fetch("https://example.com/page")
    assert "full_content" in result.trace_metadata
    assert "hello world body" in result.trace_metadata["full_content"]
