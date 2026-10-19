import asyncio
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import httpx
from streamlit.testing.v1 import AppTest

from app.core.enums import AssetType, ProviderType
from app.services.news_service import NewsService
from app.services.orchestrator import collect_headlines, analyze_collection
from app.streamlit_support import collection_rows, collection_csv, analysis_window


def rss():
    date = format_datetime(datetime.now(timezone.utc))
    return f"""<rss version="2.0"><channel><title>Example</title><item>
    <title>Apple earnings rise after strong sales</title>
    <link>https://example.com/apple-earnings</link>
    <description>Apple reported stronger quarterly profit.</description>
    <pubDate>{date}</pubDate><source url="https://example.com">Example News</source>
    </item></channel></rss>"""


def sample_collection():
    article = NewsService.parse_rss_feed_content(rss(), "Example", ProviderType.DIRECT_RSS)[0]
    providers = {"Example RSS": {"status": "SUCCESS", "count": 1, "fresh_count": 1,
                                "url": "https://example.com/rss", "error": ""}}
    with patch("app.services.orchestrator.NewsService.fetch_news_for_session", return_value=( [article], providers)):
        return asyncio.run(collect_headlines("AAPL", "Last 24 hours", 40))


def test_collect_does_not_call_quote_resolution_or_model():
    with patch("app.services.ticker_service.TickerService.validate_and_resolve",
               side_effect=AssertionError("quote API must not block collection")):
        collection = sample_collection()
    assert len(collection.articles) == 1
    assert collection.asset.company_name == "Apple"
    assert collection_rows(collection)[0]["Status"] == "Ready for analysis"
    assert b"Apple earnings rise" in collection_csv(collection)


def test_query_uses_alternatives_and_removes_corporate_suffix():
    query = NewsService.build_search_query("AAPL", "Apple Inc.", 25)
    assert '"AAPL" OR "Apple"' in query
    assert "Inc." not in query
    assert "when:25h" in query


def test_non_feed_html_is_a_failure_not_empty_success():
    with pytest.raises(ValueError, match="RSS or Atom"):
        NewsService.parse_rss_feed_content("<html><body>Access denied</body></html>",
                                          "Source", ProviderType.DIRECT_RSS)


def test_atom_updated_timestamp_is_supported():
    content = """<feed xmlns="http://www.w3.org/2005/Atom"><title>News</title>
    <entry><id>1</id><title>Apple earnings</title><updated>2026-10-19T12:00:00Z</updated>
    <link href="https://example.com/1"/></entry></feed>"""
    article = NewsService.parse_rss_feed_content(content, "Feed", ProviderType.DIRECT_RSS)[0]
    assert article.published_at_utc == datetime(2026, 10, 19, 12, tzinfo=timezone.utc)


def test_yahoo_null_url_objects_do_not_abort_all_items():
    article = NewsService.parse_yfinance_items([{"content": {
        "title": "Apple earnings", "canonicalUrl": None, "clickThroughUrl": None,
        "pubDate": "2026-10-19T12:00:00Z"}}], "AAPL")[0]
    assert article.title == "Apple earnings"


def test_sources_run_independently_and_report_failures():
    def fetch(url, name, provider):
        if "news.google.com" in url:
            response = httpx.Response(429, request=httpx.Request("GET", url))
            raise httpx.HTTPStatusError("rate limit", request=response.request, response=response)
        return NewsService.parse_rss_feed_content(rss(), name, provider)
    session = analysis_window("Last 24 hours", AssetType.EQUITY)
    with patch.object(NewsService, "_fetch_feed_httpx_sync", side_effect=fetch):
        articles, providers = asyncio.run(NewsService.fetch_news_for_session("AAPL", "Apple", session))
    assert len(articles) == 1  # Same story across feeds is deduplicated.
    assert providers["Yahoo Finance RSS"]["status"] == "SUCCESS"
    assert providers["Google News"]["error"] == "Rate limited (429)"
    assert providers["CNBC direct RSS"]["count"] == 1


def test_all_failed_feeds_still_return_collection_diagnostics():
    with patch.object(NewsService, "_fetch_feed_httpx_sync", side_effect=httpx.ConnectError("offline")):
        collection = asyncio.run(collect_headlines("AAPL", "Last 24 hours", 10))
    assert collection.articles == []
    assert all(p["status"] == "FAILED" for p in collection.provider_outcomes.values())


def test_collected_headlines_remain_visible_if_model_fails():
    collection = sample_collection()
    script = str(Path(__file__).parents[1] / "streamlit_app.py")
    app = AppTest.from_file(script).run(timeout=20)
    with patch("app.services.orchestrator.collect_headlines", return_value=collection):
        next(b for b in app.button if b.label == "Fetch live headlines").click().run(timeout=20)
    assert not app.exception
    assert len(app.session_state["collection"]["articles"]) == 1
    assert "result" not in app.session_state
    assert any(m.label == "Unique headlines fetched" and m.value == "1" for m in app.metric)
    with patch("app.services.orchestrator.analyze_collection", side_effect=RuntimeError("Model unavailable")):
        next(b for b in app.button if b.label == "Analyze collected headlines with FinBERT").click().run(timeout=20)
    assert not app.exception
    assert len(app.session_state["collection"]["articles"]) == 1
    assert any("remain available" in e.value for e in app.error)
