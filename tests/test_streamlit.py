import asyncio
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.core.enums import AssetType, ExclusionReason, ProviderType
from app.core.models import Asset, IngestedArticle
from app.services.orchestrator import analyze
from app.streamlit_support import (
    analysis_window, article_rows, csv_export, demo_result,
    display_verdict, evidence_counts, safe_url,
)


def test_demo_is_explicit_and_serializable():
    result = demo_result()
    assert result.data_status == "DEMO"
    assert "PRE-SCORED" in result.model_versions["finbert"]
    assert result.model_dump_json()
    assert evidence_counts(result)["retained"] == 4


def test_excluded_articles_do_not_inflate_retained_or_publishers():
    result = demo_result()
    result.articles[0].exclusion_reason = ExclusionReason.DUPLICATE_SYNDICATION
    result.articles[1].exclusion_reason = ExclusionReason.TIMESTAMP_OUT_OF_WINDOW
    counts = evidence_counts(result)
    assert counts["retained"] == 2
    assert counts["directional_publishers"] == 1
    assert counts["positive"] == 0
    assert counts["neutral"] == 1


def test_no_news_and_insufficient_evidence_have_separate_labels():
    result = demo_result()
    result.data_status = "NO_FRESH_NEWS"
    assert display_verdict(result) == "No fresh news"
    result.data_status = "NO_RELEVANT_NEWS"
    assert display_verdict(result) == "No relevant news"
    result.data_status = "OK"
    result.articles = result.articles[-1:]
    assert display_verdict(result) == "Insufficient evidence"


def test_csv_handles_formula_injection_and_has_full_evidence():
    result = demo_result()
    result.articles[0].title = '=HYPERLINK("https://example.com")'
    text = csv_export(result).decode("utf-8-sig")
    assert "'=HYPERLINK" in text
    assert len(article_rows(result)) == 4


def test_untrusted_urls_are_rejected():
    assert safe_url("javascript:alert(1)") is None
    assert safe_url("file:///etc/passwd") is None
    assert safe_url("https://example.com/news") == "https://example.com/news"


def test_crypto_always_has_24_hour_window():
    window = analysis_window("US market session", AssetType.CRYPTO)
    assert window.window_duration_hours == 24
    assert window.window_end_utc - window.window_start_utc == timedelta(hours=24)


def run_fixture(news, providers):
    reference = demo_result()
    asset = Asset("AAPL", "Apple Inc.", "NASDAQ", asset_type=AssetType.EQUITY)
    class FakeScorer:
        async def score_articles_async(self, articles, clusters):
            return [reference.articles[i].model_copy(update={
                "article_id": a.article_id, "title": a.title,
                "published_at_utc": a.published_at_utc,
            }) for i, a in enumerate(articles)]
    with patch("app.services.orchestrator.TickerService.validate_and_resolve", return_value=asset), \
         patch("app.services.orchestrator.NewsService.fetch_news_for_session",
               new=AsyncMock(return_value=(news, providers))):
        return asyncio.run(analyze("AAPL", "Last 24 hours", 40, FakeScorer()))


def test_partial_feed_outage_still_returns_result_without_database():
    now = demo_result().session.run_timestamp_utc
    news = [IngestedArticle(
        article_id="one", provider=ProviderType.FIXTURE, source_name="Reuters",
        canonical_url="https://example.com/1", title="Apple earnings rise",
        published_at_utc=now - timedelta(hours=1), retrieved_at_utc=now,
    )]
    result = run_fixture(news, {"yfinance": {"status": "SUCCESS"},
                                "rss_google_news_general": {"status": "FAILED"}})
    assert result.metrics.articles_retained == 1
    assert result.run_id
    assert any("unavailable" in w for w in result.warnings)


def test_no_news_skips_model_and_returns_explicit_status():
    result = run_fixture([], {"yfinance": {"status": "SUCCESS"}})
    assert result.data_status == "NO_FRESH_NEWS"
    assert result.decision_confidence == 0


def test_stale_headline_never_reaches_scorer():
    now = demo_result().session.run_timestamp_utc
    news = [IngestedArticle(
        article_id="old", provider=ProviderType.FIXTURE, source_name="Reuters",
        canonical_url="https://example.com/old", title="Apple earnings rise",
        published_at_utc=now - timedelta(days=3), retrieved_at_utc=now,
    )]
    result = run_fixture(news, {"yfinance": {"status": "SUCCESS"}})
    assert result.metrics.articles_scored_local == 0
    assert result.metrics.excluded_out_of_window == 1


def test_streamlit_initial_screen_and_demo():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py")).run(timeout=20)
    assert not app.exception
    next(b for b in app.button if b.label == "Explore a demo").click().run(timeout=20)
    assert not app.exception
    assert any("DEMO" in x.value for x in app.warning)
    assert len(app.metric) == 4
    assert len(app.tabs) == 4


def test_streamlit_filter_and_history():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py")).run(timeout=20)
    next(b for b in app.button if b.label == "Explore a demo").click().run(timeout=20)
    next(s for s in app.selectbox if s.label == "Filter tone").select("Negative").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    next(b for b in app.button if b.label == "Clear session history").click().run()
    assert not app.exception
    assert not app.session_state["history"]


def test_optional_audit_rejects_invalid_json():
    import pytest
    from app.services.llm_service import LLMService
    with pytest.raises(ValueError):
        LLMService._parse_llm_response("not JSON", 3)


def test_explicit_equity_alias_does_not_poison_commodity_lookup():
    from app.services.ticker_service import TickerService
    class OfflineTicker:
        fast_info = None
        info = {}
    TickerService.clear_cache()
    with patch("app.services.ticker_service.yf.Ticker", return_value=OfflineTicker()):
        equity = TickerService.validate_and_resolve("$GOLD")
        commodity = TickerService.validate_and_resolve("GOLD")
    assert equity.asset_type == AssetType.EQUITY
    assert commodity.asset_type == AssetType.COMMODITY
    assert commodity.yf_symbol == "GC=F"
    TickerService.clear_cache()


def test_scorer_preserves_probabilities_and_does_not_load_for_empty_input():
    from app.services.finbert_service import FinBERTService
    scorer = FinBERTService()
    assert asyncio.run(scorer.score_articles_async([], {})) == []
    assert scorer._pipeline is None
    now = demo_result().session.run_timestamp_utc
    article = IngestedArticle(
        article_id="scored", provider=ProviderType.FIXTURE, source_name="Reuters",
        canonical_url="https://example.com", title="Apple profits rise",
        published_at_utc=now, retrieved_at_utc=now,
    )
    scorer._pipeline = lambda texts, **kwargs: [[
        {"label": "positive", "score": .8},
        {"label": "negative", "score": .05},
        {"label": "neutral", "score": .15},
    ]]
    result = asyncio.run(scorer.score_articles_async([article], {"scored": "c1"}))[0]
    assert abs(result.finbert_directional_score - .75) < 1e-9
    assert result.cluster_id == "c1"
