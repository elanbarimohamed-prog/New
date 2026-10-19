"""Streamlit analysis pipeline. Returns results directly, without a web API or database."""
from __future__ import annotations
import asyncio
import time
import uuid
from app.config import settings
from app.core.enums import PipelineMode
from app.core.exceptions import NewsUnavailableException
from app.core.models import AnalysisResult, PipelineMetrics, ScoredArticle
from app.services.aggregator_service import AggregatorService
from app.services.deduplication_service import DeduplicationService
from app.services.news_service import NewsService
from app.services.ticker_service import TickerService
from app.streamlit_support import analysis_window, evidence_counts


async def analyze(ticker, window_mode, article_limit, scorer, enable_llm=False, progress=None):
    started = time.perf_counter()
    latencies = {}
    def report(fraction, text):
        if progress:
            progress(fraction, text)

    report(.05, "Resolving the asset...")
    asset = await asyncio.to_thread(TickerService.validate_and_resolve, ticker)
    session = analysis_window(window_mode, asset.asset_type)
    report(.15, "Collecting recent headlines from news feeds...")
    step = time.perf_counter()
    raw, providers = await NewsService.fetch_news_for_session(
        asset.ticker, asset.company_name, session, asset.asset_type, asset.yf_symbol)
    latencies["Collect news"] = (time.perf_counter() - step) * 1000
    if not providers or all(p.get("status") == "FAILED" for p in providers.values()):
        raise NewsUnavailableException()
    failed = [name for name, status in providers.items() if status.get("status") == "FAILED"]
    warnings = []
    if failed:
        warnings.append("Some feeds were unavailable: " + ", ".join(failed)
                        + ". This result uses the feeds that responded.")

    report(.35, "Checking timestamps, relevance, and repeated headlines...")
    step = time.perf_counter()
    fresh, stale_sample, stale_count, undated_count = DeduplicationService.filter_by_session_window(raw, session)
    retained, excluded, clusters = DeduplicationService.process_and_deduplicate(
        fresh, asset.ticker, asset.company_name, max_articles=article_limit)
    latencies["Filter news"] = (time.perf_counter() - step) * 1000
    scored = []
    if retained:
        report(.50, f"Loading FinBERT and scoring {len(retained)} headlines. First run downloads the model...")
        step = time.perf_counter()
        scored = await scorer.score_articles_async(retained, clusters)
        latencies["FinBERT"] = (time.perf_counter() - step) * 1000

    mode = PipelineMode.DETERMINISTIC_FINBERT_ONLY
    facts, summary, telemetry = [], "", {}
    if enable_llm and settings.OPENROUTER_API_KEY:
        from app.services.llm_service import LLMService
        report(.80, "Auditing potential market catalysts...")
        step = time.perf_counter()
        audit = LLMService()
        try:
            scored, facts, summary, telemetry = await audit.verify_and_synthesize(
                asset.ticker, asset.company_name, scored)
        finally:
            if audit.client:
                await audit.client.close()
        latencies["Catalyst audit"] = (time.perf_counter() - step) * 1000
        if telemetry.get("status") == "SUCCESS":
            mode = PipelineMode.HYBRID_FINBERT_LLM
        else:
            warnings.append("Optional AI audit unavailable; FinBERT analysis is still available.")

    report(.92, "Combining scores and preparing evidence...")
    decision, confidence, score, quality, rationale, extra = AggregatorService.aggregate_and_decide(
        asset.ticker, asset.company_name, session, scored, facts, summary, mode)
    evidence = list(scored)
    for article, reason, _note in stale_sample + excluded:
        evidence.append(ScoredArticle(
            article_id=article.article_id, provider=article.provider,
            source_name=article.source_name,
            canonical_publisher=DeduplicationService.normalize_publisher(article.source_name),
            canonical_url=article.canonical_url, title=article.title,
            published_at_utc=article.published_at_utc,
            lead_text=f"{article.title}. {article.summary}"[:1000],
            finbert_positive=0, finbert_negative=0, finbert_neutral=1,
            finbert_directional_score=0, finbert_confidence=0, exclusion_reason=reason))
    status = "NO_FRESH_NEWS" if not fresh else ("NO_RELEVANT_NEWS" if not retained else "OK")
    result = AnalysisResult(
        run_id=str(uuid.uuid4()), ticker=asset.ticker, company_name=asset.company_name,
        asset_type=asset.asset_type, session=session, decision=decision,
        decision_confidence=confidence, sentiment_score=score, evidence_quality=quality,
        disclaimer="News sentiment research. Scores are not forecasts or investment recommendations.",
        rationale=rationale, articles=evidence, warnings=warnings + extra,
        pipeline_mode=mode, data_status=status, llm_summary=summary or None,
        metrics=PipelineMetrics(
            total_sources=len(providers), distinct_clusters=len({a.cluster_id for a in scored}),
            articles_ingested=len(raw), articles_retained=len(scored),
            articles_deduplicated=len(excluded), articles_scored_local=len(scored),
            articles_sent_to_llm=telemetry.get("articles_sent", 0),
            excluded_out_of_window=stale_count, excluded_missing_timestamp=undated_count,
            total_latency_ms=(time.perf_counter() - started) * 1000, stage_latencies=latencies,
            prompt_tokens=telemetry.get("prompt_tokens", 0),
            completion_tokens=telemetry.get("completion_tokens", 0),
            estimated_cost_usd=telemetry.get("estimated_cost_usd", 0)),
        model_versions={"finbert": settings.FINBERT_MODEL_NAME, "llm": telemetry.get("model", "NONE")},
        config_version=settings.get_config_hash())
    result.metrics.distinct_publishers = evidence_counts(result)["directional_publishers"]
    report(1.0, "Analysis complete.")
    return result

