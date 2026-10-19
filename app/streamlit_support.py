"""Pure presentation helpers, also used by tests and CSV export."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import io
import re
from urllib.parse import urlsplit

from app.core.constants import DIRECTIONAL_THRESHOLD
from app.core.enums import AssetType, DecisionVerdict, PipelineMode, ProviderType, SessionType
from app.core.models import AnalysisResult, PipelineMetrics, ScoredArticle
from app.core.session_window import SessionWindow, compute_session_window
from app.services.aggregator_service import AggregatorService


def analysis_window(mode: str, asset_type: AssetType, now: datetime | None = None) -> SessionWindow:
    """Crypto uses a rolling window even if the US-session option was selected."""
    now = now or datetime.now(timezone.utc)
    session = compute_session_window(now)
    if mode == "US market session" and asset_type != AssetType.CRYPTO:
        return session
    start = session.run_timestamp_utc - timedelta(hours=24)
    return SessionWindow(
        market_timezone="UTC",
        session_type=SessionType.OVERNIGHT_PRE_OPEN,
        is_market_open=False,
        window_start_utc=start,
        window_start_et=start.astimezone(session.run_timestamp_et.tzinfo),
        window_end_utc=session.run_timestamp_utc,
        window_end_et=session.run_timestamp_et,
        window_duration_hours=24.0,
        run_timestamp_utc=session.run_timestamp_utc,
        run_timestamp_et=session.run_timestamp_et,
    )


def article_tone(article: ScoredArticle) -> str:
    if article.finbert_directional_score >= DIRECTIONAL_THRESHOLD:
        return "Positive"
    if article.finbert_directional_score <= -DIRECTIONAL_THRESHOLD:
        return "Negative"
    return "Neutral"


def evidence_counts(result: AnalysisResult) -> dict:
    retained = [a for a in result.articles if a.exclusion_reason is None]
    directional = [a for a in retained if article_tone(a) != "Neutral"]
    publishers = {a.canonical_publisher for a in directional
                  if a.canonical_publisher and a.canonical_publisher != "Unknown Aggregator"}
    counts = Counter(article_tone(a) for a in retained)
    return {"retained": len(retained), "directional_publishers": len(publishers),
            "positive": counts["Positive"], "negative": counts["Negative"],
            "neutral": counts["Neutral"]}


def display_verdict(result: AnalysisResult) -> str:
    if result.data_status == "NO_FRESH_NEWS":
        return "No fresh news"
    if result.data_status == "NO_RELEVANT_NEWS":
        return "No relevant news"
    counts = evidence_counts(result)
    if counts["positive"] + counts["negative"] < 2:
        return "Insufficient evidence"
    return {DecisionVerdict.BUY: "Positive sentiment", DecisionVerdict.SELL: "Negative sentiment",
            DecisionVerdict.HOLD: "Mixed / inconclusive"}[result.decision]


def safe_url(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        return url if parsed.scheme in {"http", "https"} and parsed.netloc else None
    except ValueError:
        return None


def csv_cell(value):
    """Prevent external headlines being interpreted as spreadsheet formulas."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def article_rows(result: AnalysisResult) -> list[dict]:
    return [{"Headline": a.title, "Publisher": a.canonical_publisher,
             "Published (UTC)": a.published_at_utc.isoformat() if a.published_at_utc else "Undated",
             "Tone": article_tone(a) if a.exclusion_reason is None else "Excluded",
             "Score": a.finbert_directional_score,
             "Catalyst": a.catalyst_category.value if a.catalyst_category else "",
             "Exclusion reason": a.exclusion_reason.value if a.exclusion_reason else "",
             "Link": safe_url(a.canonical_url) or ""} for a in result.articles]


def csv_export(result: AnalysisResult) -> bytes:
    rows = article_rows(result)
    output = io.StringIO(newline="")
    fields = ["Headline", "Publisher", "Published (UTC)", "Tone", "Score", "Catalyst", "Exclusion reason", "Link"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: csv_cell(v) for k, v in row.items()} for row in rows)
    return output.getvalue().encode("utf-8-sig")


def export_name(ticker: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", ticker)


def demo_result() -> AnalysisResult:
    """Fictional, pre-scored examples. Never changes the real model's state."""
    from uuid import uuid4
    session = analysis_window("Last 24 hours", AssetType.EQUITY)
    samples = [
        ("Example company reports stronger quarterly earnings", "Example News A", .86, .04, .10),
        ("Example company wins a major supply contract", "Example News B", .78, .06, .16),
        ("Example company faces a product recall", "Example News C", .05, .82, .13),
        ("Example company announces shareholder meeting date", "Example News A", .07, .06, .87),
    ]
    articles = []
    for i, (title, publisher, pos, neg, neu) in enumerate(samples):
        score = pos - neg if abs(pos - neg) >= DIRECTIONAL_THRESHOLD else 0.0
        articles.append(ScoredArticle(
            article_id=f"demo-{i}", provider=ProviderType.FIXTURE,
            source_name=publisher, canonical_publisher=publisher, canonical_url="",
            title=title, lead_text=title,
            published_at_utc=session.run_timestamp_utc - timedelta(hours=i + 1),
            finbert_positive=pos, finbert_negative=neg, finbert_neutral=neu,
            finbert_directional_score=score, finbert_confidence=max(pos, neg),
            final_weight=max(pos, neg) * (1 - neu) if score else 0,
            cluster_id=f"demo-{i}",
        ))
    verdict, confidence, score, quality, rationale, warnings = AggregatorService.aggregate_and_decide(
        "DEMO", "Fictional example company", session, articles, [],
        pipeline_mode=PipelineMode.DETERMINISTIC_FINBERT_ONLY,
    )
    return AnalysisResult(
        run_id=str(uuid4()), ticker="DEMO", company_name="Fictional example company",
        asset_type=AssetType.EQUITY, session=session, decision=verdict,
        decision_confidence=confidence, sentiment_score=score, evidence_quality=quality,
        disclaimer="Research tool. Sentiment is not a forecast or investment recommendation.",
        rationale=rationale, articles=articles, warnings=warnings,
        pipeline_mode=PipelineMode.DETERMINISTIC_FINBERT_ONLY,
        metrics=PipelineMetrics(articles_ingested=4, articles_retained=4, distinct_publishers=3),
        data_status="DEMO", model_versions={"finbert": "PRE-SCORED FICTIONAL EXAMPLES", "llm": "NONE"},
    )
