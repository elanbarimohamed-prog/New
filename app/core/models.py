from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field
from app.core.enums import (
    AssetType,
    DecisionVerdict,
    DirectionalPolarity,
    CatalystCategory,
    ProviderType,
    PipelineMode,
    CacheStatus,
    ExclusionReason,
)
from app.core.session_window import SessionWindow


@dataclass
class Asset:
    ticker: str
    company_name: str
    exchange: str
    sector: str = ""
    asset_type: AssetType = AssetType.EQUITY
    is_active: bool = True
    yf_symbol: str = ""


class IngestedArticle(BaseModel):
    article_id: str                      # Deterministic hash of canonical_url + title
    provider: ProviderType               # YFINANCE | FINNHUB | GOOGLE_RSS_AGGREGATOR | FIXTURE
    source_name: str                     # e.g., "Reuters", "CNBC", "PR Newswire"
    canonical_url: str                   # Cleaned URL without utm_* tracking
    title: str                           # Sanitized headline
    summary: str = ""                    # Summary or lead text (max 1000 chars)
    published_at_utc: datetime | None = None # None means missing or untrusted
    retrieved_at_utc: datetime           # Timestamp when ingested
    raw_provider_metadata: dict = Field(default_factory=dict)


class ScoredArticle(BaseModel):
    article_id: str
    provider: ProviderType
    source_name: str
    canonical_publisher: str             # Normalized publisher root entity
    canonical_url: str
    title: str
    published_at_utc: datetime | None = None # Timezone-aware UTC timestamp or None if undated
    lead_text: str
    finbert_positive: float
    finbert_negative: float
    finbert_neutral: float
    finbert_directional_score: float     # In [-1.0, 1.0]
    finbert_confidence: float            # In [0.0, 1.0]
    is_llm_catalyst: bool = False
    catalyst_category: CatalystCategory | None = None
    exclusion_reason: ExclusionReason | None = None
    final_weight: float = 0.0
    effective_weight: float = 0.0        # recency_decay * max(0.2, final_weight)
    age_hours: float = 0.0               # Hours elapsed from publication to session run
    cluster_id: str = ""                 # Group ID for syndicated articles


class ObservedFact(BaseModel):
    statement: str
    cited_article_id: str                # Must match a retained ScoredArticle ID
    catalyst_category: CatalystCategory
    impact: DirectionalPolarity


class PipelineRationale(BaseModel):
    summary: str
    observed_facts: list[ObservedFact] = []
    inferences: list[str] = []


class PipelineMetrics(BaseModel):
    total_sources: int = 0
    distinct_publishers: int = 0
    distinct_clusters: int = 0
    articles_ingested: int = 0
    articles_retained: int = 0
    articles_deduplicated: int = 0
    articles_scored_local: int = 0
    articles_sent_to_llm: int = 0
    excluded_out_of_window: int = 0
    excluded_missing_timestamp: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    stage_latencies: dict[str, float] = {}


class AnalysisResult(BaseModel):
    run_id: str
    ticker: str
    company_name: str
    asset_type: AssetType
    session: SessionWindow
    decision: DecisionVerdict
    decision_confidence: float           # In [0.0, 1.0]
    sentiment_score: float               # In [-1.0, 1.0]
    evidence_quality: float              # In [0.0, 1.0]
    disclaimer: str
    rationale: PipelineRationale
    metrics: PipelineMetrics
    articles: list[ScoredArticle]
    warnings: list[str] = []
    pipeline_mode: PipelineMode = PipelineMode.HYBRID_FINBERT_LLM
    cache_status: CacheStatus = CacheStatus.MISS
    data_status: str = "OK"              # "OK" | "NO_FRESH_NEWS"
    llm_summary: str | None = None
    model_versions: dict[str, str] = {}
    config_version: str = ""


