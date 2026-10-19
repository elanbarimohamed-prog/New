import hashlib
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Environment
    APP_ENV: str = "development"
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite:///./data/sentiment.db"

    # Model Identifiers
    FINBERT_MODEL_NAME: str = "ProsusAI/finbert"
    LLM_MODEL_ID: str = ""
    LLM_FALLBACK_MODEL_ID: str = ""
    ENABLE_LLM: bool = True  # Per-run opt-in is controlled by the Streamlit interface.

    # External Provider Credentials
    OPENROUTER_API_KEY: str = ""

    # Security & Networking
    TRUST_PROXY_HEADERS: bool = False
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Budget & Token Controls
    MAX_RETAINED_ARTICLES: int = 350
    MAX_LLM_CANDIDATES: int = 100
    MAX_EXCERPT_TOKENS: int = 150
    MAX_PROMPT_TOKENS: int = 8000
    MAX_COMPLETION_TOKENS: int = 1000
    MAX_COST_PER_RUN_USD: float = 0.05

    # Rate Limiting
    RATE_LIMIT_ANALYSIS_PER_MINUTE: int = 15
    RATE_LIMIT_JOBS_PER_MINUTE: int = 60

    # Pipeline Execution Timeout (in seconds)
    PIPELINE_TIMEOUT_SECONDS: int = 90

    # Cache Buckets (in minutes)
    CACHE_INTRADAY_BUCKET_MINUTES: int = 15
    CACHE_OFF_MARKET_BUCKET_MINUTES: int = 60

    # Model Token Pricing Table (USD per 1,000,000 tokens)
    MODEL_PRICING: dict[str, dict[str, float]] = {
        "openai/gpt-4o-mini": {"prompt_price_per_m": 0.15, "completion_price_per_m": 0.60},
        "deepseek/deepseek-chat": {"prompt_price_per_m": 0.14, "completion_price_per_m": 0.28},
        "nvidia/nemotron-3-ultra-550b-a55b:free": {"prompt_price_per_m": 0.0, "completion_price_per_m": 0.0},
    }

    def estimate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate token cost in USD based on model pricing table."""
        pricing = self.MODEL_PRICING.get(
            model_id,
            {"prompt_price_per_m": 0.20, "completion_price_per_m": 0.80}  # Conservative default
        )
        prompt_cost = (prompt_tokens * pricing["prompt_price_per_m"]) / 1_000_000.0
        comp_cost = (completion_tokens * pricing["completion_price_per_m"]) / 1_000_000.0
        return round(prompt_cost + comp_cost, 6)

    def get_config_hash(self) -> str:
        """Deterministic fingerprint of model IDs and pipeline limits for cache keying."""
        raw = f"{self.FINBERT_MODEL_NAME}|{self.LLM_MODEL_ID}|{self.MAX_RETAINED_ARTICLES}|{self.MAX_LLM_CANDIDATES}|{self.ENABLE_LLM}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


settings = Settings()

