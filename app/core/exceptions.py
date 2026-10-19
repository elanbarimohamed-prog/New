class SentimentAppException(Exception):
    """Base exception for the application."""
    def __init__(self, message: str, status_code: int = 500, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class InvalidTickerException(SentimentAppException):
    """Raised when a ticker fails syntax validation."""
    def __init__(self, ticker: str, reason: str = "Invalid ticker format"):
        super().__init__(
            message=f"Invalid ticker '{ticker}': {reason}",
            status_code=422,
            details={"ticker": ticker, "reason": reason}
        )


class UnsupportedAssetException(SentimentAppException):
    """Raised when an asset is outside the supported v1 scope (e.g., Forex, Crypto, Commodity)."""
    def __init__(self, raw_input: str, detected_type: str):
        super().__init__(
            message=(
                f"Asset '{raw_input}' ({detected_type}) is not supported in Version 1.0. "
                "Version 1.0 strictly evaluates US-listed Equities and ETFs. "
                "Forex, crypto, and commodities have distinct directional semantics and are scheduled for v2.0."
            ),
            status_code=422,
            details={"raw_input": raw_input, "detected_type": detected_type}
        )


class TickerNotFoundException(SentimentAppException):
    """Raised when a ticker cannot be resolved with the market data provider."""
    def __init__(self, ticker: str, reason: str = ""):
        msg = f"Ticker '{ticker}' could not be resolved or is not an actively listed US equity/ETF."
        if reason:
            msg = f"{msg} ({reason})"
        super().__init__(
            message=msg,
            status_code=404,
            details={"ticker": ticker, "reason": reason} if reason else {"ticker": ticker}
        )


class ProviderException(SentimentAppException):
    """Raised when an external data provider encounters an unrecoverable error."""
    def __init__(self, provider: str, error: str):
        super().__init__(
            message=f"Provider '{provider}' error: {error}",
            status_code=502,
            details={"provider": provider, "error": error}
        )


class BudgetExceededException(SentimentAppException):
    """Raised when prompt or cost limits are exceeded."""
    def __init__(self, reason: str):
        super().__init__(
            message=f"Budget limit reached: {reason}",
            status_code=429,
            details={"reason": reason}
        )


class RateLimitExceededException(SentimentAppException):
    """Raised when an IP rate limit is exceeded."""
    def __init__(self, retry_after_seconds: int = 60):
        super().__init__(
            message="Rate limit exceeded. Please wait before submitting another request.",
            status_code=429,
            details={"retry_after_seconds": retry_after_seconds}
        )


class JobNotFoundException(SentimentAppException):
    """Raised when a requested background job is not found."""
    def __init__(self, job_id: str):
        super().__init__(
            message=f"Job '{job_id}' not found.",
            status_code=404,
            details={"job_id": job_id}
        )


class AnalysisRunNotFoundException(SentimentAppException):
    """Raised when an analysis run record is not found."""
    def __init__(self, run_id: str):
        super().__init__(
            message=f"Analysis run '{run_id}' not found.",
            status_code=404,
            details={"run_id": run_id}
        )


class FreshnessViolationException(SentimentAppException):
    """Raised when an out-of-window article violates the core freshness invariant."""
    def __init__(
        self,
        article_id: str = "",
        timestamp_str: str = "",
        window_start: str = "",
        window_end: str = "",
        message: str | None = None,
    ):
        msg = message or (
            f"Core Invariant Violation: Article '{article_id}' timestamp ({timestamp_str}) "
            f"lies outside session window [{window_start}, {window_end}]."
        )
        super().__init__(
            message=msg,
            status_code=500,
            details={
                "article_id": article_id,
                "timestamp": timestamp_str,
                "window_start": window_start,
                "window_end": window_end,
            }
        )


class NewsUnavailableException(SentimentAppException):
    """Raised when all external news providers fail (total outage)."""
    def __init__(self, message: str = "External news providers are currently unavailable."):
        super().__init__(
            message=message,
            status_code=503,
            details={"reason": "ALL_PROVIDERS_FAILED"}
        )


