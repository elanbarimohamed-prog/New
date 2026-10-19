import logging
import re
import sys
from typing import Any

# Pattern to redact API keys and secrets in logs
SECRET_PATTERN = re.compile(r"(sk-[a-zA-Z0-9_\-]{15,}|Bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE)


class RedactingFormatter(logging.Formatter):
    """Custom formatter that automatically redacts API keys and tokens from log messages."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return SECRET_PATTERN.sub("[REDACTED_SECRET]", msg)


def setup_logger(name: str = "market_sentiment") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = RedactingFormatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()

