"""SentimentPulse application package."""
from pathlib import Path
from app_bootstrap import source_fingerprint

# Mark fresh imports so normal reruns do not discard current module objects.
_sentiment_source_fingerprint = source_fingerprint(Path(__file__).resolve().parent)

