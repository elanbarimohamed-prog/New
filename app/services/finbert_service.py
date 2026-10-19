"""Thread-safe, lazy CPU FinBERT scorer shared across Streamlit reruns."""
from __future__ import annotations
import asyncio
import threading
from app.config import settings
from app.core.constants import DIRECTIONAL_THRESHOLD
from app.core.models import ScoredArticle
from app.services.deduplication_service import DeduplicationService


class FinBERTService:
    def __init__(self):
        self._pipeline = None
        self._inference_lock = threading.Lock()

    def _score_sync(self, articles, cluster_map):
        # Tokenizers and inference are protected when the cached resource is shared.
        with self._inference_lock:
            if self._pipeline is None:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
                torch.set_num_threads(2)
                tokenizer = AutoTokenizer.from_pretrained(settings.FINBERT_MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(settings.FINBERT_MODEL_NAME)
                model.eval()
                self._pipeline = pipeline("text-classification", model=model,
                                          tokenizer=tokenizer, device=-1)
            texts = [f"{a.title}. {a.summary}".strip() for a in articles]
            outputs = self._pipeline(texts, top_k=None, truncation=True, max_length=512, batch_size=4)
            scored = []
            for art, predictions in zip(articles, outputs):
                probs = {p["label"].lower(): float(p["score"]) for p in predictions}
                pos, neg, neu = (probs.get(k, 0.0) for k in ("positive", "negative", "neutral"))
                difference = pos - neg
                score = difference if abs(difference) >= DIRECTIONAL_THRESHOLD else 0.0
                scored.append(ScoredArticle(
                    article_id=art.article_id, provider=art.provider, source_name=art.source_name,
                    canonical_publisher=DeduplicationService.normalize_publisher(art.source_name),
                    canonical_url=art.canonical_url, title=art.title,
                    published_at_utc=art.published_at_utc,
                    lead_text=f"{art.title}. {art.summary}".strip()[:1000],
                    finbert_positive=pos, finbert_negative=neg, finbert_neutral=neu,
                    finbert_directional_score=score, finbert_confidence=max(pos, neg),
                    final_weight=max(pos, neg) * (1 - neu) if score else 0,
                    cluster_id=cluster_map.get(art.article_id, art.article_id)))
            return scored

    async def score_articles_async(self, articles, cluster_map):
        if not articles:
            return []
        return await asyncio.to_thread(self._score_sync, articles, cluster_map)

