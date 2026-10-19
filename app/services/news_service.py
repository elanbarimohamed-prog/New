import asyncio
import hashlib
import html
import math
import re
import time
import urllib.parse
from datetime import datetime, timezone
import feedparser
import httpx
import yfinance as yf

from app.config import settings
from app.core.models import IngestedArticle
from app.core.enums import ProviderType, AssetType
from app.core.session_window import SessionWindow
from app.utils.logging import logger

EQUITY_PUBLISHER_DOMAINS = {
    "Reuters": "reuters.com",
    "Bloomberg": "bloomberg.com",
    "CNBC": "cnbc.com",
    "Wall Street Journal": "wsj.com",
    "MarketWatch": "marketwatch.com",
    "Seeking Alpha": "seekingalpha.com",
    "Motley Fool": "fool.com",
}

FOREX_COMMODITY_PUBLISHER_DOMAINS = {
    "Reuters": "reuters.com",
    "Bloomberg": "bloomberg.com",
    "CNBC": "cnbc.com",
    "Wall Street Journal": "wsj.com",
    "MarketWatch": "marketwatch.com",
}


class NewsService:
    @staticmethod
    def _create_article_id(canonical_url: str, title: str) -> str:
        raw = f"{canonical_url.strip()}|{title.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _clean_html(text: str) -> str:
        """Strips HTML tags, collapses whitespace, and unescapes HTML entities."""
        if not text:
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = html.unescape(cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _clean_entry_title(cls, raw_title: str, publisher: str) -> str:
        """Clean HTML and strip trailing ' - <Publisher>' when matching source."""
        title = cls._clean_html(raw_title)
        if publisher:
            pattern = rf"\s*-\s*{re.escape(publisher.strip())}\s*$"
            title = re.sub(pattern, "", title, flags=re.IGNORECASE).strip()
        return title

    @classmethod
    def _clean_entry_summary(cls, raw_summary: str, cleaned_title: str, publisher: str) -> str:
        """
        Drop summary when, after HTML stripping, case, and whitespace normalisation,
        it equals the title or the title plus publisher.
        """
        summary = cls._clean_html(raw_summary)
        norm_sum = re.sub(r"\s+", " ", summary).strip().lower()
        norm_title = re.sub(r"\s+", " ", cleaned_title).strip().lower()
        norm_title_pub = re.sub(r"\s+", " ", f"{cleaned_title} {publisher}").strip().lower()

        if norm_sum in (norm_title, norm_title_pub) or not norm_sum:
            return ""
        return summary[:1000]

    @classmethod
    def parse_yfinance_items(cls, news_items: list[dict], yf_ticker: str) -> list[IngestedArticle]:
        """
        Parses raw yfinance news items supporting both the current schema:
        {"id": ..., "content": {"title", "summary", "pubDate" (ISO-8601, Z), "provider": {"displayName"}, ...}}
        and legacy keys (providerPublishTime, top-level title, publisher, link).
        Never defaults timestamps to now().
        """
        articles: list[IngestedArticle] = []
        now = datetime.now(timezone.utc)

        for item in news_items:
            if not isinstance(item, dict):
                continue

            content = item.get("content") if isinstance(item.get("content"), dict) else {}

            # Title
            title = content.get("title") or item.get("title", "")
            title = cls._clean_html(title)
            if not title:
                continue

            # Summary
            summary = content.get("summary") or content.get("description") or item.get("summary", "")
            summary = cls._clean_html(summary)

            # Link
            canonical_url = (
                content.get("canonicalUrl", {}).get("url")
                or content.get("clickThroughUrl", {}).get("url")
                or item.get("link", f"https://finance.yahoo.com/quote/{yf_ticker}")
            )

            # Publisher: Real provider displayName as publisher
            provider_dict = item.get("provider") if isinstance(item.get("provider"), dict) else {}
            content_provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
            publisher = (
                provider_dict.get("displayName")
                or content_provider.get("displayName")
                or item.get("publisher", "Yahoo Finance")
            ).strip()

            # Timestamp: Never default to now()
            pub_time: datetime | None = None
            pub_date_str = content.get("pubDate") or item.get("pubDate")
            if pub_date_str and isinstance(pub_date_str, str):
                try:
                    pub_time = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    if pub_time.tzinfo is None:
                        pub_time = pub_time.replace(tzinfo=timezone.utc)
                except Exception:
                    pub_time = None

            if pub_time is None:
                pub_ts = item.get("providerPublishTime") or content.get("providerPublishTime")
                if pub_ts is not None:
                    try:
                        pub_time = datetime.fromtimestamp(float(pub_ts), tz=timezone.utc)
                    except Exception:
                        pub_time = None

            art_id = cls._create_article_id(canonical_url, title)
            articles.append(
                IngestedArticle(
                    article_id=art_id,
                    provider=ProviderType.YFINANCE,
                    source_name=f"Yahoo - {publisher}" if publisher else "Yahoo Finance",
                    canonical_url=canonical_url,
                    title=title,
                    summary=summary[:1000] if summary else "",
                    published_at_utc=pub_time,
                    retrieved_at_utc=now,
                    raw_provider_metadata={"yf_publisher": publisher},
                )
            )

        return articles

    @classmethod
    def parse_rss_feed_content(
        cls,
        feed_content: str | bytes,
        default_source_name: str,
        provider: ProviderType,
    ) -> list[IngestedArticle]:
        """
        Parses an RSS feed from a string/bytes payload.
        Published time comes from published_parsed, otherwise None.
        Extracts source from entry.source.title; if absent, falls back to default_source_name
        and sets raw_provider_metadata['publisher_inferred'] = True.
        """
        feed = feedparser.parse(feed_content)
        articles: list[IngestedArticle] = []
        now = datetime.now(timezone.utc)

        for entry in getattr(feed, "entries", []):
            raw_title = getattr(entry, "title", "")
            raw_summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
            link = getattr(entry, "link", "")

            # Publisher from entry.source.title
            publisher = ""
            inferred = False
            source_obj = getattr(entry, "source", None)
            if source_obj:
                if isinstance(source_obj, dict):
                    publisher = (source_obj.get("title") or source_obj.get("value") or "").strip()
                else:
                    publisher = (getattr(source_obj, "title", "") or getattr(source_obj, "value", "") or "").strip()

            if not publisher:
                publisher = default_source_name
                inferred = True

            title = cls._clean_entry_title(raw_title, publisher)
            if not title:
                continue

            summary = cls._clean_entry_summary(raw_summary, title, publisher)

            # Publication time
            pub_time: datetime | None = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pub_time = None

            raw_meta: dict = {"rss_source": default_source_name, "publisher": publisher}
            if inferred:
                raw_meta["publisher_inferred"] = True

            art_id = cls._create_article_id(link, title)
            articles.append(
                IngestedArticle(
                    article_id=art_id,
                    provider=provider,
                    source_name=publisher,
                    canonical_url=link,
                    title=title,
                    summary=summary,
                    published_at_utc=pub_time,
                    retrieved_at_utc=now,
                    raw_provider_metadata=raw_meta,
                )
            )

        return articles

    @classmethod
    def _fetch_yfinance_news_sync(cls, yf_ticker: str) -> list[IngestedArticle]:
        """Fetches recent news articles from yfinance API synchronously."""
        ticker_obj = yf.Ticker(yf_ticker)
        news_items = getattr(ticker_obj, "news", [])
        if not isinstance(news_items, list):
            return []
        return cls.parse_yfinance_items(news_items, yf_ticker)

    @classmethod
    def _fetch_feed_httpx_sync(
        cls, url: str, default_source_name: str, provider: ProviderType
    ) -> list[IngestedArticle]:
        """
        Fetches an RSS feed via httpx with per-provider timeouts (connect 4.0s, read/total 8.0s),
        then parses the XML content via feedparser.
        A well-formed feed with 0 entries is SUCCESS.
        """
        timeout = httpx.Timeout(8.0, connect=4.0)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content
            return cls.parse_rss_feed_content(content, default_source_name, provider)

    @classmethod
    async def fetch_news_for_session(
        cls,
        ticker: str,
        company_name: str,
        session: SessionWindow,
        asset_type: AssetType = AssetType.EQUITY,
        yf_symbol: str = "",
    ) -> tuple[list[IngestedArticle], dict[str, dict]]:
        """
        Massive multi-source news ingestion bounded to current session lookback:
        1. Yahoo Finance API news
        2. Forex dedicated feeds if Forex
        3. Google News RSS bounded by when:{N}h and financial site: sweeps
        """
        yf_ticker = yf_symbol or ticker
        provider_outcomes: dict[str, dict] = {}
        all_articles: list[IngestedArticle] = []
        seen_urls = set()
        seen_titles = set()

        def add_article(art: IngestedArticle):
            url_norm = art.canonical_url.split("?")[0].rstrip("/").lower()
            title_norm = art.title.strip().lower()
            if url_norm and url_norm in seen_urls:
                return
            if title_norm and title_norm in seen_titles:
                return
            seen_urls.add(url_norm)
            seen_titles.add(title_norm)
            all_articles.append(art)

        # 1. Fetch Yahoo Finance API News
        t0 = time.time()
        try:
            yf_items = await asyncio.to_thread(cls._fetch_yfinance_news_sync, yf_ticker)
            for item in yf_items:
                add_article(item)
            provider_outcomes["yfinance"] = {
                "status": "SUCCESS",
                "count": len(yf_items),
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        except Exception as e:
            logger.warning(f"Error fetching yfinance news for {yf_ticker}: {e}")
            provider_outcomes["yfinance"] = {
                "status": "FAILED",
                "error": str(e),
                "count": 0,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }

        # 2. Dedicated Forex feeds
        if asset_type == AssetType.FOREX or "/" in ticker or "=X" in yf_ticker:
            t_fx = time.time()
            fx_feed_configs = [
                (
                    "Investing.com Forex",
                    "https://www.investing.com/rss/news_1.rss",
                    ProviderType.INVESTING_FOREX_RSS,
                ),
                (
                    "FXStreet",
                    "https://www.fxstreet.com/rss/news",
                    ProviderType.FXSTREET_RSS,
                ),
                (
                    "Yahoo Finance Forex RSS",
                    f"https://finance.yahoo.com/rss/headline?s={yf_ticker}",
                    ProviderType.YFINANCE,
                ),
            ]
            for feed_name, feed_url, prov_type in fx_feed_configs:
                key = f"forex_{feed_name.lower().replace(' ', '_').replace('.', '_')}"
                t_f = time.time()
                try:
                    items = await asyncio.to_thread(cls._fetch_feed_httpx_sync, feed_url, feed_name, prov_type)
                    for item in items:
                        add_article(item)
                    provider_outcomes[key] = {
                        "status": "SUCCESS",
                        "count": len(items),
                        "latency_ms": round((time.time() - t_f) * 1000, 1),
                    }
                except Exception as e:
                    logger.warning(f"Failed direct feed {feed_name}: {e}")
                    provider_outcomes[key] = {
                        "status": "FAILED",
                        "error": str(e),
                        "count": 0,
                        "latency_ms": round((time.time() - t_f) * 1000, 1),
                    }

        # 3. Build Google News RSS query with when:{N}h bound
        lookback_hours = session.window_duration_hours if session else 24.0
        n_hours = min(72, max(1, math.ceil(lookback_hours) + 1))
        when_tag = f"when:{n_hours}h"

        # Determine base search subject: For crypto, forex, commodities use asset name
        if asset_type in (AssetType.CRYPTO, AssetType.FOREX, AssetType.COMMODITY):
            base_query = company_name or ticker
        else:
            base_query = f"{ticker} {company_name}".strip()

        domain_map = (
            FOREX_COMMODITY_PUBLISHER_DOMAINS
            if asset_type in (AssetType.FOREX, AssetType.COMMODITY)
            else EQUITY_PUBLISHER_DOMAINS
        )

        # 4. Fetch Google News RSS general query and per-publisher sweeps in parallel
        t_rss = time.time()
        feed_tasks = []
        feed_keys = []

        # 4a. General query (no site constraint)
        gen_query = f"{base_query} {when_tag}"
        encoded_gen = urllib.parse.quote_plus(gen_query)
        gen_url = f"https://news.google.com/rss/search?q={encoded_gen}&hl=en-US&gl=US&ceid=US:en"
        feed_tasks.append(
            asyncio.to_thread(
                cls._fetch_feed_httpx_sync, gen_url, "Google News", ProviderType.GOOGLE_RSS_AGGREGATOR
            )
        )
        feed_keys.append("rss_google_news_general")

        # 4b. Site sweeps
        for pub_name, domain in domain_map.items():
            sweep_query = f"{base_query} site:{domain} {when_tag}"
            encoded_sweep = urllib.parse.quote_plus(sweep_query)
            sweep_url = f"https://news.google.com/rss/search?q={encoded_sweep}&hl=en-US&gl=US&ceid=US:en"
            feed_tasks.append(
                asyncio.to_thread(
                    cls._fetch_feed_httpx_sync, sweep_url, pub_name, ProviderType.GOOGLE_RSS_AGGREGATOR
                )
            )
            feed_keys.append(f"rss_{pub_name.lower().replace(' ', '_').replace('.', '_')}")

        # Execute RSS tasks with per-provider isolation
        rss_results = await asyncio.gather(*feed_tasks, return_exceptions=True)
        for key, res in zip(feed_keys, rss_results):
            if isinstance(res, list):
                for item in res:
                    add_article(item)
                provider_outcomes[key] = {
                    "status": "SUCCESS",
                    "count": len(res),
                }
            else:
                provider_outcomes[key] = {
                    "status": "FAILED",
                    "error": str(res),
                    "count": 0,
                }

        provider_outcomes["google_news_rss_total"] = {
            "status": "SUCCESS" if any(provider_outcomes[k]["status"] == "SUCCESS" for k in feed_keys) else "FAILED",
            "count": sum(provider_outcomes[k]["count"] for k in feed_keys if provider_outcomes[k]["status"] == "SUCCESS"),
            "latency_ms": round((time.time() - t_rss) * 1000, 1),
        }

        logger.info(
            f"Ingestion sweep completed for {ticker} ({asset_type.value}): "
            f"{len(all_articles)} unique headlines ingested across {len(provider_outcomes)} feeds."
        )
        return all_articles, provider_outcomes

