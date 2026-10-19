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
                (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url")
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
        if not getattr(feed, "version", "") and not getattr(feed, "entries", []):
            raise ValueError("The source did not return an RSS or Atom feed.")
        articles: list[IngestedArticle] = []
        now = datetime.now(timezone.utc)

        for entry in getattr(feed, "entries", []):
            raw_title = getattr(entry, "title", "")
            raw_summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
            link = getattr(entry, "link", "")
            # Yahoo's direct symbol RSS often omits <source>; recover the
            # original publisher from its normal Yahoo news <guid> URL.
            inferred_origin = ""
            if provider == ProviderType.YFINANCE:
                guid = str(getattr(entry, "id", ""))
                if guid.startswith(("http://", "https://")):
                    inferred_origin = urllib.parse.urlsplit(guid).hostname or ""

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
                publisher = inferred_origin or default_source_name
                inferred = True

            title = cls._clean_entry_title(raw_title, publisher)
            if not title:
                continue

            summary = cls._clean_entry_summary(raw_summary, title, publisher)

            # Publication time
            pub_time: datetime | None = None
            parsed_date = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
            if parsed_date:
                try:
                    pub_time = datetime(*parsed_date[:6], tzinfo=timezone.utc)
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
        timeout = httpx.Timeout(10.0, connect=4.0)
        with httpx.Client(timeout=timeout, follow_redirects=True, headers={
            "User-Agent": "SentimentPulse/1.0 (public RSS reader)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9",
        }) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content
            return cls.parse_rss_feed_content(content, default_source_name, provider)

    @staticmethod
    def build_search_query(ticker: str, company_name: str, hours: int,
                           asset_type=AssetType.EQUITY) -> str:
        # An AND query such as "AAPL Apple Inc." misses headlines that only say Apple.
        core = re.sub(r"\b(incorporated|inc|corporation|corp|limited|ltd|plc)\b", "", company_name,
                      flags=re.IGNORECASE)
        core = re.sub(r"\s+", " ", core).strip(" .")
        aliases = {
            "GC=F": ["gold"], "SI=F": ["silver"], "CL=F": ["crude oil", "WTI"],
            "BZ=F": ["Brent oil"], "NG=F": ["natural gas"], "HG=F": ["copper"],
            "BTC-USD": ["bitcoin", "BTC"], "ETH-USD": ["ethereum", "ETH"],
            "SOL-USD": ["solana", "SOL"],
        }
        terms = [ticker.replace('"', ""), core.replace('"', ""), *aliases.get(ticker, [])]
        terms = list(dict.fromkeys(t for t in terms if t))
        subject = "(" + " OR ".join('"' + t + '"' for t in terms) + ")"
        finance = " (stock OR shares OR earnings OR profit OR revenue OR dividend OR analyst)" if asset_type in (
            AssetType.EQUITY, AssetType.ETF) else ""
        return f"{subject}{finance} when:{hours}h"

    @classmethod
    async def fetch_news_for_session(cls, ticker, company_name, session,
                                     asset_type=AssetType.EQUITY, yf_symbol=""):
        """Fetch public RSS sources concurrently, independently of FinBERT and quote APIs."""
        symbol = urllib.parse.quote(yf_symbol or ticker, safe="")
        hours = min(72, max(1, math.ceil(session.window_duration_hours) + 1))
        query = cls.build_search_query(ticker, company_name, hours, asset_type)
        def google_url(q):
            return "https://news.google.com/rss/search?" + urllib.parse.urlencode({
                "q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        feeds = [
            ("Yahoo Finance RSS", f"https://finance.yahoo.com/rss/headline?s={symbol}", ProviderType.YFINANCE),
            ("Google News", google_url(query), ProviderType.GOOGLE_RSS_AGGREGATOR),
            ("CNBC direct RSS", "https://www.cnbc.com/id/100003114/device/rss/rss.html", ProviderType.DIRECT_RSS),
        ]
        domains = (FOREX_COMMODITY_PUBLISHER_DOMAINS
                   if asset_type in (AssetType.FOREX, AssetType.COMMODITY)
                   else EQUITY_PUBLISHER_DOMAINS)
        for publisher, domain in domains.items():
            feeds.append((f"{publisher} via Google News",
                          google_url(f"{query} site:{domain}"), ProviderType.GOOGLE_RSS_AGGREGATOR))
        if asset_type == AssetType.FOREX:
            feeds.extend([
                ("FXStreet RSS", "https://www.fxstreet.com/rss/news", ProviderType.FXSTREET_RSS),
                ("Investing.com Forex RSS", "https://www.investing.com/rss/news_1.rss", ProviderType.INVESTING_FOREX_RSS),
            ])

        async def get_feed(name, url, provider):
            started = time.perf_counter()
            try:
                articles = await asyncio.to_thread(cls._fetch_feed_httpx_sync, url, name, provider)
                # RSS channel defaults identify the collector when publisher information is absent.
                return articles, {
                    "status": "SUCCESS", "count": len(articles),
                    "fresh_count": sum(session.is_timestamp_in_window(a.published_at_utc) for a in articles),
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "url": url, "error": ""}
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                reason = "Rate limited" if code == 429 else ("Access denied by source" if code in (401, 403) else "HTTP error")
                return [], {"status": "FAILED", "count": 0, "fresh_count": 0,
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                            "url": url, "error": f"{reason} ({code})"}
            except Exception as exc:
                reason = "Request timed out" if isinstance(exc, httpx.TimeoutException) else (
                    "Invalid feed response" if isinstance(exc, ValueError) else "Connection or feed parsing error")
                return [], {"status": "FAILED", "count": 0, "fresh_count": 0,
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                            "url": url, "error": reason}

        results = await asyncio.gather(*(get_feed(*feed) for feed in feeds))
        outcomes, articles, seen_urls, seen_titles = {}, [], set(), set()
        for (name, _url, _provider), (items, outcome) in zip(feeds, results):
            outcomes[name] = outcome
            for article in items:
                url = article.canonical_url.split("?")[0].rstrip("/").lower()
                title = article.title.strip().lower()
                if (url and url in seen_urls) or (title and title in seen_titles):
                    continue
                if url:
                    seen_urls.add(url)
                seen_titles.add(title)
                articles.append(article)
        logger.info("Collected %s unique headlines for %s across %s feeds.", len(articles), ticker, len(feeds))
        return articles, outcomes

