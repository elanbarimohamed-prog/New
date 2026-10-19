"""Run with: python -m streamlit run streamlit_app.py"""
from __future__ import annotations

import asyncio
import html
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import streamlit as st

st.set_page_config(page_title="SentimentPulse", page_icon="📊", layout="wide")

# Set server configuration before importing the analysis modules. There is no
# per-user key widget and no mutation of shared model settings during a run.
try:
    for key in ("OPENROUTER_API_KEY", "LLM_MODEL_ID", "LLM_FALLBACK_MODEL_ID", "MAX_COST_PER_RUN_USD"):
        value = st.secrets.get(key)
        if value is not None:
            os.environ[key] = str(value)
except st.errors.StreamlitSecretNotFoundError:
    pass

from app.config import settings
from app.core.models import AnalysisResult, NewsCollection
from app.streamlit_support import (
    article_rows, article_tone, csv_export, demo_result, display_verdict,
    evidence_counts, export_name, safe_url, collection_rows, collection_csv,
)

st.markdown("""
<style>
.block-container {max-width:1240px;padding-top:2.2rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {border-right:1px solid #22324a;}
.eyebrow {font-size:.74rem;letter-spacing:.18em;color:#65cef6;font-weight:700;}
.intro {font-size:1.05rem;color:#9bacbf;max-width:730px;line-height:1.65;}
.result-box {padding:1.4rem 1.6rem;border:1px solid #2a3d58;border-radius:16px;
background:linear-gradient(125deg,#122239,#0e192b);margin:.8rem 0 1.3rem;}
.result-box h2 {margin:.35rem 0;color:#edf5ff;}
.result-box p {color:#a7b9cf;margin:0;}
.small-label {font-size:.8rem;color:#79d1f4;letter-spacing:.08em;}
[data-testid="stMetric"] {background:#111e32;border:1px solid #21324b;
border-radius:12px;padding:1rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_scorer():
    from app.services.finbert_service import FinBERTService
    return FinBERTService()


def fetch_collection(ticker, window_mode, article_limit, progress):
    from app.services.orchestrator import collect_headlines
    return asyncio.run(collect_headlines(ticker, window_mode, article_limit, progress))


def score_collection(collection, enable_llm, progress):
    from app.services.orchestrator import analyze_collection
    return asyncio.run(analyze_collection(collection, get_scorer(), enable_llm, progress))


def remember(result):
    runs = st.session_state.setdefault("history", [])
    runs.insert(0, result.model_dump(mode="json"))
    del runs[15:]
    st.session_state["result"] = runs[0]


with st.sidebar:
    st.markdown("### ◉ SentimentPulse")
    st.caption("FINANCIAL NEWS · EXPLAINED")
    st.divider()
    st.markdown("**Analysis settings**")
    window_mode = st.selectbox("News window", ["Last 24 hours", "US market session"])
    article_limit = st.slider("Maximum headlines to score", 10, 100, 40, step=10)
    can_audit = bool(settings.OPENROUTER_API_KEY and settings.LLM_MODEL_ID)
    enable_llm = st.checkbox("Enable AI catalyst audit", disabled=not can_audit,
                             help="Optional OpenRouter audit. Selected catalysts get extra weight.")
    if not can_audit:
        st.caption("Optional audit: configure an OpenRouter key and model in app secrets.")
    force_refresh = st.checkbox("Fetch fresh results", help="Skip this session's five-minute result cache.")
    st.divider()
    if st.button("Explore a demo", use_container_width=True):
        st.session_state.pop("collection", None)
        remember(demo_result())
    st.caption("Demo uses fictional headlines and fixed example scores. No news or AI calls.")
    runs = st.session_state.get("history", [])
    if runs:
        st.divider()
        st.markdown("**Your session history**")
        selection = st.selectbox(
            "Previous analyses", range(len(runs)),
            format_func=lambda i: f"{runs[i]['ticker']} · {runs[i]['session']['run_timestamp_utc'][11:16]} UTC",
        )
        if st.button("Open selected analysis", use_container_width=True):
            st.session_state.pop("collection", None)
            st.session_state["result"] = runs[selection]
        if st.button("Clear session history", use_container_width=True):
            st.session_state["history"] = []
            st.session_state.pop("result", None)
            st.session_state.pop("collection", None)
            st.session_state["collection_cache"] = {}
            st.session_state["analysis_cache"] = {}
            st.rerun()
    st.caption("History is private to this browser session and can reset. Download results to keep them.")
    st.markdown("[Original project](https://github.com/elem160/Market-Sentiment-2)")

st.markdown('<div class="eyebrow">MARKET INTELLIGENCE / NEWS SENTIMENT</div>', unsafe_allow_html=True)
st.title("Read the market's mood.")
st.markdown('<p class="intro">Bring recent financial headlines into focus. Inspect the tone, '
            'follow the evidence, and understand what shaped the result.</p>', unsafe_allow_html=True)

with st.form("analysis_form"):
    left, right = st.columns([4, 1], vertical_alignment="bottom")
    with left:
        ticker = st.text_input("Asset symbol", value="AAPL", max_chars=16,
                               placeholder="AAPL, NVDA, EUR/USD, GC=F, BTC-USD")
    with right:
        submitted = st.form_submit_button("Fetch live headlines", type="primary", use_container_width=True)
st.caption("Try AAPL · NVDA · SPY · EUR/USD · GC=F · BTC-USD")

if submitted:
    ticker = ticker.strip().upper()
    if not ticker:
        st.warning("Enter an asset symbol first.")
    else:
        st.session_state.pop("result", None)
        key = ("headlines-v2", ticker, window_mode, article_limit)
        cache = st.session_state.setdefault("collection_cache", {})
        cached = cache.get(key)
        if not force_refresh and cached and time.time() - cached["saved_at"] < 300:
            st.session_state["collection"] = cached["collection"]
            st.info("Loaded your recently fetched headlines. Use Fetch fresh results to refresh.")
        else:
            st.session_state.pop("collection", None)
            with st.status("Fetching live headlines", expanded=True) as status:
                bar = st.progress(0.0)
                stage_text = st.empty()
                def progress(value, text):
                    bar.progress(value)
                    stage_text.write(text)
                try:
                    collection = fetch_collection(ticker, window_mode, article_limit, progress)
                    snapshot = collection.model_dump(mode="json")
                    st.session_state["collection"] = snapshot
                    if any(p.get("status") == "SUCCESS" for p in collection.provider_outcomes.values()):
                        cache[key] = {"saved_at": time.time(), "collection": snapshot}
                        if len(cache) > 15:
                            del cache[next(iter(cache))]
                    status.update(label=f"Fetched {len(collection.articles)} unique headlines",
                                  state="complete" if collection.articles else "error", expanded=False)
                except Exception as exc:
                    status.update(label="Headline collection could not complete", state="error")
                    st.error("Could not collect news. Check the symbol and try again.")
                    from app.utils.logging import logger
                    logger.error("Headline collection failed (%s)", type(exc).__name__)

snapshot = st.session_state.get("collection")
if snapshot:
    collection = NewsCollection.model_validate(snapshot)
    rows = collection_rows(collection)
    ready_count = sum(r["Status"] == "Ready for analysis" for r in rows)
    successful = sum(p.get("status") == "SUCCESS" for p in collection.provider_outcomes.values())
    st.markdown("### Live headline collection")
    st.write(f"**{collection.asset.ticker} · {collection.asset.company_name}**")
    st.caption(f"Fetched {collection.session.run_timestamp_utc:%d %b %Y, %H:%M} UTC. "
               "These are real feed results; no AI model was required to collect them.")
    cols = st.columns(3)
    cols[0].metric("Unique headlines fetched", len(collection.articles))
    cols[1].metric("Ready for analysis", ready_count)
    cols[2].metric("Feeds responding", f"{successful}/{len(collection.provider_outcomes)}")
    if successful == 0:
        st.error("Every feed request failed on this host. Open Source connection status below for the reasons.")
    elif not rows:
        st.info("The feeds responded but returned no articles for this search.")
    with st.expander("Source connection status", expanded=successful == 0):
        status_rows = [{"Feed": name, "Status": p.get("status", ""),
                        "Returned": p.get("count", 0), "Inside window": p.get("fresh_count", 0),
                        "Response (ms)": p.get("latency_ms", 0), "Details": p.get("error", ""),
                        "Feed URL": safe_url(p.get("url", "")) or ""}
                       for name, p in collection.provider_outcomes.items()]
        st.dataframe(status_rows, hide_index=True, use_container_width=True,
                     column_config={"Feed URL": st.column_config.LinkColumn("Feed URL")})
    st.caption("Google News searches index publishers; a responding feed does not imply direct publisher API access. "
               "Yahoo RSS may report the feed name when an original publisher is not supplied.")
    if rows:
        eligible_only = st.checkbox("Only show headlines ready for analysis", value=True)
        search = st.text_input("Search collected headlines", placeholder="Search a headline or publisher")
        shown = [r for r in rows if (not eligible_only or r["Status"] == "Ready for analysis")
                 and (not search or search.casefold() in (r["Headline"] + " " + r["Publisher / feed"]).casefold())]
        st.dataframe([{k: v for k, v in r.items() if k != "Article ID"} for r in shown],
                     hide_index=True, use_container_width=True,
                     column_config={"Link": st.column_config.LinkColumn("Read source")})
        st.download_button("Download collected headlines · CSV", collection_csv(collection),
                           file_name=f"{export_name(collection.asset.ticker)}_collected_headlines.csv",
                           mime="text/csv", key="collection_csv")
        if shown:
            inspected_id = st.selectbox("Read a collected headline", [r["Article ID"] for r in shown],
                                        format_func=lambda value: next(r["Headline"] for r in shown
                                                                       if r["Article ID"] == value))
            article = next(a for a in collection.articles if a.article_id == inspected_id)
            st.write(article.summary or "No summary was supplied by this feed.")
            if safe_url(article.canonical_url):
                st.link_button("Open news article", article.canonical_url)
    if st.button("Analyze collected headlines with FinBERT", type="primary", disabled=ready_count == 0):
        if enable_llm and time.time() - st.session_state.get("last_audit_request", 0) < 60:
            st.warning("Please wait one minute between AI audit requests.")
        else:
            if enable_llm:
                st.session_state["last_audit_request"] = time.time()
            with st.status("Analyzing collected headlines", expanded=True) as status:
                bar = st.progress(0.0)
                message = st.empty()
                def progress(value, text):
                    bar.progress(value)
                    message.write(text)
                try:
                    result = score_collection(collection, enable_llm, progress)
                    remember(result)
                    status.update(label="Sentiment analysis complete", state="complete", expanded=False)
                except Exception as exc:
                    st.session_state.pop("result", None)
                    status.update(label="AI analysis unavailable", state="error")
                    st.error("Headlines were collected successfully and remain available above. "
                             "FinBERT could not finish loading or scoring. Check the hosting logs for model errors.")
                    from app.utils.logging import logger
                    logger.exception("FinBERT analysis failed (%s)", type(exc).__name__)
    st.divider()

data = st.session_state.get("result")
if not data and not snapshot:
    st.divider()
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 01 · Collect")
        st.write("Find recent headlines from multiple news feeds.")
    with b:
        st.markdown("#### 02 · Understand")
        st.write("FinBERT classifies financial text as positive, negative, or neutral.")
    with c:
        st.markdown("#### 03 · Inspect")
        st.write("See the score and inspect each headline behind it.")
    st.info("Start with a symbol above, or explore the demo in the sidebar.")
elif data:
    result = AnalysisResult.model_validate(data)
    counts = evidence_counts(result)
    if result.data_status == "DEMO":
        st.warning("DEMO — Fictional headlines and pre-assigned scores. This is not live market data.")
    if result.data_status == "NO_RELEVANT_NEWS":
        st.info("Fresh headlines were found, but none passed relevance and duplicate filters.")
    title = html.escape(f"{result.ticker} · {result.company_name}")
    label = html.escape(display_verdict(result))
    st.markdown(f'<div class="result-box"><div class="small-label">{title}</div>'
                f'<h2>{label}</h2><p>Based on the tone of news in the selected window.</p></div>',
                unsafe_allow_html=True)
    st.caption(f"Analyzed {result.session.run_timestamp_utc:%d %b %Y, %H:%M} UTC · "
               f"News window: {result.session.window_start_utc:%d %b %H:%M} → "
               f"{result.session.window_end_utc:%d %b %H:%M} UTC")
    cols = st.columns(4)
    cols[0].metric("Sentiment", f"{result.sentiment_score:+.3f}", help="Directional weighted score from −1 to +1.")
    cols[1].metric("Retained headlines", counts["retained"])
    cols[2].metric("Directional publishers", counts["directional_publishers"],
                   help="Publishers contributing positive or negative retained headlines.")
    cols[3].metric("Evidence quality", f"{result.evidence_quality:.0%}",
                   help="A rule-based measure of volume and diversity; not verified accuracy.")
    if result.data_status == "NO_FRESH_NEWS":
        st.info("No fresh dated news passed the chosen window. No direction can be inferred.")
    for warning in result.warnings:
        st.caption(warning)
    overview, evidence, details, learn = st.tabs(["Overview", "Headline evidence", "Run details", "How it works"])
    with overview:
        st.markdown("#### The headline balance")
        import pandas as pd
        chart = pd.DataFrame({"Headlines": [counts["positive"], counts["negative"], counts["neutral"]]},
                             index=["Positive", "Negative", "Neutral"])
        st.bar_chart(chart, color="#38BDF8")
        st.write(f"**{counts['positive']} positive · {counts['negative']} negative · {counts['neutral']} neutral**")
        if result.llm_summary:
            st.markdown("#### Optional AI catalyst summary")
            st.write(result.llm_summary)
        for inference in result.rationale.inferences:
            st.write("• " + inference)
        st.caption(f"Heuristic confidence: {result.decision_confidence:.0%}. "
                   "This is not the probability of a correct prediction.")
    with evidence:
        show_excluded = st.checkbox("Include excluded headlines", value=False)
        tone = st.selectbox("Filter tone", ["All", "Positive", "Negative", "Neutral"])
        rows = [row for row in article_rows(result)
                if (show_excluded or row["Tone"] != "Excluded")
                and (tone == "All" or row["Tone"] == tone)]
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True,
                         column_config={"Link": st.column_config.LinkColumn("Source link")})
            inspected = st.selectbox("Inspect a headline", [r["Headline"] for r in rows])
            article = next(a for a in result.articles if a.title == inspected)
            st.write(article.lead_text)
            if article.exclusion_reason:
                st.info("Excluded: " + article.exclusion_reason.value)
            else:
                st.write(f"FinBERT: positive {article.finbert_positive:.1%}, "
                         f"negative {article.finbert_negative:.1%}, neutral {article.finbert_neutral:.1%}.")
                st.caption(f"Directional score {article.finbert_directional_score:+.3f} · "
                           f"Effective weight {article.effective_weight:.3f}")
            if safe_url(article.canonical_url):
                st.link_button("Read original source", article.canonical_url)
        else:
            st.info("No headlines match this filter.")
    with details:
        st.json({"data_status": result.data_status, "pipeline": result.pipeline_mode.value,
                 "models": result.model_versions, "metrics": result.metrics.model_dump(mode="json")})
        st.caption("Excluded out-of-window stories are sampled in the evidence table; the metrics contain full counts.")
    with learn:
        st.markdown("""
**1. Collect:** Search Yahoo Finance and Google News RSS, plus dedicated forex feeds when relevant.

**2. Clean:** Reject undated, stale, unrelated, and repeated headlines.

**3. Classify:** FinBERT reads headlines and available summaries, not necessarily full articles.
The directional score is positive probability minus negative probability.

**4. Combine:** More recent headlines carry more weight, with a 14-hour half-life.
Neutral headlines are excluded from the directional average.

**5. Label:** Positive or negative labels require enough directional headlines, an imbalance
of at least 1.5 to 1, a score beyond ±0.20, and at least two directional publishers.
Other cases remain mixed or insufficient.

**Optional audit:** An external language model identifies potential catalysts and boosts their
weights by 25%. It does not fact-check stories or remove every non-catalyst.

**Market windows:** Last 24 hours is a rolling window. US market session uses the original
New York clock. Cryptocurrency always uses the rolling 24-hour window.

**Limits:** News tone can differ from price direction. Repeated events, headline ambiguity,
and unavailable feeds can affect results. Confidence and evidence quality are heuristic scores.
""")
    st.divider()
    downloads = st.columns(2)
    with downloads[0]:
        st.download_button("Download headline evidence · CSV", csv_export(result),
                           file_name=f"{export_name(result.ticker)}_headlines.csv",
                           mime="text/csv", use_container_width=True)
    with downloads[1]:
        st.download_button("Download complete analysis · JSON", result.model_dump_json(indent=2),
                           file_name=f"{export_name(result.ticker)}_analysis.json",
                           mime="application/json", use_container_width=True)

st.caption("Research and learning tool. News sentiment is not an investment recommendation.")
