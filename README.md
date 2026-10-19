# SentimentPulse — Streamlit edition

A Streamlit adaptation of [elem160/Market-Sentiment-2](https://github.com/elem160/Market-Sentiment-2).
Collect recent financial headlines, classify their tone with local CPU FinBERT,
inspect the evidence, and export results.

## Start here

**News first:** enter a symbol and click **Fetch live headlines**. This makes real
RSS requests and displays headlines, publication times, source links, and per-feed
connection status without loading FinBERT. Then click **Analyze collected headlines
with FinBERT** if you want sentiment scores. If model inference fails, the collected
headlines remain readable and downloadable.

- Main application file: **streamlit_app.py**
- Recommended Python: **3.12** (3.11 is also supported by the selected main dependencies).
- No Node.js, React build, FastAPI server, database, or API key is required for the basic app.
- Click **Explore a demo** to try the interface immediately with clearly labelled fictional examples.
- A first live analysis downloads FinBERT. Allow extra time and keep internet access available.

## Deploy on Streamlit Community Cloud

1. Extract this ZIP.
2. Create a GitHub repository under your account.
3. Upload the **contents** of this project folder to the repository root, including
   the app folder, requirements.txt, streamlit_app.py, LICENSE, and .streamlit/config.toml.
   Do not upload the ZIP as your app source.
4. Go to https://share.streamlit.io/ and choose **Create app**.
5. Select your repository and branch (usually main).
6. Set **Main file path** to **streamlit_app.py**.
7. In **Advanced settings**, select Python **3.12**.
8. Deploy. The basic app works without secrets.

Official instructions:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

The CPU PyTorch wheel is selected explicitly on Linux to avoid installing CUDA packages.
Model loading remains memory intensive. A sleeping/restarted host may need to reload or download
the model. Hosting resources and news-provider access can vary; deployment itself has to be
verified on the target account. Reducing the headline slider reduces inference time, not model size.

## Run locally on Windows

Open PowerShell inside this extracted project folder:

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py

Open the Local URL printed by Streamlit, normally http://localhost:8501.
You can also use start_windows.bat after creating the environment and installing dependencies.
This uses the environment's Python directly, so PowerShell activation is unnecessary.

On macOS/Linux:

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m streamlit run streamlit_app.py

## Optional external AI audit

FinBERT alone provides the core analysis. The audit is off by default.
To enable the checkbox, add these values to Community Cloud's **Secrets** settings:

    OPENROUTER_API_KEY = "your-key"
    LLM_MODEL_ID = "an-available-openrouter-model-id"
    LLM_FALLBACK_MODEL_ID = ""
    MAX_COST_PER_RUN_USD = 0.05

For local use, copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and edit it.
Never commit secrets.toml or .env.

Choose an actual currently available model from your provider account. The audit sends
candidate headlines and publisher names to OpenRouter and its selected model provider.
Only explicitly configured models are tried. If unavailable, the app keeps the FinBERT result.
Cost checks use approximate token counts and configured model-price estimates, not a billing cap.
Verify pricing and enforce account spending limits with the provider.

## What is included

- Multi-asset symbol handling: stocks, ETFs, forex, futures/commodities, crypto.
- Direct Yahoo Finance symbol RSS, Google News financial searches, CNBC direct RSS,
  and forex-specific feeds. Sources are fetched concurrently with individual timeouts.
- Freshness, relevance, and duplicate filtering.
- CPU FinBERT model shared safely across Streamlit reruns and browser sessions.
- Rolling 24-hour window or the original New York market-session window.
  Crypto always uses a rolling 24-hour window.
- Positive/negative/mixed sentiment labels; distinct no-news and insufficient-evidence states.
- Correct retained-article counts and directional-publisher counts.
- Per-headline probabilities, exclusion reasons, source links, and filters.
- CSV and complete JSON downloads.
- Five-minute per-session analysis cache and a maximum of 15 session-history entries.
- Optional AI catalyst audit and a fictional offline interface demo.

## Changes from the original

- React/FastAPI and SQLite job polling were replaced with Streamlit forms and direct results.
  Partial RSS outages therefore do not depend on fetching an unsaved database run.
- Model loading is lazy; the demo does not import PyTorch or download weights.
- Inference is serialized with a lock to protect the shared model and bound CPU usage.
- Headlines are scored in batches of four; the default scoring cap is 40.
- The original BUY/HOLD/SELL engine remains visible in exported JSON as a research classification;
  the dashboard presents news-tone labels instead.
- The optional audit closes its HTTP client before each event loop ends, checks each configured
  fallback's estimated cost, rejects malformed JSON, and bounds catalog size.
- No permanent/shared search-history database is included. History belongs to the browser session.
  It can disappear on disconnect or restart. Download results to retain them.
- Shared model configuration is server-managed; visitors cannot submit API keys into global state.

## How to read the result

FinBERT estimates positive, negative, and neutral text probabilities.
Directional score = positive probability minus negative probability.
Differences below 0.15 in magnitude count as neutral.

The original aggregator combines directional scores with confidence and recency weights,
using a 14-hour half-life. Positive/negative direction requires at least two headlines
in that direction, a count ratio of at least 1.5, a composite score beyond +0.20/-0.20,
and at least two recognized publishers among directional evidence.

The optional audit boosts selected catalyst weights by 25%; it does not independently
verify news or automatically remove all other headlines.

Confidence and evidence quality are heuristic scores, not calibrated probabilities.
The app processes headlines and available short summaries rather than complete paid articles.
News tone does not establish fair value, future return, or a profitable trading strategy.

## Troubleshooting

- **Module not found:** install requirements with the same Python used to run Streamlit.
- **Model download fails:** check host network access to Hugging Face; retry after a transient outage.
- **No fresh news:** try Last 24 hours or a widely covered asset; undated/stale stories are excluded.
- **Some feeds unavailable:** expand **Source connection status** to see each feed's
  returned count, fresh count, response time, and errors. Collection uses the surviving feeds.
- **App exceeds host memory:** use a host with more memory; lowering article count cannot shrink weights.
- **AI checkbox disabled:** configure both OPENROUTER_API_KEY and LLM_MODEL_ID in secrets.
- **Streamlit entrypoint not found:** streamlit_app.py must be at the selected repository path.

## Tests

    python -m pip install -r requirements-dev.txt
    python -m pytest -q

The default tests use simulated news and model outputs. They check the Streamlit interface,
filtering, evidence counts, exports, window handling, and error paths. They do not prove
market predictive accuracy or live-provider uptime.

## Attribution

Original project: https://github.com/elem160/Market-Sentiment-2

This is a separate Streamlit adaptation, not an official release of that project.
The original MIT copyright and license are preserved in LICENSE.
