# Validation

Verified locally with Python 3.12 and Streamlit 1.50.0:

- Initial Streamlit screen and offline demo render through Streamlit AppTest.
- Demo evidence filters, history, and session reset.
- CSV export, including formula-injection handling, and JSON serialization.
- Retained counts exclude rejected headlines; publisher counts use directional evidence.
- No-news, no-relevant-news, and insufficient-evidence labels.
- Cryptocurrency rolling window.
- Simulated partial news-provider outage returns a directly usable result.
- Stale news is excluded before scoring.
- Invalid external-AI JSON is rejected.
- Explicit equity aliases do not override later commodity lookups.
- Lazy scorer and batch-output mapping.

The automated tests use simulated news and model outputs.
Live RSS endpoints were subsequently checked for AAPL, and their returned XML was
parsed and rendered through the real collection pipeline and Streamlit AppTest.
The automated suite also verifies collection without quote API/model calls and
preservation of headlines after a model failure.

Real FinBERT weights/inference, authenticated OpenRouter calls,
and deployment on Streamlit Community Cloud have not been validated in this environment.
The full PyTorch runtime test was attempted but could not finish because the
workspace ran out of disk space during dependency extraction. Temporary downloads
were removed. This does not establish a deployment failure, but live testing is
still required on the target host.

The ZIP contains source, configuration, tests, license, and instructions only.
It intentionally excludes downloaded models, API keys, environments, and temporary caches.
