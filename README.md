# longtail-txcat

Code and frozen caches for *Categorizing the Long Tail: An Empirical Study of Web-Search-Augmented
Fallback for Embedding-Based Transaction Classification* (Surya Parida, Independent Researcher).

Setup: `curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv --python 3.12 .venv && uv pip install -e ".[dev]"`
Data: `.venv/bin/python data/download.py dc oklahoma mcc_codes` (datasets are not redistributed).
Tests: `.venv/bin/pytest`.
