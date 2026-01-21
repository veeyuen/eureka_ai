# Eureka AI — Source-Anchored Metrics, Canonicalisation & Diffing (Streamlit)

A Streamlit app that extracts numeric metrics from web sources, canonicalises them into stable metric identities, and compares runs over time (Analysis → Evolution) to show changes, stability, and evidence quality.

This project is designed for **source-anchored** metric tracking (e.g., EV market metrics), where every number is tied back to the sources and extraction evidence used to derive it.

> ⚠️ **Not financial advice.** Outputs are best-effort extractions from sources and may be incomplete or incorrect. Always verify against primary sources.

---

## What this app does

### Core capabilities
- **Analysis run (baseline):**
  - Fetches sources, extracts candidate numbers, selects point estimates
  - Builds a **canonical metric identity** for each metric (stable keys)
  - Persists a baseline that future runs can diff against

- **Evolution run (current):**
  - Fetches current sources (optionally with injected URLs)
  - Rebuilds canonical metrics using the same identity rules as Analysis
  - Computes a **canonical-first diff** to produce the “Metric Changes” table
  - Displays:
    - current vs previous value
    - increased / decreased / unchanged classification
    - stability score
    - source reliability and evidence quality

### UI panels you’ll see
- **Metric Changes (Diff Panel V2)** — canonical-first comparison of baseline vs current
- **Data Visualization** — charts when `visualization_data` / `comparison_bars` exist
- **Sources & Reliability** — reliability classification + freshness signals
- **Evidence Quality Scores** — source quality, numeric consistency, citation density, consensus, overall

---

## High-level architecture

### Two pipelines, one canonical identity system
1. **Analysis pipeline**
   - Extracts and selects values
   - Materialises `primary_metrics_canonical` using canonical key rules
   - Persists results (including snapshot hashes and/or baseline caches)

2. **Evolution pipeline**
   - Rehydrates the baseline from Analysis
   - Computes “current” canonical metrics
   - Joins baseline ↔ current via canonical keys
   - Emits `metric_changes_v2` (rows shown in UI)

### Canonical keys (the spine)
Canonical keys are the “identity backbone” that ensures the *same semantic metric* lands on the *same key* across runs, enabling deterministic diffing.

A canonical key is derived from semantic attributes such as:
- metric meaning (e.g., sales / market share / market size)
- geography scope (global / region / country)
- time scope (FY / YTD / quarter / year)
- unit family + unit (units, %, USD bn, etc.)

---

## Quick start

### Prerequisites
- Python (version depends on your environment; use whatever the project’s lock/requirements specify)
- Streamlit

### Run locally
```bash
# Create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies (adjust for your repo)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run FIX2D92_full_codebase_streamlit_safe.py
