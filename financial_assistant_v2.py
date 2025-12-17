# =========================================================
# YUREEKA AI RESEARCH ASSISTANT v7.5
# With Web Search, Evidence-Based Verification, Confidence Scoring
# SerpAPI Output with Evolution Layer Version
# Updated SerpAPI parameters for stable output
# Deterministic Output From LLM
# Anchored Evolution Analysis Using JSON As Input Into Model
# =========================================================

import os
import re
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import base64
import hashlib
import numpy as np
import difflib
import google.generativeai as genai
from typing import Dict, List, Optional, Any, Union, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from collections import Counter
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# =========================================================
# 1. CONFIGURATION & API KEY VALIDATION
# =========================================================

def load_api_keys():
    """Load and validate API keys from secrets or environment"""
    try:
        PERPLEXITY_KEY = st.secrets.get("PERPLEXITY_API_KEY") or os.getenv("PERPLEXITY_API_KEY", "")
        GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        SERPAPI_KEY = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY", "")
        SCRAPINGDOG_KEY = st.secrets.get("SCRAPINGDOG_KEY") or os.getenv("SCRAPINGDOG_KEY", "")
    except Exception:
        PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY", "")
        GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
        SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
        SCRAPINGDOG_KEY = os.getenv("SCRAPINGDOG_KEY", "")

    # Validate critical keys
    if not PERPLEXITY_KEY or len(PERPLEXITY_KEY) < 10:
        st.error("❌ PERPLEXITY_API_KEY is missing or invalid")
        st.stop()

    if not GEMINI_KEY or len(GEMINI_KEY) < 10:
        st.error("❌ GEMINI_API_KEY is missing or invalid")
        st.stop()

    return PERPLEXITY_KEY, GEMINI_KEY, SERPAPI_KEY, SCRAPINGDOG_KEY

PERPLEXITY_KEY, GEMINI_KEY, SERPAPI_KEY, SCRAPINGDOG_KEY = load_api_keys()
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Configure Gemini
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# =========================================================
# 2. PYDANTIC MODELS
# =========================================================

class MetricDetail(BaseModel):
    """Individual metric with name, value, and unit"""
    name: str = Field(..., description="Metric name")
    value: Union[float, int, str] = Field(..., description="Metric value")
    unit: str = Field(default="", description="Unit of measurement")
    model_config = ConfigDict(extra='ignore')

class TopEntityDetail(BaseModel):
    """Entity in top_entities list"""
    name: str = Field(..., description="Entity name")
    share: Optional[str] = Field(None, description="Market share")
    growth: Optional[str] = Field(None, description="Growth rate")
    model_config = ConfigDict(extra='ignore')

class TrendForecastDetail(BaseModel):
    """Trend forecast item"""
    trend: str = Field(..., description="Trend description")
    direction: Optional[str] = Field(None, description="Direction indicator")
    timeline: Optional[str] = Field(None, description="Timeline")
    model_config = ConfigDict(extra='ignore')

class VisualizationData(BaseModel):
    chart_labels: List[str] = Field(default_factory=list)
    chart_values: List[Union[float, int]] = Field(default_factory=list)
    chart_title: Optional[str] = Field("Trend Analysis")
    chart_type: Optional[str] = Field("line")
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    model_config = ConfigDict(extra='ignore')

class ComparisonBar(BaseModel):
    """Comparison bar chart data"""
    title: str = Field("Comparison", description="Chart title")
    categories: List[str] = Field(default_factory=list)
    values: List[Union[float, int]] = Field(default_factory=list)
    model_config = ConfigDict(extra='ignore')

class BenchmarkTable(BaseModel):
    """Benchmark table row"""
    category: str
    value_1: Union[float, int, str] = Field(default=0, description="Numeric value or string")
    value_2: Union[float, int, str] = Field(default=0, description="Numeric value or string")
    model_config = ConfigDict(extra='ignore')

class Action(BaseModel):
    """Investment/action recommendation"""
    recommendation: str = Field("Neutral", description="Buy/Hold/Sell/Neutral")
    confidence: str = Field("Medium", description="High/Medium/Low")
    rationale: str = Field("", description="Reasoning")
    model_config = ConfigDict(extra='ignore')

class LLMResponse(BaseModel):
    """Complete LLM response schema"""
    executive_summary: str = Field(..., description="High-level summary")
    primary_metrics: Dict[str, MetricDetail] = Field(default_factory=dict)
    key_findings: List[str] = Field(default_factory=list)
    top_entities: List[TopEntityDetail] = Field(default_factory=list)
    trends_forecast: List[TrendForecastDetail] = Field(default_factory=list)
    visualization_data: Optional[VisualizationData] = None
    comparison_bars: Optional[ComparisonBar] = None
    benchmark_table: Optional[List[BenchmarkTable]] = None
    sources: List[str] = Field(default_factory=list)
    confidence: Union[float, int] = Field(default=75)
    freshness: Optional[str] = Field(None)
    action: Optional[Action] = None
    model_config = ConfigDict(extra='ignore')

# =========================================================
# 3. PROMPTS
# =========================================================

RESPONSE_TEMPLATE = """
{
  "executive_summary": "1-2 sentence high-level answer",
  "primary_metrics": {
    "metric_1": {"name": "Key Metric 1", "value": 25.5, "unit": "%"},
    "metric_2": {"name": "Key Metric 2", "value": 623, "unit": "$B"}
  },
  "key_findings": [
    "Finding 1 with quantified impact",
    "Finding 2 explaining drivers"
  ],
  "top_entities": [
    {"name": "Entity 1", "share": "25%", "growth": "15%"}
  ],
  "trends_forecast": [
    {"trend": "Trend description", "direction": "↑", "timeline": "2025-2027"}
  ],
  "visualization_data": {
    "chart_labels": ["2023", "2024", "2025"],
    "chart_values": [100, 120, 145],
    "chart_title": "Market Growth",
    "chart_type": "line"
  },
  "comparison_bars": {
    "title": "Market Share",
    "categories": ["A", "B", "C"],
    "values": [45, 30, 25]
  },
  "benchmark_table": [
    {"category": "Company A", "value_1": 25.5, "value_2": 623}
  ],
  "sources": ["source1.com"],
  "confidence": 87,
  "freshness": "Dec 2024",
  "action": {
    "recommendation": "Buy/Hold/Neutral/Sell",
    "confidence": "High/Medium/Low",
    "rationale": "1-sentence reason"
  }
}
"""

SYSTEM_PROMPT = f"""You are a professional market research analyst.

CRITICAL RULES:
1. Return ONLY valid JSON. NO markdown, NO code blocks, NO extra text.
2. NO citation references like [1][2] inside strings.
3. Use double quotes for all keys and string values.
4. NO trailing commas in arrays or objects.
5. Escape internal quotes with backslash.

NUMERIC FIELD RULES (IMPORTANT):
- In benchmark_table: value_1 and value_2 MUST be numbers (never "N/A", "null", or text)
- If data unavailable, use 0 for benchmark_table values
- In primary_metrics: values can be numbers or strings with units (e.g., "25.5" or "25.5 billion")
- In top_entities: share and growth can be strings (e.g., "25%")

REQUIRED FIELDS (provide substantive data):

**executive_summary** - MUST be 4-6 complete sentences covering:
  • Sentence 1: Direct answer with specific quantitative data (market size, revenue, units, etc.)
  • Sentence 2: Major players or regional breakdown with percentages/numbers
  • Sentence 3: Key growth drivers or market dynamics
  • Sentence 4: Future outlook with projected CAGR, timeline, or target values
  • Sentence 5 (optional): Challenge, risk, or competitive dynamic

  BAD (too short): "The EV market is growing rapidly due to government policies."

  GOOD: "The global electric vehicle market reached 14.2 million units sold in 2023, representing 18% of total auto sales. China dominates with 60% market share, followed by Europe (25%) and North America (10%). Growth is driven by battery cost reductions (down 89% since 2010), expanding charging infrastructure, and stricter emission regulations in over 20 countries. The market is projected to grow at 21% CAGR through 2030, reaching 40 million units annually. However, supply chain constraints for lithium and cobalt remain key challenges."

- primary_metrics (3+ metrics with numbers)
- key_findings (3+ findings with quantitative details)
- top_entities (3+ companies/countries with market share %)
- trends_forecast (2+ trends with timelines)
- visualization_data (MUST have chart_labels and chart_values)
- benchmark_table (if included, value_1 and value_2 must be NUMBERS, not "N/A")

Even if web data is sparse, use your knowledge to provide complete, detailed analysis.

Output ONLY this JSON structure:
{RESPONSE_TEMPLATE}
"""

EVOLUTION_PROMPT_TEMPLATE = """You are a market research analyst performing an UPDATE ANALYSIS.

You have been given a PREVIOUS ANALYSIS from {time_ago}. Your task is to:
1. Search for CURRENT data on the same metrics and entities
2. Identify what has CHANGED vs what has STAYED THE SAME
3. Provide updated values where data has changed
4. Flag any metrics/entities that are no longer relevant or have new entries

PREVIOUS ANALYSIS:
==================
Question: {previous_question}
Timestamp: {previous_timestamp}

Previous Executive Summary:
{previous_summary}

Previous Key Metrics:
{previous_metrics}

Previous Top Entities:
{previous_entities}

Previous Key Findings:
{previous_findings}
==================

CRITICAL RULES:
1. Return ONLY valid JSON. NO markdown, NO code blocks.
2. For EACH metric, indicate if it INCREASED, DECREASED, or stayed UNCHANGED
3. Keep the SAME metric names as previous analysis for easy comparison
4. If a metric is no longer available, mark it as "discontinued"
5. If there's a NEW important metric, add it with status "new"

REQUIRED OUTPUT FORMAT:
{{
  "executive_summary": "Updated 4-6 sentence summary noting key changes since last analysis",
  "analysis_delta": {{
    "time_since_previous": "{time_ago}",
    "overall_trend": "improving/declining/stable",
    "major_changes": ["Change 1", "Change 2"],
    "data_freshness": "Q4 2024"
  }},
  "primary_metrics": {{
    "metric_key": {{
      "name": "Same metric name as before",
      "previous_value": 100,
      "current_value": 110,
      "unit": "$B",
      "change_pct": 10.0,
      "direction": "increased/decreased/unchanged",
      "status": "updated/discontinued/new"
    }}
  }},
  "key_findings": [
    "[UNCHANGED] Finding that remains true",
    "[UPDATED] Finding with new data",
    "[NEW] Completely new finding",
    "[REMOVED] Finding no longer relevant - reason"
  ],
  "top_entities": [
    {{
      "name": "Company A",
      "previous_share": "25%",
      "current_share": "27%",
      "previous_rank": 1,
      "current_rank": 1,
      "change": "increased",
      "status": "updated"
    }}
  ],
  "trends_forecast": [
    {{"trend": "Trend description", "direction": "↑", "timeline": "2025-2027", "confidence": "high/medium/low"}}
  ],
  "visualization_data": {{
    "chart_labels": ["Previous", "Current"],
    "chart_values": [100, 110],
    "chart_title": "Market Size Evolution"
  }},
  "sources": ["source1.com", "source2.com"],
  "confidence": 85,
  "freshness": "Dec 2024",
  "drift_summary": {{
    "metrics_changed": 2,
    "metrics_unchanged": 3,
    "entities_reshuffled": 1,
    "findings_updated": 4,
    "overall_stability_pct": 75
  }}
}}

NOW, search for CURRENT information to UPDATE the previous analysis.
Focus on finding CHANGES to the metrics and entities listed above.

User Question: {query}
"""

# =========================================================
# 4. MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner="🔧 Loading AI models...")
def load_models():
    """Load and cache sentence transformer and classifier"""
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return classifier, embedder
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

domain_classifier, embedder = load_models()

# =========================================================
# 5. JSON REPAIR FUNCTIONS
# =========================================================

def repair_llm_response(data: dict) -> dict:
    """
    Repair common LLM JSON structure issues:
    - Convert primary_metrics from list to dict
    - Ensure top_entities and trends_forecast are lists
    - Fix benchmark_table numeric values
    - Add missing required fields
    """

    # Fix primary_metrics: list → dict
    if "primary_metrics" in data and isinstance(data["primary_metrics"], list):
        new_metrics = {}
        for i, item in enumerate(data["primary_metrics"]):
            if isinstance(item, dict):
                raw_name = item.get("name", f"metric_{i+1}")
                key = re.sub(r'[^a-z0-9_]', '', raw_name.lower().replace(" ", "_"))
                if not key:
                    key = f"metric_{i+1}"

                original_key = key
                j = 1
                while key in new_metrics:
                    key = f"{original_key}_{j}"
                    j += 1

                item.setdefault("name", raw_name)
                item.setdefault("value", "N/A")
                item.setdefault("unit", "")

                new_metrics[key] = item

        data["primary_metrics"] = new_metrics

    # Fix top_entities: ensure list
    if "top_entities" in data:
        if isinstance(data["top_entities"], dict):
            data["top_entities"] = list(data["top_entities"].values())
        elif not isinstance(data["top_entities"], list):
            data["top_entities"] = []

    # Fix trends_forecast: ensure list
    if "trends_forecast" in data:
        if isinstance(data["trends_forecast"], dict):
            data["trends_forecast"] = list(data["trends_forecast"].values())
        elif not isinstance(data["trends_forecast"], list):
            data["trends_forecast"] = []

    # Fix visualization_data: handle old 'labels'/'values' keys
    if "visualization_data" in data and isinstance(data["visualization_data"], dict):
        viz = data["visualization_data"]
        if "labels" in viz and "chart_labels" not in viz:
            viz["chart_labels"] = viz.pop("labels")
        if "values" in viz and "chart_values" not in viz:
            viz["chart_values"] = viz.pop("values")

    # Fix benchmark_table numeric values
    if "benchmark_table" in data and isinstance(data["benchmark_table"], list):
        cleaned_table = []
        for row in data["benchmark_table"]:
            if isinstance(row, dict):
                # Ensure category exists
                if "category" not in row:
                    row["category"] = "Unknown"

                # Fix numeric fields
                for key in ["value_1", "value_2"]:
                    if key not in row:
                        row[key] = 0
                        continue

                    val = row[key]

                    # Convert "N/A" and similar to 0
                    if isinstance(val, str):
                        val_upper = val.upper().strip()
                        if val_upper in ["N/A", "NA", "NULL", "NONE", "", "-", "—"]:
                            row[key] = 0
                        else:
                            # Try to parse numeric strings like "25.5", "$100", "1,234"
                            try:
                                cleaned = re.sub(r'[^\d.-]', '', val)
                                if cleaned:
                                    row[key] = float(cleaned) if '.' in cleaned else int(cleaned)
                                else:
                                    row[key] = 0
                            except (ValueError, TypeError):
                                row[key] = 0
                    elif not isinstance(val, (int, float)):
                        row[key] = 0

                cleaned_table.append(row)

        data["benchmark_table"] = cleaned_table

    return data

def validate_numeric_fields(data: dict, context: str = "LLM Response") -> None:
    """Log warnings for non-numeric values in expected numeric fields"""

    # Check benchmark_table
    if "benchmark_table" in data and isinstance(data["benchmark_table"], list):
        for i, row in enumerate(data["benchmark_table"]):
            if isinstance(row, dict):
                for key in ["value_1", "value_2"]:
                    val = row.get(key)
                    if isinstance(val, str):
                        st.warning(
                            f"⚠️ {context}: benchmark_table[{i}].{key} is string: '{val}' "
                            f"(will be converted to 0)"
                        )

def preclean_json(raw: str) -> str:
    """Remove markdown fences and citations before parsing"""
    if not raw or not isinstance(raw, str):
        return ""

    text = raw.strip()

    # Remove code fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # Remove citations
    text = re.sub(r'\[web:\d+\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)

    return text

def parse_json_safely(json_str: str, context: str = "LLM") -> dict:
    """
    Parse JSON with aggressive error recovery:
    1. Pre-clean
    2. Extract JSON object
    3. Fix common issues (trailing commas, unquoted keys)
    4. Iterative quote repair for unterminated strings
    """
    if not json_str:
        return {}

    cleaned = preclean_json(json_str)

    # Extract main JSON object
    match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if not match:
        st.warning(f"⚠️ No JSON object found in {context} response")
        return {}

    json_content = match.group(0)

    # Structural repairs
    try:
        # Fix unquoted keys: {key: → {"key":
        json_content = re.sub(r'([\{\,]\s*)([a-zA-Z_][a-zA-Z0-9_\-]+)(\s*):', r'\1"\2"\3:', json_content)

        # Remove trailing commas
        json_content = re.sub(r',\s*([\]\}])', r'\1', json_content)

        # Fix boolean/null capitalization
        json_content = json_content.replace(': True', ': true')
        json_content = json_content.replace(': False', ': false')
        json_content = json_content.replace(': Null', ': null')

    except Exception as e:
        st.warning(f"Regex repair failed: {e}")

    # Iterative parsing with quote repair
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            if "Unterminated string" not in e.msg:
                if attempt == 0:
                    st.warning(f"JSON parse error in {context}: {e.msg[:100]}")
                break

            # Find and escape unescaped quote near error position
            error_pos = e.pos
            found = False
            for i in range(error_pos - 1, max(0, error_pos - 150), -1):
                if i < len(json_content) and json_content[i] == '"':
                    if i == 0 or json_content[i-1] != '\\':
                        json_content = json_content[:i] + '\\"' + json_content[i+1:]
                        found = True
                        break

            if not found:
                break

    st.error(f"❌ Failed to parse JSON from {context} after {max_attempts} attempts")
    return {}

# =========================================================
# 6. WEB SEARCH FUNCTIONS
#   SERPAPI STABILITY CONFIGURATION
# =========================================================

# Fixed parameters to prevent geo/personalization variance

SERPAPI_STABILITY_CONFIG = {
    "gl": "us",                    # Fixed country
    "hl": "en",                    # Fixed language
    "google_domain": "google.com", # Fixed domain
    "nfpr": "1",                   # No auto-query correction
    "safe": "active",              # Consistent safe search
    "device": "desktop",           # Fixed device type
    "no_cache": "false",           # Allow Google caching (more stable)
}

# Preferred domains for consistent sourcing (sorted by priority)
PREFERRED_SOURCE_DOMAINS = [
    "statista.com", "reuters.com", "bloomberg.com", "imf.org", "wsj.com", "bcg.com", "opec.org",
    "worldbank.org", "mckinsey.com", "deloitte.com", "spglobal.com", "ft.com", "pwc.com", "semiconductors.org",
    "ft.com", "economist.com", "wsj.com", "forbes.com", "cnbc.com", "kpmg.com", "eia.org"
]

# Search results cache
_search_cache: Dict[str, Tuple[List[Dict], datetime]] = {}
SEARCH_CACHE_TTL_HOURS = 24

def get_search_cache_key(query: str) -> str:
    """Generate stable cache key for search query"""
    normalized = re.sub(r'\s+', ' ', query.lower().strip())
    normalized = re.sub(r'\b(today|current|latest|now|recent)\b', '', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()[:16]

def get_cached_search_results(query: str) -> Optional[List[Dict]]:
    """Get cached search results if still valid"""
    cache_key = get_search_cache_key(query)
    if cache_key in _search_cache:
        cached_results, cached_time = _search_cache[cache_key]
        if datetime.now() - cached_time < timedelta(hours=SEARCH_CACHE_TTL_HOURS):
            return cached_results
        del _search_cache[cache_key]
    return None

def cache_search_results(query: str, results: List[Dict]):
    """Cache search results"""
    cache_key = get_search_cache_key(query)
    _search_cache[cache_key] = (results, datetime.now())

# =========================================================
# LLM RESPONSE CACHE - Prevents variance on identical inputs
# =========================================================
_llm_cache: Dict[str, Tuple[str, datetime]] = {}
LLM_CACHE_TTL_HOURS = 24  # Cache LLM responses for 24 hours

def get_llm_cache_key(query: str, web_context: Dict) -> str:
    """Generate cache key from query + source URLs"""
    # Include source URLs so cache invalidates if sources change
    source_urls = sorted(web_context.get("sources", [])[:5])
    cache_input = f"{query.lower().strip()}|{'|'.join(source_urls)}"
    return hashlib.md5(cache_input.encode()).hexdigest()[:20]

def get_cached_llm_response(query: str, web_context: Dict) -> Optional[str]:
    """Get cached LLM response if still valid"""
    cache_key = get_llm_cache_key(query, web_context)
    if cache_key in _llm_cache:
        cached_response, cached_time = _llm_cache[cache_key]
        if datetime.now() - cached_time < timedelta(hours=LLM_CACHE_TTL_HOURS):
            return cached_response
        del _llm_cache[cache_key]
    return None

def cache_llm_response(query: str, web_context: Dict, response: str):
    """Cache LLM response"""
    cache_key = get_llm_cache_key(query, web_context)
    _llm_cache[cache_key] = (response, datetime.now())


def sort_results_deterministically(results: List[Dict]) -> List[Dict]:
    """Sort results for consistent ordering"""
    def sort_key(r):
        link = r.get("link", "").lower()
        # Priority: preferred domains first, then alphabetical
        priority = 999
        for i, domain in enumerate(PREFERRED_SOURCE_DOMAINS):
            if domain in link:
                priority = i
                break
        return (priority, link)
    return sorted(results, key=sort_key)


def classify_source_reliability(source: str) -> str:
    """Classify source as High/Medium/Low quality"""
    source = source.lower() if isinstance(source, str) else ""

    high = ["gov", "imf", "worldbank", "central bank", "fed", "ecb", "reuters", "spglobal", "economist", "mckinsey", "bcg", "cognitive market research",
            "financial times", "wsj", "oecd", "bloomberg", "tradingeconomics", "deloitte", "hsbc", "imarc", "booz allen", "bakerinstitute.org", "wef",
           "kpmg", "semiconductors.org", "eu", "iea", "world bank", "opec", "jpmorgan", "citibank", "goldmansachs", "j.p. morgan", "oecd",
           "world bank", "sec", "federalreserve", "bls", "bea"]
    medium = ["wikipedia", "forbes", "cnbc", "yahoo", "ceic", "statista", "trendforce", "digitimes", "idc", "gartner", "marketwatch", "fortune", "investopedia"]
    low = ["blog", "medium.com", "wordpress", "ad", "promo"]

    for h in high:
        if h in source:
            return "✅ High"
    for m in medium:
        if m in source:
            return "⚠️ Medium"
    for l in low:
        if l in source:
            return "❌ Low"

    return "⚠️ Medium"

def source_quality_score(sources: List[str]) -> float:
    """Calculate average source quality (0-100)"""
    if not sources:
        return 50.0  # Lower default when no sources

    weights = {"✅ High": 100, "⚠️ Medium": 60, "❌ Low": 30}
    scores = [weights.get(classify_source_reliability(s), 60) for s in sources]
    return sum(scores) / len(scores) if scores else 50.0

@st.cache_data(ttl=3600, show_spinner=False)
def search_serpapi(query: str, num_results: int = 10) -> List[Dict]:
    """Search Google via SerpAPI with stability controls"""
    if not SERPAPI_KEY:
        return []

    # Check cache first (this is the ONLY cache we use - removed @st.cache_data to avoid conflicts)
    cached = get_cached_search_results(query)
    if cached:
        st.info("📦 Using cached search results")
        return cached

    # Aggressive query normalization for consistent searches
    query_normalized = query.lower().strip()

    # Remove temporal words that cause variance
    query_normalized = re.sub(r'\b(latest|current|today|now|recent|new|upcoming|this year|this month)\b', '', query_normalized)

    # Normalize whitespace
    query_normalized = re.sub(r'\s+', ' ', query_normalized).strip()

    # Add year for consistency
    if not re.search(r'\b20\d{2}\b', query_normalized):
        query_normalized = f"{query_normalized} 2024"

    # Build search terms
    query_lower = query_normalized
    industry_kw = ["industry", "market", "sector", "size", "growth", "players"]

    if any(kw in query_lower for kw in industry_kw):
        search_terms = f"{query_normalized} market size growth statistics"
        tbm, tbs = "", ""  # Organic results (more stable than news)
    else:
        search_terms = f"{query_normalized} finance economics data"
        tbm, tbs = "", ""  # Use organic for stability

    params = {
        "engine": "google",
        "q": search_terms,
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "tbm": tbm,
        "tbs": tbs,
        **SERPAPI_STABILITY_CONFIG  # Add fixed location params
    }

    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []

        # Prefer organic results (more stable than news)
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "source": item.get("source", "")
            })

        # Fall back to news only if no organic results
        if not results:
            for item in data.get("news_results", [])[:num_results]:
                src = item.get("source", {})
                source_name = src.get("name", "") if isinstance(src, dict) else str(src)
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date": item.get("date", ""),
                    "source": source_name
                })

        # Sort deterministically
        results = sort_results_deterministically(results)
        results = results[:num_results]

        # Cache results
        if results:
            cache_search_results(query, results)

        return results

    except Exception as e:
        st.warning(f"⚠️ SerpAPI error: {e}")
        return []


def scrape_url(url: str) -> Optional[str]:
    """Scrape webpage content via ScrapingDog"""
    if not SCRAPINGDOG_KEY:
        return None

    params = {
        "api_key": SCRAPINGDOG_KEY,
        "url": url,
        "dynamic": "false"
    }

    try:
        resp = requests.get("https://api.scrapingdog.com/scrape", params=params, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)

        return clean_text[:3000]

    except Exception as e:
        st.warning(f"⚠️ Scraping error for {url[:50]}: {e}")
        return None

def fetch_web_context(query: str, num_sources: int = 3) -> Dict:
    """Search web and scrape top sources"""
    search_results = search_serpapi(query, num_results=10)

    # Show what SerpAPI has found
    source_counts = {
    "total": len(search_results),
    "high_quality": sum(1 for r in search_results if "✅" in classify_source_reliability(r.get("link", ""))),
    "used_for_scraping": min(num_sources, len(search_results))
    }
    st.info(f"🔍 SerpAPI: **{source_counts['total']} total** | **{source_counts['high_quality']} high-quality** | Scraping **{source_counts['used_for_scraping']}**")


    if not search_results:
        return {
            "search_results": [],
            "scraped_content": {},
            "summary": "",
            "sources": [],
            "source_reliability": []
        }

    # Scrape top sources
    scraped_content = {}
    if SCRAPINGDOG_KEY:
        progress = st.progress(0)
        st.info(f"🔍 Scraping top {num_sources} sources...")

        for i, result in enumerate(search_results[:num_sources]):
            url = result["link"]
            content = scrape_url(url)
            if content:
                scraped_content[url] = content
                st.success(f"✓ {i+1}/{num_sources}: {result['source']}")
            progress.progress((i + 1) / num_sources)

        progress.empty()

    # Build context summary
    context_parts = []
    reliabilities = []

    for r in search_results:
        date_str = f" ({r['date']})" if r['date'] else ""
        reliability = classify_source_reliability(r.get("link", "") + " " + r.get("source", ""))
        reliabilities.append(reliability)

        context_parts.append(
            f"**{r['title']}**{date_str}\n"
            f"Source: {r['source']} [{reliability}]\n"
            f"{r['snippet']}\n"
            f"URL: {r['link']}"
        )

    return {
        "search_results": search_results,
        "scraped_content": scraped_content,
        "summary": "\n\n---\n\n".join(context_parts),
        "sources": [r["link"] for r in search_results],
        "source_reliability": reliabilities
    }

# =========================================================
# 7. LLM QUERY FUNCTIONS
# =========================================================

def query_perplexity(query: str, web_context: Dict, temperature: float = 0.0) -> str:
    """Query Perplexity API with web context - deterministic settings"""

    # Check LLM cache first
    cached_response = get_cached_llm_response(query, web_context)
    if cached_response:
        st.info("📦 Using cached LLM response (identical sources)")
        return cached_response

    search_count = len(web_context.get("search_results", []))

    # Build enhanced prompt
    if not web_context.get("summary") or search_count < 2:
        enhanced_query = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User Question: {query}\n\n"
            f"Web search returned {search_count} results. "
            f"Use your knowledge to provide complete analysis with all required fields."
        )
    else:
        context_section = (
            "LATEST WEB RESEARCH:\n"
            f"{web_context['summary']}\n\n"
        )

        if web_context.get('scraped_content'):
            context_section += "\nDETAILED CONTENT:\n"
            for url, content in list(web_context['scraped_content'].items())[:2]:
                context_section += f"\n{url}:\n{content[:800]}...\n"

        enhanced_query = f"{context_section}\n{SYSTEM_PROMPT}\n\nUser Question: {query}"

    # API request
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "temperature": 0.0,      # DETERMINISTIC: No randomness
        "max_tokens": 2000,
        "top_p": 1.0,            # DETERMINISTIC: No nucleus sampling
        "messages": [{"role": "user", "content": enhanced_query}]
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data:
            raise Exception("No 'choices' in Perplexity response")

        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise Exception("Empty Perplexity response")

        # Parse and repair
        parsed = parse_json_safely(content, "Perplexity")
        if not parsed:
            return create_fallback_response(query, search_count, web_context)

        repaired = repair_llm_response(parsed)

        # Debug helper
        validate_numeric_fields(repaired, "Perplexity")

        # Validate with Pydantic
        try:
            llm_obj = LLMResponse.model_validate(repaired)

            # Merge web sources
            if web_context.get("sources"):
                existing = llm_obj.sources or []
                merged = list(dict.fromkeys(existing + web_context["sources"]))
                llm_obj.sources = merged[:10]
                llm_obj.freshness = "Current (web-enhanced)"

                result = llm_obj.model_dump_json()
                # Cache the successful response
                cache_llm_response(query, web_context, result)
                return result

        except ValidationError as e:
            st.warning(f"⚠️ Pydantic validation failed: {e}")
            return create_fallback_response(query, search_count, web_context)

    except Exception as e:
        st.error(f"❌ Perplexity API error: {e}")
        return create_fallback_response(query, search_count, web_context)


def create_fallback_response(query: str, search_count: int, web_context: Dict) -> str:
    """Create fallback response matching schema"""
    fallback = LLMResponse(
        executive_summary=f"Analysis of '{query}' completed with {search_count} web sources. Schema validation used fallback structure.",
        primary_metrics={
            "sources": MetricDetail(name="Web Sources", value=search_count, unit="sources"),
            "quality": MetricDetail(name="Data Quality", value=70, unit="%")
        },
        key_findings=[
            f"Web search found {search_count} relevant sources.",
            "Primary model output required fallback due to format issues.",
            "Manual review of raw data recommended for accuracy."
        ],
        top_entities=[
            TopEntityDetail(name="Source 1", share="N/A", growth="N/A")
        ],
        trends_forecast=[
            TrendForecastDetail(trend="Schema validation used fallback", direction="⚠️", timeline="Now")
        ],
        visualization_data=VisualizationData(
            chart_labels=["Attempt"],
            chart_values=[search_count],
            chart_title="Search Results"
        ),
        sources=web_context.get("sources", []),
        confidence=60,
        freshness="Current (fallback)"
    )

    return fallback.model_dump_json()

# =========================================================
# 7B. ANCHORED EVOLUTION QUERY
# =========================================================

def format_previous_metrics(metrics: Dict) -> str:
    """Format previous metrics for prompt"""
    if not metrics:
        return "No previous metrics available"

    lines = []
    for key, m in metrics.items():
        if isinstance(m, dict):
            lines.append(f"- {m.get('name', key)}: {m.get('value', 'N/A')} {m.get('unit', '')}")
    return "\n".join(lines) if lines else "No metrics"

def format_previous_entities(entities: List) -> str:
    """Format previous entities for prompt"""
    if not entities:
        return "No previous entities available"

    lines = []
    for i, e in enumerate(entities, 1):
        if isinstance(e, dict):
            lines.append(f"{i}. {e.get('name', 'Unknown')}: {e.get('share', 'N/A')} share, {e.get('growth', 'N/A')} growth")
    return "\n".join(lines) if lines else "No entities"

def format_previous_findings(findings: List) -> str:
    """Format previous findings for prompt"""
    if not findings:
        return "No previous findings available"

    lines = [f"- {f}" for f in findings if f]
    return "\n".join(lines) if lines else "No findings"

def calculate_time_ago(timestamp_str: str) -> str:
    """Calculate human-readable time difference"""
    try:
        prev_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        delta = datetime.now() - prev_time.replace(tzinfo=None)

        hours = delta.total_seconds() / 3600
        if hours < 24:
            return f"{hours:.1f} hours ago"
        elif hours < 168:  # 7 days
            return f"{hours/24:.1f} days ago"
        elif hours < 720:  # 30 days
            return f"{hours/168:.1f} weeks ago"
        else:
            return f"{hours/720:.1f} months ago"
    except:
        return "unknown time ago"

def query_perplexity_anchored(query: str, previous_data: Dict, web_context: Dict, temperature: float = 0.1) -> str:
    """
    Query Perplexity with previous analysis as anchor.
    This produces an evolution-aware response that tracks changes.
    """

    prev_response = previous_data.get("primary_response", {})
    prev_timestamp = previous_data.get("timestamp", "")
    prev_question = previous_data.get("question", query)

    time_ago = calculate_time_ago(prev_timestamp)

    # Build the anchored prompt
    anchored_prompt = EVOLUTION_PROMPT_TEMPLATE.format(
        time_ago=time_ago,
        previous_question=prev_question,
        previous_timestamp=prev_timestamp,
        previous_summary=prev_response.get("executive_summary", "No previous summary"),
        previous_metrics=format_previous_metrics(prev_response.get("primary_metrics", {})),
        previous_entities=format_previous_entities(prev_response.get("top_entities", [])),
        previous_findings=format_previous_findings(prev_response.get("key_findings", [])),
        query=query
    )

    # Add web context if available
    if web_context.get("summary"):
        anchored_prompt = f"CURRENT WEB RESEARCH:\n{web_context['summary']}\n\n{anchored_prompt}"

    # API request
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }


    payload = {
        "model": "sonar",
        "temperature": 0.0,      # DETERMINISTIC
        "max_tokens": 2500,
        "top_p": 1.0,            # DETERMINISTIC
        "messages": [{"role": "user", "content": anchored_prompt}]
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data:
            raise Exception("No choices in response")

        content = data["choices"][0]["message"]["content"]
        if not content:
            raise Exception("Empty response")

        # Parse JSON
        parsed = parse_json_safely(content, "Perplexity-Anchored")
        if not parsed:
            return create_anchored_fallback(query, previous_data, web_context)

        # Add sources from web context
        if web_context.get("sources"):
            existing = parsed.get("sources", [])
            parsed["sources"] = list(dict.fromkeys(existing + web_context["sources"]))[:10]

        return json.dumps(parsed)

    except Exception as e:
        st.error(f"❌ Anchored query error: {e}")
        return create_anchored_fallback(query, previous_data, web_context)

def create_anchored_fallback(query: str, previous_data: Dict, web_context: Dict) -> str:
    """Create fallback for anchored evolution query"""
    prev_response = previous_data.get("primary_response", {})

    fallback = {
        "executive_summary": f"Evolution analysis for '{query}' - model returned invalid format. Showing previous data.",
        "analysis_delta": {
            "time_since_previous": calculate_time_ago(previous_data.get("timestamp", "")),
            "overall_trend": "unknown",
            "major_changes": ["Unable to determine changes - API error"],
            "data_freshness": "Unknown"
        },
        "primary_metrics": prev_response.get("primary_metrics", {}),
        "key_findings": ["[UNCHANGED] " + f for f in prev_response.get("key_findings", [])[:3]],
        "top_entities": prev_response.get("top_entities", []),
        "trends_forecast": prev_response.get("trends_forecast", []),
        "sources": web_context.get("sources", []),
        "confidence": 50,
        "freshness": "Fallback",
        "drift_summary": {
            "metrics_changed": 0,
            "metrics_unchanged": len(prev_response.get("primary_metrics", {})),
            "entities_reshuffled": 0,
            "findings_updated": 0,
            "overall_stability_pct": 100
        }
    }
    return json.dumps(fallback)

# =========================================================
# 8. VALIDATION & SCORING
# =========================================================


def parse_number_with_unit(val_str: str) -> float:
    """Parse numbers like '58.3B', '$123M', '1,234' to base unit (millions)"""
    if not val_str:
        return 0.0

    # Clean and extract
    val_str = str(val_str).replace('$', '').replace(',', '').strip()

    # Check for unit suffix
    multiplier = 1.0  # Base = millions
    if val_str.endswith('B') or val_str.endswith('b'):
        multiplier = 1000.0  # Billions to millions
        val_str = val_str[:-1]
    elif val_str.endswith('M') or val_str.endswith('m'):
        multiplier = 1.0
        val_str = val_str[:-1]
    elif val_str.endswith('K') or val_str.endswith('k'):
        multiplier = 0.001  # Thousands to millions
        val_str = val_str[:-1]

    try:
        return float(val_str) * multiplier
    except (ValueError, TypeError):
        return 0.0

def numeric_consistency_with_sources(primary_data: dict, web_context: dict) -> float:
    """Compare primary numbers vs source numbers"""
    primary_metrics = primary_data.get("primary_metrics", {})
    primary_numbers = []

    for metric in primary_metrics.values():
        if isinstance(metric, dict):
            val = metric.get("value")
            num = parse_number_with_unit(str(val))
            if num > 0:
                primary_numbers.append(num)

    if not primary_numbers:
        return 50.0  # Neutral when no metrics to compare

    # Extract source numbers with same parsing
    source_numbers = []
    search_results = web_context.get("search_results", [])

    for result in search_results:
        snippet = str(result.get("snippet", ""))
        # Match patterns like "$58.3B", "123M", "456 billion"
        patterns = [
            r'\$?(\d+(?:\.\d+)?)\s*([BbMmKk])',  # $58.3B
            r'(\d+(?:\.\d+)?)\s*(billion|million|thousand)',  # 58.3 billion
        ]

        for pattern in patterns:
            matches = re.findall(pattern, snippet, re.IGNORECASE)
            for num, unit in matches:
                source_numbers.append(parse_number_with_unit(f"{num}{unit[0].upper()}"))

    if not source_numbers:
        return 50.0  # Neutral when no source numbers found

    # Check agreement (within 25% tolerance)
    agreements = 0
    for p_num in primary_numbers:
        for s_num in source_numbers:
            if abs(p_num - s_num) / max(p_num, s_num, 1) < 0.25:
                agreements += 1
                break

    # Scale: 0 agreements = 30%, all agreements = 95%
    agreement_ratio = agreements / len(primary_numbers)
    agreement_pct = 30.0 + (agreement_ratio * 65.0)
    return min(agreement_pct, 95.0)

def source_consensus(web_context: dict) -> float:
    """
    Calculate source consensus based on proportion of high-quality sources.
    Returns continuous score 0-100 based on quality distribution.
    """
    reliabilities = web_context.get("source_reliability", [])

    if not reliabilities:
        return 50.0  # Neutral when no sources

    total = len(reliabilities)
    high_count = sum(1 for r in reliabilities if "✅" in str(r))
    medium_count = sum(1 for r in reliabilities if "⚠️" in str(r))
    low_count = sum(1 for r in reliabilities if "❌" in str(r))

    # Weighted score: High=100, Medium=60, Low=30
    weighted_sum = (high_count * 100) + (medium_count * 60) + (low_count * 30)
    consensus_score = weighted_sum / total

    # Bonus for having multiple high-quality sources
    if high_count >= 3:
        consensus_score = min(100, consensus_score + 10)
    elif high_count >= 2:
        consensus_score = min(100, consensus_score + 5)

    return round(consensus_score, 1)

def evidence_based_veracity(primary_data: dict, web_context: dict) -> dict:
    """
    Evidence-driven veracity scoring.
    Returns breakdown of component scores and overall score (0-100).
    """
    breakdown = {}

    # 1. SOURCE QUALITY (35% weight)
    sources = primary_data.get("sources", [])
    src_score = source_quality_score(sources)
    breakdown["source_quality"] = round(src_score, 1)

    # 2. NUMERIC CONSISTENCY (30% weight)
    num_score = numeric_consistency_with_sources(primary_data, web_context)
    breakdown["numeric_consistency"] = round(num_score, 1)

    # 3. CITATION DENSITY (20% weight)
    # FIXED: Higher score when sources support findings, not penalize detail
    sources_count = len(sources)
    findings_count = len(primary_data.get("key_findings", []))
    metrics_count = len(primary_data.get("primary_metrics", {}))

    # Total claims = findings + metrics
    total_claims = findings_count + metrics_count

    if total_claims == 0:
        citations_score = 40.0  # Low score for no claims
    else:
        # Ratio of sources to claims - ideal is ~0.5-1.0 sources per claim
        ratio = sources_count / total_claims
        if ratio >= 1.0:
            citations_score = 90.0  # Well-supported
        elif ratio >= 0.5:
            citations_score = 70.0 + (ratio - 0.5) * 40  # 70-90 range
        elif ratio >= 0.25:
            citations_score = 50.0 + (ratio - 0.25) * 80  # 50-70 range
        else:
            citations_score = ratio * 200  # 0-50 range

    breakdown["citation_density"] = round(min(citations_score, 95.0), 1)

    # 4. SOURCE CONSENSUS (15% weight)
    consensus_score = source_consensus(web_context)
    breakdown["source_consensus"] = round(consensus_score, 1)

    # Calculate weighted total
    total_score = (
        breakdown["source_quality"] * 0.35 +
        breakdown["numeric_consistency"] * 0.30 +
        breakdown["citation_density"] * 0.20 +
        breakdown["source_consensus"] * 0.15
    )

    breakdown["overall"] = round(total_score, 1)

    return breakdown

def calculate_final_confidence(
    base_conf: float,
    evidence_score: float
) -> float:
    """
    Calculate final confidence score.

    Formula balances model confidence with evidence quality:
    - Evidence has higher weight (65%) as it's more objective
    - Model confidence (35%) is adjusted by evidence quality

    This ensures:
    - High model + High evidence → High final (~85-90%)
    - High model + Low evidence → Medium final (~55-65%)
    - Low model + High evidence → Medium-High final (~70-80%)
    - Low model + Low evidence → Low final (~40-50%)
    """

    # Normalize inputs to 0-100 range
    base_conf = max(0, min(100, base_conf))
    evidence_score = max(0, min(100, evidence_score))

    # 1. EVIDENCE COMPONENT (65% weight) - Primary driver
    evidence_component = evidence_score * 0.65

    # 2. MODEL COMPONENT (35% weight) - Adjusted by evidence quality
    # When evidence is weak, model confidence is discounted
    evidence_multiplier = 0.5 + (evidence_score / 200)  # Range: 0.5 to 1.0
    model_component = base_conf * evidence_multiplier * 0.35

    final = evidence_component + model_component

    # Ensure result is in valid range
    return round(max(0, min(100, final)), 1)

# =========================================================
# 8A. EVOLUTION LAYER - TRACK CHANGES OVER TIME
# =========================================================

def parse_numeric_for_comparison(value: Any) -> Optional[float]:
    """Parse any value to float for comparison"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r'[,$%]', '', value.strip())
        multiplier = 1.0
        if 'trillion' in cleaned.lower() or cleaned.lower().endswith('t'):
            multiplier = 1000000
            cleaned = re.sub(r'[tT](?:rillion)?', '', cleaned)
        elif 'billion' in cleaned.lower() or cleaned.lower().endswith('b'):
            multiplier = 1000
            cleaned = re.sub(r'[bB](?:illion)?', '', cleaned)
        elif 'million' in cleaned.lower() or cleaned.lower().endswith('m'):
            multiplier = 1
            cleaned = re.sub(r'[mM](?:illion)?', '', cleaned)
        try:
            return float(cleaned.strip()) * multiplier
        except:
            return None
    return None

def calculate_percent_change(old_val: Any, new_val: Any) -> Optional[float]:
    """Calculate percent change between two values"""
    old_num = parse_numeric_for_comparison(old_val)
    new_num = parse_numeric_for_comparison(new_val)

    if old_num is None or new_num is None:
        return None
    if old_num == 0:
        return None if new_num == 0 else 100.0

    return ((new_num - old_num) / abs(old_num)) * 100

def normalize_metric_name(name: str) -> str:
    """Normalize metric name for matching"""
    if not name:
        return ""
    norm = re.sub(r'[^\w\s]', '', name.lower().strip())
    norm = re.sub(r'\s+', ' ', norm)
    return norm

def fuzzy_match_names(name1: str, name2: str, threshold: float = 0.7) -> bool:
    """Check if two names match using fuzzy matching"""
    n1 = normalize_metric_name(name1)
    n2 = normalize_metric_name(name2)

    if n1 == n2:
        return True

    # Check if one contains the other
    if n1 in n2 or n2 in n1:
        return True

    # Fuzzy ratio
    ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
    return ratio >= threshold

def compare_metrics(old_metrics: Dict, new_metrics: Dict) -> List[Dict]:
    """Compare metrics between old and new responses"""
    changes = []

    # Track which new metrics have been matched
    matched_new_keys = set()

    # Compare old metrics to new
    for old_key, old_m in old_metrics.items():
        if not isinstance(old_m, dict):
            continue

        old_name = old_m.get("name", old_key)
        old_val = old_m.get("value")
        old_unit = old_m.get("unit", "")

        # Find matching new metric
        matched = False
        for new_key, new_m in new_metrics.items():
            if new_key in matched_new_keys:
                continue
            if not isinstance(new_m, dict):
                continue

            new_name = new_m.get("name", new_key)

            if fuzzy_match_names(old_name, new_name):
                matched_new_keys.add(new_key)
                matched = True

                new_val = new_m.get("value")
                new_unit = new_m.get("unit", "")
                pct_change = calculate_percent_change(old_val, new_val)

                # Determine direction
                if pct_change is None:
                    direction = "unchanged"
                elif abs(pct_change) < 1:
                    direction = "unchanged"
                elif pct_change > 0:
                    direction = "increased"
                else:
                    direction = "decreased"

                changes.append({
                    "name": new_name,
                    "old_value": f"{old_val} {old_unit}".strip(),
                    "new_value": f"{new_val} {new_unit}".strip(),
                    "change_pct": pct_change,
                    "direction": direction,
                    "status": "updated"
                })
                break

        if not matched:
            changes.append({
                "name": old_name,
                "old_value": f"{old_val} {old_unit}".strip(),
                "new_value": "N/A",
                "change_pct": None,
                "direction": "removed",
                "status": "removed"
            })

    # Find new metrics that weren't in old
    for new_key, new_m in new_metrics.items():
        if new_key in matched_new_keys:
            continue
        if not isinstance(new_m, dict):
            continue

        new_name = new_m.get("name", new_key)
        new_val = new_m.get("value")
        new_unit = new_m.get("unit", "")

        changes.append({
            "name": new_name,
            "old_value": "N/A",
            "new_value": f"{new_val} {new_unit}".strip(),
            "change_pct": None,
            "direction": "new",
            "status": "new"
        })

    return changes

def compare_entities(old_entities: List, new_entities: List) -> List[Dict]:
    """Compare top entities between old and new responses"""
    changes = []

    # Build lookup for old entities
    old_lookup = {}
    for i, e in enumerate(old_entities):
        if isinstance(e, dict):
            name = e.get("name", "").lower().strip()
            old_lookup[name] = {"rank": i + 1, "share": e.get("share"), "growth": e.get("growth")}

    # Build lookup for new entities
    new_lookup = {}
    for i, e in enumerate(new_entities):
        if isinstance(e, dict):
            name = e.get("name", "").lower().strip()
            new_lookup[name] = {"rank": i + 1, "share": e.get("share"), "growth": e.get("growth"), "original_name": e.get("name")}

    # Compare
    all_names = set(old_lookup.keys()) | set(new_lookup.keys())

    for name in all_names:
        old_data = old_lookup.get(name)
        new_data = new_lookup.get(name)

        if old_data and new_data:
            rank_change = old_data["rank"] - new_data["rank"]  # Positive = moved up
            share_change = calculate_percent_change(old_data["share"], new_data["share"])

            if rank_change > 0:
                direction = "moved_up"
            elif rank_change < 0:
                direction = "moved_down"
            else:
                direction = "unchanged"

            changes.append({
                "name": new_data.get("original_name", name),
                "old_rank": old_data["rank"],
                "new_rank": new_data["rank"],
                "rank_change": rank_change,
                "old_share": old_data["share"],
                "new_share": new_data["share"],
                "share_change_pct": share_change,
                "direction": direction,
                "status": "updated"
            })
        elif old_data:
            changes.append({
                "name": name,
                "old_rank": old_data["rank"],
                "new_rank": None,
                "rank_change": None,
                "old_share": old_data["share"],
                "new_share": None,
                "share_change_pct": None,
                "direction": "removed",
                "status": "removed"
            })
        elif new_data:
            changes.append({
                "name": new_data.get("original_name", name),
                "old_rank": None,
                "new_rank": new_data["rank"],
                "rank_change": None,
                "old_share": None,
                "new_share": new_data["share"],
                "share_change_pct": None,
                "direction": "new",
                "status": "new"
            })

    # Sort by new rank
    changes.sort(key=lambda x: x.get("new_rank") or 999)
    return changes

def compare_findings(old_findings: List[str], new_findings: List[str]) -> List[Dict]:
    """Compare key findings using semantic similarity"""
    changes = []
    matched_new = set()

    for old_f in old_findings:
        if not old_f:
            continue

        best_match = None
        best_similarity = 0

        for i, new_f in enumerate(new_findings):
            if i in matched_new or not new_f:
                continue

            # Simple word overlap similarity
            old_words = set(normalize_metric_name(old_f).split())
            new_words = set(normalize_metric_name(new_f).split())

            if old_words and new_words:
                overlap = len(old_words & new_words) / len(old_words | new_words)
                if overlap > best_similarity:
                    best_similarity = overlap
                    best_match = (i, new_f)

        if best_match and best_similarity > 0.5:
            matched_new.add(best_match[0])
            changes.append({
                "old_finding": old_f,
                "new_finding": best_match[1],
                "similarity": round(best_similarity * 100, 1),
                "status": "retained" if best_similarity > 0.8 else "modified"
            })
        else:
            changes.append({
                "old_finding": old_f,
                "new_finding": None,
                "similarity": 0,
                "status": "removed"
            })

    # New findings
    for i, new_f in enumerate(new_findings):
        if i not in matched_new and new_f:
            changes.append({
                "old_finding": None,
                "new_finding": new_f,
                "similarity": 0,
                "status": "new"
            })

    return changes

def calculate_stability_score(metric_changes: List[Dict], entity_changes: List[Dict], finding_changes: List[Dict]) -> Dict:
    """Calculate overall stability score"""

    # Metrics stability (40% weight)
    if metric_changes:
        unchanged_metrics = sum(1 for m in metric_changes if m.get("direction") == "unchanged")
        small_change_metrics = sum(1 for m in metric_changes if m.get("change_pct") is not None and abs(m.get("change_pct", 0)) < 10)
        metrics_stability = ((unchanged_metrics + small_change_metrics * 0.5) / len(metric_changes)) * 100
    else:
        metrics_stability = 100

    # Entity stability (30% weight)
    if entity_changes:
        unchanged_entities = sum(1 for e in entity_changes if e.get("direction") == "unchanged")
        entities_stability = (unchanged_entities / len(entity_changes)) * 100
    else:
        entities_stability = 100

    # Findings stability (30% weight)
    if finding_changes:
        retained_findings = sum(1 for f in finding_changes if f.get("status") in ["retained", "modified"])
        findings_stability = (retained_findings / len(finding_changes)) * 100
    else:
        findings_stability = 100

    overall = (metrics_stability * 0.4 + entities_stability * 0.3 + findings_stability * 0.3)

    return {
        "metrics_stability": round(metrics_stability, 1),
        "entities_stability": round(entities_stability, 1),
        "findings_stability": round(findings_stability, 1),
        "overall_stability": round(overall, 1)
    }

def analyze_evolution(old_data: Dict, new_data: Dict) -> Dict:
    """Analyze evolution between two analysis snapshots"""

    # Extract responses
    old_response = old_data.get("primary_response", {})
    new_response = new_data.get("primary_response", {})

    # Calculate time delta
    old_time = old_data.get("timestamp", "")
    new_time = new_data.get("timestamp", "")

    try:
        old_dt = datetime.fromisoformat(old_time.replace("Z", "+00:00")) if old_time else None
        new_dt = datetime.fromisoformat(new_time.replace("Z", "+00:00")) if new_time else None
        time_delta_hours = (new_dt - old_dt).total_seconds() / 3600 if old_dt and new_dt else None
    except:
        time_delta_hours = None

    # Compare components
    metric_changes = compare_metrics(
        old_response.get("primary_metrics", {}),
        new_response.get("primary_metrics", {})
    )

    entity_changes = compare_entities(
        old_response.get("top_entities", []),
        new_response.get("top_entities", [])
    )

    finding_changes = compare_findings(
        old_response.get("key_findings", []),
        new_response.get("key_findings", [])
    )

    # Calculate stability
    stability = calculate_stability_score(metric_changes, entity_changes, finding_changes)

    # Confidence evolution
    old_conf = old_data.get("final_confidence", 0)
    new_conf = new_data.get("final_confidence", 0)
    conf_change = new_conf - old_conf

    return {
        "old_timestamp": old_time,
        "new_timestamp": new_time,
        "time_delta_hours": round(time_delta_hours, 1) if time_delta_hours else None,
        "metric_changes": metric_changes,
        "entity_changes": entity_changes,
        "finding_changes": finding_changes,
        "stability": stability,
        "confidence_change": {
            "old": old_conf,
            "new": new_conf,
            "change": round(conf_change, 1)
        }
    }

# =========================================================
# 8B. EVOLUTION DASHBOARD RENDERING
# =========================================================

def render_evolution_results(evolution_data: Dict, query: str):
    """Render the anchored evolution analysis results"""

    st.header("📈 Evolution Analysis Results")
    st.markdown(f"**Query:** {query}")

    # Parse the evolution response
    try:
        if isinstance(evolution_data, str):
            data = json.loads(evolution_data)
        else:
            data = evolution_data
    except:
        st.error("Failed to parse evolution data")
        return

    # Delta summary
    delta = data.get("analysis_delta", {})
    drift = data.get("drift_summary", {})

    st.subheader("📊 Change Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Time Since Previous", delta.get("time_since_previous", "Unknown"))
    col2.metric("Overall Trend", delta.get("overall_trend", "Unknown").title())
    col3.metric("Stability Score", f"{drift.get('overall_stability_pct', 0)}%")

    # Trend indicator
    trend = delta.get("overall_trend", "stable").lower()
    if trend == "improving":
        col4.success("📈 Improving")
    elif trend == "declining":
        col4.error("📉 Declining")
    else:
        col4.info("➡️ Stable")

    # Major changes
    major_changes = delta.get("major_changes", [])
    if major_changes:
        st.markdown("**🔔 Major Changes:**")
        for change in major_changes:
            st.markdown(f"- {change}")

    st.markdown("---")

    # Executive Summary
    st.subheader("📋 Updated Executive Summary")
    st.markdown(f"**{data.get('executive_summary', 'No summary')}**")

    st.markdown("---")

    # Metrics with change indicators
    st.subheader("💰 Metrics Evolution")
    metrics = data.get("primary_metrics", {})

    if metrics:
        metric_rows = []
        for key, m in metrics.items():
            if isinstance(m, dict):
                direction = m.get("direction", "unchanged")
                status = m.get("status", "updated")

                # Icons
                if direction == "increased":
                    icon = "📈"
                elif direction == "decreased":
                    icon = "📉"
                elif status == "new":
                    icon = "🆕"
                elif status == "discontinued":
                    icon = "❌"
                else:
                    icon = "➡️"

                change_pct = m.get("change_pct")
                change_str = f"{change_pct:+.1f}%" if change_pct is not None else "-"

                metric_rows.append({
                    "": icon,
                    "Metric": m.get("name", key),
                    "Previous": f"{m.get('previous_value', 'N/A')} {m.get('unit', '')}".strip(),
                    "Current": f"{m.get('current_value', m.get('value', 'N/A'))} {m.get('unit', '')}".strip(),
                    "Change": change_str,
                    "Status": status.title()
                })

        if metric_rows:
            st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No metrics data")

    st.markdown("---")

    # Entities with ranking changes
    st.subheader("🏢 Entity Rankings Evolution")
    entities = data.get("top_entities", [])

    if entities:
        entity_rows = []
        for e in entities:
            if isinstance(e, dict):
                change = e.get("change", "unchanged")
                status = e.get("status", "updated")

                if change == "increased" or e.get("current_rank", 99) < e.get("previous_rank", 99):
                    icon = "⬆️"
                elif change == "decreased" or e.get("current_rank", 0) > e.get("previous_rank", 0):
                    icon = "⬇️"
                elif status == "new":
                    icon = "🆕"
                else:
                    icon = "➡️"

                entity_rows.append({
                    "": icon,
                    "Entity": e.get("name", "Unknown"),
                    "Prev Rank": e.get("previous_rank", "-"),
                    "Curr Rank": e.get("current_rank", "-"),
                    "Prev Share": e.get("previous_share", "-"),
                    "Curr Share": e.get("current_share", e.get("share", "-")),
                    "Status": status.title()
                })

        if entity_rows:
            st.dataframe(pd.DataFrame(entity_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No entity data")

    st.markdown("---")

    # Findings with change tags
    st.subheader("🔍 Key Findings Evolution")
    findings = data.get("key_findings", [])

    if findings:
        for finding in findings:
            if finding.startswith("[UNCHANGED]"):
                st.markdown(f"➡️ {finding.replace('[UNCHANGED]', '').strip()}")
            elif finding.startswith("[UPDATED]"):
                st.markdown(f"✏️ **{finding.replace('[UPDATED]', '').strip()}**")
            elif finding.startswith("[NEW]"):
                st.success(f"🆕 {finding.replace('[NEW]', '').strip()}")
            elif finding.startswith("[REMOVED]"):
                st.warning(f"❌ ~~{finding.replace('[REMOVED]', '').strip()}~~")
            else:
                st.markdown(f"- {finding}")
    else:
        st.info("No findings data")

    st.markdown("---")

    # Drift Summary
    st.subheader("📉 Drift Summary")

    drift_cols = st.columns(4)
    drift_cols[0].metric("Metrics Changed", drift.get("metrics_changed", 0))
    drift_cols[1].metric("Metrics Unchanged", drift.get("metrics_unchanged", 0))
    drift_cols[2].metric("Entities Reshuffled", drift.get("entities_reshuffled", 0))
    drift_cols[3].metric("Findings Updated", drift.get("findings_updated", 0))

    # Stability gauge
    stability = drift.get("overall_stability_pct", 0)
    if stability >= 80:
        st.success(f"🟢 **High Stability ({stability}%)** - Data is consistent with previous analysis")
    elif stability >= 60:
        st.warning(f"🟡 **Moderate Stability ({stability}%)** - Some changes detected")
    else:
        st.error(f"🔴 **Low Stability ({stability}%)** - Significant drift from previous analysis")

    st.markdown("---")

    # Sources
    st.subheader("🔗 Sources")
    sources = data.get("sources", [])
    if sources:
        cols = st.columns(2)
        for i, src in enumerate(sources[:8], 1):
            cols[(i-1) % 2].markdown(f"{i}. [{src[:50]}...]({src})")

# =========================================================
# 9. DASHBOARD RENDERING
# =========================================================

def detect_x_label_dynamic(labels: list) -> str:
    """Enhanced X-axis detection with better region matching"""
    if not labels:
        return "Category"

    # Convert to lowercase for comparison
    label_texts = [str(l).lower().strip() for l in labels]
    all_text = ' '.join(label_texts)

    # 1. GEOGRAPHIC REGIONS (PRIORITY 1)
    region_keywords = [
        'north america', 'asia pacific', 'asia-pacific', 'apac', 'europe', 'emea',
        'latin america', 'latam', 'middle east', 'africa', 'oceania',
        'rest of world', 'row', 'china', 'usa', 'india', 'japan', 'germany'
    ]

    # Count how many labels contain region keywords
    region_matches = sum(
        1 for label in label_texts
        if any(keyword in label for keyword in region_keywords)
    )

    # If 40%+ of labels are regions → "Regions"
    if region_matches / len(labels) >= 0.4:
        return "Regions"

    # 2. YEARS (e.g., 2023, 2024, 2025)
    year_pattern = r'\b(19|20)\d{2}\b'
    year_count = sum(1 for label in label_texts if re.search(year_pattern, label))
    if year_count / len(labels) > 0.5:
        return "Years"

    # 3. QUARTERS (Q1, Q2, Q3, Q4)
    quarter_pattern = r'\bq[1-4]\b'
    quarter_count = sum(1 for label in label_texts if re.search(quarter_pattern, label, re.IGNORECASE))
    if quarter_count >= 2:
        return "Quarters"

    # 4. MONTHS
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_count = sum(1 for label in label_texts if any(month in label for month in months))
    if month_count >= 3:
        return "Months"

    # 5. COMPANIES (common suffixes)
    company_keywords = ['inc', 'corp', 'ltd', 'llc', 'gmbh', 'ag', 'sa', 'plc']
    company_count = sum(1 for label in label_texts if any(kw in label for kw in company_keywords))
    if company_count >= 2:
        return "Companies"

    # 6. PRODUCTS/SEGMENTS (if contains "segment", "product", "category")
    if any(word in all_text for word in ['segment', 'product line', 'category', 'type']):
        return "Segments"

    # Default
    return "Categories"

def detect_y_label_dynamic(values: list) -> str:
    """Fully dynamic Y-axis label based on magnitude + context"""
    if not values:
        return "Value"

    numeric_values = []
    for v in values:
        try:
            numeric_values.append(abs(float(v)))
        except (ValueError, TypeError):
            continue

    if not numeric_values:
        return "Value"

    avg_mag = np.mean(numeric_values)
    max_mag = max(numeric_values)

    # Non-overlapping ranges with clear boundaries
    # 1. BILLIONS (large market sizes)
    if max_mag > 100 or avg_mag > 50:
        return "USD B"

    # 2. MILLIONS (medium values)
    elif max_mag > 10 or avg_mag > 5:
        return "USD M"

    # 3. PERCENTAGES (typical 0-100 range, but also small decimals)
    elif max_mag <= 100 and avg_mag <= 50:
        # Check if values look like percentages (mostly 0-100)
        if all(0 <= v <= 100 for v in numeric_values):
            return "Percent %"
        else:
            return "USD K"

    # 4. Default
    else:
        return "Units"

def render_dashboard(
    primary_json: str,
    final_conf: float,
    web_context: Dict,
    base_conf: float,
    user_question: str,
    veracity_scores: Optional[Dict] = None,
    source_reliability: Optional[List[str]] = None,
):
    """Render the analysis dashboard"""

    # Parse primary response
    try:
        data = json.loads(primary_json)
    except Exception as e:
        st.error(f"❌ Cannot render dashboard: {e}")
        st.code(primary_json[:1000])
        return

    # Header
    st.header("📊 Yureeka Market Report")
    st.markdown(f"**Question:** {user_question}")

    # Confidence row
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Confidence", f"{final_conf:.1f}%")
    col2.metric("Base Model", f"{base_conf:.1f}%")
    if veracity_scores:
        col3.metric("Evidence", f"{veracity_scores.get('overall', 0):.1f}%")
    else:
        col3.metric("Evidence", "N/A")

    st.markdown("---")

    # Executive Summary
    st.subheader("📋 Executive Summary")
    st.markdown(f"**{data.get('executive_summary', 'No summary available')}**")

    st.markdown("---")

    # Key Metrics
    st.subheader("💰 Key Metrics")
    metrics = data.get('primary_metrics', {})

    if metrics and isinstance(metrics, dict):
        metric_rows = []
        for key, detail in list(metrics.items())[:6]:
            if isinstance(detail, dict):
                metric_rows.append({
                    "Metric": detail.get("name", key),
                    "Value": f"{detail.get('value', 'N/A')} {detail.get('unit', '')}".strip()
                })

        if metric_rows:
            st.table(pd.DataFrame(metric_rows))
    else:
        st.info("No metrics available")

    st.markdown("---")

    # Key Findings
    st.subheader("🔍 Key Findings")
    findings = data.get('key_findings', [])
    for i, finding in enumerate(findings[:8], 1):
        if finding:
            st.markdown(f"**{i}.** {finding}")

    st.markdown("---")

    # Top Entities
    entities = data.get('top_entities', [])
    if entities:
        st.subheader("🏢 Top Market Players")
        entity_data = []
        for ent in entities:
            if isinstance(ent, dict):
                entity_data.append({
                    "Entity": ent.get("name", "N/A"),
                    "Share": ent.get("share", "N/A"),
                    "Growth": ent.get("growth", "N/A")
                })

        if entity_data:
            st.dataframe(pd.DataFrame(entity_data), hide_index=True, use_container_width=True)

    # Trends Forecast
    trends = data.get('trends_forecast', [])
    if trends:
        st.subheader("📈 Trends & Forecast")
        trend_data = []
        for trend in trends:
            if isinstance(trend, dict):
                trend_data.append({
                    "Trend": trend.get("trend", "N/A"),
                    "Direction": trend.get("direction", "→"),
                    "Timeline": trend.get("timeline", "N/A")
                })

        if trend_data:
            st.table(pd.DataFrame(trend_data))

    st.markdown("---")

    # Visualization - FIXED indentation
    st.subheader("📊 Data Visualization")
    viz = data.get('visualization_data')

    if viz and isinstance(viz, dict):
        labels = viz.get("chart_labels", [])
        values = viz.get("chart_values", [])
        title = viz.get("chart_title", "Trend Analysis")
        chart_type = viz.get("chart_type", "line")

        if labels and values and len(labels) == len(values):
            try:
                numeric_values = [float(v) for v in values[:10]]

                # Detect axis labels
                x_label = viz.get("x_axis_label") or detect_x_label_dynamic(labels)
                y_label = viz.get("y_axis_label") or detect_y_label_dynamic(numeric_values)

                # Create DataFrame
                df_viz = pd.DataFrame({
                    "x": labels[:10],
                    "y": numeric_values
                })

                # Create chart
                if chart_type == "bar":
                    fig = px.bar(df_viz, x="x", y="y", title=title)
                else:
                    fig = px.line(df_viz, x="x", y="y", title=title, markers=True)

                # Fix axis labels
                fig.update_layout(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    title_font_size=16,
                    font=dict(size=12),
                    xaxis=dict(tickangle=-45) if len(labels) > 5 else {}
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.info(f"⚠️ Chart rendering failed: {e}")
        else:
            st.info("📊 Visualization data incomplete or missing")
    else:
        st.info("📊 No visualization data available")

    # Comparison Bars
    comp = data.get('comparison_bars')
    if comp and isinstance(comp, dict):
        cats = comp.get("categories", [])
        vals = comp.get("values", [])
        if cats and vals and len(cats) == len(vals):
            try:
                df_comp = pd.DataFrame({"Category": cats, "Value": vals})
                fig = px.bar(df_comp, x="Category", y="Value",
                           title=comp.get("title", "Comparison"), text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass

    st.markdown("---")

    # Sources - COMPACT VERSION
    st.subheader("🔗 Sources & Reliability")
    all_sources = data.get('sources', []) or web_context.get('sources', [])

    if not all_sources:
        st.info("No sources found")
    else:
        st.success(f"📊 Found {len(all_sources)} sources")

    # Compact display - 2 columns, short URLs
    cols = st.columns(2)
    for i, src in enumerate(all_sources[:10], 1):
        col = cols[(i-1) % 2]
        short_url = src[:60] + "..." if len(src) > 60 else src
        reliability = classify_source_reliability(str(src))

        col.markdown(f"**{i}.** [{short_url}]({src})<br><small>{reliability}</small>",
                    unsafe_allow_html=True)

    # Metadata
    col_fresh, col_action = st.columns(2)
    with col_fresh:
        freshness = data.get('freshness', 'Current')
        st.metric("Data Freshness", freshness)

    st.markdown("---")

    # Veracity Scores - EVIDENCE-BASED
    if veracity_scores:
        st.subheader("✅ Evidence Quality Scores")
        cols = st.columns(5)
        metrics_display = [
            ("Sources", "source_quality"),
            ("Numbers", "numeric_consistency"),
            ("Citations", "citation_density"),
            ("Consensus", "source_consensus"),
            ("Overall", "overall")
        ]
        for i, (label, key) in enumerate(metrics_display):
            cols[i].metric(label, f"{veracity_scores.get(key, 0):.0f}%")

    # Web Context
    if web_context and web_context.get("search_results"):
        with st.expander("🌐 Web Search Details"):
            for i, result in enumerate(web_context["search_results"][:5]):
                st.markdown(f"**{i+1}. {result.get('title')}**")
                st.caption(f"{result.get('source')} - {result.get('date')}")
                st.write(result.get('snippet', ''))
                st.caption(f"[{result.get('link')}]({result.get('link')})")
                st.markdown("---")

# =========================================================
# 10. MAIN APPLICATION
# =========================================================

def main():
    st.set_page_config(
        page_title="Yureeka Market Report",
        page_icon="💹",
        layout="wide"
    )

    st.title("💹 Yureeka Market Intelligence")

    # Info section
    col_info, col_status = st.columns([3, 1])
    with col_info:
        st.markdown("""
        **Yureeka** provides AI-powered market research and analysis for finance,
        economics, and business questions.
        Powered by evidence-based verification and real-time web search.

        *Currently in prototype stage.*
        """)

    # Create tabs
    tab1, tab2 = st.tabs(["🔍 New Analysis", "📈 Evolution Analysis"])

    # =====================
    # TAB 1: NEW ANALYSIS
    # =====================
    with tab1:

    # User input
        query = st.text_input(
            "Enter your question about markets, industries, finance, or economics:",
            placeholder="e.g., What is the size of the global EV battery market?"
            )

        # Options
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_web = st.checkbox(
                "Enable web search (recommended)",
                value=bool(SERPAPI_KEY),
                disabled=not SERPAPI_KEY
            )

        # Analysis button
        if st.button("🔍 Analyze", type="primary") and query:

            # Validate query
            if len(query.strip()) < 5:
                st.error("❌ Please enter a question with at least 5 characters")
                return

            query = query.strip()[:500]  # Limit length

            # Web search
            web_context = {}
            if use_web:
                with st.spinner("🌐 Searching the web..."):
                    web_context = fetch_web_context(query, num_sources=3)

            if not web_context or not web_context.get("search_results"):
                st.info("💡 Using AI knowledge without web search")
                web_context = {
                    "search_results": [],
                    "scraped_content": {},
                    "summary": "",
                    "sources": [],
                    "source_reliability": []
                }

            # Primary model query
            with st.spinner("🤖 Analyzing with primary model..."):
                primary_response = query_perplexity(query, web_context)

            if not primary_response:
                st.error("❌ Primary model failed to respond")
                return

            # Parse primary response
            try:
                primary_data = json.loads(primary_response)
            except Exception as e:
                st.error(f"❌ Failed to parse primary response: {e}")
                st.code(primary_response[:1000])
                return

            # Evidence-based veracity scoring (single call)
            with st.spinner("✅ Verifying evidence quality..."):
                veracity_scores = evidence_based_veracity(primary_data, web_context)

            # Calculate confidence
            base_conf = float(primary_data.get("confidence", 75))

            # Final confidence calculation
            final_conf = calculate_final_confidence(
                base_conf,
                veracity_scores["overall"]
            )

            # Download JSON
            output = {
                "question": query,
                "timestamp": datetime.now().isoformat(),
                "primary_response": primary_data,
                "final_confidence": final_conf,
                "veracity_scores": veracity_scores,
                "web_sources": web_context.get("sources", [])
            }

            json_bytes = json.dumps(output, indent=2, ensure_ascii=False).encode('utf-8')
            filename = f"yureeka_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            st.download_button(
                label="💾 Download Analysis JSON",
                data=json_bytes,
                file_name=filename,
                mime="application/json"
            )

            # Render dashboard
            render_dashboard(
                primary_response,
                final_conf,
                web_context,
                base_conf,
                query,
                veracity_scores,
                web_context.get("source_reliability", [])
            )

            # Debug info
            with st.expander("🔧 Debug Information"):
                st.write("**Confidence Breakdown:**")
                st.json({
                    "base_confidence": base_conf,
                    "evidence_score": veracity_scores["overall"],
                    "final_confidence": final_conf,
                    "veracity_breakdown": veracity_scores
                })
                st.write("**Primary Model Response:**")
                st.json(primary_data)

            # Debug info
            with st.expander("🔧 Debug Information"):
                st.write("**Confidence Breakdown:**")
                st.json({
                    "base_confidence": base_conf,
                    "evidence_score": veracity_scores["overall"],
                    "final_confidence": final_conf,
                    "veracity_breakdown": veracity_scores
                })
                st.write("**Primary Model Response:**")
                st.json(primary_data)

    # =====================
    # TAB 2: EVOLUTION TRACKER (ANCHORED)
    # =====================
    with tab2:
        st.markdown("""
        #
        Upload a previous Yureeka analysis to track how the data has evolved.

        **How it works:**
        - Your previous analysis is fed to the model as context
        - The model searches for UPDATES to the same metrics/entities
        - Changes are explicitly tracked (increased/decreased/new/removed)
        - Stability score indicates how much has drifted
        """)

        # Upload previous analysis
        uploaded_file = st.file_uploader(
            "📁 Upload previous Yureeka JSON analysis",
            type=['json'],
            key="evolution_upload"
        )

        previous_data = None
        if uploaded_file:
            try:
                previous_data = json.load(uploaded_file)
                prev_response = previous_data.get("primary_response", {})
                prev_timestamp = previous_data.get("timestamp", "Unknown")

                st.success(f"✅ Loaded: {previous_data.get('question', 'Unknown query')}")
                st.caption(f"📅 Previous analysis from: {prev_timestamp}")

                # Show previous analysis summary
                with st.expander("📋 Previous Analysis Summary", expanded=False):
                    st.write(f"**Confidence:** {previous_data.get('final_confidence', 'N/A')}%")
                    st.write(f"**Summary:** {prev_response.get('executive_summary', 'N/A')}")

                    st.write("**Previous Metrics:**")
                    for k, m in list(prev_response.get("primary_metrics", {}).items())[:5]:
                        if isinstance(m, dict):
                            st.write(f"- {m.get('name', k)}: {m.get('value')} {m.get('unit', '')}")

                    st.write("**Previous Top Entities:**")
                    for i, e in enumerate(prev_response.get("top_entities", [])[:5], 1):
                        if isinstance(e, dict):
                            st.write(f"{i}. {e.get('name')}: {e.get('share', 'N/A')}")

            except Exception as e:
                st.error(f"❌ Failed to load JSON: {e}")
                previous_data = None

        # Options
        col1, col2 = st.columns(2)
        with col1:
            use_web_evo = st.checkbox(
                "Enable web search for current data",
                value=bool(SERPAPI_KEY),
                disabled=not SERPAPI_KEY,
                key="web_evolution"
            )

        with col2:
            use_same_query = st.checkbox(
                "Use same query as previous",
                value=True,
                key="same_query"
            )

        # Query input
        if use_same_query and previous_data:
            evolution_query = previous_data.get("question", "")
            st.info(f"📝 Using previous query: {evolution_query}")
        else:
            evolution_query = st.text_input(
                "Query for evolution analysis:",
                value=previous_data.get("question", "") if previous_data else "",
                key="evolution_query"
            )

        # Run evolution analysis
        if st.button("🔄 Run Evolution Analysis", type="primary", key="evolution_btn"):
            if not previous_data:
                st.error("❌ Please upload a previous analysis JSON first")
            elif not evolution_query or len(evolution_query.strip()) < 5:
                st.error("❌ Please enter a valid query")
            else:
                evolution_query = evolution_query.strip()[:500]

                # Web search for current data
                web_context_evo = {}
                if use_web_evo:
                    with st.spinner("🌐 Searching for current data..."):
                        web_context_evo = fetch_web_context(evolution_query, num_sources=3)

                if not web_context_evo:
                    web_context_evo = {
                        "search_results": [], "scraped_content": {},
                        "summary": "", "sources": [], "source_reliability": []
                    }

                # Run anchored evolution analysis
                with st.spinner("🤖 Analyzing evolution (anchored to previous)..."):
                    evolution_response = query_perplexity_anchored(
                        evolution_query,
                        previous_data,
                        web_context_evo
                    )

                if evolution_response:
                    # Parse response
                    try:
                        evolution_parsed = json.loads(evolution_response)
                    except:
                        st.error("❌ Failed to parse evolution response")
                        evolution_parsed = None

                    if evolution_parsed:
                        # Build output for download
                        evolution_output = {
                            "question": evolution_query,
                            "timestamp": datetime.now().isoformat(),
                            "analysis_type": "evolution",
                            "previous_timestamp": previous_data.get("timestamp"),
                            "primary_response": evolution_parsed,
                            "drift_summary": evolution_parsed.get("drift_summary", {}),
                            "web_sources": web_context_evo.get("sources", [])
                        }

                        # Download button
                        st.download_button(
                            label="💾 Download Evolution Analysis",
                            data=json.dumps(evolution_output, indent=2, ensure_ascii=False).encode('utf-8'),
                            file_name=f"yureeka_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                        # Render evolution dashboard
                        render_evolution_results(evolution_parsed, evolution_query)
                else:
                    st.error("❌ Evolution analysis failed")

if __name__ == "__main__":
    main()

