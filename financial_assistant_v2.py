# =========================================================
# YUREEKA AI RESEARCH ASSISTANT v7.1 - STABLE OUTPUT
# Based on v7.0 with added stability controls
# Changes marked with: # STABILITY ADDITION
# =========================================================

import os
import re
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import base64
import hashlib  # STABILITY ADDITION
import difflib  # STABILITY ADDITION
from typing import Dict, List, Optional, Any, Union, Tuple  # STABILITY ADDITION: Added Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import google.generativeai as genai
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta  # STABILITY ADDITION: Added timedelta
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

    if not PERPLEXITY_KEY or len(PERPLEXITY_KEY) < 10:
        st.error("❌ PERPLEXITY_API_KEY is missing or invalid")
        st.stop()

    if not GEMINI_KEY or len(GEMINI_KEY) < 10:
        st.error("❌ GEMINI_API_KEY is missing or invalid")
        st.stop()

    return PERPLEXITY_KEY, GEMINI_KEY, SERPAPI_KEY, SCRAPINGDOG_KEY

PERPLEXITY_KEY, GEMINI_KEY, SERPAPI_KEY, SCRAPINGDOG_KEY = load_api_keys()
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# =========================================================
# STABILITY ADDITION: NEW SECTION - STABILITY CONFIGURATION
# =========================================================

# Cache TTL by data type (in hours)
CACHE_TTL_HOURS = {
    "market_size": 168,      # 1 week - market sizes change slowly
    "industry_data": 72,     # 3 days
    "company_rankings": 24,  # 1 day
    "stock_price": 0.5,      # 30 min - prices change rapidly
    "economic_indicator": 24,
    "default": 24
}

# Keywords for detecting data type
DATA_TYPE_KEYWORDS = {
    "market_size": ["market size", "industry size", "total market", "tam", "market value", "market worth"],
    "stock_price": ["stock price", "share price", "trading at", "stock quote", "ticker"],
    "company_rankings": ["market share", "top companies", "leading players", "market leaders", "top players"],
    "economic_indicator": ["gdp", "inflation", "unemployment", "interest rate", "cpi", "ppi"],
}

# Anchoring thresholds - if new value within X% of old, keep old value
METRIC_ANCHOR_THRESHOLD = 0.10   # 10% for numeric metrics
ENTITY_ANCHOR_THRESHOLD = 0.15  # 15% for entity market shares

# In-memory response cache
_response_cache: Dict[str, Tuple[dict, datetime]] = {}

def detect_data_type(query: str) -> str:
    """Detect data type from query to determine cache TTL and stability rules"""
    query_lower = query.lower()
    for dtype, keywords in DATA_TYPE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return dtype
    return "default"

def get_cache_ttl_hours(query: str) -> int:
    """Get cache TTL in hours based on query data type"""
    dtype = detect_data_type(query)
    return CACHE_TTL_HOURS.get(dtype, CACHE_TTL_HOURS["default"])

def get_query_cache_key(query: str, sources: List[str] = None) -> str:
    """Generate stable cache key from query and sources"""
    # Normalize query
    normalized = re.sub(r'\s+', ' ', query.lower().strip())
    # Remove time-sensitive words that shouldn't affect caching
    normalized = re.sub(r'\b(today|current|latest|now|recent|this year)\b', '2024', normalized)

    content = normalized
    if sources:
        # Include top sources in key for consistency
        content += "|" + "|".join(sorted(s.lower()[:50] for s in sources[:3]))

    return hashlib.md5(content.encode()).hexdigest()[:16]

def get_cached_response(query: str, sources: List[str] = None) -> Optional[dict]:
    """Get cached response if still valid"""
    cache_key = get_query_cache_key(query, sources)

    if cache_key in _response_cache:
        cached_data, cached_time = _response_cache[cache_key]
        ttl_hours = get_cache_ttl_hours(query)

        if datetime.now() - cached_time < timedelta(hours=ttl_hours):
            return cached_data
        else:
            # Expired - remove from cache
            del _response_cache[cache_key]

    return None

def cache_response(query: str, response_data: dict, sources: List[str] = None):
    """Cache response with timestamp"""
    cache_key = get_query_cache_key(query, sources)
    _response_cache[cache_key] = (response_data, datetime.now())

def parse_numeric_value(value: Any) -> Optional[float]:
    """Parse any value to float for comparison"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r'[,$%]', '', value.strip())
        multiplier = 1.0
        if 'billion' in cleaned.lower() or cleaned.lower().endswith('b'):
            multiplier = 1000
            cleaned = re.sub(r'[bB](?:illion)?', '', cleaned)
        elif 'million' in cleaned.lower() or cleaned.lower().endswith('m'):
            multiplier = 1
            cleaned = re.sub(r'[mM](?:illion)?', '', cleaned)
        elif 'trillion' in cleaned.lower() or cleaned.lower().endswith('t'):
            multiplier = 1000000
            cleaned = re.sub(r'[tT](?:rillion)?', '', cleaned)
        try:
            return float(cleaned.strip()) * multiplier
        except:
            return None
    return None

def values_within_threshold(old_val: Any, new_val: Any, threshold: float) -> bool:
    """Check if two values are within threshold percentage of each other"""
    old_num = parse_numeric_value(old_val)
    new_num = parse_numeric_value(new_val)

    if old_num is None or new_num is None:
        return False
    if old_num == 0:
        return new_num == 0

    pct_diff = abs(old_num - new_num) / abs(old_num)
    return pct_diff <= threshold

def normalize_metric_name(name: str) -> str:
    """Normalize metric name for matching across responses"""
    if not name:
        return ""
    norm = re.sub(r'[^\w\s]', '', name.lower().strip())
    norm = re.sub(r'\s+', ' ', norm)

    # Common synonyms
    synonyms = {
        "market_size": ["market size", "total market", "market value", "industry size"],
        "growth_rate": ["growth rate", "cagr", "growth", "annual growth"],
        "market_share": ["market share", "share", "market portion"],
    }

    for canonical, variants in synonyms.items():
        if any(v in norm for v in variants):
            return canonical
    return norm

def fuzzy_match_metric_names(name1: str, name2: str, threshold: float = 0.7) -> bool:
    """Check if two metric names refer to the same thing"""
    n1 = normalize_metric_name(name1)
    n2 = normalize_metric_name(name2)

    if n1 == n2:
        return True

    # Fuzzy match
    ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
    return ratio >= threshold

def apply_anchoring(new_response: dict, previous_response: dict) -> dict:
    """
    Apply anchoring: keep previous values if new values are within threshold.
    This reduces volatility from LLM randomness.
    """
    if not previous_response:
        return new_response

    # Anchor metrics
    old_metrics = previous_response.get("primary_metrics", {})
    new_metrics = new_response.get("primary_metrics", {})

    if old_metrics and new_metrics:
        for old_key, old_m in old_metrics.items():
            if not isinstance(old_m, dict):
                continue

            old_name = old_m.get("name", old_key)
            old_val = old_m.get("value")

            # Find matching metric in new response
            for new_key, new_m in new_metrics.items():
                if not isinstance(new_m, dict):
                    continue

                new_name = new_m.get("name", new_key)

                if fuzzy_match_metric_names(old_name, new_name):
                    new_val = new_m.get("value")

                    # If within threshold, keep old value
                    if values_within_threshold(old_val, new_val, METRIC_ANCHOR_THRESHOLD):
                        new_metrics[new_key]["value"] = old_val
                        new_metrics[new_key]["_anchored"] = True
                    break

    # Anchor entity market shares
    old_entities = previous_response.get("top_entities", [])
    new_entities = new_response.get("top_entities", [])

    if old_entities and new_entities:
        old_shares = {}
        for e in old_entities:
            if isinstance(e, dict):
                name = e.get("name", "").lower().strip()
                old_shares[name] = e.get("share")

        for new_e in new_entities:
            if isinstance(new_e, dict):
                name = new_e.get("name", "").lower().strip()
                if name in old_shares:
                    old_share = old_shares[name]
                    new_share = new_e.get("share")

                    if values_within_threshold(old_share, new_share, ENTITY_ANCHOR_THRESHOLD):
                        new_e["share"] = old_share
                        new_e["_anchored"] = True

    return new_response

# END STABILITY ADDITION

# =========================================================
# 2. PYDANTIC MODELS
# =========================================================

class MetricDetail(BaseModel):
    name: str = Field(..., description="Metric name")
    value: Union[float, int, str] = Field(..., description="Metric value")
    unit: str = Field(default="", description="Unit of measurement")
    model_config = ConfigDict(extra='ignore')

class TopEntityDetail(BaseModel):
    name: str = Field(..., description="Entity name")
    share: Optional[str] = Field(None, description="Market share")
    growth: Optional[str] = Field(None, description="Growth rate")
    model_config = ConfigDict(extra='ignore')

class TrendForecastDetail(BaseModel):
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
    title: str = Field("Comparison", description="Chart title")
    categories: List[str] = Field(default_factory=list)
    values: List[Union[float, int]] = Field(default_factory=list)
    model_config = ConfigDict(extra='ignore')

class BenchmarkTable(BaseModel):
    category: str
    value_1: Union[float, int, str] = Field(default=0)
    value_2: Union[float, int, str] = Field(default=0)
    model_config = ConfigDict(extra='ignore')

class Action(BaseModel):
    recommendation: str = Field("Neutral")
    confidence: str = Field("Medium")
    rationale: str = Field("")
    model_config = ConfigDict(extra='ignore')

class LLMResponse(BaseModel):
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
# 3. PROMPTS - STABILITY ADDITION: Enhanced for consistent output
# =========================================================

RESPONSE_TEMPLATE = """
{
  "executive_summary": "4-6 sentences with specific numbers",
  "primary_metrics": {
    "metric_1": {"name": "Key Metric 1", "value": 25.5, "unit": "%"},
    "metric_2": {"name": "Key Metric 2", "value": 623, "unit": "$B"}
  },
  "key_findings": ["Finding 1", "Finding 2"],
  "top_entities": [{"name": "Entity 1", "share": "25%", "growth": "15%"}],
  "trends_forecast": [{"trend": "Trend description", "direction": "↑", "timeline": "2025-2027"}],
  "visualization_data": {
    "chart_labels": ["2023", "2024", "2025"],
    "chart_values": [100, 120, 145],
    "chart_title": "Market Growth",
    "chart_type": "line"
  },
  "sources": ["source1.com"],
  "confidence": 87,
  "freshness": "Dec 2024"
}
"""

# STABILITY ADDITION: Enhanced system prompt with stability rules
SYSTEM_PROMPT = f"""You are a professional market research analyst providing CONSISTENT, VERIFIABLE data.

CRITICAL RULES:
1. Return ONLY valid JSON. NO markdown, NO code blocks, NO extra text.
2. NO citation references like [1][2] inside strings.
3. Use double quotes for all keys and string values.
4. NO trailing commas in arrays or objects.

STABILITY RULES (IMPORTANT FOR CONSISTENT OUTPUT):
- Round market sizes to nearest $0.1B (e.g., $58.3B not $58.27B)
- Round percentages to nearest 0.1% (e.g., 21.5% not 21.47%)
- Use consistent units: USD billions for market sizes, % for growth rates
- Entity rankings should be based on market share and remain stable unless clear evidence contradicts
- Use CAGR for growth projections when available
- For forecasts, use most commonly cited industry projections

NUMERIC PRECISION RULES:
- Market sizes > $1B: Round to 1 decimal (58.3B)
- Market sizes < $1B: Round to whole millions (847M)
- Percentages: Always 1 decimal (21.5%)
- Rankings: Only change if market share data clearly supports different order

REQUIRED FIELDS:
- executive_summary: 4-6 sentences with specific quantitative data
- primary_metrics: 3+ metrics with numbers and consistent units
- key_findings: 3+ findings with quantitative details
- top_entities: 3+ companies/countries with market share %
- trends_forecast: 2+ trends with timelines
- visualization_data: chart_labels and chart_values arrays

Output ONLY this JSON structure:
{RESPONSE_TEMPLATE}
"""

# STABILITY ADDITION: Function to build prompt with previous values for anchoring
def build_stable_prompt(query: str, web_context: Dict, previous_response: Optional[dict] = None) -> str:
    """
    Build prompt with optional anchoring to previous values.
    If previous_response provided, includes it as reference for stability.
    """
    search_count = len(web_context.get("search_results", []))

    # Base context
    if not web_context.get("summary") or search_count < 2:
        context_section = f"Web search returned {search_count} results. Use your knowledge for complete analysis."
    else:
        context_section = f"LATEST WEB RESEARCH:\n{web_context['summary']}\n"
        if web_context.get('scraped_content'):
            context_section += "\nDETAILED CONTENT:\n"
            for url, content in list(web_context['scraped_content'].items())[:2]:
                context_section += f"\n{url}:\n{content[:800]}...\n"

    # STABILITY ADDITION: Add anchoring section if previous response exists
    anchor_section = ""
    if previous_response:
        anchor_section = "\n\nPREVIOUS ANALYSIS (use as anchor - only deviate if sources clearly contradict):\n"

        # Previous metrics
        prev_metrics = previous_response.get("primary_metrics", {})
        if prev_metrics:
            anchor_section += "Previous Metrics:\n"
            for k, m in list(prev_metrics.items())[:5]:
                if isinstance(m, dict):
                    anchor_section += f"  - {m.get('name', k)}: {m.get('value')} {m.get('unit', '')}\n"

        # Previous entity rankings
        prev_entities = previous_response.get("top_entities", [])
        if prev_entities:
            anchor_section += "Previous Entity Rankings:\n"
            for i, e in enumerate(prev_entities[:5], 1):
                if isinstance(e, dict):
                    anchor_section += f"  {i}. {e.get('name')}: {e.get('share', 'N/A')}\n"

        anchor_section += """
ANCHORING RULES:
- If new data is within 10% of previous values, KEEP the previous value for stability
- Only update metrics if sources show CLEAR evidence of change
- Maintain entity rankings unless market share data explicitly contradicts
"""

    return f"{context_section}{anchor_section}\n{SYSTEM_PROMPT}\n\nUser Question: {query}"

# =========================================================
# 4. MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner="🔧 Loading AI models...")
def load_models():
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return classifier, embedder
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

domain_classifier, embedder = load_models()

# =========================================================
# 5. JSON REPAIR FUNCTIONS (unchanged from v7)
# =========================================================

def repair_llm_response(data: dict) -> dict:
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

    if "top_entities" in data:
        if isinstance(data["top_entities"], dict):
            data["top_entities"] = list(data["top_entities"].values())
        elif not isinstance(data["top_entities"], list):
            data["top_entities"] = []

    if "trends_forecast" in data:
        if isinstance(data["trends_forecast"], dict):
            data["trends_forecast"] = list(data["trends_forecast"].values())
        elif not isinstance(data["trends_forecast"], list):
            data["trends_forecast"] = []

    if "visualization_data" in data and isinstance(data["visualization_data"], dict):
        viz = data["visualization_data"]
        if "labels" in viz and "chart_labels" not in viz:
            viz["chart_labels"] = viz.pop("labels")
        if "values" in viz and "chart_values" not in viz:
            viz["chart_values"] = viz.pop("values")

    if "benchmark_table" in data and isinstance(data["benchmark_table"], list):
        for row in data["benchmark_table"]:
            if isinstance(row, dict):
                row.setdefault("category", "Unknown")
                for key in ["value_1", "value_2"]:
                    val = row.get(key, 0)
                    if isinstance(val, str):
                        if val.upper().strip() in ["N/A", "NA", "NULL", "NONE", "", "-"]:
                            row[key] = 0
                        else:
                            try:
                                cleaned = re.sub(r'[^\d.-]', '', val)
                                row[key] = float(cleaned) if cleaned else 0
                            except:
                                row[key] = 0
    return data

def preclean_json(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = re.sub(r'\[web:\d+\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

def parse_json_safely(json_str: str, context: str = "LLM") -> dict:
    if not json_str:
        return {}
    cleaned = preclean_json(json_str)
    match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if not match:
        return {}
    json_content = match.group(0)

    try:
        json_content = re.sub(r'([\{\,]\s*)([a-zA-Z_][a-zA-Z0-9_\-]+)(\s*):', r'\1"\2"\3:', json_content)
        json_content = re.sub(r',\s*([\]\}])', r'\1', json_content)
        json_content = json_content.replace(': True', ': true').replace(': False', ': false')
    except:
        pass

    for _ in range(10):
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            if "Unterminated string" not in str(e):
                break
            pos = getattr(e, 'pos', 0)
            for i in range(pos - 1, max(0, pos - 150), -1):
                if i < len(json_content) and json_content[i] == '"' and (i == 0 or json_content[i-1] != '\\'):
                    json_content = json_content[:i] + '\\"' + json_content[i+1:]
                    break
            else:
                break
    return {}

# =========================================================
# 6. WEB SEARCH FUNCTIONS (unchanged from v7)
# =========================================================

def classify_source_reliability(source: str) -> str:
    source = source.lower() if isinstance(source, str) else ""
    high = ["gov", "imf", "worldbank", "central bank", "fed", "ecb", "reuters", "spglobal", "economist", "mckinsey", "bcg", "cognitive market research",
            "financial times", "wsj", "oecd", "bloomberg", "tradingeconomics", "deloitte", "hsbc", "imarc", "booz", "bakerinstitute.org",
           "kpmg", "semiconductors.org", "eu", "iea", "world bank", "opec", "jp morgan", "citibank", "goldman sachs", "j.p. morgan"]
    medium = ["wikipedia", "forbes", "cnbc", "yahoo", "statista", "ceic"]
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
    if not sources:
        return 50.0
    weights = {"✅ High": 100, "⚠️ Medium": 60, "❌ Low": 30}
    scores = [weights.get(classify_source_reliability(s), 60) for s in sources]
    return sum(scores) / len(scores) if scores else 50.0

@st.cache_data(ttl=3600, show_spinner=False)
def search_serpapi(query: str, num_results: int = 5) -> List[Dict]:
    if not SERPAPI_KEY:
        return []

    # STABILITY ADDITION: Normalize query for consistent search
    query_normalized = re.sub(r'\b(latest|current|today|now)\b', '2024', query.lower())

    query_lower = query_normalized
    if any(kw in query_lower for kw in ["industry", "market", "sector", "size", "growth"]):
        search_terms = f"{query_normalized} market size growth trends 2024"
        tbm, tbs = "", ""
    else:
        search_terms = f"{query_normalized} finance economics data"
        tbm, tbs = "nws", "qdr:m"

    params = {"engine": "google", "q": search_terms, "api_key": SERPAPI_KEY, "num": num_results, "tbm": tbm, "tbs": tbs}

    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("news_results", [])[:num_results]:
            src = item.get("source", {})
            results.append({
                "title": item.get("title", ""), "link": item.get("link", ""),
                "snippet": item.get("snippet", ""), "date": item.get("date", ""),
                "source": src.get("name", "") if isinstance(src, dict) else str(src)
            })

        if not results:
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""), "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""), "date": "", "source": item.get("source", "")
                })

        # STABILITY ADDITION: Sort for consistent ordering
        results.sort(key=lambda x: (x.get("source", "").lower(), x.get("link", "")))
        return results[:num_results]
    except:
        return []

def scrape_url(url: str) -> Optional[str]:
    if not SCRAPINGDOG_KEY:
        return None
    try:
        resp = requests.get("https://api.scrapingdog.com/scrape",
                          params={"api_key": SCRAPINGDOG_KEY, "url": url, "dynamic": "false"}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)[:3000]
    except:
        return None

def fetch_web_context(query: str, num_sources: int = 3) -> Dict:
    search_results = search_serpapi(query, num_results=5)
    if not search_results:
        return {"search_results": [], "scraped_content": {}, "summary": "", "sources": [], "source_reliability": []}

    scraped_content = {}
    if SCRAPINGDOG_KEY:
        progress = st.progress(0)
        for i, result in enumerate(search_results[:num_sources]):
            content = scrape_url(result["link"])
            if content:
                scraped_content[result["link"]] = content
            progress.progress((i + 1) / num_sources)
        progress.empty()

    context_parts, reliabilities = [], []
    for r in search_results:
        reliability = classify_source_reliability(r.get("link", "") + " " + r.get("source", ""))
        reliabilities.append(reliability)
        context_parts.append(f"**{r['title']}**\nSource: {r['source']} [{reliability}]\n{r['snippet']}\nURL: {r['link']}")

    return {
        "search_results": search_results, "scraped_content": scraped_content,
        "summary": "\n\n---\n\n".join(context_parts), "sources": [r["link"] for r in search_results],
        "source_reliability": reliabilities
    }

# =========================================================
# 7. LLM QUERY FUNCTIONS - STABILITY ADDITION: Modified for stable output
# =========================================================

def query_perplexity(
    query: str,
    web_context: Dict,
    temperature: float = 0.0,  # STABILITY CHANGE: Default to 0 for deterministic output
    previous_response: Optional[dict] = None,  # STABILITY ADDITION
    force_refresh: bool = False  # STABILITY ADDITION
) -> Tuple[str, bool]:  # STABILITY CHANGE: Returns (response, was_cached)
    """
    Query Perplexity API with stability controls.
    Returns (response_json, was_cached) tuple.
    """

    sources = web_context.get("sources", [])

    # STABILITY ADDITION: Check cache first
    if not force_refresh:
        cached = get_cached_response(query, sources)
        if cached:
            return json.dumps(cached), True

    search_count = len(web_context.get("search_results", []))

    # STABILITY ADDITION: Use stable prompt builder
    enhanced_query = build_stable_prompt(query, web_context, previous_response)

    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}

    # STABILITY CHANGE: temperature=0 for deterministic output
    payload = {
        "model": "sonar",
        "temperature": temperature,  # 0 = deterministic
        "max_tokens": 2000,
        "top_p": 1.0,  # STABILITY CHANGE: No nucleus sampling
        "messages": [{"role": "user", "content": enhanced_query}]
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data:
            raise Exception("No choices in response")

        content = data["choices"][0]["message"]["content"]
        if not content:
            raise Exception("Empty response")

        parsed = parse_json_safely(content, "Perplexity")
        if not parsed:
            return create_fallback_response(query, search_count, web_context), False

        repaired = repair_llm_response(parsed)

        # STABILITY ADDITION: Apply anchoring if previous response exists
        if previous_response:
            repaired = apply_anchoring(repaired, previous_response)

        try:
            llm_obj = LLMResponse.model_validate(repaired)
            if web_context.get("sources"):
                existing = llm_obj.sources or []
                llm_obj.sources = list(dict.fromkeys(existing + web_context["sources"]))[:10]
                llm_obj.freshness = "Current (web-enhanced)"

            result = json.loads(llm_obj.model_dump_json())

            # STABILITY ADDITION: Cache the response
            cache_response(query, result, sources)

            return json.dumps(result), False

        except ValidationError:
            return create_fallback_response(query, search_count, web_context), False

    except Exception as e:
        st.error(f"❌ Perplexity API error: {e}")
        return create_fallback_response(query, search_count, web_context), False

def query_gemini(query: str) -> str:
    """Query Gemini API (unchanged from v7)"""
    prompt = f"{SYSTEM_PROMPT}\n\nUser query: {query}"

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=2000)  # STABILITY: temp=0
        )
        content = getattr(response, "text", None)
        if not content:
            raise Exception("Empty response")

        parsed = parse_json_safely(content, "Gemini")
        if not parsed:
            return create_fallback_response(query, 0, {})

        repaired = repair_llm_response(parsed)
        try:
            llm_obj = LLMResponse.model_validate(repaired)
            return llm_obj.model_dump_json()
        except ValidationError:
            return create_fallback_response(query, 0, {})
    except Exception:
        return create_fallback_response(query, 0, {})

def create_fallback_response(query: str, search_count: int, web_context: Dict) -> str:
    fallback = LLMResponse(
        executive_summary=f"Analysis of '{query}' with {search_count} sources. Fallback structure used.",
        primary_metrics={"sources": MetricDetail(name="Web Sources", value=search_count, unit="sources")},
        key_findings=[f"Found {search_count} sources", "Fallback response generated"],
        top_entities=[TopEntityDetail(name="N/A", share="N/A", growth="N/A")],
        trends_forecast=[TrendForecastDetail(trend="Fallback", direction="⚠️", timeline="N/A")],
        visualization_data=VisualizationData(chart_labels=["N/A"], chart_values=[0], chart_title="No Data"),
        sources=web_context.get("sources", []),
        confidence=50,
        freshness="Current (fallback)"
    )
    return fallback.model_dump_json()

# =========================================================
# 8. VALIDATION & SCORING (unchanged from v7)
# =========================================================

@st.cache_data
def get_embedding(text: str):
    return embedder.encode(text)

def semantic_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    try:
        v1 = get_embedding(text1)
        v2 = get_embedding(text2)
        return round(float(util.cos_sim(v1, v2).item()) * 100, 2)
    except:
        return 0.0

def parse_number_with_unit(val_str: str) -> float:
    if not val_str:
        return 0.0
    val_str = str(val_str).replace('$', '').replace(',', '').strip()
    multiplier = 1.0
    if val_str.endswith(('B', 'b')):
        multiplier = 1000.0
        val_str = val_str[:-1]
    elif val_str.endswith(('M', 'm')):
        val_str = val_str[:-1]
    elif val_str.endswith(('K', 'k')):
        multiplier = 0.001
        val_str = val_str[:-1]
    try:
        return float(val_str) * multiplier
    except:
        return 0.0

def numeric_consistency_with_sources(primary_data: dict, web_context: dict) -> float:
    primary_metrics = primary_data.get("primary_metrics", {})
    primary_numbers = []
    for metric in primary_metrics.values():
        if isinstance(metric, dict):
            num = parse_number_with_unit(str(metric.get("value", "")))
            if num > 0:
                primary_numbers.append(num)

    if not primary_numbers:
        return 50.0

    source_numbers = []
    for result in web_context.get("search_results", []):
        snippet = str(result.get("snippet", ""))
        patterns = [r'\$?(\d+(?:\.\d+)?)\s*([BbMmKk])', r'(\d+(?:\.\d+)?)\s*(billion|million)']
        for pattern in patterns:
            for num, unit in re.findall(pattern, snippet, re.IGNORECASE):
                source_numbers.append(parse_number_with_unit(f"{num}{unit[0].upper()}"))

    if not source_numbers:
        return 50.0

    agreements = sum(1 for p in primary_numbers if any(abs(p - s) / max(p, s, 1) < 0.25 for s in source_numbers))
    return min(30.0 + (agreements / len(primary_numbers) * 65.0), 95.0)

def source_consensus(web_context: dict) -> float:
    reliabilities = web_context.get("source_reliability", [])
    if not reliabilities:
        return 50.0

    high = sum(1 for r in reliabilities if "✅" in str(r))
    medium = sum(1 for r in reliabilities if "⚠️" in str(r))
    low = sum(1 for r in reliabilities if "❌" in str(r))
    total = len(reliabilities)

    score = (high * 100 + medium * 60 + low * 30) / total
    if high >= 3:
        score = min(100, score + 10)
    elif high >= 2:
        score = min(100, score + 5)
    return round(score, 1)

def evidence_based_veracity(primary_data: dict, web_context: dict) -> dict:
    sources = primary_data.get("sources", [])
    src_score = source_quality_score(sources)
    num_score = numeric_consistency_with_sources(primary_data, web_context)

    sources_count = len(sources)
    total_claims = len(primary_data.get("key_findings", [])) + len(primary_data.get("primary_metrics", {}))

    if total_claims == 0:
        citations_score = 40.0
    else:
        ratio = sources_count / total_claims
        citations_score = min(90.0 if ratio >= 1.0 else 70.0 + (ratio - 0.5) * 40 if ratio >= 0.5 else 50.0 + (ratio - 0.25) * 80 if ratio >= 0.25 else ratio * 200, 95.0)

    consensus_score = source_consensus(web_context)

    return {
        "source_quality": round(src_score, 1),
        "numeric_consistency": round(num_score, 1),
        "citation_density": round(citations_score, 1),
        "source_consensus": round(consensus_score, 1),
        "overall": round(src_score * 0.35 + num_score * 0.30 + citations_score * 0.20 + consensus_score * 0.15, 1)
    }

def calculate_final_confidence(base_conf: float, evidence_score: float) -> float:
    base_conf = max(0, min(100, base_conf))
    evidence_score = max(0, min(100, evidence_score))
    evidence_component = evidence_score * 0.65
    evidence_multiplier = 0.5 + (evidence_score / 200)
    model_component = base_conf * evidence_multiplier * 0.35
    return round(max(0, min(100, evidence_component + model_component)), 1)

# =========================================================
# 9. DASHBOARD RENDERING - STABILITY ADDITION: Show stability info
# =========================================================

def detect_x_label_dynamic(labels: list) -> str:
    if not labels:
        return "Category"
    label_texts = [str(l).lower() for l in labels]
    all_text = ' '.join(label_texts)

    region_kw = ['north america', 'asia pacific', 'europe', 'latin america', 'middle east', 'china', 'usa', 'india']
    if sum(1 for l in label_texts if any(k in l for k in region_kw)) / len(labels) >= 0.4:
        return "Regions"
    if sum(1 for l in label_texts if re.search(r'\b20\d{2}\b', l)) / len(labels) > 0.5:
        return "Years"
    if sum(1 for l in label_texts if re.search(r'\bq[1-4]\b', l, re.I)) >= 2:
        return "Quarters"
    return "Categories"

def detect_y_label_dynamic(values: list) -> str:
    if not values:
        return "Value"
    try:
        nums = [abs(float(v)) for v in values]
        avg, mx = np.mean(nums), max(nums)
        if mx > 500 or avg > 100:
            return "USD B"
        elif mx > 50 or avg > 10:
            return "USD M"
        elif all(0 <= v <= 100 for v in nums):
            return "Percent %"
    except:
        pass
    return "Value"

# STABILITY ADDITION: New function to render stability information
def render_stability_info(was_cached: bool, query: str, data: dict):
    """Render stability status panel"""
    st.subheader("🔒 Output Stability")

    cols = st.columns(4)

    # Cache status
    data_type = detect_data_type(query)
    ttl = get_cache_ttl_hours(query)

    if was_cached:
        cols[0].metric("Cache Status", "✅ Cached", help="Response from cache - guaranteed stable")
    else:
        cols[0].metric("Cache Status", "🔄 Fresh", help="New response generated")

    cols[1].metric("Data Type", data_type.replace("_", " ").title())
    cols[2].metric("Cache TTL", f"{ttl}h")

    # Count anchored values
    anchored_count = 0
    for m in data.get("primary_metrics", {}).values():
        if isinstance(m, dict) and m.get("_anchored"):
            anchored_count += 1
    for e in data.get("top_entities", []):
        if isinstance(e, dict) and e.get("_anchored"):
            anchored_count += 1

    cols[3].metric("Anchored Values", anchored_count, help="Values kept from previous for stability")

    # Explanation expander
    with st.expander("ℹ️ How stability works"):
        st.markdown(f"""
        **Stability Controls Applied:**
        1. **Response Caching**: Same query returns cached response for {ttl} hours
        2. **Temperature = 0**: LLM generates deterministically
        3. **Value Anchoring**: New values within 10% of previous are kept unchanged
        4. **Query Normalization**: "latest"/"current" → "2024" for consistent search
        5. **Structured Prompting**: Explicit rounding rules for numbers

        **Data Type: {data_type.replace("_", " ").title()}**
        - Cache TTL: {ttl} hours
        - {"Market data changes slowly - high stability expected" if data_type == "market_size" else "Standard caching applied"}
        """)

def render_dashboard(
    primary_json: str,
    final_conf: float,
    web_context: Dict,
    base_conf: float,
    user_question: str,
    veracity_scores: Dict,
    source_reliability: List[str],
    was_cached: bool = False  # STABILITY ADDITION
):
    try:
        data = json.loads(primary_json)
    except Exception as e:
        st.error(f"❌ Cannot render dashboard: {e}")
        return

    st.header("📊 Yureeka Market Report")
    st.markdown(f"**Question:** {user_question}")

    # Confidence row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Confidence", f"{final_conf:.1f}%")
    col2.metric("Model Confidence", f"{base_conf:.1f}%")
    col3.metric("Evidence Score", f"{veracity_scores.get('overall', 0):.1f}%")
    # STABILITY ADDITION
    col4.metric("Cache Status", "📦 Cached" if was_cached else "🔄 Fresh")

    # STABILITY ADDITION: Show stability info
    st.markdown("---")
    render_stability_info(was_cached, user_question, data)

    st.markdown("---")

    # Executive Summary
    st.subheader("📋 Executive Summary")
    st.markdown(f"**{data.get('executive_summary', 'No summary available')}**")
    st.markdown("---")

    # Key Metrics
    st.subheader("💰 Key Metrics")
    metrics = data.get('primary_metrics', {})
    if metrics:
        rows = []
        for k, m in list(metrics.items())[:6]:
            if isinstance(m, dict):
                # STABILITY ADDITION: Show anchored indicator
                anchored = " 🔗" if m.get("_anchored") else ""
                rows.append({
                    "Metric": m.get("name", k) + anchored,
                    "Value": f"{m.get('value', 'N/A')} {m.get('unit', '')}".strip()
                })
        if rows:
            st.table(pd.DataFrame(rows))
    st.markdown("---")

    # Key Findings
    st.subheader("🔍 Key Findings")
    for i, finding in enumerate(data.get('key_findings', [])[:8], 1):
        if finding:
            st.markdown(f"**{i}.** {finding}")
    st.markdown("---")

    # Top Entities
    entities = data.get('top_entities', [])
    if entities:
        st.subheader("🏢 Top Market Players")
        entity_data = []
        for e in entities:
            if isinstance(e, dict):
                # STABILITY ADDITION: Show anchored indicator
                anchored = " 🔗" if e.get("_anchored") else ""
                entity_data.append({
                    "Entity": e.get("name", "N/A") + anchored,
                    "Share": e.get("share", "N/A"),
                    "Growth": e.get("growth", "N/A")
                })
        if entity_data:
            st.dataframe(pd.DataFrame(entity_data), hide_index=True, use_container_width=True)

    # Trends
    trends = data.get('trends_forecast', [])
    if trends:
        st.subheader("📈 Trends & Forecast")
        trend_data = [{"Trend": t.get("trend", "N/A"), "Direction": t.get("direction", "→"), "Timeline": t.get("timeline", "N/A")}
                     for t in trends if isinstance(t, dict)]
        if trend_data:
            st.table(pd.DataFrame(trend_data))
    st.markdown("---")

    # Visualization
    st.subheader("📊 Data Visualization")
    viz = data.get('visualization_data')
    if viz and isinstance(viz, dict):
        labels = viz.get("chart_labels", [])
        values = viz.get("chart_values", [])
        if labels and values and len(labels) == len(values):
            try:
                nums = [float(v) for v in values[:10]]
                df = pd.DataFrame({"x": labels[:10], "y": nums})
                chart_type = viz.get("chart_type", "line")
                fig = px.bar(df, x="x", y="y", title=viz.get("chart_title", "Trend")) if chart_type == "bar" else px.line(df, x="x", y="y", title=viz.get("chart_title", "Trend"), markers=True)
                fig.update_layout(xaxis_title=detect_x_label_dynamic(labels), yaxis_title=detect_y_label_dynamic(nums))
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Chart rendering failed")
    st.markdown("---")

    # Sources
    st.subheader("🔗 Sources")
    all_sources = data.get('sources', []) or web_context.get('sources', [])
    if all_sources:
        cols = st.columns(2)
        for i, src in enumerate(all_sources[:10], 1):
            reliability = classify_source_reliability(str(src))
            cols[(i-1) % 2].markdown(f"**{i}.** [{src[:50]}...]({src}) {reliability}", unsafe_allow_html=True)
    st.markdown("---")

    # Evidence Quality
    if veracity_scores:
        st.subheader("✅ Evidence Quality Scores")
        cols = st.columns(5)
        for i, (label, key) in enumerate([("Sources", "source_quality"), ("Numbers", "numeric_consistency"),
                                          ("Citations", "citation_density"), ("Consensus", "source_consensus"), ("Overall", "overall")]):
            cols[i].metric(label, f"{veracity_scores.get(key, 0):.0f}%")

    # Web Context
    if web_context.get("search_results"):
        with st.expander("🌐 Web Search Details"):
            for i, r in enumerate(web_context["search_results"][:5], 1):
                st.markdown(f"**{i}. {r.get('title')}**")
                st.caption(f"{r.get('source')} - {r.get('date')}")
                st.write(r.get('snippet', ''))
                st.markdown("---")

# =========================================================
# 10. MAIN APPLICATION - STABILITY ADDITION: Added controls
# =========================================================

def main():
    st.set_page_config(page_title="Yureeka Market Report", page_icon="💹", layout="wide")

    st.title("💹 Yureeka Market Intelligence")

    # STABILITY ADDITION: Version info with stability features
    st.markdown("""
    **Yureeka v7.1** - Now with output stability controls.

    *Features: Response caching, value anchoring, deterministic LLM output*
    """)

    # STABILITY ADDITION: Cache status in sidebar
    with st.sidebar:
        st.subheader("🔒 Stability Controls")
        st.write(f"**Cached queries:** {len(_response_cache)}")

        for key, (data, time) in list(_response_cache.items())[:3]:
            age = (datetime.now() - time).total_seconds() / 3600
            st.caption(f"• `{key[:8]}...` - {age:.1f}h ago")

        if st.button("🗑️ Clear Cache"):
            _response_cache.clear()
            st.success("Cache cleared!")
            st.rerun()

        st.markdown("---")
        st.markdown("""
        **Stability Features:**
        - 🌡️ Temperature = 0
        - 📦 Response caching
        - 🔗 Value anchoring
        - 📝 Structured prompts
        """)

    # Main tabs
    tab1, tab2 = st.tabs(["🔍 New Analysis", "📊 Compare with Previous"])

    with tab1:
        query = st.text_input(
            "Enter your question about markets, industries, finance, or economics:",
            placeholder="e.g., What is the size of the global EV battery market?",
            key="query_new"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            use_web = st.checkbox("Enable web search", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY, key="web_new")
        with col2:
            # STABILITY ADDITION
            force_refresh = st.checkbox("Force refresh (skip cache)", value=False, key="refresh_new")
        with col3:
            if query:
                dtype = detect_data_type(query)
                ttl = get_cache_ttl_hours(query)
                st.caption(f"Type: {dtype} | Cache: {ttl}h")

        if st.button("🔍 Analyze", type="primary", key="analyze_new") and query:
            run_analysis(query.strip()[:500], use_web, force_refresh, None)

    with tab2:
        st.markdown("### Compare with Previous Analysis")
        st.markdown("Upload a previous Yureeka JSON to use as anchor for stable comparison.")

        uploaded = st.file_uploader("Upload previous JSON output", type=['json'], key="upload")

        previous_data = None
        if uploaded:
            try:
                previous_data = json.load(uploaded)
                st.success(f"✅ Loaded: {previous_data.get('question', 'Unknown')}")
                st.caption(f"Timestamp: {previous_data.get('timestamp', 'Unknown')}")

                # Show previous metrics
                prev_resp = previous_data.get("primary_response", {})
                prev_metrics = prev_resp.get("primary_metrics", {})
                if prev_metrics:
                    with st.expander("📋 Previous Metrics (will be used for anchoring)"):
                        for k, m in list(prev_metrics.items())[:5]:
                            if isinstance(m, dict):
                                st.write(f"**{m.get('name', k)}**: {m.get('value')} {m.get('unit', '')}")
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")

        query2 = st.text_input(
            "Query (or use same as previous):",
            value=previous_data.get("question", "") if previous_data else "",
            key="query_compare"
        )

        col1, col2 = st.columns(2)
        with col1:
            use_web2 = st.checkbox("Enable web search", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY, key="web_compare")
        with col2:
            force_refresh2 = st.checkbox("Force refresh", value=False, key="refresh_compare")

        if st.button("🔄 Run Comparison", type="primary", key="analyze_compare") and query2:
            prev_resp = previous_data.get("primary_response") if previous_data else None
            run_analysis(query2.strip()[:500], use_web2, force_refresh2, prev_resp)

def run_analysis(query: str, use_web: bool, force_refresh: bool, previous_response: Optional[dict]):
    """Run analysis with stability controls"""

    if len(query.strip()) < 5:
        st.error("❌ Please enter a question with at least 5 characters")
        return

    # Web search
    web_context = {"search_results": [], "scraped_content": {}, "summary": "", "sources": [], "source_reliability": []}
    if use_web:
        with st.spinner("🌐 Searching the web..."):
            web_context = fetch_web_context(query, num_sources=3)

        if web_context.get("search_results"):
            st.success(f"Found {len(web_context['search_results'])} sources")
        else:
            st.info("💡 Using AI knowledge without web search")

    # Query LLM with stability controls
    with st.spinner("🤖 Analyzing (with stability controls)..."):
        primary_response, was_cached = query_perplexity(
            query,
            web_context,
            temperature=0.0,  # Deterministic
            previous_response=previous_response,
            force_refresh=force_refresh
        )

    if was_cached:
        st.info("📦 Response loaded from cache (guaranteed stable)")

    if not primary_response:
        st.error("❌ Analysis failed")
        return

    try:
        primary_data = json.loads(primary_response)
    except Exception as e:
        st.error(f"❌ Parse error: {e}")
        return

    # Veracity scoring
    with st.spinner("✅ Verifying evidence quality..."):
        veracity_scores = evidence_based_veracity(primary_data, web_context)

    base_conf = float(primary_data.get("confidence", 75))
    final_conf = calculate_final_confidence(base_conf, veracity_scores["overall"])

    # Build output
    output = {
        "question": query,
        "timestamp": datetime.now().isoformat(),
        "primary_response": primary_data,
        "final_confidence": final_conf,
        "veracity_scores": veracity_scores,
        "web_sources": web_context.get("sources", []),
        # STABILITY ADDITION
        "stability": {
            "was_cached": was_cached,
            "data_type": detect_data_type(query),
            "cache_ttl_hours": get_cache_ttl_hours(query),
            "had_previous_anchor": previous_response is not None
        }
    }

    # Download button
    st.download_button(
        "💾 Download Analysis JSON",
        json.dumps(output, indent=2, ensure_ascii=False).encode('utf-8'),
        f"yureeka_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "application/json"
    )

    # Render dashboard
    render_dashboard(
        primary_response,
        final_conf,
        web_context,
        base_conf,
        query,
        veracity_scores,
        web_context.get("source_reliability", []),
        was_cached  # STABILITY ADDITION
    )

    # Debug info
    with st.expander("🔧 Debug Information"):
        st.json({
            "base_confidence": base_conf,
            "evidence_score": veracity_scores["overall"],
            "final_confidence": final_conf,
            "was_cached": was_cached,
            "cache_key": get_query_cache_key(query, web_context.get("sources")),
            "data_type": detect_data_type(query),
            "cache_ttl": get_cache_ttl_hours(query),
            "anchored_metrics": [k for k, v in primary_data.get("primary_metrics", {}).items()
                                if isinstance(v, dict) and v.get("_anchored")],
            "veracity_breakdown": veracity_scores
        })

if __name__ == "__main__":
    main()

