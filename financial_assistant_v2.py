# =========================================================
# YUREEKA AI RESEARCH ASSISTANT v9.0 - HYBRID VERIFICATION
# Combines v7 speed with v8 accuracy
# Fast source scoring + Targeted claim verification
# =========================================================

import os
import re
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import google.generativeai as genai
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from collections import Counter
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from dataclasses import dataclass, field

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
# 3. VERIFICATION DATA STRUCTURES
# =========================================================

@dataclass
class ExtractedClaim:
    """A verifiable claim extracted from LLM output"""
    text: str
    claim_type: str  # 'numeric', 'entity', 'general'
    numeric_value: Optional[float] = None
    numeric_unit: Optional[str] = None
    source_field: str = ""
    priority: int = 1  # 1=high (numeric), 2=medium (entity), 3=low (general)

@dataclass
class ClaimVerification:
    """Result of verifying a single claim"""
    claim: ExtractedClaim
    is_grounded: bool
    confidence: float
    supporting_source: Optional[str] = None
    supporting_text: Optional[str] = None
    match_type: str = "none"  # 'numeric', 'semantic', 'keyword', 'none'

@dataclass
class VeracityResult:
    """Complete veracity assessment"""
    source_quality: float = 0.0
    claim_grounding: float = 0.0
    numeric_accuracy: float = 0.0
    source_agreement: float = 0.0
    response_completeness: float = 0.0
    overall: float = 0.0
    claims_extracted: int = 0
    claims_verified: int = 0
    verifications: List[dict] = field(default_factory=list)
    unverified_claims: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

# =========================================================
# 4. PROMPTS
# =========================================================

#RESPONSE_TEMPLATE = '''
#{
#  "executive_summary": "4-6 sentence comprehensive answer with specific numbers",
#  "primary_metrics": {
#    "metric_1": {"name": "Key Metric", "value": 25.5, "unit": "%"}
#  },
#  "key_findings": ["Finding 1 with data", "Finding 2"],
#  "top_entities": [{"name": "Company", "share": "25%", "growth": "15%"}],
#  "trends_forecast": [{"trend": "Description", "direction": "↑", "timeline": "2025-2027"}],
#  "visualization_data": {
#    "chart_labels": ["2023", "2024", "2025"],
#    "chart_values": [100, 120, 145],
#    "chart_title": "Market Growth",
#    "chart_type": "line"
#  },
#  "sources": ["source1.com"],
#  "confidence": 85,
#  "freshness": "Dec 2024"
#}
#'''

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


# =========================================================
# 5. MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner="🔧 Loading AI models...")
def load_models():
    try:
        nli_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return nli_classifier, embedder
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

nli_classifier, embedder = load_models()

# =========================================================
# 6. SOURCE CLASSIFICATION (v7 style - fast)
# =========================================================

HIGH_QUALITY_DOMAINS = {
    "gov", "imf.org", "worldbank.org", "federalreserve.gov", "reuters.com",
    "spglobal.com", "economist.com", "mckinsey.com", "ft.com", "wsj.com",
    "bloomberg.com", "deloitte.com", "kpmg.com", "sec.gov", "nature.com",
    "oecd.org", "iea.org", "who.int", "un.org", "europa.eu"
}

MEDIUM_QUALITY_DOMAINS = {
    "wikipedia.org", "forbes.com", "cnbc.com", "statista.com", "investopedia.com",
    "marketwatch.com", "businessinsider.com", "yahoo.com"
}

LOW_QUALITY_DOMAINS = {"blog", "medium.com", "wordpress", "reddit.com", "quora.com"}

def classify_source_reliability(source: str) -> Tuple[str, int]:
    """Fast source classification - returns (label, score)"""
    source_lower = source.lower() if isinstance(source, str) else ""
    for domain in HIGH_QUALITY_DOMAINS:
        if domain in source_lower:
            return ("✅ High", 95)
    for domain in MEDIUM_QUALITY_DOMAINS:
        if domain in source_lower:
            return ("⚠️ Medium", 60)
    for domain in LOW_QUALITY_DOMAINS:
        if domain in source_lower:
            return ("❌ Low", 25)
    return ("⚠️ Unknown", 45)

def calculate_source_quality_fast(sources: List[str]) -> Tuple[float, dict]:
    """v7-style fast source quality calculation"""
    if not sources:
        return (0.0, {"count": 0, "high": 0, "medium": 0, "low": 0})

    scores = []
    high, medium, low = 0, 0, 0

    for src in sources:
        label, score = classify_source_reliability(src)
        scores.append(score)
        if "High" in label:
            high += 1
        elif "Medium" in label or "Unknown" in label:
            medium += 1
        else:
            low += 1

    avg_score = sum(scores) / len(scores)

    # Bonus for multiple high-quality sources
    if high >= 3:
        avg_score = min(100, avg_score + 10)
    elif high >= 2:
        avg_score = min(100, avg_score + 5)

    return (avg_score, {"count": len(sources), "high": high, "medium": medium, "low": low})

# =========================================================
# 7. JSON REPAIR FUNCTIONS
# =========================================================

def repair_llm_response(data: dict) -> dict:
    """Repair common LLM JSON structure issues"""
    if "primary_metrics" in data and isinstance(data["primary_metrics"], list):
        new_metrics = {}
        for i, item in enumerate(data["primary_metrics"]):
            if isinstance(item, dict):
                raw_name = item.get("name", f"metric_{i+1}")
                key = re.sub(r'[^a-z0-9_]', '', raw_name.lower().replace(" ", "_")) or f"metric_{i+1}"
                j = 1
                original_key = key
                while key in new_metrics:
                    key = f"{original_key}_{j}"
                    j += 1
                item.setdefault("name", raw_name)
                item.setdefault("value", "N/A")
                item.setdefault("unit", "")
                new_metrics[key] = item
        data["primary_metrics"] = new_metrics

    for field in ["top_entities", "trends_forecast"]:
        if field in data:
            if isinstance(data[field], dict):
                data[field] = list(data[field].values())
            elif not isinstance(data[field], list):
                data[field] = []

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
                                row[key] = float(re.sub(r'[^\d.-]', '', val) or 0)
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
# 8. WEB SEARCH FUNCTIONS
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def search_serpapi(query: str, num_results: int = 5) -> List[Dict]:
    if not SERPAPI_KEY:
        return []

    query_lower = query.lower()
    if any(kw in query_lower for kw in ["industry", "market", "sector", "size", "growth"]):
        search_terms = f"{query} market size growth trends 2024"
        tbm, tbs = "", ""
    else:
        search_terms = f"{query} finance economics data"
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
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "source": src.get("name", "") if isinstance(src, dict) else str(src)
            })

        if not results:
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date": "",
                    "source": item.get("source", "")
                })
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
        return '\n'.join(chunk for chunk in chunks if chunk)[:5000]
    except:
        return None

def fetch_web_context(query: str, num_sources: int = 3) -> Dict:
    search_results = search_serpapi(query, num_results=5)
    if not search_results:
        return {"search_results": [], "scraped_content": {}, "all_text": "", "sources": [], "source_scores": []}

    scraped_content = {}
    if SCRAPINGDOG_KEY:
        progress = st.progress(0)
        for i, result in enumerate(search_results[:num_sources]):
            content = scrape_url(result["link"])
            if content:
                scraped_content[result["link"]] = content
            progress.progress((i + 1) / num_sources)
        progress.empty()

    # Build combined text for verification
    all_text_parts = []
    source_scores = []

    for r in search_results:
        _, score = classify_source_reliability(r.get("link", "") + " " + r.get("source", ""))
        source_scores.append(score)
        all_text_parts.append(r.get("snippet", ""))

    for content in scraped_content.values():
        all_text_parts.append(content)

    return {
        "search_results": search_results,
        "scraped_content": scraped_content,
        "all_text": "\n\n".join(all_text_parts),
        "sources": [r["link"] for r in search_results],
        "source_scores": source_scores
    }

# =========================================================
# 9. CLAIM EXTRACTION (Targeted - v8 style but limited)
# =========================================================

def parse_number_with_context(text: str) -> List[Tuple[float, str, str]]:
    """Extract numbers with units and surrounding context"""
    results = []
    patterns = [
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(trillion|T)\b', 1_000_000, "T"),
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(billion|B)\b', 1_000, "B"),
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(million|M)\b', 1, "M"),
        (r'(\d+(?:\.\d+)?)\s*%', 1, "%"),
        (r'(\d+(?:\.\d+)?)\s*CAGR', 1, "CAGR"),
    ]

    for pattern, multiplier, unit_label in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = float(match.group(1)) * multiplier
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                context = text[start:end].strip()
                results.append((value, unit_label, context))
            except:
                continue
    return results

def extract_priority_claims(primary_data: dict, max_claims: int = 8) -> List[ExtractedClaim]:
    """
    Extract only the most important claims for verification.
    Prioritizes: numeric claims > entity claims > general claims
    """
    claims = []

    # Priority 1: Numeric claims from executive_summary (most important)
    summary = primary_data.get("executive_summary", "")
    if summary:
        numbers = parse_number_with_context(summary)
        for value, unit, context in numbers[:3]:  # Limit to top 3
            claims.append(ExtractedClaim(
                text=context, claim_type="numeric", numeric_value=value,
                numeric_unit=unit, source_field="executive_summary", priority=1
            ))

    # Priority 1: Numeric claims from primary_metrics
    for key, metric in list(primary_data.get("primary_metrics", {}).items())[:3]:
        if isinstance(metric, dict):
            name = metric.get("name", key)
            value = metric.get("value")
            unit = metric.get("unit", "")
            if value is not None:
                try:
                    numeric_val = float(str(value).replace(',', '').replace('$', ''))
                    claims.append(ExtractedClaim(
                        text=f"{name}: {value} {unit}".strip(), claim_type="numeric",
                        numeric_value=numeric_val, numeric_unit=unit,
                        source_field="primary_metrics", priority=1
                    ))
                except:
                    pass

    # Priority 2: Entity claims (market share)
    for entity in primary_data.get("top_entities", [])[:2]:
        if isinstance(entity, dict):
            name = entity.get("name", "")
            share = entity.get("share", "")
            if name and share and share != "N/A":
                claims.append(ExtractedClaim(
                    text=f"{name} market share {share}", claim_type="entity",
                    source_field="top_entities", priority=2
                ))

    # Priority 3: Key findings (only if we have room)
    if len(claims) < max_claims:
        for finding in primary_data.get("key_findings", [])[:2]:
            if isinstance(finding, str) and len(finding) > 20:
                # Check if it has numbers
                numbers = parse_number_with_context(finding)
                if numbers:
                    value, unit, _ = numbers[0]
                    claims.append(ExtractedClaim(
                        text=finding[:100], claim_type="numeric",
                        numeric_value=value, numeric_unit=unit,
                        source_field="key_findings", priority=2
                    ))
                elif len(claims) < max_claims - 1:
                    claims.append(ExtractedClaim(
                        text=finding[:100], claim_type="general",
                        source_field="key_findings", priority=3
                    ))

    # Sort by priority and return top N
    claims.sort(key=lambda c: c.priority)
    return claims[:max_claims]

# =========================================================
# 10. HYBRID CLAIM VERIFICATION
# =========================================================

@st.cache_data
def get_embedding(text: str):
    return embedder.encode(text[:512])

def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity (0.0-1.0)"""
    if not text1 or not text2:
        return 0.0
    try:
        v1 = get_embedding(text1)
        v2 = get_embedding(text2)
        return float(util.cos_sim(v1, v2).item())
    except:
        return 0.0

def quick_keyword_match(claim_text: str, source_text: str) -> float:
    """Fast keyword overlap check (v7 style)"""
    claim_words = set(re.findall(r'\b\w{4,}\b', claim_text.lower()))
    source_words = set(re.findall(r'\b\w{4,}\b', source_text.lower()))

    if not claim_words:
        return 0.0

    overlap = len(claim_words & source_words)
    return overlap / len(claim_words)

def verify_numeric_claim(claim: ExtractedClaim, source_text: str) -> Tuple[bool, float, str]:
    """
    Verify numeric claim with unit awareness (v8 style).
    Returns (is_match, confidence, matching_context)
    """
    if claim.numeric_value is None:
        return (False, 0.0, "")

    source_numbers = parse_number_with_context(source_text)
    if not source_numbers:
        return (False, 0.0, "")

    best_match = (False, 0.0, "")
    claim_unit = (claim.numeric_unit or "").upper()

    for source_value, source_unit, source_context in source_numbers:
        # Unit compatibility check
        source_unit_upper = source_unit.upper()
        unit_compatible = (
            claim_unit == source_unit_upper or
            (claim_unit in ["B", "BILLION"] and source_unit_upper in ["B", "BILLION"]) or
            (claim_unit in ["M", "MILLION"] and source_unit_upper in ["M", "MILLION"]) or
            (claim_unit == "%" and source_unit_upper in ["%", "PERCENT", "CAGR"]) or
            (claim_unit == "CAGR" and source_unit_upper in ["%", "CAGR"])
        )

        if not unit_compatible:
            continue

        # Calculate relative difference
        max_val = max(abs(claim.numeric_value), abs(source_value), 0.001)
        rel_diff = abs(claim.numeric_value - source_value) / max_val

        # Graduated confidence
        if rel_diff < 0.03:      # Within 3%
            confidence = 0.98
        elif rel_diff < 0.08:    # Within 8%
            confidence = 0.90
        elif rel_diff < 0.15:    # Within 15%
            confidence = 0.75
        elif rel_diff < 0.25:    # Within 25%
            confidence = 0.55
        else:
            confidence = 0.0

        if confidence > best_match[1]:
            best_match = (confidence >= 0.55, confidence, source_context)

    return best_match

def verify_claim_hybrid(claim: ExtractedClaim, web_context: dict) -> ClaimVerification:
    """
    Hybrid verification: tries fast methods first, then semantic if needed.
    """
    all_text = web_context.get("all_text", "")
    search_results = web_context.get("search_results", [])
    scraped_content = web_context.get("scraped_content", {})

    best = ClaimVerification(claim=claim, is_grounded=False, confidence=0.0, match_type="none")

    # Strategy 1: Numeric verification for numeric claims (fast and accurate)
    if claim.claim_type == "numeric" and claim.numeric_value is not None:
        # Check snippets first (faster)
        for result in search_results:
            is_match, conf, ctx = verify_numeric_claim(claim, result.get("snippet", ""))
            if is_match and conf > best.confidence:
                best = ClaimVerification(
                    claim=claim, is_grounded=True, confidence=conf,
                    supporting_source=result.get("link"), supporting_text=ctx[:150],
                    match_type="numeric"
                )
                if conf > 0.85:  # Good enough, stop searching
                    return best

        # Check scraped content if not found
        if not best.is_grounded:
            for url, content in scraped_content.items():
                is_match, conf, ctx = verify_numeric_claim(claim, content)
                if is_match and conf > best.confidence:
                    best = ClaimVerification(
                        claim=claim, is_grounded=True, confidence=conf,
                        supporting_source=url, supporting_text=ctx[:150],
                        match_type="numeric"
                    )
                    if conf > 0.85:
                        return best

    # Strategy 2: Quick keyword match (v7 style - fast)
    keyword_score = quick_keyword_match(claim.text, all_text)
    if keyword_score > 0.5 and keyword_score > best.confidence:
        best = ClaimVerification(
            claim=claim, is_grounded=keyword_score > 0.6,
            confidence=keyword_score * 0.8,  # Discount keyword matches
            supporting_text="Keyword match in sources",
            match_type="keyword"
        )

    # Strategy 3: Semantic similarity (v8 style - slower, use sparingly)
    # Only for non-numeric claims or if nothing found yet
    if not best.is_grounded and claim.claim_type != "numeric":
        # Check snippets only (faster than full content)
        for result in search_results:
            snippet = result.get("snippet", "")
            if len(snippet) > 30:
                sim = semantic_similarity(claim.text, snippet)
                if sim > 0.55 and sim > best.confidence:
                    best = ClaimVerification(
                        claim=claim, is_grounded=sim > 0.6,
                        confidence=sim,
                        supporting_source=result.get("link"),
                        supporting_text=snippet[:150],
                        match_type="semantic"
                    )

    return best

# =========================================================
# 11. SOURCE AGREEMENT CHECK (NEW in v9)
# =========================================================

def check_source_agreement(primary_data: dict, web_context: dict) -> Tuple[float, dict]:
    """
    Check if multiple sources agree on key facts.
    Extracts key numbers from LLM response and checks how many sources mention similar values.
    """
    # Extract key numbers from response
    summary = primary_data.get("executive_summary", "")
    response_numbers = parse_number_with_context(summary)

    if not response_numbers:
        return (50.0, {"message": "No numbers to verify agreement"})

    search_results = web_context.get("search_results", [])
    scraped_content = web_context.get("scraped_content", {})

    agreements = []

    for resp_value, resp_unit, resp_context in response_numbers[:3]:  # Top 3 numbers
        sources_agreeing = 0
        sources_checked = 0

        # Check each source
        for result in search_results:
            snippet = result.get("snippet", "")
            source_numbers = parse_number_with_context(snippet)
            sources_checked += 1

            for src_value, src_unit, _ in source_numbers:
                if resp_unit.upper() == src_unit.upper():
                    max_val = max(abs(resp_value), abs(src_value), 0.001)
                    if abs(resp_value - src_value) / max_val < 0.20:
                        sources_agreeing += 1
                        break

        if sources_checked > 0:
            agreements.append(sources_agreeing / sources_checked)

    if not agreements:
        return (50.0, {"message": "Could not check agreement"})

    avg_agreement = sum(agreements) / len(agreements)
    score = avg_agreement * 100

    return (min(score, 95.0), {"numbers_checked": len(agreements), "avg_agreement": avg_agreement})

# =========================================================
# 12. RESPONSE COMPLETENESS CHECK (NEW in v9)
# =========================================================

def check_response_completeness(primary_data: dict) -> Tuple[float, dict]:
    """
    Check if LLM response has all required fields with substantive content.
    """
    checks = {
        "executive_summary": len(primary_data.get("executive_summary", "")) > 100,
        "primary_metrics": len(primary_data.get("primary_metrics", {})) >= 2,
        "key_findings": len(primary_data.get("key_findings", [])) >= 2,
        "top_entities": len(primary_data.get("top_entities", [])) >= 1,
        "trends_forecast": len(primary_data.get("trends_forecast", [])) >= 1,
        "visualization_data": bool(primary_data.get("visualization_data", {}).get("chart_values")),
        "sources": len(primary_data.get("sources", [])) >= 1,
    }

    passed = sum(checks.values())
    total = len(checks)
    score = (passed / total) * 100

    return (score, {"passed": passed, "total": total, "checks": checks})

# =========================================================
# 13. HYBRID VERACITY SCORING
# =========================================================

def calculate_veracity_hybrid(primary_data: dict, web_context: dict) -> VeracityResult:
    """
    Hybrid veracity scoring combining v7 speed with v8 accuracy.

    Components:
    1. Source Quality (20%): v7-style fast domain checking
    2. Claim Grounding (30%): v8-style but limited to priority claims
    3. Numeric Accuracy (25%): v8-style with unit awareness
    4. Source Agreement (15%): NEW - do multiple sources agree?
    5. Response Completeness (10%): NEW - is response well-formed?
    """
    result = VeracityResult()

    # 1. SOURCE QUALITY (20%) - v7 style, fast
    sources = primary_data.get("sources", []) + web_context.get("sources", [])
    sources = list(dict.fromkeys(sources))  # Dedupe
    sq_score, sq_details = calculate_source_quality_fast(sources)
    result.source_quality = sq_score
    result.details["source_quality"] = sq_details

    # 2 & 3. CLAIM GROUNDING + NUMERIC ACCURACY (30% + 25%) - v8 style but targeted
    claims = extract_priority_claims(primary_data, max_claims=8)
    result.claims_extracted = len(claims)

    verifications = []
    numeric_verifications = []

    for claim in claims:
        v = verify_claim_hybrid(claim, web_context)
        verifications.append(v)
        if claim.claim_type == "numeric":
            numeric_verifications.append(v)

    # Claim grounding score
    if verifications:
        grounded = sum(1 for v in verifications if v.is_grounded)
        avg_conf = sum(v.confidence for v in verifications) / len(verifications)
        result.claims_verified = grounded
        result.claim_grounding = (grounded / len(verifications) * 0.6 + avg_conf * 0.4) * 100
        result.unverified_claims = [v.claim.text[:60] for v in verifications if not v.is_grounded]
    else:
        result.claim_grounding = 50.0  # Neutral if no claims

    # Numeric accuracy score
    if numeric_verifications:
        num_grounded = sum(1 for v in numeric_verifications if v.is_grounded and v.match_type == "numeric")
        num_avg_conf = sum(v.confidence for v in numeric_verifications if v.match_type == "numeric")
        num_avg_conf = num_avg_conf / len(numeric_verifications) if numeric_verifications else 0
        result.numeric_accuracy = (num_grounded / len(numeric_verifications) * 0.5 + num_avg_conf * 0.5) * 100
    else:
        result.numeric_accuracy = 50.0  # Neutral if no numeric claims

    result.details["claim_grounding"] = {
        "total": len(verifications),
        "grounded": result.claims_verified,
        "unverified": result.unverified_claims[:3]
    }

    # Store verifications for display
    result.verifications = [
        {
            "claim": v.claim.text[:80],
            "type": v.claim.claim_type,
            "grounded": v.is_grounded,
            "confidence": round(v.confidence, 2),
            "match_type": v.match_type,
            "source": (v.supporting_source[:40] + "...") if v.supporting_source else None
        }
        for v in verifications
    ]

    # 4. SOURCE AGREEMENT (15%) - NEW
    sa_score, sa_details = check_source_agreement(primary_data, web_context)
    result.source_agreement = sa_score
    result.details["source_agreement"] = sa_details

    # 5. RESPONSE COMPLETENESS (10%) - NEW
    rc_score, rc_details = check_response_completeness(primary_data)
    result.response_completeness = rc_score
    result.details["response_completeness"] = rc_details

    # OVERALL WEIGHTED SCORE
    result.overall = (
        result.source_quality * 0.20 +
        result.claim_grounding * 0.30 +
        result.numeric_accuracy * 0.25 +
        result.source_agreement * 0.15 +
        result.response_completeness * 0.10
    )

    return result

# =========================================================
# 14. FINAL CONFIDENCE CALCULATION
# =========================================================

def calculate_final_confidence(base_conf: float, evidence_score: float) -> float:
    """
    Calculate final confidence balancing model and evidence.

    Uses v8's evidence multiplier range but v7's simpler formula.

    - High model + High evidence → ~85-90%
    - High model + Low evidence → ~35-45% (heavy penalty)
    - Low model + High evidence → ~70-75%
    - Low model + Low evidence → ~25-35%
    """
    base_conf = max(0, min(100, base_conf))
    evidence_score = max(0, min(100, evidence_score))

    # Evidence component (65% weight)
    evidence_component = evidence_score * 0.65

    # Model component (35% weight) - discounted by weak evidence
    # Multiplier: 0.25 at 0% evidence, 1.0 at 100% evidence
    evidence_multiplier = 0.25 + (evidence_score / 100) * 0.75
    model_component = base_conf * evidence_multiplier * 0.35

    return round(max(0, min(100, evidence_component + model_component)), 1)

# =========================================================
# 15. LLM QUERY FUNCTIONS
# =========================================================

def query_perplexity(query: str, web_context: Dict, temperature: float = 0.1) -> str:
    search_count = len(web_context.get("search_results", []))

    if not web_context.get("all_text") or search_count < 2:
        enhanced_query = f"{SYSTEM_PROMPT}\n\nUser Question: {query}\n\nProvide comprehensive analysis."
    else:
        # Build context from search results
        context_parts = []
        for r in web_context.get("search_results", [])[:3]:
            context_parts.append(f"Source: {r.get('source', 'Unknown')}\n{r.get('snippet', '')}")

        # Add scraped content excerpts
        for url, content in list(web_context.get("scraped_content", {}).items())[:2]:
            context_parts.append(f"From {url}:\n{content[:600]}")

        context_section = "\n\n---\n\n".join(context_parts)
        enhanced_query = f"WEB RESEARCH:\n{context_section}\n\n{SYSTEM_PROMPT}\n\nQuestion: {query}"

    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar", "temperature": temperature, "max_tokens": 2000, "top_p": 0.8,
        "messages": [{"role": "user", "content": enhanced_query}]
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data:
            raise Exception("No choices")

        content = data["choices"][0]["message"]["content"]
        if not content:
            raise Exception("Empty response")

        parsed = parse_json_safely(content, "Perplexity")
        if not parsed:
            return create_fallback_response(query, search_count, web_context)

        repaired = repair_llm_response(parsed)

        try:
            llm_obj = LLMResponse.model_validate(repaired)
            if web_context.get("sources"):
                existing = llm_obj.sources or []
                llm_obj.sources = list(dict.fromkeys(existing + web_context["sources"]))[:10]
                llm_obj.freshness = "Current (web-enhanced)"
            return llm_obj.model_dump_json()
        except ValidationError:
            return create_fallback_response(query, search_count, web_context)
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return create_fallback_response(query, search_count, web_context)

def create_fallback_response(query: str, search_count: int, web_context: Dict) -> str:
    fallback = LLMResponse(
        executive_summary=f"Analysis of '{query}' with {search_count} sources. Using fallback structure.",
        primary_metrics={"sources": MetricDetail(name="Sources", value=search_count, unit="")},
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
# 16. DASHBOARD RENDERING
# =========================================================

def detect_axis_labels(labels: list, values: list) -> Tuple[str, str]:
    """Detect appropriate axis labels"""
    x_label = "Category"
    y_label = "Value"

    if labels:
        label_text = ' '.join(str(l).lower() for l in labels)
        if any(k in label_text for k in ['america', 'europe', 'asia', 'china', 'africa']):
            x_label = "Region"
        elif re.search(r'\b20\d{2}\b', label_text):
            x_label = "Year"
        elif re.search(r'\bq[1-4]\b', label_text, re.I):
            x_label = "Quarter"

    if values:
        try:
            nums = [abs(float(v)) for v in values]
            avg = sum(nums) / len(nums)
            if avg > 100:
                y_label = "USD Billions"
            elif avg > 1:
                y_label = "USD Millions"
            elif all(0 <= n <= 100 for n in nums):
                y_label = "Percent"
        except:
            pass

    return x_label, y_label

def render_dashboard(primary_json: str, final_conf: float, web_context: Dict, base_conf: float,
                    user_question: str, veracity: VeracityResult):
    try:
        data = json.loads(primary_json)
    except Exception as e:
        st.error(f"❌ Cannot render: {e}")
        return

    st.header("📊 Yureeka Market Report")
    st.markdown(f"**Question:** {user_question}")

    # Confidence metrics
    cols = st.columns(4)
    cols[0].metric("Final Confidence", f"{final_conf:.0f}%",
                   help="Combined score from model confidence and evidence quality")
    cols[1].metric("Model Confidence", f"{base_conf:.0f}%",
                   help="LLM's self-reported confidence")
    cols[2].metric("Evidence Score", f"{veracity.overall:.0f}%",
                   help="Quality of supporting evidence")
    cols[3].metric("Claims Verified", f"{veracity.claims_verified}/{veracity.claims_extracted}",
                   help="Claims found in sources")

    st.markdown("---")

    # Executive Summary
    st.subheader("📋 Executive Summary")
    st.markdown(f"**{data.get('executive_summary', 'No summary')}**")
    st.markdown("---")

    # Metrics
    st.subheader("💰 Key Metrics")
    metrics = data.get('primary_metrics', {})
    if metrics:
        rows = [{"Metric": m.get("name", k), "Value": f"{m.get('value', 'N/A')} {m.get('unit', '')}".strip()}
                for k, m in list(metrics.items())[:6] if isinstance(m, dict)]
        if rows:
            st.table(pd.DataFrame(rows))
    st.markdown("---")

    # Findings
    st.subheader("🔍 Key Findings")
    for i, f in enumerate(data.get('key_findings', [])[:6], 1):
        if f:
            st.markdown(f"**{i}.** {f}")
    st.markdown("---")

    # Entities & Trends in columns
    col1, col2 = st.columns(2)

    with col1:
        entities = data.get('top_entities', [])
        if entities:
            st.subheader("🏢 Top Players")
            entity_df = pd.DataFrame([
                {"Entity": e.get("name", "N/A"), "Share": e.get("share", "N/A"), "Growth": e.get("growth", "N/A")}
                for e in entities if isinstance(e, dict)
            ])
            if not entity_df.empty:
                st.dataframe(entity_df, hide_index=True, use_container_width=True)

    with col2:
        trends = data.get('trends_forecast', [])
        if trends:
            st.subheader("📈 Trends")
            trend_df = pd.DataFrame([
                {"Trend": t.get("trend", "N/A")[:50], "Direction": t.get("direction", "→")}
                for t in trends if isinstance(t, dict)
            ])
            if not trend_df.empty:
                st.dataframe(trend_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Visualization
    viz = data.get('visualization_data')
    if viz and isinstance(viz, dict):
        labels = viz.get("chart_labels", [])
        values = viz.get("chart_values", [])
        if labels and values and len(labels) == len(values):
            st.subheader("📊 Visualization")
            try:
                nums = [float(v) for v in values[:10]]
                x_label, y_label = detect_axis_labels(labels, nums)
                df = pd.DataFrame({"x": labels[:10], "y": nums})

                chart_type = viz.get("chart_type", "line")
                if chart_type == "bar":
                    fig = px.bar(df, x="x", y="y", title=viz.get("chart_title", "Analysis"))
                else:
                    fig = px.line(df, x="x", y="y", title=viz.get("chart_title", "Analysis"), markers=True)

                fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass

    st.markdown("---")

    # Sources
    st.subheader("🔗 Sources")
    sources = list(dict.fromkeys(data.get('sources', []) + web_context.get('sources', [])))[:8]
    if sources:
        cols = st.columns(2)
        for i, src in enumerate(sources):
            label, _ = classify_source_reliability(src)
            cols[i % 2].markdown(f"{i+1}. [{src[:45]}...]({src}) {label}")

    st.markdown("---")

    # Evidence Quality Breakdown
    st.subheader("✅ Evidence Quality")
    cols = st.columns(5)
    metrics_info = [
        ("Sources", veracity.source_quality, "Quality of source domains"),
        ("Grounding", veracity.claim_grounding, "Claims verified in sources"),
        ("Numeric", veracity.numeric_accuracy, "Numbers match sources"),
        ("Agreement", veracity.source_agreement, "Multiple sources agree"),
        ("Complete", veracity.response_completeness, "Response has all fields"),
    ]
    for i, (name, score, tooltip) in enumerate(metrics_info):
        cols[i].metric(name, f"{score:.0f}%", help=tooltip)

    # Claim verification details
    with st.expander("🔬 Claim Verification Details"):
        if veracity.verifications:
            for v in veracity.verifications:
                icon = "✅" if v["grounded"] else "❌"
                conf_str = f"{v['confidence']:.0%}" if v['confidence'] else "0%"
                st.markdown(f"{icon} **{v['claim'][:70]}...**")
                st.caption(f"Type: {v['type']} | Match: {v['match_type']} | Confidence: {conf_str}")
                if v.get("source"):
                    st.caption(f"Source: {v['source']}")
                st.markdown("---")

        if veracity.unverified_claims:
            st.warning("**Unverified claims:**")
            for claim in veracity.unverified_claims[:3]:
                st.markdown(f"- ❓ {claim}")

    # Web search details
    if web_context.get("search_results"):
        with st.expander("🌐 Web Search Results"):
            for i, r in enumerate(web_context["search_results"][:5], 1):
                st.markdown(f"**{i}. {r.get('title', 'No title')}**")
                st.caption(f"{r.get('source', 'Unknown')} | {r.get('date', '')}")
                st.write(r.get('snippet', '')[:200])
                st.markdown("---")

# =========================================================
# 17. MAIN APPLICATION
# =========================================================

def main():
    st.set_page_config(page_title="Yureeka Market Intelligence", page_icon="💹", layout="wide")
    st.title("💹 Yureeka Market Intelligence")

    st.markdown("""
    **Yureeka v9** - Hybrid verification combining fast source scoring with targeted claim verification.

    *Features: Priority claim extraction, unit-aware numeric matching, source agreement checking*
    """)

    query = st.text_input(
        "Enter your market research question:",
        placeholder="e.g., What is the global electric vehicle market size and growth forecast?"
    )

    col1, col2 = st.columns(2)
    with col1:
        use_web = st.checkbox("Enable web search", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY)
    with col2:
        show_debug = st.checkbox("Show debug info", value=False)

    if st.button("🔍 Analyze", type="primary") and query:
        if len(query.strip()) < 5:
            st.error("Please enter a longer question")
            return

        query = query.strip()[:500]

        # Web search
        web_context = {"search_results": [], "scraped_content": {}, "all_text": "", "sources": [], "source_scores": []}
        if use_web:
            with st.spinner("🌐 Searching web..."):
                web_context = fetch_web_context(query, num_sources=3)

            if web_context.get("search_results"):
                st.success(f"Found {len(web_context['search_results'])} sources")
            else:
                st.info("No web results, using AI knowledge only")

        # Query LLM
        with st.spinner("🤖 Analyzing..."):
            response = query_perplexity(query, web_context)

        if not response:
            st.error("Analysis failed")
            return

        try:
            primary_data = json.loads(response)
        except:
            st.error("Failed to parse response")
            return

        # Veracity scoring
        with st.spinner("✅ Verifying claims..."):
            veracity = calculate_veracity_hybrid(primary_data, web_context)

        # Confidence
        base_conf = float(primary_data.get("confidence", 75))
        final_conf = calculate_final_confidence(base_conf, veracity.overall)

        # Download button
        output = {
            "question": query,
            "timestamp": datetime.now().isoformat(),
            "response": primary_data,
            "confidence": {"base": base_conf, "evidence": veracity.overall, "final": final_conf},
            "veracity": {
                "source_quality": veracity.source_quality,
                "claim_grounding": veracity.claim_grounding,
                "numeric_accuracy": veracity.numeric_accuracy,
                "source_agreement": veracity.source_agreement,
                "response_completeness": veracity.response_completeness,
                "overall": veracity.overall,
                "claims_extracted": veracity.claims_extracted,
                "claims_verified": veracity.claims_verified,
                "verifications": veracity.verifications
            },
            "sources": web_context.get("sources", [])
        }

        st.download_button(
            "💾 Download Report",
            json.dumps(output, indent=2),
            f"yureeka_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )

        # Render dashboard
        render_dashboard(response, final_conf, web_context, base_conf, query, veracity)

        # Debug info
        if show_debug:
            with st.expander("🔧 Debug Information", expanded=True):
                st.subheader("Confidence Calculation")
                evidence_mult = 0.25 + (veracity.overall / 100) * 0.75
                st.json({
                    "model_confidence": base_conf,
                    "evidence_score": round(veracity.overall, 1),
                    "evidence_multiplier": round(evidence_mult, 2),
                    "evidence_component": round(veracity.overall * 0.65, 1),
                    "model_component": round(base_conf * evidence_mult * 0.35, 1),
                    "final_confidence": final_conf
                })

                st.subheader("Veracity Breakdown")
                st.json({
                    "source_quality (20%)": round(veracity.source_quality, 1),
                    "claim_grounding (30%)": round(veracity.claim_grounding, 1),
                    "numeric_accuracy (25%)": round(veracity.numeric_accuracy, 1),
                    "source_agreement (15%)": round(veracity.source_agreement, 1),
                    "response_completeness (10%)": round(veracity.response_completeness, 1),
                    "overall": round(veracity.overall, 1)
                })

                st.subheader("Claims Extracted")
                st.write(f"Total: {veracity.claims_extracted}, Verified: {veracity.claims_verified}")

                st.subheader("Raw Response")
                st.json(primary_data)

if __name__ == "__main__":
    main()
