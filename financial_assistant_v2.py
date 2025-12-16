# =========================================================
# YUREEKA AI RESEARCH ASSISTANT v8.0 - CLAIM-LEVEL VERIFICATION
# With Web Search, Semantic Grounding, Evidence-Based Scoring
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
from dataclasses import dataclass

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
    claim_type: str  # 'numeric', 'entity', 'trend', 'general'
    numeric_value: Optional[float] = None
    numeric_unit: Optional[str] = None
    source_field: str = ""

@dataclass
class ClaimVerification:
    """Result of verifying a single claim"""
    claim: ExtractedClaim
    is_grounded: bool
    confidence: float
    supporting_source: Optional[str] = None
    supporting_text: Optional[str] = None
    match_type: str = "none"

# =========================================================
# 4. PROMPTS
# =========================================================

RESPONSE_TEMPLATE = '''
{
  "executive_summary": "4-6 sentence comprehensive answer",
  "primary_metrics": {
    "metric_1": {"name": "Key Metric 1", "value": 25.5, "unit": "%"}
  },
  "key_findings": ["Finding 1", "Finding 2"],
  "top_entities": [{"name": "Entity 1", "share": "25%", "growth": "15%"}],
  "trends_forecast": [{"trend": "Description", "direction": "↑", "timeline": "2025-2027"}],
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
'''

SYSTEM_PROMPT = f"""You are a professional market research analyst.

CRITICAL RULES:
1. Return ONLY valid JSON. NO markdown, NO code blocks, NO extra text.
2. NO citation references like [1][2] inside strings.
3. Use double quotes for all keys and string values.
4. NO trailing commas.
5. Escape internal quotes with backslash.

REQUIRED FIELDS:
- executive_summary: 4-6 sentences with quantitative data
- primary_metrics: 3+ metrics with numbers
- key_findings: 3+ findings with details
- top_entities: 3+ companies/countries
- trends_forecast: 2+ trends with timelines
- visualization_data: chart_labels and chart_values

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
# 6. SOURCE CLASSIFICATION
# =========================================================


HIGH_QUALITY_DOMAINS = {
    "gov", "imf", "worldbank", "central bank", "fed", "ecb", "reuters", "spglobal", "economist", "mckinsey", "bcg", "cognitive market research", 
    "financial times", "wsj", "oecd", "bloomberg", "tradingeconomics", "deloitte", "hsbc", "imarc", "booz", "bakerinstitute.org",
    "kpmg", "semiconductors.org", "eu", "iea", "world bank", "opec", "jp morgan", "citibank", "goldman sachs", "j.p. morgan"
}

MEDIUM_QUALITY_DOMAINS = {
    "wikipedia.org", "forbes.com", "cnbc.com", "statista.com", "investopedia.com"
}

LOW_QUALITY_DOMAINS = {"blog", "medium.com", "wordpress", "reddit.com", "quora.com"}

def classify_source_reliability(source: str) -> Tuple[str, int]:
    source_lower = source.lower() if isinstance(source, str) else ""
    for domain in HIGH_QUALITY_DOMAINS:
        if domain in source_lower:
            return ("✅ High", 100)
    for domain in MEDIUM_QUALITY_DOMAINS:
        if domain in source_lower:
            return ("⚠️ Medium", 60)
    for domain in LOW_QUALITY_DOMAINS:
        if domain in source_lower:
            return ("❌ Low", 25)
    return ("⚠️ Medium", 50)

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
                        val_upper = val.upper().strip()
                        if val_upper in ["N/A", "NA", "NULL", "NONE", "", "-"]:
                            row[key] = 0
                        else:
                            try:
                                cleaned = re.sub(r'[^\d.-]', '', val)
                                row[key] = float(cleaned) if cleaned else 0
                            except:
                                row[key] = 0
    return data

def preclean_json(raw: str) -> str:
    if not raw or not isinstance(raw, str):
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
            if "Unterminated string" not in e.msg:
                break
            for i in range(e.pos - 1, max(0, e.pos - 150), -1):
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
        search_terms = f"{query} finance economics"
        tbm, tbs = "nws", "qdr:m"
    
    params = {"engine": "google", "q": search_terms, "api_key": SERPAPI_KEY, "num": num_results, "tbm": tbm, "tbs": tbs}
    
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
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
        return {"search_results": [], "scraped_content": {}, "summary": "", "sources": [], "source_reliability": [], "source_scores": []}
    
    scraped_content = {}
    if SCRAPINGDOG_KEY:
        progress = st.progress(0)
        for i, result in enumerate(search_results[:num_sources]):
            content = scrape_url(result["link"])
            if content:
                scraped_content[result["link"]] = content
            progress.progress((i + 1) / num_sources)
        progress.empty()
    
    context_parts, reliabilities, source_scores = [], [], []
    for r in search_results:
        reliability_label, reliability_score = classify_source_reliability(r.get("link", "") + " " + r.get("source", ""))
        reliabilities.append(reliability_label)
        source_scores.append(reliability_score)
        context_parts.append(f"**{r['title']}**\nSource: {r['source']} [{reliability_label}]\n{r['snippet']}\nURL: {r['link']}")
    
    return {
        "search_results": search_results,
        "scraped_content": scraped_content,
        "summary": "\n\n---\n\n".join(context_parts),
        "sources": [r["link"] for r in search_results],
        "source_reliability": reliabilities,
        "source_scores": source_scores
    }

# =========================================================
# 9. CLAIM EXTRACTION
# =========================================================

def parse_number_with_context(text: str) -> List[Tuple[float, str, str]]:
    """Extract numbers with units and context"""
    results = []
    patterns = [
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(trillion|T)\b', 1_000_000),
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(billion|B)\b', 1_000),
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(million|M)\b', 1),
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(thousand|K)\b', 0.001),
        (r'(\d+(?:\.\d+)?)\s*%', 1),
    ]
    
    for pattern, multiplier in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = float(match.group(1)) * multiplier
                unit = match.group(2) if len(match.groups()) > 1 else "%"
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                results.append((value, unit, context))
            except:
                continue
    return results

def extract_claims_from_response(primary_data: dict) -> List[ExtractedClaim]:
    """Extract verifiable claims from LLM response"""
    claims = []
    
    # From executive_summary
    summary = primary_data.get("executive_summary", "")
    if summary:
        numbers = parse_number_with_context(summary)
        for value, unit, context in numbers:
            claims.append(ExtractedClaim(text=context, claim_type="numeric", numeric_value=value, 
                                        numeric_unit=unit, source_field="executive_summary"))
        sentences = [s.strip() for s in summary.split('.') if len(s.strip()) > 20]
        for sentence in sentences[:5]:
            if not any(c.text in sentence for c in claims):
                claims.append(ExtractedClaim(text=sentence, claim_type="general", source_field="executive_summary"))
    
    # From key_findings
    for finding in primary_data.get("key_findings", [])[:5]:
        if isinstance(finding, str) and len(finding) > 10:
            numbers = parse_number_with_context(finding)
            if numbers:
                for value, unit, context in numbers:
                    claims.append(ExtractedClaim(text=finding, claim_type="numeric", numeric_value=value,
                                                numeric_unit=unit, source_field="key_findings"))
            else:
                claims.append(ExtractedClaim(text=finding, claim_type="general", source_field="key_findings"))
    
    # From primary_metrics
    for key, metric in primary_data.get("primary_metrics", {}).items():
        if isinstance(metric, dict):
            name = metric.get("name", key)
            value = metric.get("value")
            unit = metric.get("unit", "")
            if value is not None:
                try:
                    numeric_val = float(str(value).replace(',', '').replace('$', ''))
                    claims.append(ExtractedClaim(text=f"{name}: {value} {unit}".strip(), claim_type="numeric",
                                                numeric_value=numeric_val, numeric_unit=unit, source_field="primary_metrics"))
                except:
                    claims.append(ExtractedClaim(text=f"{name}: {value} {unit}".strip(), claim_type="general",
                                                source_field="primary_metrics"))
    
    # From top_entities
    for entity in primary_data.get("top_entities", [])[:5]:
        if isinstance(entity, dict):
            name = entity.get("name", "")
            share = entity.get("share", "")
            if name and share:
                claims.append(ExtractedClaim(text=f"{name} has {share} market share", claim_type="entity",
                                            source_field="top_entities"))
    
    return claims

# =========================================================
# 10. CLAIM VERIFICATION (CORE LOGIC)
# =========================================================

@st.cache_data
def get_embedding(text: str):
    return embedder.encode(text)

def semantic_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    try:
        v1 = get_embedding(text1[:512])
        v2 = get_embedding(text2[:512])
        return float(util.cos_sim(v1, v2).item())
    except:
        return 0.0

def check_numeric_match(claim_value: float, claim_unit: str, source_text: str) -> Tuple[bool, float, str]:
    """Check if numeric claim matches source numbers with context awareness"""
    source_numbers = parse_number_with_context(source_text)
    if not source_numbers:
        return (False, 0.0, "")
    
    best_match = (False, 0.0, "")
    
    for source_value, source_unit, source_context in source_numbers:
        # Check unit compatibility
        unit_compatible = (
            claim_unit.lower() in source_unit.lower() or
            source_unit.lower() in claim_unit.lower() or
            (claim_unit == "%" and source_unit == "%") or
            (claim_unit in ["B", "billion"] and source_unit in ["B", "billion"]) or
            (claim_unit in ["M", "million"] and source_unit in ["M", "million"])
        )
        
        if not unit_compatible:
            continue
        
        # Calculate relative difference
        if max(claim_value, source_value) > 0:
            rel_diff = abs(claim_value - source_value) / max(claim_value, source_value)
        else:
            rel_diff = 1.0
        
        if rel_diff < 0.05:
            confidence = 0.95
        elif rel_diff < 0.10:
            confidence = 0.85
        elif rel_diff < 0.20:
            confidence = 0.70
        elif rel_diff < 0.30:
            confidence = 0.50
        else:
            confidence = 0.0
        
        if confidence > best_match[1]:
            best_match = (confidence > 0.5, confidence, source_context)
    
    return best_match

def verify_claim_against_sources(claim: ExtractedClaim, web_context: dict) -> ClaimVerification:
    """Verify a single claim against web sources"""
    best_verification = ClaimVerification(claim=claim, is_grounded=False, confidence=0.0, match_type="none")
    
    source_texts = []
    for result in web_context.get("search_results", []):
        if result.get("snippet"):
            source_texts.append((result["snippet"], result.get("link", ""), "snippet"))
    
    for url, content in web_context.get("scraped_content", {}).items():
        if content:
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
            for para in paragraphs[:20]:
                source_texts.append((para, url, "scraped"))
    
    if not source_texts:
        return best_verification
    
    # Numeric matching for numeric claims
    if claim.claim_type == "numeric" and claim.numeric_value is not None:
        for source_text, source_url, _ in source_texts:
            is_match, confidence, context = check_numeric_match(claim.numeric_value, claim.numeric_unit or "", source_text)
            if is_match and confidence > best_verification.confidence:
                best_verification = ClaimVerification(claim=claim, is_grounded=True, confidence=confidence,
                                                     supporting_source=source_url, supporting_text=context[:200], match_type="numeric")
    
    # Semantic matching for all claims
    for source_text, source_url, _ in source_texts:
        similarity = semantic_similarity(claim.text, source_text)
        if similarity > 0.6 and similarity > best_verification.confidence:
            best_verification = ClaimVerification(claim=claim, is_grounded=True, confidence=similarity,
                                                 supporting_source=source_url, supporting_text=source_text[:200], match_type="semantic")
    
    # Partial credit for topic mention
    if not best_verification.is_grounded:
        claim_keywords = set(claim.text.lower().split())
        for source_text, source_url, _ in source_texts:
            source_words = set(source_text.lower().split())
            overlap = len(claim_keywords & source_words) / max(len(claim_keywords), 1)
            if overlap > 0.3:
                best_verification = ClaimVerification(claim=claim, is_grounded=False, confidence=overlap * 0.4,
                                                     supporting_source=source_url, supporting_text="Topic mentioned", match_type="partial")
                break
    
    return best_verification

# =========================================================
# 11. EVIDENCE-BASED VERACITY SCORING (REDESIGNED)
# =========================================================

def calculate_source_quality_score(web_context: dict) -> Tuple[float, dict]:
    """Calculate source quality based on actual sources used"""
    source_scores = web_context.get("source_scores", [])
    if not source_scores:
        return (0.0, {"message": "No sources available", "count": 0})
    
    avg_score = sum(source_scores) / len(source_scores)
    high_count = sum(1 for s in source_scores if s >= 80)
    
    return (avg_score, {"average_quality": avg_score, "source_count": len(source_scores), "high_quality_count": high_count})

def calculate_claim_grounding_score(verifications: List[ClaimVerification]) -> Tuple[float, dict]:
    """Calculate what percentage of claims are grounded in sources"""
    if not verifications:
        return (0.0, {"message": "No claims to verify", "total": 0})
    
    grounded_count = sum(1 for v in verifications if v.is_grounded)
    total_confidence = sum(v.confidence for v in verifications)
    grounding_rate = grounded_count / len(verifications)
    avg_confidence = total_confidence / len(verifications)
    
    score = (grounding_rate * 0.6 + avg_confidence * 0.4) * 100
    
    return (score, {
        "total_claims": len(verifications),
        "grounded_claims": grounded_count,
        "grounding_rate": grounding_rate,
        "average_confidence": avg_confidence,
        "ungrounded": [v.claim.text[:50] for v in verifications if not v.is_grounded][:3]
    })

def calculate_numeric_accuracy_score(verifications: List[ClaimVerification]) -> Tuple[float, dict]:
    """Calculate accuracy of numeric claims specifically"""
    numeric_verifications = [v for v in verifications if v.claim.claim_type == "numeric"]
    
    if not numeric_verifications:
        return (50.0, {"message": "No numeric claims to verify", "count": 0})
    
    verified_count = sum(1 for v in numeric_verifications if v.is_grounded and v.match_type == "numeric")
    total_confidence = sum(v.confidence for v in numeric_verifications if v.match_type == "numeric")
    
    if verified_count == 0:
        return (30.0, {"total": len(numeric_verifications), "verified": 0})
    
    accuracy_rate = verified_count / len(numeric_verifications)
    avg_confidence = total_confidence / verified_count
    score = (accuracy_rate * 0.5 + avg_confidence * 0.5) * 100
    
    return (score, {"total_numeric_claims": len(numeric_verifications), "verified_claims": verified_count,
                   "accuracy_rate": accuracy_rate, "average_match_confidence": avg_confidence})

def calculate_source_coverage_score(verifications: List[ClaimVerification], web_context: dict) -> Tuple[float, dict]:
    """Calculate how well claims are distributed across sources"""
    if not verifications:
        return (0.0, {"message": "No verifications"})
    
    sources_used = set(v.supporting_source for v in verifications if v.supporting_source)
    total_sources = len(web_context.get("sources", []))
    grounded_verifications = [v for v in verifications if v.is_grounded]
    
    if total_sources == 0:
        return (0.0, {"message": "No sources available"})
    if len(sources_used) == 0:
        return (20.0, {"message": "No claims linked to sources"})
    
    coverage = len(sources_used) / total_sources
    
    if len(grounded_verifications) > 0:
        source_counts = Counter(v.supporting_source for v in grounded_verifications if v.supporting_source)
        max_from_one = max(source_counts.values()) if source_counts else 0
        distribution = 1 - (max_from_one / len(grounded_verifications))
    else:
        distribution = 0
    
    score = (coverage * 0.5 + distribution * 0.5) * 100
    
    return (min(score, 95.0), {"sources_available": total_sources, "sources_supporting_claims": len(sources_used),
                              "coverage_rate": coverage, "distribution_score": distribution})

def evidence_based_veracity(primary_data: dict, web_context: dict) -> dict:
    """
    Comprehensive evidence-based veracity scoring.
    
    Components:
    1. Source Quality (25%): Quality of web sources found
    2. Claim Grounding (35%): Are LLM claims supported by sources?
    3. Numeric Accuracy (25%): Do numbers match source numbers?
    4. Source Coverage (15%): Are claims spread across multiple sources?
    """
    
    claims = extract_claims_from_response(primary_data)
    verifications = [verify_claim_against_sources(c, web_context) for c in claims]
    
    source_quality, sq_details = calculate_source_quality_score(web_context)
    claim_grounding, cg_details = calculate_claim_grounding_score(verifications)
    numeric_accuracy, na_details = calculate_numeric_accuracy_score(verifications)
    source_coverage, sc_details = calculate_source_coverage_score(verifications, web_context)
    
    overall = (source_quality * 0.25 + claim_grounding * 0.35 + numeric_accuracy * 0.25 + source_coverage * 0.15)
    
    return {
        "source_quality": round(source_quality, 1),
        "claim_grounding": round(claim_grounding, 1),
        "numeric_accuracy": round(numeric_accuracy, 1),
        "source_coverage": round(source_coverage, 1),
        "overall": round(overall, 1),
        "details": {
            "source_quality": sq_details,
            "claim_grounding": cg_details,
            "numeric_accuracy": na_details,
            "source_coverage": sc_details,
            "total_claims_extracted": len(claims),
            "total_claims_grounded": sum(1 for v in verifications if v.is_grounded)
        },
        "verifications": [
            {"claim": v.claim.text[:100], "grounded": v.is_grounded, "confidence": round(v.confidence, 2),
             "match_type": v.match_type, "source": v.supporting_source[:50] if v.supporting_source else None}
            for v in verifications[:10]
        ]
    }

# =========================================================
# 12. FINAL CONFIDENCE CALCULATION (REDESIGNED)
# =========================================================

def calculate_final_confidence(base_conf: float, evidence_score: float) -> float:
    """
    Calculate final confidence with proper evidence weighting.
    
    - High model (90) + High evidence (85) → ~85%
    - High model (90) + Low evidence (30)  → ~40%
    - Low model (50)  + High evidence (85) → ~75%
    - Low model (50)  + Low evidence (30)  → ~30%
    """
    base_conf = max(0, min(100, base_conf))
    evidence_score = max(0, min(100, evidence_score))
    
    # Evidence component (70% weight)
    evidence_component = evidence_score * 0.70
    
    # Model component (30% weight) - discounted by weak evidence
    evidence_multiplier = 0.2 + (evidence_score / 100) * 0.8  # Range: 0.2 to 1.0
    model_component = base_conf * evidence_multiplier * 0.30
    
    return round(max(0, min(100, evidence_component + model_component)), 1)

# =========================================================
# 13. LLM QUERY FUNCTIONS
# =========================================================

def query_perplexity(query: str, web_context: Dict, temperature: float = 0.1) -> str:
    search_count = len(web_context.get("search_results", []))
    
    if not web_context.get("summary") or search_count < 2:
        enhanced_query = f"{SYSTEM_PROMPT}\n\nUser Question: {query}\n\nWeb search returned {search_count} results."
    else:
        context_section = f"LATEST WEB RESEARCH:\n{web_context['summary']}\n\n"
        if web_context.get('scraped_content'):
            context_section += "\nDETAILED CONTENT:\n"
            for url, content in list(web_context['scraped_content'].items())[:2]:
                context_section += f"\n{url}:\n{content[:800]}...\n"
        enhanced_query = f"{context_section}\n{SYSTEM_PROMPT}\n\nUser Question: {query}"
    
    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
    payload = {"model": "sonar", "temperature": temperature, "max_tokens": 2000, "top_p": 0.8,
               "messages": [{"role": "user", "content": enhanced_query}]}
    
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
        st.error(f"❌ Perplexity API error: {e}")
        return create_fallback_response(query, search_count, web_context)

def create_fallback_response(query: str, search_count: int, web_context: Dict) -> str:
    fallback = LLMResponse(
        executive_summary=f"Analysis of '{query}' with {search_count} web sources. Fallback structure used.",
        primary_metrics={"sources": MetricDetail(name="Web Sources", value=search_count, unit="sources")},
        key_findings=[f"Web search found {search_count} sources.", "Primary model required fallback."],
        top_entities=[TopEntityDetail(name="Source 1", share="N/A", growth="N/A")],
        trends_forecast=[TrendForecastDetail(trend="Fallback used", direction="⚠️", timeline="Now")],
        visualization_data=VisualizationData(chart_labels=["Attempt"], chart_values=[search_count], chart_title="Search Results"),
        sources=web_context.get("sources", []),
        confidence=60,
        freshness="Current (fallback)"
    )
    return fallback.model_dump_json()

# =========================================================
# 14. DASHBOARD RENDERING
# =========================================================

def detect_x_label_dynamic(labels: list) -> str:
    if not labels:
        return "Category"
    label_texts = [str(l).lower().strip() for l in labels]
    
    region_keywords = ['north america', 'asia pacific', 'europe', 'latin america', 'middle east', 'africa', 'china', 'usa', 'india']
    if sum(1 for l in label_texts if any(k in l for k in region_keywords)) / len(labels) >= 0.4:
        return "Regions"
    if sum(1 for l in label_texts if re.search(r'\b(19|20)\d{2}\b', l)) / len(labels) > 0.5:
        return "Years"
    if sum(1 for l in label_texts if re.search(r'\bq[1-4]\b', l, re.I)) >= 2:
        return "Quarters"
    return "Categories"

def detect_y_label_dynamic(values: list) -> str:
    if not values:
        return "Value"
    numeric_values = [abs(float(v)) for v in values if str(v).replace('.','').replace('-','').isdigit()]
    if not numeric_values:
        return "Value"
    avg_mag, max_mag = np.mean(numeric_values), max(numeric_values)
    if max_mag > 500 or avg_mag > 100:
        return "USD B"
    elif max_mag > 50 or avg_mag > 10:
        return "USD M"
    elif all(0 <= v <= 100 for v in numeric_values):
        return "Percent %"
    return "Value"

def render_dashboard(primary_json: str, final_conf: float, web_context: Dict, base_conf: float,
                    user_question: str, veracity_scores: Optional[Dict] = None):
    try:
        data = json.loads(primary_json)
    except Exception as e:
        st.error(f"❌ Cannot render dashboard: {e}")
        return
    
    st.header("📊 Yureeka Market Report")
    st.markdown(f"**Question:** {user_question}")
    
    # Confidence row
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Confidence", f"{final_conf:.1f}%")
    col2.metric("Model Confidence", f"{base_conf:.1f}%")
    col3.metric("Evidence Score", f"{veracity_scores.get('overall', 0):.1f}%" if veracity_scores else "N/A")
    
    st.markdown("---")
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    st.markdown(f"**{data.get('executive_summary', 'No summary available')}**")
    st.markdown("---")
    
    # Key Metrics
    st.subheader("💰 Key Metrics")
    metrics = data.get('primary_metrics', {})
    if metrics:
        metric_rows = [{"Metric": d.get("name", k), "Value": f"{d.get('value', 'N/A')} {d.get('unit', '')}".strip()}
                      for k, d in list(metrics.items())[:6] if isinstance(d, dict)]
        if metric_rows:
            st.table(pd.DataFrame(metric_rows))
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
        entity_data = [{"Entity": e.get("name", "N/A"), "Share": e.get("share", "N/A"), "Growth": e.get("growth", "N/A")}
                      for e in entities if isinstance(e, dict)]
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
                numeric_values = [float(v) for v in values[:10]]
                df_viz = pd.DataFrame({"x": labels[:10], "y": numeric_values})
                chart_type = viz.get("chart_type", "line")
                fig = px.bar(df_viz, x="x", y="y", title=viz.get("chart_title", "Trend")) if chart_type == "bar" else px.line(df_viz, x="x", y="y", title=viz.get("chart_title", "Trend"), markers=True)
                fig.update_layout(xaxis_title=detect_x_label_dynamic(labels), yaxis_title=detect_y_label_dynamic(numeric_values))
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
            reliability_label, _ = classify_source_reliability(str(src))
            cols[(i-1) % 2].markdown(f"**{i}.** [{src[:50]}...]({src}) {reliability_label}", unsafe_allow_html=True)
    st.markdown("---")
    
    # Evidence Quality
    if veracity_scores:
        st.subheader("✅ Evidence Quality Analysis")
        cols = st.columns(4)
        for i, (label, key) in enumerate([("Source Quality", "source_quality"), ("Claim Grounding", "claim_grounding"),
                                          ("Numeric Accuracy", "numeric_accuracy"), ("Source Coverage", "source_coverage")]):
            cols[i].metric(label, f"{veracity_scores.get(key, 0):.0f}%")
        
        with st.expander("🔬 Claim Verification Details"):
            details = veracity_scores.get("details", {})
            cg = details.get("claim_grounding", {})
            st.write(f"**Claims:** {cg.get('total_claims', 0)} total, {cg.get('grounded_claims', 0)} grounded ({cg.get('grounding_rate', 0):.1%})")
            for v in veracity_scores.get("verifications", []):
                icon = "✅" if v["grounded"] else "❌"
                st.write(f"{icon} {v['claim'][:60]}... (conf: {v['confidence']:.0%}, {v['match_type']})")
    
    # Web Context
    if web_context and web_context.get("search_results"):
        with st.expander("🌐 Web Search Details"):
            for i, r in enumerate(web_context["search_results"][:5]):
                st.markdown(f"**{i+1}. {r.get('title')}**")
                st.caption(f"{r.get('source')} - {r.get('date')}")
                st.write(r.get('snippet', ''))
                st.markdown("---")

# =========================================================
# 15. MAIN APPLICATION
# =========================================================

def main():
    st.set_page_config(page_title="Yureeka Market Report", page_icon="💹", layout="wide")
    st.title("💹 Yureeka Market Intelligence")
    
    st.markdown("""
    **Yureeka** provides AI-powered market research with claim-level verification.
    *Version 8.0 - With semantic claim grounding*
    """)
    
    query = st.text_input("Enter your question about markets, industries, finance, or economics:",
                         placeholder="e.g., What is the size of the global EV battery market?")
    
    use_web = st.checkbox("Enable web search (recommended)", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY)
    
    if st.button("🔍 Analyze", type="primary") and query:
        if len(query.strip()) < 5:
            st.error("❌ Please enter a question with at least 5 characters")
            return
        
        query = query.strip()[:500]
        
        # Web search
        web_context = {}
        if use_web:
            with st.spinner("🌐 Searching the web..."):
                web_context = fetch_web_context(query, num_sources=3)
        
        if not web_context or not web_context.get("search_results"):
            st.info("💡 Using AI knowledge without web search")
            web_context = {"search_results": [], "scraped_content": {}, "summary": "", "sources": [],
                          "source_reliability": [], "source_scores": []}
        
        # Query LLM
        with st.spinner("🤖 Analyzing..."):
            primary_response = query_perplexity(query, web_context)
        
        if not primary_response:
            st.error("❌ Primary model failed")
            return
        
        try:
            primary_data = json.loads(primary_response)
        except Exception as e:
            st.error(f"❌ Parse error: {e}")
            return
        
        # Veracity scoring
        with st.spinner("✅ Verifying claims..."):
            veracity_scores = evidence_based_veracity(primary_data, web_context)
        
        base_conf = float(primary_data.get("confidence", 75))
        final_conf = calculate_final_confidence(base_conf, veracity_scores["overall"])
        
        # Download
        output = {
            "question": query,
            "timestamp": datetime.now().isoformat(),
            "primary_response": primary_data,
            "final_confidence": final_conf,
            "veracity_scores": veracity_scores,
            "web_sources": web_context.get("sources", [])
        }
        
        st.download_button("💾 Download JSON", json.dumps(output, indent=2).encode(),
                          f"yureeka_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")
        
        # Render
        render_dashboard(primary_response, final_conf, web_context, base_conf, query, veracity_scores)
        
        # Debug
        with st.expander("🔧 Debug"):
            st.json({
                "model_confidence": base_conf,
                "evidence_score": veracity_scores["overall"],
                "final_confidence": final_conf,
                "evidence_multiplier": round(0.2 + (veracity_scores["overall"] / 100) * 0.8, 2),
                "veracity_breakdown": {k: veracity_scores[k] for k in ["source_quality", "claim_grounding", "numeric_accuracy", "source_coverage", "overall"]}
            })

if __name__ == "__main__":
    main()

