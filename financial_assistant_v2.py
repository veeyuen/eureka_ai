# =========================================================
# YUREEKA AI RESEARCH ASSISTANT v10.0 - EVOLUTION TRACKING
# Builds on v9 Hybrid with temporal comparison and stability metrics
# =========================================================

import os
import re
import json
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import google.generativeai as genai
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from collections import Counter
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from dataclasses import dataclass, field
import hashlib

# =========================================================
# 1. CONFIGURATION & API KEY VALIDATION
# =========================================================

def load_api_keys():
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
# 3. EVOLUTION DATA STRUCTURES
# =========================================================

@dataclass
class MetricChange:
    """Tracks change in a single metric"""
    name: str
    old_value: Optional[float]
    new_value: Optional[float]
    old_unit: str = ""
    new_unit: str = ""
    change_absolute: Optional[float] = None
    change_percent: Optional[float] = None
    direction: str = "unchanged"  # 'up', 'down', 'unchanged', 'new', 'removed'

@dataclass
class EntityChange:
    """Tracks change in entity ranking/share"""
    name: str
    old_rank: Optional[int]
    new_rank: Optional[int]
    old_share: Optional[str]
    new_share: Optional[str]
    status: str = "unchanged"  # 'unchanged', 'moved_up', 'moved_down', 'new', 'removed'

@dataclass
class FindingChange:
    """Tracks changes in key findings"""
    finding: str
    status: str  # 'retained', 'new', 'removed', 'modified'
    similarity_to_old: Optional[float] = None
    matched_old_finding: Optional[str] = None

@dataclass 
class EvolutionResult:
    """Complete evolution analysis"""
    previous_timestamp: str
    current_timestamp: str
    time_delta_hours: float
    
    # Metric changes
    metric_changes: List[MetricChange] = field(default_factory=list)
    metrics_stability: float = 0.0
    
    # Entity changes
    entity_changes: List[EntityChange] = field(default_factory=list)
    entities_stability: float = 0.0
    
    # Finding changes
    finding_changes: List[FindingChange] = field(default_factory=list)
    findings_stability: float = 0.0
    
    # Confidence evolution
    old_confidence: float = 0.0
    new_confidence: float = 0.0
    confidence_change: float = 0.0
    
    # Overall stability
    overall_stability: float = 0.0
    
    # Summary
    summary: str = ""
    
    def to_dict(self) -> dict:
        return {
            "previous_timestamp": self.previous_timestamp,
            "current_timestamp": self.current_timestamp,
            "time_delta_hours": self.time_delta_hours,
            "metric_changes": [
                {"name": m.name, "old": m.old_value, "new": m.new_value, 
                 "change_pct": m.change_percent, "direction": m.direction}
                for m in self.metric_changes
            ],
            "metrics_stability": self.metrics_stability,
            "entity_changes": [
                {"name": e.name, "old_rank": e.old_rank, "new_rank": e.new_rank,
                 "old_share": e.old_share, "new_share": e.new_share, "status": e.status}
                for e in self.entity_changes
            ],
            "entities_stability": self.entities_stability,
            "finding_changes": [
                {"finding": f.finding[:100], "status": f.status, "similarity": f.similarity_to_old}
                for f in self.finding_changes
            ],
            "findings_stability": self.findings_stability,
            "confidence_evolution": {
                "old": self.old_confidence,
                "new": self.new_confidence,
                "change": self.confidence_change
            },
            "overall_stability": self.overall_stability,
            "summary": self.summary
        }

# =========================================================
# 4. VERIFICATION DATA STRUCTURES (from v9)
# =========================================================

@dataclass
class ExtractedClaim:
    text: str
    claim_type: str
    numeric_value: Optional[float] = None
    numeric_unit: Optional[str] = None
    source_field: str = ""
    priority: int = 1

@dataclass
class ClaimVerification:
    claim: ExtractedClaim
    is_grounded: bool
    confidence: float
    supporting_source: Optional[str] = None
    supporting_text: Optional[str] = None
    match_type: str = "none"

@dataclass
class VeracityResult:
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
# 5. PROMPTS
# =========================================================

RESPONSE_TEMPLATE = '''
{
  "executive_summary": "4-6 sentence comprehensive answer with specific numbers",
  "primary_metrics": {
    "metric_1": {"name": "Key Metric", "value": 25.5, "unit": "%"}
  },
  "key_findings": ["Finding 1 with data", "Finding 2"],
  "top_entities": [{"name": "Company", "share": "25%", "growth": "15%"}],
  "trends_forecast": [{"trend": "Description", "direction": "↑", "timeline": "2025-2027"}],
  "visualization_data": {
    "chart_labels": ["2023", "2024", "2025"],
    "chart_values": [100, 120, 145],
    "chart_title": "Market Growth",
    "chart_type": "line"
  },
  "sources": ["source1.com"],
  "confidence": 85,
  "freshness": "Dec 2024"
}
'''

SYSTEM_PROMPT = f"""You are a professional market research analyst.

CRITICAL RULES:
1. Return ONLY valid JSON. NO markdown, NO code blocks.
2. NO citation references like [1][2] inside strings.
3. Use double quotes for all keys and string values.
4. NO trailing commas.

REQUIRED FIELDS with substantive data:
- executive_summary: 4-6 sentences with specific quantitative data
- primary_metrics: 3+ metrics with numbers and units
- key_findings: 3+ findings with quantitative details
- top_entities: 3+ companies/countries with market share
- trends_forecast: 2+ trends with timelines
- visualization_data: chart_labels and chart_values arrays
- sources: list of source URLs
- confidence: 0-100 score

Output ONLY this JSON:
{RESPONSE_TEMPLATE}
"""

# =========================================================
# 6. MODEL LOADING
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
# 7. SOURCE CLASSIFICATION
# =========================================================

HIGH_QUALITY_DOMAINS = {
    "gov", "imf.org", "worldbank.org", "federalreserve.gov", "reuters.com",
    "spglobal.com", "economist.com", "mckinsey.com", "ft.com", "wsj.com",
    "bloomberg.com", "deloitte.com", "kpmg.com", "sec.gov", "nature.com"
}

MEDIUM_QUALITY_DOMAINS = {
    "wikipedia.org", "forbes.com", "cnbc.com", "statista.com", "investopedia.com"
}

LOW_QUALITY_DOMAINS = {"blog", "medium.com", "wordpress", "reddit.com", "quora.com"}

def classify_source_reliability(source: str) -> Tuple[str, int]:
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
    if not sources:
        return (0.0, {"count": 0})
    scores = [classify_source_reliability(src)[1] for src in sources]
    avg_score = sum(scores) / len(scores)
    high_count = sum(1 for s in scores if s >= 80)
    if high_count >= 3:
        avg_score = min(100, avg_score + 10)
    elif high_count >= 2:
        avg_score = min(100, avg_score + 5)
    return (avg_score, {"count": len(sources), "high": high_count})

# =========================================================
# 8. JSON REPAIR & WEB FUNCTIONS (from v9)
# =========================================================

def repair_llm_response(data: dict) -> dict:
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
    
    for fld in ["top_entities", "trends_forecast"]:
        if fld in data:
            if isinstance(data[fld], dict):
                data[fld] = list(data[fld].values())
            elif not isinstance(data[fld], list):
                data[fld] = []
    
    if "visualization_data" in data and isinstance(data["visualization_data"], dict):
        viz = data["visualization_data"]
        if "labels" in viz and "chart_labels" not in viz:
            viz["chart_labels"] = viz.pop("labels")
        if "values" in viz and "chart_values" not in viz:
            viz["chart_values"] = viz.pop("values")
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
    
    all_text_parts = [r.get("snippet", "") for r in search_results]
    all_text_parts.extend(scraped_content.values())
    source_scores = [classify_source_reliability(r.get("link", ""))[1] for r in search_results]
    
    return {
        "search_results": search_results, "scraped_content": scraped_content,
        "all_text": "\n\n".join(all_text_parts), "sources": [r["link"] for r in search_results],
        "source_scores": source_scores
    }

# =========================================================
# 9. EVOLUTION ANALYSIS FUNCTIONS
# =========================================================

def parse_numeric_value(value: Any) -> Optional[float]:
    """Parse a value to float, handling strings with units"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove common formatting
        cleaned = re.sub(r'[,$%]', '', value.strip())
        # Handle billions/millions
        multiplier = 1.0
        if cleaned.lower().endswith('b') or 'billion' in cleaned.lower():
            multiplier = 1000
            cleaned = re.sub(r'[bB](?:illion)?', '', cleaned)
        elif cleaned.lower().endswith('m') or 'million' in cleaned.lower():
            multiplier = 1
            cleaned = re.sub(r'[mM](?:illion)?', '', cleaned)
        elif cleaned.lower().endswith('k') or 'thousand' in cleaned.lower():
            multiplier = 0.001
            cleaned = re.sub(r'[kK](?:thousand)?', '', cleaned)
        
        try:
            return float(cleaned.strip()) * multiplier
        except:
            return None
    return None

def compare_metrics(old_metrics: Dict, new_metrics: Dict) -> Tuple[List[MetricChange], float]:
    """Compare metrics between old and new responses"""
    changes = []
    
    # Get all metric keys
    old_keys = set(old_metrics.keys()) if old_metrics else set()
    new_keys = set(new_metrics.keys()) if new_metrics else set()
    all_keys = old_keys | new_keys
    
    stable_count = 0
    total_count = 0
    
    for key in all_keys:
        old_metric = old_metrics.get(key, {}) if old_metrics else {}
        new_metric = new_metrics.get(key, {}) if new_metrics else {}
        
        old_val = parse_numeric_value(old_metric.get("value")) if old_metric else None
        new_val = parse_numeric_value(new_metric.get("value")) if new_metric else None
        
        name = new_metric.get("name") or old_metric.get("name") or key
        old_unit = old_metric.get("unit", "") if old_metric else ""
        new_unit = new_metric.get("unit", "") if new_metric else ""
        
        change = MetricChange(
            name=name,
            old_value=old_val,
            new_value=new_val,
            old_unit=old_unit,
            new_unit=new_unit
        )
        
        if old_val is None and new_val is not None:
            change.direction = "new"
        elif old_val is not None and new_val is None:
            change.direction = "removed"
        elif old_val is not None and new_val is not None:
            change.change_absolute = new_val - old_val
            if old_val != 0:
                change.change_percent = ((new_val - old_val) / abs(old_val)) * 100
            
            # Determine direction and stability
            if change.change_percent is not None:
                if abs(change.change_percent) < 5:
                    change.direction = "unchanged"
                    stable_count += 1
                elif change.change_percent > 0:
                    change.direction = "up"
                else:
                    change.direction = "down"
            total_count += 1
        
        changes.append(change)
    
    stability = (stable_count / total_count * 100) if total_count > 0 else 100.0
    return changes, stability

def compare_entities(old_entities: List, new_entities: List) -> Tuple[List[EntityChange], float]:
    """Compare entity rankings between old and new responses"""
    changes = []
    
    # Build name -> (rank, share) maps
    old_map = {}
    for i, e in enumerate(old_entities or []):
        if isinstance(e, dict):
            name = e.get("name", "").lower().strip()
            if name:
                old_map[name] = (i + 1, e.get("share", ""))
    
    new_map = {}
    for i, e in enumerate(new_entities or []):
        if isinstance(e, dict):
            name = e.get("name", "").lower().strip()
            if name:
                new_map[name] = (i + 1, e.get("share", ""))
    
    all_names = set(old_map.keys()) | set(new_map.keys())
    stable_count = 0
    
    for name in all_names:
        old_rank, old_share = old_map.get(name, (None, None))
        new_rank, new_share = new_map.get(name, (None, None))
        
        # Find display name (prefer new)
        display_name = name.title()
        for e in (new_entities or []) + (old_entities or []):
            if isinstance(e, dict) and e.get("name", "").lower().strip() == name:
                display_name = e.get("name", name.title())
                break
        
        change = EntityChange(
            name=display_name,
            old_rank=old_rank,
            new_rank=new_rank,
            old_share=old_share,
            new_share=new_share
        )
        
        if old_rank is None:
            change.status = "new"
        elif new_rank is None:
            change.status = "removed"
        elif old_rank == new_rank:
            change.status = "unchanged"
            stable_count += 1
        elif new_rank < old_rank:
            change.status = "moved_up"
        else:
            change.status = "moved_down"
        
        changes.append(change)
    
    stability = (stable_count / len(all_names) * 100) if all_names else 100.0
    return changes, stability

@st.cache_data
def get_embedding(text: str):
    return embedder.encode(text[:512])

def semantic_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    try:
        v1 = get_embedding(text1)
        v2 = get_embedding(text2)
        return float(util.cos_sim(v1, v2).item())
    except:
        return 0.0

def compare_findings(old_findings: List[str], new_findings: List[str]) -> Tuple[List[FindingChange], float]:
    """Compare key findings using semantic similarity"""
    changes = []
    
    old_findings = [f for f in (old_findings or []) if isinstance(f, str) and f.strip()]
    new_findings = [f for f in (new_findings or []) if isinstance(f, str) and f.strip()]
    
    matched_old = set()
    retained_count = 0
    
    # For each new finding, find best match in old
    for new_f in new_findings:
        best_match = None
        best_sim = 0.0
        
        for i, old_f in enumerate(old_findings):
            if i in matched_old:
                continue
            sim = semantic_similarity(new_f, old_f)
            if sim > best_sim:
                best_sim = sim
                best_match = (i, old_f)
        
        if best_match and best_sim > 0.7:
            matched_old.add(best_match[0])
            if best_sim > 0.9:
                changes.append(FindingChange(
                    finding=new_f, status="retained",
                    similarity_to_old=best_sim, matched_old_finding=best_match[1]
                ))
                retained_count += 1
            else:
                changes.append(FindingChange(
                    finding=new_f, status="modified",
                    similarity_to_old=best_sim, matched_old_finding=best_match[1]
                ))
        else:
            changes.append(FindingChange(finding=new_f, status="new"))
    
    # Mark unmatched old findings as removed
    for i, old_f in enumerate(old_findings):
        if i not in matched_old:
            changes.append(FindingChange(finding=old_f, status="removed"))
    
    total = len(old_findings) + len(new_findings) - len(matched_old)
    stability = (retained_count / max(len(new_findings), 1)) * 100
    return changes, stability

def analyze_evolution(old_data: dict, new_data: dict) -> EvolutionResult:
    """
    Comprehensive evolution analysis between two snapshots.
    """
    result = EvolutionResult(
        previous_timestamp=old_data.get("timestamp", "Unknown"),
        current_timestamp=new_data.get("timestamp", datetime.now().isoformat()),
        time_delta_hours=0.0
    )
    
    # Calculate time delta
    try:
        old_time = datetime.fromisoformat(result.previous_timestamp.replace('Z', '+00:00'))
        new_time = datetime.fromisoformat(result.current_timestamp.replace('Z', '+00:00'))
        result.time_delta_hours = (new_time - old_time).total_seconds() / 3600
    except:
        pass
    
    # Get response data
    old_response = old_data.get("response") or old_data.get("primary_response", {})
    new_response = new_data.get("response") or new_data.get("primary_response", {})
    
    # Compare metrics
    result.metric_changes, result.metrics_stability = compare_metrics(
        old_response.get("primary_metrics", {}),
        new_response.get("primary_metrics", {})
    )
    
    # Compare entities
    result.entity_changes, result.entities_stability = compare_entities(
        old_response.get("top_entities", []),
        new_response.get("top_entities", [])
    )
    
    # Compare findings
    result.finding_changes, result.findings_stability = compare_findings(
        old_response.get("key_findings", []),
        new_response.get("key_findings", [])
    )
    
    # Confidence evolution
    old_conf = old_data.get("confidence", {})
    new_conf = new_data.get("confidence", {})
    
    if isinstance(old_conf, dict):
        result.old_confidence = old_conf.get("final", old_response.get("confidence", 75))
    else:
        result.old_confidence = float(old_conf) if old_conf else 75
    
    if isinstance(new_conf, dict):
        result.new_confidence = new_conf.get("final", new_response.get("confidence", 75))
    else:
        result.new_confidence = float(new_conf) if new_conf else 75
    
    result.confidence_change = result.new_confidence - result.old_confidence
    
    # Overall stability (weighted average)
    result.overall_stability = (
        result.metrics_stability * 0.40 +
        result.entities_stability * 0.30 +
        result.findings_stability * 0.30
    )
    
    # Generate summary
    result.summary = generate_evolution_summary(result)
    
    return result

def generate_evolution_summary(evolution: EvolutionResult) -> str:
    """Generate human-readable evolution summary"""
    parts = []
    
    # Time context
    if evolution.time_delta_hours > 0:
        if evolution.time_delta_hours < 24:
            parts.append(f"Compared to {evolution.time_delta_hours:.1f} hours ago:")
        else:
            days = evolution.time_delta_hours / 24
            parts.append(f"Compared to {days:.1f} days ago:")
    
    # Stability assessment
    if evolution.overall_stability >= 80:
        parts.append("Data is highly stable with minimal changes.")
    elif evolution.overall_stability >= 60:
        parts.append("Data shows moderate stability with some updates.")
    elif evolution.overall_stability >= 40:
        parts.append("Significant changes detected in the data.")
    else:
        parts.append("Major revisions - data has substantially changed.")
    
    # Metric highlights
    up_metrics = [m for m in evolution.metric_changes if m.direction == "up"]
    down_metrics = [m for m in evolution.metric_changes if m.direction == "down"]
    
    if up_metrics:
        names = [m.name for m in up_metrics[:2]]
        parts.append(f"📈 Increased: {', '.join(names)}")
    if down_metrics:
        names = [m.name for m in down_metrics[:2]]
        parts.append(f"📉 Decreased: {', '.join(names)}")
    
    # Entity changes
    new_entities = [e for e in evolution.entity_changes if e.status == "new"]
    if new_entities:
        names = [e.name for e in new_entities[:2]]
        parts.append(f"🆕 New players: {', '.join(names)}")
    
    # Confidence
    if abs(evolution.confidence_change) > 5:
        direction = "increased" if evolution.confidence_change > 0 else "decreased"
        parts.append(f"Confidence {direction} by {abs(evolution.confidence_change):.1f}%")
    
    return " ".join(parts)

# =========================================================
# 10. VERACITY SCORING (from v9)
# =========================================================

def parse_number_with_context(text: str) -> List[Tuple[float, str, str]]:
    results = []
    patterns = [
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(trillion|T)\b', 1_000_000, "T"),
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(billion|B)\b', 1_000, "B"),
        (r'\$?\s*(\d+(?:\.\d+)?)\s*(million|M)\b', 1, "M"),
        (r'(\d+(?:\.\d+)?)\s*%', 1, "%"),
    ]
    for pattern, mult, unit in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = float(match.group(1)) * mult
                start, end = max(0, match.start() - 40), min(len(text), match.end() + 40)
                results.append((value, unit, text[start:end].strip()))
            except:
                continue
    return results

def extract_priority_claims(primary_data: dict, max_claims: int = 8) -> List[ExtractedClaim]:
    claims = []
    summary = primary_data.get("executive_summary", "")
    if summary:
        for value, unit, context in parse_number_with_context(summary)[:3]:
            claims.append(ExtractedClaim(text=context, claim_type="numeric", numeric_value=value,
                                        numeric_unit=unit, source_field="executive_summary", priority=1))
    
    for key, metric in list(primary_data.get("primary_metrics", {}).items())[:3]:
        if isinstance(metric, dict):
            val = parse_numeric_value(metric.get("value"))
            if val is not None:
                claims.append(ExtractedClaim(
                    text=f"{metric.get('name', key)}: {metric.get('value')} {metric.get('unit', '')}".strip(),
                    claim_type="numeric", numeric_value=val, numeric_unit=metric.get("unit", ""),
                    source_field="primary_metrics", priority=1
                ))
    
    for entity in primary_data.get("top_entities", [])[:2]:
        if isinstance(entity, dict) and entity.get("name") and entity.get("share"):
            claims.append(ExtractedClaim(
                text=f"{entity['name']} market share {entity['share']}", claim_type="entity",
                source_field="top_entities", priority=2
            ))
    
    return claims[:max_claims]

def quick_keyword_match(claim_text: str, source_text: str) -> float:
    claim_words = set(re.findall(r'\b\w{4,}\b', claim_text.lower()))
    source_words = set(re.findall(r'\b\w{4,}\b', source_text.lower()))
    if not claim_words:
        return 0.0
    return len(claim_words & source_words) / len(claim_words)

def verify_numeric_claim(claim: ExtractedClaim, source_text: str) -> Tuple[bool, float, str]:
    if claim.numeric_value is None:
        return (False, 0.0, "")
    source_numbers = parse_number_with_context(source_text)
    if not source_numbers:
        return (False, 0.0, "")
    
    best = (False, 0.0, "")
    claim_unit = (claim.numeric_unit or "").upper()
    
    for src_val, src_unit, src_ctx in source_numbers:
        src_unit_up = src_unit.upper()
        compatible = (claim_unit == src_unit_up or
                     (claim_unit in ["B", "BILLION"] and src_unit_up in ["B", "BILLION"]) or
                     (claim_unit in ["M", "MILLION"] and src_unit_up in ["M", "MILLION"]) or
                     (claim_unit == "%" and src_unit_up in ["%", "PERCENT"]))
        if not compatible:
            continue
        
        max_val = max(abs(claim.numeric_value), abs(src_val), 0.001)
        rel_diff = abs(claim.numeric_value - src_val) / max_val
        
        conf = 0.98 if rel_diff < 0.03 else 0.90 if rel_diff < 0.08 else 0.75 if rel_diff < 0.15 else 0.55 if rel_diff < 0.25 else 0.0
        if conf > best[1]:
            best = (conf >= 0.55, conf, src_ctx)
    return best

def verify_claim_hybrid(claim: ExtractedClaim, web_context: dict) -> ClaimVerification:
    all_text = web_context.get("all_text", "")
    search_results = web_context.get("search_results", [])
    scraped_content = web_context.get("scraped_content", {})
    
    best = ClaimVerification(claim=claim, is_grounded=False, confidence=0.0, match_type="none")
    
    if claim.claim_type == "numeric" and claim.numeric_value is not None:
        for result in search_results:
            is_match, conf, ctx = verify_numeric_claim(claim, result.get("snippet", ""))
            if is_match and conf > best.confidence:
                best = ClaimVerification(claim=claim, is_grounded=True, confidence=conf,
                                        supporting_source=result.get("link"), supporting_text=ctx[:150], match_type="numeric")
                if conf > 0.85:
                    return best
        
        if not best.is_grounded:
            for url, content in scraped_content.items():
                is_match, conf, ctx = verify_numeric_claim(claim, content)
                if is_match and conf > best.confidence:
                    best = ClaimVerification(claim=claim, is_grounded=True, confidence=conf,
                                            supporting_source=url, supporting_text=ctx[:150], match_type="numeric")
                    if conf > 0.85:
                        return best
    
    kw_score = quick_keyword_match(claim.text, all_text)
    if kw_score > 0.5 and kw_score * 0.8 > best.confidence:
        best = ClaimVerification(claim=claim, is_grounded=kw_score > 0.6, confidence=kw_score * 0.8,
                                supporting_text="Keyword match", match_type="keyword")
    
    if not best.is_grounded and claim.claim_type != "numeric":
        for result in search_results:
            snippet = result.get("snippet", "")
            if len(snippet) > 30:
                sim = semantic_similarity(claim.text, snippet)
                if sim > 0.55 and sim > best.confidence:
                    best = ClaimVerification(claim=claim, is_grounded=sim > 0.6, confidence=sim,
                                            supporting_source=result.get("link"), supporting_text=snippet[:150], match_type="semantic")
    return best

def check_source_agreement(primary_data: dict, web_context: dict) -> Tuple[float, dict]:
    summary = primary_data.get("executive_summary", "")
    response_numbers = parse_number_with_context(summary)
    if not response_numbers:
        return (50.0, {"message": "No numbers"})
    
    search_results = web_context.get("search_results", [])
    agreements = []
    
    for resp_val, resp_unit, _ in response_numbers[:3]:
        sources_agreeing = 0
        for result in search_results:
            for src_val, src_unit, _ in parse_number_with_context(result.get("snippet", "")):
                if resp_unit.upper() == src_unit.upper():
                    max_val = max(abs(resp_val), abs(src_val), 0.001)
                    if abs(resp_val - src_val) / max_val < 0.20:
                        sources_agreeing += 1
                        break
        if search_results:
            agreements.append(sources_agreeing / len(search_results))
    
    if not agreements:
        return (50.0, {})
    return (min(sum(agreements) / len(agreements) * 100, 95.0), {"checked": len(agreements)})

def check_response_completeness(primary_data: dict) -> Tuple[float, dict]:
    checks = {
        "summary": len(primary_data.get("executive_summary", "")) > 100,
        "metrics": len(primary_data.get("primary_metrics", {})) >= 2,
        "findings": len(primary_data.get("key_findings", [])) >= 2,
        "entities": len(primary_data.get("top_entities", [])) >= 1,
        "trends": len(primary_data.get("trends_forecast", [])) >= 1,
        "viz": bool(primary_data.get("visualization_data", {}).get("chart_values")),
        "sources": len(primary_data.get("sources", [])) >= 1,
    }
    return (sum(checks.values()) / len(checks) * 100, checks)

def calculate_veracity_hybrid(primary_data: dict, web_context: dict) -> VeracityResult:
    result = VeracityResult()
    
    sources = list(dict.fromkeys(primary_data.get("sources", []) + web_context.get("sources", [])))
    sq_score, sq_details = calculate_source_quality_fast(sources)
    result.source_quality = sq_score
    result.details["source_quality"] = sq_details
    
    claims = extract_priority_claims(primary_data, max_claims=8)
    result.claims_extracted = len(claims)
    
    verifications = [verify_claim_hybrid(c, web_context) for c in claims]
    numeric_vf = [v for v in verifications if v.claim.claim_type == "numeric"]
    
    if verifications:
        grounded = sum(1 for v in verifications if v.is_grounded)
        avg_conf = sum(v.confidence for v in verifications) / len(verifications)
        result.claims_verified = grounded
        result.claim_grounding = (grounded / len(verifications) * 0.6 + avg_conf * 0.4) * 100
        result.unverified_claims = [v.claim.text[:60] for v in verifications if not v.is_grounded]
    else:
        result.claim_grounding = 50.0
    
    if numeric_vf:
        num_grounded = sum(1 for v in numeric_vf if v.is_grounded and v.match_type == "numeric")
        num_conf = sum(v.confidence for v in numeric_vf if v.match_type == "numeric") / len(numeric_vf) if numeric_vf else 0
        result.numeric_accuracy = (num_grounded / len(numeric_vf) * 0.5 + num_conf * 0.5) * 100
    else:
        result.numeric_accuracy = 50.0
    
    result.verifications = [
        {"claim": v.claim.text[:80], "type": v.claim.claim_type, "grounded": v.is_grounded,
         "confidence": round(v.confidence, 2), "match_type": v.match_type}
        for v in verifications
    ]
    
    sa_score, _ = check_source_agreement(primary_data, web_context)
    result.source_agreement = sa_score
    
    rc_score, rc_details = check_response_completeness(primary_data)
    result.response_completeness = rc_score
    result.details["completeness"] = rc_details
    
    result.overall = (result.source_quality * 0.20 + result.claim_grounding * 0.30 +
                     result.numeric_accuracy * 0.25 + result.source_agreement * 0.15 +
                     result.response_completeness * 0.10)
    return result

def calculate_final_confidence(base_conf: float, evidence_score: float) -> float:
    base_conf = max(0, min(100, base_conf))
    evidence_score = max(0, min(100, evidence_score))
    evidence_component = evidence_score * 0.65
    evidence_multiplier = 0.25 + (evidence_score / 100) * 0.75
    model_component = base_conf * evidence_multiplier * 0.35
    return round(max(0, min(100, evidence_component + model_component)), 1)

# =========================================================
# 11. LLM QUERY FUNCTIONS
# =========================================================

def query_perplexity(query: str, web_context: Dict, temperature: float = 0.1) -> str:
    search_count = len(web_context.get("search_results", []))
    
    if not web_context.get("all_text") or search_count < 2:
        enhanced_query = f"{SYSTEM_PROMPT}\n\nUser Question: {query}\n\nProvide comprehensive analysis."
    else:
        context_parts = [f"Source: {r.get('source', 'Unknown')}\n{r.get('snippet', '')}" 
                        for r in web_context.get("search_results", [])[:3]]
        for url, content in list(web_context.get("scraped_content", {}).items())[:2]:
            context_parts.append(f"From {url}:\n{content[:600]}")
        enhanced_query = f"WEB RESEARCH:\n{chr(10).join(context_parts)}\n\n{SYSTEM_PROMPT}\n\nQuestion: {query}"
    
    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
    payload = {"model": "sonar", "temperature": temperature, "max_tokens": 2000, "top_p": 0.8,
               "messages": [{"role": "user", "content": enhanced_query}]}
    
    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        if "choices" not in data:
            raise Exception("No choices")
        content = data["choices"][0]["message"]["content"]
        if not content:
            raise Exception("Empty")
        
        parsed = parse_json_safely(content)
        if not parsed:
            return create_fallback_response(query, search_count, web_context)
        
        repaired = repair_llm_response(parsed)
        try:
            llm_obj = LLMResponse.model_validate(repaired)
            if web_context.get("sources"):
                llm_obj.sources = list(dict.fromkeys((llm_obj.sources or []) + web_context["sources"]))[:10]
                llm_obj.freshness = "Current (web-enhanced)"
            return llm_obj.model_dump_json()
        except ValidationError:
            return create_fallback_response(query, search_count, web_context)
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return create_fallback_response(query, search_count, web_context)

def create_fallback_response(query: str, search_count: int, web_context: Dict) -> str:
    fallback = LLMResponse(
        executive_summary=f"Analysis of '{query}' with {search_count} sources. Fallback.",
        primary_metrics={"sources": MetricDetail(name="Sources", value=search_count, unit="")},
        key_findings=[f"Found {search_count} sources", "Fallback response"],
        top_entities=[TopEntityDetail(name="N/A", share="N/A", growth="N/A")],
        trends_forecast=[TrendForecastDetail(trend="Fallback", direction="⚠️", timeline="N/A")],
        visualization_data=VisualizationData(chart_labels=["N/A"], chart_values=[0]),
        sources=web_context.get("sources", []), confidence=50, freshness="Current (fallback)"
    )
    return fallback.model_dump_json()

# =========================================================
# 12. EVOLUTION VISUALIZATION
# =========================================================

def render_evolution_panel(evolution: EvolutionResult):
    """Render the evolution comparison panel"""
    
    st.subheader("📊 Evolution Analysis")
    
    # Summary
    st.info(evolution.summary)
    
    # Stability meters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "green" if evolution.overall_stability >= 70 else "orange" if evolution.overall_stability >= 40 else "red"
        st.metric("Overall Stability", f"{evolution.overall_stability:.0f}%",
                 help="How stable is the data compared to previous snapshot")
    
    with col2:
        st.metric("Metrics Stability", f"{evolution.metrics_stability:.0f}%",
                 help="Stability of numeric metrics")
    
    with col3:
        st.metric("Entities Stability", f"{evolution.entities_stability:.0f}%",
                 help="Stability of entity rankings")
    
    with col4:
        st.metric("Findings Stability", f"{evolution.findings_stability:.0f}%",
                 help="Stability of key findings")
    
    st.markdown("---")
    
    # Metric changes
    if evolution.metric_changes:
        st.subheader("📈 Metric Changes")
        
        metric_data = []
        for m in evolution.metric_changes:
            if m.direction != "unchanged":
                icon = {"up": "📈", "down": "📉", "new": "🆕", "removed": "❌"}.get(m.direction, "➡️")
                old_str = f"{m.old_value:.2f}" if m.old_value is not None else "N/A"
                new_str = f"{m.new_value:.2f}" if m.new_value is not None else "N/A"
                change_str = f"{m.change_percent:+.1f}%" if m.change_percent is not None else "-"
                
                metric_data.append({
                    "Metric": f"{icon} {m.name}",
                    "Previous": old_str,
                    "Current": new_str,
                    "Change": change_str
                })
        
        if metric_data:
            st.dataframe(pd.DataFrame(metric_data), hide_index=True, use_container_width=True)
        else:
            st.success("All metrics unchanged (within 5% tolerance)")
    
    # Entity changes
    if evolution.entity_changes:
        st.subheader("🏢 Entity Ranking Changes")
        
        entity_data = []
        for e in evolution.entity_changes:
            if e.status != "unchanged":
                icon = {"moved_up": "⬆️", "moved_down": "⬇️", "new": "🆕", "removed": "❌"}.get(e.status, "➡️")
                old_rank = f"#{e.old_rank}" if e.old_rank else "N/A"
                new_rank = f"#{e.new_rank}" if e.new_rank else "N/A"
                
                entity_data.append({
                    "Entity": f"{icon} {e.name}",
                    "Previous Rank": old_rank,
                    "Current Rank": new_rank,
                    "Previous Share": e.old_share or "N/A",
                    "Current Share": e.new_share or "N/A"
                })
        
        if entity_data:
            st.dataframe(pd.DataFrame(entity_data), hide_index=True, use_container_width=True)
        else:
            st.success("All entity rankings unchanged")
    
    # Finding changes
    if evolution.finding_changes:
        with st.expander("🔍 Finding Changes Detail"):
            new_findings = [f for f in evolution.finding_changes if f.status == "new"]
            removed_findings = [f for f in evolution.finding_changes if f.status == "removed"]
            modified_findings = [f for f in evolution.finding_changes if f.status == "modified"]
            
            if new_findings:
                st.markdown("**🆕 New Findings:**")
                for f in new_findings[:3]:
                    st.markdown(f"- {f.finding[:100]}...")
            
            if removed_findings:
                st.markdown("**❌ Removed Findings:**")
                for f in removed_findings[:3]:
                    st.markdown(f"- ~~{f.finding[:100]}...~~")
            
            if modified_findings:
                st.markdown("**✏️ Modified Findings:**")
                for f in modified_findings[:3]:
                    st.markdown(f"- {f.finding[:100]}... (similarity: {f.similarity_to_old:.0%})")
    
    # Confidence trend
    st.subheader("📊 Confidence Evolution")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Previous Confidence", f"{evolution.old_confidence:.1f}%")
    with col2:
        st.metric("Current Confidence", f"{evolution.new_confidence:.1f}%")
    with col3:
        delta_color = "normal" if abs(evolution.confidence_change) < 5 else "off"
        st.metric("Change", f"{evolution.confidence_change:+.1f}%")

def render_stability_chart(evolution: EvolutionResult):
    """Render stability radar chart"""
    categories = ['Metrics', 'Entities', 'Findings', 'Overall']
    values = [
        evolution.metrics_stability,
        evolution.entities_stability,
        evolution.findings_stability,
        evolution.overall_stability
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name='Stability'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Stability Profile"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 13. DASHBOARD RENDERING
# =========================================================

def detect_axis_labels(labels: list, values: list) -> Tuple[str, str]:
    x_label, y_label = "Category", "Value"
    if labels:
        label_text = ' '.join(str(l).lower() for l in labels)
        if any(k in label_text for k in ['america', 'europe', 'asia', 'china']):
            x_label = "Region"
        elif re.search(r'\b20\d{2}\b', label_text):
            x_label = "Year"
    if values:
        try:
            nums = [abs(float(v)) for v in values]
            avg = sum(nums) / len(nums)
            y_label = "USD Billions" if avg > 100 else "USD Millions" if avg > 1 else "Percent" if all(0 <= n <= 100 for n in nums) else "Value"
        except:
            pass
    return x_label, y_label

def render_dashboard(primary_json: str, final_conf: float, web_context: Dict, base_conf: float,
                    user_question: str, veracity: VeracityResult, evolution: Optional[EvolutionResult] = None):
    try:
        data = json.loads(primary_json)
    except Exception as e:
        st.error(f"❌ Cannot render: {e}")
        return
    
    st.header("📊 Yureeka Market Report")
    st.markdown(f"**Question:** {user_question}")
    
    # Confidence metrics
    cols = st.columns(4)
    cols[0].metric("Final Confidence", f"{final_conf:.0f}%")
    cols[1].metric("Model Confidence", f"{base_conf:.0f}%")
    cols[2].metric("Evidence Score", f"{veracity.overall:.0f}%")
    cols[3].metric("Claims Verified", f"{veracity.claims_verified}/{veracity.claims_extracted}")
    
    # Evolution panel (if comparing)
    if evolution:
        st.markdown("---")
        render_evolution_panel(evolution)
        with st.expander("📈 Stability Radar"):
            render_stability_chart(evolution)
    
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
    
    # Entities & Trends
    col1, col2 = st.columns(2)
    with col1:
        entities = data.get('top_entities', [])
        if entities:
            st.subheader("🏢 Top Players")
            df = pd.DataFrame([{"Entity": e.get("name"), "Share": e.get("share"), "Growth": e.get("growth")}
                              for e in entities if isinstance(e, dict)])
            if not df.empty:
                st.dataframe(df, hide_index=True, use_container_width=True)
    
    with col2:
        trends = data.get('trends_forecast', [])
        if trends:
            st.subheader("📈 Trends")
            df = pd.DataFrame([{"Trend": t.get("trend", "")[:50], "Direction": t.get("direction", "→")}
                              for t in trends if isinstance(t, dict)])
            if not df.empty:
                st.dataframe(df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Visualization
    viz = data.get('visualization_data')
    if viz and isinstance(viz, dict):
        labels, values = viz.get("chart_labels", []), viz.get("chart_values", [])
        if labels and values and len(labels) == len(values):
            st.subheader("📊 Visualization")
            try:
                nums = [float(v) for v in values[:10]]
                x_label, y_label = detect_axis_labels(labels, nums)
                df = pd.DataFrame({"x": labels[:10], "y": nums})
                fig = px.bar(df, x="x", y="y", title=viz.get("chart_title", "Analysis")) if viz.get("chart_type") == "bar" else px.line(df, x="x", y="y", title=viz.get("chart_title", "Analysis"), markers=True)
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
    
    # Evidence Quality
    st.subheader("✅ Evidence Quality")
    cols = st.columns(5)
    for i, (name, score) in enumerate([("Sources", veracity.source_quality), ("Grounding", veracity.claim_grounding),
                                        ("Numeric", veracity.numeric_accuracy), ("Agreement", veracity.source_agreement),
                                        ("Complete", veracity.response_completeness)]):
        cols[i].metric(name, f"{score:.0f}%")
    
    with st.expander("🔬 Claim Verification"):
        for v in veracity.verifications:
            icon = "✅" if v["grounded"] else "❌"
            st.markdown(f"{icon} **{v['claim'][:70]}...** ({v['match_type']}, {v['confidence']:.0%})")

# =========================================================
# 14. MAIN APPLICATION
# =========================================================

def main():
    st.set_page_config(page_title="Yureeka Evolution", page_icon="📈", layout="wide")
    st.title("📈 Yureeka Market Intelligence - Evolution Tracking")
    
    st.markdown("""
    **Yureeka v10** - Track how market data evolves over time.
    
    *Features: Load previous analyses, compare metrics, track stability*
    """)
    
    # Tabs for different modes
    tab1, tab2 = st.tabs(["🔍 New Analysis", "📊 Compare with Previous"])
    
    with tab1:
        query = st.text_input("Enter your market research question:",
                             placeholder="e.g., What is the global electric vehicle market size?",
                             key="new_query")
        
        col1, col2 = st.columns(2)
        with col1:
            use_web = st.checkbox("Enable web search", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY, key="new_web")
        
        if st.button("🔍 Analyze", type="primary", key="new_analyze") and query:
            run_analysis(query, use_web, None)
    
    with tab2:
        st.markdown("### Compare with Previous Analysis")
        
        uploaded_file = st.file_uploader("Upload previous Yureeka JSON output", type=['json'], key="upload")
        
        previous_data = None
        if uploaded_file:
            try:
                previous_data = json.load(uploaded_file)
                st.success(f"✅ Loaded: {previous_data.get('question', 'Unknown query')}")
                st.caption(f"Timestamp: {previous_data.get('timestamp', 'Unknown')}")
                
                # Show previous metrics summary
                prev_response = previous_data.get("response") or previous_data.get("primary_response", {})
                prev_metrics = prev_response.get("primary_metrics", {})
                if prev_metrics:
                    with st.expander("📋 Previous Metrics"):
                        for k, m in list(prev_metrics.items())[:5]:
                            if isinstance(m, dict):
                                st.write(f"**{m.get('name', k)}**: {m.get('value')} {m.get('unit', '')}")
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")
        
        query2 = st.text_input("Enter query (or use same as previous):",
                              value=previous_data.get("question", "") if previous_data else "",
                              key="compare_query")
        
        col1, col2 = st.columns(2)
        with col1:
            use_web2 = st.checkbox("Enable web search", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY, key="compare_web")
        
        if st.button("🔄 Run Comparison", type="primary", key="compare_analyze") and query2 and previous_data:
            run_analysis(query2, use_web2, previous_data)

def run_analysis(query: str, use_web: bool, previous_data: Optional[dict]):
    """Run analysis with optional evolution comparison"""
    
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
    
    base_conf = float(primary_data.get("confidence", 75))
    final_conf = calculate_final_confidence(base_conf, veracity.overall)
    
    # Evolution analysis (if comparing)
    evolution = None
    if previous_data:
        with st.spinner("📊 Analyzing evolution..."):
            current_data = {
                "question": query,
                "timestamp": datetime.now().isoformat(),
                "response": primary_data,
                "confidence": {"base": base_conf, "evidence": veracity.overall, "final": final_conf}
            }
            evolution = analyze_evolution(previous_data, current_data)
    
    # Build output
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
            "claims_verified": veracity.claims_verified
        },
        "sources": web_context.get("sources", [])
    }
    
    # Add evolution data if comparing
    if evolution:
        output["evolution"] = evolution.to_dict()
    
    # Download button
    st.download_button(
        "💾 Download Report",
        json.dumps(output, indent=2),
        f"yureeka_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "application/json"
    )
    
    # Render dashboard
    render_dashboard(response, final_conf, web_context, base_conf, query, veracity, evolution)
    
    # Debug info
    with st.expander("🔧 Debug"):
        st.json({
            "confidence": {"base": base_conf, "evidence": veracity.overall, "final": final_conf},
            "veracity": {k: round(getattr(veracity, k), 1) for k in 
                        ["source_quality", "claim_grounding", "numeric_accuracy", "source_agreement", "response_completeness", "overall"]}
        })
        if evolution:
            st.json(evolution.to_dict())

if __name__ == "__main__":
    main()

