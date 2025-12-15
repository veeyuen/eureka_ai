# =========================================================
# YUREEKA AI RESEARCH ASSISTANT v6.0 - REFACTORED
# With Web Search, Multi-Model Validation, Confidence Scoring
# =========================================================

import os
import re
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import base64
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from collections import Counter
import google.generativeai as genai
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
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
# 2. PYDANTIC MODELS (FIXED)
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
    """Standardized visualization data"""
    chart_labels: List[str] = Field(default_factory=list, description="X-axis labels")
    chart_values: List[Union[float, int]] = Field(default_factory=list, description="Y-axis values")
    chart_title: Optional[str] = Field("Trend Analysis", description="Chart title")
    chart_type: Optional[str] = Field("line", description="Chart type")
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
    value_1: Union[float, int]
    value_2: Union[float, int]
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

REQUIRED FIELDS (provide substantive data):
- executive_summary: Write 4-6 sentences covering:
  * Direct answer to the question (1 sentence)
  * Market size/scale with specific numbers (1 sentence)
  * Key drivers or growth factors (1-2 sentences)
  * Future outlook or trend direction (1-2 sentence)
  Example: "The global EV battery market reached $58.3B in 2023. Strong growth is driven by falling lithium prices and government mandates. China dominates with 60% market share, followed by Europe at 25%. The market is projected to grow at 18.5% CAGR through 2030 as battery costs decline further."
  
- primary_metrics (3+ metrics with numbers)
- key_findings (3+ findings)
- top_entities (3+ companies/countries)
- trends_forecast (2+ trends)
- visualization_data (MUST have chart_labels and chart_values)

Even if web data is sparse, use your knowledge to provide complete analysis.

Output ONLY this JSON structure:
{RESPONSE_TEMPLATE}
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
# 5. JSON REPAIR FUNCTIONS (CRITICAL FIXES)
# =========================================================

def repair_llm_response(data: dict) -> dict:
    """
    Repair common LLM JSON structure issues:
    - Convert primary_metrics from list to dict
    - Ensure top_entities and trends_forecast are lists
    - Add missing required fields
    """
    
    # Fix primary_metrics: list → dict
    if "primary_metrics" in data and isinstance(data["primary_metrics"], list):
        new_metrics = {}
        for i, item in enumerate(data["primary_metrics"]):
            if isinstance(item, dict):
                # Generate unique key from name
                raw_name = item.get("name", f"metric_{i+1}")
                key = re.sub(r'[^a-z0-9_]', '', raw_name.lower().replace(" ", "_"))
                if not key:
                    key = f"metric_{i+1}"
                
                # Ensure uniqueness
                original_key = key
                j = 1
                while key in new_metrics:
                    key = f"{original_key}_{j}"
                    j += 1
                
                # Ensure required fields
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
    
    return data

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
# =========================================================

def classify_source_reliability(source: str) -> str:
    """Classify source as High/Medium/Low quality"""
    source = source.lower() if isinstance(source, str) else ""
    
    high = ["gov", "imf", "worldbank", "central bank", "fed", "ecb", "reuters", 
            "financial times", "wsj", "oecd", "bloomberg", "tradingeconomics"]
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
    """Calculate average source quality (0-100)"""
    if not sources:
        return 60.0
    
    weights = {"✅ High": 100, "⚠️ Medium": 60, "❌ Low": 30}
    scores = [weights.get(classify_source_reliability(s), 60) for s in sources]
    return sum(scores) / len(scores) if scores else 60.0

@st.cache_data(ttl=3600, show_spinner=False)
def search_serpapi(query: str, num_results: int = 5) -> List[Dict]:
    """Search Google via SerpAPI"""
    if not SERPAPI_KEY:
        return []
    
    # Smart query classification
    query_lower = query.lower()
    industry_kw = ["industry", "market", "sector", "size", "growth", "players"]
    macro_kw = ["gdp", "inflation", "unemployment", "interest", "fed"]
    
    if any(kw in query_lower for kw in industry_kw):
        search_terms = f"{query} market size growth trends 2024"
        tbm, tbs = "", ""
    elif any(kw in query_lower for kw in macro_kw):
        search_terms = f"{query} latest data"
        tbm, tbs = "nws", "qdr:m"
    else:
        search_terms = f"{query} finance economics"
        tbm, tbs = "nws", "qdr:m"
    
    params = {
        "engine": "google",
        "q": search_terms,
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "tbm": tbm,
        "tbs": tbs
    }
    
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
        
        # Sort for consistency
        results.sort(key=lambda x: (x.get("source", "").lower(), x.get("link", "")))
        return results[:num_results]
    
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
    search_results = search_serpapi(query, num_results=5)
    
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
# 7. LLM QUERY FUNCTIONS (FIXED)
# =========================================================

def query_perplexity(query: str, web_context: Dict, temperature: float = 0.1) -> str:
    """Query Perplexity API with web context"""
    
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
        "temperature": temperature,
        "max_tokens": 2000,
        "top_p": 0.8,
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
        
        # Validate with Pydantic
        try:
            llm_obj = LLMResponse.model_validate(repaired)
            
            # Merge web sources
            if web_context.get("sources"):
                existing = llm_obj.sources or []
                merged = list(dict.fromkeys(existing + web_context["sources"]))
                llm_obj.sources = merged[:10]
                llm_obj.freshness = "Current (web-enhanced)"
            
            return llm_obj.model_dump_json()
        
        except ValidationError as e:
            st.warning(f"⚠️ Pydantic validation failed: {e}")
            return create_fallback_response(query, search_count, web_context)
    
    except Exception as e:
        st.error(f"❌ Perplexity API error: {e}")
        return create_fallback_response(query, search_count, web_context)

def query_gemini(query: str) -> str:
    """Query Gemini API"""
    prompt = f"{SYSTEM_PROMPT}\n\nUser query: {query}"
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2000
            )
        )
        
        content = getattr(response, "text", None)
        if not content or not content.strip():
            raise Exception("Empty Gemini response")
        
        # Parse and repair
        parsed = parse_json_safely(content, "Gemini")
        if not parsed:
            return create_fallback_response(query, 0, {})
        
        repaired = repair_llm_response(parsed)
        
        try:
            llm_obj = LLMResponse.model_validate(repaired)
            return llm_obj.model_dump_json()
        except ValidationError:
            return create_fallback_response(query, 0, {})
    
    except Exception as e:
        st.warning(f"⚠️ Gemini API error: {e}")
        return create_fallback_response(query, 0, {})

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
# 8. VALIDATION & SCORING (FIXED)
# =========================================================

@st.cache_data
def get_embedding(text: str):
    """Cache embeddings for performance"""
    return embedder.encode(text)

def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity score (0-100)"""
    if not text1 or not text2:
        return 0.0
    
    try:
        v1 = get_embedding(text1)
        v2 = get_embedding(text2)
        sim = util.cos_sim(v1, v2)
        return round(float(sim.item()) * 100, 2)
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return 0.0

def numeric_alignment(metrics1: Dict, metrics2: Dict) -> Optional[float]:
    """Compare numeric metrics between two responses"""
    if not metrics1 or not metrics2:
        return None
    
    # Extract values from MetricDetail dicts
    v1_map = {k: v.get("value") for k, v in metrics1.items() if isinstance(v, dict)}
    v2_map = {k: v.get("value") for k, v in metrics2.items() if isinstance(v, dict)}
    
    total_diff = 0
    count = 0
    
    for key in v1_map:
        if key in v2_map:
            try:
                val1 = float(v1_map[key])
                val2 = float(v2_map[key])
                
                if val1 == 0 and val2 == 0:
                    continue
                
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    continue
                
                diff = abs(val1 - val2) / max_val
                total_diff += diff
                count += 1
            
            except (ValueError, TypeError):
                continue
    
    if count == 0:
        return None
    
    alignment = 1 - (total_diff / count)
    return round(alignment * 100, 2)

def compare_lists(list1: List[str], list2: List[str]) -> float:
    """Compare two lists of text using semantic similarity"""
    if not list1 or not list2:
        return 0.0
    
    scores = []
    for text1 in list1:
        best_match = max([semantic_similarity(text1, text2) for text2 in list2], default=0)
        scores.append(best_match)
    
    return round(np.mean(scores), 2) if scores else 0.0

def multi_modal_verification(json1: Dict, json2: Dict) -> Dict[str, float]:
    """
    Compare two LLM outputs across multiple dimensions:
    - Executive summaries (semantic)
    - Key findings (list similarity)
    - Primary metrics (numeric alignment)
    - Visualization data (chart similarity)
    """
    
    # Summary comparison
    summary_score = semantic_similarity(
        json1.get("executive_summary", ""),
        json2.get("executive_summary", "")
    )
    
    # Findings comparison
    findings_score = compare_lists(
        json1.get("key_findings", []),
        json2.get("key_findings", [])
    )
    
    # Metrics comparison
    metrics_score = numeric_alignment(
        json1.get("primary_metrics", {}),
        json2.get("primary_metrics", {})
    ) or 0.0
    
    # Visualization comparison
    viz1 = json1.get("visualization_data", {}) or {}
    viz2 = json2.get("visualization_data", {}) or {}
    
    labels1 = viz1.get("chart_labels", [])
    labels2 = viz2.get("chart_labels", [])
    values1 = viz1.get("chart_values", [])
    values2 = viz2.get("chart_values", [])
    
    viz_score = 0.0
    if labels1 == labels2 and values1 and values2 and len(values1) == len(values2):
        try:
            v1 = np.array([float(x) for x in values1])
            v2 = np.array([float(x) for x in values2])
            diff = np.abs(v1 - v2)
            max_vals = np.maximum(np.abs(v1), np.abs(v2))
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = np.where(max_vals != 0, diff / max_vals, 0)
            viz_score = max(0, 100 - np.nanmean(rel_diff) * 100)
        except:
            pass
    
    # Overall weighted score
    weights = {"summary": 0.3, "findings": 0.25, "metrics": 0.25, "viz": 0.2}
    overall = (
        summary_score * weights["summary"] +
        findings_score * weights["findings"] +
        metrics_score * weights["metrics"] +
        viz_score * weights["viz"]
    )
    
    return {
        "summary_score": summary_score,
        "findings_score": findings_score,
        "metrics_score": metrics_score,
        "viz_score": viz_score,
        "overall_score": round(overall, 2)
    }

def calculate_final_confidence(
    base_conf: float,
    sem_conf: float,
    num_conf: Optional[float],
    src_conf: float,
    veracity_conf: float
) -> float:
    """
    Calculate weighted final confidence score.
    Note: Veracity is kept separate as it's a cross-model comparison metric.
    """
    
    components = [
        ("base", base_conf, 0.30),
        ("semantic", sem_conf, 0.25),
        ("source", src_conf, 0.25),
        ("numeric", num_conf or 0, 0.20 if num_conf else 0)
    ]
    
    total_weight = sum(w for _, _, w in components)
    weighted_sum = sum(score * weight for _, score, weight in components)
    
    final = weighted_sum / total_weight if total_weight > 0 else 70.0
    return round(final, 2)

# =========================================================
# 9. DASHBOARD RENDERING (FIXED)
# =========================================================

def render_dashboard(
    primary_json: str,
    final_conf: float,
    sem_conf: float,
    num_conf: Optional[float],
    web_context: Dict,
    base_conf: float,
    src_conf: float,
    user_question: str,
    secondary_json: Optional[str] = None,
    veracity_scores: Optional[Dict] = None,
    show_secondary: bool = False
):
    """Render complete analysis dashboard"""
    
    # Parse primary response
    try:
        data = json.loads(primary_json)
    except Exception as e:
        st.error(f"❌ Cannot render dashboard: {e}")
        st.code(primary_json[:1000])
        return
    
    # Header
    st.header("📊 Yureeka Market Intelligence")
    st.markdown(f"**Question:** {user_question}")
    
    # Confidence metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Confidence", f"{final_conf:.1f}%")
    col2.metric("Base Model", f"{base_conf:.1f}%")
    col3.metric("Semantic", f"{sem_conf:.1f}%")
    if num_conf:
        col4.metric("Numeric", f"{num_conf:.1f}%")
    else:
        col4.metric("Numeric", "N/A")
    
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
    
    # Visualization
    st.subheader("📊 Data Visualization")
    viz = data.get('visualization_data')
    
    if viz and isinstance(viz, dict):
        labels = viz.get("chart_labels", [])
        values = viz.get("chart_values", [])
        title = viz.get("chart_title", "Market Trend")
        chart_type = viz.get("chart_type", "line")
        
        if labels and values and len(labels) == len(values):
            try:
                df_viz = pd.DataFrame({
                    "Category": labels[:10],
                    "Value": [float(v) for v in values[:10]]
                })
                
                if chart_type == "bar":
                    fig = px.bar(df_viz, x="Category", y="Value", title=title)
                else:
                    fig = px.line(df_viz, x="Category", y="Value", title=title, markers=True)
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Chart data available but rendering failed: {e}")
    
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
    
    # Sources
    sources = data.get('sources', [])
    if sources:
        st.subheader("🔗 Sources")
        for i, src in enumerate(sources[:10], 1):
            if src and src.startswith("http"):
                reliability = classify_source_reliability(src)
                st.markdown(f"**{i}.** [{src}]({src}) {reliability}")
    
    # Metadata
    col_fresh, col_action = st.columns(2)
    with col_fresh:
        freshness = data.get('freshness', 'Current')
        st.metric("Data Freshness", freshness)
    
    with col_action:
        action = data.get('action')
        if action and isinstance(action, dict):
            rec = action.get("recommendation", "Neutral")
            conf = action.get("confidence", "Medium")
            st.metric("Recommendation", f"{rec} ({conf})")
    
    st.markdown("---")
    
    # Veracity Scores
    if veracity_scores:
        st.subheader("✅ Cross-Model Verification")
        cols = st.columns(5)
        metrics = [
            ("Summary", "summary_score"),
            ("Findings", "findings_score"),
            ("Metrics", "metrics_score"),
            ("Viz", "viz_score"),
            ("Overall", "overall_score")
        ]
        for i, (label, key) in enumerate(metrics):
            cols[i].metric(label, f"{veracity_scores.get(key, 0):.1f}%")
    
    # Secondary Model (Optional)
    if show_secondary and secondary_json:
        with st.expander("🔍 Secondary Model Output (Gemini)"):
            try:
                sec_data = json.loads(secondary_json)
                st.json(sec_data)
            except:
                st.code(secondary_json[:2000])
    
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
        page_title="Yureeka Research Assistant",
        page_icon="💹",
        layout="wide"
    )
    
    st.title("💹 Yureeka AI Research Assistant")
    
    # Info section
    col_info, col_status = st.columns([3, 1])
    with col_info:
        st.markdown("""
        **Yureeka** provides AI-powered market research and analysis for finance, 
        economics, and business questions. 
        Powered by multi-model verification and real-time web search.
        
        *Currently in prototype stage.*
        """)

  # UNCOMMENT SECTION BELOW TO SHOW WEB SEARCH ENABLED
  #  with col_status:
  #      web_status = "✅ Enabled" if SERPAPI_KEY else "⚠️ Disabled"
  #      st.metric("Web Search", web_status)
    
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
    
    with col_opt2:
        show_secondary = st.checkbox("Show secondary model output", value=False)
    
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
        
        # Secondary model validation
        with st.spinner("✅ Validating with secondary model..."):
            secondary_response = query_gemini(query)
        
        try:
            secondary_data = json.loads(secondary_response)
        except:
            secondary_data = {}
        
        # Calculate verification scores
        veracity_scores = multi_modal_verification(primary_data, secondary_data)
        
        # Calculate confidence scores
        base_conf = float(primary_data.get("confidence", 75))
        sem_conf = semantic_similarity(
            primary_data.get("executive_summary", ""),
            secondary_data.get("executive_summary", "")
        )
        num_conf = numeric_alignment(
            primary_data.get("primary_metrics", {}),
            secondary_data.get("primary_metrics", {})
        )
        src_conf = source_quality_score(primary_data.get("sources", []))
        
        final_conf = calculate_final_confidence(
            base_conf, sem_conf, num_conf, src_conf,
            veracity_scores["overall_score"]
        )
        
        # Download JSON
        output = {
            "question": query,
            "timestamp": datetime.now().isoformat(),
            "primary_response": primary_data,
            "secondary_response": secondary_data,
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
            sem_conf,
            num_conf,
            web_context,
            base_conf,
            src_conf,
            query,
            secondary_response,
            veracity_scores,
            show_secondary
        )
        
        # Debug info
        with st.expander("🔧 Debug Information"):
            st.write("**Confidence Breakdown:**")
            st.json({
                "base_confidence": base_conf,
                "semantic_similarity": sem_conf,
                "numeric_alignment": num_conf,
                "source_quality": src_conf,
                "final_confidence": final_conf
            })
            
            st.write("**Primary Model Response:**")
            st.json(primary_data)
            
            if show_secondary:
                st.write("**Secondary Model Response:**")
                st.json(secondary_data)

if __name__ == "__main__":
    main()
