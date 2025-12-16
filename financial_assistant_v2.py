# =========================================================
# YUREEKA AI RESEARCH ASSISTANT v7.0 - CORRECTED
# With Web Search, Evidence-Based Verification, Confidence Scoring
# =========================================================

import os
import re
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import base64
from typing import Dict, List, Optional, Any, Union
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import google.generativeai as genai
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
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
# =========================================================

def classify_source_reliability(source: str) -> str:
    """Classify source as High/Medium/Low quality"""
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
    """Calculate average source quality (0-100)"""
    if not sources:
        return 50.0  # Lower default when no sources
    
    weights = {"✅ High": 100, "⚠️ Medium": 60, "❌ Low": 30}
    scores = [weights.get(classify_source_reliability(s), 60) for s in sources]
    return sum(scores) / len(scores) if scores else 50.0

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
# 7. LLM QUERY FUNCTIONS
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

        validate_numeric_fields(repaired, "Gemini")
        
        try:
            llm_obj = LLMResponse.model_validate(repaired)
            return llm_obj.model_dump_json()
        except ValidationError:
            return create_fallback_response(query, 0, {})
    
    except Exception as e:
        st.info("Secondary model response unavailable")
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
# 8. VALIDATION & SCORING
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

if __name__ == "__main__":
    main()
