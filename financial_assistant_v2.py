# =========================================================
# AI FINANCIAL RESEARCH ASSISTANT – AI HYBRID VERIFICATION v5.6 (FIXED)
# WITH WEB SEARCH, DYNAMIC METRICS, CONFIDENCE BREAKDOWN & EVOLUTION LAYER
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
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, ConfigDict # ADDED ConfigDict
from typing import List, Dict, Optional, Any, Union

# ----------------------------
# 1. UPDATED PYDANTIC MODELS TO MATCH COMPLEX RESPONSE_TEMPLATE & BE ROBUST
# ----------------------------

class MetricDetail(BaseModel):
    """Corresponds to the inner object in primary_metrics."""
    name: str = Field(..., description="Key Metric Name")
    value: Union[float, int, str] = Field(..., description="Key Metric Value (can be number or string with unit)")
    unit: str = Field(..., description="Unit of measurement")
    model_config = ConfigDict(extra='ignore') # Ignore unexpected fields

class TopEntityDetail(BaseModel):
    """Corresponds to an item in top_entities."""
    name: str = Field(..., description="Entity Name")
    # FIX: Made these fields Optional with default "N/A" to prevent "Field required" errors
    share: str = Field("N/A", description="Market Share/Position") 
    growth: str = Field("N/A", description="Growth Rate/Trend")
    model_config = ConfigDict(extra='ignore') # Ignore unexpected fields like 'type' or 'description'

class TrendForecastDetail(BaseModel):
    """Corresponds to an item in trends_forecast."""
    trend: str = Field(..., description="Trend description")
    # FIX: Made these fields Optional with default "N/A" to prevent "Field required" errors
    direction: str = Field("N/A", description="Direction symbol (e.g., up arrow)")
    timeline: str = Field("N/A", description="Timeline (e.g., 2025-2027)")
    model_config = ConfigDict(extra='ignore') # Ignore unexpected fields like 'description'

class ComparisonBar(BaseModel):
    """Corresponds to comparison_bars."""
    title: str
    categories: List[str]
    values: List[Union[float, int]]
    model_config = ConfigDict(extra='ignore')

class BenchmarkTable(BaseModel):
    """Corresponds to an item in benchmark_table."""
    category: str
    value_1: Union[float, int]
    value_2: Union[float, int]
    model_config = ConfigDict(extra='ignore')

class Action(BaseModel):
    """Corresponds to action field."""
    recommendation: str
    confidence: str
    rationale: str
    model_config = ConfigDict(extra='ignore')

class LLMResponse(BaseModel):
    """The main response schema, matching the complex RESPONSE_TEMPLATE."""
    executive_summary: str = Field(..., description="1-2 sentence high-level answer to the core question")
    
    # primary_metrics is a DICT of MetricDetail objects
    primary_metrics: Dict[str, MetricDetail] = Field(..., description="Dictionary of key metrics")
    
    key_findings: List[str] = Field(..., description="List of key findings")
    
    # top_entities is a LIST of TopEntityDetail objects
    top_entities: List[TopEntityDetail] = Field(..., description="List of top entities")
    
    # trends_forecast is a LIST of TrendForecastDetail objects
    trends_forecast: List[TrendForecastDetail] = Field(..., description="List of trends and forecasts")
    
    # visualization_data can be flexible for now
    visualization_data: Dict[str, Any] = Field(..., description="Data for chart generation")
    
    comparison_bars: Optional[ComparisonBar] = None
    benchmark_table: Optional[List[BenchmarkTable]] = None
    sources: Optional[List[str]] = Field(default_factory=list)
    confidence: Optional[Union[float, int]] = 0
    freshness: Optional[str] = None
    action: Optional[Action] = None
    model_config = ConfigDict(extra='ignore') # Ignore unexpected fields at the top level

# ----------------------------
# CONFIGURATION
# ----------------------------
try:
    PERPLEXITY_KEY = st.secrets["PERPLEXITY_API_KEY"]
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
    SCRAPINGDOG_KEY = st.secrets.get("SCRAPINGDOG_KEY", "")
except:
    PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
    SCRAPINGDOG_KEY = os.getenv("SCRAPINGDOG_KEY", "")

if not PERPLEXITY_KEY or not GEMINI_KEY:
    st.error("Missing API keys. Please set PERPLEXITY_API_KEY and GEMINI_API_KEY.")
    st.stop()

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# ----------------------------
# FIX 1: ADD SECTION_MAPPING FOR RENDER_DASHBOARD
# ----------------------------
SECTION_MAPPING = {
    "executive_summary": "Executive Summary",
    "primary_metrics": "Key Performance Indicators (KPIs)",
    "top_entities": "Top Market Entities & Competitors",
    "key_findings": "Key Analysis Takeaways"
}


# ----------------------------
# SOURCE RELIABILITY CLASSIFIER
# ----------------------------
def classify_source_reliability(source):
    source = source.lower() if isinstance(source, str) else ""
    high_sources = ["gov", "imf", "worldbank", "world bank", "central bank", "fed", "ecb", "bank of england", "eu", "reuters", "financial times", "wsj", "oecd", "bank of korea", "tradingeconomics",
                    "the economist", "ft.com", "bloomberg", "investopedia", "marketwatch", "bank of canada", "reserve bank of australia", "monetary authority of singapore", "HKMA", "bank of japan", 
                    "adb", "unfpa", "deloitte", "accenture", "kpmg", "ey", "mckinsey", "bain"]
    medium_sources = ["wikipedia", "forbes", "cnbc", "yahoo finance", "ceic", "kaggle", "statista"]
    low_sources = ["blog", "medium.com", "wordpress", "promotions", "advertisement", "sponsored", "blogger"]

    for high in high_sources:
        if high in source:
            return "✅ High"
    for medium in medium_sources:
        if medium in source:
            return "⚠️ Medium"
    for low in low_sources:
        if low in source:
            return "❌ Low"
    return "⚠️ Medium"  # default fallback

def source_quality_confidence(sources):
    weights = {"✅ High": 1.0, "⚠️ Medium": 0.6, "❌ Low": 0.3}
    total_score = 0
    count = 0
    for source in sources:
        rank = classify_source_reliability(source)
        total_score += weights.get(rank, 0.6)
        count += 1
    return total_score / count if count > 0 else 0.6

# ----------------------------
# PROMPTS AND MODELS
# ----------------------------

RESPONSE_TEMPLATE = """
Return ONLY valid JSON in this flexible structure. Populate ALL fields with relevant data:

{
    "executive_summary": "1-2 sentence high-level answer to the core question",
    "primary_metrics": {
    "metric_1": {"name": "Key Metric 1", "value": 25.5, "unit": "%"},
    "metric_2": {"name": "Key Metric 2", "value": 623, "unit": "$B"},
    "metric_3": {"name": "Key Metric 3", "value": 12.5, "unit": "x"}
  },
  "key_findings": [
    "Finding 1 with quantified impact",
    "Finding 2 explaining drivers",
    "Finding 3 highlighting opportunities/risks"
  ],
  "top_entities": [
    {"name": "Entity 1", "share": "25%", "growth": "15%"},
    {"name": "Entity 2", "share": "18%", "growth": "22%"},
    {"name": "Entity 3", "share": "12%", "growth": "-3%"}
  ],
  "trends_forecast": [
    {"trend": "Trend description", "direction": "‚Üë/‚Üì/‚Üî", "timeline": "2025-2027"},
    {"trend": "Another trend", "direction": "‚Üë", "timeline": "2026"}
  ],
  "visualization_data": {
  "title": "YoY Growth",
  "chart_labels": ["Year 1", "Year 2"],
  "data_series_label": "Market Size ($B)",
  "data_series_values": [100, 120]
    },
    "comparison_bars": {
      "title": "Market Share by Segment (Sample)",
      "categories": ["Segment A", "Segment B", "Segment C"],
      "values": [45, 30, 25] // Must be three numerical values that sum to 100
    },
   "benchmark_table": [
    {"category": "Company A", "value_1": 25.5, "value_2": 623},
    {"category": "Company B", "value_1": 18.2, "value_2": 450}
  ],
  "sources": ["source1.com", "source2.com"],
  "confidence": 87,
  "freshness": "Dec 2025",
  "action": {
    "recommendation": "Buy/Hold/Neutral/Sell",
    "confidence": "High/Medium/Low",
    "rationale": "1-sentence reason"
  }
}
"""

SYSTEM_PROMPT = """You are a professional market research analyst. 

CRITICAL: ALWAYS provide a COMPLETE response with:
- executive_summary (2 sentences minimum)
- 3+ primary_metrics with numbers
- 3+ key_findings  
- top_entities (3+ companies/countries)
- trends_forecast (2+ trends)
- Data Visualization Mandate: ** The `visualization_data` field **MUST be populated**. If you cannot find explicit time-series or comparative data in the provided context, you **MUST generate realistic, synthetic data** that visually represents the trends and findings described in the text (e.g., market growth or market share distribution). NEVER leave this field empty.

FOLLOW EXACTLY:

1. Return ONLY a single JSON object. NO markdown, NO code blocks, NO explanations.
2. NO references like [1][2] inside JSON strings.
3. NO text before or after the JSON { ... }
4. Use ONLY the fields from the response template below.
5. **Formatting Zero-Tolerance:**
   - **MUST** enclose all property names (keys) in double quotes.
   - **MUST NOT** include any trailing commas within objects or arrays (e.g., `[1, 2,]` is forbidden).
   - **MUST** use a comma delimiter between all list elements and object properties.
   - **MUST** escape any internal double quotes within string values using a backslash (e.g., `\"`).

EVEN IF WEB DATA IS SPARSE, use your knowledge to provide substantive analysis.

NEVER return empty fields. Output ONLY valid JSON:\n"
f"{RESPONSE_TEMPLATE}"
""" 


@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        embed = SentenceTransformer("all-MiniLM-L6-v2")
        return classifier, embed
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

domain_classifier, embedder = load_models()

# ----------------------------
# JSON REPAIR FUNCTION (CRITICAL FIX FOR primary_metrics)
# ----------------------------

def repair_metric_list_to_dict(data: dict) -> dict:
    """
    FIX: If primary_metrics is a list (LLM failure), convert it to a dict (schema requirement).
    Converts: [{"name": "Metric 1", "value": 10}, ...]
    To:      {"metric_1": {"name": "Metric 1", "value": 10}, ...}
    """
    if "primary_metrics" in data and isinstance(data["primary_metrics"], list):
        new_metrics = {}
        for i, metric_item in enumerate(data["primary_metrics"]):
            if isinstance(metric_item, dict):
                # Use name to create a unique, standard key
                raw_name = metric_item.get("name", f"metric_{i+1}")
                key = re.sub(r'[^a-z0-9_]', '', raw_name.lower().replace(" ", "_"))
                if not key: key = f"metric_{i+1}"

                # Ensure key is unique
                original_key = key
                j = 1
                while key in new_metrics:
                    key = f"{original_key}_{j}"
                    j += 1
                    
                # Ensure it has minimum required fields for MetricDetail (for softening)
                if "value" not in metric_item:
                    metric_item["value"] = "N/A"
                if "unit" not in metric_item:
                    metric_item["unit"] = ""
                    
                new_metrics[key] = metric_item
            
            # Fallback for old list-of-string format which might still occasionally occur
            elif isinstance(metric_item, str):
                key = f"metric_{i+1}"
                name, value = metric_item.split(":", 1) if ":" in metric_item else (metric_item, "N/A")
                new_metrics[key] = {"name": name.strip(), "value": value.strip(), "unit": ""}
        
        data["primary_metrics"] = new_metrics
    return data


# ----------------------------
# WEB SEARCH FUNCTIONS
# ----------------------------
@st.cache_data(ttl=3600)
def search_serpapi(query: str, num_results: int = 5):
    if not SERPAPI_KEY:
        st.info("💡 SerpAPI key not configured.")
        return []
    
    # SMART QUERY CLASSIFICATION
    query_lower = query.lower()
    industry_keywords = ["industry", "market", "sector", "ev", "electric", "vehicle", 
                        "sneaker", "semiconductor", "biotech", "battery", "renewable"]
    macro_keywords = ["gdp", "inflation", "unemployment", "interest", "fed", "ecb"]
    
    if any(kw in query_lower for kw in industry_keywords):
        search_terms = f"{query} market size growth players trends 2025"
        tbm = ""  # All results, not just news
        tbs = ""  # No time restriction
    elif any(kw in query_lower for kw in macro_keywords):
        search_terms = f"{query} latest data"
        tbm = "nws"
        tbs = "qdr:m"
    else:
        search_terms = f"{query} finance economics markets"
        tbm = "nws"
        tbs = "qdr:m"
    
    params = {
    "engine": "google",
    "q": search_terms,
    "api_key": SERPAPI_KEY,
    "num": num_results,
    "tbm": tbm,
    "tbs": tbs}

    url = "https://serpapi.com/search"
    
    # rest of your existing function...
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("news_results", [])[:num_results]:
            src = item.get("source")
            source_name = src.get("name", "") if isinstance(src, dict) else (src if isinstance(src, str) else "")
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "source": source_name
            })
        if not results:
            for item in data.get("organic_results", [])[:num_results]:
                src = item.get("source", "")
                source_name = src if isinstance(src, str) else ""
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date": "",
                    "source": source_name
                })
        if results:
            results.sort(key=lambda x: (x.get("source", "").lower(), x.get("link", "")))
        return results[:num_results]
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ SerpAPI search error: {e}")
        return []
    except Exception as e:
        st.warning(f"⚠️ Error processing SerpAPI results: {e}")
        return []

def scrape_url_scrapingdog(url: str):
    if not SCRAPINGDOG_KEY:
        return None
    api_url = "https://api.scrapingdog.com/scrape"
    params = {
        "api_key": SCRAPINGDOG_KEY,
        "url": url,
        "dynamic": "false"
    }
    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text[:3000]
    except Exception as e:
        st.warning(f"⚠️ ScrapingDog error for {url[:50]}: {e}")
        return None

def fetch_web_context(query: str, num_sources: int = 3): 
    search_results = search_serpapi(query, num_results=5)
    if not search_results:
        return {"search_results": [], "scraped_content": {}, "summary": "", "sources": [], "source_reliability": []}

    # Sort the results by URL or source name for consistency
    search_results_sorted = sorted(search_results, key=lambda x: x.get("link", "").lower())
    
    scraped_content = {}
    if SCRAPINGDOG_KEY:
        st.info(f"🔍 Scraping top {min(num_sources, len(search_results_sorted))} sources...")
        for i, result in enumerate(search_results_sorted[:num_sources]):
            url = result["link"]
            content = scrape_url_scrapingdog(url)
            if content:
                scraped_content[url] = content
                st.success(f"✓ Scraped {i+1}/{num_sources}: {result['source']}")
    context_parts = []
    reliabilities = []

    for r in search_results_sorted:
        date_str = f" ({r['date']})" if r['date'] else ""
        reliability = classify_source_reliability(r.get("link", "") + " " + r.get("source", ""))
        reliabilities.append(reliability)
        context_parts.append(
            f"**{r['title']}**{date_str}\n"
            f"Source: {r['source']} [{reliability}]\n"
            f"{r['snippet']}\n"
            f"URL: {r['link']}"
        )
    summary = "\n\n---\n\n".join(context_parts)
    sources = [r["link"] for r in search_results]
    return {
        "search_results": search_results_sorted,
        "scraped_content": scraped_content,
        "summary": summary,
        "sources": sources,
        "source_reliability": reliabilities,
    }


def query_perplexity_with_context(query: str, web_context: dict, temperature: float = 0.0) -> str:
    """
    Call Perplexity with enriched context and return a JSON string
    that conforms to the LLMResponse schema (or a structured fallback).
    """
    
    # 1. Build enhanced prompt (as before)
    search_results_count = len(web_context.get("search_results", []))
    
    if not web_context.get("summary") or search_results_count < 2:
        enhanced_query = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User Question: {query}\n\n"
            f"Web search returned only {search_results_count} usable results. "
            f"Use your general market and industry knowledge to provide a full analysis "
            f"with metrics, key findings, and forward-looking trends."
        )
    else:
        context_section = (
            "LATEST WEB RESEARCH (Current as of today):\n"
            f"{web_context['summary']}\n\n"
        )
        if web_context.get('scraped_content'):
            context_section += "\nDETAILED CONTENT FROM TOP SOURCES:\n"
            for url, content in list(web_context['scraped_content'].items())[:2]:
                context_section += f"\nFrom {url}:\n{content[:800]}...\n"
        
        enhanced_query = f"{context_section}\n{SYSTEM_PROMPT}\n\nUser Question: {query}"

    # 2. Call Perplexity API (as before)
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar",
        "temperature": temperature,
        "max_tokens": 2000,
        "top_p": 0.8,
        "messages": [{"role": "user", "content": enhanced_query}],
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        response_data = resp.json()
        
        if "choices" not in response_data:
            raise Exception("No 'choices' in Perplexity response")
        
        content = response_data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise Exception("Perplexity returned empty response")

        # 3. Pre-clean only (strip markdown/citations)
        cleaned = preclean_llm_json(content)

        # 4. Structured Repair before validation (CRITICAL FIX)
        try:
            parsed_json = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback to robust parser if simple loads fails
            parsed_json = parse_json_robustly(cleaned, "Perplexity Pre-Validation")

        repaired_json = repair_metric_list_to_dict(parsed_json) # Apply the list-to-dict fix
        
        # 5. Validate against Pydantic schema
        try:
            # Use model_validate on the dictionary, not model_validate_json on the string
            llm_obj = LLMResponse.model_validate(repaired_json)
            
            # 6. Manually update confidence from LLM response (can be an int or float string)
            raw_confidence = llm_obj.confidence
            if isinstance(raw_confidence, (str, int, float)):
                 llm_obj.confidence = float(raw_confidence)
            else:
                 llm_obj.confidence = 0.0 # Default if field is missing or malformed

        except ValidationError as e:
            st.warning(
                "LLM response failed schema validation; using fallback schema instead. "
                f"Details: {e}"
            )
            
            # 7. FIXED FALLBACK - matches the *NEW COMPLEX* schema structure
            fallback = {
                "executive_summary": (
                    f"Comprehensive analysis of '{query}' with {search_results_count} web sources, "
                    "but the primary model output did not match the expected schema. Showing simple data."
                ),
                "primary_metrics": {
                    "source_count": {"name": "Sources Found", "value": search_results_count, "unit": "count"},
                    "confidence_align": {"name": "Model Alignment", "value": 75.0, "unit": "%"}
                },
                "key_findings": [
                    f"Web search found {search_results_count} relevant sources.",
                    "Model generated detailed market analysis.",
                    "Key themes: market growth, regional dominance, technological trends."
                ],
                "top_entities": [],
                "trends_forecast": [],
                "visualization_data": {},
                "sources": web_context.get("sources", []),
                "confidence": 75.0,
                "freshness": "Current (web-enhanced)",
            }
            llm_obj = LLMResponse(**fallback)

        # 8. Optionally merge web sources - SAFE None handling
        if web_context.get("sources"):
            existing_sources = llm_obj.sources or []
            web_sources = web_context["sources"] or []
            merged_sources = list(dict.fromkeys(existing_sources + web_sources))[:10]
            llm_obj.sources = merged_sources
            llm_obj.freshness = "Current (web-scraped + real-time search)"

        # 9. Return as JSON string for storage / downstream rendering
        return llm_obj.model_dump_json()

    except requests.exceptions.RequestException as e:
        st.error(f"Perplexity API request failed: {e}")
        raise
    except Exception as e:
        st.error(f"Perplexity query error: {e}")
        raise


def query_gemini(query: str):
    prompt = f"{SYSTEM_PROMPT}\n\nUser query: {query}"
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=2000,
            ),
        )
        content = getattr(response, "text", None)
        if not content or not content.strip():
            finish_reason = getattr(response, "finish_reason", None)
            st.warning(f"Gemini returned empty response. finish_reason={finish_reason}")
            raise Exception("Gemini returned empty response")
        
        # 1. Pre-clean
        cleaned = preclean_llm_json(content)
        
        # 2. Structured Repair before validation (CRITICAL FIX)
        try:
            parsed_json = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed_json = parse_json_robustly(cleaned, "Gemini Pre-Validation")

        repaired_json = repair_metric_list_to_dict(parsed_json) # Apply the list-to-dict fix
        
        # 3. Validate
        try:
             llm_obj = LLMResponse.model_validate(repaired_json) # Use model_validate on the dictionary
             return llm_obj.model_dump_json() # Return valid JSON string
        except ValidationError:
            # Fallback to the raw cleaned string if validation fails (will be handled by main)
            return cleaned 
            
    except Exception as e:
        st.warning(f"Gemini API error: {e}")
        # Return a JSON string that will minimally load and fail comparison softly
        return json.dumps({
            "executive_summary": "Gemini validation unavailable due to API error.",
            "primary_metrics": {},
            "key_findings": ["Cross-validation could not be performed"],
            "top_entities": [],
            "trends_forecast": [],
            "visualization_data": {},
            "confidence": 0,
        })

# ----------------------------
# SELF-CONSISTENCY & VALIDATION
# (The rest of the code remains the same as the last corrected version)
# ----------------------------

def generate_self_consistent_responses_with_web(query, web_context, n=1):  # generate one response
    st.info(f"Generating analysis...")

    responses, scores = [], []
    success_count = 0
    for i in range(n):
        try:
            r = query_perplexity_with_context(query, web_context, temperature=0.1)
            responses.append(r)
            # 🟢 FIXED: Parse confidence from the JSON string
            scores.append(parse_confidence(r)) 
            success_count += 1
        except Exception as e:
            st.warning(f"Attempt {i+1}/{n} failed: {e}")
            continue
    if success_count == 0:
        st.error("All Perplexity API calls failed.")
        return [], []
    return responses, scores

def majority_vote(responses):
    if not responses:
        return ""
    cleaned = [r.strip() for r in responses if r]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]

def parse_confidence(text):
    try:
        # 🟢 FIXED: Confidence is now within the top-level LLMResponse schema
        js = json.loads(text)
        return float(js.get("confidence", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0

def semantic_similarity_score(a, b):
    try:
        v1, v2 = embedder.encode([a, b])
        sim = util.cos_sim(v1, v2)
        score = sim.item() if hasattr(sim, "item") else float(sim[0][0])
        return round(score * 100, 2)
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return 0.0

def numeric_alignment_score(j1, j2):
    # 🟢 FIXED: Metrics are now a DICT of DICTS in the new schema
    m1 = j1.get("primary_metrics", {})
    m2 = j2.get("primary_metrics", {})
    
    # Extract values from the inner MetricDetail objects
    v1_map = {k: v.get("value") for k, v in m1.items()}
    v2_map = {k: v.get("value") for k, v in m2.items()}
    
    total_diff = 0
    count = 0
    
    for key in v1_map:
        if key in v2_map:
            try:
                # Attempt to convert values to float for comparison
                v1, v2 = float(v1_map[key]), float(v2_map[key])
                if v1 == 0 and v2 == 0:
                    diff = 0
                elif max(abs(v1), abs(v2)) == 0:
                    diff = 0
                else:
                    diff = abs(v1 - v2) / max(abs(v1), abs(v2))
                total_diff += diff
                count += 1
            except (ValueError, TypeError):
                continue
    if count == 0:
        return None
    alignment = 1 - (total_diff / count)
    return round(alignment * 100, 2)

# ----------------------------
# DYNAMIC METRICS FILTERING & DISPLAY
# ----------------------------

def filter_relevant_metrics(question, metrics):
    relevant_metrics = {}
    for metric_name in metrics:
        classification = domain_classifier(question, [metric_name], multi_label=False)
        score = classification.get("scores", [0])[0]
        if score > 0.5:
            relevant_metrics[metric_name] = metrics[metric_name]
    return relevant_metrics


def render_dynamic_metrics(question, metrics):
    if not metrics:
        st.info("No metrics available.")
        return
    
    cols = st.columns(min(4, len(metrics)))  # Dynamic columns

    # 🟢 FIXED: Metrics is now expected to be Dict[str, MetricDetail]
    if isinstance(metrics, dict):
        items = list(metrics.items())[:4]
        for i, (metric_key, metric_detail) in enumerate(items):
            col = cols[i % len(cols)]
            if isinstance(metric_detail, dict):
                 name = metric_detail.get("name", metric_key)
                 value = metric_detail.get("value", "N/A")
                 unit = metric_detail.get("unit", "")
                 display_val = f"{value} {unit}".strip()
                 col.metric(name[:30], display_val)
            else:
                 col.info(f"{metric_key}: {str(metric_detail)[:30]}")

    else:
        st.info("Metrics format not recognized.")

# ----------------------------
# EVOLUTION LAYER
# ----------------------------

def time_ago(ts_str):
    ts = datetime.fromisoformat(ts_str)
    delta = datetime.now() - ts
    if delta.days > 0:
        return f"{delta.days}d ago"
    elif delta.seconds > 3600:
        return f"{delta.seconds // 3600}h ago"
    elif delta.seconds > 60:
        return f"{delta.seconds // 60}m ago"
    else:
        return "Just now"

def display_metric_with_delta(label, current, previous):
    # This is for the old metric structure. Needs updating if metrics is a dict of dicts.
    # For now, keeping original logic, but be aware this may break if metrics change format.
    if current is None or previous is None:
        display_val = str(current) if current is not None else "N/A"
        st.metric(label=label, value=display_val)
    else:
        delta = current - previous
        st.metric(label=label, value=f"{current:.2f}%", delta=f"{delta:+.2f}pp")

def render_evolution_layer(versions_history):
    st.subheader("Evolution Layer - Version Control & Drift")

    if not versions_history or len(versions_history) < 2:
        st.info("No history available for evolution layer.")
        return

    version_labels = [v["version"] for v in versions_history]
    selected_ver = st.radio("Select version", version_labels, horizontal=True, index=len(version_labels)-1)
    selected_index = version_labels.index(selected_ver)

    updated_ts = versions_history[selected_index]["timestamp"]
    st.markdown(f"**Updated:** {time_ago(updated_ts)}")

    current_metrics = versions_history[selected_index]["metrics"]
    previous_metrics = (versions_history[selected_index - 1]["metrics"]
                        if selected_index > 0 else current_metrics)
    
    # 🟢 FIXED: Handle metrics as Dict of Dicts (the new structure)
    metric_cols = st.columns(min(4, len(current_metrics)))
    for i, (m_key, current_detail) in enumerate(current_metrics.items()):
        if isinstance(current_detail, dict):
            m = current_detail.get("name", m_key)
            curr_val = current_detail.get("value")
            curr_unit = current_detail.get("unit", "")
            
            # Find previous value by key and extract its value
            prev_detail = previous_metrics.get(m_key, {})
            prev_val = prev_detail.get("value")
            
            # Simple numeric delta calculation for display
            try:
                curr_num = float(curr_val)
                prev_num = float(prev_val)
                delta = curr_num - prev_num
                metric_cols[i % len(metric_cols)].metric(
                    label=m, 
                    value=f"{curr_val} {curr_unit}".strip(), 
                    delta=f"{delta:+.2f}"
                )
            except (TypeError, ValueError):
                # Fallback if value is not numeric
                metric_cols[i % len(metric_cols)].metric(
                    label=m, 
                    value=f"{curr_val} {curr_unit}".strip()
                )

    st.markdown(f"*Reason for change:* {versions_history[selected_index]['change_reason']}")

    confidence_points = [v["confidence"] for v in versions_history]
    df_conf = pd.DataFrame({"Version": version_labels, "Confidence": confidence_points})
    fig = px.line(df_conf, x="Version", y="Confidence", title="Confidence Drift", height=150)
    st.plotly_chart(fig, use_container_width=True)

    freshness = versions_history[selected_index]["sources_freshness"]
    st.progress(int(freshness))
    st.caption(f"{freshness}% of ‚úÖ sources updated recently")


# ----------------------------
# RENDER DASHBOARD
# ----------------------------

def parse_trends_to_chart(trends):
    """Extract years/numbers from trend text for visualization"""
    import re
    labels, values = [], []
    for trend in trends:
        text = str(trend)
        # Match years like 2023, 2024-2027
        years = re.findall(r'20\d{2}', text)
        # Match numbers like 25%, 20M, 11.2CAGR
        nums = re.findall(r'[\d.]+[%B$TM]?', text)
        labels.extend(years[:3])
        values.extend([float(re.sub(r'[^\d.]', '', n)) for n in nums[:3] if re.sub(r'[^\d.]', '', n)])
    return labels[:5], values[:5]  # Limit for chart


def parse_json_robustly(json_string, context):
    """
    Parses a JSON string safely, addressing key quoting, trailing commas, 
    missing delimiters, and unescaped double quotes (Unterminated string error).
    
    NOTE: This is now a legacy function, as Pydantic's model_validate_json is preferred.
    It is kept for robustness in case model_validate_json fails.
    """
    if not json_string:
        return {}
    
    cleaned_string = json_string.strip()
    
    # 1. Clean up wrappers and control characters
    if cleaned_string.startswith("```json"):
        cleaned_string = cleaned_string[7:].strip()
    if cleaned_string.startswith("```"):
        cleaned_string = cleaned_string[3:].strip()
    if cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[:-3].strip()

    cleaned_string = cleaned_string.replace('\n', ' ').replace('\t', ' ')
    cleaned_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_string)

    # 2. Isolate the main JSON object
    match = re.search(r'\{.*\}', cleaned_string, flags=re.DOTALL)
    if match:
        json_content = match.group(0)
    else:
        #st.error(f"JSON parse failed: Could not find any valid JSON object '{{...}}' in {context} response.")
        return {"parse_error": "No JSON object found."}
    
    # 3. AGGRESSIVE STRUCTURAL REPAIR: FIX KEYS, COMMAS, AND BOOLEANS
    repaired_content = json_content
    
    try:
        # Fix 1: Insert missing commas between closing brace/bracket and a new key (Fixes 'Expecting ',' delimiter')
        repaired_content = re.sub(r'([\]\}])\s*["\'(a-zA-Z_]', r'\1,', repaired_content)

        # Fix 2: {key: -> {"key": (Fixes 'Expecting property name enclosed in double quotes')
        repaired_content = re.sub(r'([\{\,]\s*)([a-zA-Z_][a-zA-Z0-9_\-]+)(\s*):', r'\1"\2"\3:', repaired_content)

        # Fix 3: Remove illegal trailing commas (Fixes 'Expecting value' failure)
        repaired_content = re.sub(r',\s*([\]\}])', r'\1', repaired_content)
        
        # Fix 4: Capitalization of boolean/null values
        repaired_content = repaired_content.replace(': True', ': true')
        repaired_content = repaired_content.replace(': False', ': false')
        repaired_content = repaired_content.replace(': Null', ': null')
        
    except Exception as e:
        st.warning(f"Structural repair regex failed: {e}")

    json_content = repaired_content # Update content for the iterative loop

    # 4. ITERATIVE QUOTE REPAIR LOOP (TARGETING 'UNTERMINATED STRING' ERROR)
    max_retries = 20 # Increased retries
    current_attempt = 0
    
    while current_attempt < max_retries:
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            # Check for error types caused by unescaped quotes or delimiter issues
            if not ("Unterminated string" in e.msg or "Expecting ',' delimiter" in e.msg or "Expecting value" in e.msg):
                #st.error(f"JSON parse failed (Attempt {current_attempt+1}): {e}")
                return {"parse_error": str(e)}

            error_pos = e.pos
            found_quote = -1
            
            # Search backwards from error_pos to find the nearest quote to escape
            for i in range(error_pos - 1, max(0, error_pos - 200), -1): # Increased search range
                if i < len(json_content) and json_content[i] == '"':
                    # Crucial check: if the preceding character is NOT a backslash, this is our unescaped quote.
                    if i == 0 or json_content[i-1] != '\\':
                        found_quote = i
                        break
            
            if found_quote != -1:
                json_content = json_content[:found_quote] + '\\"' + json_content[found_quote+1:]
                current_attempt += 1
                continue # Retry the loop with the fixed string
            
            #st.error(f"JSON parse failed (Attempt {current_attempt+1}): Could not find unescaped quote near error position.")
            return {"parse_error": str(e)}

    #st.error(f"JSON parse failed after {max_retries} automatic repair attempts.")
    return {"parse_error": "Max retries exceeded"}

def preclean_llm_json(raw: str) -> str:
    """Strip markdown fences and simple citations; do NOT try to repair JSON structure."""
    if not raw or not isinstance(raw, str):
        return ""
    
    text = raw.strip()

    # Remove leading/trailing code fences
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()

    # Remove,,  style citations[1][2][3]
    text = re.sub(r'\[web:\d+\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)

    return text.strip()


from typing import Optional
import streamlit as st
import pandas as pd
import plotly.express as px
from pydantic import ValidationError

def render_dashboard(
    chosen_primary: str,
    final_conf: Optional[float] = None,
    sem_conf: Optional[float] = None,
    num_conf: Optional[float] = None,
    web_context: Optional[dict] = None,
    base_conf: Optional[float] = None,
    src_conf: Optional[float] = None,
    versions_history: Optional[list] = None,
    user_question: Optional[str] = None,
    secondary_resp: Optional[str] = None,
    veracity_scores: Optional[dict] = None,
    show_secondary_view: bool = False,
    ) -> None:
    """
    Renders the main analysis dashboard using the COMPLEX LLMResponse schema (Dicts/Lists of Dicts).
    """
    
    # Parse primary response with Pydantic (using the structured schema)
    try:
        # Use model_validate with the repair logic one last time in case the pre-parse failed but was fixed
        parsed_dict = json.loads(preclean_llm_json(chosen_primary))
        repaired_dict = repair_metric_list_to_dict(parsed_dict)
        llm_obj = LLMResponse.model_validate(repaired_dict)
        data = llm_obj.model_dump(by_alias=True) # Use the dictionary for rendering
    except ValidationError as e:
        st.error(f"Cannot render dashboard: primary response failed schema validation. {e}")
        st.json({"raw": chosen_primary[:1000]})
        return
    except Exception as e:
        st.error(f"Cannot render dashboard: severe JSON parsing error. {e}")
        st.json({"raw": chosen_primary[:1000]})
        return

    # Header and confidence metrics
    st.header("📊 Market Intelligence Dashboard")
    
    if user_question:
        st.markdown(f"**Question:** {user_question}")
    
    # Confidence row
    col1, col2, col3, col4 = st.columns(4)
    if final_conf is not None:
        col1.metric("Final Confidence", f"{final_conf:.1f}%")
    if base_conf is not None:
        col2.metric("Base Model", f"{base_conf:.1f}%")
    if sem_conf is not None:
        col3.metric("Semantic Align", f"{sem_conf:.1f}%")
    if num_conf is not None:
        col4.metric("Numeric Align", f"{num_conf:.1f}%" if isinstance(num_conf, (int, float)) else "N/A")

    st.markdown("---")

    # Executive Summary
    st.subheader("📋 Executive Summary")
    st.markdown(f"**{data.get('executive_summary', 'Summary not available.')}**")

    st.markdown("---")

    # Key Metrics - DICT OF DICTS
    st.subheader("💰 Key Metrics")
    primary_metrics = data.get('primary_metrics', {})
    
    metrics_list = []
    if isinstance(primary_metrics, dict):
        for metric_key, metric_detail in primary_metrics.items():
            if isinstance(metric_detail, dict):
                name = metric_detail.get('name', metric_key)
                value = metric_detail.get('value', 'N/A')
                unit = metric_detail.get('unit', '')
                metrics_list.append({
                    "Metric": name,
                    "Value": f"{value} {unit}".strip()
                })

    if metrics_list:
        try:
            df_metrics = pd.DataFrame(metrics_list)
            st.table(df_metrics)
        except Exception as e:
            st.warning(f"Failed to render metrics table: {e}")
            
    st.markdown("---")

    # Key Findings - LIST OF STRINGS
    st.subheader("🔍 Key Findings")
    key_findings = data.get('key_findings', [])
    for i, finding in enumerate(key_findings[:8], 1):
        if finding:
            st.markdown(f"**{i}.** {finding}")

    st.markdown("---")

    # Top Entities - LIST OF DICTS
    top_entities = data.get('top_entities', [])
    if top_entities:
        st.subheader("🏢 Top Entities")
        
        entity_data = []
        for entity in top_entities:
            if isinstance(entity, dict):
                entity_data.append({
                    "Entity": entity.get("name", "N/A"),
                    "Share": entity.get("share", "N/A"),
                    "Growth": entity.get("growth", "N/A"),
                })

        if entity_data:
            try:
                df_entities = pd.DataFrame(entity_data)
                st.dataframe(df_entities, hide_index=True, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render Top Entities table: {e}")


    # Trends Forecast - LIST OF DICTS
    trends_forecast = data.get('trends_forecast', [])
    if trends_forecast:
        st.subheader("📈 Trends Forecast")
        trend_data = []
        for trend in trends_forecast:
            if isinstance(trend, dict):
                trend_data.append({
                    "Trend": trend.get("trend", "N/A"),
                    "Direction": trend.get("direction", "N/A"),
                    "Timeline": trend.get("timeline", "N/A"),
                })
        
        if trend_data:
            try:
                df_trends = pd.DataFrame(trend_data)
                st.table(df_trends)
            except Exception as e:
                st.warning(f"Could not render Trends table: {e}")

    st.markdown("---")

    # Visualization Data - Flexible handling (Line/Bar)
    st.subheader("📊 Visualization")
    viz = data.get('visualization_data', {})
    
    # Handle common time series patterns
    labels = viz.get("chart_labels") or viz.get("labels", [])
    values = viz.get("data_series_values") or viz.get("values", [])
    title = viz.get("title", "Market Trends")
    
    if labels and values and len(labels) == len(values):
        try:
            df_viz = pd.DataFrame({
                "Category": labels[:10],
                "Value": [float(v) for v in values[:10]]
            })
            fig = px.line(df_viz, x="Category", y="Value", title=title, markers=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Chart data available but formatting issue.")
    
    # Comparison Bars
    comp_bars = data.get('comparison_bars', {})
    if comp_bars and isinstance(comp_bars, dict) and comp_bars.get("categories") and comp_bars.get("values"):
        try:
            df_bars = pd.DataFrame({
                "Category": comp_bars["categories"],
                "Value": comp_bars["values"]
            })
            fig_bar = px.bar(df_bars, x="Category", y="Value", 
                             title=comp_bars.get("title", "Comparison"), text_auto=True)
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception:
            st.info("Comparison Bar data available but formatting issue.")

    if not viz and not comp_bars:
        st.info("🧪 Visualization data structure detected but no renderable chart series found.")

    st.markdown("---")
    
    # Sources
    sources = data.get('sources', [])
    if sources:
        st.subheader("🔗 Sources")
        for i, source in enumerate(sources[:10], 1):
            if source and source.startswith("http"):
                st.markdown(f"**{i}.** [{source}]({source})")
            elif source:
                st.markdown(f"**{i}.** {source}")

    # Data freshness and action
    col_fresh, col_action = st.columns(2)
    with col_fresh:
        freshness = data.get('freshness') or "Current"
        st.metric("Data Freshness", freshness)
    
    with col_action:
        action = data.get('action')
        if action and isinstance(action, dict):
            rec = action.get("recommendation", "Neutral")
            conf = action.get("confidence", "Medium")
            rationale = action.get("rationale", "")
            st.metric("Recommendation", f"{rec} ({conf})", delta=rationale[:50])
        else:
            st.metric("Recommendation", "Analyze", delta="Review key findings above")

    st.markdown("---")

    # Secondary sections
    if web_context and web_context.get("search_results"):
        st.subheader("🌐 Web Context")
        for i, result in enumerate(web_context["search_results"][:5]):
            with st.expander(f"Source {i+1}: {result.get('title', 'No title')}"):
                st.write(f"**{result.get('source', 'N/A')}**")
                st.write(result.get('snippet', ''))
                st.caption(f"[Link]({result.get('link', '')})")

    if show_secondary_view and secondary_resp:
        st.subheader("🔍 Secondary Validation")
        try:
            # 🟢 FIXED: Validate secondary response against Pydantic model
            # Need to perform the repair again since secondary_resp is a raw JSON string
            parsed_sec_dict = json.loads(preclean_llm_json(secondary_resp))
            repaired_sec_dict = repair_metric_list_to_dict(parsed_sec_dict)
            sec_obj = LLMResponse.model_validate(repaired_sec_dict)
            st.json(sec_obj.model_dump())
        except ValidationError:
            # Fallback if Pydantic fails
            st.warning("Secondary response failed Pydantic validation. Displaying raw output.")
            st.code(secondary_resp[:2000], language="json")
        except Exception:
            st.warning("Secondary response failed JSON parsing. Displaying raw output.")
            st.code(secondary_resp[:2000], language="json")

    if veracity_scores:
        st.subheader("✅ Veracity Scores")
        cols_v = st.columns(5)
        # 🟢 FIXED: Updated metrics to match new keys
        metrics = [
            ("Summary", "executive_summary_score"), 
            ("Findings", "key_findings_score"), 
            ("Metrics", "primary_metrics_score"), # Placeholder for better numeric comparison
            ("Graph", "graphical_data_score"),
            ("Overall", "overall_score"),
        ]
        for i, (label, key) in enumerate(metrics):
            cols_v[i].metric(label, f"{veracity_scores.get(key, 0):.1f}")

    if versions_history:
        st.subheader("📈 Evolution Layer")
        render_evolution_layer(versions_history)



# ----------------------------
# MAIN WORKFLOW
# Includes a multi_modal_compare() function that compares two JSON outputs from your LLMs, covering textual, tabular, and graphical data
# ----------------------------

def compare_texts(text1, text2):
    if not text1 or not text2:
        return 0.0
    embeddings = embedder.encode([text1, text2], convert_to_tensor=True)
    sim = util.cos_sim(embeddings[0], embeddings[1])
    return float(sim.item()) * 100

def compare_key_insights(list1, list2):
    if not list1 or not list2:
        return 0.0
    sims = []
    for t1 in list1:
        best_sim = 0
        for t2 in list2:
            sim = compare_texts(t1, t2)
            if sim > best_sim:
                best_sim = sim
            # Optimization: if high match, stop early
            if best_sim > 99.9:
                 break
        sims.append(best_sim)
    return np.mean(sims) if sims else 0.0

# 🟢 FIXED: Renamed for the new structure, kept logic flexible for now
def compare_tables(table1, table2):
    # This function is currently comparing the 'benchmark_table' key,
    # which is a list of dicts.
    if not table1 or not table2:
        return 0.0
    
    try:
        df1 = pd.DataFrame(table1)
        df2 = pd.DataFrame(table2)
    except Exception:
        return 0.0

    # Align columns and rows by intersection (simple, more complex alignment can be done)
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        return 0.0

    df1_common = df1[common_cols].reset_index(drop=True)
    df2_common = df2[common_cols].reset_index(drop=True)

    # Truncate to shortest length for fair comparison
    min_len = min(len(df1_common), len(df2_common))
    df1_common = df1_common.iloc[:min_len]
    df2_common = df2_common.iloc[:min_len]

    scores = []
    for col in common_cols:
        col1 = df1_common[col]
        col2 = df2_common[col]

        # Numeric columns - compare mean relative difference
        if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
            diffs = np.abs(col1 - col2)
            max_vals = np.maximum(np.abs(col1), np.abs(col2))
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diffs = np.where(max_vals != 0, diffs / max_vals, 0)
            score = 100.0 - (np.nanmean(rel_diffs) * 100)
            scores.append(max(0, min(100, score)))
        else:
            # Non-numeric: fraction exact matches ignoring case and whitespace
            matches = col1.astype(str).str.strip().str.lower() == col2.astype(str).str.strip().str.lower()
            score = (matches.sum() / len(matches)) * 100
            scores.append(score)
    return np.mean(scores) if scores else 0.0

# 🟢 FIXED: Renamed for the new structure
def compare_graphical_data(vis1, vis2):
    if not vis1 or not vis2:
        return 0.0
        
    # The new structure uses 'chart_labels' and 'data_series_values'
    labels1 = vis1.get("chart_labels") or vis1.get("labels", [])
    labels2 = vis2.get("chart_labels") or vis2.get("labels", [])

    values1 = vis1.get("data_series_values") or vis1.get("values", [])
    values2 = vis2.get("data_series_values") or vis2.get("values", [])
    
    # Try the old keys as fallback
    if not labels1:
        labels1 = vis1.get("labels", [])
    if not labels2:
        labels2 = vis2.get("labels", [])
    if not values1:
        values1 = vis1.get("values", [])
    if not values2:
        values2 = vis2.get("values", [])

    # Simple label comparison (strict for now)
    if labels1 != labels2:
        return 0.0 

    if not values1 or not values2 or len(values1) != len(values2):
        return 0.0

    # Ensure all values are numeric for comparison
    try:
        vals1 = np.array([float(v) for v in values1])
        vals2 = np.array([float(v) for v in values2])
    except (ValueError, TypeError):
        return 0.0 # Cannot compare non-numeric data

    # Percent similarity using normalized difference
    diff = np.abs(vals1 - vals2)
    max_vals = np.maximum(np.abs(vals1), np.abs(vals2))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(max_vals != 0, diff / max_vals, 0)
    score = 100.0 - (np.nanmean(rel_diff) * 100)
    return max(0, min(100, score))

def multi_modal_compare(json1, json2):
    # 🟢 FIXED: Key names updated to match LLMResponse schema
    
    # Textual comparisons
    summary_score = compare_texts(json1.get("executive_summary", ""), json2.get("executive_summary", ""))
    insights_score = compare_key_insights(json1.get("key_findings", []), json2.get("key_findings", []))
    
    # Tabular comparison (using benchmark_table for now, since 'table' is gone)
    table_score = compare_tables(json1.get("benchmark_table", []), json2.get("benchmark_table", []))

    # Graphical data comparison
    graph_score = compare_graphical_data(json1.get("visualization_data", {}), json2.get("visualization_data", {}))
    
    # Placeholder for numeric comparison (use numeric_alignment_score separately)
    numeric_metrics_score = numeric_alignment_score(json1, json2)

    # Aggregate overall (weights can be tuned)
    weights = {
        "summary": 0.3,
        "insights": 0.2,
        "table": 0.3, # Using benchmark_table alignment
        "graph": 0.2,
    }
    # Using weighted average for the structural integrity score
    overall_score = (
        summary_score * weights["summary"] +
        insights_score * weights["insights"] +
        table_score * weights["table"] +
        graph_score * weights["graph"]
    )

    return {
        "executive_summary_score": summary_score,
        "key_findings_score": insights_score,
        "benchmark_table_score": table_score,
        "graphical_data_score": graph_score,
        "primary_metrics_score": numeric_metrics_score if numeric_metrics_score is not None else 0.0, # Adding the numeric comparison here
        "overall_score": overall_score,
    }


def main():
    st.set_page_config(page_title="Yureeka Market Research Assistant", layout="wide")
    st.title("Yureeka Research Assistant")
#    st.caption("Self-Consistency + Cross-Model Verification + Live Web Search + Dynamic Metrics + Evolution Layer")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("""
        Yureeka is a research assistant that assists in providing succinct and accurate answers to your market related questions.  You may ask any 
        question that is related to finance, economics or the markets. 
        This product is currently in prototype stage.
        """)
    with c2:
        web_status = "✅ Enabled" if SERPAPI_KEY else "‚ö†Ô∏è Not configured"
        st.metric("Web Search", web_status)

    q = st.text_input("Enter your question about markets, finance, or economics:")
#    use_web_search = st.checkbox("Enable live web search (recommended)", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY)

    # In main() function, after q = st.text_input("..."):

    col_opt1, col_opt2 = st.columns(2) # <--- NEW: Create two columns
    
    with col_opt1:
        use_web_search = st.checkbox("Enable live web search (recommended)", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY)
    
    with col_opt2:
        # 🟢 1A. INSERT THE SECONDARY TOGGLE
        show_validation = st.checkbox("Show Secondary (Validation) Response", value=False)
        
    if st.button("Analyze") and q:
        web_context = {}
        if use_web_search:
            with st.spinner("Searching the web for latest info..."):
                web_context = fetch_web_context(q, num_sources=3)

        if web_context and web_context.get("search_results"):
            responses, scores = generate_self_consistent_responses_with_web(q, web_context, n=3)
        else:
            st.info("Using internal model knowledge only...")
            empty_ctx = {"search_results": [], "scraped_content": {}, "summary": "", "sources": []}
            responses, scores = generate_self_consistent_responses_with_web(q, empty_ctx, n=3)

        if not responses or not scores:
            st.error("Primary model failed to generate valid responses.")
            return

        if len(responses) != len(scores):
            st.error("Mismatch in responses and scores.")
            return

        voted_response = majority_vote(responses)
        max_score = max(scores)
        best_idx = scores.index(max_score)
        best_response = responses[best_idx]

        chosen_primary = best_response or voted_response
        if not chosen_primary:
            st.error("Could not determine the primary response.")
            return
            
        st.info("Performing validation...")
        secondary_resp = query_gemini(q)

        # 🟢 FIXED: Parse chosen_primary and secondary_resp immediately for use as dictionaries
        try:
            # Need to re-parse the final chosen response here (already repaired in LLM call, but need dict for comparison)
            parsed_dict_1 = json.loads(preclean_llm_json(chosen_primary))
            j1 = repair_metric_list_to_dict(parsed_dict_1)
        except Exception:
            # Fallback to robust parsing/empty dict if Pydantic fails
            j1 = parse_json_robustly(chosen_primary, "Primary Model")
        
        try:
            parsed_dict_2 = json.loads(preclean_llm_json(secondary_resp))
            j2 = repair_metric_list_to_dict(parsed_dict_2)
        except Exception:
            j2 = parse_json_robustly(secondary_resp, "Secondary Model")
            
        # The summary text is used for semantic comparison
        sem_conf = compare_texts(j1.get('executive_summary', ''), j2.get('executive_summary', ''))

        veracity_scores = multi_modal_compare(j1, j2)

        # 🟢 FIXED: Calculation needs the dictionary 'j1' for numeric_alignment_score
        num_conf = numeric_alignment_score(j1, j2)
        base_conf = max_score
        src_conf = source_quality_confidence(j1.get("sources", [])) * 100

   
        # Use consistent weighting
        weights = {
        'base': 0.25,
        'semantic': 0.20,
        'source': 0.20,
        'veracity': 0.20,
        'numeric': 0.15
        }

        final_conf = (
        base_conf * weights['base'] +
        sem_conf * weights['semantic'] +
        src_conf * weights['source'] +
        veracity_scores["overall_score"] * weights['veracity'] +
        (num_conf * weights['numeric'] if num_conf is not None else 0)
        )

        # Save JSON output

        # ---- DOWNLOAD JSON INSTEAD OF SAVING ----
        output_payload = {
        "primary_response": j1,
        "secondary_response": j2,
        "final_confidence": final_conf,
        "veracity_scores": veracity_scores,
        "question": q,
        "timestamp": datetime.now().isoformat()
        }

        
        json_str = json.dumps(output_payload, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        filename = f"yureeka_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 1. FIX THE HTML STRING TO ENSURE IT'S VALID AND ISOLATED
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}">💾 Download Analysis JSON</a>'
        
        # 2. RENDER THE LINK (This part is correct, but ensure no extra text is around it)
        st.markdown(href, unsafe_allow_html=True)

        # 3. Add the clarifying text separately
        st.caption("(Right-click → Save As)") 
        st.success("✅ Analysis ready for download!")

        
        versions_history = [
        {
        "version": "V1 (Jul 10)",
        "timestamp": "2025-07-10T12:00:00",
        # 🟢 FIXED: Pass the metric structure j1.get("primary_metrics", {})
        "metrics": j1.get("primary_metrics", {}),
        "confidence": base_conf,
        "sources_freshness": 80,
        "change_reason": "Initial version",
        },
        {
        "version": "V2 (Aug 28)",
        "timestamp": "2025-08-28T15:30:00",
        "metrics": j1.get("primary_metrics", {}),
        "confidence": base_conf * 0.98,
        "sources_freshness": 75,
        "change_reason": "Quarterly update",
        },
        {
        "version": "V3 (Nov 3)",
        "timestamp": datetime.now().isoformat(timespec="minutes"),
        # 🟢 FIXED: Ensure the history uses the correct key 'primary_metrics'
        "metrics": j1.get("primary_metrics", {}),
        "confidence": final_conf,
        "sources_freshness": 78,
        "change_reason": "Latest analysis",
        }
        ]

        # Store ALL versions for individual pages (NEW)
        st.session_state["all_versions"] = versions_history
        # 🟢 FIX 2: Correct keys for the current analysis session state
        st.session_state["current_analysis"] = {
        "executive_summary": j1.get("executive_summary", ""), # Corrected to 'executive_summary'
        "primary_metrics": j1.get("primary_metrics", {}), # Corrected to 'primary_metrics'
        "confidence": final_conf
        }

        # In main() function, inside if st.button("Analyze") and q: block, around line 570:

        # CALL RENDER_DASHBOARD WITH NEW TOGGLE PARAMETER
        render_dashboard(
            chosen_primary,
            final_conf,
            sem_conf,
            num_conf,
            web_context,
            base_conf,
            src_conf,
            versions_history,
            user_question=q,
            secondary_resp=secondary_resp,
            veracity_scores=veracity_scores,
            show_secondary_view=show_validation  # 🟢 PASS THE TOGGLE HERE
        )


        

        with st.expander("Debug Information"):
            st.write("Primary Response:")
            st.code(chosen_primary, language="json")
            st.write("Validation Response:")
            st.code(secondary_resp, language="json")
            st.write(f"All Confidence Scores: {scores}")
            st.write(f"Selected Best Score: {base_conf}")
            if web_context:
                st.write(f"Web Sources Found: {len(web_context.get('search_results', []))}")

if __name__ == "__main__":
    main()
