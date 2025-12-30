# ===============================================================================
# YUREEKA AI RESEARCH ASSISTANT v7.26
# With Web Search, Evidence-Based Verification, Confidence Scoring
# SerpAPI Output with Evolution Layer Version
# Updated SerpAPI parameters for stable output
# Deterministic Output From LLM
# Deterministic Evolution Core Using Python Diff Engine
# Anchored Evolution Analysis Using JSON As Input Into Model
# Implementation of Source-Based Evolution
# Saving of JSON output Files into Google Sheets
# Canonical Metric Registry + Semantic Hashing of Findings
# Removal of Evolution Decisions from LLM
# Further Enhancements to Minimize Evolution Drift (Metric)
# Saving of Extraction Cache in JSON
# Prioritize High Quality Sources With Source Freshness Tracking
# Timestamps = Timezone Naive
# Improved Stability of Handling of Duplicate Canonicalized IDs
# Deterministic Main and Side Topic Extractor
# Range Aware Canonical Metrics
# Range + Source Attribution
# Proxy Labeler + Geo Tagging
# Improved Main Topic + Side Topic Extractor Using Deterministic-->NLP-->LLM layer
# Guardrails For Main + Side Topic Handling
# Numeric Consistency Scores
# Multi-Side Enumerations
# Show More Detail in Dashboard
# Dashboard Unit Presentation Fixes (Main + Evolution)
# More Precise Extractor
# Domain-Agnostic Question Profiling
# Baseline Caching Contains HTTP Validators + Numeric Data
# URL canonicalization
# ================================================================================

import io
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
import gspread
import google.generativeai as genai
from pypdf import PdfReader
from pathlib import Path
from google.oauth2.service_account import Credentials
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from collections import Counter
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# =========================================================
# GOOGLE SHEETS HISTORY STORAGE
# =========================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
MAX_HISTORY_ITEMS = 50

@st.cache_resource
def get_google_sheet():
    """Connect to Google Sheet (cached connection)"""
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=SCOPES
        )
        client = gspread.authorize(creds)
        spreadsheet_name = st.secrets.get("google_sheets", {}).get("spreadsheet_name", "Yureeka_JSON")
        sheet = client.open(spreadsheet_name).sheet1

        # Ensure headers exist - handle response object
        try:
            headers = sheet.row_values(1)
            if not headers or len(headers) == 0 or headers[0] != "id":
                # update() returns a response object in newer gspread - ignore it
                _ = sheet.update('A1:E1', [["id", "timestamp", "question", "confidence", "data"]])
        except gspread.exceptions.APIError:
            _ = sheet.update('A1:E1', [["id", "timestamp", "question", "confidence", "data"]])
        except Exception:
            pass  # Headers probably already exist

        return sheet

    except gspread.exceptions.SpreadsheetNotFound:
        st.error("❌ Spreadsheet not found. Create 'Yureeka_History' and share with service account.")
        return None
    except Exception as e:
        error_str = str(e)
        # Ignore Response [200] - it's actually success
        if "Response [200]" in error_str:
            # This means the connection worked, try to return the sheet anyway
            try:
                creds = Credentials.from_service_account_info(
                    dict(st.secrets["gcp_service_account"]),
                    scopes=SCOPES
                )
                client = gspread.authorize(creds)
                spreadsheet_name = st.secrets.get("google_sheets", {}).get("spreadsheet_name", "Yureeka_History")
                return client.open(spreadsheet_name).sheet1
            except:
                pass
        st.error(f"❌ Failed to connect to Google Sheets: {e}")
        return None


def generate_analysis_id() -> str:
    """Generate unique ID for analysis"""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]}"

def add_to_history(analysis: Dict) -> bool:
    """Save analysis to Google Sheet"""
    sheet = get_google_sheet()
    if not sheet:
        # Fallback to session state
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        st.session_state.analysis_history.append(analysis)
        return False

    try:
        analysis_id = generate_analysis_id()
        row = [
            analysis_id,
            analysis.get("timestamp", datetime.now().isoformat()),
            analysis.get("question", "")[:100],  # Truncate for display
            str(analysis.get("final_confidence", "")),
            json.dumps(analysis, default=str)  # Full JSON data
        ]
        sheet.append_row(row, value_input_option='RAW')
        return True
    except Exception as e:
        st.warning(f"⚠️ Failed to save to Google Sheets: {e}")
        # Fallback to session state
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        st.session_state.analysis_history.append(analysis)
        return False

def get_history(limit: int = MAX_HISTORY_ITEMS) -> List[Dict]:
    """Load analysis history from Google Sheet"""
    sheet = get_google_sheet()
    if not sheet:
        # Fallback to session state
        return st.session_state.get('analysis_history', [])

    try:
        # Get all rows (skip header)
        all_rows = sheet.get_all_values()[1:]

        # Parse and return most recent
        history = []
        for row in all_rows[-limit:]:
            if len(row) >= 5:
                try:
                    data = json.loads(row[4])
                    data['_sheet_id'] = row[0]  # Keep track of sheet row ID
                    history.append(data)
                except json.JSONDecodeError:
                    continue

        return history
    except Exception as e:
        st.warning(f"⚠️ Failed to load from Google Sheets: {e}")
        return st.session_state.get('analysis_history', [])

def get_analysis_by_id(analysis_id: str) -> Optional[Dict]:
    """Get a specific analysis by ID"""
    sheet = get_google_sheet()
    if not sheet:
        return None

    try:
        # Find row with matching ID
        cell = sheet.find(analysis_id)
        if cell:
            row = sheet.row_values(cell.row)
            if len(row) >= 5:
                return json.loads(row[4])
    except Exception as e:
        st.warning(f"⚠️ Failed to find analysis: {e}")

    return None

def delete_from_history(analysis_id: str) -> bool:
    """Delete an analysis from history"""
    sheet = get_google_sheet()
    if not sheet:
        return False

    try:
        cell = sheet.find(analysis_id)
        if cell:
            sheet.delete_rows(cell.row)
            return True
    except Exception as e:
        st.warning(f"⚠️ Failed to delete: {e}")

    return False

def clear_history() -> bool:
    """Clear all history (keep headers)"""
    sheet = get_google_sheet()
    if not sheet:
        st.session_state.analysis_history = []
        return True

    try:
        # Get row count
        all_rows = sheet.get_all_values()
        if len(all_rows) > 1:
            # Delete all rows except header
            sheet.delete_rows(2, len(all_rows))
        return True
    except Exception as e:
        st.warning(f"⚠️ Failed to clear history: {e}")
        return False

def format_history_label(analysis: Dict) -> str:
    """Format a history item for dropdown display"""
    timestamp = analysis.get('timestamp', '')
    question = analysis.get('question', 'Unknown query')[:40]
    confidence = analysis.get('final_confidence', '')

    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now()
        delta = now - dt.replace(tzinfo=None)

        if delta.total_seconds() < 3600:
            time_str = f"{int(delta.total_seconds() / 60)}m ago"
        elif delta.total_seconds() < 86400:
            time_str = f"{int(delta.total_seconds() / 3600)}h ago"
        elif delta.days == 1:
            time_str = "Yesterday"
        elif delta.days < 7:
            time_str = f"{delta.days}d ago"
        else:
            time_str = dt.strftime("%b %d")
    except:
        time_str = timestamp[:10] if timestamp else "Unknown"

    conf_str = f" ({confidence:.0f}%)" if isinstance(confidence, (int, float)) else ""
    return f"{time_str}: {question}...{conf_str}"

def get_history_options() -> List[Tuple[str, int]]:
    """Get formatted history options for dropdown"""
    history = get_history()
    options = []
    for i, analysis in enumerate(reversed(history)):  # Most recent first
        label = format_history_label(analysis)
        actual_index = len(history) - 1 - i
        options.append((label, actual_index))
    return options

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
  "executive_summary": "3-4 sentence high-level answer",
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
  "freshness": "Dec 2024"
}
"""



SYSTEM_PROMPT = f"""You are a professional market research analyst.

CRITICAL RULES:
1. Return ONLY valid JSON. NO markdown, NO code blocks, NO extra text.
2. NO citation references like [1][2] inside strings.
3. Use double quotes for all keys and string values.
4. NO trailing commas in arrays or objects.
5. Escape internal quotes with backslash.
6. If the prompt includes "Query Structure", you MUST follow it:
   - Treat "MAIN QUESTION" as the primary topic and address it FIRST.
   - Treat "SIDE QUESTIONS" as secondary topics and address them AFTER the main topic.
   - Do NOT let a side question replace the main question just because it is more specific.
   - In executive_summary, clearly separate: "Main:" then "Side:" when side questions exist.


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

    - Convert primary_metrics from list -> dict (stable keys)
    - Normalize MetricDetail fields so currency+unit do NOT get lost:
        "29.8 S$B" / "S$29.8B" / "S$29.8 billion" -> value=29.8, unit="S$B"
        "$204.7B" -> value=204.7, unit="$B"
        "9.8%" -> value=9.8, unit="%"
    - Ensure top_entities and trends_forecast are lists
    - Fix visualization_data legacy keys (labels/values)
    - Fix benchmark_table numeric values
    - Remove 'action' block entirely (no longer used)
    - Add minimal required fields if missing

    NOTE: This function is intentionally conservative: it normalizes obvious formatting
    without trying to "invent" missing values.
    """
    if not isinstance(data, dict):
        return {}

    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            return list(x.values())
        return []

    def _coerce_number(s: str):
        try:
            return float(str(s).replace(",", "").strip())
        except Exception:
            return None

    def _normalize_metric_item(item: dict) -> dict:
        """
        Normalize a single metric dict in-place-ish and return it.

        Goal: preserve currency + magnitude in `unit`, keep `value` numeric when possible.
        """
        if not isinstance(item, dict):
            return {"name": "N/A", "value": "N/A", "unit": ""}

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            name = "N/A"
        item["name"] = name

        raw_val = item.get("value")
        raw_unit = item.get("unit")

        unit = (raw_unit or "")
        if not isinstance(unit, str):
            unit = str(unit)

        # If already numeric and unit looks okay, keep as-is
        if isinstance(raw_val, (int, float)) and isinstance(unit, str):
            item["unit"] = unit.strip()
            return item

        # Try to parse string value forms like:
        # "S$29.8B", "29.8 S$B", "$ 204.7 billion", "9.8%", "12 percent"
        if isinstance(raw_val, str):
            txt = raw_val.strip()

            # Also allow unit to carry the number sometimes (rare but happens)
            # e.g. value="29.8", unit="S$B" is already fine.
            # But if unit is empty and txt contains unit, we extract.
            # Percent detection
            if re.search(r'(%|\bpercent\b)', txt, flags=re.I):
                num = _coerce_number(re.sub(r'[^0-9\.\-\,]+', '', txt))
                if num is not None:
                    item["value"] = num
                    item["unit"] = "%"
                    return item

            # Currency detection
            currency = ""
            # Normalize currency tokens in either value or unit
            combo = f"{txt} {unit}".strip()

            if re.search(r'\bSGD\b', combo, flags=re.I) or "S$" in combo.upper():
                currency = "S$"
            elif re.search(r'\bUSD\b', combo, flags=re.I) or "$" in combo:
                currency = "$"

            # Magnitude detection
            # Accept: T/B/M/K, or words
            mag = ""
            if re.search(r'\btrillion\b', combo, flags=re.I):
                mag = "T"
            elif re.search(r'\bbillion\b', combo, flags=re.I):
                mag = "B"
            elif re.search(r'\bmillion\b', combo, flags=re.I):
                mag = "M"
            elif re.search(r'\bthousand\b', combo, flags=re.I):
                mag = "K"
            else:
                m = re.search(r'([TBMK])\b', combo.replace(" ", ""), flags=re.I)
                if m:
                    mag = m.group(1).upper()

            # Extract numeric
            num = _coerce_number(re.sub(r'[^0-9\.\-\,]+', '', txt))
            if num is not None:
                # If unit was present and meaningful (and already includes %), keep it
                if unit.strip() == "%":
                    item["value"] = num
                    item["unit"] = "%"
                    return item

                # Build unit as currency+magnitude when any found
                # If neither found, keep existing unit (may be e.g. "years", "points")
                if currency or mag:
                    item["value"] = num
                    item["unit"] = f"{currency}{mag}".strip()
                    return item

                # No currency/mag detected: keep unit if provided; else blank
                item["value"] = num
                item["unit"] = unit.strip()
                return item

            # If we can’t parse into a number, at least preserve the original text
            item["value"] = txt
            item["unit"] = unit.strip()
            return item

        # Non-string, non-numeric (None, dict, list, etc.)
        if raw_val is None or raw_val == "":
            item["value"] = "N/A"
        else:
            item["value"] = str(raw_val)

        item["unit"] = unit.strip()
        return item

    # -------------------------
    # primary_metrics normalization
    # -------------------------
    metrics = data.get("primary_metrics")

    # list -> dict
    if isinstance(metrics, list):
        new_metrics = {}
        for i, item in enumerate(metrics):
            if not isinstance(item, dict):
                continue
            item = _normalize_metric_item(item)

            raw_name = item.get("name", f"metric_{i+1}")
            key = re.sub(r'[^a-z0-9_]', '', str(raw_name).lower().replace(" ", "_")).strip("_")
            if not key:
                key = f"metric_{i+1}"

            original_key = key
            j = 1
            while key in new_metrics:
                key = f"{original_key}_{j}"
                j += 1

            new_metrics[key] = item

        data["primary_metrics"] = new_metrics

    elif isinstance(metrics, dict):
        # Normalize each metric dict entry
        cleaned = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                cleaned[str(k)] = _normalize_metric_item(v)
            else:
                # If someone stored a scalar, wrap it
                cleaned[str(k)] = _normalize_metric_item({"name": str(k), "value": v, "unit": ""})
        data["primary_metrics"] = cleaned

    else:
        data["primary_metrics"] = {}

    # -------------------------
    # list-like fields
    # -------------------------
    data["top_entities"] = _to_list(data.get("top_entities"))
    data["trends_forecast"] = _to_list(data.get("trends_forecast"))
    data["key_findings"] = _to_list(data.get("key_findings"))

    # Ensure strings in key_findings
    data["key_findings"] = [str(x) for x in data["key_findings"] if x is not None and str(x).strip()]

    # -------------------------
    # visualization_data legacy keys
    # -------------------------
    if isinstance(data.get("visualization_data"), dict):
        viz = data["visualization_data"]
        if "labels" in viz and "chart_labels" not in viz:
            viz["chart_labels"] = viz.pop("labels")
        if "values" in viz and "chart_values" not in viz:
            viz["chart_values"] = viz.pop("values")

        # Coerce chart_labels/values types gently
        if "chart_labels" in viz and not isinstance(viz["chart_labels"], list):
            viz["chart_labels"] = [str(viz["chart_labels"])]
        if "chart_values" in viz and not isinstance(viz["chart_values"], list):
            viz["chart_values"] = [viz["chart_values"]]

    # -------------------------
    # benchmark_table numeric cleaning
    # -------------------------
    if isinstance(data.get("benchmark_table"), list):
        cleaned_table = []
        for row in data["benchmark_table"]:
            if not isinstance(row, dict):
                continue

            if "category" not in row:
                row["category"] = "Unknown"

            for key in ["value_1", "value_2"]:
                if key not in row:
                    row[key] = 0
                    continue

                val = row.get(key)
                if isinstance(val, str):
                    val_upper = val.upper().strip()
                    if val_upper in ["N/A", "NA", "NULL", "NONE", "", "-", "—"]:
                        row[key] = 0
                    else:
                        try:
                            cleaned = re.sub(r'[^\d.-]', '', val)
                            row[key] = float(cleaned) if '.' in cleaned else int(cleaned) if cleaned else 0
                        except Exception:
                            row[key] = 0
                elif isinstance(val, (int, float)):
                    pass
                else:
                    row[key] = 0

            cleaned_table.append(row)

        data["benchmark_table"] = cleaned_table

    # -------------------------
    # Remove action block entirely
    # -------------------------
    data.pop("action", None)

    # -------------------------
    # Minimal required top-level fields
    # -------------------------
    if not isinstance(data.get("executive_summary"), str) or not data.get("executive_summary", "").strip():
        data["executive_summary"] = "No executive summary provided."

    if not isinstance(data.get("sources"), list):
        data["sources"] = []

    if "confidence" not in data:
        data["confidence"] = 60

    if not isinstance(data.get("freshness"), str) or not data.get("freshness", "").strip():
        data["freshness"] = "Current"

    return data


def validate_numeric_fields(data: dict, context: str = "LLM Response") -> None:
    """
    Guardrail logger (and gentle coercer) for numeric lists used in charts/tables.

    We keep this lightweight: warn when strings appear where numbers are expected,
    and attempt to coerce when safe.
    """
    if not isinstance(data, dict):
        return

    # Check benchmark_table
    if "benchmark_table" in data and isinstance(data["benchmark_table"], list):
        for i, row in enumerate(data["benchmark_table"]):
            if isinstance(row, dict):
                for key in ["value_1", "value_2"]:
                    val = row.get(key)
                    if isinstance(val, str):
                        st.warning(
                            f"⚠️ {context}: benchmark_table[{i}].{key} is string: '{val}' (coercing to 0 if invalid)"
                        )
                        try:
                            cleaned = re.sub(r"[^\d\.\-]", "", val)
                            row[key] = float(cleaned) if cleaned else 0
                        except Exception:
                            row[key] = 0

    # Check visualization_data chart_values
    viz = data.get("visualization_data")
    if isinstance(viz, dict):
        vals = viz.get("chart_values")
        if isinstance(vals, list):
            new_vals = []
            for j, v in enumerate(vals):
                if isinstance(v, (int, float)):
                    new_vals.append(v)
                elif isinstance(v, str):
                    try:
                        cleaned = re.sub(r"[^\d\.\-]", "", v)
                        new_vals.append(float(cleaned) if cleaned else 0.0)
                        st.warning(f"⚠️ {context}: visualization_data.chart_values[{j}] is string: '{v}' (coerced)")
                    except Exception:
                        new_vals.append(0.0)
                else:
                    new_vals.append(0.0)
            viz["chart_values"] = new_vals


def preclean_json(raw: str) -> str:
    """
    Remove markdown fences and common citation markers before JSON parsing.
    Conservative: tries not to destroy legitimate JSON content.
    """
    if not raw or not isinstance(raw, str):
        return ""

    text = raw.strip()

    # Remove leading/trailing code fences (```json ... ```)
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)

    text = text.strip()

    # Remove common citation formats the model may append
    # [web:1], [1], (1) etc. (but avoid killing array syntax by being specific)
    text = re.sub(r'\[web:\d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!")\[\d+\](?!")', '', text)   # not inside quotes
    text = re.sub(r'(?<!")\(\d+\)(?!")', '', text)   # not inside quotes

    return text.strip()


def parse_json_safely(json_str: str, context: str = "LLM") -> dict:
    """
    Parse JSON with aggressive error recovery:
    1) Pre-clean markdown/citations
    2) Extract the *first* JSON object
    3) Repair common issues (unquoted keys, trailing commas, True/False/Null)
    4) Try parsing; if it fails, attempt a small set of pragmatic fixes
    """
    if json_str is None:
        return {}
    if not isinstance(json_str, str):
        json_str = str(json_str)

    if not json_str.strip():
        return {}

    cleaned = preclean_json(json_str)

    # Extract first JSON object (most LLM outputs are one object)
    match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if not match:
        st.warning(f"⚠️ No JSON object found in {context} response")
        return {}

    json_content = match.group(0)

    # Structural repairs
    try:
        # Fix unquoted keys: {key: -> {"key":
        json_content = re.sub(
            r'([\{\,]\s*)([a-zA-Z_][a-zA-Z0-9_\-]*)(\s*):',
            r'\1"\2"\3:',
            json_content
        )

        # Remove trailing commas
        json_content = re.sub(r',\s*([\]\}])', r'\1', json_content)

        # Fix boolean/null capitalization
        json_content = re.sub(r':\s*True\b', ': true', json_content)
        json_content = re.sub(r':\s*False\b', ': false', json_content)
        json_content = re.sub(r':\s*Null\b', ': null', json_content)

    except Exception as e:
        st.warning(f"⚠️ {context}: Regex repair failed: {e}")

    # Attempt parse with a few passes
    attempts = 0
    last_err = None

    while attempts < 6:
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            last_err = e
            msg = (e.msg or "").lower()

            # Pass 1: replace smart quotes
            if attempts == 0:
                json_content = (
                    json_content.replace("“", '"')
                                .replace("”", '"')
                                .replace("’", "'")
                )

            # Pass 2: single-quote keys/strings -> double quotes (limited)
            elif attempts == 1:
                # Only do this if it looks like single quotes dominate
                if json_content.count("'") > json_content.count('"'):
                    json_content = re.sub(r"\'", '"', json_content)

            # Pass 3: try removing control characters
            elif attempts == 2:
                json_content = re.sub(r"[\x00-\x1F\x7F]", "", json_content)

            # Pass 4: if unterminated string, try escaping a quote near the error
            elif "unterminated string" in msg or "unterminated" in msg:
                pos = e.pos
                # Try escaping a quote a bit before pos
                for i in range(pos - 1, max(0, pos - 200), -1):
                    if i < len(json_content) and json_content[i] == '"':
                        if i == 0 or json_content[i - 1] != "\\":
                            json_content = json_content[:i] + '\\"' + json_content[i + 1:]
                            break

            # Pass 5+: give up
            attempts += 1
            continue

    st.error(f"❌ Failed to parse JSON from {context}: {str(last_err)[:180] if last_err else 'unknown error'}")
    return {}




def parse_query_structure_safe(json_str: str, user_question: str) -> Dict:
    """
    Parse LLM-derived query structure with guaranteed deterministic fallback.
    Never raises, never returns empty dict.
    """
    parsed = parse_json_safely(json_str, context="LLM Query Structure")

    if isinstance(parsed, dict) and parsed:
        # Minimal schema validation
        if "main" in parsed or "category" in parsed:
            return parsed

    # 🔒 Deterministic fallback (NO LLM)
    return {
        "category": "unknown",
        "category_confidence": 0.0,
        "main": user_question,
        "side": []
    }


def extract_json_object(text: str) -> Optional[Dict]:
    """
    Best-effort extraction of the first JSON object from a string.
    Returns dict or None.
    """
    if not text or not isinstance(text, str):
        return None

    # Common cleanup
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Fast path
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Regex: first {...} block (non-greedy)
    try:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None
        candidate = m.group(0)
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


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
    """Search web and scrape top sources (with per-source meta + cached numeric candidates)."""
    search_results = search_serpapi(query, num_results=10)

    source_counts = {
        "total": len(search_results),
        "high_quality": sum(
            1 for r in search_results
            if "✅" in classify_source_reliability(r.get("link", ""))
        ),
        "used_for_scraping": min(num_sources, len(search_results))
    }
    st.info(
        f"🔍 Sources Found: **{source_counts['total']} total** | "
        f"**{source_counts['high_quality']} high-quality** | "
        f"Scraping **{source_counts['used_for_scraping']}**"
    )

    if not search_results:
        return {
            "search_results": [],
            "scraped_content": {},
            "scraped_meta": {},
            "summary": "",
            "sources": [],
            "source_reliability": []
        }

    scraped_content: Dict[str, str] = {}
    scraped_meta: Dict[str, Dict] = {}

    # Best-effort scrape (only if key exists)
    if SCRAPINGDOG_KEY:
        progress = st.progress(0)
        st.info(f"🔍 Scraping top {num_sources} sources...")

        for i, result in enumerate(search_results[:num_sources]):
            url = result.get("link")
            if not url:
                progress.progress((i + 1) / num_sources)
                continue

            # Use the robust fetcher (records URL_FETCH_META)
            content, status_msg = fetch_url_content_with_status(url)

            # Store meta (headers/fingerprint/etc.)
            meta = {}
            try:
                meta = dict((globals().get("URL_FETCH_META") or {}).get(url) or {})
            except Exception:
                meta = {}

            meta.update({
                "url": url,
                "status_detail": status_msg,
                "source": result.get("source"),
                "title": result.get("title"),
                "date": result.get("date"),
            })

            if content:
                scraped_content[url] = content

                # Extract and cache numeric candidates now (so later evolution can reuse)
                extracted = []
                try:
                    extracted = extract_numbers_with_context_pdf(content) if status_msg == "success_pdf" else extract_numbers_with_context(content)
                except Exception:
                    extracted = []

                compact = [{
                    "value": n.get("value"),
                    "unit": n.get("unit"),
                    "raw": n.get("raw"),
                    "source_url": url,
                    "context": (n.get("context", "")[:220] if isinstance(n.get("context"), str) else "")
                } for n in (extracted or [])]

                meta["numbers_found"] = len(compact)
                meta["extracted_numbers"] = compact

                # Ensure an extract_hash exists (fingerprint of cleaned content)
                if not meta.get("extract_hash"):
                    try:
                        fp = fingerprint_text(content)
                        meta["extract_hash"] = fp
                        meta["fingerprint"] = meta.get("fingerprint") or fp
                    except Exception:
                        pass

                st.success(f"✓ {i+1}/{num_sources}: {result.get('source', '')}")
            else:
                meta["numbers_found"] = meta.get("numbers_found", 0) or 0
                meta["extracted_numbers"] = meta.get("extracted_numbers", []) or []
                st.warning(f"⚠️ {i+1}/{num_sources}: {result.get('source', '')} ({status_msg})")

            scraped_meta[url] = meta
            progress.progress((i + 1) / num_sources)

        progress.empty()

    # Build context summary
    context_parts = []
    reliabilities = []

    for r in search_results:
        date_str = f" ({r.get('date')})" if r.get('date') else ""
        reliability = classify_source_reliability((r.get("link", "") or "") + " " + (r.get("source", "") or ""))
        reliabilities.append(reliability)

        context_parts.append(
            f"**{r.get('title', '')}**{date_str}\n"
            f"Source: {r.get('source', '')} [{reliability}]\n"
            f"{r.get('snippet', '')}\n"
            f"URL: {r.get('link', '')}"
        )

    return {
        "search_results": search_results,
        "scraped_content": scraped_content,
        "scraped_meta": scraped_meta,  # ✅ NEW
        "summary": "\n\n---\n\n".join(context_parts),
        "sources": [r.get("link") for r in search_results if r.get("link")],
        "source_reliability": reliabilities
    }



def fingerprint_text(text: str) -> str:
    """Stable short fingerprint for fetched content (for debugging + determinism checks)."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]

def unit_clean_first_letter(unit: str) -> str:
    """Normalize units to first letter (T/B/M/K/%), ignoring $ and spaces."""
    if not unit:
        return ""
    u = unit.replace("$", "").replace(" ", "").strip().upper()
    return u[0] if u else ""

# =========================================================
# 7. LLM QUERY FUNCTIONS
# =========================================================

def query_perplexity(query: str, web_context: Dict, query_structure: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Query Perplexity and return a validated JSON string (LLMResponse-compatible).
    Removes 'action' and excludes None fields from output JSON.
    """
    if not PERPLEXITY_KEY:
        st.error("❌ PERPLEXITY_KEY not set.")
        return None

    query_structure = query_structure or {}
    structure_txt = ""
    ordering_contract = ""

    try:
        structure_txt, ordering_contract = build_query_structure_prompt(query_structure)
    except Exception:
        structure_txt = ""
        ordering_contract = ""

    # Web context: show top sources + snippets
    sources = (web_context.get("sources", []) if isinstance(web_context, dict) else []) or []
    search_results = (web_context.get("search_results", []) if isinstance(web_context, dict) else []) or []
    search_count = int(web_context.get("search_count", len(search_results)) if isinstance(web_context, dict) else 0)

    context_section = "WEB CONTEXT:\n"
    for url in sources[:6]:
        content = (web_context.get("scraped_content", {}) or {}).get(url) if isinstance(web_context, dict) else None
        if content:
            context_section += f"\n{url}:\n{str(content)[:800]}...\n"
        else:
            context_section += f"\n{url}\n"

    enhanced_query = (
        f"{context_section}\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"User Question: {query}\n\n"
        f"{structure_txt}\n\n"
        f"{ordering_contract}\n"
        f"Web search returned {search_count} results.\n"
        f"Return ONLY valid JSON matching the template and include all required fields."
    )

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "temperature": 0.0,
        "max_tokens": 2400,
        "top_p": 1.0,
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

        parsed = parse_json_safely(content, "Perplexity")
        if not parsed:
            return create_fallback_response(query, search_count, web_context)

        repaired = repair_llm_response(parsed)

        # Ensure action is removed even if present
        repaired.pop("action", None)

        validate_numeric_fields(repaired, "Perplexity")

        try:
            llm_obj = LLMResponse.model_validate(repaired)

            # Ensure action not present (belt + suspenders)
            if hasattr(llm_obj, "action"):
                llm_obj.action = None

            # Merge web sources
            if isinstance(web_context, dict) and web_context.get("sources"):
                existing = llm_obj.sources or []
                merged = list(dict.fromkeys(existing + web_context["sources"]))
                llm_obj.sources = merged[:10]
                llm_obj.freshness = "Current (web-enhanced)"

            result = llm_obj.model_dump_json(exclude_none=True)
            cache_llm_response(query, web_context, result)
            return result

        except ValidationError as e:
            st.warning(f"⚠️ Pydantic validation failed: {e}")
            return create_fallback_response(query, search_count, web_context)

    except Exception as e:
        st.error(f"❌ Perplexity API error: {e}")
        return create_fallback_response(query, search_count, web_context)


def query_perplexity_raw(prompt: str, max_tokens: int = 400, timeout: int = 30) -> str:
    """
    Raw Perplexity call that returns text only.
    IMPORTANT: Does NOT attempt to validate as LLMResponse.
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""

def create_fallback_response(query: str, search_count: int, web_context: Dict) -> str:
    """Create fallback response matching schema, excluding None fields and removing action."""
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
        freshness="Current (fallback)",
        action=None
    )

    return fallback.model_dump_json(exclude_none=True)


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
    """
    Parse a numeric string into a comparable base scale.
    Returns a float in "millions" for currency/volume-like values.
    Percentages are returned as their numeric value (e.g., "9.8%" -> 9.8).

    Handles:
      - $58.3B, 58.3B, S$29.8B, 29.8 S$B, USD 21.18 B
      - 58.3 billion, 58.3 bn, 58.3 million, 58.3 mn, 570 thousand
      - 570,000 (interpreted as an absolute count -> converted to millions)
      - 9.8% (kept as 9.8)
    """
    if val_str is None:
        return 0.0

    s = str(val_str).strip()
    if not s:
        return 0.0

    s_low = s.lower()

    # If it's a percentage, return the raw percent number (not millions)
    if "%" in s_low:
        m = re.search(r'(-?\d+(?:\.\d+)?)', s_low)
        if not m:
            return 0.0
        try:
            return float(m.group(1))
        except Exception:
            return 0.0

    # Normalize: remove commas and common currency tokens/symbols
    # (keep letters because we need bn/mn/b/m/k detection)
    s_low = s_low.replace(",", " ")
    for token in ["s$", "usd", "sgd", "us$", "$", "€", "£", "aud", "cad"]:
        s_low = s_low.replace(token, " ")

    # Collapse whitespace
    s_low = re.sub(r"\s+", " ", s_low).strip()

    # Extract the first number
    m = re.search(r'(-?\d+(?:\.\d+)?)', s_low)
    if not m:
        return 0.0

    try:
        num = float(m.group(1))
    except Exception:
        return 0.0

    # Look at the remaining text after the number for unit words/suffix
    tail = s_low[m.end():].strip()

    # Decide multiplier (base = millions)
    # billions -> *1000, millions -> *1, thousands -> *0.001
    multiplier = 1.0

    # Word-based units
    if re.search(r'\b(trillion|tn)\b', tail):
        multiplier = 1_000_000.0  # trillion -> million
    elif re.search(r'\b(billion|bn)\b', tail):
        multiplier = 1000.0
    elif re.search(r'\b(million|mn)\b', tail):
        multiplier = 1.0
    elif re.search(r'\b(thousand|k)\b', tail):
        multiplier = 0.001
    else:
        # Suffix-style units (possibly with spaces), e.g. "29.8 b", "21.18 b", "58.3m"
        # We only look at the very first letter-ish token in tail.
        t0 = tail[:4].strip()  # enough to catch "b", "m", "k"
        if t0.startswith("b"):
            multiplier = 1000.0
        elif t0.startswith("m"):
            multiplier = 1.0
        elif t0.startswith("k"):
            multiplier = 0.001
        else:
            # No unit detected. If it's a big integer like 570000 (jobs, people),
            # interpret as an absolute count and convert to millions.
            # (570000 -> 0.57 million)
            if abs(num) >= 10000 and float(num).is_integer():
                multiplier = 1.0 / 1_000_000.0
            else:
                multiplier = 1.0

    return num * multiplier


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

def numeric_consistency_with_sources_v2(primary_data: dict, web_context: dict) -> float:
    """
    Stable numeric consistency (0-100):
    - Evidence text: search_results snippets + web_context summary + scraped_content
    - Unit-aware parsing via parse_number_with_unit()
    - Range-aware (supports min/max if metric has a 'range' dict)
    - Downweights proxy metrics (is_proxy=True) so they don't tank the score
    """

    try:
        # Prefer canonical metrics if available (has is_proxy, range, etc.)
        metrics = primary_data.get("primary_metrics_canonical") or primary_data.get("primary_metrics") or {}
        if not isinstance(metrics, dict) or not metrics:
            return 50.0

        # -----------------------------
        # Build evidence text corpus
        # -----------------------------
        texts = []

        # 1) snippets
        sr = (web_context or {}).get("search_results") or []
        if isinstance(sr, list):
            for r in sr:
                if isinstance(r, dict):
                    snip = r.get("snippet", "")
                    if isinstance(snip, str) and snip.strip():
                        texts.append(snip)

        # 2) summary
        summary = (web_context or {}).get("summary") or ""
        if isinstance(summary, str) and summary.strip():
            texts.append(summary)

        # 3) scraped_content
        scraped = (web_context or {}).get("scraped_content") or {}
        if isinstance(scraped, dict):
            for _, content in scraped.items():
                if isinstance(content, str) and content.strip():
                    texts.append(content)

        evidence_text = "\n".join(texts)
        if not evidence_text.strip():
            return 45.0  # no evidence stored

        # -----------------------------
        # Extract numeric candidates from evidence text
        # -----------------------------
        # Keep this broad; parse_number_with_unit will normalize.
        patterns = [
            r'\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[BbMmKk]\b',                 # 29.8B, 570K, 1.2M
            r'\$?\s?\d+(?:\.\d+)?\s*(?:billion|million|thousand|bn|mn)\b',      # 29.8 billion, 29.8 bn
            r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b',                               # 570,000
            r'\b\d+(?:\.\d+)?\s*%\b',                                          # 9.8%
        ]

        evidence_numbers = []
        lowered = evidence_text.lower()

        for pat in patterns:
            for m in re.findall(pat, lowered, flags=re.IGNORECASE):
                n = parse_number_with_unit(str(m))
                if n and n > 0:
                    evidence_numbers.append(n)

        # If nothing extracted, don’t penalize too hard
        if not evidence_numbers:
            return 50.0

        # -----------------------------
        # Verify each metric against evidence numbers (tolerance match)
        # -----------------------------
        def _metric_candidates(m: dict) -> list:
            """Return list of candidate numeric values for a metric (range-aware)."""
            out = []
            if not isinstance(m, dict):
                return out

            # Range support: check min/max if present
            rng = m.get("range") if isinstance(m.get("range"), dict) else None
            if rng:
                if rng.get("min") is not None:
                    out.append(rng.get("min"))
                if rng.get("max") is not None:
                    out.append(rng.get("max"))

            # Also check main value
            if m.get("value") is not None:
                out.append(m.get("value"))

            return out

        def _parse_metric_num(val, unit_hint: str = "") -> float:
            # build a value+unit string so parse_number_with_unit has a chance
            if val is None:
                return 0.0
            s = str(val)
            if unit_hint and unit_hint.lower() not in s.lower():
                s = f"{s} {unit_hint}"
            return parse_number_with_unit(s)

        def _is_supported(target: float, evidence_nums: list, rel_tol: float = 0.25) -> bool:
            # same tolerance approach as v1 (25%)
            if not target or target <= 0:
                return False
            for e in evidence_nums:
                if e <= 0:
                    continue
                if abs(target - e) / max(target, e, 1) < rel_tol:
                    return True
            return False

        supported_w = 0.0
        total_w = 0.0

        for _, m in metrics.items():
            if not isinstance(m, dict):
                continue

            unit = str(m.get("unit") or "").strip()

            # proxy weighting
            is_proxy = bool(m.get("is_proxy"))
            w = 0.5 if is_proxy else 1.0

            cands = _metric_candidates(m)
            if not cands:
                continue

            # parse candidates into numeric values
            parsed_targets = []
            for c in cands:
                n = _parse_metric_num(c, unit_hint=unit)
                if n and n > 0:
                    parsed_targets.append(n)

            if not parsed_targets:
                continue

            total_w += w

            # supported if ANY candidate matches evidence
            if any(_is_supported(t, evidence_numbers, rel_tol=0.25) for t in parsed_targets):
                supported_w += w

        if total_w <= 0:
            return 50.0

        ratio = supported_w / total_w
        # Map: keep a soft floor so one miss doesn't tank the whole run
        score = 30.0 + (ratio * 65.0)  # same scale as v1 (30..95)
        return min(max(score, 20.0), 95.0)

    except Exception:
        return 45.0



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
    num_score = numeric_consistency_with_sources_v2(primary_data, web_context)
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
# 8A. DETERMINISTIC DIFF ENGINE
# Pure Python computation - no LLM variance
# =========================================================

@dataclass
class MetricDiff:
    """Single metric change record"""
    name: str
    old_value: Optional[float]
    new_value: Optional[float]
    old_raw: str  # Original string representation
    new_raw: str
    unit: str
    change_pct: Optional[float]
    change_type: str  # 'increased', 'decreased', 'unchanged', 'added', 'removed'

@dataclass
class EntityDiff:
    """Single entity ranking change record"""
    name: str
    old_rank: Optional[int]
    new_rank: Optional[int]
    old_share: Optional[str]
    new_share: Optional[str]
    rank_change: Optional[int]  # Positive = moved up
    change_type: str  # 'moved_up', 'moved_down', 'unchanged', 'added', 'removed'

@dataclass
class FindingDiff:
    """Single finding change record"""
    old_text: Optional[str]
    new_text: Optional[str]
    similarity: float  # 0-100
    change_type: str  # 'retained', 'modified', 'added', 'removed'

@dataclass
class EvolutionDiff:
    """Complete diff between two analyses"""
    old_timestamp: str
    new_timestamp: str
    time_delta_hours: Optional[float]
    metric_diffs: List[MetricDiff]
    entity_diffs: List[EntityDiff]
    finding_diffs: List[FindingDiff]
    stability_score: float  # 0-100
    summary_stats: Dict[str, int]

# =========================================================
# CANONICAL METRIC REGISTRY & SEMANTIC FINDING HASH
# Add this section after the dataclass definitions (around line 1587)
# =========================================================

# ------------------------------------
# CANONICAL METRIC REGISTRY
# Removes LLM control over metric identity
# ------------------------------------

# Metric type definitions with aliases
METRIC_REGISTRY = {
    # Market Size metrics
    "market_size": {
        "canonical_name": "Market Size",
        "aliases": [
            "market size", "market value", "market cap", "total market",
            "global market", "market valuation", "industry size",
            "total addressable market", "tam", "market worth"
        ],
        "unit_type": "currency",
        "category": "size"
    },
    "market_size_current": {
        "canonical_name": "Current Market Size",
        "aliases": [
            "2024 market size", "2025 market size", "current market",
            "present market size", "today market", "current year market",
            "market size 2024", "market size 2025"
        ],
        "unit_type": "currency",
        "category": "size"
    },
    "market_size_projected": {
        "canonical_name": "Projected Market Size",
        "aliases": [
            "projected market", "forecast market", "future market",
            "2026 market", "2027 market", "2028 market", "2029 market", "2030 market",
            "market projection", "expected market size", "estimated market"
        ],
        "unit_type": "currency",
        "category": "size"
    },

    # Growth metrics
    "cagr": {
        "canonical_name": "CAGR",
        "aliases": [
            "cagr", "compound annual growth", "compound growth rate",
            "annual growth rate", "growth rate", "yearly growth"
        ],
        "unit_type": "percentage",
        "category": "growth"
    },
    "yoy_growth": {
        "canonical_name": "YoY Growth",
        "aliases": [
            "yoy growth", "year over year", "year-over-year",
            "annual growth", "yearly growth rate", "growth percentage"
        ],
        "unit_type": "percentage",
        "category": "growth"
    },

    # Revenue metrics
    "revenue": {
        "canonical_name": "Revenue",
        "aliases": [
            "revenue", "sales", "total revenue", "annual revenue",
            "yearly revenue", "total sales", "gross revenue"
        ],
        "unit_type": "currency",
        "category": "financial"
    },

    # Market share
    "market_share": {
        "canonical_name": "Market Share",
        "aliases": [
            "market share", "share", "market portion", "market percentage",
            "share of market"
        ],
        "unit_type": "percentage",
        "category": "share"
    },

    # Volume metrics
    "units_sold": {
        "canonical_name": "Units Sold",
        "aliases": [
            "units sold", "unit sales", "volume", "units shipped",
            "shipments", "deliveries", "production volume"
        ],
        "unit_type": "count",
        "category": "volume"
    },

    # Pricing
    "average_price": {
        "canonical_name": "Average Price",
        "aliases": [
            "average price", "avg price", "mean price", "asp",
            "average selling price", "unit price"
        ],
        "unit_type": "currency",
        "category": "pricing"
    },
    # -------------------------
    # Country / Macro metrics
    # -------------------------
    "gdp": {
        "canonical_name": "GDP",
        "aliases": ["gdp", "gross domestic product", "economic output"],
        "unit_type": "currency",
        "category": "macro"
    },
    "gdp_per_capita": {
        "canonical_name": "GDP per Capita",
        "aliases": ["gdp per capita", "gdp/capita", "income per person", "per capita gdp"],
        "unit_type": "currency",
        "category": "macro"
    },
    "gdp_growth": {
        "canonical_name": "GDP Growth",
        "aliases": ["gdp growth", "economic growth", "growth rate of gdp", "real gdp growth"],
        "unit_type": "percentage",
        "category": "macro"
    },
    "population": {
        "canonical_name": "Population",
        "aliases": ["population", "population size", "number of people"],
        "unit_type": "count",
        "category": "macro"
    },
    "exports": {
        "canonical_name": "Exports",
        "aliases": ["exports", "export value", "total exports"],
        "unit_type": "currency",
        "category": "trade"
    },
    "imports": {
        "canonical_name": "Imports",
        "aliases": ["imports", "import value", "total imports"],
        "unit_type": "currency",
        "category": "trade"
    },
    "inflation": {
        "canonical_name": "Inflation",
        "aliases": ["inflation", "cpi", "consumer price index", "inflation rate"],
        "unit_type": "percentage",
        "category": "macro"
    },
    "interest_rate": {
        "canonical_name": "Interest Rate",
        "aliases": ["interest rate", "policy rate", "benchmark rate", "central bank rate"],
        "unit_type": "percentage",
        "category": "macro"
    }

}

# Year extraction pattern
YEAR_PATTERN = re.compile(r'(20\d{2})')

# ------------------------------------
# DETERMINISTIC QUESTION SIGNALS
# Drives metric table templates (no LLM)
# ------------------------------------

QUESTION_CATEGORY_TEMPLATES = {
    "country": [
        "gdp",
        "gdp_per_capita",
        "gdp_growth",
        "population",
        "exports",
        "imports",
        "inflation",
        "interest_rate",
    ],
    "industry": [
        "market_size_current",
        "market_size_projected",
        "cagr",
        "revenue",
        "market_share",
        "units_sold",
        "average_price",
    ],
}

def get_expected_metric_ids_for_category(category: str) -> List[str]:
    """
    Domain-agnostic mapping from a template/category string to expected metric IDs.

    Backward compatible:
      - accepts legacy categories like 'country', 'industry', 'company', 'generic'
      - also accepts template IDs like 'ENTITY_OVERVIEW_MARKET_LIGHT_V1', etc.

    NOTE:
    - This function returns a *default* set for a given template/category.
    - The profiler (classify_question_signals) can override/compose expected_metric_ids dynamically.
    """
    c_raw = (category or "unknown").strip()
    c = c_raw.lower().strip()

    # -------------------------
    # New generalized templates
    # -------------------------
    if c in {"entity_overview_country_light_v1", "entity_overview_country_v1"}:
        return [
            "population",
            "gdp_nominal",
            "gdp_per_capita",
            "gdp_growth",
            "inflation",
            "currency",
            "unemployment",
            "exports",
            "imports",
            "top_industries",
        ]

    if c in {"entity_overview_market_light_v1"}:
        return [
            "market_size_current",
            "market_size_projected",
            "cagr",
            "key_trends",
            "top_players",
        ]

    if c in {"entity_overview_market_heavy_v1"}:
        return [
            "market_size_current",
            "market_size_projected",
            "cagr",
            "key_trends",
            "top_players",
            "key_regions",
            "segments",
            "market_share",
            "revenue",
            "units_sold",
            "average_price",
        ]

    if c in {"entity_overview_company_light_v1", "entity_overview_company_v1"}:
        return [
            "revenue",
            "growth",
            "gross_margin",
            "operating_margin",
            "net_income",
            "market_cap",
            "valuation_multiple",
        ]

    if c in {"entity_overview_product_light_v1", "entity_overview_product_v1"}:
        return [
            "average_price",
            "units_sold",
            "market_share",
            "growth",
            "key_trends",
        ]

    if c in {"entity_overview_topic_v1", "generic_v1"}:
        return []

    # -------------------------
    # Legacy categories (still supported)
    # -------------------------
    if c == "country":
        return get_expected_metric_ids_for_category("ENTITY_OVERVIEW_COUNTRY_LIGHT_V1")

    if c == "industry":
        # legacy industry defaults to light market
        return get_expected_metric_ids_for_category("ENTITY_OVERVIEW_MARKET_LIGHT_V1")

    if c == "company":
        return get_expected_metric_ids_for_category("ENTITY_OVERVIEW_COMPANY_LIGHT_V1")

    if c == "generic":
        return []

    # fallback
    return []


def classify_question_signals(query: str) -> Dict[str, Any]:
    """
    Deterministically classify query and return:
      - category: high-level bucket used for templates (country | industry | company | generic)
      - expected_metric_ids: list[str]
      - signals: list[str] (debuggable reasons)
      - years: list[int]
      - regions: list[str]
      - intents: list[str] (market_size, growth_forecast, competitive_landscape, pricing, regulation, consumer_demand, supply_chain, investment, macro_outlook)
    """
    q_raw = (query or "").strip()
    q = q_raw.lower().strip()
    signals: List[str] = []

    if not q:
        return {
            "category": "generic",
            "expected_metric_ids": [],
            "signals": ["empty_query"],
            "years": [],
            "regions": [],
            "intents": []
        }

    # -------------------------
    # 1) Extract years (deterministic)
    # -------------------------
    years: List[int] = []
    try:
        year_matches = re.findall(r"\b(19|20)\d{2}\b", q_raw)
        # The regex above returns the first group; re-run with a non-capturing group to capture full year strings.
        year_matches_full = re.findall(r"\b(?:19|20)\d{2}\b", q_raw)
        years = sorted({int(y) for y in year_matches_full})
        if years:
            signals.append(f"years:{','.join(map(str, years[:8]))}")
    except Exception:
        years = []

    # -------------------------
    # 2) Extract regions/countries (best-effort deterministic; spaCy if available)
    # -------------------------
    regions: List[str] = []
    try:
        nlp = _try_spacy_nlp()
        if nlp:
            doc = nlp(q_raw)
            gpes = [ent.text.strip() for ent in getattr(doc, "ents", []) if ent.label_ in ("GPE", "LOC")]
            regions = []
            for g in gpes:
                if g and g.lower() not in [x.lower() for x in regions]:
                    regions.append(g)
            if regions:
                signals.append(f"regions_spacy:{','.join(regions[:6])}")
    except Exception:
        pass

    # Fallback: very lightweight region tokens
    if not regions:
        region_tokens = [
            "singapore", "malaysia", "indonesia", "thailand", "vietnam", "philippines",
            "china", "india", "japan", "korea", "australia",
            "usa", "united states", "europe", "uk", "united kingdom",
            "asean", "southeast asia", "sea", "global", "worldwide"
        ]
        hits = [t for t in region_tokens if t in q]
        if hits:
            # Keep original casing loosely (title-case single words)
            regions = [h.title() if " " not in h else h.upper() if h in ("usa", "uk") else h.title() for h in hits[:6]]
            signals.append(f"regions_kw:{','.join(hits[:6])}")

    # -------------------------
    # 3) Intent detection (domain-agnostic)
    # -------------------------
    intent_patterns: Dict[str, List[str]] = {
        "market_size": ["market size", "tam", "total addressable market", "how big", "size of the market", "market value"],
        "growth_forecast": ["cagr", "forecast", "projection", "by 20", "growth rate", "expected to", "outlook", "trend"],
        "competitive_landscape": ["key players", "competitors", "market share", "top companies", "leading players", "who are the players"],
        "pricing": ["pricing", "price", "asp", "average selling price", "cost", "margins"],
        "consumer_demand": ["demand", "users", "penetration", "adoption", "consumer", "customer", "behavior"],
        "supply_chain": ["supply", "capacity", "production", "manufacturing", "inventory", "shipment", "lead time"],
        "regulation": ["regulation", "policy", "law", "compliance", "tax", "tariff", "subsidy"],
        "investment": ["investment", "capex", "funding", "valuation", "roi", "profit", "ebitda"],
        "macro_outlook": ["gdp", "inflation", "interest rate", "policy rate", "exports", "imports", "currency", "exchange rate", "per capita"],
    }

    intents: List[str] = []
    for intent, pats in intent_patterns.items():
        if any(p in q for p in pats):
            intents.append(intent)

    # Small disambiguation: "by 2030" etc. strongly suggests forecast if years exist
    if years and "growth_forecast" not in intents and any(yr >= 2025 for yr in years):
        intents.append("growth_forecast")

    if intents:
        signals.append(f"intents:{','.join(intents[:10])}")

    # -------------------------
    # 4) Category decision (template driver)
    # -------------------------
    # Keep it coarse: country vs industry vs company vs generic
    country_kw = [
        "gdp", "per capita", "population", "exports", "imports",
        "inflation", "cpi", "interest rate", "policy rate", "central bank",
        "currency", "exchange rate"
    ]
    company_kw = ["revenue", "earnings", "profit", "ebitda", "guidance", "quarter", "fy", "10-k", "10q", "balance sheet"]
    industry_kw = [
        "market", "industry", "sector", "tam", "cagr", "market size", "market share",
        "key players", "competitors", "pricing", "forecast", "outlook"
    ]

    country_hits = [k for k in country_kw if k in q]
    company_hits = [k for k in company_kw if k in q]
    industry_hits = [k for k in industry_kw if k in q]

    # If macro intent is present, strongly bias to country
    if "macro_outlook" in intents and (regions or country_hits):
        category = "country"
        signals.append("category_rule:macro_outlook_bias_country")
    elif company_hits and not industry_hits:
        category = "company"
        signals.append(f"category_rule:company_keywords:{','.join(company_hits[:5])}")
    elif industry_hits and not country_hits:
        category = "industry"
        signals.append(f"category_rule:industry_keywords:{','.join(industry_hits[:5])}")
    elif industry_hits and country_hits:
        # tie-break: if market sizing/competitive signals exist -> industry; if macro_outlook -> country
        if "macro_outlook" in intents:
            category = "country"
            signals.append("category_rule:mixed_signals_macro_wins")
        else:
            category = "industry"
            signals.append("category_rule:mixed_signals_default_to_industry")
    else:
        category = "generic"
        signals.append("category_rule:no_template_keywords")

    # -------------------------
    # 5) Expected metric IDs (category + intent)
    # -------------------------
    expected_metric_ids: List[str] = []
    try:
        expected_metric_ids = get_expected_metric_ids_for_category(category) or []
    except Exception:
        expected_metric_ids = []

    # Add a few intent-driven metric IDs (only if your registry supports them)
    intent_metric_suggestions = {
        "market_size": ["market_size", "market_size_2024", "market_size_2025"],
        "growth_forecast": ["cagr", "forecast_period", "market_size_2030"],
        "competitive_landscape": ["market_share", "top_players"],
        "pricing": ["avg_price", "asp"],
        "consumer_demand": ["users", "penetration", "arpu"],
        "supply_chain": ["capacity", "shipments"],
        "investment": ["capex", "profit", "ebitda"],
        "macro_outlook": ["gdp", "inflation", "interest_rate", "exchange_rate"],
    }

    for intent in intents:
        for mid in intent_metric_suggestions.get(intent, []):
            if mid not in expected_metric_ids:
                expected_metric_ids.append(mid)

    return {
        "category": category,
        "expected_metric_ids": expected_metric_ids,
        "signals": signals,
        "years": years,
        "regions": regions,
        "intents": intents
    }


    def _contains_any(needle_list: List[str]) -> bool:
        return any(k in q for k in needle_list)

    # -------------------------
    # Determine intents
    # -------------------------
    intents: List[str] = []
    for intent, kws in intent_triggers.items():
        if _contains_any(kws):
            intents.append(intent)

    if intents:
        signals.append("intents:" + ",".join(sorted(set(intents))))

    # -------------------------
    # Determine entity_kind (best-effort heuristic)
    # -------------------------
    is_marketish = _contains_any(market_entity_kw) or any(i in intents for i in ["size", "growth", "forecast", "share", "segments", "players", "regions"])
    is_companyish = _contains_any(company_entity_kw) and not _contains_any(country_entity_kw)
    is_countryish = _contains_any(country_entity_kw) and not is_companyish
    is_productish = _contains_any(product_entity_kw) and not (is_marketish or is_countryish or is_companyish)

    if is_countryish:
        entity_kind = "country"
        signals.append("entity_kind:country")
    elif is_companyish:
        entity_kind = "company"
        signals.append("entity_kind:company")
    elif is_productish:
        entity_kind = "product"
        signals.append("entity_kind:product")
    elif is_marketish:
        entity_kind = "market"
        signals.append("entity_kind:market")
    else:
        entity_kind = "topic_general"
        signals.append("entity_kind:topic_general")

    # -------------------------
    # Determine scope
    # -------------------------
    is_comparative = _contains_any(comparative_kw)
    is_forecasty = _contains_any(forecast_kw) or bool(YEAR_PATTERN.findall(q_raw))

    # Broad overview should win when user explicitly asks for general explainer
    # BUT: if they also mention measurable intents (size/growth/forecast/etc.), treat as metrics_light.
    is_broad_phrase = _contains_any(broad_phrases)

    if is_comparative:
        scope = "comparative"
        signals.append("scope:comparative")
    elif is_forecasty and any(i in intents for i in ["forecast", "growth", "size"]):
        scope = "forecast_specific"
        signals.append("scope:forecast_specific")
    elif is_broad_phrase and not intents:
        scope = "broad_overview"
        signals.append("scope:broad_overview")
    else:
        # metrics light vs heavy
        heavy_asks = ["segments", "share", "volume", "regions", "players"]
        heavy_requested = any(i in intents for i in heavy_asks)
        if heavy_requested:
            scope = "metrics_heavy"
            signals.append("scope:metrics_heavy")
        else:
            scope = "metrics_light"
            signals.append("scope:metrics_light")

    # -------------------------
    # Map entity_kind -> category (backward compatible)
    # -------------------------
    if entity_kind == "country":
        category = "country"
    elif entity_kind == "company":
        category = "company"
    elif entity_kind in {"market", "product"}:
        category = "industry"
    else:
        category = "generic"

    # -------------------------
    # Choose generalized template + tiers
    # -------------------------
    # Tier meanings:
    #  1 = high extractability (size/growth/forecast)
    #  2 = medium (players/regions/basic segments)
    #  3 = low (granular channels, detailed splits) -> only if explicitly asked
    if category == "country":
        metric_template_id = "ENTITY_OVERVIEW_COUNTRY_LIGHT_V1" if scope != "metrics_heavy" else "ENTITY_OVERVIEW_COUNTRY_LIGHT_V1"
        metric_tiers_enabled = [1]
    elif category == "company":
        metric_template_id = "ENTITY_OVERVIEW_COMPANY_LIGHT_V1"
        metric_tiers_enabled = [1]
    elif category == "industry":
        if scope in {"metrics_heavy", "comparative"}:
            metric_template_id = "ENTITY_OVERVIEW_MARKET_HEAVY_V1"
            metric_tiers_enabled = [1, 2]
        else:
            metric_template_id = "ENTITY_OVERVIEW_MARKET_LIGHT_V1"
            metric_tiers_enabled = [1]
    else:
        metric_template_id = "ENTITY_OVERVIEW_TOPIC_V1"
        metric_tiers_enabled = []

    # -------------------------
    # Build expected_metric_ids dynamically from intents (domain-agnostic)
    # -------------------------
    # Slot -> metric id mapping (kept generic; avoids tourism specialization)
    # If you later add more canonical IDs, expand these mappings.
    market_slot_to_id = {
        "size_current": "market_size_current",
        "size_projected": "market_size_projected",
        "growth_cagr": "cagr",
        "growth_yoy": "growth",
        "share_key": "market_share",
        "volume_current": "units_sold",
        "price_avg": "average_price",
        "players_top": "top_players",
        "regions_key": "key_regions",
        "segments_basic": "segments",
        "trends": "key_trends",
        "revenue": "revenue",
    }

    company_slot_to_id = {
        "revenue": "revenue",
        "growth": "growth",
        "gross_margin": "gross_margin",
        "operating_margin": "operating_margin",
        "net_income": "net_income",
        "market_cap": "market_cap",
        "valuation_multiple": "valuation_multiple",
        "trends": "key_trends",
    }

    country_slot_to_id = {
        "population": "population",
        "gdp_nominal": "gdp_nominal",
        "gdp_per_capita": "gdp_per_capita",
        "gdp_growth": "gdp_growth",
        "inflation": "inflation",
        "currency": "currency",
        "unemployment": "unemployment",
        "exports": "exports",
        "imports": "imports",
        "top_industries": "top_industries",
        "trends": "key_trends",
    }

    # Determine slots from intents
    slots: List[str] = []
    if entity_kind == "country":
        # For countries: macro defaults if broad, otherwise macro intents
        if scope == "broad_overview":
            slots = ["population", "gdp_nominal", "gdp_per_capita", "gdp_growth", "inflation", "currency", "top_industries"]
        else:
            # If user asks for macro (or didn’t specify), still give a tight macro set
            slots = ["population", "gdp_nominal", "gdp_growth", "inflation", "currency"]
            if "macro" in intents:
                slots += ["unemployment", "exports", "imports"]

        mapper = country_slot_to_id

    elif entity_kind == "company":
        slots = ["revenue", "growth", "gross_margin", "operating_margin", "net_income", "market_cap", "valuation_multiple"]
        mapper = company_slot_to_id

    elif entity_kind in {"market", "product"}:
        # Tier 1 core (always when metrics_* scope)
        if scope == "broad_overview":
            slots = ["trends", "players_top"]
        else:
            slots = ["size_current", "growth_cagr"]
            if "forecast" in intents:
                slots.append("size_projected")
            if "trends" in intents:
                slots.append("trends")
            # Tier 2 (only when explicitly asked or heavy scope)
            if scope in {"metrics_heavy", "comparative"}:
                if "players" in intents:
                    slots.append("players_top")
                if "regions" in intents:
                    slots.append("regions_key")
                if "segments" in intents:
                    slots.append("segments_basic")
                if "share" in intents:
                    slots.append("share_key")
                if "volume" in intents:
                    slots.append("volume_current")
                if "price" in intents:
                    slots.append("price_avg")
            else:
                # metrics_light: include players/trends only if asked
                if "players" in intents:
                    slots.append("players_top")
                if "regions" in intents:
                    slots.append("regions_key")

        mapper = market_slot_to_id

    else:
        # topic_general
        slots = []
        mapper = {}

    expected_metric_ids = []
    for s in slots:
        mid = mapper.get(s)
        if mid:
            expected_metric_ids.append(mid)

    # If still empty but template provides defaults, use template defaults
    if not expected_metric_ids:
        expected_metric_ids = get_expected_metric_ids_for_category(metric_template_id)

    # De-dup while preserving order
    seen = set()
    expected_metric_ids = [x for x in expected_metric_ids if not (x in seen or seen.add(x))]

    # -------------------------
    # Preferred source classes (generic)
    # -------------------------
    if category == "country":
        preferred_source_classes = ["official_stats", "government", "reputable_org", "reference"]
    elif category == "company":
        preferred_source_classes = ["official_filings", "investor_relations", "reputable_org", "news"]
    elif category == "industry":
        preferred_source_classes = ["industry_association", "reputable_org", "official_stats", "news", "research_portal"]
    else:
        preferred_source_classes = ["reference", "official_stats", "reputable_org"]

    # Attach year detection signal
    years = sorted(set(YEAR_PATTERN.findall(q_raw))) if YEAR_PATTERN.findall(q_raw) else []
    if years:
        signals.append("years_detected:" + ",".join(years))

    return {
        "category": category,
        "expected_metric_ids": expected_metric_ids,
        "signals": signals,
        "entity_kind": entity_kind,
        "scope": scope,
        "metric_template_id": metric_template_id,
        "metric_tiers_enabled": metric_tiers_enabled,
        "preferred_source_classes": preferred_source_classes,
        "intents": sorted(set(intents)),
    }


def get_canonical_metric_id(metric_name: str) -> Tuple[str, str]:
    """
    Map a metric name to its canonical ID and display name.

    Returns:
        Tuple of (canonical_id, canonical_display_name)

    Example:
        "2024 Market Size" -> ("market_size_2024", "Market Size (2024)")
        "Global Market Value" -> ("market_size", "Market Size")
        "CAGR 2024-2030" -> ("cagr_2024_2030", "CAGR (2024-2030)")
    """
    if not metric_name:
        return ("unknown", "Unknown Metric")

    name_lower = metric_name.lower().strip()
    name_normalized = re.sub(r'[^\w\s]', ' ', name_lower)
    name_normalized = re.sub(r'\s+', ' ', name_normalized).strip()

    # Extract years
    years = YEAR_PATTERN.findall(metric_name)
    year_suffix = "_".join(sorted(years)) if years else ""

    # Find best matching registry entry
    best_match_id = None
    best_match_score = 0

    for metric_id, config in METRIC_REGISTRY.items():
        for alias in config["aliases"]:
            # Remove years from alias for comparison
            alias_no_year = YEAR_PATTERN.sub('', alias).strip()
            name_no_year = YEAR_PATTERN.sub('', name_normalized).strip()

            # Exact match
            if alias_no_year == name_no_year:
                best_match_id = metric_id
                best_match_score = 1.0
                break

            # Containment match
            if alias_no_year in name_no_year or name_no_year in alias_no_year:
                score = len(alias_no_year) / max(len(name_no_year), 1)
                if score > best_match_score:
                    best_match_id = metric_id
                    best_match_score = score

            # Word overlap match
            alias_words = set(alias_no_year.split())
            name_words = set(name_no_year.split())
            if alias_words and name_words:
                overlap = len(alias_words & name_words) / len(alias_words | name_words)
                if overlap > best_match_score:
                    best_match_id = metric_id
                    best_match_score = overlap

        if best_match_score == 1.0:
            break

    # Build canonical ID and display name
    if best_match_id and best_match_score > 0.4:
        config = METRIC_REGISTRY[best_match_id]
        canonical_base = best_match_id
        display_name = config["canonical_name"]

        if year_suffix:
            canonical_id = f"{canonical_base}_{year_suffix}"
            if len(years) == 1:
                display_name = f"{display_name} ({years[0]})"
            else:
                display_name = f"{display_name} ({'-'.join(years)})"
        else:
            canonical_id = canonical_base

        return (canonical_id, display_name)

    # Fallback: create ID from normalized name
    fallback_id = re.sub(r'\s+', '_', name_normalized)
    if year_suffix:
        fallback_id = f"{fallback_id}_{year_suffix}" if year_suffix not in fallback_id else fallback_id

    return (fallback_id, metric_name)

# ------------------------------------
# GEO + PROXY TAGGING (DETERMINISTIC)
# ------------------------------------

import re
from typing import Dict, Any, Tuple, List, Optional

REGION_KEYWORDS = {
    "APAC": ["apac", "asia pacific", "asia-pacific"],
    "SOUTHEAST_ASIA": ["southeast asia", "asean", "sea "],  # note space to reduce false matches
    "ASIA": ["asia"],
    "EUROPE": ["europe", "eu", "emea"],
    "NORTH_AMERICA": ["north america"],
    "LATAM": ["latin america", "latam"],
    "MIDDLE_EAST": ["middle east", "mena"],
    "AFRICA": ["africa"],
    "OCEANIA": ["oceania", "australia", "new zealand"],
}

GLOBAL_KEYWORDS = ["global", "worldwide", "world", "international", "across the world"]

# Minimal country map (expand deterministically over time)
COUNTRY_KEYWORDS = {
    "Singapore": ["singapore", "sg"],
    "United States": ["united states", "usa", "u.s.", "us"],
    "United Kingdom": ["united kingdom", "uk", "u.k.", "britain", "england"],
    "China": ["china", "prc"],
    "Japan": ["japan"],
    "India": ["india"],
    "Indonesia": ["indonesia"],
    "Malaysia": ["malaysia"],
    "Thailand": ["thailand"],
    "Vietnam": ["vietnam"],
    "Philippines": ["philippines"],
}

def infer_geo_scope(*texts: str) -> Dict[str, str]:
    """
    Deterministically infer geography from text.
    Returns {"geo_scope": "local|regional|global|unknown", "geo_name": "<name or ''>"}.
    Priority: country > region > global.
    """
    combined = " ".join([t for t in texts if isinstance(t, str) and t.strip()]).lower()
    if not combined:
        return {"geo_scope": "unknown", "geo_name": ""}

    # 1) Country/local (most specific)
    for country, kws in COUNTRY_KEYWORDS.items():
        for kw in kws:
            if kw in combined:
                return {"geo_scope": "local", "geo_name": country}

    # 2) Region
    for region_name, kws in REGION_KEYWORDS.items():
        for kw in kws:
            if kw in combined:
                pretty = region_name.replace("_", " ").title()
                return {"geo_scope": "regional", "geo_name": pretty}

    # 3) Global
    for kw in GLOBAL_KEYWORDS:
        if kw in combined:
            return {"geo_scope": "global", "geo_name": "Global"}

    return {"geo_scope": "unknown", "geo_name": ""}


# ---- Proxy labeling ----
# "Proxy" = adjacent metric that can help approximate the target but isn't the target definition.
# You can expand these sets deterministically.

PROXY_PATTERNS = [
    # (pattern, proxy_type, reason_template)
    (r"\bapparel\b|\bfashion\b|\bclothing\b", "adjacent_category", "Uses apparel/fashion as an adjacent proxy for streetwear."),
    (r"\bfootwear\b|\bsneaker\b|\bshoes\b", "subsegment", "Uses footwear/sneakers as a subsegment proxy for the broader market."),
    (r"\bresale\b|\bsecondary market\b", "channel_proxy", "Uses resale/secondary-market measures as a channel proxy."),
    (r"\be-?commerce\b|\bonline sales\b|\bsocial commerce\b", "channel_proxy", "Uses e-commerce indicators as a channel proxy."),
    (r"\btourism\b|\bvisitor\b|\btravel retail\b", "demand_driver", "Uses tourism indicators as a demand-driver proxy."),
    (r"\bsearch interest\b|\bgoogle trends\b|\bweb traffic\b", "interest_proxy", "Uses interest/attention measures as a proxy."),
]

# These are words that signal "core market size" style metrics (usually non-proxy if they match the user topic).
CORE_MARKET_PATTERNS = [
    r"\bmarket size\b",
    r"\bmarket value\b",
    r"\brevenue\b",
    r"\bsales\b",
    r"\bcagr\b",
    r"\bgrowth\b",
    r"\bprojected\b|\bforecast\b",
]

def infer_proxy_label(
    metric_name: str,
    question_text: str = "",
    category_hint: str = "",
    *extra_context: str
) -> Dict[str, Any]:
    """
    Deterministically label a metric as proxy/non-proxy.

    Returns fields:
      is_proxy: bool
      proxy_type: str
      proxy_reason: str
      proxy_confidence: float (0-1)
      proxy_target: str (best-guess target topic)
    """
    name = (metric_name or "").lower().strip()
    q = (question_text or "").lower().strip()
    ctx = " ".join([c for c in extra_context if isinstance(c, str)]).lower()

    combined = " ".join([name, q, ctx]).strip()

    # Default: not proxy
    out = {
        "is_proxy": False,
        "proxy_type": "",
        "proxy_reason": "",
        "proxy_confidence": 0.0,
        "proxy_target": ""
    }

    if not combined:
        return out

    # Best-effort target topic extraction (very light heuristic)
    # If you already have question signals elsewhere, you can pass them in category_hint/question_text.
    # Here we just keep a short phrase if present.
    proxy_target = ""
    if "streetwear" in q:
        proxy_target = "streetwear"
    elif "semiconductor" in q:
        proxy_target = "semiconductors"
    elif "battery" in q:
        proxy_target = "batteries"
    out["proxy_target"] = proxy_target

    # If metric name itself looks like core market patterns AND includes the target keyword, treat as non-proxy.
    # (prevents incorrectly labeling "Singapore streetwear market size" as proxy)
    core_like = any(re.search(p, name) for p in CORE_MARKET_PATTERNS)
    if core_like:
        # If it explicitly contains the topic keyword, strongly non-proxy
        if proxy_target and proxy_target in name:
            return out
        # If it says "streetwear market" in name, non-proxy even if target not detected
        if "streetwear" in name:
            return out

    # Detect proxies using patterns.
    for pat, ptype, reason in PROXY_PATTERNS:
        if re.search(pat, combined):
            out["is_proxy"] = True
            out["proxy_type"] = ptype
            out["proxy_reason"] = reason
            # Confidence: stronger if pattern appears in metric name; weaker if only in context.
            if re.search(pat, name):
                out["proxy_confidence"] = 0.9
            elif re.search(pat, ctx):
                out["proxy_confidence"] = 0.7
            else:
                out["proxy_confidence"] = 0.6
            return out

    return out


def merge_group_geo(group: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Choose the most frequent geo tag within a merged group deterministically.
    Returns (geo_scope, geo_name).
    """
    items = []
    for g in group:
        s = g.get("geo_scope", "unknown")
        n = g.get("geo_name", "")
        if s and s != "unknown":
            items.append((s, n))

    if not items:
        return "unknown", ""

    counts: Dict[str, int] = {}
    for s, n in items:
        k = f"{s}|{n}"
        counts[k] = counts.get(k, 0) + 1

    best_k = max(counts.items(), key=lambda kv: kv[1])[0]  # deterministic tie via insertion order after stable sort
    s, n = best_k.split("|", 1)
    return s, n


def merge_group_proxy(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge proxy labels for duplicates deterministically.
    If ANY member is proxy -> merged metric is proxy.
    Choose the highest-confidence proxy candidate.
    """
    best = None
    best_conf = -1.0

    for g in group:
        is_proxy = bool(g.get("is_proxy", False))
        conf = float(g.get("proxy_confidence", 0.0) or 0.0)
        if is_proxy and conf > best_conf:
            best_conf = conf
            best = g

    if best is None:
        return {
            "is_proxy": False,
            "proxy_type": "",
            "proxy_reason": "",
            "proxy_confidence": 0.0,
            "proxy_target": "",
        }

    return {
        "is_proxy": True,
        "proxy_type": best.get("proxy_type", ""),
        "proxy_reason": best.get("proxy_reason", ""),
        "proxy_confidence": float(best.get("proxy_confidence", 0.0) or 0.0),
        "proxy_target": best.get("proxy_target", ""),
    }

def canonicalize_metrics(
    metrics: Dict,
    merge_duplicates_to_range: bool = True,
    question_text: str = "",
    category_hint: str = ""
) -> Dict:
    """
    Convert metrics to canonical IDs, but NEVER merge across incompatible dimensions.

    Key fix:
      - Adds deterministic 'dimension' classification and incorporates it into canonical keys.
      - Prevents revenue vs unit-sales from merging just because the year matches.
      - Keeps your geo + proxy tagging behavior.

    Output:
      canonicalized[canonical_key] -> metric dict with:
        - canonical_id (base id)
        - canonical_key (dimension-safe id you should use everywhere downstream)
        - dimension (currency | unit_sales | percent | count | index | unknown)
        - name (dimension-corrected display name)
    """
    if not isinstance(metrics, dict):
        return {}

    def infer_metric_dimension(metric_name: str, unit_raw: str) -> str:
        n = (metric_name or "").lower()
        u = (unit_raw or "").strip().lower()

        # Percent
        if "%" in u or "percent" in n or "share" in n or "cagr" in n:
            return "percent"

        # Currency signals
        currency_tokens = ["$", "s$", "usd", "sgd", "eur", "€", "gbp", "£", "jpy", "¥", "cny", "rmb", "aud", "cad"]
        if any(t in u for t in currency_tokens) or any(t in n for t in ["revenue", "market value", "valuation", "value (", "usd", "sgd", "eur"]):
            return "currency"

        # Unit sales / shipments
        unit_tokens = ["unit", "units", "sold", "sales volume", "shipments", "registrations", "deliveries", "vehicles", "pcs", "pieces", "volume"]
        if any(t in n for t in unit_tokens):
            return "unit_sales"

        # Pure counts
        if any(t in n for t in ["count", "number of", "install base", "installed base", "users", "subscribers"]) and "revenue" not in n:
            return "count"

        # Index / score
        if any(t in n for t in ["index", "score", "rating"]):
            return "index"

        return "unknown"

    def display_name_for_dimension(original_display: str, dim: str) -> str:
        # If registry mapped it wrongly (e.g. "Revenue (2025)" but actually unit sales),
        # override display name to avoid misleading labels.
        if not original_display:
            return original_display

        od = original_display.strip()
        od_low = od.lower()

        if dim == "unit_sales":
            # Replace "Revenue" phrasing if it exists
            if "revenue" in od_low or "market value" in od_low or "valuation" in od_low:
                return re.sub(r"(?i)revenue|market value|valuation", "Unit Sales", od).strip()
            # If it just says "Sales", keep but make explicit
            if od_low.startswith("sales"):
                return "Unit Sales" + od[len("Sales"):]
            if "sales" in od_low:
                return re.sub(r"(?i)sales", "Unit Sales", od).strip()
            return od

        if dim == "currency":
            # If it says "Unit Sales" but unit is currency, flip back
            if "unit sales" in od_low:
                return re.sub(r"(?i)unit sales", "Revenue", od).strip()
            return od

        if dim == "percent":
            # Prefer "Share" / "CAGR" style if it looks like one
            if "unit sales" in od_low or "revenue" in od_low:
                return od  # don’t aggressively rename; leave as-is
            return od

        return od

    candidates = []

    for key, metric in metrics.items():
        if not isinstance(metric, dict):
            continue

        original_name = metric.get("name", key)
        canonical_id, canonical_name = get_canonical_metric_id(original_name)

        raw_unit = (metric.get("unit") or "").strip()
        unit_norm = raw_unit.upper()

        dim = infer_metric_dimension(str(original_name), raw_unit)

        # Dimension-safe canonical key (this is what you group/merge on)
        canonical_key = f"{canonical_id}__{dim}"

        parsed_val = parse_to_float(metric.get("value"))
        value_for_sort = parsed_val if parsed_val is not None else str(metric.get("value", ""))

        stable_sort_key = (
            str(original_name).lower().strip(),
            dim,
            unit_norm,
            str(value_for_sort),
            str(key),
        )

        geo = infer_geo_scope(
            str(original_name),
            str(metric.get("context_snippet", "")),
            str(metric.get("source", "")),
            str(metric.get("source_url", "")),
        )

        proxy = infer_proxy_label(
            str(original_name),
            str(question_text),
            str(category_hint),
            str(metric.get("context_snippet", "")),
            str(metric.get("source", "")),
            str(metric.get("source_url", "")),
        )

        candidates.append({
            "canonical_id": canonical_id,
            "canonical_key": canonical_key,
            "canonical_name": display_name_for_dimension(canonical_name, dim),
            "original_name": original_name,
            "metric": metric,
            "unit": unit_norm,
            "parsed_val": parsed_val,
            "dimension": dim,
            "stable_sort_key": stable_sort_key,
            "geo_scope": geo["geo_scope"],
            "geo_name": geo["geo_name"],
            **proxy,
        })

    candidates.sort(key=lambda x: x["stable_sort_key"])

    # Group by canonical_key (NOT canonical_id)
    grouped: Dict[str, List[Dict]] = {}
    for c in candidates:
        grouped.setdefault(c["canonical_key"], []).append(c)

    canonicalized: Dict[str, Dict] = {}

    for ckey, group in grouped.items():
        # Single metric or no merge requested
        if len(group) == 1 or not merge_duplicates_to_range:
            g = group[0]
            m = g["metric"]
            canonicalized[ckey] = {
                **m,
                "name": g["canonical_name"],
                "canonical_id": g["canonical_id"],
                "canonical_key": ckey,
                "dimension": g["dimension"],
                "original_name": g["original_name"],
                "geo_scope": g.get("geo_scope", "unknown"),
                "geo_name": g.get("geo_name", ""),
                "is_proxy": bool(g.get("is_proxy", False)),
                "proxy_type": g.get("proxy_type", ""),
                "proxy_reason": g.get("proxy_reason", ""),
                "proxy_confidence": float(g.get("proxy_confidence", 0.0) or 0.0),
                "proxy_target": g.get("proxy_target", ""),
            }
            continue

        # Merge duplicates within SAME dimension-safe canonical_key
        base = group[0]
        base_metric = dict(base["metric"])
        base_metric["name"] = base["canonical_name"]
        base_metric["canonical_id"] = base["canonical_id"]
        base_metric["canonical_key"] = ckey
        base_metric["dimension"] = base["dimension"]

        geo_scope, geo_name = merge_group_geo(group)
        base_metric["geo_scope"] = geo_scope
        base_metric["geo_name"] = geo_name

        merged_proxy = merge_group_proxy(group)
        base_metric.update(merged_proxy)

        vals = [g["parsed_val"] for g in group if g["parsed_val"] is not None]
        raw_vals = [str(g["metric"].get("value", "")) for g in group]
        orig_names = [g["original_name"] for g in group]

        units = [g["unit"] for g in group if g["unit"]]
        unit_base = units[0] if units else (base_metric.get("unit") or "")
        base_metric["unit"] = unit_base

        base_metric["original_names"] = orig_names
        base_metric["raw_values"] = raw_vals

        if vals:
            vals_sorted = sorted(vals)
            vmin, vmax = vals_sorted[0], vals_sorted[-1]
            vmed = vals_sorted[len(vals_sorted) // 2]
            base_metric["value"] = vmed
            base_metric["range"] = {
                "min": vmin,
                "max": vmax,
                "candidates": vals_sorted,
                "n": len(vals_sorted),
            }
        else:
            base_metric["range"] = {"min": None, "max": None, "candidates": [], "n": 0}

        canonicalized[ckey] = base_metric

    return canonicalized


def freeze_metric_schema(canonical_metrics: Dict) -> Dict:
    """
    Lock metric identity + expected schema for future evolution.

    Key fix:
      - Stores canonical_key (dimension-safe)
      - Stores dimension + unit family
      - Keywords include dimension hints to improve later matching
    """
    frozen = {}
    if not isinstance(canonical_metrics, dict):
        return frozen

    def unit_family(unit_raw: str) -> str:
        u = (unit_raw or "").strip().lower()
        if not u:
            return "unknown"
        if "%" in u:
            return "percent"
        if any(t in u for t in ["$", "s$", "usd", "sgd", "eur", "€", "gbp", "£", "jpy", "¥", "cny", "rmb"]):
            return "currency"
        if any(t in u for t in ["b", "bn", "billion", "m", "mn", "million", "k", "thousand", "t", "trillion"]):
            # magnitude without currency symbol -> keep as magnitude
            return "magnitude"
        return "other"

    for ckey, m in canonical_metrics.items():
        if not isinstance(m, dict):
            continue

        dim = (m.get("dimension") or "").strip() or "unknown"
        name = m.get("name")
        unit = (m.get("unit") or "").strip()
        uf = unit_family(unit)

        # Keywords: name + dimension token to prevent cross-dimension matches later
        kws = extract_context_keywords(name or "") or []
        if dim and dim not in kws:
            kws.append(dim)
        if uf and uf not in kws:
            kws.append(uf)

        frozen[ckey] = {
            "canonical_key": ckey,
            "canonical_id": m.get("canonical_id") or ckey.split("__", 1)[0],
            "dimension": dim,
            "name": name,
            "unit": unit_clean_first_letter(unit.upper()),
            "unit_family": uf,
            "keywords": kws[:30],
        }

    return frozen



# =========================================================
# RANGE + SOURCE ATTRIBUTION (DETERMINISTIC, NO LLM)
# =========================================================

def normalize_unit_tag(u: str) -> str:
    """
    Normalize unit to a compact tag for matching:
    - Currency magnitudes: T/B/M
    - Percent: %
    - Unknown: ""
    """
    if not u:
        return ""
    ul = str(u).strip().lower()
    if "%" in ul:
        return "%"

    # common currency magnitude patterns
    if "trillion" in ul or ul.endswith("t"):
        return "T"
    if "billion" in ul or ul.endswith("b"):
        return "B"
    if "million" in ul or ul.endswith("m"):
        return "M"

    # if you store units like "billion USD"
    if ul.startswith("t"):
        return "T"
    if ul.startswith("b"):
        return "B"
    if ul.startswith("m"):
        return "M"

    return ""


def to_billions(value: float, unit_tag: str) -> Optional[float]:
    """Convert T/B/M tagged values into billions. Leaves % unchanged (returns None for % here)."""
    try:
        v = float(value)
    except Exception:
        return None

    if unit_tag == "T":
        return v * 1000.0
    if unit_tag == "B":
        return v
    if unit_tag == "M":
        return v / 1000.0
    return None


def build_metric_keywords(metric_name: str) -> List[str]:
    """Reuse your existing keyword extractor, but ensure we always have something."""
    kws = extract_context_keywords(metric_name) or []
    # Add simple fallback tokens (deterministic)
    for t in re.findall(r"[a-zA-Z]{4,}", str(metric_name).lower()):
        if t not in kws:
            kws.append(t)
    return kws[:25]


def extract_numbers_from_scraped_sources(
    scraped_content: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Deterministically extract numeric candidates from all scraped source texts.
    Returns list of {url, value, unit_tag, raw, context}.
    """
    candidates: List[Dict[str, Any]] = []
    if not isinstance(scraped_content, dict):
        return candidates

    for url, content in scraped_content.items():
        if not content or not isinstance(content, str) or len(content) < 200:
            continue

        nums = extract_numbers_with_context(content)  # you already have this function
        for n in nums:
            unit_tag = normalize_unit_tag(n.get("unit", ""))
            candidates.append({
                "url": url,
                "value": n.get("value"),
                "unit_tag": unit_tag,
                "raw": n.get("raw", ""),
                "context": (n.get("context") or ""),
            })

    return candidates


def attribute_span_to_sources(
    metric_name: str,
    metric_unit: str,
    scraped_content: Dict[str, str],
    rel_tol: float = 0.08,
) -> Dict[str, Any]:
    """
    Build a deterministic span (min/mid/max) for a metric, and attribute min/max to sources.
    Uses only scraped content + regex extractions (NO LLM).
    """
    unit_tag_hint = normalize_unit_tag(metric_unit)
    keywords = build_metric_keywords(metric_name)

    all_candidates = extract_numbers_from_scraped_sources(scraped_content)

    filtered: List[Dict[str, Any]] = []

    for c in all_candidates:
        ctx = c.get("context", "")
        if not ctx:
            continue

        # Context match gate
        ctx_score = calculate_context_match(keywords, ctx)
        if ctx_score <= 0.0:
            continue

        # Unit gate:
        # If metric is %, require %.
        if unit_tag_hint == "%":
            if c.get("unit_tag") != "%":
                continue
            # keep as-is, no scaling
            val_norm = c.get("value")
        else:
            # currency magnitude matching: allow T/B/M and convert to billions
            # if candidate unit_tag missing, skip (too risky)
            if c.get("unit_tag") not in ("T", "B", "M"):
                continue
            val_norm = to_billions(c.get("value"), c.get("unit_tag"))
            if val_norm is None:
                continue

        filtered.append({
            **c,
            "value_norm": val_norm,
            "ctx_score": float(ctx_score),
        })

    if not filtered:
        return {
            "span": None,
            "source_attribution": None,
            "evidence": []
        }

    # Choose min/max deterministically:
    # - primary key: numeric value_norm
    # - tie-breaker: higher ctx_score
    # - final tie-breaker: url lexicographic
    def min_key(x):
        return (x["value_norm"], -x["ctx_score"], str(x.get("url", "")))

    def max_key(x):
        return (-x["value_norm"], -x["ctx_score"], str(x.get("url", "")))

    min_item = sorted(filtered, key=min_key)[0]
    max_item = sorted(filtered, key=max_key)[0]

    vmin = float(min_item["value_norm"])
    vmax = float(max_item["value_norm"])
    vmid = (vmin + vmax) / 2.0

    # For % metrics, keep % in the same numeric scale; for currency, we standardize to "billion USD"
    if unit_tag_hint == "%":
        unit_out = "%"
    else:
        unit_out = "billion USD"

    evidence = []
    for it in sorted(filtered, key=lambda x: (-x["ctx_score"], str(x.get("url", ""))))[:12]:
        evidence.append({
            "url": it.get("url"),
            "raw": it.get("raw"),
            "unit_tag": it.get("unit_tag"),
            "value_norm": it.get("value_norm"),
            "context_snippet": (it.get("context") or "")[:220],
            "context_score": round(float(it.get("ctx_score", 0.0)) * 100, 1),
        })

    return {
        "span": {
            "min": round(vmin, 4),
            "mid": round(vmid, 4),
            "max": round(vmax, 4),
            "unit": unit_out
        },
        "source_attribution": {
            "min": {
                "url": min_item.get("url"),
                "raw": min_item.get("raw"),
                "value_norm": min_item.get("value_norm"),
                "context_snippet": (min_item.get("context") or "")[:220],
                "context_score": round(float(min_item.get("ctx_score", 0.0)) * 100, 1),
            },
            "max": {
                "url": max_item.get("url"),
                "raw": max_item.get("raw"),
                "value_norm": max_item.get("value_norm"),
                "context_snippet": (max_item.get("context") or "")[:220],
                "context_score": round(float(max_item.get("ctx_score", 0.0)) * 100, 1),
            }
        },
        "evidence": evidence
    }


def add_range_and_source_attribution_to_canonical_metrics(
    canonical_metrics: Dict[str, Dict[str, Any]],
    web_context: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Enrich canonical metrics with:
      - value_span (min/mid/max)
      - source_attribution (min/max)
      - evidence list (top candidates)
    Deterministic + no LLM.
    """
    if not isinstance(canonical_metrics, dict):
        return {}

    scraped = {}
    try:
        scraped = (web_context or {}).get("scraped_content", {}) or {}
    except Exception:
        scraped = {}

    enriched = {}
    for cid, m in canonical_metrics.items():
        metric_name = m.get("name") or m.get("original_name") or cid
        metric_unit = m.get("unit") or ""

        span_pack = attribute_span_to_sources(
            metric_name=metric_name,
            metric_unit=metric_unit,
            scraped_content=scraped
        )

        mm = dict(m)
        if span_pack.get("span"):
            mm["value_span"] = span_pack["span"]
        if span_pack.get("source_attribution"):
            mm["source_attribution"] = span_pack["source_attribution"]
        if span_pack.get("evidence"):
            mm["evidence"] = span_pack["evidence"]

        enriched[cid] = mm

    return enriched




# ------------------------------------
# SEMANTIC FINDING HASH
# Removes wording-based churn from findings comparison
# ------------------------------------

# Semantic components to extract from findings
FINDING_PATTERNS = {
    # Growth/decline patterns
    "growth": [
        r'(?:grow(?:ing|th)?|increas(?:e|ing)|expand(?:ing)?|ris(?:e|ing)|up)\s*(?:by|at|of)?\s*(\d+(?:\.\d+)?)\s*%?',
        r'(\d+(?:\.\d+)?)\s*%?\s*(?:growth|increase|expansion|rise)',
    ],
    "decline": [
        r'(?:declin(?:e|ing)|decreas(?:e|ing)|fall(?:ing)?|drop(?:ping)?|down)\s*(?:by|at|of)?\s*(\d+(?:\.\d+)?)\s*%?',
        r'(\d+(?:\.\d+)?)\s*%?\s*(?:decline|decrease|drop|fall)',
    ],

    # Value patterns
    "value": [
        r'\$\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)?',
        r'(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)',
    ],

    # Ranking patterns
    "rank": [
        r'(?:lead(?:ing|er)?|top|first|largest|biggest|#1|number one)',
        r'(?:second|#2|runner.?up)',
        r'(?:third|#3)',
    ],

    # Trend patterns
    "trend_up": [
        r'(?:bullish|optimistic|positive|strong|robust|accelerat)',
    ],
    "trend_down": [
        r'(?:bearish|pessimistic|negative|weak|slow(?:ing)?|decelerat)',
    ],

    # Entity patterns (will be filled dynamically)
    "entities": []
}

# Common stop words to remove
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'also', 'now', 'new'
}


def extract_semantic_components(finding: str) -> Dict[str, Any]:
    """
    Extract semantic components from a finding.

    Example:
        "The market is growing at 15% annually" ->
        {
            "direction": "up",
            "percentage": 15.0,
            "subject": "market",
            "timeframe": "annual",
            "entities": [],
            "keywords": ["market", "growing", "annually"]
        }
    """
    if not finding:
        return {}

    finding_lower = finding.lower()
    components = {
        "direction": None,
        "percentage": None,
        "value": None,
        "value_unit": None,
        "subject": None,
        "timeframe": None,
        "entities": [],
        "keywords": []
    }

    # Extract direction
    for pattern in FINDING_PATTERNS["growth"]:
        match = re.search(pattern, finding_lower)
        if match:
            components["direction"] = "up"
            if match.groups():
                try:
                    components["percentage"] = float(match.group(1))
                except:
                    pass
            break

    if not components["direction"]:
        for pattern in FINDING_PATTERNS["decline"]:
            match = re.search(pattern, finding_lower)
            if match:
                components["direction"] = "down"
                if match.groups():
                    try:
                        components["percentage"] = float(match.group(1))
                    except:
                        pass
                break

    # Extract trend sentiment
    if not components["direction"]:
        for pattern in FINDING_PATTERNS["trend_up"]:
            if re.search(pattern, finding_lower):
                components["direction"] = "up"
                break
        for pattern in FINDING_PATTERNS["trend_down"]:
            if re.search(pattern, finding_lower):
                components["direction"] = "down"
                break

    # Extract value
    for pattern in FINDING_PATTERNS["value"]:
        match = re.search(pattern, finding_lower)
        if match:
            try:
                components["value"] = float(match.group(1))
                if len(match.groups()) > 1 and match.group(2):
                    components["value_unit"] = match.group(2)[0].upper()
            except:
                pass
            break

    # Extract timeframe
    timeframe_patterns = {
        "annual": r'\b(?:annual(?:ly)?|year(?:ly)?|per year|yoy|y-o-y)\b',
        "quarterly": r'\b(?:quarter(?:ly)?|q[1-4])\b',
        "monthly": r'\b(?:month(?:ly)?|per month)\b',
        "2024": r'\b2024\b',
        "2025": r'\b2025\b',
        "2026": r'\b2026\b',
        "2030": r'\b2030\b',
    }
    for tf_name, tf_pattern in timeframe_patterns.items():
        if re.search(tf_pattern, finding_lower):
            components["timeframe"] = tf_name
            break

    # Extract subject keywords
    words = re.findall(r'\b[a-z]{3,}\b', finding_lower)
    keywords = [w for w in words if w not in STOP_WORDS]
    components["keywords"] = keywords[:10]  # Limit to top 10

    # Identify likely subject
    subject_candidates = ["market", "industry", "sector", "segment", "revenue", "sales", "demand", "supply"]
    for word in keywords:
        if word in subject_candidates:
            components["subject"] = word
            break

    return components


def compute_semantic_hash(finding: str) -> str:
    """
    Compute a semantic hash for a finding that's invariant to wording changes.

    Two findings with the same meaning should produce the same hash.

    Example:
        "The market is growing at 15% annually" -> "up_15.0_market_annual"
        "Annual growth rate stands at 15%" -> "up_15.0_market_annual"
    """
    components = extract_semantic_components(finding)

    # Build hash components in consistent order
    hash_parts = []

    # Direction
    if components.get("direction"):
        hash_parts.append(components["direction"])

    # Percentage (rounded to avoid float issues)
    if components.get("percentage") is not None:
        hash_parts.append(f"{components['percentage']:.1f}")

    # Value with unit
    if components.get("value") is not None:
        val_str = f"{components['value']:.1f}"
        if components.get("value_unit"):
            val_str += components["value_unit"]
        hash_parts.append(val_str)

    # Subject
    if components.get("subject"):
        hash_parts.append(components["subject"])

    # Timeframe
    if components.get("timeframe"):
        hash_parts.append(components["timeframe"])

    # If we have enough components, use them for hash
    if len(hash_parts) >= 2:
        return "_".join(hash_parts)

    # Fallback: use sorted keywords
    keywords = sorted(components.get("keywords", []))[:5]
    if keywords:
        return "_".join(keywords)

    # Last resort: normalized text hash
    normalized = re.sub(r'\s+', ' ', finding.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def compute_semantic_finding_diffs(old_findings: List[str], new_findings: List[str]) -> List[FindingDiff]:
    """
    Compute finding diffs using semantic hashing instead of text similarity.

    This ensures that findings with the same meaning but different wording
    are recognized as the same finding.
    """
    diffs = []
    matched_new_indices = set()

    # Compute hashes for all findings
    old_hashes = [(f, compute_semantic_hash(f), extract_semantic_components(f)) for f in old_findings if f]
    new_hashes = [(f, compute_semantic_hash(f), extract_semantic_components(f)) for f in new_findings if f]

    # Match by semantic hash
    for old_text, old_hash, old_components in old_hashes:
        best_match_idx = None
        best_match_score = 0

        for i, (new_text, new_hash, new_components) in enumerate(new_hashes):
            if i in matched_new_indices:
                continue

            # Exact hash match = same finding
            if old_hash == new_hash:
                best_match_idx = i
                best_match_score = 100
                break

            # Component-based similarity
            score = compute_component_similarity(old_components, new_components)
            if score > best_match_score:
                best_match_score = score
                best_match_idx = i

        if best_match_idx is not None and best_match_score >= 60:
            matched_new_indices.add(best_match_idx)
            new_text = new_hashes[best_match_idx][0]

            if best_match_score >= 90:
                change_type = 'retained'
            else:
                change_type = 'modified'

            diffs.append(FindingDiff(
                old_text=old_text,
                new_text=new_text,
                similarity=best_match_score,
                change_type=change_type
            ))
        else:
            # Finding removed
            diffs.append(FindingDiff(
                old_text=old_text,
                new_text=None,
                similarity=0,
                change_type='removed'
            ))

    # Find added findings
    for i, (new_text, new_hash, new_components) in enumerate(new_hashes):
        if i not in matched_new_indices:
            diffs.append(FindingDiff(
                old_text=None,
                new_text=new_text,
                similarity=0,
                change_type='added'
            ))

    return diffs


def compute_component_similarity(comp1: Dict, comp2: Dict) -> float:
    """
    Compute similarity between two finding component dictionaries.
    Returns a score from 0-100.
    """
    if not comp1 or not comp2:
        return 0

    score = 0
    weights = {
        "direction": 25,
        "percentage": 25,
        "value": 20,
        "subject": 15,
        "timeframe": 10,
        "keywords": 5
    }

    # Direction match
    if comp1.get("direction") and comp2.get("direction"):
        if comp1["direction"] == comp2["direction"]:
            score += weights["direction"]
    elif not comp1.get("direction") and not comp2.get("direction"):
        score += weights["direction"] * 0.5  # Both neutral

    # Percentage match (within 2% tolerance)
    if comp1.get("percentage") is not None and comp2.get("percentage") is not None:
        diff = abs(comp1["percentage"] - comp2["percentage"])
        if diff <= 2:
            score += weights["percentage"]
        elif diff <= 5:
            score += weights["percentage"] * 0.5

    # Value match (within 10% tolerance)
    if comp1.get("value") is not None and comp2.get("value") is not None:
        v1, v2 = comp1["value"], comp2["value"]
        # Normalize by unit
        if comp1.get("value_unit") == comp2.get("value_unit"):
            if v1 > 0 and v2 > 0:
                ratio = min(v1, v2) / max(v1, v2)
                if ratio >= 0.9:
                    score += weights["value"]
                elif ratio >= 0.8:
                    score += weights["value"] * 0.5

    # Subject match
    if comp1.get("subject") and comp2.get("subject"):
        if comp1["subject"] == comp2["subject"]:
            score += weights["subject"]

    # Timeframe match
    if comp1.get("timeframe") and comp2.get("timeframe"):
        if comp1["timeframe"] == comp2["timeframe"]:
            score += weights["timeframe"]

    # Keyword overlap
    kw1 = set(comp1.get("keywords", []))
    kw2 = set(comp2.get("keywords", []))
    if kw1 and kw2:
        overlap = len(kw1 & kw2) / len(kw1 | kw2)
        score += weights["keywords"] * overlap

    return score


# ------------------------------------
# UPDATED METRIC DIFF COMPUTATION
# Using canonical IDs
# ------------------------------------

def compute_metric_diffs_canonical(old_metrics: Dict, new_metrics: Dict) -> List[MetricDiff]:
    """
    Compute metric diffs using canonical IDs for stable matching.
    Range-aware via get_metric_value_span + spans_overlap.
    """
    diffs: List[MetricDiff] = []

    old_canonical = canonicalize_metrics(old_metrics)
    new_canonical = canonicalize_metrics(new_metrics)

    matched_new_ids = set()

    # Match by canonical ID
    for old_id, old_m in old_canonical.items():
        old_name = old_m.get("name", old_id)

        old_span = get_metric_value_span(old_m)
        old_raw = str(old_m.get("value", ""))
        old_unit = old_span.get("unit") or old_m.get("unit", "")
        old_val = old_span.get("mid")

        # -------------------------
        # Direct canonical ID match
        # -------------------------
        if old_id in new_canonical:
            new_m = new_canonical[old_id]
            matched_new_ids.add(old_id)

            new_raw = str(new_m.get("value", ""))
            new_span = get_metric_value_span(new_m)
            new_val = new_span.get("mid")
            new_unit = new_span.get("unit") or new_m.get("unit", old_unit)

            if spans_overlap(old_span, new_span, rel_tol=0.05):
                change_pct = 0.0
                change_type = "unchanged"
            else:
                change_pct = compute_percent_change(old_val, new_val)
                if change_pct is None or abs(change_pct) < 0.5:
                    change_type = "unchanged"
                elif change_pct > 0:
                    change_type = "increased"
                else:
                    change_type = "decreased"

            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=new_val,
                old_raw=old_raw,
                new_raw=new_raw,
                unit=new_unit or old_unit,
                change_pct=change_pct,
                change_type=change_type
            ))
            continue  # important: don't fall into base-ID matching

        # -------------------------
        # Base ID match (strip years)
        # -------------------------
        base_id = re.sub(r'_\d{4}(?:_\d{4})*$', '', old_id)
        found = False

        for new_id, new_m in new_canonical.items():
            if new_id in matched_new_ids:
                continue

            new_base_id = re.sub(r'_\d{4}(?:_\d{4})*$', '', new_id)
            if base_id != new_base_id:
                continue

            matched_new_ids.add(new_id)
            found = True

            new_raw = str(new_m.get("value", ""))
            new_span = get_metric_value_span(new_m)
            new_val = new_span.get("mid")
            new_unit = new_span.get("unit") or new_m.get("unit", old_unit)

            if spans_overlap(old_span, new_span, rel_tol=0.05):
                change_pct = 0.0
                change_type = "unchanged"
            else:
                change_pct = compute_percent_change(old_val, new_val)
                if change_pct is None or abs(change_pct) < 0.5:
                    change_type = "unchanged"
                elif change_pct > 0:
                    change_type = "increased"
                else:
                    change_type = "decreased"

            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=new_val,
                old_raw=old_raw,
                new_raw=new_raw,
                unit=new_unit or old_unit,
                change_pct=change_pct,
                change_type=change_type
            ))
            break

        if not found:
            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=None,
                old_raw=old_raw,
                new_raw="",
                unit=old_unit,
                change_pct=None,
                change_type="removed"
            ))

    # Added metrics
    for new_id, new_m in new_canonical.items():
        if new_id in matched_new_ids:
            continue

        new_name = new_m.get("name", new_id)
        new_raw = str(new_m.get("value", ""))
        new_span = get_metric_value_span(new_m)
        new_val = new_span.get("mid")
        new_unit = new_span.get("unit") or new_m.get("unit", "")

        diffs.append(MetricDiff(
            name=new_name,
            old_value=None,
            new_value=new_val,
            old_raw="",
            new_raw=new_raw,
            unit=new_unit,
            change_pct=None,
            change_type="added"
        ))

    return diffs


# ------------------------------------
# NUMERIC PARSING (DETERMINISTIC)
# ------------------------------------

def parse_to_float(value: Any) -> Optional[float]:
    """
    Deterministically parse any value to float.
    Returns None if unparseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    # Clean string
    cleaned = value.strip().upper()
    cleaned = re.sub(r'[,$]', '', cleaned)

    # Handle empty/NA
    if cleaned in ['', 'N/A', 'NA', 'NULL', 'NONE', '-', '—']:
        return None

    # Extract multiplier
    multiplier = 1.0
    if 'TRILLION' in cleaned or cleaned.endswith('T'):
        multiplier = 1_000_000
        cleaned = re.sub(r'T(?:RILLION)?', '', cleaned)
    elif 'BILLION' in cleaned or cleaned.endswith('B'):
        multiplier = 1_000
        cleaned = re.sub(r'B(?:ILLION)?', '', cleaned)
    elif 'MILLION' in cleaned or cleaned.endswith('M'):
        multiplier = 1
        cleaned = re.sub(r'M(?:ILLION)?', '', cleaned)
    elif 'THOUSAND' in cleaned or cleaned.endswith('K'):
        multiplier = 0.001
        cleaned = re.sub(r'K(?:THOUSAND)?', '', cleaned)

    # Handle percentages
    if '%' in cleaned:
        cleaned = cleaned.replace('%', '')
        # Don't apply multiplier to percentages
        multiplier = 1.0

    try:
        return float(cleaned.strip()) * multiplier
    except (ValueError, TypeError):
        return None

def get_metric_value_span(metric: Dict) -> Dict[str, Any]:
    """
    Return a numeric span for a metric to support range-aware canonical metrics.

    Output:
      {
        "min": float|None,
        "max": float|None,
        "mid": float|None,   # representative value (median if range, else parsed value)
        "unit": str,         # normalized (upper/stripped), preserves %/$ units if present
        "is_range": bool
      }
    """
    if not isinstance(metric, dict):
        return {"min": None, "max": None, "mid": None, "unit": "", "is_range": False}

    unit = (metric.get("unit") or "").strip()

    # If metric already has a range object, prefer it
    r = metric.get("range")
    if isinstance(r, dict):
        vmin = r.get("min")
        vmax = r.get("max")
        # ensure numeric
        try:
            vmin = float(vmin) if vmin is not None else None
        except Exception:
            vmin = None
        try:
            vmax = float(vmax) if vmax is not None else None
        except Exception:
            vmax = None

        # Representative = median of candidates if provided, else midpoint of min/max
        candidates = r.get("candidates")
        nums = []
        if isinstance(candidates, list):
            for c in candidates:
                try:
                    nums.append(float(c))
                except Exception:
                    pass
        if nums:
            nums_sorted = sorted(nums)
            mid = nums_sorted[len(nums_sorted) // 2]
        else:
            mid = None
            if vmin is not None and vmax is not None:
                mid = (vmin + vmax) / 2.0
            elif vmin is not None:
                mid = vmin
            elif vmax is not None:
                mid = vmax

        return {
            "min": vmin,
            "max": vmax,
            "mid": mid,
            "unit": unit,
            "is_range": True
        }

    # Non-range metric: parse single numeric value
    val = parse_to_float(metric.get("value"))
    return {
        "min": val,
        "max": val,
        "mid": val,
        "unit": unit,
        "is_range": False
    }


def spans_overlap(a: Dict[str, Any], b: Dict[str, Any], rel_tol: float = 0.05) -> bool:
    """
    Decide whether two spans overlap "enough" to be considered stable.
    rel_tol provides a small widening to avoid false drift from rounding.
    """
    if not a or not b:
        return False
    a_min, a_max = a.get("min"), a.get("max")
    b_min, b_max = b.get("min"), b.get("max")

    if a_min is None or a_max is None or b_min is None or b_max is None:
        return False

    # Widen spans slightly
    a_pad = max(abs(a_max), abs(a_min), 1.0) * rel_tol
    b_pad = max(abs(b_max), abs(b_min), 1.0) * rel_tol

    a_min2, a_max2 = a_min - a_pad, a_max + a_pad
    b_min2, b_max2 = b_min - b_pad, b_max + b_pad

    return not (a_max2 < b_min2 or b_max2 < a_min2)


def compute_percent_change(old_val: Optional[float], new_val: Optional[float]) -> Optional[float]:
    """
    Compute percent change. Returns None if either value is None or old is 0.
    """
    if old_val is None or new_val is None:
        return None
    if old_val == 0:
        return None if new_val == 0 else float('inf')
    return round(((new_val - old_val) / abs(old_val)) * 100, 2)

# ------------------------------------
# NAME MATCHING (DETERMINISTIC)
# ------------------------------------

def normalize_name(name: str) -> str:
    """Normalize name for matching"""
    if not name:
        return ""
    n = name.lower().strip()
    n = re.sub(r'[^\w\s]', '', n)
    n = re.sub(r'\s+', ' ', n)
    return n

def name_similarity(name1: str, name2: str) -> float:
    """Compute similarity ratio between two names (0-1)"""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    # Check containment
    if n1 in n2 or n2 in n1:
        return 0.9
    # Sequence matcher
    return difflib.SequenceMatcher(None, n1, n2).ratio()

def find_best_match(name: str, candidates: List[str], threshold: float = 0.7) -> Optional[str]:
    """Find best matching name from candidates"""
    best_match = None
    best_score = threshold
    for candidate in candidates:
        score = name_similarity(name, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate
    return best_match

# =========================================================
# DETERMINISTIC QUERY STRUCTURE EXTRACTION
# - Classify query into a known category (country / industry / etc.)
# - Extract main question + "side questions" deterministically
# - Optional: spaCy dependency parse (if installed)
# - Optional: embedding similarity (if sentence-transformers/sklearn installed)
# =========================================================

SIDE_CONNECTOR_PATTERNS = [
    r"\bimpact of\b",
    r"\beffect of\b",
    r"\binfluence of\b",
    r"\brole of\b",
    r"\bdriven by\b",
    r"\bcaused by\b",
    r"\bdue to\b",
    r"\bincluding\b",
    r"\bincluding but not limited to\b",
    r"\bwith a focus on\b",
    r"\bespecially\b",
    r"\bnotably\b",
    r"\bplus\b",
    r"\bas well as\b",
    r"\band also\b",
    r"\bvs\b",
    r"\bversus\b",
]

QUESTION_CATEGORIES = {
    "country": {
        "signals": [
            "gdp", "gdp per capita", "population", "inflation", "interest rate",
            "exports", "imports", "trade balance", "currency", "fx", "central bank",
            "unemployment", "fiscal", "budget", "debt", "sovereign", "country"
        ],
        "template_sections": [
            "GDP & GDP per capita", "Growth rates", "Population & demographics",
            "Key industries", "Exports & imports", "Currency & FX trends",
            "Interest rates & inflation", "Risks & outlook"
        ],
    },
    "industry": {
        "signals": [
            "market size", "tam", "cagr", "industry", "sector", "market",
            "key players", "competitive landscape", "drivers", "challenges",
            "regulation", "technology trends", "forecast"
        ],
        "template_sections": [
            "Total Addressable Market (TAM) / Market size", "Growth rates (CAGR/YoY)",
            "Key players", "Key drivers", "Challenges & risks",
            "Technology trends", "Regulatory / environmental factors", "Outlook"
        ],
    },
    "company": {
        "signals": [
            "revenue", "earnings", "profit", "margins", "guidance",
            "business model", "segments", "customers", "competitors",
            "valuation", "multiple", "pe ratio", "cash flow"
        ],
        "template_sections": [
            "Business overview", "Revenue / profitability", "Segments",
            "Competitive position", "Key risks", "Guidance / outlook"
        ],
    },
    "unknown": {
        "signals": [],
        "template_sections": [],
    }
}

def _normalize_q(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def _cleanup_clause(text: str) -> str:
    t = _normalize_q(text)
    t = re.sub(r"^[,;:\-\s]+", "", t)
    t = re.sub(r"[,;:\-\s]+$", "", t)
    return t

def detect_query_category(query: str) -> Dict[str, Any]:
    """
    Deterministically classify query category using keyword signals.
    Returns: {"category": "...", "confidence": 0-1, "matched_signals": [...]}
    """
    q = (query or "").lower()
    best_cat = "unknown"
    best_hits = 0
    best_matched = []

    for cat, cfg in QUESTION_CATEGORIES.items():
        if cat == "unknown":
            continue
        matched = [s for s in cfg["signals"] if s in q]
        hits = len(matched)
        if hits > best_hits:
            best_hits = hits
            best_cat = cat
            best_matched = matched

    # simple confidence: saturate after ~6 hits
    conf = min(best_hits / 6.0, 1.0) if best_hits > 0 else 0.0
    return {"category": best_cat, "confidence": round(conf, 2), "matched_signals": best_matched[:8]}

# =========================================================
# 3A+. LAYERED QUERY STRUCTURE PARSER (Deterministic -> NLP -> Embeddings -> LLM fallback)
# =========================================================

_QUERY_SPLIT_PATTERNS = [
    r"\bas well as\b",
    r"\balong with\b",
    r"\bin addition to\b",
    r"\band\b",
    r"\bplus\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r",",
    r";",
]

_COUNTRY_OVERVIEW_SIGNALS = [
    "in general", "overview", "tell me about", "general", "profile", "facts about",
    "economy", "population", "gdp", "currency", "exports", "imports",
]

def _normalize_q(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q

def _split_clauses_deterministic(q: str) -> List[str]:
    """
    Deterministically split a question into ordered clauses.

    Supports:
    - comma/connector splits (",", "and", "as well as", "in addition to", etc.)
    - multi-side enumerations like:
        "in addition to: 1. X 2. Y"
        "including: (1) X (2) Y"
        "as well as: • X • Y"
    """
    if not isinstance(q, str):
        return []

    s = q.strip()
    if not s:
        return []

    # Normalize whitespace early (keep original casing if present; upstream may lowercase already)
    s = re.sub(r"\s+", " ", s).strip()

    # --- Step A: If there's an enumeration intro, split head vs tail ---
    # Examples: "in addition to:", "including:", "plus:", "as well as:"
    enum_intro = re.search(
        r"\b(in addition to|in addition|including|in addition to the following|as well as|plus)\b\s*:?\s*",
        s,
        flags=re.IGNORECASE,
    )

    head = s
    tail = ""

    if enum_intro:
        # Split at the FIRST occurrence of the enum phrase
        idx = enum_intro.start()
        # head is everything before the phrase if it exists, otherwise keep whole string
        # but we usually want "Tell me about X in general" to remain in head.
        head = s[:idx].strip().rstrip(",")
        tail = s[enum_intro.end():].strip()

        # If head is empty (e.g., query begins with "In addition to:"), treat everything as head
        if not head:
            head = s
            tail = ""

    clauses: List[str] = []

    # --- Step B: Split head using your existing connector patterns ---
    if head:
        parts = [head]
        for pat in _QUERY_SPLIT_PATTERNS:
            next_parts = []
            for p in parts:
                next_parts.extend(re.split(pat, p, flags=re.IGNORECASE))
            parts = next_parts

        for p in parts:
            p = p.strip(" ,;:.").strip()
            if p:
                clauses.append(p)

    # --- Step C: If tail exists, split as enumerated items/bullets ---
    if tail:
        # Split on "1.", "1)", "(1)", "•", "-", "*"
        # Keep it robust: find item starts, then slice.
        item_start = re.compile(r"(?:^|\s)(?:\(?\d+\)?[\.\)]|[•\-\*])\s+", flags=re.IGNORECASE)
        starts = [m.start() for m in item_start.finditer(tail)]

        if starts:
            # Build slices using detected starts
            spans = []
            for i, st0 in enumerate(starts):
                st = st0
                # Move start to the start of token (strip leading whitespace)
                while st < len(tail) and tail[st].isspace():
                    st += 1
                en = starts[i + 1] if i + 1 < len(starts) else len(tail)
                spans.append((st, en))

            for st, en in spans:
                item = tail[st:en].strip(" ,;:.").strip()
                # Remove the leading bullet/number token again (safety)
                item = re.sub(r"^\(?\d+\)?[\.\)]\s+", "", item)
                item = re.sub(r"^[•\-\*]\s+", "", item)
                item = item.strip(" ,;:.").strip()
                if item:
                    clauses.append(item)
        else:
            # If tail doesn't look enumerated, fall back to normal splitter on tail
            parts = [tail]
            for pat in _QUERY_SPLIT_PATTERNS:
                next_parts = []
                for p in parts:
                    next_parts.extend(re.split(pat, p, flags=re.IGNORECASE))
                parts = next_parts

            for p in parts:
                p = p.strip(" ,;:.").strip()
                if p:
                    clauses.append(p)

    # Final cleanup + dedupe while preserving order
    out: List[str] = []
    seen = set()
    for c in clauses:
        c2 = c.strip()
        if not c2:
            continue
        if c2.lower() in seen:
            continue
        seen.add(c2.lower())
        out.append(c2)

    return out



def _dedupe_clauses(clauses: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in clauses:
        c2 = c.strip().lower()
        if not c2 or c2 in seen:
            continue
        seen.add(c2)
        out.append(c.strip())
    return out

def _choose_main_and_side(clauses: List[str]) -> Tuple[str, List[str]]:
    """
    Pick 'main' as the first clause; side = remainder.
    Deterministic, stable across runs.
    """
    clauses = _dedupe_clauses(clauses)
    if not clauses:
        return "", []
    main = clauses[0]
    side = clauses[1:]
    return main, side

def _try_spacy_nlp():
    """
    Optional NLP layer. If spaCy is installed, use it; otherwise return None.
    """
    try:
        import spacy  # type: ignore
        # Avoid heavy model loading; prefer blank model with sentencizer if no model available.
        try:
            nlp = spacy.load("en_core_web_sm")  # common if installed
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        return None

def _nlp_refine_clauses(query: str, clauses: List[str]) -> Dict[str, Any]:
    """
    Use dependency/NER cues to:
      - detect country-overview questions
      - improve main-vs-side decision (coordination / 'as well as' patterns)
    Returns partial overrides: {"main":..., "side":[...], "hints":{...}}
    """
    nlp = _try_spacy_nlp()
    if not nlp:
        return {"hints": {"nlp_used": False}}

    doc = nlp(_normalize_q(query))
    # Named entities that look like places
    gpes = [ent.text for ent in getattr(doc, "ents", []) if ent.label_ in ("GPE", "LOC")]
    gpes_norm = [g.strip() for g in gpes if g and len(g.strip()) > 1]

    # Coordination hint: if query has "as well as" or "and", keep deterministic split,
    # but try to pick the more "general" clause as main when overview signals exist.
    overview_hit = any(sig in (query or "").lower() for sig in _COUNTRY_OVERVIEW_SIGNALS)
    hints = {
        "nlp_used": True,
        "gpe_entities": gpes_norm[:5],
        "overview_signal_hit": bool(overview_hit),
    }

    main, side = _choose_main_and_side(clauses)

    # If overview signals + place entity present, bias main to the overview clause
    if overview_hit and gpes_norm:
        # choose clause with strongest overview signal density
        def score_clause(c: str) -> int:
            c = c.lower()
            return sum(1 for sig in _COUNTRY_OVERVIEW_SIGNALS if sig in c)
        scored = sorted([(score_clause(c), c) for c in clauses], key=lambda x: (-x[0], x[1]))
        if scored and scored[0][0] > 0:
            main = scored[0][1]
            side = [c for c in clauses if c != main]

    return {"main": main, "side": side, "hints": hints}

def _embedding_category_vote(query: str) -> Dict[str, Any]:
    """
    Deterministic 'embedding-like' similarity using TF-IDF (no external model downloads).
    Produces a category suggestion + confidence based on similarity to category descriptors.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    except Exception:
        return {"category": "unknown", "confidence": 0.0, "method": "tfidf_unavailable"}

    q = _normalize_q(query).lower()
    if not q:
        return {"category": "unknown", "confidence": 0.0, "method": "tfidf_empty"}

    # Build deterministic descriptors from your registry
    cat_texts = []
    cat_names = []
    for cat, cfg in (QUESTION_CATEGORIES or {}).items():
        if not isinstance(cfg, dict) or cat == "unknown":
            continue
        signals = " ".join(cfg.get("signals", [])[:50])
        sections = " ".join((cfg.get("template_sections", []) or [])[:50])
        descriptor = f"{cat} {signals} {sections}".strip()
        if descriptor:
            cat_names.append(cat)
            cat_texts.append(descriptor)

    if not cat_texts:
        return {"category": "unknown", "confidence": 0.0, "method": "tfidf_no_registry"}

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)
    X = vec.fit_transform(cat_texts + [q])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()

    best_idx = int(sims.argmax()) if sims.size else 0
    best_sim = float(sims[best_idx]) if sims.size else 0.0
    best_cat = cat_names[best_idx] if cat_names else "unknown"

    # Map cosine similarity (~0-1) into a conservative confidence
    conf = max(0.0, min(best_sim / 0.35, 1.0))  # 0.35 sim ~= "high"
    return {"category": best_cat, "confidence": round(conf, 2), "method": "tfidf"}

def _llm_fallback_query_structure(query: str, web_context: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Last resort: ask LLM to output ONLY a small JSON query-structure object.
    Guardrail: do NOT let the LLM invent extra side questions unless the user explicitly enumerated them.
    This path must NOT validate against LLMResponse.
    """
    try:
        q = str(query or "").strip()
        if not q:
            return None

        # --- Detect explicit enumeration / list structure in the USER query ---
        # If the user wrote a list (1., 2), bullets, etc.), it's reasonable to accept multiple sides.
        enum_patterns = [
            r"(^|\n)\s*\d+\s*[\.\)]\s+",     # 1.  / 2)
            r"(^|\n)\s*[-•*]\s+",           # - item / • item
            r"(^|\n)\s*[a-zA-Z]\s*[\.\)]\s+"  # a) / b. etc.
        ]
        has_explicit_enumeration = any(re.search(p, q, flags=re.MULTILINE) for p in enum_patterns)

        # Deterministic baseline (what the system already extracted)
        # We use this to clamp LLM hallucinations.
        det_clauses = _split_clauses_deterministic(_normalize_q(q))
        det_main, det_side = _choose_main_and_side(det_clauses)
        det_side = _dedupe_clauses([s.strip() for s in (det_side or []) if isinstance(s, str) and s.strip()])

        prompt = (
            "Extract a query structure.\n"
            "Return ONLY valid JSON with keys:\n"
            "  category: one of [country, industry, company, finance, market, unknown]\n"
            "  category_confidence: number 0-1\n"
            "  main: string (the main question/topic)\n"
            "  side: array of strings (side questions)\n"
            "No extra keys, no commentary.\n\n"
            f"Query: {q}"
        )

        raw = query_perplexity_raw(prompt, max_tokens=250, timeout=30)

        # Parse
        if isinstance(raw, dict):
            parsed = raw
        else:
            if raw is None:
                raw = ""
            if not isinstance(raw, str):
                raw = str(raw)
            parsed = parse_json_safely(raw, "LLM Query Structure")

        if not isinstance(parsed, dict) or parsed.get("main") is None:
            return None

        # --- Clean/normalize fields ---
        llm_main = str(parsed.get("main") or "").strip()
        llm_side = parsed.get("side") if isinstance(parsed.get("side"), list) else []
        llm_side = [str(s).strip() for s in llm_side if s is not None and str(s).strip()]

        # Reject "invented" side items that look like generic outline bullets
        # (Only apply this rejection when the user did NOT explicitly enumerate a list.)
        def _looks_like_outline_item(s: str) -> bool:
            s2 = s.lower().strip()
            bad_starts = (
                "overview", "key", "key stats", "statistics", "major statistics",
                "policies", "infrastructure", "recent trends", "post-covid", "covid",
                "challenges", "opportunities", "drivers", "headwinds",
                "background", "introduction"
            )
            return any(s2.startswith(b) for b in bad_starts)

        # --- Guardrail policy ---
        # If user didn't enumerate, do NOT accept LLM expansion of side questions.
        if not has_explicit_enumeration:
            # Keep deterministic sides only. (You can allow 1 LLM side if deterministic found none.)
            final_side = det_side
            if not final_side and llm_side:
                # Allow at most one side item as a fallback, but avoid outline-like additions.
                cand = llm_side[0]
                final_side = [] if _looks_like_outline_item(cand) else [cand]
        else:
            # User enumerated: accept multiple sides, but still de-dupe and keep deterministic items first
            merged = []
            for s in (det_side + llm_side):
                s = str(s).strip()
                if not s:
                    continue
                if s not in merged:
                    merged.append(s)
            final_side = merged

        # If LLM main is empty or fragment-y, keep deterministic main
        bad_prefixes = ("as well as", "as well", "and ", "also ", "plus ", "as for ")
        if not llm_main or any(llm_main.lower().startswith(p) for p in bad_prefixes):
            llm_main = (det_main or "").strip()

        # Return only allowed keys
        out = {
            "category": parsed.get("category", "unknown") or "unknown",
            "category_confidence": parsed.get("category_confidence", 0.0),
            "main": llm_main,
            "side": final_side,
        }
        return out

    except Exception:
        return None


def _split_side_candidates(query: str) -> List[str]:
    """
    Deterministic splitting into clause candidates.
    We keep it conservative to avoid over-splitting.
    """
    q = _normalize_q(query)
    # Pull quoted strings as strong side-topic candidates
    quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", q)
    q_wo_quotes = re.sub(r"['\"][^'\"]{2,80}['\"]", " ", q)

    # Split on major separators
    parts = re.split(r"[;]|(?:\s+-\s+)|(?:\s+—\s+)", q_wo_quotes)
    parts = [p for p in (_cleanup_clause(x) for x in parts) if p]

    # Further split on side connectors
    connector_re = "(" + "|".join(SIDE_CONNECTOR_PATTERNS) + ")"
    expanded = []
    for p in parts:
        # break into chunks but keep connector words in-place by splitting into sentences first
        sub = re.split(r"\.\s+|\?\s+|\!\s+", p)
        for s in sub:
            s = _cleanup_clause(s)
            if not s:
                continue
            # if contains connector, split into [before, after...] using first connector
            m = re.search(connector_re, s.lower())
            if m:
                idx = m.start()
                before = _cleanup_clause(s[:idx])
                after = _cleanup_clause(s[idx:])
                if before:
                    expanded.append(before)
                if after:
                    expanded.append(after)
            else:
                expanded.append(s)

    # Add quoted items as standalone candidates (often side topics)
    for qitem in quoted:
        cleaned = _cleanup_clause(qitem)
        if cleaned:
            expanded.append(cleaned)

    # De-dupe while preserving order (deterministic)
    seen = set()
    out = []
    for x in expanded:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _extract_spacy_side_topics(query: str) -> List[str]:
    """
    Optional: use spaCy dependency parse if available.
    Extracts objects of 'impact/effect/role/influence' patterns.
    """
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load("en_core_web_sm")  # type: ignore
        except Exception:
            return []
    except Exception:
        return []

    doc = nlp(query)
    side = []

    triggers = {"impact", "effect", "influence", "role"}
    for token in doc:
        if token.lemma_.lower() in triggers:
            # Look for "of X" attached to trigger
            for child in token.children:
                if child.dep_ == "prep" and child.text.lower() == "of":
                    pobj = next((c for c in child.children if c.dep_ in ("pobj", "dobj", "obj")), None)
                    if pobj is not None:
                        # take subtree as phrase
                        phrase = " ".join(t.text for t in pobj.subtree)
                        phrase = _cleanup_clause(phrase)
                        if phrase and phrase.lower() not in (s.lower() for s in side):
                            side.append(phrase)
    return side[:5]

def _embedding_similarity(a: str, b: str) -> Optional[float]:
    """
    Optional: compute cosine similarity using:
      - sentence-transformers (preferred) OR
      - sklearn TF-IDF fallback
    Returns None if unavailable.
    """
    a = _normalize_q(a)
    b = _normalize_q(b)
    if not a or not b:
        return None

    # 1) sentence-transformers (if installed)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([a, b], normalize_embeddings=True)
        sim = float(np.dot(emb[0], emb[1]))
        return max(min(sim, 1.0), -1.0)
    except Exception:
        pass

    # 2) sklearn TF-IDF cosine similarity (deterministic)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform([a, b])
        sim = float(cosine_similarity(X[0], X[1])[0, 0])
        return max(min(sim, 1.0), -1.0)
    except Exception:
        return None

def extract_query_structure(query: str) -> Dict[str, Any]:
    """
    Layered query structure extraction:
      1) Deterministic clause split -> main/side
      2) Deterministic category from keyword signals (detect_query_category)
      3) Optional NLP refinement (spaCy if available)
      4) Deterministic similarity vote (TF-IDF)
      5) LLM fallback ONLY if confidence remains low
    """
    q = _normalize_q(query)
    clauses = _split_clauses_deterministic(q)
    main, side = _choose_main_and_side(clauses)

    # --- Layer 1: deterministic keyword category ---
    det_cat = detect_query_category(q)
    category = det_cat.get("category", "unknown")
    cat_conf = float(det_cat.get("confidence", 0.0))

    debug = {
        "deterministic": {
            "clauses": clauses,
            "main": main,
            "side": side,
            "category": category,
            "confidence": cat_conf,
            "matched_signals": det_cat.get("matched_signals", []),
        }
    }

    # --- Layer 2: NLP refinement (optional) ---
    nlp_out = _nlp_refine_clauses(q, clauses)
    if isinstance(nlp_out, dict):
        hints = nlp_out.get("hints", {})
        debug["nlp"] = hints or {"nlp_used": False}

        # Override main/side if NLP produced them (guard against fragment-y mains)
        nlp_main = (nlp_out.get("main") or "").strip()
        if nlp_main:
            bad_prefixes = ("as well as", "as well", "and ", "also ", "plus ", "as for ")
            if not any(nlp_main.lower().startswith(p) for p in bad_prefixes):
                main = nlp_main

        if isinstance(nlp_out.get("side"), list):
            side = nlp_out["side"]

        # If NLP detects a place + overview cue, bias to "country"
        gpes = (hints or {}).get("gpe_entities", []) if isinstance(hints, dict) else []
        overview_hit = (hints or {}).get("overview_signal_hit", False) if isinstance(hints, dict) else False
        if overview_hit and gpes and cat_conf < 0.45:
            category = "country"
            cat_conf = max(cat_conf, 0.55)

    # --- Layer 3: embedding-style category vote ---
    emb_vote = _embedding_category_vote(q)
    debug["similarity_vote"] = emb_vote

    emb_cat = emb_vote.get("category", "unknown")
    emb_conf = float(emb_vote.get("confidence", 0.0))

    if cat_conf < 0.40 and emb_cat != "unknown" and emb_conf >= 0.45:
        category = emb_cat
        cat_conf = max(cat_conf, min(0.75, emb_conf))

    # --- Layer 4: LLM fallback if still ambiguous ---
    if cat_conf < 0.30:
        llm = _llm_fallback_query_structure(q)
        debug["llm_fallback_used"] = bool(llm)

        if isinstance(llm, dict):
            category = llm.get("category", category) or category
            try:
                cat_conf = float(llm.get("category_confidence", cat_conf))
            except Exception:
                pass

            llm_main = (llm.get("main") or "").strip()
            llm_side = llm.get("side") if isinstance(llm.get("side"), list) else []

            det_main = (main or "").strip()
            det_side = side or []

            def _overview_score(s: str) -> int:
                if not s:
                    return 0
                s2 = s.lower()
                signals = [
                    "in general", "overview", "background", "basic facts",
                    "at a glance", "tell me about", "describe", "introduction"
                ]
                return sum(1 for sig in signals if sig in s2)

            def _is_bad_main(s: str) -> bool:
                if not s or len(s) < 8:
                    return True
                return s.lower().startswith(
                    ("as well as", "as well", "and ", "also ", "plus ", "as for ")
                )

            merged_side = []
            for s in det_side + llm_side:
                s = str(s).strip()
                if s and s not in merged_side:
                    merged_side.append(s)

            det_score = _overview_score(det_main)
            llm_score = _overview_score(llm_main)

            if llm_main and not _is_bad_main(llm_main):
                if not det_main or llm_score > det_score:
                    main = llm_main

            side = merged_side

    side = _dedupe_clauses([s.strip() for s in (side or []) if s.strip()])

    return {
        "category": category or "unknown",
        "category_confidence": round(max(0.0, min(cat_conf, 1.0)), 2),
        "main": (main or "").strip(),
        "side": side,
        "debug": debug,
    }


def format_query_structure_for_prompt(qs: Optional[Dict[str, Any]]) -> str:
    if not qs or not isinstance(qs, dict):
        return ""

    parts = []
    parts.append("STRUCTURED QUESTION (DETERMINISTIC):")
    parts.append(f"- Category: {qs.get('category','unknown')} (conf {qs.get('category_confidence','')})")
    parts.append(f"- Main (answer this FIRST): {qs.get('main','')}")
    side = qs.get("side") or []

    if side:
        parts.append("- Side questions (answer AFTER main, in this exact order):")
        for i, s in enumerate(side[:10], 1):
            parts.append(f"  {i}. {s}")

    tmpl = qs.get("template_sections") or []
    if tmpl:
        parts.append("- Recommended response sections (use as headings if helpful):")
        for t in tmpl[:10]:
            parts.append(f"  - {t}")

    # Hard behavioral instruction to the LLM (kept short and explicit)
    parts.append(
        "RESPONSE RULES:\n"
        "1) Start by answering the MAIN request with general context.\n"
        "2) Then answer EACH side question explicitly (label them).\n"
        "3) Metrics/findings can include both main + side, but do not ignore the main.\n"
        "4) If you provide tourism/industry metrics, ALSO provide basic country/overview facts when main is an overview."
    )

    return "\n".join(parts).strip()


# ------------------------------------
# METRIC DIFF COMPUTATION
# ------------------------------------

def compute_metric_diffs(old_metrics: Dict, new_metrics: Dict) -> List[MetricDiff]:
    """
    Compute deterministic diffs between metric dictionaries.
    Returns list of MetricDiff objects.
    """
    diffs = []
    matched_new_keys = set()

    # Build lookup for new metrics by normalized name
    new_by_name = {}
    for key, m in new_metrics.items():
        if isinstance(m, dict):
            name = m.get('name', key)
            new_by_name[normalize_name(name)] = (key, m)

    # Process old metrics
    for old_key, old_m in old_metrics.items():
        if not isinstance(old_m, dict):
            continue

        old_name = old_m.get('name', old_key)
        old_raw = str(old_m.get('value', ''))
        old_unit = old_m.get('unit', '')
        old_val = parse_to_float(old_m.get('value'))

        # Find matching new metric
        norm_name = normalize_name(old_name)
        match = new_by_name.get(norm_name)

        if not match:
            # Try fuzzy matching
            best = find_best_match(old_name, [m.get('name', k) for k, m in new_metrics.items() if isinstance(m, dict)])
            if best:
                for k, m in new_metrics.items():
                    if isinstance(m, dict) and m.get('name', k) == best:
                        match = (k, m)
                        break

        if match:
            new_key, new_m = match
            matched_new_keys.add(new_key)

            new_raw = str(new_m.get('value', ''))
            new_val = parse_to_float(new_m.get('value'))
            new_unit = new_m.get('unit', old_unit)

            change_pct = compute_percent_change(old_val, new_val)

            # Determine change type
            if change_pct is None:
                change_type = 'unchanged'
            elif abs(change_pct) < 0.5:  # Less than 0.5% change = unchanged
                change_type = 'unchanged'
            elif change_pct > 0:
                change_type = 'increased'
            else:
                change_type = 'decreased'

            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=new_val,
                old_raw=old_raw,
                new_raw=new_raw,
                unit=new_unit or old_unit,
                change_pct=change_pct,
                change_type=change_type
            ))
        else:
            # Metric was removed
            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=None,
                old_raw=old_raw,
                new_raw='',
                unit=old_unit,
                change_pct=None,
                change_type='removed'
            ))

    # Find added metrics
    for new_key, new_m in new_metrics.items():
        if new_key in matched_new_keys:
            continue
        if not isinstance(new_m, dict):
            continue

        new_name = new_m.get('name', new_key)
        new_raw = str(new_m.get('value', ''))
        new_val = parse_to_float(new_m.get('value'))
        new_unit = new_m.get('unit', '')

        diffs.append(MetricDiff(
            name=new_name,
            old_value=None,
            new_value=new_val,
            old_raw='',
            new_raw=new_raw,
            unit=new_unit,
            change_pct=None,
            change_type='added'
        ))

    return diffs

# ------------------------------------
# ENTITY DIFF COMPUTATION
# ------------------------------------

def compute_entity_diffs(old_entities: List, new_entities: List) -> List[EntityDiff]:
    """
    Compute deterministic diffs between entity rankings.
    """
    diffs = []

    # Build lookups with ranks
    old_lookup = {}
    for i, e in enumerate(old_entities):
        if isinstance(e, dict):
            name = normalize_name(e.get('name', ''))
            old_lookup[name] = {
                'rank': i + 1,
                'share': e.get('share'),
                'original_name': e.get('name', '')
            }

    new_lookup = {}
    for i, e in enumerate(new_entities):
        if isinstance(e, dict):
            name = normalize_name(e.get('name', ''))
            new_lookup[name] = {
                'rank': i + 1,
                'share': e.get('share'),
                'original_name': e.get('name', '')
            }

    # All unique names
    all_names = set(old_lookup.keys()) | set(new_lookup.keys())

    for norm_name in all_names:
        old_data = old_lookup.get(norm_name)
        new_data = new_lookup.get(norm_name)

        if old_data and new_data:
            # Entity exists in both
            rank_change = old_data['rank'] - new_data['rank']  # Positive = moved up

            if rank_change > 0:
                change_type = 'moved_up'
            elif rank_change < 0:
                change_type = 'moved_down'
            else:
                change_type = 'unchanged'

            diffs.append(EntityDiff(
                name=new_data['original_name'],
                old_rank=old_data['rank'],
                new_rank=new_data['rank'],
                old_share=old_data['share'],
                new_share=new_data['share'],
                rank_change=rank_change,
                change_type=change_type
            ))
        elif old_data:
            # Entity removed
            diffs.append(EntityDiff(
                name=old_data['original_name'],
                old_rank=old_data['rank'],
                new_rank=None,
                old_share=old_data['share'],
                new_share=None,
                rank_change=None,
                change_type='removed'
            ))
        else:
            # Entity added
            diffs.append(EntityDiff(
                name=new_data['original_name'],
                old_rank=None,
                new_rank=new_data['rank'],
                old_share=None,
                new_share=new_data['share'],
                rank_change=None,
                change_type='added'
            ))

    # Sort by new rank (added entities at end)
    diffs.sort(key=lambda x: x.new_rank if x.new_rank else 999)
    return diffs

# ------------------------------------
# FINDING DIFF COMPUTATION
# ------------------------------------

def compute_finding_diffs(old_findings: List[str], new_findings: List[str]) -> List[FindingDiff]:
    """
    Compute deterministic diffs between findings using text similarity.
    """
    diffs = []
    matched_new_indices = set()

    # Match old findings to new
    for old_f in old_findings:
        if not old_f:
            continue

        best_match_idx = None
        best_similarity = 0.5  # Minimum threshold

        for i, new_f in enumerate(new_findings):
            if i in matched_new_indices or not new_f:
                continue

            sim = name_similarity(old_f, new_f)  # Reuse name similarity for text
            if sim > best_similarity:
                best_similarity = sim
                best_match_idx = i

        if best_match_idx is not None:
            matched_new_indices.add(best_match_idx)
            similarity_pct = round(best_similarity * 100, 1)

            if similarity_pct >= 90:
                change_type = 'retained'
            else:
                change_type = 'modified'

            diffs.append(FindingDiff(
                old_text=old_f,
                new_text=new_findings[best_match_idx],
                similarity=similarity_pct,
                change_type=change_type
            ))
        else:
            # Finding removed
            diffs.append(FindingDiff(
                old_text=old_f,
                new_text=None,
                similarity=0,
                change_type='removed'
            ))

    # Find added findings
    for i, new_f in enumerate(new_findings):
        if i in matched_new_indices or not new_f:
            continue

        diffs.append(FindingDiff(
            old_text=None,
            new_text=new_f,
            similarity=0,
            change_type='added'
        ))

    return diffs

# =========================================================
# 8C. DETERMINISTIC SOURCE EXTRACTION
# Extract metrics/entities directly from web snippets - NO LLM
# =========================================================

def extract_metrics_from_sources(web_context: Dict) -> Dict:
    """
    Extract numeric metrics directly from web search snippets.
    100% deterministic - no LLM involved.
    """
    extracted = {}
    search_results = web_context.get("search_results", [])

    # Patterns to match common metric formats
    patterns = [
        # Market size patterns
        (r'\$\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)\b', 'market_size'),
        (r'market\s+size[:\s]+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'market_size'),
        (r'valued\s+at\s+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'market_size'),
        (r'worth\s+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'market_size'),

        # Growth rate patterns
        (r'CAGR[:\s]+of?\s*(\d+(?:\.\d+)?)\s*%', 'cagr'),
        (r'(\d+(?:\.\d+)?)\s*%\s*CAGR', 'cagr'),
        (r'grow(?:th|ing)?\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%', 'growth_rate'),

        # Revenue patterns
        (r'revenue[:\s]+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'revenue'),

        # Year-specific values
        (r'(?:in\s+)?20\d{2}[:\s]+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'year_value'),
    ]

    all_matches = []

    for result in search_results:
        snippet = result.get("snippet", "")
        title = result.get("title", "")
        source = result.get("source", "")
        text = f"{title} {snippet}".lower()

        for pattern, metric_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    value_str, unit = match[0], match[1] if len(match) > 1 else ''
                else:
                    value_str, unit = match, ''

                try:
                    value = float(value_str)

                    # Normalize unit
                    unit_lower = unit.lower() if unit else ''
                    if unit_lower in ['t', 'trillion']:
                        unit_normalized = 'T'
                        value_in_billions = value * 1000
                    elif unit_lower in ['b', 'billion']:
                        unit_normalized = 'B'
                        value_in_billions = value
                    elif unit_lower in ['m', 'million']:
                        unit_normalized = 'M'
                        value_in_billions = value / 1000
                    elif unit_lower == '%':
                        unit_normalized = '%'
                        value_in_billions = value  # Keep as-is for percentages
                    else:
                        unit_normalized = ''
                        value_in_billions = value

                    all_matches.append({
                        'type': metric_type,
                        'value': value,
                        'unit': unit_normalized,
                        'value_normalized': value_in_billions,
                        'source': source,
                        'raw': f"{value_str} {unit}".strip()
                    })
                except (ValueError, TypeError):
                    continue

    # Deduplicate and select best matches by type
    metrics_by_type = {}
    for match in all_matches:
        mtype = match['type']
        if mtype not in metrics_by_type:
            metrics_by_type[mtype] = []
        metrics_by_type[mtype].append(match)

    # For each type, take the most common value (mode) or median
    metric_counter = 0
    for mtype, matches in metrics_by_type.items():
        if not matches:
            continue

        # Group by similar values (within 10%)
        value_groups = []
        for m in matches:
            added = False
            for group in value_groups:
                if group and abs(m['value_normalized'] - group[0]['value_normalized']) / max(group[0]['value_normalized'], 0.001) < 0.1:
                    group.append(m)
                    added = True
                    break
            if not added:
                value_groups.append([m])

        # Take the largest group (most consensus)
        if value_groups:
            best_group = max(value_groups, key=len)
            representative = best_group[0]

            metric_counter += 1
            metric_key = f"extracted_{mtype}_{metric_counter}"

            # Map type to readable name
            type_names = {
                'market_size': 'Market Size',
                'cagr': 'CAGR',
                'growth_rate': 'Growth Rate',
                'revenue': 'Revenue',
                'year_value': 'Market Value'
            }

            extracted[metric_key] = {
                'name': type_names.get(mtype, mtype.replace('_', ' ').title()),
                'value': representative['value'],
                'unit': f"${representative['unit']}" if representative['unit'] in ['T', 'B', 'M'] else representative['unit'],
                'source_count': len(best_group),
                'sources': list(set(m['source'] for m in best_group))[:3]
            }

    return extracted


def extract_entities_from_sources(web_context: Dict) -> List[Dict]:
    """
    Extract company/entity names from web search snippets.
    100% deterministic - no LLM involved.
    """
    search_results = web_context.get("search_results", [])

    # Common market leaders that appear in financial contexts
    known_entities = [
        # Tech
        'apple', 'microsoft', 'google', 'alphabet', 'amazon', 'meta', 'facebook',
        'nvidia', 'tesla', 'intel', 'amd', 'qualcomm', 'broadcom', 'cisco',
        'ibm', 'oracle', 'salesforce', 'adobe', 'netflix', 'uber', 'airbnb',
        # Finance
        'jpmorgan', 'goldman sachs', 'morgan stanley', 'bank of america',
        'wells fargo', 'citigroup', 'blackrock', 'vanguard', 'fidelity',
        # Auto
        'toyota', 'volkswagen', 'ford', 'gm', 'general motors', 'honda',
        'bmw', 'mercedes', 'byd', 'nio', 'rivian', 'lucid',
        # Pharma
        'pfizer', 'johnson & johnson', 'roche', 'novartis', 'merck',
        'abbvie', 'eli lilly', 'astrazeneca', 'moderna', 'gilead',
        # Energy
        'exxon', 'chevron', 'shell', 'bp', 'totalenergies', 'conocophillips',
        # Consumer
        'walmart', 'costco', 'home depot', 'nike', 'starbucks', 'mcdonalds',
        'coca-cola', 'pepsi', 'procter & gamble', 'unilever',
        # Regions (for market share by region)
        'north america', 'europe', 'asia pacific', 'asia-pacific', 'apac',
        'china', 'united states', 'japan', 'germany', 'india', 'uk',
        'latin america', 'middle east', 'africa'
    ]

    entity_mentions = {}

    for result in search_results:
        snippet = result.get("snippet", "").lower()
        title = result.get("title", "").lower()
        text = f"{title} {snippet}"

        for entity in known_entities:
            if entity in text:
                # Try to extract market share if mentioned
                share_pattern = rf'{re.escape(entity)}[^.]*?(\d+(?:\.\d+)?)\s*%'
                share_match = re.search(share_pattern, text, re.IGNORECASE)

                share = None
                if share_match:
                    share = f"{share_match.group(1)}%"

                if entity not in entity_mentions:
                    entity_mentions[entity] = {'count': 0, 'shares': []}

                entity_mentions[entity]['count'] += 1
                if share:
                    entity_mentions[entity]['shares'].append(share)

    # Sort by mention count and build list
    sorted_entities = sorted(entity_mentions.items(), key=lambda x: x[1]['count'], reverse=True)

    entities = []
    for entity_name, data in sorted_entities[:10]:  # Top 10
        # Use most common share if available
        share = None
        if data['shares']:
            # Take the most common share value
            share_counts = Counter(data['shares'])
            share = share_counts.most_common(1)[0][0]

        entities.append({
            'name': entity_name.title(),
            'share': share,
            'growth': None,  # Can't reliably extract growth from snippets
            'mention_count': data['count']
        })

    return entities

# ------------------------------------
# STABILITY SCORE COMPUTATION
# ------------------------------------

def compute_stability_score(
    metric_diffs: List[MetricDiff],
    entity_diffs: List[EntityDiff],
    finding_diffs: List[FindingDiff]
) -> float:
    """
    Compute overall stability score (0-100).
    Higher = more stable (less change).
    """
    scores = []

    # Metric stability (40% weight)
    if metric_diffs:
        stable_metrics = sum(1 for m in metric_diffs if m.change_type == 'unchanged')
        small_change = sum(1 for m in metric_diffs if m.change_pct and abs(m.change_pct) < 10)
        metric_score = ((stable_metrics + small_change * 0.5) / len(metric_diffs)) * 100
        scores.append(('metrics', metric_score, 0.4))

    # Entity stability (35% weight)
    if entity_diffs:
        stable_entities = sum(1 for e in entity_diffs if e.change_type == 'unchanged')
        entity_score = (stable_entities / len(entity_diffs)) * 100
        scores.append(('entities', entity_score, 0.35))

    # Finding stability (25% weight)
    if finding_diffs:
        retained = sum(1 for f in finding_diffs if f.change_type in ['retained', 'modified'])
        finding_score = (retained / len(finding_diffs)) * 100
        scores.append(('findings', finding_score, 0.25))

    if not scores:
        return 100.0

    # Weighted average
    total_weight = sum(s[2] for s in scores)
    weighted_sum = sum(s[1] * s[2] for s in scores)
    return round(weighted_sum / total_weight, 1)

# ------------------------------------
# MAIN DIFF COMPUTATION
# ------------------------------------

def compute_evolution_diff(old_analysis: Dict, new_analysis: Dict) -> EvolutionDiff:
    """
    Main entry point: compute complete deterministic diff between two analyses.
    """
    old_response = old_analysis.get('primary_response', {})
    new_response = new_analysis.get('primary_response', {})

    # Timestamps
    old_ts = old_analysis.get('timestamp', '')
    new_ts = new_analysis.get('timestamp', '')

    # Calculate time delta
    time_delta = None
    try:
        old_dt = datetime.fromisoformat(old_ts.replace('Z', '+00:00'))
        new_dt = datetime.fromisoformat(new_ts.replace('Z', '+00:00'))
        time_delta = round((new_dt.replace(tzinfo=None) - old_dt.replace(tzinfo=None)).total_seconds() / 3600, 1)
    except:
        pass

    # Compute diffs using CANONICAL metric registry for stable matching
    metric_diffs = compute_metric_diffs_canonical(
        old_response.get('primary_metrics', {}),
        new_response.get('primary_metrics', {})
    )

    entity_diffs = compute_entity_diffs(
        old_response.get('top_entities', []),
        new_response.get('top_entities', [])
    )

    # Use SEMANTIC finding comparison (stable across wording changes)
    finding_diffs = compute_semantic_finding_diffs(
        old_response.get('key_findings', []),
        new_response.get('key_findings', [])
    )

    # Compute stability
    stability = compute_stability_score(metric_diffs, entity_diffs, finding_diffs)

    # Summary stats
    summary_stats = {
        'metrics_increased': sum(1 for m in metric_diffs if m.change_type == 'increased'),
        'metrics_decreased': sum(1 for m in metric_diffs if m.change_type == 'decreased'),
        'metrics_unchanged': sum(1 for m in metric_diffs if m.change_type == 'unchanged'),
        'metrics_added': sum(1 for m in metric_diffs if m.change_type == 'added'),
        'metrics_removed': sum(1 for m in metric_diffs if m.change_type == 'removed'),
        'entities_moved_up': sum(1 for e in entity_diffs if e.change_type == 'moved_up'),
        'entities_moved_down': sum(1 for e in entity_diffs if e.change_type == 'moved_down'),
        'entities_unchanged': sum(1 for e in entity_diffs if e.change_type == 'unchanged'),
        'entities_added': sum(1 for e in entity_diffs if e.change_type == 'added'),
        'entities_removed': sum(1 for e in entity_diffs if e.change_type == 'removed'),
        'findings_retained': sum(1 for f in finding_diffs if f.change_type == 'retained'),
        'findings_modified': sum(1 for f in finding_diffs if f.change_type == 'modified'),
        'findings_added': sum(1 for f in finding_diffs if f.change_type == 'added'),
        'findings_removed': sum(1 for f in finding_diffs if f.change_type == 'removed'),
    }

    return EvolutionDiff(
        old_timestamp=old_ts,
        new_timestamp=new_ts,
        time_delta_hours=time_delta,
        metric_diffs=metric_diffs,
        entity_diffs=entity_diffs,
        finding_diffs=finding_diffs,
        stability_score=stability,
        summary_stats=summary_stats
    )

# ------------------------------------
# LLM EXPLANATION (ONLY INTERPRETS DIFFS)
# ------------------------------------

def generate_diff_explanation_prompt(diff: EvolutionDiff, query: str) -> str:
    """
    Generate prompt for LLM to EXPLAIN computed diffs (not discover them).
    """
    # Build metric changes text
    metric_changes = []
    for m in diff.metric_diffs:
        if m.change_type == 'increased':
            metric_changes.append(f"- {m.name}: {m.old_raw} → {m.new_raw} ({m.change_pct:+.1f}%) INCREASED")
        elif m.change_type == 'decreased':
            metric_changes.append(f"- {m.name}: {m.old_raw} → {m.new_raw} ({m.change_pct:+.1f}%) DECREASED")
        elif m.change_type == 'added':
            metric_changes.append(f"- {m.name}: NEW metric added with value {m.new_raw}")
        elif m.change_type == 'removed':
            metric_changes.append(f"- {m.name}: REMOVED (was {m.old_raw})")

    # Build entity changes text
    entity_changes = []
    for e in diff.entity_diffs:
        if e.change_type == 'moved_up':
            entity_changes.append(f"- {e.name}: Rank {e.old_rank} → {e.new_rank} (moved UP {e.rank_change} positions)")
        elif e.change_type == 'moved_down':
            entity_changes.append(f"- {e.name}: Rank {e.old_rank} → {e.new_rank} (moved DOWN {abs(e.rank_change)} positions)")
        elif e.change_type == 'added':
            entity_changes.append(f"- {e.name}: NEW entrant at rank {e.new_rank}")
        elif e.change_type == 'removed':
            entity_changes.append(f"- {e.name}: DROPPED OUT (was rank {e.old_rank})")

    # Build findings changes text
    finding_changes = []
    for f in diff.finding_diffs:
        if f.change_type == 'added':
            finding_changes.append(f"- NEW: {f.new_text}")
        elif f.change_type == 'removed':
            finding_changes.append(f"- REMOVED: {f.old_text}")
        elif f.change_type == 'modified':
            finding_changes.append(f"- MODIFIED: '{f.old_text[:50]}...' → '{f.new_text[:50]}...'")

    prompt = f"""You are a market analyst explaining changes between two analysis snapshots.

    QUERY: {query}
    TIME ELAPSED: {diff.time_delta_hours:.1f} hours
    STABILITY SCORE: {diff.stability_score:.0f}%

    COMPUTED METRIC CHANGES:
    {chr(10).join(metric_changes) if metric_changes else "No significant metric changes"}

    COMPUTED ENTITY RANKING CHANGES:
    {chr(10).join(entity_changes) if entity_changes else "No ranking changes"}

    COMPUTED FINDING CHANGES:
    {chr(10).join(finding_changes) if finding_changes else "No finding changes"}

    SUMMARY STATS:
    - Metrics: {diff.summary_stats['metrics_increased']} increased, {diff.summary_stats['metrics_decreased']} decreased, {diff.summary_stats['metrics_unchanged']} unchanged
    - Entities: {diff.summary_stats['entities_moved_up']} moved up, {diff.summary_stats['entities_moved_down']} moved down
    - Findings: {diff.summary_stats['findings_added']} new, {diff.summary_stats['findings_removed']} removed

    YOUR TASK: Provide a 3-5 sentence executive interpretation of these changes.
    - What is the overall trend (improving/declining/stable)?
    - What are the most significant changes and why might they have occurred?
    - What should stakeholders pay attention to?

    Return ONLY a JSON object:
    {{
        "trend": "improving/declining/stable",
        "headline": "One sentence summary of key change",
        "interpretation": "3-5 sentence detailed interpretation",
        "watch_items": ["Item 1 to monitor", "Item 2 to monitor"]
    }}
    """
    return prompt

def get_llm_explanation(diff: EvolutionDiff, query: str) -> Dict:
    """
    Ask LLM to explain the computed diffs (not discover them).
    """
    prompt = generate_diff_explanation_prompt(diff, query)

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 500,
        "top_p": 1.0,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        parsed = parse_json_safely(content, "Explanation")
        if parsed:
            return parsed
    except Exception as e:
        st.warning(f"LLM explanation failed: {e}")

    # Fallback
    return {
        "trend": "stable" if diff.stability_score >= 70 else "changing",
        "headline": f"Analysis shows {diff.stability_score:.0f}% stability over {diff.time_delta_hours:.0f} hours",
        "interpretation": "Unable to generate detailed interpretation.",
        "watch_items": []
    }


# =========================================================
# 8B. EVOLUTION DASHBOARD RENDERING
# =========================================================

def render_evolution_results(diff: EvolutionDiff, explanation: Dict, query: str):
    """Render deterministic evolution results"""

    st.header("📈 Evolution Analysis")
    st.markdown(f"**Query:** {query}")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    if diff.time_delta_hours:
        if diff.time_delta_hours < 24:
            time_str = f"{diff.time_delta_hours:.1f}h"
        else:
            time_str = f"{diff.time_delta_hours/24:.1f}d"
        col1.metric("Time Elapsed", time_str)
    else:
        col1.metric("Time Elapsed", "Unknown")

    col2.metric("Stability", f"{diff.stability_score:.0f}%")

    trend = explanation.get('trend', 'stable')
    trend_icon = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}.get(trend, '➡️')
    col3.metric("Trend", f"{trend_icon} {trend.title()}")

    # Stability indicator
    if diff.stability_score >= 80:
        col4.success("🟢 Highly Stable")
    elif diff.stability_score >= 60:
        col4.warning("🟡 Moderate Changes")
    else:
        col4.error("🔴 Significant Drift")

    # Headline
    st.info(f"**{explanation.get('headline', 'Analysis complete')}**")

    st.markdown("---")

    # Interpretation
    st.subheader("📋 Interpretation")
    st.markdown(explanation.get('interpretation', 'No interpretation available'))

    # Watch items
    watch_items = explanation.get('watch_items', [])
    if watch_items:
        st.markdown("**🔔 Watch Items:**")
        for item in watch_items:
            st.markdown(f"- {item}")

    st.markdown("---")

    # Metric Changes Table
    st.subheader("💰 Metric Changes")
    if diff.metric_diffs:
        metric_rows = []

        def _fmt_currency_first(raw: str, unit: str) -> str:
            """
            Formats evolution metrics as:
            - S$29.8B
            - $120M
            - 29.8%
            """
            raw = (raw or "").strip()
            unit = (unit or "").strip()

            if not raw or raw == "-":
                return "-"

            # If already currency-first, trust it
            if raw.startswith("S$") or raw.startswith("$"):
                return raw

            # Percent case
            if unit == "%":
                return f"{raw}%"

            # Detect currency from unit
            currency = ""
            scale = unit.replace(" ", "")

            if scale.upper().startswith("SGD"):
                currency = "S$"
                scale = scale[3:]
            elif scale.upper().startswith("USD"):
                currency = "$"
                scale = scale[3:]
            elif scale.startswith("S$"):
                currency = "S$"
                scale = scale[2:]
            elif scale.startswith("$"):
                currency = "$"
                scale = scale[1:]

            # Human-readable units
            if unit.lower().endswith("billion"):
                return f"{currency}{raw} billion".strip()
            if unit.lower().endswith("million"):
                return f"{currency}{raw} million".strip()

            # Compact units (B/M/K)
            if scale.upper() in {"B", "M", "K"}:
                return f"{currency}{raw}{scale}".strip()

            # Fallback
            return f"{currency}{raw} {unit}".strip()

        for m in diff.metric_diffs:
            icon = {
                'increased': '📈', 'decreased': '📉', 'unchanged': '➡️',
                'added': '🆕', 'removed': '❌'
            }.get(m.change_type, '•')

            change_str = f"{m.change_pct:+.1f}%" if m.change_pct is not None else "-"

            prev_raw = m.old_raw or "-"
            curr_raw = m.new_raw or "-"

            metric_rows.append({
                "": icon,
                "Metric": m.name,
                "Previous": _fmt_currency_first(prev_raw, getattr(m, "unit", "") or ""),
                "Current":  _fmt_currency_first(curr_raw, getattr(m, "unit", "") or ""),
                "Change": change_str,
                "Status": m.change_type.replace('_', ' ').title()
            })

        st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No metrics to compare")

    st.markdown("---")

    # Entity Changes Table
    st.subheader("🏢 Entity Ranking Changes")
    if diff.entity_diffs:
        entity_rows = []
        for e in diff.entity_diffs:
            icon = {
                'moved_up': '⬆️', 'moved_down': '⬇️', 'unchanged': '➡️',
                'added': '🆕', 'removed': '❌'
            }.get(e.change_type, '•')

            rank_str = f"{e.rank_change:+d}" if e.rank_change else "-"

            entity_rows.append({
                "": icon,
                "Entity": e.name,
                "Old Rank": e.old_rank or "-",
                "New Rank": e.new_rank or "-",
                "Rank Δ": rank_str,
                "Old Share": e.old_share or "-",
                "New Share": e.new_share or "-"
            })
        st.dataframe(pd.DataFrame(entity_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No entities to compare")

    st.markdown("---")

    # Finding Changes
    st.subheader("🔍 Finding Changes")
    if diff.finding_diffs:
        added = [f for f in diff.finding_diffs if f.change_type == 'added']
        removed = [f for f in diff.finding_diffs if f.change_type == 'removed']
        modified = [f for f in diff.finding_diffs if f.change_type == 'modified']

        if added:
            st.markdown("**🆕 New Findings:**")
            for f in added:
                st.success(f"• {f.new_text}")

        if removed:
            st.markdown("**❌ Removed Findings:**")
            for f in removed:
                st.error(f"• ~~{f.old_text}~~")

        if modified:
            st.markdown("**✏️ Modified Findings:**")
            for f in modified:
                st.warning(f"• {f.new_text} *(similarity: {f.similarity:.0f}%)*")
    else:
        st.info("No findings to compare")

    st.markdown("---")

    # Summary Stats
    st.subheader("📊 Change Summary")
    stats = diff.summary_stats

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Metrics:**")
        st.write(f"📈 {stats['metrics_increased']} increased")
        st.write(f"📉 {stats['metrics_decreased']} decreased")
        st.write(f"➡️ {stats['metrics_unchanged']} unchanged")

    with col2:
        st.markdown("**Entities:**")
        st.write(f"⬆️ {stats['entities_moved_up']} moved up")
        st.write(f"⬇️ {stats['entities_moved_down']} moved down")
        st.write(f"🆕 {stats['entities_added']} new")

    with col3:
        st.markdown("**Findings:**")
        st.write(f"✅ {stats['findings_retained']} retained")
        st.write(f"✏️ {stats['findings_modified']} modified")
        st.write(f"🆕 {stats['findings_added']} new")


# =========================================================
# 8D. SOURCE-ANCHORED EVOLUTION
# Re-fetch the SAME sources from previous analysis for true stability
# Enhanced fetch_url_content function to use scrapingdog as fallback
# =========================================================

def _extract_pdf_text_from_bytes(pdf_bytes: bytes, max_pages: int = 6, max_chars: int = 7000) -> Optional[str]:
    """
    Extract readable text from PDF bytes deterministically.
    Limits pages/chars for speed and consistent output.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for i, page in enumerate(reader.pages[:max_pages]):
            t = page.extract_text() or ""
            t = t.replace("\x00", " ").strip()
            if t:
                texts.append(t)
        joined = "\n".join(texts).strip()
        if len(joined) < 200:
            return None
        return joined[:max_chars]
    except Exception:
        return None



def fetch_url_content(url: str) -> Optional[str]:
    """Fetch content from a specific URL with ScrapingDog fallback"""

    def extract_text(html: str) -> Optional[str]:
        """Extract clean text from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        clean_text = ' '.join(line for line in lines if line)
        return clean_text[:5000] if len(clean_text) > 200 else None

    # Try 1: Direct request
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        if 'captcha' not in resp.text.lower():
            content = extract_text(resp.text)
            if content:
                return content
    except:
        pass

    # Try 2: ScrapingDog API
    if SCRAPINGDOG_KEY:
        try:
            api_url = "https://api.scrapingdog.com/scrape"
            params = {"api_key": SCRAPINGDOG_KEY, "url": url, "dynamic": "false"}
            resp = requests.get(api_url, params=params, timeout=30)
            if resp.status_code == 200:
                content = extract_text(resp.text)
                if content:
                    return content
        except:
            pass

    return None

def fetch_url_content_with_status(url: str) -> Tuple[Optional[str], str]:
    """
    Fetch content from a URL and return (text, status_msg).

    Fixes:
      - Normalizes bare domains like "singstat.gov.sg" -> "https://singstat.gov.sg"
      - Rejects non-http(s) schemes
      - Keeps previous behavior for HTML/PDF and blocked/captcha detection
    """
    # ---- URL normalization ----
    def _normalize_url(u: str) -> Optional[str]:
        if not u or not isinstance(u, str):
            return None
        u = u.strip()

        # Reject non-web schemes
        bad_prefixes = ("mailto:", "javascript:", "data:", "file:", "ftp:")
        if u.lower().startswith(bad_prefixes):
            return None

        # If it's just a bare domain, add https://
        if "://" not in u:
            u = "https://" + u

        return u

    url_norm = _normalize_url(url)
    if not url_norm:
        return None, "invalid_url"

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        resp = requests.get(url_norm, headers=headers, timeout=25, allow_redirects=True)

        if resp.status_code == 304:
            return None, "not_modified"

        if resp.status_code >= 400:
            if resp.status_code in (403, 429, 503):
                return None, f"blocked_or_rate_limited:{resp.status_code}"
            return None, f"http_error:{resp.status_code}"

        raw_bytes = resp.content or b""
        if not raw_bytes:
            return None, "empty_response"

        content_type = (resp.headers.get("Content-Type") or "").lower()
        is_pdf = url_norm.lower().endswith(".pdf") or ("application/pdf" in content_type)

        if is_pdf:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
                page_texts = []
                for i in range(len(doc)):
                    t = doc.load_page(i).get_text("text") or ""
                    t = re.sub(r"[ \t]+", " ", t)
                    t = re.sub(r"\n{3,}", "\n\n", t).strip()
                    if t:
                        page_texts.append((i, t))

                if not page_texts:
                    return None, "pdf_no_text"

                full_text = "\n\n".join([f"[PDF_PAGE_{pno}]\n{txt}" for (pno, txt) in page_texts])

                low = full_text.lower()
                if any(s in low for s in ["captcha", "verify you are human", "access denied", "unusual traffic"]):
                    return None, "captcha_or_blocked"

                return full_text, "success_pdf"
            except Exception:
                return None, "pdf_extract_failed"

        # HTML / text
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw_bytes, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n")
        except Exception:
            text = raw_bytes.decode("utf-8", errors="ignore")

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        low = text.lower()
        if any(s in low for s in [
            "captcha", "verify you are human", "access denied", "unusual traffic",
            "enable javascript", "your request has been blocked", "robot"
        ]):
            return None, "captcha_or_blocked"

        if len(text) < 250:
            return None, "too_short_or_unusable"

        return text, "success"

    except requests.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, f"exception:{type(e).__name__}"


def extract_numbers_from_text(text: str) -> List[Dict]:
    """Extract all numbers with context from text"""
    if not text:
        return []

    numbers = []

    # Pattern: number with optional unit, capture surrounding context
    pattern = r'(\$?\d+(?:\.\d+)?)\s*(trillion|billion|million|%|T|B|M)?'

    for match in re.finditer(pattern, text, re.IGNORECASE):
        value_str = match.group(1).replace('$', '')
        unit = match.group(2) or ''

        try:
            value = float(value_str)
        except:
            continue

        # Skip very small or very large unlikely values
        if value == 0 or value > 1000000:
            continue

        # Get context (150 chars before and after)
        start = max(0, match.start() - 150)
        end = min(len(text), match.end() + 150)
        context = text[start:end].lower()

        # Normalize unit
        unit_map = {'trillion': 'T', 't': 'T', 'billion': 'B', 'b': 'B', 'million': 'M', 'm': 'M', '%': '%'}
        unit_norm = unit_map.get(unit.lower(), '') if unit else ''

        numbers.append({
            'value': value,
            'unit': unit_norm,
            'context': context,
            'raw': f"{value_str}{unit}"
        })

    return numbers


def _parse_iso_dt(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        ts2 = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def now_utc() -> datetime:
    """Timezone-aware UTC now (prevents naive/aware datetime bugs)."""
    return datetime.now(timezone.utc)


def _normalize_number_to_parse_base(value: float, unit: str) -> float:
    u = (unit or "").strip().upper()
    if u == "T":
        return value * 1_000_000
    if u == "B":
        return value * 1_000
    if u == "M":
        return value * 1
    if u == "K":
        return value * 0.001
    if u == "%":
        return value
    return value


# =========================================================
# ROBUST EVOLUTION HELPERS (DETERMINISTIC)
# =========================================================

NON_DATA_CONTEXT_HINTS = [
    "table of contents", "cookie", "privacy", "terms", "copyright",
    "subscribe", "newsletter", "login", "sign in", "nav", "footer"
]

def fingerprint_text(text: str) -> str:
    """Stable short fingerprint for fetched content (deterministic)."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]

def normalize_unit(unit: str) -> str:
    """Normalize unit to one of: T/B/M/%/'' (deterministic)."""
    if not unit:
        return ""
    u = unit.strip().upper().replace("USD", "").replace("$", "").replace(" ", "")
    if u in ["TRILLION", "T"]:
        return "T"
    if u in ["BILLION", "B"]:
        return "B"
    if u in ["MILLION", "M"]:
        return "M"
    if u in ["PERCENT", "%"]:
        return "%"
    if u in ["K", "THOUSAND"]:
        return "K"
    return u

def normalize_currency_prefix(raw: str) -> bool:
    """True if looks like a currency number ($/USD)."""
    if not raw:
        return False
    s = raw.strip().upper()
    return s.startswith("$") or " USD" in s or s.startswith("USD")

def is_likely_junk_context(context: str) -> bool:
    """Reject contexts that are likely not data-bearing (deterministic)."""
    if not context:
        return False
    c = context.lower()
    return any(h in c for h in NON_DATA_CONTEXT_HINTS)

def parse_human_number(value_str: str, unit: str) -> Optional[float]:
    """
    Parse number + unit into a comparable float scale.
    - For T/B/M: returns value in billions (B) to compare apples-to-apples.
    - For %: returns numeric percent.
    """
    if value_str is None:
        return None

    s = str(value_str).strip()
    if not s:
        return None

    # remove currency symbols/commas/space
    s = s.replace("$", "").replace(",", "").strip()

    # handle parentheses for negatives e.g. (12.3)
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()

    try:
        v = float(s)
    except Exception:
        return None

    u = normalize_unit(unit)

    # Normalize magnitudes into BILLIONS for currency-like units
    if u == "T":
        return v * 1000.0
    if u == "B":
        return v
    if u == "M":
        return v / 1000.0
    if u == "K":
        return v / 1_000_000.0

    # Percent: keep as percent number
    if u == "%":
        return v

    # Unknown unit: leave as-is (still useful for ratio filtering)
    return v

def build_prev_numbers(prev_metrics: Dict) -> Dict[str, Dict]:
    """
    Build previous metric lookup keyed by metric_name string.
    Stores:
      - parsed numeric value (for matching)
      - normalized unit (for gating)
      - raw display string INCLUDING currency/magnitude (for dashboards + evolution JSON)
      - raw_value/raw_unit for debugging
    """
    def _format_raw_display(value: Any, unit: str) -> str:
        v = "" if value is None else str(value).strip()
        u = (unit or "").strip()

        if not v:
            return ""

        # Currency prefix handling (SGD/USD keywords OR symbol prefixes)
        currency = ""
        u_nospace = u.replace(" ", "")

        if u_nospace.upper().startswith("SGD"):
            currency = "S$"
            u_tail = u_nospace[3:]
        elif u_nospace.upper().startswith("USD"):
            currency = "$"
            u_tail = u_nospace[3:]
        elif u_nospace.startswith("S$"):
            currency = "S$"
            u_tail = u_nospace[2:]
        elif u_nospace.startswith("$"):
            currency = "$"
            u_tail = u_nospace[1:]
        else:
            u_tail = u_nospace

        # Percent special case
        if u_tail == "%":
            return f"{v}%"

        # Word scales
        if "billion" in u.lower():
            return f"{currency}{v} billion".strip()
        if "million" in u.lower():
            return f"{currency}{v} million".strip()

        # Compact suffix (B/M/K/T)
        if u_tail.upper() in {"T", "B", "M", "K"}:
            return f"{currency}{v}{u_tail.upper()}".strip()

        # Fallback
        return f"{currency}{v} {u}".strip()

    prev_numbers: Dict[str, Dict] = {}
    for key, metric in (prev_metrics or {}).items():
        if not isinstance(metric, dict):
            continue

        metric_name = metric.get("name", key)
        raw_value = metric.get("value", "")
        raw_unit = metric.get("unit", "")

        val = parse_human_number(str(raw_value), raw_unit)
        if val is None:
            continue

        prev_numbers[metric_name] = {
            "value": val,
            "unit": normalize_unit(raw_unit),
            "raw": _format_raw_display(raw_value, raw_unit),   # ✅ now includes currency + unit
            "raw_value": raw_value,
            "raw_unit": raw_unit,
            "keywords": extract_context_keywords(metric_name),
        }

    return prev_numbers

def compute_source_anchored_diff(previous_data: Dict) -> Dict:
    """
    Source-anchored evolution analysis.

    Key fix in this drop-in:
      - Normalize every baseline/source URL so bare domains become https://...
      - This prevents requests.exceptions.MissingSchema and allows extraction + caching.

    NOTE: This version does not change your matching rules; it fixes fetch reliability.
    """
    prev_response = previous_data.get("primary_response", {}) or {}
    prev_sources = prev_response.get("sources", []) or previous_data.get("web_sources", []) or []
    prev_sources = [s for s in prev_sources if isinstance(s, str) and s.strip()]

    # Baseline cache (previous evolution run)
    baseline_sources_cache = (
        (previous_data.get("results", {}) or {}).get("source_results", [])
        or previous_data.get("source_results", [])
        or []
    )
    if isinstance(baseline_sources_cache, dict):
        baseline_sources_cache = baseline_sources_cache.get("source_results") or []
    if not isinstance(baseline_sources_cache, list):
        baseline_sources_cache = []

    def _normalize_url(u: str) -> Optional[str]:
        if not u or not isinstance(u, str):
            return None
        u = u.strip()
        bad_prefixes = ("mailto:", "javascript:", "data:", "file:", "ftp:")
        if u.lower().startswith(bad_prefixes):
            return None
        if "://" not in u:
            u = "https://" + u
        return u

    # Normalize the sources we will check (keep originals for display if you want)
    sources_to_check = []
    for s in prev_sources:
        nu = _normalize_url(s)
        if nu:
            sources_to_check.append(nu)

    # If prev_sources are empty or invalid, fall back to baseline source_results urls
    if not sources_to_check:
        for sr in baseline_sources_cache:
            if isinstance(sr, dict):
                nu = _normalize_url(sr.get("url", ""))
                if nu:
                    sources_to_check.append(nu)

    # De-dup and cap
    sources_to_check = list(dict.fromkeys(sources_to_check))[:8]

    source_results = []
    all_current_numbers = []

    for url in sources_to_check:
        content, status_msg = fetch_url_content_with_status(url)

        if content:
            try:
                nums = extract_numbers_with_context_pdf(content) if status_msg == "success_pdf" else extract_numbers_with_context(content)
            except Exception:
                nums = []

            compact = []
            for n in nums or []:
                if not isinstance(n, dict):
                    continue
                compact.append({
                    "value": n.get("value"),
                    "unit": n.get("unit"),
                    "raw": n.get("raw"),
                    "source_url": url,
                    "context": (n.get("context", "")[:220] if isinstance(n.get("context"), str) else "")
                })

            source_results.append({
                "url": url,
                "status": "fetched_extracted" if compact else "fetched_unusable",
                "status_detail": status_msg if compact else f"{status_msg}_but_no_numbers",
                "numbers_found": len(compact),
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "extracted_numbers": compact,
            })

            # Keep full candidates for matching layer if you already do that downstream
            for n in nums or []:
                if isinstance(n, dict):
                    n["source_url"] = url
                    all_current_numbers.append(n)
        else:
            source_results.append({
                "url": url,
                "status": "failed",
                "status_detail": status_msg,
                "numbers_found": 0,
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "extracted_numbers": [],
            })

    # Return minimal structure (your downstream matcher can enrich this)
    return {
        "status": "success",
        "sources_checked": len(sources_to_check),
        "sources_fetched": sum(1 for r in source_results if r.get("status", "").startswith("fetched")),
        "source_results": source_results,
        "numbers_extracted_total": sum(int(r.get("numbers_found") or 0) for r in source_results),
        # keep these if your renderer expects them
        "metric_changes": [],
        "stability_score": None,
        "interpretation": "Fetch/extraction completed. (Matching layer not shown in this minimal drop-in.)"
    }


def extract_context_keywords(metric_name: str) -> List[str]:
    """Extract meaningful keywords from metric name for matching"""
    name_lower = metric_name.lower()
    keywords = []

    # Year patterns
    year_match = re.findall(r'20\d{2}', metric_name)
    keywords.extend(year_match)

    # Common metric keywords
    keyword_patterns = [
        'market size', 'revenue', 'sales', 'growth', 'cagr', 'share',
        'projected', 'forecast', 'estimate', 'actual',
        'q1', 'q2', 'q3', 'q4', 'quarter',
        'annual', 'yearly', 'monthly',
        'billion', 'million', 'trillion',
        'semiconductor', 'chip', 'memory', 'logic', 'analog'
    ]

    for pattern in keyword_patterns:
        if pattern in name_lower:
            keywords.append(pattern)

    return keywords


def extract_numbers_with_context(text: str) -> List[Dict]:
    """
    Extract numbers with surrounding context for matching.

    Hardening fixes (based on latest Germany run):
    - Absolutely prevent 4-digit years from acquiring magnitude units (B/M/K/T) or currency raw formats
    - Prevent '$2022B' artifacts by:
        * only treating '$' as currency if it is immediately adjacent to a non-year number OR the local window contains USD cues
        * never prepending '$' to plain year tokens
    - Infer word scales (million/billion) ONLY for non-year numbers
    - Reject 'times' artifacts like '2-3 times' -> '3t'
    """
    if not text:
        return []

    numbers: List[Dict] = []

    # Numeric token boundaries
    num_token = r"(?<!\d)(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)"

    # Capture optional currency symbol only if it directly precedes the number (no spaces),
    # OR as a separate group so we can validate later.
    # Example accepted: "$190.237", "S$29.8"
    # Example rejected downstream: "$2022" (year-like)
    pattern = rf"(?<!\w)(S\$|\$)?\s*({num_token})\s*(%|T|B|M|K|bn|mn|trillion|billion|million|thousand)?"

    unit_map = {"bn": "B", "mn": "M"}

    def _window(s: str, pos: int, left: int = 100, right: int = 100) -> str:
        return (s[max(0, pos - left):min(len(s), pos + right)] or "")

    def _is_year_token(num_str: str) -> bool:
        return bool(re.fullmatch(r"(19|20)\d{2}", (num_str or "").strip()))

    def _looks_like_times_phrase(s: str, pos: int) -> bool:
        w = _window(s, pos, 35, 35).lower()
        return (" times" in w) or ("x the" in w) or (" x " in w)

    def _detect_word_scale(s: str, pos: int) -> str:
        w = _window(s, pos, 90, 90).lower()
        if "billion" in w:
            return "B"
        if "million" in w:
            return "M"
        if "thousand" in w:
            return "K"
        if "trillion" in w:
            return "T"
        return ""

    def _currency_is_valid(sym: str, num_str: str, pos: int) -> bool:
        """
        '$' is only valid currency if:
        - number is NOT a year token, AND
        - either symbol is 'S$' OR local window suggests money (usd, dollars, sgd, € etc.)
        This prevents '$2022' being treated as money.
        """
        if not sym:
            return False
        if _is_year_token(num_str):
            return False
        if sym == "S$":
            return True
        w = _window(text, pos, 80, 80).lower()
        return any(k in w for k in ["usd", "dollar", "dollars", "us$", "sgd", "€", "eur", "million", "billion", "trillion"])

    def _normalize_unit(u_raw: str) -> str:
        if not u_raw:
            return ""
        u = u_raw.strip()
        u = unit_map.get(u.lower(), u)
        # Word units
        if u.lower() == "billion":
            return "B"
        if u.lower() == "million":
            return "M"
        if u.lower() == "thousand":
            return "K"
        if u.lower() == "trillion":
            return "T"
        return normalize_unit(u)

    for m in re.finditer(pattern, text, re.IGNORECASE):
        sym = (m.group(1) or "").strip()
        num_str = (m.group(2) or "").strip()
        unit_raw = (m.group(3) or "").strip()

        # Context window
        start = max(0, m.start() - 220)
        end = min(len(text), m.end() + 220)
        context = (text[start:end] or "").lower()

        if is_likely_junk_context(context):
            continue

        # Reject times artifacts if unit looks like magnitude
        u_norm = _normalize_unit(unit_raw)
        if u_norm in {"T", "B", "M", "K"} and _looks_like_times_phrase(text, m.start()):
            continue

        # If this is a year token, we do NOT allow:
        # - currency symbols
        # - inferred word scales
        # - explicit magnitude units
        if _is_year_token(num_str):
            # if unit explicitly indicates magnitude, reject
            if u_norm in {"T", "B", "M", "K"}:
                continue
            # drop currency-marked years entirely (stops "$2023" pollution)
            if sym in {"$", "S$"}:
                continue

            # Keep plain year as unitless, but it will usually be filtered later by matching gates
            parsed_year = parse_human_number(num_str, "")
            if parsed_year is None:
                continue
            numbers.append({
                "value": float(parsed_year),
                "unit": "",
                "context": context,
                "raw": num_str
            })
            continue

        # For non-years: infer unit if missing and word-scale nearby
        if not u_norm:
            inferred = _detect_word_scale(text, m.start())
            if inferred:
                u_norm = inferred

        parsed = parse_human_number(num_str, u_norm)
        if parsed is None:
            continue

        # Validate currency symbol usage
        currency_prefix = ""
        if sym and _currency_is_valid(sym, num_str, m.start()):
            currency_prefix = sym

        # Build raw output
        if u_norm == "%":
            raw_out = f"{num_str}%"
        elif u_norm in {"T", "B", "M", "K"}:
            raw_out = f"{currency_prefix}{num_str}{u_norm}".strip()
        else:
            raw_out = f"{currency_prefix}{num_str}".strip()

        # Sanity guard
        if abs(parsed) > 10_000_000:
            continue

        numbers.append({
            "value": float(parsed),
            "unit": u_norm,
            "context": context,
            "raw": raw_out
        })

    return numbers

def extract_numbers_with_context_pdf(text: str) -> List[Dict]:
    """
    PDF-specialized extractor wrapper.

    Strategy:
    - Run normal extraction
    - Add additional filtering aimed at PDF front-matter noise (ISSN/ISBN/doi)
    - Prefer contexts that contain domain-relevant table language (GDP, growth, %, forecast, services, industry, etc.)

    Returns the same schema as extract_numbers_with_context().
    """
    if not text:
        return []

    base = extract_numbers_with_context(text) or []

    def _bad_pdf_context(ctx: str) -> bool:
        c = (ctx or "").lower()
        bad = [
            "issn", "isbn", "doi", "catalogue", "kc-", "legal notice",
            "reproduction is authorised", "all rights reserved", "printed by",
            "manuscript completed", "luxembourg:", "©", "copyright"
        ]
        return any(b in c for b in bad)

    def _good_pdf_context(ctx: str) -> bool:
        c = (ctx or "").lower()
        good = [
            "gdp", "gross domestic product", "growth", "forecast", "projection",
            "services", "industry", "manufacturing", "agriculture", "share",
            "percent", "%", "table", "figure", "chart"
        ]
        return any(g in c for g in good)

    cleaned = []
    for n in base:
        ctx = n.get("context", "") or ""
        if _bad_pdf_context(ctx):
            continue
        # If PDF extraction is still noisy, keep only numbers with "good" context or obvious %/currency markers.
        raw = (n.get("raw") or "").lower()
        u = normalize_unit(n.get("unit", ""))
        if _good_pdf_context(ctx) or u == "%" or any(sym in raw for sym in ["€", "eur", "$", "s$"]):
            cleaned.append(n)

    # If we filtered too hard, fall back to base
    return cleaned if len(cleaned) >= max(10, len(base) // 3) else base


def calculate_context_match(keywords: List[str], context: str) -> float:
    """Calculate how well keywords match the context (deterministic)."""
    if not context:
        return 0.0

    context_lower = context.lower()

    # If no keywords, give a small baseline (we'll rely more on value_score)
    if not keywords:
        return 0.25

    # Year keywords MUST match if present
    year_keywords = [kw for kw in keywords if re.fullmatch(r"20\d{2}", kw)]
    if year_keywords:
        if not any(y in context_lower for y in year_keywords):
            return 0.0

    matches = sum(1 for kw in keywords if kw.lower() in context_lower)

    # Instead of hard "matches < 2 = reject", scale smoothly:
    match_ratio = matches / max(len(keywords), 1)

    # If nothing matches, reject
    if matches == 0:
        return 0.0

    # Score between 0.35 and 1.0 depending on ratio
    return 0.35 + (match_ratio * 0.65)

def render_source_anchored_results(results: Dict, query: str):
    """Render the results of source-anchored evolution analysis"""

    st.header("📈 Source-Anchored Evolution Analysis")
    st.markdown(f"**Question:** {query}")

    # Summary metrics
    summary = results.get("summary", {}) or {}
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Metrics", summary.get("total_metrics", 0))
    col2.metric("Compared", summary.get("metrics_compared", 0))
    col3.metric("Changed", summary.get("metrics_changed", 0))
    col4.metric("Unchanged", summary.get("metrics_unchanged", 0))

    stability_score = results.get("stability_score", None)
    if stability_score is None:
        col5.metric("Stability", "N/A")
    else:
        col5.metric("Stability", f"{stability_score:.0f}%")

    # Interpretation
    interpretation = results.get("interpretation", "")
    if interpretation:
        if "Low confidence" in interpretation or "blocked" in interpretation.lower():
            st.warning(interpretation)
        else:
            st.info(interpretation)

    st.markdown("---")

    # Metric changes
    st.subheader("🧮 Metric Changes")
    changes = results.get("metric_changes", []) or []

    if not changes:
        st.info("No metric comparisons available.")
    else:
        display_rows = []
        for ch in changes:
            prev_val = ch.get("previous_value", "N/A")
            cur_val = ch.get("current_value", "N/A")

            cp = ch.get("change_pct", None)
            if cp is None:
                cp_disp = "N/A"
            else:
                cp_disp = f"{cp:+.1f}%"

            display_rows.append(
                {
                    "Metric": ch.get("metric", "N/A"),
                    "Previous": prev_val,
                    "Current": cur_val,
                    "Change": cp_disp,
                    "Confidence": f"{int(ch.get('match_confidence', 0))}%",
                    "Status": ch.get("status", ""),
                }
            )

        df = pd.DataFrame(display_rows)
        st.dataframe(df, hide_index=True, width="stretch")

    st.markdown("---")

    # Sources checked
    st.subheader("🔗 Sources Checked")
    source_results = results.get("source_results", []) or []

    if not source_results:
        st.info("No source results recorded.")
        return

    # Render status with the newer status codes
    def _status_icon(sr: Dict) -> str:
        status = (sr.get("status") or "").lower()
        numbers_found = int(sr.get("numbers_found") or 0)

        if status == "fetched_extracted" and numbers_found > 0:
            return "✅"
        if status == "fetched_unusable":
            return "⚠️"
        if status == "failed_but_reused_cache":
            return "♻️"
        if status.startswith("failed"):
            return "❌"
        if status.startswith("fetched"):
            return "⚠️"
        return "•"

    for sr in source_results:
        url = sr.get("url", "")
        status = sr.get("status", "unknown")
        detail = sr.get("status_detail", "")
        numbers_found = sr.get("numbers_found", 0)
        icon = _status_icon(sr)

        if url:
            st.markdown(
                f"{icon} **[{url}]({url})**  \n"
                f"- Status: `{status}`  \n"
                f"- Detail: {detail}  \n"
                f"- Numbers found: {numbers_found}"
            )
        else:
            st.markdown(
                f"{icon} **(missing url)**  \n"
                f"- Status: `{status}`  \n"
                f"- Detail: {detail}  \n"
                f"- Numbers found: {numbers_found}"
            )


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

# =========================================================
# 3A. QUESTION CATEGORIZATION + SIGNALS (DETERMINISTIC)
# =========================================================

def categorize_question_signals(query: str, qs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a question_profile used for structured reporting.

    IMPORTANT:
      - category must follow query_structure if provided (single source of truth).
      - signals can be rich, but must not contradict the chosen category.
    """
    qs = qs or {}
    q = (query or "").strip()

    # Prefer category/main/side from query_structure when available
    category = (qs.get("category") or "").strip() or "unknown"
    main_q = (qs.get("main") or "").strip() or q
    side_qs = qs.get("side") if isinstance(qs.get("side"), list) else []

    # Deterministic signals (richer classifier)
    base = classify_question_signals(q) or {}

    # Force category + expected_metric_ids to match query_structure category
    # (but preserve other extracted info like years/regions/intents)
    signals: Dict[str, Any] = {}
    signals["category"] = category

    # Carry over extracted fields
    signals["years"] = base.get("years", []) or []
    signals["regions"] = base.get("regions", []) or []
    signals["intents"] = base.get("intents", []) or []

    # Keep raw signals for debugging
    raw_hits = list(base.get("signals") or [])
    signals["raw_signals"] = raw_hits

    def _signal_consistent_with_category(sig: str, cat: str) -> bool:
        s = (sig or "").lower()
        c = (cat or "").lower()
        if not s:
            return False

        # If final category is country, drop industry/company category-rule strings
        if c == "country":
            if "industry_keywords" in s or "mixed_signals_default_to_industry" in s or "company_keywords" in s:
                return False

        # If final category is industry, drop explicit country-rule strings
        if c == "industry":
            if "macro_outlook_bias_country" in s or "country_keywords" in s:
                return False

        return True

    signals["signals"] = [s for s in raw_hits if _signal_consistent_with_category(s, category)]

    # Expected metric IDs: always determined by the final category, then lightly enriched by intents (optional)
    expected_metric_ids: List[str] = []
    try:
        expected_metric_ids = get_expected_metric_ids_for_category(category) or []
    except Exception:
        expected_metric_ids = []

    # Optional: enrich with intent-based suggestions (won't remove anything)
    intent_metric_suggestions = {
        "market_size": ["market_size", "market_size_2024", "market_size_2025"],
        "growth_forecast": ["cagr", "market_size_2030"],
        "competitive_landscape": ["market_share", "top_players"],
        "pricing": ["avg_price", "asp"],
        "consumer_demand": ["users", "penetration", "arpu"],
        "supply_chain": ["capacity", "shipments"],
        "investment": ["capex", "profit", "ebitda"],
        "macro_outlook": ["gdp", "inflation", "interest_rate", "exchange_rate"],
    }

    intents = signals.get("intents") or []
    for intent in intents:
        for mid in intent_metric_suggestions.get(intent, []):
            if mid not in expected_metric_ids:
                expected_metric_ids.append(mid)

    signals["expected_metric_ids"] = expected_metric_ids

    profile: Dict[str, Any] = {
        "category": category,
        "signals": signals,
        "main_question": main_q,
        "side_questions": side_qs,
    }

    # Keep debug for traceability
    if qs.get("debug") is not None:
        profile["debug_query_structure"] = qs.get("debug")

    return profile


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

    # -------------------------
    # Parse primary response
    # -------------------------
    try:
        data = json.loads(primary_json)
    except Exception as e:
        st.error(f"❌ Cannot render dashboard: {e}")
        st.code(primary_json[:1000])
        return

    # -------------------------
    # Local helper: metric formatting (unit-safe)
    # -------------------------
    def _format_metric_value(m: Dict) -> str:
        """
        Format metric values cleanly:
        - Currency before number: $204.7B, S$29.8B
        - Compact units (B, M, K)
        - Proper separators
        """
        if not isinstance(m, dict):
            return "N/A"

        val = m.get("value")
        unit = (m.get("unit") or "").strip()

        if val is None or val == "":
            return "N/A"

        try:
            num = float(str(val).replace(",", ""))
        except Exception:
            return f"{val}{unit}".strip()

        unit = unit.replace(" ", "")
        currency_prefix = ""

        if unit.upper().startswith("S$"):
            currency_prefix = "S$"
            unit = unit[2:]
        elif unit.upper().startswith("$"):
            currency_prefix = "$"
            unit = unit[1:]
        elif unit.upper().startswith("USD"):
            currency_prefix = "$"
            unit = unit.replace("USD", "")
        elif unit.upper().startswith("SGD"):
            currency_prefix = "S$"
            unit = unit.replace("SGD", "")

        if unit.upper() == "B":
            formatted = f"{num:.2f}".rstrip("0").rstrip(".") + "B"
        elif unit.upper() == "M":
            formatted = f"{num:.2f}".rstrip("0").rstrip(".") + "M"
        elif unit.upper() == "K":
            formatted = f"{num:.2f}".rstrip("0").rstrip(".") + "K"
        elif unit == "%":
            return f"{num:.1f}%"
        else:
            formatted = f"{int(num):,}" if abs(num) >= 1000 else f"{num:g}"
            if unit:
                formatted = f"{formatted} {unit}"

        return f"{currency_prefix}{formatted}".strip()

    # -------------------------
    # Header & confidence
    # -------------------------
    st.header("📊 Yureeka Market Report")
    st.markdown(f"**Question:** {user_question}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Confidence", f"{final_conf:.1f}%")
    col2.metric("Base Model", f"{base_conf:.1f}%")
    col3.metric(
        "Evidence",
        f"{veracity_scores.get('overall', 0):.1f}%" if veracity_scores else "N/A"
    )

    st.markdown("---")

    # -------------------------
    # Executive Summary
    # -------------------------
    st.subheader("📋 Executive Summary")
    st.markdown(f"**{data.get('executive_summary', 'No summary available')}**")
    st.markdown("---")

    # -------------------------
    # Key Metrics
    # -------------------------
    st.subheader("💰 Key Metrics")
    metrics = data.get("primary_metrics") or {}

    question_category = data.get("question_category") or (data.get("question_profile", {}) or {}).get("category")
    question_signals = data.get("question_signals") or (data.get("question_profile", {}) or {}).get("signals", {})
    side_questions = data.get("side_questions") or (data.get("question_profile", {}) or {}).get("side_questions", [])
    expected_ids = data.get("expected_metric_ids") or question_signals.get("expected_metric_ids", [])

    rows: List[Dict[str, str]] = []

    if question_category:
        rows.append({"Metric": "Question Category", "Value": str(question_category)})

    if isinstance(question_signals, dict):
        if question_signals.get("years"):
            rows.append({"Metric": "Years (detected)", "Value": ", ".join(map(str, question_signals["years"]))})
        if question_signals.get("regions"):
            rows.append({"Metric": "Regions (detected)", "Value": ", ".join(map(str, question_signals["regions"]))})

    if side_questions:
        rows.append({"Metric": "Side Questions", "Value": "; ".join(map(str, side_questions))})

    try:
        canon = canonicalize_metrics(metrics) if isinstance(metrics, dict) else {}
    except Exception:
        canon = metrics or {}

    by_base: Dict[str, List[Dict]] = {}
    for cid, m in (canon or {}).items():
        base = re.sub(r'_\d{4}(?:_\d{4})*$', '', str(cid))
        by_base.setdefault(base, []).append(m)

    if expected_ids:
        for base_id in expected_ids:
            mlist = by_base.get(base_id)
            if mlist:
                rows.append({"Metric": mlist[0].get("name", base_id), "Value": _format_metric_value(mlist[0])})
            else:
                rows.append({"Metric": base_id.replace("_", " ").title(), "Value": "N/A"})

    for cid, m in (canon or {}).items():
        base = re.sub(r'_\d{4}(?:_\d{4})*$', '', str(cid))
        if expected_ids and base in expected_ids:
            continue
        rows.append({"Metric": m.get("name", cid), "Value": _format_metric_value(m)})

    if rows:
        st.table(pd.DataFrame(rows[:10]))
        if len(rows) > 10:
            with st.expander(f"Show all key metrics ({len(rows)})"):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No metrics available")

    st.markdown("---")

    # -------------------------
    # Key Findings
    # -------------------------
    st.subheader("🔍 Key Findings")
    findings = data.get("key_findings") or []
    for i, f in enumerate(findings[:8], 1):
        if f:
            st.markdown(f"**{i}.** {f}")

    if len(findings) > 8:
        with st.expander(f"Show all key findings ({len(findings)})"):
            for i, f in enumerate(findings, 1):
                if f:
                    st.markdown(f"**{i}.** {f}")

    st.markdown("---")

    # -------------------------
    # Top Entities
    # -------------------------
    entities = data.get("top_entities") or []
    if entities:
        st.subheader("🏢 Top Market Players")
        df_ent = pd.DataFrame([
            {
                "Entity": e.get("name", "N/A"),
                "Share": e.get("share", "N/A"),
                "Growth": e.get("growth", "N/A"),
            }
            for e in entities if isinstance(e, dict)
        ])

        if not df_ent.empty:
            st.dataframe(df_ent.head(8), hide_index=True, width="stretch")
            if len(df_ent) > 8:
                with st.expander(f"Show all market players ({len(df_ent)})"):
                    st.dataframe(df_ent, hide_index=True, width="stretch")

    st.markdown("---")

    # -------------------------
    # Trends & Forecast
    # -------------------------
    trends = data.get("trends_forecast") or []
    if trends:
        st.subheader("📈 Trends & Forecast")
        df_trends = pd.DataFrame([
            {
                "Trend": t.get("trend", "N/A"),
                "Direction": t.get("direction", "→"),
                "Timeline": t.get("timeline", "N/A"),
            }
            for t in trends if isinstance(t, dict)
        ])
        st.table(df_trends.head(8))
        if len(df_trends) > 8:
            with st.expander(f"Show all trends ({len(df_trends)})"):
                st.dataframe(df_trends, hide_index=True, width="stretch")

    st.markdown("---")

    # -------------------------
    # Visualization
    # -------------------------
    st.subheader("📊 Data Visualization")
    viz = data.get("visualization_data")

    if isinstance(viz, dict):
        labels = viz.get("chart_labels") or []
        values = viz.get("chart_values") or []
        title = viz.get("chart_title", "Trend Analysis")
        chart_type = viz.get("chart_type", "line")

        if labels and values and len(labels) == len(values):
            try:
                df_viz = pd.DataFrame({"x": labels[:10], "y": [float(v) for v in values[:10]]})
                fig = px.bar(df_viz, x="x", y="y", title=title) if chart_type == "bar" else px.line(
                    df_viz, x="x", y="y", title=title, markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"⚠️ Chart rendering failed: {e}")
        else:
            st.info("📊 Visualization data incomplete")
    else:
        st.info("📊 No visualization data available")

    st.markdown("---")

    # -------------------------
    # Sources
    # -------------------------
    st.subheader("🔗 Sources & Reliability")
    sources = data.get("sources") or (web_context.get("sources") if isinstance(web_context, dict) else []) or []

    if not sources:
        st.info("No sources found")
    else:
        st.success(f"📊 Found {len(sources)} sources")
        cols = st.columns(2)
        for i, s in enumerate(sources[:10], 1):
            col = cols[(i - 1) % 2]
            short = s[:60] + "..." if len(s) > 60 else s
            col.markdown(
                f"**{i}.** [{short}]({s})<br><small>{classify_source_reliability(str(s))}</small>",
                unsafe_allow_html=True
            )

        if len(sources) > 10:
            with st.expander(f"Show all sources ({len(sources)})"):
                st.dataframe(pd.DataFrame({
                    "#": range(1, len(sources) + 1),
                    "Source": sources,
                    "Reliability": [classify_source_reliability(str(s)) for s in sources],
                }), hide_index=True, width="stretch")

    st.markdown("---")

    # -------------------------
    # Veracity Scores
    # -------------------------
    if veracity_scores:
        st.subheader("✅ Evidence Quality Scores")
        cols = st.columns(5)
        for i, (label, key) in enumerate([
            ("Sources", "source_quality"),
            ("Numbers", "numeric_consistency"),
            ("Citations", "citation_density"),
            ("Consensus", "source_consensus"),
            ("Overall", "overall"),
        ]):
            cols[i].metric(label, f"{veracity_scores.get(key, 0):.0f}%")


def render_native_comparison(baseline: Dict, compare: Dict):
    """Render a clean comparison between two analyses"""

    st.header("📊 Analysis Comparison")

    # Time info
    baseline_time = baseline.get('timestamp', '')
    compare_time = compare.get('timestamp', '')

    try:
        baseline_dt = datetime.fromisoformat(baseline_time.replace('Z', '+00:00'))
        compare_dt = datetime.fromisoformat(compare_time.replace('Z', '+00:00'))
        delta = compare_dt - baseline_dt
        if delta.days > 0:
            delta_str = f"{delta.days}d {delta.seconds // 3600}h"
        else:
            delta_str = f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
    except:
        delta_str = "Unknown"

    # Overview row
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline", baseline_time[:16] if baseline_time else "N/A")
    col2.metric("Current", compare_time[:16] if compare_time else "N/A")
    col3.metric("Time Delta", delta_str)

    st.markdown("---")

    # Extract metrics
    baseline_metrics = baseline.get('primary_response', {}).get('primary_metrics', {})
    compare_metrics = compare.get('primary_response', {}).get('primary_metrics', {})

    # Build metric diff table
    st.subheader("💰 Metric Changes")

    diff_rows = []
    stability_count = 0
    total_count = 0

    # Canonicalize metrics for stable matching
    baseline_canonical = canonicalize_metrics(baseline_metrics)
    compare_canonical = canonicalize_metrics(compare_metrics)

    # Build lookup by canonical ID
    baseline_by_id = {}
    compare_by_id = {}

    for cid, m in baseline_canonical.items():
        baseline_by_id[cid] = m

    for cid, m in compare_canonical.items():
        compare_by_id[cid] = m

    all_ids = set(baseline_by_id.keys()).intersection(compare_by_id.keys())

    for cid in sorted(all_ids):
        baseline_m = baseline_by_id.get(cid)
        compare_m = compare_by_id.get(cid)

        # Use canonical name for display, fallback to original
        display_name = cid
        if baseline_m and baseline_m.get('name'):
            display_name = baseline_m['name']


        if baseline_m and compare_m:
            old_val = baseline_m.get('value', 'N/A')
            new_val = compare_m.get('value', 'N/A')
            unit = compare_m.get('unit', baseline_m.get('unit', ''))

            old_num = parse_to_float(old_val)
            new_num = parse_to_float(new_val)

            if old_num is not None and new_num is not None and old_num != 0:
                change_pct = ((new_num - old_num) / abs(old_num)) * 100

                if abs(change_pct) < 1:
                    icon, reason = "➡️", "No change"
                    stability_count += 1
                elif abs(change_pct) < 5:
                    icon, reason = "➡️", "Minor change"
                    stability_count += 1
                elif change_pct > 0:
                    icon, reason = "📈", "Increased"
                else:
                    icon, reason = "📉", "Decreased"

                delta_str = f"{change_pct:+.1f}%"
            else:
                icon, delta_str, reason = "➡️", "-", "Non-numeric"
                stability_count += 1

            diff_rows.append({
                '': icon,
                'Metric': display_name,
                'Old': _fmt_currency_first(str(old_val), str(unit)),
                'New': _fmt_currency_first(str(new_val), str(unit)),
                'Δ': delta_str,
                'Reason': reason
            })
            total_count += 1

        elif baseline_m:
            old_val = baseline_m.get('value', 'N/A')
            unit = baseline_m.get('unit', '')
            diff_rows.append({
                '': '❌',
                'Metric': display_name,
                'Old': f"{old_val} {unit}".strip(),
                'New': '-',
                'Δ': '-',
                'Reason': 'Removed'
            })
            total_count += 1
        else:
            new_val = compare_m.get('value', 'N/A')
            unit = compare_m.get('unit', '')
            diff_rows.append({
                '': '🆕',
                'Metric': display_name,
                'Old': '-',
                'New': f"{new_val} {unit}".strip(),
                'Δ': '-',
                'Reason': 'New'
            })
            total_count += 1

    if diff_rows:
        st.dataframe(pd.DataFrame(diff_rows), hide_index=True, use_container_width=True)

        # Show canonical ID mapping for debugging
        with st.expander("🔧 Canonical ID Mapping (Debug)"):
            st.write("**How metrics were matched:**")

            baseline_canonical = canonicalize_metrics(baseline_metrics)
            compare_canonical = canonicalize_metrics(compare_metrics)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Baseline Metrics:**")
                for cid, m in baseline_canonical.items():
                    original = m.get('original_name', 'N/A')
                    canonical = m.get('name', 'N/A')
                    st.caption(f"`{cid}`")
                    st.write(f"  {original} → {canonical}")

            with col2:
                st.write("**Current Metrics:**")
                for cid, m in compare_canonical.items():
                    original = m.get('original_name', 'N/A')
                    canonical = m.get('name', 'N/A')
                    st.caption(f"`{cid}`")
                    st.write(f"  {original} → {canonical}")
    else:
        st.info("No metrics to compare")

    # Stability score
    stability_pct = (stability_count / total_count * 100) if total_count > 0 else 100

    st.markdown("---")
    st.subheader("📊 Stability Score")

    col1, col2, col3 = st.columns(3)
    col1.metric("Stable Metrics", f"{stability_count}/{total_count}")
    col2.metric("Stability", f"{stability_pct:.0f}%")

    if stability_pct >= 80:
        col3.success("🟢 Highly Stable")
    elif stability_pct >= 60:
        col3.warning("🟡 Moderate Changes")
    else:
        col3.error("🔴 Significant Drift")

    # Confidence comparison
    st.markdown("---")
    st.subheader("🎯 Confidence Change")

    col1, col2, col3 = st.columns(3)
    baseline_conf = baseline.get('final_confidence', 0)
    compare_conf = compare.get('final_confidence', 0)
    conf_change = compare_conf - baseline_conf if isinstance(baseline_conf, (int, float)) and isinstance(compare_conf, (int, float)) else 0

    col1.metric("Baseline", f"{baseline_conf:.1f}%" if isinstance(baseline_conf, (int, float)) else "N/A")
    col2.metric("Current", f"{compare_conf:.1f}%" if isinstance(compare_conf, (int, float)) else "N/A")
    col3.metric("Change", f"{conf_change:+.1f}%")

    # Download comparison
    st.markdown("---")
    comparison_output = {
        "comparison_timestamp": datetime.now().isoformat(),
        "baseline": baseline,
        "current": compare,
        "stability_score": stability_pct,
        "metrics_compared": total_count,
        "metrics_stable": stability_count
    }

    st.download_button(
        label="💾 Download Comparison Report",
        data=json.dumps(comparison_output, indent=2, ensure_ascii=False).encode('utf-8'),
        file_name=f"yureeka_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

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
        query = st.text_input(
            "Enter your question about markets, industries, finance, or economics:",
            placeholder="e.g., What is the size of the global EV battery market?"
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_web = st.checkbox(
                "Enable web search (recommended)",
                value=bool(SERPAPI_KEY),
                disabled=not SERPAPI_KEY
            )

        if st.button("🔍 Analyze", type="primary") and query:
            if len(query.strip()) < 5:
                st.error("❌ Please enter a question with at least 5 characters")
                return

            query = query.strip()[:500]

            query_structure = extract_query_structure(query) or {}
            question_profile = categorize_question_signals(query, qs=query_structure)
            question_signals = question_profile.get("signals", {}) or {}

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

            with st.spinner("🤖 Analyzing query..."):
                primary_response = query_perplexity(query, web_context, query_structure=query_structure)

            if not primary_response:
                st.error("❌ Primary model failed to respond")
                return

            try:
                primary_data = json.loads(primary_response)
            except Exception as e:
                st.error(f"❌ Failed to parse primary response: {e}")
                st.code(primary_response[:1000])
                return

            with st.spinner("✅ Verifying evidence quality..."):
                veracity_scores = evidence_based_veracity(primary_data, web_context)

            base_conf = float(primary_data.get("confidence", 75))
            final_conf = calculate_final_confidence(base_conf, veracity_scores.get("overall", 0))

            # Optional: canonicalize + attribution + schema freeze (only if your codebase defines these)
            try:
                if primary_data.get("primary_metrics"):
                    primary_data["primary_metrics_canonical"] = canonicalize_metrics(
                        primary_data.get("primary_metrics", {}),
                        merge_duplicates_to_range=True,
                        question_text=query,
                        category_hint=str(primary_data.get("question_category", ""))
                    )
                if primary_data.get("primary_metrics_canonical"):
                    primary_data["primary_metrics_canonical"] = add_range_and_source_attribution_to_canonical_metrics(
                        primary_data.get("primary_metrics_canonical", {}),
                        web_context
                    )
                if primary_data.get("primary_metrics_canonical"):
                    primary_data["metric_schema_frozen"] = freeze_metric_schema(
                        primary_data["primary_metrics_canonical"]
                    )
            except Exception:
                pass

            # Hash key findings (optional)
            try:
                if primary_data.get("key_findings"):
                    findings_with_hash = []
                    for finding in primary_data.get("key_findings", []):
                        if finding:
                            findings_with_hash.append({
                                "text": finding,
                                "semantic_hash": compute_semantic_hash(finding)
                            })
                    primary_data["key_findings_hashed"] = findings_with_hash
            except Exception:
                pass

            # Build output
            output = {
                "question": query,
                "question_profile": question_profile,
                "question_category": question_profile.get("category"),
                "question_signals": question_signals,
                "side_questions": question_profile.get("side_questions", []),
                "timestamp": now_utc().isoformat(),
                "primary_response": primary_data,
                "final_confidence": final_conf,
                "veracity_scores": veracity_scores,
                "web_sources": web_context.get("sources", []),
            }

            # Save baseline numeric cache if available (optional; your codebase may already do this)
            try:
                baseline_sources_cache = []
                scraped = (web_context or {}).get("scraped_content") or {}
                for url, content in scraped.items():
                    if not content:
                        continue
                    nums = extract_numbers_with_context(content)
                    baseline_sources_cache.append({
                        "url": url,
                        "fetched_at": now_utc().isoformat(),
                        "fingerprint": fingerprint_text(content),
                        "extracted_numbers": [
                            {
                                "value": n.get("value"),
                                "unit": n.get("unit"),
                                "raw": n.get("raw"),
                                "context_snippet": (n.get("context") or "")[:200]
                            }
                            for n in (nums or [])
                        ]
                    })
                if baseline_sources_cache:
                    output["baseline_sources_cache"] = baseline_sources_cache
            except Exception:
                pass

            with st.spinner("💾 Saving to history..."):
                if add_to_history(output):
                    st.success("✅ Analysis saved to Google Sheets")
                else:
                    st.warning("⚠️ Saved to session only (Google Sheets unavailable)")

            json_bytes = json.dumps(output, indent=2, ensure_ascii=False).encode("utf-8")
            filename = f"yureeka_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            st.download_button(
                label="💾 Download Analysis JSON",
                data=json_bytes,
                file_name=filename,
                mime="application/json"
            )

            render_dashboard(
                primary_response,
                final_conf,
                web_context,
                base_conf,
                query,
                veracity_scores,
                web_context.get("source_reliability", [])
            )

            with st.expander("🔧 Debug Information"):
                st.write("**Confidence Breakdown:**")
                st.json({
                    "base_confidence": base_conf,
                    "evidence_score": veracity_scores.get("overall", 0),
                    "final_confidence": final_conf,
                    "veracity_breakdown": veracity_scores
                })
                st.write("**Primary Model Response:**")
                st.json(primary_data)

    # =====================
    # TAB 2: EVOLUTION ANALYSIS
    # =====================
    with tab2:
        st.markdown("""
        ### 📈 Track the evolution of key metrics over time using **deterministic source-anchored analysis**.

        **How it works:**
        - Select a baseline from your history (stored in Google Sheets)
        - Re-fetches the **exact same sources** from that analysis
        - Extracts current numbers using regex (no LLM variance)
        - Computes deterministic diffs with context-aware matching
        """)

        with st.sidebar:
            st.subheader("📚 History")

            if st.button("🔄 Refresh"):
                st.cache_resource.clear()
                st.rerun()

            sheet = get_google_sheet()
            if sheet:
                st.success("✅ Google Sheets connected")
            else:
                st.warning("⚠️ Using session storage")

        # ✅ FIX: your codebase uses get_history(), not load_history()
        history = get_history()

        if not history:
            st.info("📭 No previous analyses found. Run an analysis in the 'New Analysis' tab first.")
            return

        baseline_options = [
            f"{i+1}. {h.get('question', 'N/A')}  ({h.get('timestamp', '')})"
            for i, h in enumerate(history)
        ]
        baseline_choice = st.selectbox("Select baseline analysis:", baseline_options)
        baseline_idx = int(baseline_choice.split(".")[0]) - 1
        baseline_data = history[baseline_idx]

        compare_method = st.selectbox(
            "Comparison method:",
            [
                "source-anchored evolution (re-fetch same sources)",
                "another saved analysis (deterministic)",
                "fresh analysis (volatile)"
            ]
        )

        compare_data = None
        if "another saved analysis" in compare_method:
            compare_options = [
                f"{i+1}. {h.get('question', 'N/A')}  ({h.get('timestamp', '')})"
                for i, h in enumerate(history) if i != baseline_idx
            ]
            if compare_options:
                compare_choice = st.selectbox("Select comparison analysis:", compare_options)
                compare_idx = int(compare_choice.split(".")[0]) - 1
                compare_data = history[compare_idx]
            else:
                st.warning("No other saved analyses to compare with.")

        st.markdown("---")

        if st.button("🧬 Run Evolution Analysis", type="primary"):

            if "source-anchored evolution" in compare_method:
                evolution_query = baseline_data.get("question", "")
                if not evolution_query:
                    st.error("❌ No question found in baseline.")
                    return

                with st.spinner("🧬 Running source-anchored evolution..."):
                    try:
                        results = run_source_anchored_evolution(baseline_data)
                    except Exception as e:
                        st.error(f"❌ Evolution failed: {e}")
                        return

                interpretation = ""
                try:
                    if results and isinstance(results, dict):
                        interpretation = results.get("interpretation", "") or ""
                except Exception:
                    interpretation = ""

                evolution_output = {
                    "question": evolution_query,
                    "timestamp": datetime.now().isoformat(),
                    "analysis_type": "source_anchored",
                    "previous_timestamp": baseline_data.get("timestamp"),
                    "results": results,
                    "interpretation": {
                        "text": interpretation,
                        "authoritative": False,
                        "source": "llm_optional"
                    }
                }

                st.download_button(
                    label="💾 Download Evolution Report",
                    data=json.dumps(evolution_output, indent=2, ensure_ascii=False).encode("utf-8"),
                    file_name=f"yureeka_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

                # ✅ FIX: guarded renderer to avoid stability_score=None formatting crashes
                render_source_anchored_results(results, evolution_query)

            elif "another saved analysis" in compare_method:
                if compare_data:
                    st.success("✅ Comparing two saved analyses (deterministic)")
                    render_native_comparison(baseline_data, compare_data)
                else:
                    st.error("❌ Please select a comparison analysis")

            else:
                st.warning("⚠️ Running fresh analysis - results may vary")

                query = baseline_data.get("question", "")
                if not query:
                    st.error("❌ No query found")
                    return

                with st.spinner("🌐 Fetching current data..."):
                    web_context = fetch_web_context(query, num_sources=3)

                if not web_context:
                    web_context = {
                        "search_results": [],
                        "scraped_content": {},
                        "summary": "",
                        "sources": [],
                        "source_reliability": []
                    }

                with st.spinner("🤖 Running analysis..."):
                    new_response = query_perplexity(query, web_context)

                if new_response:
                    try:
                        new_parsed = json.loads(new_response)
                        veracity = evidence_based_veracity(new_parsed, web_context)
                        base_conf = float(new_parsed.get("confidence", 75))
                        final_conf = calculate_final_confidence(base_conf, veracity.get("overall", 0))

                        compare_data = {
                            "question": query,
                            "timestamp": datetime.now().isoformat(),
                            "primary_response": new_parsed,
                            "final_confidence": final_conf,
                            "veracity_scores": veracity,
                            "web_sources": web_context.get("sources", [])
                        }

                        add_to_history(compare_data)
                        st.success("✅ Saved to history")

                        render_native_comparison(baseline_data, compare_data)
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
                else:
                    st.error("❌ Analysis failed")

if __name__ == "__main__":
    main()
