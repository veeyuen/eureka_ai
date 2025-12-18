# =========================================================
# YUREEKA AI RESEARCH ASSISTANT v7.9
# With Web Search, Evidence-Based Verification, Confidence Scoring
# SerpAPI Output with Evolution Layer Version
# Updated SerpAPI parameters for stable output
# Deterministic Output From LLM
# Deterministic Evolution Core Using Python Diff Engine
# Anchored Evolution Analysis Using JSON As Input Into Model
# Implementation of Source-Based Evolution
# Saving of JSON output Files into Google Sheets
# Canonical Metric Registry + Semantic Hashing of Findings
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
import gspread
import google.generativeai as genai
from google.oauth2.service_account import Credentials
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime, timedelta
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
    }
}

# Year extraction pattern
YEAR_PATTERN = re.compile(r'(20\d{2})')


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


def canonicalize_metrics(metrics: Dict) -> Dict:
    """
    Convert all metrics to use canonical IDs.

    Input: {"metric1": {"name": "2024 Market Size", "value": 100, "unit": "B"}}
    Output: {"market_size_2024": {"name": "Market Size (2024)", "value": 100, "unit": "B",
             "canonical_id": "market_size_2024", "original_name": "2024 Market Size"}}
    """
    canonicalized = {}

    for key, metric in metrics.items():
        if not isinstance(metric, dict):
            continue

        original_name = metric.get('name', key)
        canonical_id, canonical_name = get_canonical_metric_id(original_name)

        # Handle duplicate canonical IDs by appending suffix
        final_id = canonical_id
        suffix = 1
        while final_id in canonicalized:
            final_id = f"{canonical_id}_{suffix}"
            suffix += 1

        canonicalized[final_id] = {
            **metric,
            "name": canonical_name,
            "canonical_id": final_id,
            "original_name": original_name
        }

    return canonicalized


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
    """
    diffs = []

    # Canonicalize both metric sets
    old_canonical = canonicalize_metrics(old_metrics)
    new_canonical = canonicalize_metrics(new_metrics)

    matched_new_ids = set()

    # Match by canonical ID
    for old_id, old_m in old_canonical.items():
        old_name = old_m.get('name', old_id)
        old_raw = str(old_m.get('value', ''))
        old_unit = old_m.get('unit', '')
        old_val = parse_to_float(old_m.get('value'))

        # Direct canonical ID match
        if old_id in new_canonical:
            new_m = new_canonical[old_id]
            matched_new_ids.add(old_id)

            new_raw = str(new_m.get('value', ''))
            new_val = parse_to_float(new_m.get('value'))
            new_unit = new_m.get('unit', old_unit)

            change_pct = compute_percent_change(old_val, new_val)

            if change_pct is None or abs(change_pct) < 0.5:
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
            # Try base ID match (without year suffix)
            base_id = re.sub(r'_\d{4}(?:_\d{4})*$', '', old_id)
            found = False

            for new_id, new_m in new_canonical.items():
                if new_id in matched_new_ids:
                    continue
                new_base_id = re.sub(r'_\d{4}(?:_\d{4})*$', '', new_id)

                if base_id == new_base_id:
                    matched_new_ids.add(new_id)
                    found = True

                    new_raw = str(new_m.get('value', ''))
                    new_val = parse_to_float(new_m.get('value'))
                    new_unit = new_m.get('unit', old_unit)

                    change_pct = compute_percent_change(old_val, new_val)

                    if change_pct is None or abs(change_pct) < 0.5:
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
                    break

            if not found:
                # Metric removed
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
    for new_id, new_m in new_canonical.items():
        if new_id not in matched_new_ids:
            new_name = new_m.get('name', new_id)
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
        for m in diff.metric_diffs:
            icon = {
                'increased': '📈', 'decreased': '📉', 'unchanged': '➡️',
                'added': '🆕', 'removed': '❌'
            }.get(m.change_type, '•')

            change_str = f"{m.change_pct:+.1f}%" if m.change_pct is not None else "-"

            metric_rows.append({
                "": icon,
                "Metric": m.name,
                "Previous": m.old_raw or "-",
                "Current": m.new_raw or "-",
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
# =========================================================

def fetch_url_content(url: str) -> Optional[str]:
    """Fetch content from a specific URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        clean_text = ' '.join(line for line in lines if line)

        return clean_text[:5000]
    except:
        return None

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

def compute_source_anchored_diff(previous_data: Dict) -> Dict:
    """
    Re-fetch the SAME sources from previous analysis and extract current numbers.
    Uses context-aware matching to reduce false positives.
    """
    prev_response = previous_data.get('primary_response', {})
    prev_sources = previous_data.get('web_sources', []) or prev_response.get('sources', [])

    if not prev_sources:
        return {
            'status': 'no_sources',
            'message': 'No sources found in previous analysis. Please run a new analysis first.',
            'metric_changes': [],
            'source_results': []
        }

    sources_to_check = prev_sources[:5]

    # Extract previous metrics with context keywords
    prev_metrics = prev_response.get('primary_metrics', {})
    prev_numbers = {}
    for key, metric in prev_metrics.items():
        if isinstance(metric, dict):
            val = parse_to_float(metric.get('value'))
            if val is not None:
                metric_name = metric.get('name', key)

                # Extract context keywords from metric name
                keywords = extract_context_keywords(metric_name)

                prev_numbers[metric_name] = {
                    'value': val,
                    'unit': metric.get('unit', ''),
                    'raw': str(metric.get('value', '')),
                    'keywords': keywords
                }

    # Re-fetch sources and extract numbers WITH context
    source_results = []
    all_current_numbers = []

    for url in sources_to_check:
        content = fetch_url_content(url)

        if content:
            numbers = extract_numbers_with_context(content)
            source_results.append({'url': url, 'status': 'fetched', 'numbers_found': len(numbers)})
            all_current_numbers.extend(numbers)
        else:
            source_results.append({'url': url, 'status': 'failed', 'numbers_found': 0})

    # Match metrics using context-aware matching
    metric_changes = []

    for metric_name, prev_data_item in prev_numbers.items():
        prev_val = prev_data_item['value']
        prev_unit = prev_data_item['unit']
        prev_keywords = prev_data_item['keywords']

        best_match = None
        best_match_score = 0

        for curr_num in all_current_numbers:
            # Skip if units don't match (when both have units)
            if prev_unit and curr_num['unit']:
                prev_unit_clean = prev_unit.replace('$', '').replace(' ', '').upper()
                curr_unit_clean = curr_num['unit'].upper()
                if prev_unit_clean and curr_unit_clean and prev_unit_clean[0] != curr_unit_clean[0]:
                    continue

            # Check value similarity (must be within 50% for initial filter)
            if prev_val > 0 and curr_num['value'] > 0:
                ratio = curr_num['value'] / prev_val
                if not (0.5 <= ratio <= 2.0):
                    continue

                # Base score from value similarity
                value_score = 1 / (1 + abs(ratio - 1))

                # Context keyword matching score
                context_score = calculate_context_match(prev_keywords, curr_num['context'])

                # Combined score: 40% value similarity, 60% context match
                combined_score = (value_score * 0.4) + (context_score * 0.6)

                if combined_score > best_match_score:
                    best_match_score = combined_score
                    best_match = curr_num

        # Only accept matches with confidence > 60%
        if best_match and best_match_score > 0.6:
            change_pct = compute_percent_change(prev_val, best_match['value'])

            # More lenient threshold: < 5% change = unchanged
            if change_pct is None or abs(change_pct) < 5:
                change_type = 'unchanged'
            elif change_pct > 0:
                change_type = 'increased'
            else:
                change_type = 'decreased'

            metric_changes.append({
                'name': metric_name,
                'previous_value': prev_data_item['raw'],
                'current_value': best_match['raw'],
                'change_pct': change_pct,
                'change_type': change_type,
                'match_confidence': round(best_match_score * 100, 1),
                'context_snippet': best_match['context'][:100]
            })
        else:
            metric_changes.append({
                'name': metric_name,
                'previous_value': prev_data_item['raw'],
                'current_value': 'Not found (no confident match)',
                'change_pct': None,
                'change_type': 'not_found',
                'match_confidence': round(best_match_score * 100, 1) if best_match else 0,
                'context_snippet': ''
            })

    # Calculate stability - include small changes as "stable"
   # found_count = sum(1 for m in metric_changes if m['change_type'] != 'not_found')
   # stable_count = sum(1 for m in metric_changes if m['change_type'] in ['unchanged'] or
    #                   (m['change_pct'] is not None and abs(m['change_pct']) < 10))

   # stability = (stable_count / len(metric_changes)) * 100 if metric_changes else 100

    # In compute_source_anchored_diff, replace stability calculation:

    # Calculate stability - be more lenient
    # "Stable" = unchanged OR small change (< 10%) OR not found (can't compare)
    # Calculate stability - be more lenient
    found_count = sum(1 for m in metric_changes if m['change_type'] != 'not_found')
    unchanged_count = sum(1 for m in metric_changes if m['change_type'] == 'unchanged')
    stable_count = sum(1 for m in metric_changes if
                       m['change_type'] == 'unchanged' or
                       (m['change_pct'] is not None and abs(m['change_pct']) < 10))

    if found_count > 0:
        stability = (stable_count / found_count) * 100
    else:
        stability = 100  # No metrics found = can't determine instability

    return {
        'status': 'success',
        'sources_checked': len(sources_to_check),
        'sources_fetched': sum(1 for s in source_results if s['status'] == 'fetched'),
        'source_results': source_results,
        'metric_changes': metric_changes,
        'stability_score': round(stability, 1),
        'summary': {
            'total_metrics': len(metric_changes),
            'metrics_found': found_count,
            'metrics_unchanged': unchanged_count,
            'metrics_stable': stable_count,
            'metrics_increased': sum(1 for m in metric_changes if m['change_type'] == 'increased'),
            'metrics_decreased': sum(1 for m in metric_changes if m['change_type'] == 'decreased'),
            'metrics_not_found': sum(1 for m in metric_changes if m['change_type'] == 'not_found'),
        }
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
    """Extract numbers with surrounding context for better matching"""
    if not text:
        return []

    numbers = []
    # Pattern to match numbers with optional currency/unit
    pattern = r'(\$?\d+(?:\.\d+)?)\s*(trillion|billion|million|%|T|B|M)?'

    for match in re.finditer(pattern, text, re.IGNORECASE):
        value_str = match.group(1).replace('$', '').replace(',', '')
        unit = match.group(2) or ''

        try:
            value = float(value_str)
        except:
            continue

        # Skip unlikely values
        if value == 0 or value > 10000000:
            continue

        # Get larger context window (150 chars before and after)
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
            'raw': f"{value_str}{unit}".strip()
        })

    return numbers


def calculate_context_match(keywords: List[str], context: str) -> float:
    """Calculate how well keywords match the context"""
    if not keywords or not context:
        return 0.3  # Base score when no keywords

    context_lower = context.lower()

    # CRITICAL: Year keywords MUST match if present
    year_keywords = [kw for kw in keywords if re.match(r'20\d{2}', kw)]
    if year_keywords:
        year_found = any(year in context_lower for year in year_keywords)
        if not year_found:
            return 0.0  # Reject match if year doesn't match

    matches = sum(1 for kw in keywords if kw.lower() in context_lower)

    if len(keywords) == 0:
        return 0.3

    match_ratio = matches / len(keywords)

    # Scale: 0 matches = 0.1, all matches = 1.0
    return 0.1 + (match_ratio * 0.9)

def render_source_anchored_results(results: Dict, query: str):
    """Render source-anchored evolution results"""

    st.header("📈 Source-Anchored Evolution Analysis")
    st.markdown(f"**Query:** {query}")

    if results['status'] != 'success':
        st.error(f"❌ {results.get('message', 'Analysis failed')}")
        return

    # Overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sources Checked", results['sources_checked'])
    col2.metric("Sources Fetched", results['sources_fetched'])
    col3.metric("Stability", f"{results['stability_score']:.0f}%")

    summary = results['summary']
    if summary['metrics_increased'] > summary['metrics_decreased']:
        col4.success("📈 Trending Up")
    elif summary['metrics_decreased'] > summary['metrics_increased']:
        col4.error("📉 Trending Down")
    else:
        col4.info("➡️ Stable")

    st.markdown("---")

    # Source status
    st.subheader("🔗 Source Verification")
    for src in results['source_results']:
        status_icon = "✅" if src['status'] == 'fetched' else "❌"
        st.markdown(f"{status_icon} {src['url'][:60]}... ({src['numbers_found']} numbers found)")

    st.markdown("---")

    # Metric changes
    st.subheader("💰 Metric Changes")

    if results['metric_changes']:
        rows = []
        for m in results['metric_changes']:
            icon = {
                'increased': '📈',
                'decreased': '📉',
                'unchanged': '➡️',
                'not_found': '❓'
            }.get(m['change_type'], '•')

            change_str = f"{m['change_pct']:+.1f}%" if m['change_pct'] is not None else "-"

            rows.append({
                '': icon,
                'Metric': m['name'],
                'Previous': m['previous_value'],
                'Current': m['current_value'],
                'Change': change_str,
                'Confidence': f"{m['match_confidence']:.0f}%"
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No metrics to compare")

    st.markdown("---")

    # Summary stats
    st.subheader("📊 Summary")
    cols = st.columns(4)
    cols[0].metric("Total Metrics", summary['total_metrics'])
    cols[1].metric("Found in Sources", summary['metrics_found'])
    cols[2].metric("Increased", summary['metrics_increased'])
    cols[3].metric("Decreased", summary['metrics_decreased'])

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


def render_native_comparison(baseline: Dict, compare: Dict):
    """Render a clean comparison between two analyses"""

    st.header("📊 Analysis Comparison")

    # Time info
    baseline_time = baseline.get('timestamp', '')
    compare_time = compare.get('timestamp', '')

    try:
        baseline_dt = datetime.fromisoformat(baseline_time.replace('Z', '+00:00'))
        compare_dt = datetime.fromisoformat(compare_time.replace('Z', '+00:00'))
        delta = compare_dt.replace(tzinfo=None) - baseline_dt.replace(tzinfo=None)
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

    all_ids = set(baseline_by_id.keys()) | set(compare_by_id.keys())

    for cid in sorted(all_ids):
        baseline_m = baseline_by_id.get(cid)
        compare_m = compare_by_id.get(cid)

        # Use canonical name for display, fallback to original
        if baseline_m:
            display_name = baseline_m.get('name', cid)
        elif compare_m:
            display_name = compare_m.get('name', cid)
        else:
            display_name = cid

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
                'Old': f"{old_val} {unit}".strip(),
                'New': f"{new_val} {unit}".strip(),
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


            # Canonicalize metrics before saving for stable future comparisons
            if primary_data.get('primary_metrics'):
                primary_data['primary_metrics_canonical'] = canonicalize_metrics(
                    primary_data.get('primary_metrics', {})
                )

            # Compute semantic hashes for findings
            if primary_data.get('key_findings'):
                findings_with_hash = []
                for finding in primary_data.get('key_findings', []):
                    if finding:
                        findings_with_hash.append({
                            'text': finding,
                            'semantic_hash': compute_semantic_hash(finding)
                        })
                primary_data['key_findings_hashed'] = findings_with_hash

            # Build output
            output = {
                "question": query,
                "timestamp": datetime.now().isoformat(),
                "primary_response": primary_data,
                "final_confidence": final_conf,
                "veracity_scores": veracity_scores,
                "web_sources": web_context.get("sources", [])
            }

            # AUTO-SAVE TO GOOGLE SHEETS
            with st.spinner("💾 Saving to history..."):
                if add_to_history(output):
                    st.success("✅ Analysis saved to Google Sheets")
                else:
                    st.warning("⚠️ Saved to session only (Google Sheets unavailable)")

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


    # =====================
    # TAB 2: EVOLUTION TRACKER (SOURCE-ANCHORED + GOOGLE SHEETS)
    # =====================
    with tab2:
        st.markdown("""
        ### 📈 Evolution Tracker
        Track how data has changed over time using **deterministic source-anchored analysis**.

        **How it works:**
        - Select a baseline from your history (stored in Google Sheets)
        - Re-fetches the **exact same sources** from that analysis
        - Extracts current numbers using regex (no LLM variance)
        - Computes deterministic diffs with context-aware matching
        """)

        # Sidebar - History Management
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

        # Load history
        history = get_history()

        if not history:
            st.info("📭 No previous analyses found. Run an analysis in the 'New Analysis' tab first.")

            # Upload fallback
            st.markdown("---")
            uploaded_file = st.file_uploader(
                "📁 Or upload a previous Yureeka JSON",
                type=['json'],
                key="evolution_upload_fallback"
            )
            if uploaded_file:
                try:
                    uploaded_data = json.load(uploaded_file)
                    add_to_history(uploaded_data)
                    st.success("✅ Uploaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed: {e}")

        else:
            # BASELINE SELECTOR
            st.subheader("📋 Select Baseline Analysis")

            history_options = get_history_options()
            baseline_labels = [opt[0] for opt in history_options]

            baseline_selection = st.selectbox(
                "Select analysis to check for changes",
                options=range(len(baseline_labels)),
                format_func=lambda i: baseline_labels[i],
                key="baseline_select"
            )

            baseline_index = history_options[baseline_selection][1]
            baseline_data = history[baseline_index]

            # Show baseline details
            with st.expander("📋 Baseline Details", expanded=False):
                st.write(f"**Query:** {baseline_data.get('question', 'N/A')}")
                st.write(f"**Timestamp:** {baseline_data.get('timestamp', 'N/A')[:19]}")
                st.write(f"**Confidence:** {baseline_data.get('final_confidence', 'N/A')}%")

                prev_response = baseline_data.get('primary_response', {})
                st.write("**Metrics:**")
                for k, m in list(prev_response.get("primary_metrics", {}).items())[:5]:
                    if isinstance(m, dict):
                        st.write(f"- {m.get('name', k)}: {m.get('value')} {m.get('unit', '')}")

                st.write("**Sources:**")
                prev_sources = baseline_data.get('web_sources', []) or prev_response.get('sources', [])
                for i, src in enumerate(prev_sources[:5], 1):
                    st.write(f"{i}. {src[:60]}...")

            evolution_query = baseline_data.get("question", "")
            st.info(f"📝 Query: {evolution_query}")

            st.markdown("---")

            # COMPARISON OPTIONS
            st.subheader("🔍 Comparison Method")

            compare_method = st.radio(
                "How to compare:",
                options=[
                    "🔗 Re-check original sources (Deterministic - Recommended)",
                    "📋 Compare with another saved analysis (Deterministic)",
                    "🔄 Run fresh analysis (May vary between runs)"
                ],
                index=0,
                key="compare_method"
            )

            # Additional selector for history comparison
            compare_data = None
            if "another saved analysis" in compare_method:
                if len(history) < 2:
                    st.warning("Need at least 2 analyses to compare from history")
                else:
                    compare_labels = [opt[0] for opt in history_options]
                    compare_selection = st.selectbox(
                        "Compare against:",
                        options=range(len(compare_labels)),
                        format_func=lambda i: compare_labels[i],
                        index=0,
                        key="compare_history_select"
                    )
                    compare_index = history_options[compare_selection][1]
                    compare_data = history[compare_index]

            st.markdown("---")

            # RUN COMPARISON
            if st.button("🔍 Run Evolution Analysis", type="primary", key="evolution_btn"):

                if "Re-check original sources" in compare_method:
                    # =====================
                    # SOURCE-ANCHORED (DETERMINISTIC)
                    # =====================
                    st.success("✅ Using deterministic source-anchored analysis")

                    with st.spinner("🔗 Re-fetching original sources..."):
                        results = compute_source_anchored_diff(baseline_data)

                    if results['status'] != 'success':
                        st.error(f"❌ {results.get('message', 'Analysis failed')}")
                    else:
                        # Generate interpretation (only volatile part)
                        interpretation = ""
                        if results['metric_changes']:
                            changes_text = []
                            for m in results['metric_changes']:
                                if m['change_type'] == 'increased':
                                    changes_text.append(f"- {m['name']}: {m['previous_value']} → {m['current_value']} (+{m['change_pct']:.1f}%)")
                                elif m['change_type'] == 'decreased':
                                    changes_text.append(f"- {m['name']}: {m['previous_value']} → {m['current_value']} ({m['change_pct']:.1f}%)")

                            if changes_text:
                                with st.spinner("💬 Generating interpretation..."):
                                    try:
                                        changes_str = "\n".join(changes_text)
                                        explanation_prompt = (
                                            f'Based on these metric changes for "{evolution_query}":\n'
                                            f'{changes_str}\n\n'
                                            'Provide a 2-3 sentence interpretation.\n'
                                            'Return ONLY JSON: {"interpretation": "your text"}'
                                        )
                                        headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
                                        payload = {"model": "sonar", "temperature": 0.0, "max_tokens": 200, "top_p": 1.0, "messages": [{"role": "user", "content": explanation_prompt}]}
                                        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=20)
                                        explanation_data = parse_json_safely(resp.json()["choices"][0]["message"]["content"], "Explanation")
                                        interpretation = explanation_data.get('interpretation', '')
                                    except:
                                        interpretation = ""
                            else:
                                interpretation = "No significant changes detected in the metrics."

                        # Build output
                        evolution_output = {
                            "question": evolution_query,
                            "timestamp": datetime.now().isoformat(),
                            "analysis_type": "source_anchored",
                            "previous_timestamp": baseline_data.get("timestamp"),
                            "results": results,
                            "interpretation": interpretation
                        }

                        # Download
                        st.download_button(
                            label="💾 Download Evolution Report",
                            data=json.dumps(evolution_output, indent=2, ensure_ascii=False).encode('utf-8'),
                            file_name=f"yureeka_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                        # RENDER RESULTS
                        st.header("📈 Evolution Analysis Results")
                        st.markdown(f"**Query:** {evolution_query}")

                        # Overview
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Sources Checked", results['sources_checked'])
                        col2.metric("Sources Fetched", results['sources_fetched'])
                        col3.metric("Stability", f"{results['stability_score']:.0f}%")

                        summary = results['summary']
                        if summary['metrics_increased'] > summary['metrics_decreased']:
                            col4.success("📈 Trending Up")
                        elif summary['metrics_decreased'] > summary['metrics_increased']:
                            col4.error("📉 Trending Down")
                        else:
                            col4.info("➡️ Stable")

                        # Stability indicator
                        st.markdown("---")
                        if results['stability_score'] >= 80:
                            st.success(f"🟢 **High Stability ({results['stability_score']:.0f}%)** - Data consistent with baseline")
                        elif results['stability_score'] >= 60:
                            st.warning(f"🟡 **Moderate Stability ({results['stability_score']:.0f}%)** - Some changes detected")
                        else:
                            st.error(f"🔴 **Low Stability ({results['stability_score']:.0f}%)** - Significant changes")

                        # Interpretation
                        if interpretation:
                            st.info(f"**💬 Interpretation:** {interpretation}")

                        # Source verification
                        st.markdown("---")
                        st.subheader("🔗 Source Verification")
                        for src in results['source_results']:
                            if src['status'] == 'fetched':
                                st.success(f"✅ {src['url'][:70]}... ({src['numbers_found']} numbers)")
                            else:
                                st.error(f"❌ {src['url'][:70]}... (failed)")

                        # Metric changes table
                        st.markdown("---")
                        st.subheader("💰 Metric Changes")

                        if results['metric_changes']:
                            rows = []
                            for m in results['metric_changes']:
                                icon = {'increased': '📈', 'decreased': '📉', 'unchanged': '➡️', 'not_found': '❓'}.get(m['change_type'], '•')
                                change_str = f"{m['change_pct']:+.1f}%" if m['change_pct'] is not None else "-"

                                rows.append({
                                    '': icon,
                                    'Metric': m['name'],
                                    'Old': m['previous_value'],
                                    'New': m['current_value'],
                                    'Δ': change_str,
                                    'Confidence': f"{m['match_confidence']:.0f}%"
                                })

                            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                            # Show context for verification
                            with st.expander("🔍 Match Context"):
                                for m in results['metric_changes']:
                                    if m.get('context_snippet'):
                                        st.caption(f"**{m['name']}:** ...{m['context_snippet']}...")
                        else:
                            st.info("No metrics to compare")

                        # Summary
                        st.markdown("---")
                        st.subheader("📊 Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total", summary['total_metrics'])
                        col2.metric("Found", summary['metrics_found'])
                        col3.metric("📈 Up", summary['metrics_increased'])
                        col4.metric("📉 Down", summary['metrics_decreased'])

                elif "another saved analysis" in compare_method:
                    # =====================
                    # HISTORY VS HISTORY (DETERMINISTIC)
                    # =====================
                    if compare_data:
                        st.success("✅ Comparing two saved analyses (deterministic)")
                        render_native_comparison(baseline_data, compare_data)
                    else:
                        st.error("❌ Please select a comparison analysis")

                else:
                    # =====================
                    # FRESH ANALYSIS (VOLATILE)
                    # =====================
                    st.warning("⚠️ Running fresh analysis - results may vary")

                    query = baseline_data.get('question', '')
                    if not query:
                        st.error("❌ No query found")
                    else:
                        with st.spinner("🌐 Fetching current data..."):
                            web_context = fetch_web_context(query, num_sources=3)

                        if not web_context:
                            web_context = {"search_results": [], "scraped_content": {}, "summary": "", "sources": [], "source_reliability": []}

                        with st.spinner("🤖 Running analysis..."):
                            new_response = query_perplexity(query, web_context)

                        if new_response:
                            try:
                                new_parsed = json.loads(new_response)
                                veracity = evidence_based_veracity(new_parsed, web_context)
                                base_conf = float(new_parsed.get("confidence", 75))
                                final_conf = calculate_final_confidence(base_conf, veracity["overall"])

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

            # Upload additional analyses
            st.markdown("---")
            with st.expander("📁 Upload analysis file"):
                uploaded_file = st.file_uploader("Upload JSON", type=['json'], key="evo_upload")
                if uploaded_file:
                    try:
                        uploaded_data = json.load(uploaded_file)
                        add_to_history(uploaded_data)
                        st.success("✅ Added to history!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")
if __name__ == "__main__":
    main()
