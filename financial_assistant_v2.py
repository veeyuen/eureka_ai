# ===============================================================================
# YUREEKA AI RESEARCH ASSISTANT v7.20
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
# Expanded Output In Metrics/Findings/Entities With Toggle
# ================================================================================

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
    if json_str is None:
        return {}
    if not isinstance(json_str, str):
        json_str = str(json_str)

    if not json_str.strip():
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
    """Query Perplexity API with web context - deterministic settings"""

    # Check LLM cache first
    cached_response = get_cached_llm_response(query, web_context)
    if cached_response:
        st.info("📦 Using cached LLM response (identical sources)")
        return cached_response

    search_count = len(web_context.get("search_results", []))

    # Build enhanced prompt
    if not web_context.get("summary") or search_count < 2:
        structure_txt = format_query_structure_for_prompt(query_structure)
        enhanced_query = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User Question: {query}\n\n"
            f"{structure_txt}\n\n"
            f"Web search returned {search_count} results. "
            f"Use the structured question to ensure coverage of the main question AND side questions."
            f"Provide complete analysis with all required fields."
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

        structure_txt = format_query_structure_for_prompt(query_structure)
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

def classify_question_signals(query: str) -> Dict[str, Any]:
    """
    Deterministically classify query and return:
      - category: 'country' | 'industry' | 'generic'
      - expected_metric_ids: list[str] (canonical metric ids)
      - signals: list[str] (debuggable reasons)
    """
    q = (query or "").lower().strip()
    signals = []

    if not q:
        return {"category": "generic", "expected_metric_ids": [], "signals": ["empty_query"]}

    country_kw = [
        "gdp", "per capita", "population", "exports", "imports",
        "inflation", "cpi", "interest rate", "policy rate", "central bank",
        "currency", "exchange rate"
    ]
    industry_kw = [
        "market", "industry", "sector", "tam", "total addressable market",
        "cagr", "market size", "market share", "key players", "competitors",
        "pricing", "revenue", "forecast"
    ]

    country_hits = [k for k in country_kw if k in q]
    industry_hits = [k for k in industry_kw if k in q]

    if country_hits and not industry_hits:
        category = "country"
        signals.append(f"country_keywords:{','.join(country_hits[:5])}")
    elif industry_hits and not country_hits:
        category = "industry"
        signals.append(f"industry_keywords:{','.join(industry_hits[:5])}")
    elif industry_hits and country_hits:
        # tie-break: prefer industry unless query is explicitly macro-heavy
        category = "industry"
        signals.append("mixed_signals_default_to_industry")
    else:
        category = "generic"
        signals.append("no_template_keywords")

    expected = QUESTION_CATEGORY_TEMPLATES.get(category, [])
    return {"category": category, "expected_metric_ids": expected, "signals": signals}


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
    Convert metrics to canonical IDs.

    Enhancements:
      - Range-aware mode merges duplicates (same canonical_id) into a single metric span.
      - Deterministic geo tagging: geo_scope + geo_name.
      - Deterministic proxy labeling: is_proxy + proxy_type + proxy_reason (+ confidence).

    NOTE: extra fields added here persist into JSON as long as you save primary_metrics_canonical.
    """
    if not isinstance(metrics, dict):
        return {}

    candidates = []

    # ---- Collect candidates (deterministic ordering) ----
    for key, metric in metrics.items():
        if not isinstance(metric, dict):
            continue

        original_name = metric.get("name", key)
        canonical_id, canonical_name = get_canonical_metric_id(original_name)

        raw_unit = (metric.get("unit") or "").strip()
        unit = raw_unit.upper()

        parsed_val = parse_to_float(metric.get("value"))
        value_for_sort = parsed_val if parsed_val is not None else str(metric.get("value", ""))

        stable_sort_key = (
            str(original_name).lower().strip(),
            unit,
            str(value_for_sort),
            str(key),  # fallback stabilizer
        )

        # GEO inference: use metric context if present
        geo = infer_geo_scope(
            str(original_name),
            str(metric.get("context_snippet", "")),
            str(metric.get("source", "")),
            str(metric.get("source_url", "")),
        )

        # PROXY inference: use question_text + optional context

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
            "canonical_name": canonical_name,
            "original_name": original_name,
            "metric": metric,
            "unit": unit,
            "parsed_val": parsed_val,
            "stable_sort_key": stable_sort_key,

            # NEW: geo
            "geo_scope": geo["geo_scope"],
            "geo_name": geo["geo_name"],

            # NEW: proxy
            **proxy,
        })

    candidates.sort(key=lambda x: x["stable_sort_key"])

    # ---- Group by canonical_id ----
    grouped: Dict[str, List[Dict]] = {}
    for c in candidates:
        grouped.setdefault(c["canonical_id"], []).append(c)

    canonicalized: Dict[str, Dict] = {}

    for cid, group in grouped.items():
        # Single metric → keep as-is
        if len(group) == 1 or not merge_duplicates_to_range:
            g = group[0]
            m = g["metric"]
            canonicalized[cid] = {
                **m,
                "name": g["canonical_name"],
                "canonical_id": cid,
                "original_name": g["original_name"],

                # NEW: geo
                "geo_scope": g.get("geo_scope", "unknown"),
                "geo_name": g.get("geo_name", ""),

                # NEW: proxy
                "is_proxy": bool(g.get("is_proxy", False)),
                "proxy_type": g.get("proxy_type", ""),
                "proxy_reason": g.get("proxy_reason", ""),
                "proxy_confidence": float(g.get("proxy_confidence", 0.0) or 0.0),
                "proxy_target": g.get("proxy_target", ""),
            }
            continue

        # Merge duplicates → range-aware metric
        base = group[0]
        base_metric = dict(base["metric"])  # copy
        base_metric["name"] = base["canonical_name"]
        base_metric["canonical_id"] = cid

        # Merge GEO deterministically
        geo_scope, geo_name = merge_group_geo(group)
        base_metric["geo_scope"] = geo_scope
        base_metric["geo_name"] = geo_name

        # Merge PROXY deterministically
        merged_proxy = merge_group_proxy(group)
        base_metric.update(merged_proxy)

        # Collect numeric candidates
        vals = [g["parsed_val"] for g in group if g["parsed_val"] is not None]
        raw_vals = [str(g["metric"].get("value", "")) for g in group]
        orig_names = [g["original_name"] for g in group]

        # Unit handling: if units disagree, keep base unit but preserve info
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

        canonicalized[cid] = base_metric

    return canonicalized



def freeze_metric_schema(canonical_metrics: Dict) -> Dict:
    """
    Lock metric identity + expected schema for future evolution.
    """
    frozen = {}
    for cid, m in canonical_metrics.items():
        frozen[cid] = {
            "canonical_id": cid,
            "name": m.get("name"),
            "unit": unit_clean_first_letter((m.get("unit") or "").upper().strip()),
            "keywords": extract_context_keywords(m.get("name", "")),
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

def _split_clauses_deterministic(query: str) -> List[str]:
    """
    Split query into candidate clauses deterministically using conservative separators.
    We do NOT try to be 'smart' here; smartness is added in later layers.
    """
    q = _normalize_q(query).lower()
    if not q:
        return []

    # Prefer multi-word separators first by replacing with a hard delimiter.
    tmp = q
    for pat in _QUERY_SPLIT_PATTERNS:
        tmp = re.sub(pat, "|||", tmp, flags=re.IGNORECASE)

    parts = [p.strip(" .?!)(").strip() for p in tmp.split("|||")]
    parts = [p for p in parts if p and len(p) > 2]
    return parts[:10]

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
    This path must NOT validate against LLMResponse.
    """
    try:
        prompt = (
            "Extract a query structure.\n"
            "Return ONLY valid JSON with keys:\n"
            "  category: one of [country, industry, company, finance, market, unknown]\n"
            "  category_confidence: number 0-1\n"
            "  main: string (the main question/topic)\n"
            "  side: array of strings (side questions)\n"
            "No extra keys, no commentary.\n\n"
            f"Query: {query}"
        )

        raw = query_perplexity_raw(prompt, max_tokens=250, timeout=30)

        # If upstream ever returns dict-like accidentally, accept it
        if isinstance(raw, dict):
            parsed = raw
        else:
            if raw is None:
                raw = ""
            if not isinstance(raw, str):
                raw = str(raw)
            parsed = parse_json_safely(raw, "LLM Query Structure")

        if isinstance(parsed, dict) and parsed.get("main") is not None:
            return parsed

    except Exception:
        return None

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

    Output schema:
      {"category": "...", "category_confidence": 0-1, "main": "...", "side": [...], "debug": {...}}
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

        # Override main/side if NLP produced them
        if nlp_out.get("main"):
            main = nlp_out["main"]
        if isinstance(nlp_out.get("side"), list):
            side = nlp_out["side"]

        # If NLP detects a place + overview cue, bias to "country"
        gpes = (hints or {}).get("gpe_entities", []) if isinstance(hints, dict) else []
        overview_hit = (hints or {}).get("overview_signal_hit", False) if isinstance(hints, dict) else False
        if overview_hit and gpes:
            # Only override if deterministic confidence is weak
            if cat_conf < 0.45:
                category = "country"
                cat_conf = max(cat_conf, 0.55)

    # --- Layer 3: embedding-style category vote (TF-IDF cosine) ---
    emb_vote = _embedding_category_vote(q)
    debug["similarity_vote"] = emb_vote

    emb_cat = emb_vote.get("category", "unknown")
    emb_conf = float(emb_vote.get("confidence", 0.0))

    # If deterministic confidence is low, adopt embedding suggestion
    if cat_conf < 0.40 and emb_cat and emb_cat != "unknown" and emb_conf >= 0.45:
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
            main = llm.get("main", main) or main
            side = llm.get("side", side) if isinstance(llm.get("side"), list) else side

    # clean side
    side = _dedupe_clauses([s.strip() for s in (side or []) if isinstance(s, str) and s.strip()])

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
    parts.append(f"- Main: {qs.get('main','')}")
    side = qs.get("side") or []
    if side:
        parts.append("- Side questions:")
        for s in side[:5]:
            parts.append(f"  - {s}")
    tmpl = qs.get("template_sections") or []
    if tmpl:
        parts.append("- Recommended response sections:")
        for t in tmpl[:10]:
            parts.append(f"  - {t}")
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

def render_evolution_results(results: Dict, query: str):
    """Render evolution results with show-all + filter controls"""

    st.header("📈 Evolution Analysis Results")
    st.markdown(f"**Query:** {query}")

    if not isinstance(results, dict) or results.get("status") != "success":
        st.error(f"❌ {results.get('message', 'Analysis failed') if isinstance(results, dict) else 'Analysis failed'}")
        return

    def _filter_df(df: pd.DataFrame, q: str) -> pd.DataFrame:
        if not q:
            return df
        ql = q.lower().strip()
        if not ql:
            return df

        def row_match(row) -> bool:
            for v in row.values:
                try:
                    if ql in str(v).lower():
                        return True
                except Exception:
                    continue
            return False

        mask = df.apply(row_match, axis=1)
        return df[mask]

    def _render_df(title: str, rows: List[Dict[str, Any]], key_prefix: str, default_limit: int = 10):
        st.subheader(title)
        if not rows:
            st.info("No data available")
            return

        df = pd.DataFrame(rows)
        c1, c2 = st.columns([3, 1])
        with c1:
            q = st.text_input("Filter", value="", key=f"{key_prefix}_filter")
        with c2:
            show_all = st.checkbox("Show all", value=False, key=f"{key_prefix}_showall")

        df = _filter_df(df, q)
        if not show_all:
            df = df.head(default_limit)

        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(f"Showing {len(df)} row(s){' (filtered)' if q else ''}.")

    # Overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sources Checked", results.get("sources_checked", 0))
    col2.metric("Sources Fetched", results.get("sources_fetched", 0))
    col3.metric("Stability", f"{float(results.get('stability_score', 0)):.0f}%")

    summary = results.get("summary", {}) or {}
    if summary.get("metrics_increased", 0) > summary.get("metrics_decreased", 0):
        col4.success("📈 Trending Up")
    elif summary.get("metrics_decreased", 0) > summary.get("metrics_increased", 0):
        col4.error("📉 Trending Down")
    else:
        col4.info("➡️ Stable")

    st.markdown("---")

    # Source Verification (expanded + filter)
    src_rows = []
    for s in results.get("source_results", []) or []:
        if isinstance(s, dict):
            src_rows.append({
                "URL": s.get("url", ""),
                "Status": s.get("status", ""),
                "Detail": s.get("status_detail", ""),
                "Numbers Found": s.get("numbers_found", 0),
                "Fingerprint": s.get("fingerprint", ""),
                "Fetched At": s.get("fetched_at", ""),
            })
    _render_df("🔗 Source Verification", src_rows, key_prefix="evo_sources", default_limit=10)

    st.markdown("---")

    # Metric Changes (expanded + filter)
    change_rows = []
    for m in results.get("metric_changes", []) or []:
        if isinstance(m, dict):
            icon = {
                "increased": "📈",
                "decreased": "📉",
                "unchanged": "➡️",
                "not_found": "❓"
            }.get(m.get("change_type"), "•")
            cp = m.get("change_pct")
            cp_str = f"{cp:+.1f}%" if isinstance(cp, (int, float)) else "-"
            change_rows.append({
                "": icon,
                "Metric": m.get("name", ""),
                "Previous": m.get("previous_value", ""),
                "Current": m.get("current_value", ""),
                "Δ": cp_str,
                "Confidence": f"{float(m.get('match_confidence', 0)):.0f}%",
                "Context": m.get("context_snippet", ""),
            })
    _render_df("💰 Metric Changes", change_rows, key_prefix="evo_changes", default_limit=12)

    st.markdown("---")

    # Summary
    st.subheader("📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", summary.get("total_metrics", 0))
    col2.metric("Found", summary.get("metrics_found", 0))
    col3.metric("📈 Up", summary.get("metrics_increased", 0))
    col4.metric("📉 Down", summary.get("metrics_decreased", 0))


# =========================================================
# 8D. SOURCE-ANCHORED EVOLUTION
# Re-fetch the SAME sources from previous analysis for true stability
# Enhanced fetch_url_content function to use scrapingdog as fallback
# =========================================================

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
    """Fetch content and return (content, status_message)"""

    # First try: Direct request
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 403:
            raise Exception("blocked_403")
        if resp.status_code == 404:
            return None, "Page not found (404)"

        resp.raise_for_status()

        if 'captcha' in resp.text.lower() or 'blocked' in resp.text.lower():
            raise Exception("captcha_detected")

        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        clean_text = ' '.join(line for line in lines if line)

        if len(clean_text) > 200:
            return clean_text[:5000], "success"
        else:
            raise Exception("empty_content")

    except Exception as e:
        error_type = str(e)

    # Second try: ScrapingDog
    if SCRAPINGDOG_KEY:
        try:
            api_url = "https://api.scrapingdog.com/scrape"
            params = {"api_key": SCRAPINGDOG_KEY, "url": url, "dynamic": "false"}
            resp = requests.get(api_url, params=params, timeout=30)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                clean_text = ' '.join(line for line in lines if line)

                if len(clean_text) > 200:
                    return clean_text[:5000], "success_via_proxy"
        except:
            pass

    # Map error to user-friendly message
    error_messages = {
        "blocked_403": "Access blocked (403)",
        "captcha_detected": "Captcha/bot protection",
        "empty_content": "No readable content",
    }
    return None, error_messages.get(error_type, f"Failed: {error_type[:30]}")

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
    Build previous metric lookup keyed by *metric_name string*.
    Stores parsed numeric value in comparable scale + unit + raw + keywords.
    """
    prev_numbers = {}
    for key, metric in (prev_metrics or {}).items():
        if not isinstance(metric, dict):
            continue
        metric_name = metric.get("name", key)
        raw = str(metric.get("value", ""))
        unit = metric.get("unit", "")
        val = parse_human_number(raw, unit)
        if val is None:
            continue
        prev_numbers[metric_name] = {
            "value": val,
            "unit": normalize_unit(unit),
            "raw": raw,
            "keywords": extract_context_keywords(metric_name)
        }
    return prev_numbers



def compute_source_anchored_diff(previous_data: Dict) -> Dict:
    """
    Re-fetch the SAME sources from previous analysis and extract current numbers.
    Deterministic + context-aware matching.
    Fixes:
      - schema lookup (canonical_id vs display name mismatch)
      - asserts before variable init
      - baseline reuse ordering
    """
    prev_response = previous_data.get("primary_response", {}) or {}
    prev_sources = previous_data.get("web_sources", []) or prev_response.get("sources", []) or []

    # Always define this early (avoids undefined references in error paths)
    sources_to_check = prev_sources[:5]

    frozen_schema = prev_response.get("metric_schema_frozen", {}) or {}

    # --------- No sources: safe early return (no undefined vars) ----------
    if not prev_sources:
        return {
            "status": "no_sources",
            "message": "No sources found in previous analysis. Please run a new analysis first.",
            "sources_checked": 0,
            "sources_fetched": 0,
            "source_results": [],
            "metric_changes": [],
            "stability": {
                "system_stability_pct": 100.0,
                "data_change_detected": 0,
                "metrics_compared": 0
            },
            "stability_score": 100.0,
            "summary": {
                "total_metrics": 0,
                "metrics_found": 0,
                "metrics_unchanged": 0,
                "metrics_stable": 0,
                "metrics_increased": 0,
                "metrics_decreased": 0,
                "metrics_not_found": 0
            }
        }

    # --------- Build previous metrics (must happen before baseline reuse) ----------
    prev_metrics = prev_response.get("primary_metrics", {}) or {}
    prev_numbers = build_prev_numbers(prev_metrics)

    # --------- Baseline reuse gating ----------
    BASELINE_REUSE_HOURS = 24
    baseline_ts = None

    baseline_ts_raw = prev_response.get("timestamp") or previous_data.get("timestamp")
    baseline_ts = _parse_iso_dt(str(baseline_ts_raw)) if baseline_ts_raw else None

    baseline_is_recent = (
        baseline_ts is not None and
        (now_utc() - baseline_ts).total_seconds() < BASELINE_REUSE_HOURS * 3600
    )


    if baseline_is_recent:
        metric_changes = []
        for metric_name, prev_item in prev_numbers.items():
            metric_changes.append({
                "name": metric_name,
                "previous_value": prev_item["raw"],
                "current_value": prev_item["raw"],
                "change_pct": 0.0,
                "change_type": "unchanged",
                "match_confidence": 100.0,
                "context_snippet": "baseline_recent_reuse"
            })

        summary = {
            "total_metrics": len(metric_changes),
            "metrics_found": len(metric_changes),
            "metrics_unchanged": len(metric_changes),
            "metrics_stable": len(metric_changes),
            "metrics_increased": 0,
            "metrics_decreased": 0,
            "metrics_not_found": 0,
        }
        return {
            "status": "success",
            "sources_checked": len(sources_to_check),
            "sources_fetched": 0,
            "source_results": [
                {"url": u, "status": "skipped_recent_baseline", "status_detail": "baseline_recent_reuse", "numbers_found": 0}
                for u in sources_to_check
            ],
            "metric_changes": metric_changes,
            "stability": {
                "system_stability_pct": 100.0,
                "data_change_detected": 0,
                "metrics_compared": len(metric_changes)
            },
            "stability_score": 100.0,
            "summary": summary
        }

    # --------- Re-fetch sources and extract numbers ----------
    source_results = []
    all_current_numbers = []

    # Baseline cached content (optional) — stored from the original run
    baseline_sources_cache = []
    try:
        baseline_sources_cache = (
            previous_data.get("baseline_sources_cache")
            or prev_response.get("baseline_sources_cache")
            or []
        )
    except Exception:
        baseline_sources_cache = []

    # If someone stored this as a dict wrapper, normalize to a list
    if isinstance(baseline_sources_cache, dict):
        baseline_sources_cache = (
            baseline_sources_cache.get("source_results")
            or baseline_sources_cache.get("sources")
            or baseline_sources_cache.get("items")
            or []
        )

    if not isinstance(baseline_sources_cache, list):
        baseline_sources_cache = []



    for url in sources_to_check:
        content, status_msg = fetch_url_content_with_status(url)

        if content:
            content_fingerprint = fingerprint_text(content)
            numbers = extract_numbers_with_context(content)

            # Compact cache: store extracted numbers + short context snippets (avoid storing full content)


            numbers_compact = [
            {
                "value": n.get("value"),
                "unit": n.get("unit"),
                "raw": n.get("raw"),
                "context_snippet": (n.get("context") or "")[:200]
            }
            for n in numbers
        ]

            source_results.append({
                "url": url,
                "status": "fetched",
                "status_detail": status_msg,
                "numbers_found": len(numbers),
                "fingerprint": content_fingerprint,
                "fetched_at": now_utc().isoformat(),
                "extracted_numbers": numbers_compact
            })

            all_current_numbers.extend(numbers)

        else:

            # baseline_sources_cache is normalized to a LIST of dicts
            cached_numbers = []

            # Find cached extracted_numbers for this URL
            for s in baseline_sources_cache:
                if not isinstance(s, dict):
                    continue
                if s.get("url") != url:
                    continue

                extracted = s.get("extracted_numbers") or []
                if isinstance(extracted, list):
                    cached_numbers.extend(extracted)

            if cached_numbers:
                # Cache contains already-extracted numbers in the same shape as extract_numbers_with_context()
                source_results.append({
                    "url": url,
                    "status": "failed_but_reused_cache",
                    "status_detail": f"{status_msg} (reused baseline cache)",
                    "numbers_found": len(cached_numbers),
                    "fingerprint": None,
                    "fetched_at": now_utc().isoformat(),
                    "extracted_numbers": cached_numbers
                })
                all_current_numbers.extend(cached_numbers)
            else:
                source_results.append({
                    "url": url,
                    "status": "failed",
                    "status_detail": status_msg,
                    "numbers_found": 0,
                    "fetched_at": now_utc().isoformat(),
                })




    # --------- Match metrics using context + value similarity ----------
    metric_changes = []

    # Optional: use canonicalization to enforce frozen schema units (correctly)
    # frozen_schema keys are canonical_id (from freeze_metric_schema)
    # We map metric_name -> canonical_id once for lookup.
    metric_name_to_canonical = {}
    try:
        canon = canonicalize_metrics(prev_metrics)
        for cid, m in canon.items():
            # m["original_name"] is the original metric label
            metric_name_to_canonical[m.get("name", m.get("original_name", cid))] = cid
    except Exception:
        metric_name_to_canonical = {}

    for metric_name, prev_item in prev_numbers.items():
        prev_val = prev_item["value"]
        prev_unit = prev_item["unit"]
        prev_keywords = prev_item["keywords"]

        best_match = None
        best_score = 0.0

        canonical_id = metric_name_to_canonical.get(metric_name)
        schema = frozen_schema.get(canonical_id) if canonical_id else None
        locked_unit = normalize_unit(schema.get("unit")) if isinstance(schema, dict) and schema.get("unit") else ""

        for curr in all_current_numbers:
            curr_val = curr.get("value")
            curr_unit = normalize_unit(curr.get("unit", ""))

            if curr_val is None:
                continue

            # Unit gating:
            # - If we have a locked unit, require match.
            # - Else if prev unit exists and current has unit, require compatible.
            if locked_unit:
                if curr_unit and curr_unit != locked_unit:
                    continue
            else:
                if prev_unit and curr_unit and prev_unit != curr_unit:
                    continue

            # Ratio filter (avoid far-off candidates)
            if prev_val and curr_val:
                ratio = curr_val / prev_val if prev_val != 0 else 0
                if not (0.5 <= ratio <= 2.0):
                    continue
                value_score = 1.0 / (1.0 + abs(ratio - 1.0))
            else:
                continue

            context_score = calculate_context_match(prev_keywords, curr.get("context", ""))

            # Combined score (deterministic weighting)
            combined = (0.4 * value_score) + (0.6 * context_score)

            if combined > best_score:
                best_score = combined
                best_match = curr

        if best_match and best_score > 0.6:
            change_pct = compute_percent_change(prev_val, best_match["value"])

            if change_pct is None or abs(change_pct) < 5:
                change_type = "unchanged"
            elif change_pct > 0:
                change_type = "increased"
            else:
                change_type = "decreased"

            metric_changes.append({
                "name": metric_name,
                "previous_value": prev_item["raw"],
                "current_value": best_match.get("raw", ""),
                "change_pct": change_pct,
                "change_type": change_type,
                "match_confidence": round(best_score * 100, 1),
                "context_snippet": (best_match.get("context", "")[:100] if best_match.get("context") else "")
            })
        else:
            metric_changes.append({
                "name": metric_name,
                "previous_value": prev_item["raw"],
                "current_value": "Not found (no confident match)",
                "change_pct": None,
                "change_type": "not_found",
                "match_confidence": round(best_score * 100, 1) if best_match else 0.0,
                "context_snippet": ""
            })

    # --------- Stability scoring ----------
    found_count = sum(1 for m in metric_changes if m["change_type"] != "not_found")
    unchanged_count = sum(1 for m in metric_changes if m["change_type"] == "unchanged")
    stable_count = sum(
        1 for m in metric_changes
        if m["change_type"] == "unchanged" or (m["change_pct"] is not None and abs(m["change_pct"]) < 10)
    )

    if found_count > 0:
        stability = (stable_count / found_count) * 100.0
    else:
        stability = 100.0

    summary = {
        "total_metrics": len(metric_changes),
        "metrics_found": found_count,
        "metrics_unchanged": unchanged_count,
        "metrics_stable": stable_count,
        "metrics_increased": sum(1 for m in metric_changes if m["change_type"] == "increased"),
        "metrics_decreased": sum(1 for m in metric_changes if m["change_type"] == "decreased"),
        "metrics_not_found": sum(1 for m in metric_changes if m["change_type"] == "not_found"),
    }

    results_stability_score = round(stability, 1)

    # Determinism guard: source order must remain identical
    # Uncomment the 2 lines below for old method
  #  assert sources_to_check == prev_sources[:len(sources_to_check)], \
  #      "Source order mutation detected – evolution must be deterministic"
    if sources_to_check != prev_sources[:len(sources_to_check)]:
        st.error(
            "❌ Source order mutation detected.\n\n"
            "Evolution analysis requires the exact same sources in the same order "
            "to remain deterministic.\n\n"
            "Please rerun the baseline analysis."
        )
        return {
            "status": "error",
            "message": "Source order mutation detected – determinism violated",
            "sources_checked": len(sources_to_check),
            "sources_fetched": 0,
            "source_results": [],
            "metric_changes": [],
            "stability_score": 0.0,
            "summary": {}
        }



    # 🔒 Ensure status_detail always exists
    for src in source_results:
        if 'status_detail' not in src:
            src['status_detail'] = 'unknown'

    return {
        "status": "success",
        "sources_checked": len(sources_to_check),
        "sources_fetched": sum(1 for s in source_results if s["status"] == "fetched"),
        "source_results": source_results,
        "metric_changes": metric_changes,

        # structured stability
        "stability": {
            "system_stability_pct": results_stability_score,
            "data_change_detected": summary["metrics_increased"] + summary["metrics_decreased"],
            "metrics_compared": found_count
        },

        # backward compat
        "stability_score": results_stability_score,
        "summary": summary
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
    Robust against commas, currency symbols, parentheses negatives,
    and avoids many false positives (years, tiny integers, junk contexts).
    """
    if not text:
        return []

    numbers = []

    # Match:
    #   $1,234.56  billion
    #   12.3%
    #   (45.6) M
    pattern = r'(\(?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?|\(?\d+(?:\.\d+)?\)?)\s*(trillion|billion|million|thousand|%|T|B|M|K)?'

    for match in re.finditer(pattern, text, re.IGNORECASE):
        raw_num = match.group(1) or ""
        unit = (match.group(2) or "").strip()

        # Context window
        start = max(0, match.start() - 180)
        end = min(len(text), match.end() + 180)
        context = text[start:end].lower()

        # Skip obvious junk contexts
        if is_likely_junk_context(context):
            continue

        # Reject pure years (e.g., 2024) unless clearly a data value (has unit or currency)
        num_clean_for_year = raw_num.replace("$", "").replace(",", "").strip()
        if re.fullmatch(r"(19|20)\d{2}", num_clean_for_year) and not unit and "$" not in raw_num:
            continue

        # Parse numeric into comparable scale
        parsed = parse_human_number(num_clean_for_year, unit)
        if parsed is None:
            continue

        # Filter out tiny integers that often represent ranks, list counts, etc.
        # Keep if it has % or currency-ish markers.
        u_norm = normalize_unit(unit)
        is_currency_like = ("$" in raw_num) or normalize_currency_prefix(context) or u_norm in ["T", "B", "M", "K"]
        if not is_currency_like and u_norm != "%" and abs(parsed) < 3:
            continue

        # Filter extreme values that are unlikely in market metrics
        if abs(parsed) > 10_000_000:  # in billions this is astronomically large
            continue

        numbers.append({
            "value": parsed,              # normalized comparable value (billions for currency-like units)
            "unit": u_norm,               # normalized unit (T/B/M/%/K/..)
            "context": context,
            "raw": f"{raw_num}{unit}".strip()
        })

    return numbers

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

    # In the source verification display:
    for src in results['source_results']:
        if src['status'] == 'fetched':
            st.success(f"✅ {src['url'][:70]}... ({src['numbers_found']} numbers)")
        else:
            reason = src.get('status_detail', 'Unknown error')
            st.error(f"❌ {src['url'][:50]}... - {reason}")

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

# =========================================================
# 3A. QUESTION CATEGORIZATION + SIGNALS (DETERMINISTIC)
# =========================================================

def categorize_question_signals(question: str) -> Dict[str, Any]:
    """
    Deterministic categorizer + signals extractor.
    Keeps this lightweight and stable (no LLM dependency).
    """
    if not question:
        return {"category": "unknown", "signals": {}, "side_questions": []}

    q = question.strip()
    ql = q.lower()

    # --- category heuristic (deterministic) ---
    is_country = any(k in ql for k in ["gdp", "gdp per capita", "population", "exports", "imports", "currency", "interest rate", "inflation"])
    is_industry = any(k in ql for k in ["market", "industry", "tam", "total addressable", "cagr", "market size", "key players", "competitive landscape"])

    if is_country and not is_industry:
        category = "country"
    elif is_industry and not is_country:
        category = "industry"
    elif is_country and is_industry:
        category = "mixed"
    else:
        category = "general"

    # --- signals (deterministic) ---
    years = re.findall(r"\b(20\d{2})\b", q)
    regions = [r for r in ["asia", "apac", "europe", "north america", "latin america", "middle east", "africa", "china", "india", "japan", "usa", "uk"] if r in ql]

    # naive "side question" split: keep it deterministic
    # examples: "... and the impact of sneaker drops", "... including X"
    side_markers = ["impact of", "effect of", "including", "along with", "as well as", "and also"]
    side_questions = []
    for m in side_markers:
        if m in ql:
            parts = q.split(m, 1)
            if len(parts) == 2:
                candidate = parts[1].strip(" .,:;")
                if candidate and len(candidate) >= 4:
                    side_questions.append(candidate)
            break

    signals = {
        "years": sorted(list(set(years))),
        "regions": sorted(list(set(regions))),
        "contains_country_indicators": bool(is_country),
        "contains_industry_indicators": bool(is_industry),
    }

    return {
        "category": category,
        "signals": signals,
        "side_questions": side_questions
    }


def render_dashboard(
    primary_json: str,
    final_conf: float,
    web_context: Dict,
    base_conf: float,
    user_question: str,
    veracity_scores: Optional[Dict] = None,
    source_reliability: Optional[List[str]] = None,
    key_root: str = "dash",
):
    """Render the analysis dashboard"""

    # Parse primary response
    try:
        data = json.loads(primary_json)
    except Exception as e:
        st.error(f"❌ Cannot render dashboard: {e}")
        st.code(primary_json[:1000])
        return

    # -------------------------
    # Helpers
    # -------------------------
    def _format_metric_value(m: Dict) -> str:
        """Range-aware formatting; falls back to value+unit."""
        if not isinstance(m, dict):
            return "N/A"

        unit = (m.get("unit") or "").strip()

        # Case 1: explicit merged "range" dict (from canonicalize_metrics merge)
        rng = m.get("range") if isinstance(m, dict) else None
        if isinstance(rng, dict) and rng.get("min") is not None and rng.get("max") is not None:
            try:
                vmin = float(rng["min"])
                vmax = float(rng["max"])
                if vmin != vmax:
                    return f"{rng['min']}–{rng['max']} {unit}".strip()
            except Exception:
                pass

        # Case 2: span from get_metric_value_span
        try:
            span = get_metric_value_span(m)
        except Exception:
            span = None

        if isinstance(span, dict) and span.get("min") is not None and span.get("max") is not None:
            try:
                vmin = float(span["min"])
                vmax = float(span["max"])
                u = (span.get("unit") or unit or "").strip()
                if vmin != vmax:
                    return f"{span['min']}–{span['max']} {u}".strip()
                mid = span.get("mid")
                if mid is not None:
                    return f"{mid} {u}".strip()
            except Exception:
                pass

        return f"{m.get('value', 'N/A')} {unit}".strip()

    def _filter_df(df: pd.DataFrame, q: str) -> pd.DataFrame:
        """Simple contains-filter across all cells."""
        if not q:
            return df
        ql = q.lower().strip()
        if not ql:
            return df

        def row_match(row) -> bool:
            for v in row.values:
                try:
                    if ql in str(v).lower():
                        return True
                except Exception:
                    continue
            return False

        mask = df.apply(row_match, axis=1)
        return df[mask]

    def _render_table_with_controls(
        title: str,
        rows: List[Dict[str, Any]],
        key_prefix: str,
        default_limit: int = 8,
        prefer_dataframe_threshold: int = 12,
    ):
        """Show search + Show all; uses st.dataframe for large lists."""
        st.subheader(title)

        if not rows:
            st.info("No data available")
            return

        df = pd.DataFrame(rows)

        # Controls
        c1, c2 = st.columns([3, 1])
        with c1:
            q = st.text_input("Filter", value="", key=f"{key_prefix}_filter")
        with c2:
            show_all = st.checkbox("Show all", value=False, key=f"{key_prefix}_showall")

        df = _filter_df(df, q)

        if not show_all:
            df = df.head(default_limit)

        # Use dataframe for long lists; also fine for short lists (consistent UI)
        if len(df) >= prefer_dataframe_threshold:
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            st.dataframe(df, hide_index=True, width="stretch")

        # Optional: show count info
        st.caption(f"Showing {len(df)} row(s){' (filtered)' if q else ''}.")

    # -------------------------
    # Header
    # -------------------------
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

    # -------------------------
    # Key Metrics (expanded)
    # -------------------------
    metrics = data.get("primary_metrics", {}) or {}

    question_category = data.get("question_category") or (data.get("question_profile", {}) or {}).get("category")
    question_signals = data.get("question_signals") or (data.get("question_profile", {}) or {}).get("signals", {})
    side_questions = data.get("side_questions") or (data.get("question_profile", {}) or {}).get("side_questions", [])

    expected_ids = data.get("expected_metric_ids") or (
        (data.get("question_signals") or {}).get("expected_metric_ids") or []
    )

    metric_rows: List[Dict[str, Any]] = []

    # Add question-derived signals up top (if present)
    if question_category:
        metric_rows.append({"Metric": "Question Category", "Value": str(question_category)})

    if isinstance(question_signals, dict):
        years = question_signals.get("years")
        regions = question_signals.get("regions")
        if years:
            metric_rows.append({"Metric": "Question Years (detected)", "Value": ", ".join(map(str, years))})
        if regions:
            metric_rows.append({"Metric": "Regions (detected)", "Value": ", ".join(map(str, regions))})

    if side_questions:
        metric_rows.append({"Metric": "Side Question(s)", "Value": "; ".join(map(str, side_questions))})

    if isinstance(metrics, dict) and metrics:
        # Canonicalize metrics for stability and range-aware merging if you enabled it
        try:
            canon = canonicalize_metrics(metrics)
        except Exception:
            canon = metrics

        # Build lookup by base canonical id
        by_base: Dict[str, List[Dict]] = {}
        if isinstance(canon, dict):
            for cid, m in canon.items():
                base = re.sub(r'_\d{4}(?:_\d{4})*$', '', str(cid))
                by_base.setdefault(base, []).append(m)

        # If a template exists, output in template order first
        if expected_ids and isinstance(by_base, dict):
            for base_id in expected_ids:
                candidates = by_base.get(base_id, [])
                if candidates:
                    m = candidates[0]
                    metric_rows.append({"Metric": m.get("name", base_id), "Value": _format_metric_value(m)})
                else:
                    metric_rows.append({"Metric": str(base_id).replace("_", " ").title(), "Value": "N/A"})

            # Then append any remaining metrics not covered by expected_ids
            expected_set = set(expected_ids)
            if isinstance(canon, dict):
                for cid, m in canon.items():
                    base = re.sub(r'_\d{4}(?:_\d{4})*$', '', str(cid))
                    if base in expected_set:
                        continue
                    metric_rows.append({"Metric": m.get("name", cid), "Value": _format_metric_value(m)})
        else:
            # No template → show ALL available metrics (not just first 6)
            if isinstance(canon, dict):
                for cid, m in canon.items():
                    metric_rows.append({"Metric": m.get("name", cid), "Value": _format_metric_value(m)})

    #_render_table_with_controls("💰 Key Metrics", metric_rows, key_prefix="dash_metrics", default_limit=10)
    _render_table_with_controls("💰 Key Metrics", metric_rows, key_prefix=f"{key_root}_metrics", default_limit=10)

    st.markdown("---")

    # -------------------------
    # Key Findings (expanded)
    # -------------------------
    findings = data.get("key_findings", []) or []
    finding_rows = [{"#": i + 1, "Finding": f} for i, f in enumerate(findings) if f]
    #_render_table_with_controls("🔍 Key Findings", finding_rows, key_prefix="dash_findings", default_limit=10)
    _render_table_with_controls("🔍 Key Findings", finding_rows, key_prefix=f"{key_root}_findings", default_limit=10)


    st.markdown("---")

    # -------------------------
    # Top Entities (expanded)
    # -------------------------
    entities = data.get("top_entities", []) or []
    entity_rows = []
    for ent in entities:
        if isinstance(ent, dict):
            entity_rows.append({
                "Entity": ent.get("name", "N/A"),
                "Share": ent.get("share", "N/A"),
                "Growth": ent.get("growth", "N/A"),
            })
    #_render_table_with_controls("🏢 Top Market Players", entity_rows, key_prefix="dash_entities", default_limit=10)
    _render_table_with_controls("🏢 Top Market Players", entity_rows, key_prefix=f"{key_root}_entities", default_limit=10)


    st.markdown("---")

    # -------------------------
    # Trends Forecast (expanded)
    # -------------------------
    trends = data.get("trends_forecast", []) or []
    trend_rows = []
    for t in trends:
        if isinstance(t, dict):
            trend_rows.append({
                "Trend": t.get("trend", "N/A"),
                "Direction": t.get("direction", "→"),
                "Timeline": t.get("timeline", "N/A"),
            })
    #_render_table_with_controls("📈 Trends & Forecast", trend_rows, key_prefix="dash_trends", default_limit=10)
    _render_table_with_controls("📈 Trends & Forecast", trend_rows, key_prefix=f"{key_root}_trends", default_limit=10)


    st.markdown("---")

    # -------------------------
    # Visualization (leave as-is; not list heavy)
    # -------------------------
    st.subheader("📊 Data Visualization")
    viz = data.get("visualization_data")

    if viz and isinstance(viz, dict):
        labels = viz.get("chart_labels", [])
        values = viz.get("chart_values", [])
        title = viz.get("chart_title", "Trend Analysis")
        chart_type = viz.get("chart_type", "line")

        if labels and values and len(labels) == len(values):
            try:
                numeric_values = [float(v) for v in values[:10]]
                x_label = viz.get("x_axis_label") or detect_x_label_dynamic(labels)
                y_label = viz.get("y_axis_label") or detect_y_label_dynamic(numeric_values)

                df_viz = pd.DataFrame({"x": labels[:10], "y": numeric_values})

                if chart_type == "bar":
                    fig = px.bar(df_viz, x="x", y="y", title=title)
                else:
                    fig = px.line(df_viz, x="x", y="y", title=title, markers=True)

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

    # Comparison Bars (leave as-is)
    comp = data.get("comparison_bars")
    if comp and isinstance(comp, dict):
        cats = comp.get("categories", [])
        vals = comp.get("values", [])
        if cats and vals and len(cats) == len(vals):
            try:
                df_comp = pd.DataFrame({"Category": cats, "Value": vals})
                fig = px.bar(
                    df_comp, x="Category", y="Value",
                    title=comp.get("title", "Comparison"), text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    st.markdown("---")

    # -------------------------
    # Sources (expanded + filter)
    # -------------------------
    all_sources = data.get("sources", []) or (web_context.get("sources", []) if isinstance(web_context, dict) else [])
    source_rows = [{"#": i + 1, "Source": s, "Reliability": classify_source_reliability(str(s))} for i, s in enumerate(all_sources)]
    #_render_table_with_controls("🔗 Sources & Reliability", source_rows, key_prefix="dash_sources", default_limit=10)
    _render_table_with_controls("🔗 Sources & Reliability", source_rows, key_prefix=f"{key_root}_sources", default_limit=10)


    # Metadata
    col_fresh, _ = st.columns(2)
    with col_fresh:
        freshness = data.get("freshness", "Current")
        st.metric("Data Freshness", freshness)

    st.markdown("---")

    # Veracity Scores
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

    # Web Context (expanded list; still optional)
    if isinstance(web_context, dict) and web_context.get("search_results"):
        with st.expander("🌐 Web Search Details (filterable)", expanded=False):
            sr = web_context.get("search_results", []) or []
            sr_rows = []
            for i, r in enumerate(sr):
                if isinstance(r, dict):
                    sr_rows.append({
                        "#": i + 1,
                        "Title": r.get("title", ""),
                        "Source": r.get("source", ""),
                        "Date": r.get("date", ""),
                        "Snippet": r.get("snippet", ""),
                        "Link": r.get("link", ""),
                    })
            #_render_table_with_controls("Search Results", sr_rows, key_prefix="dash_search_results", default_limit=10)
            _render_table_with_controls("Search Results", sr_rows, key_prefix=f"{key_root}_search_results", default_limit=10)



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

                # -------------------------------
        # Persisted analysis output (prevents reset-on-rerun)
        # -------------------------------
        if "last_analysis" not in st.session_state:
            st.session_state["last_analysis"] = None

        # Analysis button
        if st.button("🔍 Analyze", type="primary", key="analyze_btn") and query:

            # Validate query
            if len(query.strip()) < 5:
                st.error("❌ Please enter a question with at least 5 characters")
                st.stop()

            query = query.strip()[:500]  # Limit length
            analysis_id = hashlib.md5(f"{query}|{datetime.now().isoformat()}".encode("utf-8")).hexdigest()[:10]


            # 3A: Deterministic question categorization/signals for structured reporting
            question_profile = categorize_question_signals(query)
            # Deterministic question signals (no LLM)
            question_signals = classify_question_signals(query)

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
                st.stop()

            # Parse primary response
            try:
                primary_data = json.loads(primary_response)
            except Exception as e:
                st.error(f"❌ Failed to parse primary response: {e}")
                st.code(primary_response[:1000])
                st.stop()

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
            if primary_data.get("primary_metrics"):
                primary_data["primary_metrics_canonical"] = canonicalize_metrics(
                    primary_data.get("primary_metrics", {})
                )
            if primary_data.get("primary_metrics_canonical"):
                primary_data["metric_schema_frozen"] = freeze_metric_schema(
                    primary_data["primary_metrics_canonical"]
                )

            # Compute semantic hashes for findings
            if primary_data.get("key_findings"):
                findings_with_hash = []
                for finding in primary_data.get("key_findings", []):
                    if finding:
                        findings_with_hash.append({
                            "text": finding,
                            "semantic_hash": compute_semantic_hash(finding)
                        })
                primary_data["key_findings_hashed"] = findings_with_hash

            # Attach deterministic question metadata into the primary output (so dashboard + JSON can use it)
            primary_data["question_profile"] = question_profile or {}
            primary_data["question_signals"] = question_signals or {}

            # Re-serialize primary_data AFTER injecting deterministic fields,
            # so the dashboard sees the same structure as what you save.
            primary_json = json.dumps(primary_data, ensure_ascii=False)


            # -------------------------------
            # Build output (SAVE FULL WEB CONTEXT)
            # -------------------------------
            output = {
                "question": query,
                "timestamp": datetime.now().isoformat(),
                "primary_response": primary_data,
                "final_confidence": final_conf,
                "veracity_scores": veracity_scores,

                # ✅ FULL web context + reliability (what you asked for)
                "web_context": web_context,
                "web_sources": web_context.get("sources", []),
                "source_reliability": web_context.get("source_reliability", []),
            }

            # AUTO-SAVE TO GOOGLE SHEETS
            with st.spinner("💾 Saving to history..."):
                if add_to_history(output):
                    st.success("✅ Analysis saved to Google Sheets")
                else:
                    st.warning("⚠️ Saved to session only (Google Sheets unavailable)")

            # Persist so UI widgets won't clear the page on rerun
            st.session_state["last_analysis"] = {
                "primary_json": primary_json,   # ✅ updated JSON (includes question_profile/signals)
                "final_conf": final_conf,
                "web_context": web_context,
                "base_conf": base_conf,
                "query": query,
                "veracity_scores": veracity_scores,
                "source_reliability": web_context.get("source_reliability", []),
                "output": output,
            }


            # ✅ Always re-render the last analysis on rerun (so toggles/filters don't wipe output)
            last = st.session_state.get("last_analysis")
            if isinstance(last, dict) and last.get("primary_json"):
                render_dashboard(
                    last["primary_json"],
                    last["final_conf"],
                    last["web_context"],
                    last["base_conf"],
                    last["query"],
                    last.get("veracity_scores"),
                    (last.get("web_context") or {}).get("source_reliability", []),
                    key_root=f"dash_{last.get('analysis_id','na')}",
                )


        # -------------------------------
        # Re-render last analysis on every rerun
        # (so toggles like "Show all" don't wipe the page)
        # -------------------------------
        last = st.session_state.get("last_analysis")
        if last and isinstance(last, dict) and last.get("primary_json"):
            render_dashboard(
                last["primary_json"],
                float(last.get("final_conf", 0.0)),
                last.get("web_context", {}) or {},
                float(last.get("base_conf", 0.0)),
                str(last.get("query", "")),
                last.get("veracity_scores"),
                last.get("source_reliability"),
            )



        # -------------------------------
        # Always render last output (survives reruns)
        # -------------------------------
        if st.session_state.get("last_analysis"):
            la = st.session_state["last_analysis"]

            # Download always available after reruns
            json_bytes = json.dumps(la["output"], indent=2, ensure_ascii=False).encode("utf-8")
            filename = f"yureeka_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            st.download_button(
                label="💾 Download Analysis JSON",
                data=json_bytes,
                file_name=filename,
                mime="application/json",
                key="download_analysis_json"
            )

            # Render dashboard
            render_dashboard(
                la["primary_json"],
                la["final_conf"],
                la["web_context"],
                la["base_conf"],
                la["query"],
                la["veracity_scores"],
                la["web_context"].get("source_reliability", [])
            )

            # Debug info (optional)
            with st.expander("🔧 Debug Information"):
                st.write("**Confidence Breakdown:**")
                st.json({
                    "base_confidence": la["base_conf"],
                    "evidence_score": la["veracity_scores"].get("overall", 0),
                    "final_confidence": la["final_conf"],
                    "veracity_breakdown": la["veracity_scores"]
                })
                st.write("**Primary Model Response:**")
                st.json(la["primary_data"])


    # =====================
    # TAB 2: EVOLUTION TRACKER (SOURCE-ANCHORED + GOOGLE SHEETS)
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
                            "interpretation": {
                                "text": interpretation,
                                "authoritative": False,
                                "source": "llm_optional"
                            }
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
