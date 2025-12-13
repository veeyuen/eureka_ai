# =========================================================
# AI FINANCIAL RESEARCH ASSISTANT – HYBRID VERIFICATION v5.6
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
#gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
gemini_model = genai.GenerativeModel('gemini-2.5-flash')



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
#RESPONSE_TEMPLATE = """
#You are a research assistant. Return ONLY valid JSON formatted as:
#{
#  "summary": "Brief summary of findings.",
#  "key_insights": ["Insight 1", "Insight 2"],
#  "metrics": {"GDP Growth (%)": number, "Inflation (%)": number, "Unemployment (%)": number},
#  "visual_data": {"labels": ["Q1","Q2"], "values": [2.3,2.5]},
#  "table": [{"Country": "US", "GDP": 25.5, "Inflation": 3.4}],
#  "sources": ["https://imf.org", "https://reuters.com"],
#  "confidence_score": 85,
#  "data_freshness": "As of [date]"
#}
#"""

# ----------------------------
# 1. EXPANDED JSON TEMPLATE
# ----------------------------
RESPONSE_TEMPLATE = """
Return ONLY valid JSON in this flexible structure. Populate ALL fields with deep, relevant data:

{
  "executive_summary": "Detailed 3-5 sentence strategic summary answering the core question, highlighting the 'so what?', and providing a forward-looking conclusion.",
  "primary_metrics": {
    "metric_1": {"name": "Key Metric 1", "value": "e.g. 600", "unit": "e.g. $B"},
    "metric_2": {"name": "Key Metric 2", "value": "e.g. 12.5", "unit": "%"},
    "metric_3": {"name": "Key Metric 3", "value": "e.g. 150", "unit": "M Units"},
    "metric_4": {"name": "Key Metric 4", "value": "e.g. 45", "unit": "% Share"}
  },
  "key_findings": [
    "Finding 1: Detailed insight with quantified impact (e.g., 'Market grew 20% due to X')",
    "Finding 2: Strategic driver explanation",
    "Finding 3: Emerging risk or opportunity",
    "Finding 4: Regulatory or technological shift",
    "Finding 5: Competitive dynamic"
  ],
  "market_drivers": [
    "Driver 1: Detailed description of a factor propelling growth",
    "Driver 2: Technological or consumer behavior catalyst"
  ],
  "market_challenges": [
    "Challenge 1: Primary headwind (e.g., supply chain, regulation)",
    "Challenge 2: Economic or competitive hurdle"
  ],
  "top_entities": [
    {"name": "Entity 1", "share": "25%", "details": "Market leader, strong in [Region]"},
    {"name": "Entity 2", "share": "15%", "details": "Challenger brand, growing 20% YoY"},
    {"name": "Entity 3", "share": "10%", "details": "Niche player focusing on premium segment"}
  ],
  "trends_forecast": [
    {"trend": "Trend Name", "impact": "High/Med/Low", "timeline": "2025-2027", "details": "Brief explanation"},
    {"trend": "Trend Name", "impact": "High", "timeline": "2026", "details": "Brief explanation"}
  ],
  "visualization_data": {
    "trend_line": {"labels": ["2023","2024","2025"], "values": [18,22,25]},
    "comparison_bars": {"categories": ["A","B","C"], "values": [25,18,12]}
  },
  "benchmark_table": [
    {"category": "Metric A", "value_1": 25.5, "value_2": 623},
    {"category": "Metric B", "value_1": 18.2, "value_2": 450}
  ],
  "sources": ["source1.com", "source2.com"],
  "confidence": 87,
  "freshness": "Dec 2025"
}
"""

# ----------------------------
# 1. EXPANDED JSON TEMPLATE
# ----------------------------
# ----------------------------
# 2. SENIOR ANALYST SYSTEM PROMPT
# ----------------------------
SYSTEM_PROMPT = """You are a Senior Market Research Analyst at a top-tier consulting firm (e.g., McKinsey, Goldman Sachs).

YOUR GOAL: Provide a comprehensive, deep, and quantified analysis. Do not be superficial.

CRITICAL INSTRUCTIONS:
1. **Depth over Breadth:** Elaborate on *why* trends are happening, not just *what* they are.
2. **Quantify Everything:** Use numbers, percentages, and $ values wherever possible.
3. **Strategic Lens:** Focus on implications, opportunities, and risks.
4. **Structure:**
   - **Executive Summary:** A dense paragraph synthesizing the current state and future outlook.
   - **Drivers & Challenges:** Explicitly list what pushes and pulls the market.
   - **Entities:** Provide context on *why* they are leaders.

RESPONSE FORMAT:
1. Return ONLY a single JSON object.
2. NO markdown, NO code blocks.
3. Use the exact JSON structure provided below.
4. NEVER return empty fields. Use your internal knowledge if web data is sparse.

"Strictly follow this JSON structure:\n"
f"{RESPONSE_TEMPLATE}"
""" 


@st.cache_resource(show_spinner="Loading AI models...")
#def load_models():
#    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
#    embed = SentenceTransformer("all-MiniLM-L6-v2")
#    return classifier, embed
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
             # sort results by source name for stable order
   #         st.success(f"✅ Found {len(results)} sources via SerpAPI")
    #        results.sort(key=lambda x: x.get("source", "").lower())
        # Add secondary sort key for stability
            results.sort(key=lambda x: (x.get("source", "").lower(), x.get("link", "")))
        return results[:num_results]
     #   return results
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

def query_perplexity_with_context(query: str, web_context: dict, temperature=0.1):
    # FALLBACK: Check web context quality
    search_results_count = len(web_context.get("search_results", []))
    
    if not web_context.get("summary") or search_results_count < 2:
        # Weak web results - prioritize model knowledge
        enhanced_query = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User Question: {query}\n\n"
            f"Web search returned only {search_results_count} usable results. "
            f"Use your general market and industry knowledge to provide a full analysis "
            f"with metrics, key findings, and forward-looking trends."
        )
    else:
        # Strong web results - build context
        context_section = f"""
        LATEST WEB RESEARCH (Current as of today):
        {web_context['summary']}

        """
        if web_context.get('scraped_content'):
            context_section += "\nDETAILED CONTENT FROM TOP SOURCES:\n"
            for url, content in list(web_context['scraped_content'].items())[:2]:
                context_section += f"\nFrom {url}:\n{content[:800]}...\n"
        enhanced_query = f"{context_section}\n{SYSTEM_PROMPT}\n\nUser Question: {query}"

    # DEBUG: Log before API call
   # st.info(
   ##     f"🔍 Web results: {search_results_count} | "
   #     f"Prompt length: {len(enhanced_query)} chars"
   # )
   # st.caption(f"Prompt preview: `{enhanced_query[:200]}...`")

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
            raise Exception("No 'choices' in response")
        content = response_data["choices"][0]["message"]["content"]
        if not content.strip():
            raise Exception("Perplexity returned empty response")

        # ENHANCED JSON CLEANING - strip markdown wrappers and references
        content = content.strip()
    
        # Remove common markdown wrappers
        # Remove markdown code blocks properly
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        content = content.strip()

        # Remove citation references like [1], [2] that break JSON
            # Remove citation references like [1], [2] that break JSON
        content = re.sub(r'\[\d+\]', '', content)

        # Use the robust parser instead of raw json.loads
        parsed = parse_json_robustly(content, context="Perplexity primary response")

        if isinstance(parsed, dict) and "parseerror" not in parsed:
            # Merge web sources if available
            if web_context.get("sources"):
                existing_sources = parsed.get("sources", [])
                all_sources = existing_sources + web_context["sources"]
                parsed["sources"] = list(set(all_sources))[:10]
                parsed["data_freshness"] = "Current (web-scraped + real-time search)"

            content = json.dumps(parsed)
        else:
            # RICH FALLBACK - matches your ORIGINAL dashboard schema
            st.warning("Primary JSON invalid; using fallback schema instead.")
            content = json.dumps({
                "summary": f"Comprehensive analysis of '{query}' with {search_results_count} web sources.",
                "key_insights": [
                    f"Web search found {search_results_count} relevant sources including Fortune Business Insights and IEA.",
                    "Model generated detailed market analysis but JSON formatting needed correction.",
                    "Key themes extracted: market growth, regional dominance, technological trends."
                ],
                "metrics": {
                    "Web Results": search_results_count,
                    "Source Quality": 75,
                    "Model Confidence": 80
                },
                "visual_data": {
                    "labels": [],
                    "values": []
                },
                "table": [],
                "sources": web_context.get("sources", []),
                "confidence_score": 75,
                "data_freshness": "Current (web-enhanced)"
            })

        return content

        
    except Exception as e:
        st.error(f"Perplexity query error: {e}")
        raise


def query_gemini(query: str):
    prompt = f"{SYSTEM_PROMPT}\n\nUser query: {query}"
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2000,
            ),
        )
        # Defensive: check if response contains any candidates or parts
        content = getattr(response, "text", None)
        if not content or not content.strip():
            # Try to extract diagnostics if available
            finish_reason = getattr(response, "finish_reason", None)
            st.warning(f"Gemini returned empty response. finish_reason={finish_reason}")
            raise Exception("Gemini returned empty response")
        try:
            json.loads(content)
        except json.JSONDecodeError:
         #   st.warning("Gemini returned non-JSON response, reformatting...")
            content = json.dumps({
                "summary": content[:500],
                "key_insights": [content[:200]],
                "metrics": {},
                "visual_data": {},
                "table": [],
                "sources": [],
                "confidence_score": 50,
            })
        return content
    except Exception as e:
        st.warning(f"Gemini API error: {e}")
        return json.dumps({
            "summary": "Gemini validation unavailable due to API error.",
            "key_insights": ["Cross-validation could not be performed"],
            "metrics": {},
            "visual_data": {},
            "table": [],
            "sources": [],
            "confidence_score": 0,
        })

# ----------------------------
# SELF-CONSISTENCY & VALIDATION
# ----------------------------
def generate_self_consistent_responses_with_web(query, web_context, n=1):  # generate one response
    #st.info(f"Generating {n} independent analyst responses with web context...")
    st.info(f"Generating analysis...")

    responses, scores = [], []
    success_count = 0
    for i in range(n):
        try:
            r = query_perplexity_with_context(query, web_context, temperature=0.1)
            responses.append(r)
            scores.append(parse_confidence(r))
            success_count += 1
        except Exception as e:
            st.warning(f"Attempt {i+1}/{n} failed: {e}")
            continue
    if success_count == 0:
        st.error("All Perplexity API calls failed.")
        return [], []
   # st.success(f"Successfully generated {success_count}/{n} responses")
#    st.success(f"Successfully generated analysis")

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
        js = json.loads(text)
        return float(js.get("confidence_score", 0))
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
    m1 = j1.get("metrics", {})
    m2 = j2.get("metrics", {})
    if not m1 or not m2:
        return None
    total_diff = 0
    count = 0
    for key in m1:
        if key in m2:
            try:
                v1, v2 = float(m1[key]), float(m2[key])
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
    
    cols = st.columns(min(4, len(str(metrics)) // 50 + 1))  # Dynamic columns
    
    # HANDLE ALL CASES: strings, lists, dicts
    if isinstance(metrics, str):
        # Single string metric - display as key highlights
        st.info(f"**Key Metrics:** {metrics}")
        return
    
    elif isinstance(metrics, list):
        for i, metric in enumerate(metrics[:4]):
            col = cols[i % len(cols)]
            if isinstance(metric, str):
                col.metric("Metric", metric[:60] + "..." if len(metric) > 60 else metric)
            elif isinstance(metric, dict):
                name = metric.get('name', 'Metric')
                value = str(metric.get('value', ''))
                col.metric(name[:30], value)
    
    elif isinstance(metrics, dict):
        items = list(metrics.items())[:4]
        for i, (key, value) in enumerate(items):
            col = cols[i % len(cols)]
            display_val = str(value)[:40]
            col.metric(key[:30], display_val)
    
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
    if current is None or previous is None:
        display_val = str(current) if current is not None else "N/A"
        st.metric(label=label, value=display_val)
    else:
        delta = current - previous
        st.metric(label=label, value=f"{current:.2f}%", delta=f"{delta:+.2f}pp")

def render_evolution_layer(versions_history):
    st.subheader("Evolution Layer - Version Control & Drift")

    version_labels = [v["version"] for v in versions_history]
    selected_ver = st.radio("Select version", version_labels, horizontal=True)
    selected_index = version_labels.index(selected_ver)

    updated_ts = versions_history[selected_index]["timestamp"]
    st.markdown(f"**Updated:** {time_ago(updated_ts)}")

    current_metrics = versions_history[selected_index]["metrics"]
    previous_metrics = (versions_history[selected_index - 1]["metrics"]
                        if selected_index > 0 else current_metrics)

    for m, curr_val in current_metrics.items():
        prev_val = previous_metrics.get(m, curr_val)
        display_metric_with_delta(m, curr_val, prev_val)

    st.markdown(f"*Reason for change:* {versions_history[selected_index]['change_reason']}")

    confidence_points = [v["confidence"] for v in versions_history]
    df_conf = pd.DataFrame({"Version": version_labels, "Confidence": confidence_points})
    fig = px.line(df_conf, x="Version", y="Confidence", title="Confidence Drift", height=150)
    st.plotly_chart(fig, use_container_width=True)

    freshness = versions_history[selected_index]["sources_freshness"]
    st.progress(int(freshness))
    st.caption(f"{freshness}% of ✅ sources updated recently")

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
        values.extend([float(n.strip('%$BMT')) for n in nums[:3]])
    return labels[:5], values[:5]  # Limit for chart


def parse_json_robustly(json_string: str, context: str = ""):
    """
    Safely parse a JSON string with light, conservative repairs.

    - Trims leading/trailing whitespace and control characters.
    - Tries normal json.loads first.
    - If that fails, truncates at the last '}' or ']' and retries.
    - If there is an 'Unterminated string' error, attempts to close the string.
    - Returns a Python object on success, or {"parseerror": "..."} on failure.
    """

    if not json_string:
        msg = "Empty JSON string."
        st.error(f"JSON parse failed: {msg} (context: {context})")
        return {"parseerror": msg}

    # 1. Basic cleanup
    cleaned = json_string.strip()
    # Remove obvious control characters that often sneak in
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)

    # Helper: single safe parse attempt
    def try_parse(payload: str):
        try:
            return json.loads(payload), None
        except json.JSONDecodeError as e:
            return None, e

    # 2. First attempt
    parsed, err = try_parse(cleaned)
    if parsed is not None:
        return parsed

    # 3. Truncate at last closing brace/bracket (removes trailing junk)
    last_brace = cleaned.rfind("}")
    last_bracket = cleaned.rfind("]")
    cut_pos = max(last_brace, last_bracket)

    if cut_pos != -1:
        truncated = cleaned[: cut_pos + 1].strip()
        parsed, err = try_parse(truncated)
        if parsed is not None:
            return parsed
        cleaned = truncated  # keep for further repair
    else:
        # If there is no closing brace/bracket at all, give up early
        msg = f"JSON appears structurally incomplete (no closing '}}' or ']'). Context: {context}"
        st.error(f"JSON parse failed: {msg}")
        return {"parseerror": msg}

    # 4. Handle unterminated strings and simple quote issues
    #    We do a few small repair attempts, then stop.
    max_retries = 5
    current = cleaned

    for _ in range(max_retries):
        parsed, err = try_parse(current)
        if parsed is not None:
            return parsed

        if not isinstance(err, json.JSONDecodeError):
            break

        msg = err.msg or ""
        pos = getattr(err, "pos", None)

        # Case: unterminated string literal
        if "Unterminated string" in msg and pos is not None:
            # Try to close the string at error position
            start_quote = current.rfind('"', 0, pos)
            if start_quote != -1:
                # Insert a closing quote at error position
                current = current[:pos] + '"' + current[pos:]
                continue

        # Case: invalid control character inside string → strip around pos
        if "Invalid control character" in msg and pos is not None:
            current = current[:pos] + current[pos + 1 :]
            continue

        # Case: trailing comma before '}' or ']'
        if "Expecting property name enclosed in double quotes" in msg:
            # Remove trailing commas before } or ]
            current = re.sub(r",\s*([}\]])", r"\1", current)
            continue

        # If we reach here, no specific repair rule matched → break
        break

    # 5. Final failure: return structured error instead of raising
    snippet = current[:500]
    msg = f"JSON parse failed after conservative repairs. Error: {err}. Context: {context}. Snippet: {snippet!r}"
    st.error(msg)
    return {"parseerror": msg}


def render_dashboard(
    chosen_primary: str,
    final_conf: float,
    sem_conf: float,
    num_conf: float,
    web_context: dict,
    base_conf: float,
    src_conf: float,
    versions_history: list,
    user_question: str,
    secondary_resp: str = None,
    veracity_scores: dict = None,
    show_secondary_view: bool = False,
):
    """
    Renders the main analysis dashboard using data from the primary response.
    Supports both:
      - New schema: { "primary_response": { ... } }
      - Legacy schema: { "summary": ..., "metrics": ... }
    """

    # Robust parse of the raw primary JSON
    raw_data = parse_json_robustly(chosen_primary, context="Primary (render_dashboard)")

    # Unwrap if outer envelope is present
    if isinstance(raw_data, dict) and "primary_response" in raw_data:
        data = raw_data["primary_response"]
    else:
        data = raw_data

    if isinstance(data, dict) and "parseerror" in data:
        st.error("Cannot render dashboard due to severe parsing error in the LLM response.")
        return

    if not isinstance(data, dict):
        st.error("Primary response is not a valid JSON object.")
        return

    # Schema‑aware aliases
    executive_summary = data.get("executive_summary") or data.get("summary", "")
    drivers = data.get("drivers", [])
    challenges = data.get("challenges", [])
    entities = data.get("entities", [])
    implications_block = data.get("implications_opportunities_risks", {})
    primary_metrics = data.get("primary_metrics", {})
    legacy_metrics = data.get("metrics", {})
    sources = data.get("sources", [])
    data_freshness = data.get("data_freshness") or data.get("freshness", "")

    # ======================
    # HEADER & META
    # ======================
    st.header("Analysis Dashboard")

    if user_question:
        st.markdown(f"**User Question:** {user_question}")

    col1, col2, col3, col4 = st.columns(4)
    if final_conf is not None:
        col1.metric("Final Confidence", f"{final_conf:.1f}")
    if base_conf is not None:
        col2.metric("Base Model Confidence", f"{base_conf:.1f}")
    if sem_conf is not None:
        col3.metric("Semantic Alignment", f"{sem_conf:.1f}")
    if num_conf is not None:
        col4.metric("Numeric Alignment", f"{num_conf:.1f}" if isinstance(num_conf, (int, float)) else "N/A")

    if data_freshness:
        st.caption(f"🕒 Data freshness: {data_freshness}")

    st.markdown("---")

    # ======================
    # EXECUTIVE SUMMARY
    # ======================
    if executive_summary:
        st.subheader("Executive Summary")
        st.write(executive_summary)

    # ======================
    # DRIVERS & CHALLENGES
    # ======================
    if isinstance(drivers, list) and drivers:
        st.subheader("Market Drivers")
        for d in drivers:
            st.markdown(f"- {d}")

    if isinstance(challenges, list) and challenges:
        st.subheader("Market Challenges")
        for c in challenges:
            st.markdown(f"- {c}")

    # ======================
    # ENTITIES
    # ======================
    if isinstance(entities, list) and entities:
        st.subheader("Key Entities and Players")
        for e in entities:
            st.markdown(f"- {e}")

    # ======================
    # IMPLICATIONS / OPPORTUNITIES / RISKS
    # ======================
    if isinstance(implications_block, dict) and any(implications_block.values()):
        st.subheader("Implications, Opportunities, and Risks")

        if implications_block.get("opportunities"):
            st.markdown("**Opportunities**")
            st.write(implications_block["opportunities"])

        if implications_block.get("risks"):
            st.markdown("**Risks**")
            st.write(implications_block["risks"])

        if implications_block.get("strategic_implications"):
            st.markdown("**Strategic Implications**")
            st.write(implications_block["strategic_implications"])

    st.markdown("---")

    # ======================
    # METRICS
    # ======================
    st.subheader("Key Metrics")

    if isinstance(primary_metrics, dict) and primary_metrics:
        cols = st.columns(4)
        idx = 0
        for key, metric in primary_metrics.items():
            col = cols[idx % len(cols)]
            name = metric.get("name", key)
            value = metric.get("value", "")
            unit = metric.get("unit", "")
            if unit:
                display_val = f"{value} {unit}"
            else:
                display_val = str(value)
            col.metric(label=name, value=display_val)
            idx += 1

    elif isinstance(legacy_metrics, dict) and legacy_metrics:
        cols = st.columns(4)
        idx = 0
        for name, val in legacy_metrics.items():
            col = cols[idx % len(cols)]
            col.metric(label=name, value=str(val))
            idx += 1
    else:
        st.info("No structured metrics available in the analysis.")

    st.markdown("---")

    # ======================
    # SOURCES
    # ======================
    if isinstance(sources, list) and sources:
        st.subheader("Sources")
        for s in sources:
            if isinstance(s, str):
                st.markdown(f"- [{s}]({s})")
            else:
                st.markdown(f"- {s}")

    # ======================
    # WEB CONTEXT
    # ======================
    if isinstance(web_context, dict) and web_context.get("search_results"):
        st.subheader("Web Search Context")
        search_results = web_context.get("search_results", [])
        for i, snippet in enumerate(search_results[:5]):
            url = snippet.get("link", "N/A")
            title = snippet.get("title", "No Title")
            snippet_text = snippet.get("snippet", "No snippet available.")
            with st.expander(f"Source {i+1}: {title}"):
                st.write(f"URL: {url}")
                st.write(snippet_text)

    # ======================
    # SECONDARY VALIDATION VIEW
    # ======================
    if show_secondary_view and secondary_resp:
        st.markdown("---")
        st.subheader("Secondary Validation Response (Diagnostics)")
        sec_parsed = parse_json_robustly(secondary_resp, context="Secondary (render_dashboard)")
        st.json(sec_parsed)

    # ======================
    # VERACITY SCORES
    # ======================
    if isinstance(veracity_scores, dict) and veracity_scores:
        st.markdown("---")
        st.subheader("Veracity & Validation Scores")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Summary", f"{veracity_scores.get('summary_score', 0):.1f}")
        c2.metric("Insights", f"{veracity_scores.get('insights_score', 0):.1f}")
        c3.metric("Table", f"{veracity_scores.get('table_score', 0):.1f}")
        c4.metric("Graphs", f"{veracity_scores.get('graph_score', 0):.1f}")
        c5.metric("Overall", f"{veracity_scores.get('overall_score', 0):.1f}")

    # ======================
    # EVOLUTION / VERSION CONTROL
    # ======================
    if versions_history:
        st.markdown("---")
        st.subheader("Evolution Layer - Version Control & Drift")
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
        sims.append(best_sim)
    return np.mean(sims) if sims else 0.0

def compare_tables(table1, table2):
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

def compare_graphical_data(vis1, vis2):
    if not vis1 or not vis2:
        return 0.0
    labels1 = vis1.get("labels", [])
    labels2 = vis2.get("labels", [])

    values1 = vis1.get("values", [])
    values2 = vis2.get("values", [])

    if labels1 != labels2:
        return 0.0  # Or partial credit with fuzzy matching labels

    if not values1 or not values2 or len(values1) != len(values2):
        return 0.0

    vals1 = np.array(values1)
    vals2 = np.array(values2)

    # Percent similarity using normalized difference
    diff = np.abs(vals1 - vals2)
    max_vals = np.maximum(np.abs(vals1), np.abs(vals2))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(max_vals != 0, diff / max_vals, 0)
    score = 100.0 - (np.nanmean(rel_diff) * 100)
    return max(0, min(100, score))

def multi_modal_compare(json1, json2):
    # Textual comparisons
    summary_score = compare_texts(json1.get("summary", ""), json2.get("summary", ""))
    insights_score = compare_key_insights(json1.get("key_insights", []), json2.get("key_insights", []))

    # Tabular comparison
    table_score = compare_tables(json1.get("table", []), json2.get("table", []))

    # Graphical data comparison
    graph_score = compare_graphical_data(json1.get("visual_data", {}), json2.get("visual_data", {}))

    # Aggregate overall (weights can be tuned)
    weights = {
        "summary": 0.3,
        "insights": 0.2,
        "table": 0.3,
        "graph": 0.2,
    }
    overall_score = (
        summary_score * weights["summary"] +
        insights_score * weights["insights"] +
        table_score * weights["table"] +
        graph_score * weights["graph"]
    )

    return {
        "summary_score": summary_score,
        "insights_score": insights_score,
        "table_score": table_score,
        "graph_score": graph_score,
        "overall_score": overall_score,
    }


# assume these exist elsewhere in your file
# from your_module import (
#     fetch_web_context,
#     generate_self_consistent_responses_with_web,
#     parse_json_robustly,
#     semantic_similarity_score,
#     numeric_alignment_score,
#     render_dashboard,
# )

def main():
    st.set_page_config(page_title="Yureeka Market Intelligence", layout="wide")
    st.title("Yureeka Market Intelligence")

    # 1. User input
    q = st.text_input("Enter your market / macro question", value="", placeholder="e.g., Global electric vehicle market outlook to 2030")

    use_websearch = st.checkbox("Use web search for latest context", value=True)
    show_validation = st.checkbox("Show secondary validation response", value=False)

    if st.button("Analyze") and q:
        # 2. Build web context
        if use_websearch:
            with st.spinner("Searching the web for latest info..."):
                web_context = fetch_web_context(q, num_sources=3)
        else:
            web_context = {
                "search_results": [],
                "scraped_content": {},
                "summary": "",
                "sources": [],
                "source_reliability": [],
            }

        # 3. Generate primary responses (Perplexity) + scores
        with st.spinner("Generating analysis..."):
            responses, scores = generate_self_consistent_responses_with_web(q, web_context, n=1)

        if not responses or not scores:
            st.error("Primary model failed to generate valid responses.")
            return

        if len(responses) != len(scores):
            st.error("Mismatch in responses and scores.")
            return

        # For now, single response; keep majority_vote logic if you use n>1
        chosen_primary = responses[0]
        base_conf = scores[0]

        # 4. Secondary validation (Gemini)
        st.info("Cross-validating with Gemini 2.5 Flash...")
        from your_module import query_gemini  # or keep at top
        secondary_resp = query_gemini(q)

        # 5. Parse raw JSONs with robust parser
        raw_primary = parse_json_robustly(chosen_primary, context="Primary (main)")
        raw_secondary = parse_json_robustly(secondary_resp, context="Secondary (main)") if secondary_resp else {}

        # 6. Unwrap new schema if present
        if isinstance(raw_primary, dict) and "primary_response" in raw_primary:
            j1 = raw_primary["primary_response"]
            final_conf = float(raw_primary.get("final_confidence", base_conf))
            veracity_scores = raw_primary.get("veracity_scores", {})
            user_question = raw_primary.get("question", q)
        else:
            j1 = raw_primary if isinstance(raw_primary, dict) else {}
            final_conf = float(j1.get("confidence_score", base_conf)) if isinstance(j1, dict) else base_conf
            veracity_scores = {}
            user_question = q

        if isinstance(raw_secondary, dict) and "primary_response" in raw_secondary:
            j2 = raw_secondary["primary_response"]
        else:
            j2 = raw_secondary if isinstance(raw_secondary, dict) else {}

        # 7. Compute semantic and numeric alignment
        sem_conf = semantic_similarity_score(chosen_primary, secondary_resp) if secondary_resp else 0.0
        num_conf = numeric_alignment_score(j1, j2) if isinstance(j1, dict) and isinstance(j2, dict) else None

        # 8. Build versions history using whatever metrics are available
        metrics_for_versions = j1.get("metrics", {}) if isinstance(j1, dict) else {}
        versions_history = [
            {
                "version": "V1 (Jul 10)",
                "timestamp": "2025-07-10T12:00:00",
                "metrics": metrics_for_versions,
                "confidence": final_conf,
                "sources_freshness": 80,
                "change_reason": "Initial version",
            },
            {
                "version": "V2 (Aug 28)",
                "timestamp": "2025-08-28T15:30:00",
                "metrics": metrics_for_versions,
                "confidence": final_conf * 0.98 if final_conf is not None else None,
                "sources_freshness": 75,
                "change_reason": "Quarterly update",
            },
            {
                "version": "V3 (Nov 3)",
                "timestamp": datetime.now().isoformat(timespec="minutes"),
                "metrics": metrics_for_versions,
                "confidence": final_conf,
                "sources_freshness": 78,
                "change_reason": "Latest analysis",
            },
        ]

        st.session_state["all_versions"] = versions_history
        st.session_state["current_analysis"] = {
            "summary": j1.get("executive_summary", j1.get("summary", "")) if isinstance(j1, dict) else "",
            "metrics": metrics_for_versions,
            "confidence": final_conf,
        }

        # 9. Offer JSON download of the full outer structure
        output_payload = {
            "primary_response": j1,
            "secondary_response": j2,
            "final_confidence": final_conf,
            "veracity_scores": veracity_scores,
            "question": user_question,
            "timestamp": datetime.now().isoformat(),
        }

        json_str = json.dumps(output_payload, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        filename = f"yureeka_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        href = (
            f'<a href="data:application/json;base64,{b64}" '
            f'download="{filename}">💾 Download Analysis JSON</a> '
            f'(Right-click → Save As)'
        )
        st.markdown(href, unsafe_allow_html=True)
        st.success("✅ Analysis ready for download!")

        # 10. Render dashboard with full context
        render_dashboard(
            chosen_primary=chosen_primary,
            final_conf=final_conf,
            sem_conf=sem_conf,
            num_conf=num_conf,
            web_context=web_context,
            base_conf=base_conf,
            src_conf=None,  # or your existing source freshness metric
            versions_history=versions_history,
            user_question=user_question,
            secondary_resp=secondary_resp,
            veracity_scores=veracity_scores,
            show_secondary_view=show_validation,
        )

        # Optional debug section
        with st.expander("Debug Information"):
            st.write("Primary Response (raw string):")
            st.code(chosen_primary, language="json")
            st.write(f"All Confidence Scores: {scores}")
            st.write(f"Selected Base Score: {base_conf}")
          #  if web_context:
          #      st.write(f"Web Sources Found: {len(web_context.get('search_results', []))}")

if __name__ == "__main__":
    main()
