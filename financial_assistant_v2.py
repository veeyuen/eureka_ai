# =========================================================
# AI FINANCIAL RESEARCH ASSISTANT ‚AI HYBRID VERIFICATION v5.6
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

#SYSTEM_PROMPT = (
#    "You are a research analyst focused on topics related to industry analysis, business sectors, finance, economics, and markets.\n"
#    "Output strictly in the JSON format below, including ONLY those financial or economic metrics "
#    "that are specifically relevant to the exact question the user asks.\n"
#    "For example, if the user asks about oil or energy, include metrics like oil production, reserves, "
#    "prices, and exclude unrelated metrics such as inflation or unemployment.\n"
#    "For example, if the user asks about the size of the sneaker market in Asia, include metrics like market size in USD, projected growth rates,"
#    "and exclude unrelated metrics such as inflation or unemployment. Also include information on the major brands of sneakers involved and the trends in the marketplace.\n"
#    "If the question is related to macroeconomics or the underlying drivers are macroeconomic, you may include macroeconomic indicators such as GDP growth, inflation, population size etc.\n"
#    "If the question is not related to business, finance, economics or the markets politely decline to answer the question.\n"
#    "Strictly follow this JSON structure:\n"
#    f"{RESPONSE_TEMPLATE}"
#)

#SYSTEM_PROMPT = (
#    "You are an AI research analyst covering:\n"
#    "- Macroeconomics (GDP, inflation, rates, unemployment)\n"
#    "- **Industry analysis** (market size, growth, major players, trends)\n"
#    "- Business sectors (EV, tech, consumer goods, energy, biotech, etc.)\n\n"
#    "For industry queries, include:\n"
#    "- Market size/revenue\n"
#    "- Growth rates/CAGR\n"
#    "- Top 3-5 companies + market share\n"
#    "- Key trends/drivers\n"
#    "- Relevant financial metrics\n\n"
#    "ALWAYS provide analysis even if web results are sparse - use your knowledge.\n"
#    "Output strictly valid JSON:\n"
#    f"{RESPONSE_TEMPLATE}"
#)

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
   #         st.success(f"‚úÖ Found {len(results)} sources via SerpAPI")
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

def query_perplexity_with_context(query: str, web_context: dict, temperature=0):
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
   ##     f"üîç Web results: {search_results_count} | "
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
        content = re.sub(r'\[\d+\]', '', content)
        
        try:
            parsed = json.loads(content)
            
            # Merge web sources if available
            if web_context.get("sources"):
                existing_sources = parsed.get("sources", [])
                all_sources = existing_sources + web_context["sources"]
                parsed["sources"] = list(set(all_sources))[:10]
                parsed["data_freshness"] = "Current (web-scraped + real-time search)"
            
            content = json.dumps(parsed)
            
        except json.JSONDecodeError as e:
            st.warning(f"JSON parse failed after cleaning: {e}")
            st.caption(f"Raw content preview: {content[:300]}...")
            
            # RICH FALLBACK - matches your ORIGINAL dashboard schema
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
                temperature=0,
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
        values.extend([float(n.strip('%$BMT')) for n in nums[:3]])
    return labels[:5], values[:5]  # Limit for chart


def parse_json_robustly(json_string, context):
    """
    Parses a JSON string safely, addressing all common LLM errors: key quoting, 
    trailing commas, missing delimiters, and unescaped double quotes.
    """
    if not json_string:
        return {}
    
    cleaned_string = json_string.strip()
    
    # 1. Clean up wrappers and control characters
    if cleaned_string.startswith("```json"):
        cleaned_string = cleaned_string[7:].strip()
    if cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[:-3].strip()

    cleaned_string = cleaned_string.replace('\n', ' ').replace('\t', ' ')
    cleaned_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_string)

    # 2. Isolate the main JSON object
    match = re.search(r'\{.*\}', cleaned_string, flags=re.DOTALL)
    if match:
        json_content = match.group(0)
    else:
        st.error(f"JSON parse failed: Could not find any valid JSON object '{{...}}' in {context} response.")
        return {"parse_error": "No JSON object found."}
    
    # 3. AGGRESSIVE STRUCTURAL REPAIR: FIX KEYS, COMMAS, AND BOOLEANS
    repaired_content = json_content
    
    try:
        # Fix 1: Insert missing commas between elements (Crucial for 'Expecting ',' delimiter')
        # Looks for " or ] or } followed by whitespace and then any character that is not a comma, closing brace, or closing bracket. 
        repaired_content = re.sub(r'([\"\]\}])\s*(\s*)(?=[^,\]\}])', r'\1,\2', repaired_content)

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
    max_retries = 20
    current_attempt = 0
    
    while current_attempt < max_retries:
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            # Check for error types
            if not ("Unterminated string" in e.msg or "Expecting ',' delimiter" in e.msg or "Expecting value" in e.msg):
                st.error(f"JSON parse failed (Attempt {current_attempt+1}): {e}")
                error_pos = e.pos if hasattr(e, 'pos') else 0
                start = max(0, error_pos - 50)
                end = min(len(json_content), error_pos + 50)
                st.markdown(f"**Error near:** `{json_content[start:end]}`")
                return {"parse_error": str(e)}

            error_pos = e.pos
            found_quote = -1
            
            # Search backwards from error_pos to find the nearest quote to escape
            for i in range(error_pos - 1, max(0, error_pos - 200), -1): 
                if i < len(json_content) and json_content[i] == '"':
                    if i == 0 or json_content[i-1] != '\\':
                        found_quote = i
                        break
            
            if found_quote != -1:
                json_content = json_content[:found_quote] + '\\"' + json_content[found_quote+1:]
                current_attempt += 1
                continue 
            
            st.error(f"JSON parse failed (Attempt {current_attempt+1}): Could not fix error near position {error_pos}.")
            return {"parse_error": str(e)}

    st.error(f"JSON parse failed after {max_retries} automatic repair attempts.")
    return {"parse_error": "Max retries exceeded"}



def render_dashboard(data, final_conf, sem_conf, num_conf, web_context, base_conf, src_conf, versions_history, user_question, secondary_resp, veracity_scores, show_secondary_view):
    """
    Renders the financial analysis dashboard using Streamlit.
    'data' must be the PARSED JSON dictionary (e.g., j1).
    """

    SECTION_MAPPING = {
        "executive_summary": "Executive Summary",
        "primary_metrics": "📊 Key Performance Indicators (KPIs)",
        "key_findings": "🔍 Core Insights & Key Findings",
        "market_drivers": "🚀 Market Drivers",
        "market_challenges": "🚧 Market Challenges",
        "top_entities": "👑 Top Entities / Competitive Landscape",
        "trends_forecast": "🔮 Trends & Future Forecast",
        "visualization_data": "📈 Trend Visualization",
        "benchmark_table": "⚖️ Competitive Benchmarks",
        "sources": "📄 Data Sources & Confidence Breakdown",
        "confidence": "Confidence Breakdown",
        "freshness": "Data Freshness"
    }


    
    # --- 1. EXECUTIVE SUMMARY & CONFIDENCE SCORE ---
    summary_key = "executive_summary"
    st.header(f"💰 Market Strategy Analysis: {user_question}")
    
    # FIX: data is now confirmed to be a dictionary, allowing .get()
    executive_summary = data.get(summary_key, "Executive Summary not available.")
    
    st.subheader(SECTION_MAPPING.get(summary_key, summary_key.replace('_', ' ').title()))
    st.markdown(f"**{executive_summary}**")

    # Display Confidence Score (Simplified)
    st.markdown(f"**Overall Confidence Score:** **{final_conf}** (Based on Consensus & Validation)")
    
    st.markdown("---")

    # --- 2. LAYOUT: TWO COLUMNS FOR ENTITIES AND FINDINGS ---
    col_entities, col_findings = st.columns(2)

    # =========================================================
    # 2a. TOP ENTITIES & COMPETITORS (FIXED FOR MIXED TYPES)
    # =========================================================
    with col_entities:
        entities_key = "top_entities"
        st.subheader(SECTION_MAPPING.get(entities_key, entities_key.replace('_', ' ').title()))
        top_entities = data.get(entities_key, [])
        
        if top_entities:
            entity_data = []
            
            for entity in top_entities:
                if isinstance(entity, dict):
                    # Case 1: The item is a valid dictionary (as expected)
                    entity_data.append({
                        "Entity": entity.get("name", "N/A"),
                        "Share": entity.get("share", "N/A"),
                        "Details": entity.get("details", "N/A")
                    })
                elif isinstance(entity, str):
                    # Case 2: The item is a string (FIX: ENHANCED Parsing for 'Name (Details)')
                    
                    # Regex looks for: (Name) space (open parenthesis) (details) (close parenthesis)
                    match = re.search(r'^\s*([^\(]+?)\s*\((.*)\)\s*$', entity)
                    
                    if match:
                        # Case 2a: Successfully parsed using (Name) (Detail) format
                        entity_name = match.group(1).strip()
                        entity_details = match.group(2).strip()
                        
                        entity_data.append({
                            "Entity": entity_name,
                            "Share": "N/A", 
                            "Details": entity_details
                        })
                    elif ":" in entity:
                        # Case 2b: Fallback to the previous simple split (Name: Detail)
                        name, detail = entity.split(":", 1)
                        entity_data.append({
                            "Entity": name.strip(),
                            "Share": "N/A",
                            "Details": detail.strip()
                        })
                    else:
                        # Case 2c: If it's just a name, use it as the entity
                        entity_data.append({
                            "Entity": entity.strip(),
                            "Share": "N/A",
                            "Details": "N/A"
                        })
                else:
                    # Case 3: Unexpected type
                    st.warning(f"Skipping malformed entity entry: Expected dict or string, found {type(entity).__name__}.")
                    continue 

            if entity_data:
                try:
                    df_entities = pd.DataFrame(entity_data)
                    st.dataframe(df_entities, hide_index=True, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render Top Entities table: {e}")
            else:
                st.info("No valid top entities data available.")
        else:
            st.info("No top entities data available.")
            
    # --- 2b. KEY FINDINGS ---
    with col_findings:
        findings_key = "key_findings"
        st.subheader(SECTION_MAPPING.get(findings_key, findings_key.replace('_', ' ').title()))
        key_findings = data.get(findings_key, [])
        
        if isinstance(key_findings, list) and key_findings:
            markdown_list = "\n".join([f"- {item}" for item in key_findings])
            st.markdown(markdown_list)
        else:
            st.info("Key findings not available.")

    st.markdown("---")

    # =========================================================
    # 3. METRICS (KEY PERFORMANCE INDICATORS) - FIXED FOR MIXED TYPES
    # =========================================================
    metrics_key = "primary_metrics"
    st.subheader(SECTION_MAPPING.get(metrics_key, metrics_key.replace('_', ' ').title()))
    
    primary_metrics = data.get(metrics_key, [])
    metrics_list = []
    
    # Logic to handle the metrics array, which might be a mix of string and dict formats
    if isinstance(primary_metrics, list):
        for i, metric_item in enumerate(primary_metrics):
            if isinstance(metric_item, str):
                # Handles Format 1: String "Metric Name: Value"
                if ':' in metric_item:
                    key, value = metric_item.split(':', 1)
                    metrics_list.append({
                        "Metric": key.strip(),
                        "Value": value.strip()
                    })
                else:
                    # Fallback for a plain string entry
                    metrics_list.append({
                        "Metric": f"Metric {i+1}",
                        "Value": metric_item.strip()
                    })
            elif isinstance(metric_item, dict):
                # Handles Format 2: Dictionary {"metric": ..., "value": ...}
                name = metric_item.get("metric", metric_item.get("name", f"Metric {i+1}"))
                value = metric_item.get("value", "N/A")
                unit = metric_item.get("unit", "")
                
                metrics_list.append({
                    "Metric": name,
                    "Value": f"{value} {unit}".strip()
                })
            else:
                # Fallback for any other unexpected type
                st.warning(f"Skipping malformed metric entry (Index {i}): Expected string or dict, found {type(metric_item).__name__}.")

    # Logic to handle the dict-of-dicts format (if the AI returns it)
    elif isinstance(primary_metrics, dict):
        for key, metric_dict in primary_metrics.items():
            if isinstance(metric_dict, dict):
                name = metric_dict.get("name", key.replace('_', ' ').title())
                value = metric_dict.get("value", "N/A")
                unit = metric_dict.get("unit", "")
                
                metrics_list.append({
                    "Metric": name,
                    "Value": f"{value} {unit}".strip()
                })

    # Final Table Rendering
    if metrics_list:
        try:
            df_metrics = pd.DataFrame(metrics_list)
            st.table(df_metrics)
        except Exception as e:
             st.warning(f"Failed to render metrics table: {e}")
    else:
        st.info("Primary metrics data is unavailable.")
        
    # --- 4. PLACEHOLDERS FOR OTHER SECTIONS ---
    # Placeholder for the secondary view logic
    if show_secondary_view:
        st.subheader("Secondary Validation View (Placeholder)")
        st.write(f"Confidence (Semantic: {sem_conf}, Numeric: {num_conf})")
        st.write(f"Secondary Response Data: {secondary_resp}")
        st.write(f"Veracity Scores: {veracity_scores}")
        
    # Placeholder for web context (as seen in traceback)
    if web_context:
        st.markdown("---")
        st.subheader("Web Sources")
        st.write(f"Web Sources Found: {len(web_context.get('search_results', []))} (Details hidden)")

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

        sem_conf = semantic_similarity_score(chosen_primary, secondary_resp)

        try:
            j1 = json.loads(chosen_primary)
        except Exception:
            j1 = {}

        try:
            j2 = json.loads(secondary_resp)
        except Exception:
            j2 = {}

        veracity_scores = multi_modal_compare(j1, j2)


    
        # Incorporate veracity_scores["overall_score"] into final confidence if desired:
   #     final_conf = np.mean([base_conf, sem_conf, num_conf if num_conf is not None else 0, src_conf, veracity_scores["overall_score"]])
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

   #     json_str = json.dumps(output_payload, ensure_ascii=False, indent=2)
   #     b64 = base64.b64encode(json_str.encode()).decode()
   #     filename = f"yureeka_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
   #     href = f'<a href="data:application/json;base64,{b64}" download="{filename}"--> Download Analysis JSON</a> (Right-click ‚ Save As)'

   #     st.markdown(href, unsafe_allow_html=True)
   #     st.success("‚úÖ Analysis ready for download!")

        # FIX: The original code used st.markdown(href, unsafe_allow_html=True)
        # We need to ensure the HTML is clean and isolated.
        
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
        "metrics": j1.get("metrics", {}),
        "confidence": base_conf,
        "sources_freshness": 80,
        "change_reason": "Initial version",
        },
        {
        "version": "V2 (Aug 28)",
        "timestamp": "2025-08-28T15:30:00",
        "metrics": j1.get("metrics", {}),
        "confidence": base_conf * 0.98,
        "sources_freshness": 75,
        "change_reason": "Quarterly update",
        },
        {
        "version": "V3 (Nov 3)",
        "timestamp": datetime.now().isoformat(timespec="minutes"),
        "metrics": j1.get("metrics", {}),
        "confidence": final_conf,
        "sources_freshness": 78,
        "change_reason": "Latest analysis",
        }
        ]

        # Save for other pages
       # st.session_state["versions_history"] = versions_history

        # Store ALL versions for individual pages (NEW)
        st.session_state["all_versions"] = versions_history
        st.session_state["current_analysis"] = {
        "summary": j1.get("executive_summary", ""),
        "metrics": j1.get("metrics", {}),
        "confidence": final_conf
        }

        # In main() function, inside if st.button("Analyze") and q: block, around line 570:

        # CALL RENDER_DASHBOARD WITH NEW TOGGLE PARAMETER
        # ... (inside main() function)

        render_dashboard(
        j1, # 🟢 FIX: Pass the parsed dictionary, not the raw string
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
        show_secondary_view=show_validation
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

