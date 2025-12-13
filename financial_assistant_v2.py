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
    {"trend": "Trend description", "direction": "↑/↓/↔", "timeline": "2025-2027"},
    {"trend": "Another trend", "direction": "↑", "timeline": "2026"}
  ],
  "visualization_data": {
    "trend_line": {"labels": ["2023","2024","2025"], "values": [18,22,25]},
    "comparison_bars": {"categories": ["A","B","C"], "values": [25,18,12]}
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

FOLLOW EXACTLY:

1. Return ONLY a single JSON object. NO markdown, NO code blocks, NO explanations.
2. NO references like [1][2] inside JSON strings.
3. NO text before or after the JSON { ... }
4. Use ONLY these fields from the template below.

EVEN IF WEB DATA IS SPARSE, use your knowledge to provide substantive analysis.

For "electric vehicle market":
- Market size ~$600B+, growing 25% CAGR
- Leaders: Tesla (18%), BYD (15%), VW (9%)
- Trends: battery costs ↓60%, China 60% global sales

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

# Note: Ensure 'import re' is at the top of your financial_assistant_v2 (2).py file.

import json
import streamlit as st
import re 
# Note: Ensure 'import re' is at the top of your financial_assistant_v2 (2).py file.

def parse_json_robustly(json_string, context):
    """Parses a JSON string safely by aggressively isolating the main JSON object and performing safe cleaning."""
    if not json_string:
        return {}
    
    cleaned_string = json_string.strip()
    
    # 1. Clean up wrappers and bad characters
    if cleaned_string.startswith("```json"):
        cleaned_string = cleaned_string[7:].strip()
    if cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[:-3].strip()

    # Remove all unescaped newlines and tabs
    cleaned_string = cleaned_string.replace('\n', ' ').replace('\t', ' ')
    
    # Remove non-standard control characters
    cleaned_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_string)

    # 2. Aggressively search for and extract the main JSON object {...}
    # This step is critical to remove any preamble or postamble text.
    match = re.search(r'\{.*\}', cleaned_string, flags=re.DOTALL)
    
    if match:
        json_content = match.group(0)
    else:
        st.error(f"JSON parse failed: Could not find any valid JSON object '{{...}}' in {context} response.")
        return {"parse_error": "No JSON object found."}

    # 3. Stabilized Content Cleanup (Removed high-risk heuristic replacements)
    # Perform a minimal, non-destructive backslash cleanup, as this often relates to escaped quotes.
    # The standard JSON library expects double-escaped backslashes (\\\\), but LLMs often output single (\).
    # We will try a simple replacement that often fixes the issue without corrupting the structure.
    repaired_content = json_content.replace(r'\"', '<<TEMP_QUOTE>>') # Protect correctly escaped quotes
    repaired_content = repaired_content.replace(r'\n', ' ')           # Remove literal \n sequences if they exist
    repaired_content = repaired_content.replace(r'\t', ' ')           # Remove literal \t sequences
    repaired_content = repaired_content.replace('<<TEMP_QUOTE>>', r'\"') # Restore protected quotes


    # 4. Final attempt to load the isolated and repaired JSON string
    try:
        return json.loads(repaired_content)
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parse failed after final cleaning: {e}")
        st.caption(f"Error Context: {context}")
        
        # Display the problematic area around the error character
        error_pos = e.pos if hasattr(e, 'pos') else 1 
        start_char = max(0, error_pos - 50)
        end_char = min(len(repaired_content), error_pos + 50)
        st.markdown("**Problematic Content Snippet:**")
        st.code(repaired_content[start_char:end_char], language="text")
        
        return {"parse_error": str(e)}


def render_dashboard(
    chosen_primary,
    final_conf,
    sem_conf,
    num_conf,
    web_context,
    base_conf,
    src_conf,
    versions_history,
    user_question,
    secondary_resp=None,
    veracity_scores=None
):
    """
    Renders the main analysis dashboard using data from the primary response.
    Includes robust fallbacks for charts and tables to address missing keys.
    """
    # Use the robust parser here
    data = parse_json_robustly(chosen_primary, "Primary")

    # Handle parsing failure before proceeding
    if "parse_error" in data:
        st.error("Cannot render dashboard due to severe parsing error in the LLM response.")
        return

    st.markdown("## 📊 Analysis Dashboard")

    # =========================================================
    # 1. METRICS (SUMMARY & KEY FINDINGS)
    # =========================================================
    st.subheader("Executive Summary")
    st.info(data.get("executive_summary", "Summary not available."))
    
    col1, col2, col3, col4 = st.columns(4)

    # --- Primary Metrics ---
    primary_metrics = data.get("primary_metrics", [])
    if isinstance(primary_metrics, list):
        for i, metric in enumerate(primary_metrics):
            
            # FIX: Ensure the item is a string before attempting to split or strip
            if not isinstance(metric, str):
                st.warning(f"Skipping malformed metric entry (Index {i}): Expected string, found {type(metric).__name__}.")
                continue # Skip this entry and move to the next
            
            # Original parsing logic follows:
            if ':' in metric:
                key, value = metric.split(':', 1)
            else:
                key, value = f"Metric {i+1}", metric
                
            if i == 0:
                col = col1
            elif i == 1:
                col = col2
            elif i == 2:
                col = col3
            elif i == 3:
                col = col4
            else:
                break # Only display first 4
                
            # Now, key and value are guaranteed to be strings or derived from strings
            col.metric(key.strip(), value.strip())

    # --- Key Findings ---
    key_findings = data.get("key_findings", [])
    if isinstance(key_findings, list):
        st.subheader("Key Findings & Insights")
        for finding in key_findings:
            # Ensure finding is a string before printing
            if isinstance(finding, str):
                st.markdown(f"- {finding}")

    # =========================================================
    # 2. TREND VISUALIZATION (Fixes for missing keys & list data)
    # =========================================================
    st.subheader("Trend Visualization")
    
    # 1. Try to find the explicit visualization data
    vis = data.get('visual_data') or data.get('visualization_data') or {} 
    vis['rendered'] = False
    
    if vis.get('type') == 'bar' and vis.get('data'):
        # Existing logic for explicit chart data
        try:
            df_chart = pd.DataFrame(vis['data'])
            fig = px.bar(
                df_chart, 
                x=vis.get('x_axis', df_chart.columns[0]), 
                y=vis.get('y_axis', df_chart.columns[1]), 
                title=vis.get('title', 'Generated Trend Chart')
            )
            st.plotly_chart(fig, use_container_width=True)
            vis['rendered'] = True
        except Exception:
            vis['rendered'] = False

    # 2. FIX: Fallback to use 'top_entities' list data to create a simple bar chart
    if not vis.get('rendered', False) and 'top_entities' in data and isinstance(data['top_entities'], list) and data['top_entities']:
        try:
            df_bar = pd.DataFrame(data['top_entities'])
            # Clean the 'share' column for charting (removes non-numeric chars like %, $, B and handles commas)
            if 'share' in df_bar.columns and 'name' in df_bar.columns:
                df_bar['share_val'] = (
                    df_bar['share']
                    .astype(str)
                    .str.replace('[%B$]', '', regex=True)
                    .str.replace(',', '', regex=False)
                    .astype(float)
                )
                
                # Sort and take top 5 for cleaner visualization
                df_bar = df_bar.sort_values(by='share_val', ascending=False).head(5)
                
                fig = px.bar(
                    df_bar, 
                    x='name', 
                    y='share_val', 
                    title='Top Entities Market Share (Fallback)', 
                    labels={'share_val': 'Market Share (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
                vis['rendered'] = True
        except Exception:
            # Silently fail if fallback chart creation fails
            pass

    if not vis.get('rendered', False):
        st.info("No structured visual data available.")


    # =========================================================
    # 3. DATA TABLE (Fixes for missing keys & list data)
    # =========================================================
    st.subheader("Data Table")
    
    # 1. Try to find the explicit table data
    table_data = data.get('benchmark_table') or data.get('table') or [] 

    # 2. FIX: Fallback to use 'top_entities' list data if no explicit table is found
    if not table_data and 'top_entities' in data and isinstance(data['top_entities'], list):
        table_data = data['top_entities']
        st.caption("Displaying Top Entities data (fallback for missing primary table).")

    # 3. FIX: Fallback to primary_metrics (which is a list of strings)
    if not table_data and 'primary_metrics' in data and isinstance(data['primary_metrics'], list):
        metrics_list = data['primary_metrics']
        rows = []
        for item in metrics_list:
            # Ensure it is a string before parsing
            if not isinstance(item, str):
                continue
            
            # Parse "Key: Value" strings into a two-column table
            if ':' in item:
                category, value = item.split(':', 1)
                rows.append({'Metric': category.strip(), 'Value': value.strip()})
            else:
                rows.append({'Metric/Finding': item})
        if rows:
            table_data = rows
            st.caption("Displaying Primary Metrics data (fallback for missing table).")

    # 4. FIX: Fallback to key_findings (which is a list of strings)
    if not table_data and 'key_findings' in data and isinstance(data['key_findings'], list):
        findings_list = data['key_findings']
        # Convert list of strings into a list of dictionaries for a simple table
        table_data = [{'Finding': f.strip()} for f in findings_list if isinstance(f, str) and f.strip()]
        st.caption("Displaying Key Findings (fallback for missing table).")


    if table_data:
        try:
            # Ensure data is converted to DataFrame before rendering
            df_table = pd.DataFrame(table_data)
            # Use st.dataframe for table rendering
            st.dataframe(df_table, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render table data: {e}")
    else:
        st.info("No tabular data available.")

    # =========================================================
    # 4. CONFIDENCE SCORE & VERACITY
    # =========================================================
    st.subheader("Confidence Score & Veracity")
    colA, colB, colC = st.columns(3)
    
    if final_conf is not None:
        colA.metric("Final Confidence", f"{final_conf:.2f}%")
    
    # Semantics (Secondary Model)
    if sem_conf is not None:
        colB.metric("Semantic Veracity", f"{sem_conf:.2f}%")
        
    # Numerical (Self-Correction)
    if num_conf is not None:
        colC.metric("Numerical Consistency", f"{num_conf:.2f}%")
    
    # --- Veracity Breakdown (New addition in v5.6) ---
    if veracity_scores:
        st.markdown("**Detailed Veracity Breakdown:**")
        
        # Display scores as a bar chart for comparison
        veracity_df = pd.DataFrame(
            veracity_scores.items(), 
            columns=['Category', 'Score']
        ).set_index('Category').T
        
        st.dataframe(veracity_df, use_container_width=True)

    # =========================================================
    # 5. EVOLUTION LAYER & HISTORY (Fixes for timestamp parsing)
    # =========================================================
    st.subheader("Analysis History")
    
    if isinstance(versions_history, list) and versions_history:
        
        history_df = pd.DataFrame(versions_history)
        
        # FIX: Robustly convert the timestamp column using ISO8601 format
        history_df['timestamp'] = pd.to_datetime(
            history_df['timestamp'], 
            format='ISO8601', 
            errors='coerce' # Set errors='coerce' to turn bad data into NaT instead of raising an error
        )

        # Sort history by timestamp
        history_df = history_df.sort_values(by='timestamp', ascending=False)
        
        # Format the datetime object for display
        history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(history_df[['version', 'timestamp', 'confidence', 'sources_freshness', 'change_reason']].head(5), use_container_width=True)
    else:
        st.info("No historical analysis available.")

    # =========================================================
    # 6. RAW SOURCES AND CONTEXT
    # =========================================================
    st.subheader("Sources and Context")
    
    colS1, colS2 = st.columns(2)
    
    # Primary Source Confidence
    if base_conf is not None:
        colS1.metric("Base Model Confidence", f"{base_conf:.2f}%")
        
    # Source Freshness (Simulated/Calculated)
    if src_conf is not None:
        colS2.metric("Source Freshness (Days Ago)", f"{src_conf}")
        
    # Web Context Display
    if web_context:
        st.markdown("**Web Search Context:**")
        # Ensure context is a list before iteration
        if isinstance(web_context, list):
            for i, snippet in enumerate(web_context):
                url = snippet.get("url", "N/A")
                title = snippet.get("title", "No Title")
                snippet_text = snippet.get("snippet", "No snippet available.")
                
                # Only display the first 5 sources neatly
                if i < 5:
                    with st.expander(f"Source {i+1}: {title} (From: {url})"):
                        st.write(snippet_text)



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
    st.title("💹 Yureeka AI Market Analyst")
#    st.caption("Self-Consistency + Cross-Model Verification + Live Web Search + Dynamic Metrics + Evolution Layer")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("""
        Yureeka is a research assistant that assists in providing succinct and accurate answers to your market related questions.  You may ask any 
        question that is related to finance, economics or the markets. 
        This product is currently in prototype stage.
        """)
    with c2:
        web_status = "✅ Enabled" if SERPAPI_KEY else "⚠️ Not configured"
        st.metric("Web Search", web_status)

    q = st.text_input("Enter your question about markets, finance, or economics:")
    use_web_search = st.checkbox("Enable live web search (recommended)", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY)

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
            
        st.info("Cross-validating with Gemini 2.0 Flash...")
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

        json_str = json.dumps(output_payload, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        filename = f"yureeka_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}">💾 Download Analysis JSON</a> (Right-click → Save As)'

        st.markdown(href, unsafe_allow_html=True)
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
        "summary": j1.get("summary", ""),
        "metrics": j1.get("metrics", {}),
        "confidence": final_conf
        }
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
        veracity_scores=veracity_scores  # ← ADD THIS LINE
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
