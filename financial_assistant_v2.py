# =========================================================
# AI FINANCIAL RESEARCH ASSISTANT – HYBRID VERIFICATION v5.3
# WITH WEB SEARCH INTEGRATION (SerpAPI + ScrapingDog)
# Complete standalone version - ready to deploy
# =========================================================

import os
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from collections import Counter
import google.generativeai as genai
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import re

# ----------------------------
# STEP 1: CONFIGURATION
# ----------------------------
# Try Streamlit secrets first, fallback to environment variables
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

# Configure Gemini properly
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# ----------------------------
# SOURCE RELIABILITY CLASSIFIER
# ----------------------------

versions_history = [
    {
        "version": "V1 (Jul 10)",
        "timestamp": "2025-07-10T12:00:00",
        "metrics": {"Penetration (%)": 12.5, "GDP Growth (%)": 2.3},
        "confidence": 68,
        "sources_freshness": 70,
        "change_reason": "Quarterly update"
    },
    {
        "version": "V2 (Aug 28)",
        "timestamp": "2025-08-28T15:30:00",
        "metrics": {"Penetration (%)": 13.1, "GDP Growth (%)": 2.5},
        "confidence": 74,
        "sources_freshness": 80,
        "change_reason": "New economic data release"
    },
    {
        "version": "V3 (Oct 2)",
        "timestamp": "2025-10-02T09:20:00",
        "metrics": {"Penetration (%)": 12.9, "GDP Growth (%)": 2.4},
        "confidence": 71,
        "sources_freshness": 75,
        "change_reason": "Minor adjustments"
    },
]

def classify_source_reliability(source):
    source = source.lower()
    high_sources = ["gov", "imf", "worldbank", "world bank", "central bank", "fed", "ecb", "bank of england", "eu", "ceic"]
    medium_sources = ["reuters", "bloomberg", "ft.com", "financial times", "cnbc", "wsj", "the economist"]
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
    return "⚠️ Medium"  # Default if no match

def source_quality_confidence(sources):
    weights = {"✅ High": 1.0, "⚠️ Medium": 0.6, "❌ Low": 0.3}
    total_score = 0
    count = 0
    for source in sources:
        rank = classify_source_reliability(source)
        score = weights.get(rank, 0.6)
        total_score += score
        count += 1
    return total_score / count if count > 0 else 0.6

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
    delta = current - previous
    st.metric(label=label, value=f"{current:.2f}%", delta=f"{delta:+.2f}pp")

RESPONSE_TEMPLATE = """
You are a research assistant. Return ONLY valid JSON formatted as:
{
  "summary": "Brief summary of findings.",
  "key_insights": ["Insight 1", "Insight 2"],
  "metrics": {"GDP Growth (%)": number, "Inflation (%)": number, "Unemployment (%)": number},
  "visual_data": {"labels": ["Q1","Q2"], "values": [2.3,2.5]},
  "table": [{"Country": "US", "GDP": 25.5, "Inflation": 3.4}],
  "sources": ["https://imf.org", "https://reuters.com"],
  "confidence_score": 85,
  "data_freshness": "As of [date]"
}
"""

SYSTEM_PROMPT = (
    "You are an AI research analyst that only answers topics related to business, finance, economics, or markets.\n"
    "You will reference only the most current data available to answer the query.\n"
    "If you receive a query unrelated to those topics politely decline the request.\n"
    "Output strictly in the JSON structure below:\n"
    f"{RESPONSE_TEMPLATE}"
)

# Cache model loading
@st.cache_resource
def load_models():
    """Load ML models once and cache them"""
    classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli", 
        device=-1
    )
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    return classifier, embed

# Domain classifier and embedder
ALLOWED_TOPICS = ["finance", "economics", "markets", "business", "macroeconomics"]
domain_classifier, embedder = load_models()

# ----------------------------
# STEP 2: WEB SEARCH FUNCTIONS
# ----------------------------
def search_serpapi(query: str, num_results: int = 5):
    """Search using SerpAPI (Google Search)"""
    if not SERPAPI_KEY:
        st.info("💡 SerpAPI key not configured. Add SERPAPI_KEY to secrets for enhanced web search.")
        return []
    
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": f"{query} finance economics markets",
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "tbm": "nws",  # News results
        "tbs": "qdr:m"  # Past month
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Extract news results
        for item in data.get("news_results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "source": item.get("source", {}).get("name", "")
            })
        
        # Fallback to organic results if no news
        if not results:
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date": "",
                    "source": item.get("source", "")
                })
        
        if results:
            st.success(f"✅ Found {len(results)} sources via SerpAPI")
        return results
        
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ SerpAPI search error: {e}")
        return []
    except Exception as e:
        st.warning(f"⚠️ Error processing SerpAPI results: {e}")
        return []


def scrape_url_scrapingdog(url: str):
    """Scrape content from URL using ScrapingDog"""
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
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        # Extract text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:3000]  # Limit to 3000 chars
        
    except Exception as e:
        st.warning(f"⚠️ ScrapingDog error for {url[:50]}: {e}")
        return None

def fetch_web_context(query: str, num_sources: int = 3):
    """Fetch web context by searching and optionally scraping, apply reliability bands"""
    search_results = search_serpapi(query, num_results=5)
    if not search_results:
        return {
            "search_results": [],
            "scraped_content": {},
            "summary": "",
            "sources": [],
            "source_reliability": [],
        }
    scraped_content = {}
    if SCRAPINGDOG_KEY:
        st.info(f"🔍 Scraping top {min(num_sources, len(search_results))} sources...")
        for i, result in enumerate(search_results[:num_sources]):
            url = result["link"]
            content = scrape_url_scrapingdog(url)
            if content:
                scraped_content[url] = content
                st.success(f"✓ Scraped {i+1}/{num_sources}: {result['source']}")
    # Build context and assign reliability
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
    summary = "\n\n---\n\n".join(context_parts)
    sources = [r["link"] for r in search_results]
    return {
        "search_results": search_results,
        "scraped_content": scraped_content,
        "summary": summary,
        "sources": sources,
        "source_reliability": reliabilities,
    }

# ----------------------------
# STEP 3: ENHANCED QUERY FUNCTIONS
# ----------------------------
def query_perplexity_with_context(query: str, web_context: dict, temperature=0.2):
    """Query Perplexity with web-scraped context"""
    
    # Build enhanced prompt with web context
    if web_context["summary"]:
        context_section = f"""
LATEST WEB RESEARCH (Current as of today):
{web_context['summary']}

"""
        # Add scraped content if available
        if web_context['scraped_content']:
            context_section += "\nDETAILED CONTENT FROM TOP SOURCES:\n"
            for url, content in list(web_context['scraped_content'].items())[:2]:
                context_section += f"\nFrom {url}:\n{content[:800]}...\n"
        
        enhanced_query = f"{context_section}\n{SYSTEM_PROMPT}\n\nUser Question: {query}"
    else:
        enhanced_query = f"{SYSTEM_PROMPT}\n\nUser Question: {query}"
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}", 
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sonar",
        "temperature": temperature,
        "max_tokens": 2000,
        "messages": [
            {"role": "user", "content": enhanced_query}
        ]
    }
    
    try:
        resp = requests.post(
            PERPLEXITY_URL, 
            headers=headers, 
            json=payload, 
            timeout=45
        )
        
        if resp.status_code != 200:
            error_detail = resp.text
            st.error(f"Perplexity API Error {resp.status_code}: {error_detail}")
            raise Exception(f"Perplexity returned {resp.status_code}: {error_detail}")
        
        response_data = resp.json()
        
        if "choices" not in response_data:
            raise Exception(f"No 'choices' in response")
        
        content = response_data["choices"][0]["message"]["content"]
        
        if not content or not content.strip():
            raise Exception("Perplexity returned empty response")
        
        # Validate JSON and inject sources
        try:
            parsed = json.loads(content)
            # Add our web sources
            if web_context["sources"]:
                existing_sources = parsed.get("sources", [])
                all_sources = existing_sources + web_context["sources"]
                parsed["sources"] = list(set(all_sources))[:10]  # Unique, max 10
                parsed["data_freshness"] = "Current (web-scraped + real-time search)"
            content = json.dumps(parsed)
        except json.JSONDecodeError:
            # If not valid JSON, wrap it
            st.warning("Reformatting Perplexity response...")
            content = json.dumps({
                "summary": content[:500],
                "key_insights": [content[:200]],
                "metrics": {},
                "visual_data": {},
                "table": [],
                "sources": web_context["sources"],
                "confidence_score": 50,
                "data_freshness": "Current (web-scraped)"
            })
        
        return content
        
    except Exception as e:
        st.error(f"Perplexity query error: {e}")
        raise


def query_gemini(query: str):
    """Query Gemini API with proper error handling"""
    prompt = f"{SYSTEM_PROMPT}\n\nUser query: {query}"
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
        )
        content = response.text
        
        if not content or not content.strip():
            raise Exception("Gemini returned empty response")
        
        try:
            json.loads(content)
        except json.JSONDecodeError:
            st.warning("Gemini returned non-JSON response, reformatting...")
            content = json.dumps({
                "summary": content[:500],
                "key_insights": [content[:200]],
                "metrics": {},
                "visual_data": {},
                "table": [],
                "sources": [],
                "confidence_score": 50
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
            "confidence_score": 0
        })

# ----------------------------
# STEP 4: SELF-CONSISTENCY WITH WEB SEARCH
# ----------------------------
def generate_self_consistent_responses_with_web(query, web_context, n=3):
    """Generate multiple responses with web context"""
    st.info(f"Generating {n} independent analyst responses with web context...")
    responses, scores = [], []
    success_count = 0
    
    for i in range(n):
        try:
            r = query_perplexity_with_context(query, web_context, temperature=0.8)
            responses.append(r)
            scores.append(parse_confidence(r))
            success_count += 1
        except Exception as e:
            st.warning(f"Attempt {i+1}/{n} failed: {e}")
            continue
    
    if success_count == 0:
        st.error("All Perplexity API calls failed.")
        return [], []
    
    st.success(f"Successfully generated {success_count}/{n} responses")
    return responses, scores


def majority_vote(responses):
    """Select most common response via voting"""
    if not responses:
        return ""
    cleaned = [r.strip() for r in responses if r]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def parse_confidence(text):
    """Extract confidence score from JSON response"""
    try:
        js = json.loads(text)
        conf = js.get("confidence_score", 0)
        return float(conf)
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0

# ----------------------------
# STEP 5: VALIDATION FUNCTIONS
# ----------------------------
def semantic_similarity_score(a, b):
    """Calculate semantic similarity between two texts"""
    try:
        v1, v2 = embedder.encode([a, b])
        sim = util.cos_sim(v1, v2)
        if hasattr(sim, 'item'):
            score = sim.item()
        elif hasattr(sim, 'shape'):
            score = float(sim[0][0])
        else:
            score = float(sim)
        return round(score * 100, 2)
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return 0.0


def numeric_alignment_score(j1, j2):
    """Compare numeric metrics between two JSON responses"""
    m1 = j1.get("metrics", {})
    m2 = j2.get("metrics", {})
    
    if not m1 or not m2:
        return None
    
    total_diff, count = 0, 0
    
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
# STEP 6: DASHBOARD RENDERER
# ----------------------------

def render_evolution_layer(versions_history):
    st.subheader("Evolution Layer - Version Control & Drift")

    # 1. Version selector
    version_labels = [v["version"] for v in versions_history]
    selected_ver = st.radio("Select version", version_labels, horizontal=True)
    selected_index = version_labels.index(selected_ver)

    # Show updated timestamp
    updated_ts = versions_history[selected_index]["timestamp"]
    st.markdown(f"**Updated:** {time_ago(updated_ts)}")

    # Display metrics with deltas (if previous version exists)
    current_metrics = versions_history[selected_index]["metrics"]
    if selected_index > 0:
        prev_metrics = versions_history[selected_index - 1]["metrics"]
    else:
        prev_metrics = current_metrics

    for metric_name, current_val in current_metrics.items():
        previous_val = prev_metrics.get(metric_name, current_val)
        display_metric_with_delta(metric_name, current_val, previous_val)

    # Show change reason
    st.markdown(f"*Reason for change:* {versions_history[selected_index]['change_reason']}")

    # 4. Confidence drift sparkline
    confidence_points = [v["confidence"] for v in versions_history]
    df_conf = pd.DataFrame({"Version": version_labels, "Confidence": confidence_points})
    fig = px.line(df_conf, x="Version", y="Confidence", title="Confidence Drift", height=150)
    st.plotly_chart(fig, use_container_width=True)

    # 5. Source freshness bar
    freshness = versions_history[selected_index]["sources_freshness"]
    st.progress(int(freshness))
    st.caption(f"{freshness}% of ✅ sources updated recently")



def render_dashboard(response, final_conf, sem_conf, num_conf, web_context=None, base_conf=None, src_conf=None):
    """Render the financial dashboard with results and source reliability bands"""
    
    if not response or not response.strip():
        st.error("Received empty response from model")
        return
    
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON returned by model: {e}")
        st.write("**Raw response received:**")
        st.code(response[:1000], language="text")
        return

    # Confidence and freshness display
    col1, col2 = st.columns(2)
  #  col1.metric("Overall Confidence (%)", f"{final_conf:.1f}")
    
    # Confidence breakdown display
    st.subheader("Confidence Score Breakdown")
    st.write(f"- Base model confidence: {base_conf:.1f}%")
    st.write(f"- Semantic similarity confidence: {sem_conf:.1f}%")
    st.write(f"- Numeric alignment confidence: {num_conf if num_conf is not None else 'N/A'}%")
    st.write(f"- Source quality confidence: {src_conf:.1f}%")
    st.write(f"---\n**Overall confidence: {final_conf:.1f}%**")

    freshness = data.get("data_freshness", "Unknown")
    col2.metric("Data Freshness", freshness)

  
    # Main summary
    st.header("📊 Financial Summary")
    st.write(data.get("summary", "No summary available."))

    # Key insights
    st.subheader("Key Insights")
    insights = data.get("key_insights", [])
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No key insights provided.")

    # Metrics display
    st.subheader("Metrics")
    mets = data.get("metrics", {})
    if mets:
        cols = st.columns(len(mets))
        for i, (k, v) in enumerate(mets.items()):
            cols[i].metric(k, v)
    else:
        st.info("No metrics available.")

    # Visualization
    st.subheader("Trend Visualization")
    vis = data.get("visual_data", {})
    if "labels" in vis and "values" in vis:
        try:
            df = pd.DataFrame({
                "Period": vis["labels"], 
                "Value": vis["values"]
            })
            fig = px.line(
                df, 
                x="Period", 
                y="Value", 
                title="Quarterly Trends", 
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render visualization: {e}")
    else:
        st.info("No visual data available.")

    # Data table
    st.subheader("Data Table")
    tab = data.get("table", [])
    if tab:
        try:
            st.dataframe(pd.DataFrame(tab), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render table: {e}")
    else:
        st.info("No tabular data available.")

    # Sources - PROMINENT DISPLAY WITH RELIABILITY BANDS
    st.subheader("📚 Sources & References")
    sources = data.get("sources", [])
    if sources:
        st.success(f"✅ Information from {len(sources)} sources:")
        for i, s in enumerate(sources, 1):
            rank = classify_source_reliability(s)
            st.markdown(f"{i}. [{s}]({s})  — {rank}")
    else:
        st.info("No sources cited.")
    
    # Show web search stats if available, with reliability bands
    if web_context and web_context.get("search_results"):
        with st.expander("🔍 Web Search Details"):
            st.write(f"**Sources Found:** {len(web_context['search_results'])}")
            st.write(f"**Pages Scraped:** {len(web_context.get('scraped_content', {}))}")
            for idx, result in enumerate(web_context['search_results']):
                rank = web_context.get('source_reliability', [])
                reliab = rank[idx] if idx < len(rank) else classify_source_reliability(result['link'] + ' ' + result.get('source',''))
                st.markdown(f"- **{result['title']}** ({result.get('source', 'Unknown')}) [{reliab}]")

    # Validation metrics
    st.subheader("Validation Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Semantic Similarity", f"{sem_conf:.2f}%")
    if num_conf is not None:
        col2.metric("Numeric Alignment", f"{num_conf:.2f}%")
    else:
        col2.info("No numeric data to compare")

    # Evolution layer

    render_evolution_layer(versions_history)

# ----------------------------
# STEP 7: MAIN WORKFLOW
# ----------------------------
def main():
    st.set_page_config(
        page_title="Yureeka Market Research Assistant", 
        layout="wide"
    )
    st.title("💹 Yureeka AI Market Analyst")
    st.caption("Self-Consistency + Cross-Model Verification + Live Web Search + Version control + Drift monitoring")
    
    # Show web search status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        This assistant combines:
        - **Self-consistency reasoning** via multiple Perplexity analyses
        - **Cross-model validation** using Gemini 2.0 Flash
        - **Live web search**
        - **Semantic and numeric alignment** scoring
        - Version control & drift monitoring via evolution layer
        """)
    with col2:
        web_status = "✅ Enabled" if SERPAPI_KEY else "⚠️ Not configured"
        st.metric("Web Search", web_status)

    q = st.text_input("Enter your question about markets, finance, or economics:")
    
    # Web search toggle
    use_web_search = st.checkbox(
        "🌐 Enable live web search (recommended for current data)", 
        value=bool(SERPAPI_KEY),
        disabled=not SERPAPI_KEY,
        help="Searches the web for latest information before analysis"
    )

    if st.button("Analyze") and q:
        
        # Fetch web context if enabled
        web_context = {}
        if use_web_search:
            with st.spinner("🔍 Searching the web for latest information..."):
                web_context = fetch_web_context(q, num_sources=3)
        
        # Generate responses
        if web_context and web_context.get("search_results"):
            responses, scores = generate_self_consistent_responses_with_web(q, web_context, n=3)
        else:
            st.info("📚 Proceeding with AI model knowledge only...")
            # Fallback to original method without web context
            empty_context = {"search_results": [], "scraped_content": {}, "summary": "", "sources": []}
            responses, scores = generate_self_consistent_responses_with_web(q, empty_context, n=3)
        
        if not responses or not scores:
            st.error("Primary model failed to generate any valid responses.")
            return
        
        if len(responses) != len(scores):
            st.error("Internal error: response/score alignment mismatch")
            return
        
        # Select best response
        voted_response = majority_vote(responses)
        max_score = max(scores)
        best_idx = scores.index(max_score)
        best_response = responses[best_idx]
        
        chosen_primary = best_response if best_response else voted_response

        if not chosen_primary:
            st.error("Could not determine primary response.")
            return

        # Cross-validation with Gemini
        st.info("Cross‑verifying via Gemini 2.0 Flash...")
        secondary_resp = query_gemini(q)

        # Scoring
        sem_conf = semantic_similarity_score(chosen_primary, secondary_resp)
        
        try:
            j1 = json.loads(chosen_primary)
        except json.JSONDecodeError:
            j1 = {}
        
        try:
            j2 = json.loads(secondary_resp)
        except json.JSONDecodeError:
            j2 = {}
        
        num_conf = numeric_alignment_score(j1, j2)
        base_conf = max_score

        # Calculate final confidence
       # confidence_components = [base_conf, sem_conf]
        #if num_conf is not None:
        #    confidence_components.append(num_conf)
        
        #final_conf = np.mean(confidence_components)

        # When calculating final confidence:
        # Compute source quality confidence from the decoded JSON response 'data'
        src_conf = source_quality_confidence(j1.get("sources", [])) * 100  # scale to percentage

        confidence_components = [base_conf, sem_conf]
        if num_conf is not None:
            confidence_components.append(num_conf)
        confidence_components.append(src_conf)

        final_conf = np.mean(confidence_components)

        
        # Display results
        render_dashboard(chosen_primary, final_conf, sem_conf, num_conf, web_context, base_conf, src_conf)
        
        # Debug info
        with st.expander("🔍 Debug Information"):
            st.write("**Primary Response (Perplexity):**")
            st.code(chosen_primary, language="json")
            st.write("**Validation Response (Gemini):**")
            st.code(secondary_resp, language="json")
            st.write(f"**All Confidence Scores:** {scores}")
            st.write(f"**Selected Best Score:** {base_conf}")
            if web_context:
                st.write(f"**Web Sources Found:** {len(web_context.get('search_results', []))}")

# ----------------------------
# RUN STREAMLIT
# ----------------------------
if __name__ == "__main__":
    main()
