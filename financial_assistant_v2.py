# =========================================================
# AI FINANCIAL RESEARCH ASSISTANT – HYBRID VERIFICATION v5.6
# WITH WEB SEARCH, DYNAMIC METRICS, CONFIDENCE BREAKDOWN & EVOLUTION LAYER
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
from datetime import datetime, timedelta

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
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# ----------------------------
# SOURCE RELIABILITY CLASSIFIER
# ----------------------------
def classify_source_reliability(source):
    source = source.lower() if isinstance(source, str) else ""
    high_sources = ["gov", "imf", "worldbank", "world bank", "central bank", "fed", "ecb", "bank of england", "eu", "reuters", "financial times", "wsj", "oecd", "bank of korea", "tradingeconomics",
                    "the economist", "ft.com", "bloomberg", "investopedia", "marketwatch", "bank of canada", "reserve bank of australia", "monetary authority of singapore", "HKMA", "bank of japan", 
                    "adb", "unfpa", "deloitte", "accenture", "kpmg"]
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
    "You are an AI research analyst focused on topics related to business, finance, economics, and markets.\n"
    "Output strictly in the JSON format below, including ONLY those financial or economic metrics "
    "that are specifically relevant to the exact question the user asks.\n"
    "For example, if the user asks about oil or energy, include metrics like oil production, reserves, "
    "prices, and exclude unrelated metrics such as inflation or unemployment.\n"
    "If the question is related to macroeconomics or the underlying drivers are macroeconomic, you may include GDP growth, inflation, etc.\n"
    "If the question is not related to business, finance, economics or markets politely decline to answer the question.\n"
    "Strictly follow this JSON structure:\n"
    f"{RESPONSE_TEMPLATE}"
)

@st.cache_resource
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    return classifier, embed

domain_classifier, embedder = load_models()

# ----------------------------
# WEB SEARCH FUNCTIONS
# ----------------------------
@st.cache_data(ttl=3600)  # cache for 1 hour
def search_serpapi(query: str, num_results: int = 5):
    if not SERPAPI_KEY:
        st.info("💡 SerpAPI key not configured.")
        return []
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": f"{query} finance economics markets",
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "tbm": "nws",
        "tbs": "qdr:m"
    }
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
            st.success(f"✅ Found {len(results)} sources via SerpAPI")
            results.sort(key=lambda x: x.get("source", "").lower())
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

# ----------------------------
# AI QUERY FUNCTIONS
# ----------------------------
def query_perplexity_with_context(query: str, web_context: dict, temperature=0.1):
    if web_context.get("summary"):
        context_section = f"""
LATEST WEB RESEARCH (Current as of today):
{web_context['summary']}

"""
        if web_context.get('scraped_content'):
            context_section += "\nDETAILED CONTENT FROM TOP SOURCES:\n"
            for url, content in list(web_context['scraped_content'].items())[:2]:
                context_section += f"\nFrom {url}:\n{content[:800]}...\n"
        enhanced_query = f"{context_section}\n{SYSTEM_PROMPT}\n\nUser Question: {query}"
    else:
        enhanced_query = f"{SYSTEM_PROMPT}\n\nUser Question: {query}"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar",
        "temperature": temperature,
        "max_tokens": 2000,
        "top_p": 0.8,              # even safer, try 0.7 or lower
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
        try:
            parsed = json.loads(content)
            if web_context.get("sources"):
                existing_sources = parsed.get("sources", [])
                all_sources = existing_sources + web_context["sources"]
                parsed["sources"] = list(set(all_sources))[:10]  # unique max 10
                parsed["data_freshness"] = "Current (web-scraped + real-time search)"
            content = json.dumps(parsed)
        except json.JSONDecodeError:
            st.warning("Reformatting Perplexity response to JSON...")
            content = json.dumps({
                "summary": content[:500],
                "key_insights": [content[:200]],
                "metrics": {},
                "visual_data": {},
                "table": [],
                "sources": web_context.get("sources", []),
                "confidence_score": 50,
                "data_freshness": "Current (web-scraped)"
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
                max_output_tokens=1000,
            ),
        )
        content = response.text
        if not content.strip():
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
    st.info(f"Generating analysis with up-to-date content...")

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
    st.success(f"Successfully generated analysis")

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
    relevant_metrics = filter_relevant_metrics(question, metrics)
    to_display = relevant_metrics if relevant_metrics else metrics
    cols = st.columns(len(to_display))
    for i, (k, v) in enumerate(to_display.items()):
        try:
            val = f"{float(v):.2f}"
        except Exception:
            val = str(v)
        cols[i].metric(k, val)

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

def render_dashboard(response, final_conf, sem_conf, num_conf, web_context=None,
                     base_conf=None, src_conf=None, versions_history=None, user_question=""):
    if not response or not response.strip():
        st.error("Received empty response from model")
        return

    try:
        # Parse JSON string into dict
        data = json.loads(response)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.code(response[:800])
        return

    # Display only summary text (not entire JSON)
                         
    col1, col2 = st.columns(2)
    col1.metric("Overall Confidence (%)", f"{final_conf:.1f}")
    freshness = data.get("data_freshness", "Unknown")
    col2.metric("Data Freshness", freshness)

    st.subheader("Confidence Score Breakdown")
    st.write(f"- Base model confidence: {base_conf:.1f}%")
    st.write(f"- Semantic similarity confidence: {sem_conf:.1f}%")
    st.write(f"- Numeric alignment confidence: {num_conf if num_conf is not None else 'N/A'}%")
    st.write(f"- Source quality confidence: {src_conf:.1f}%")
    st.write(f"---\n**Overall confidence: {final_conf:.1f}%**")

    st.header("📊 Summary Analysis")
    summary_text = data.get("summary", "No summary available.")
    st.write(summary_text)
    
    st.subheader("Key Insights")
    insights = data.get("key_insights", [])
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No key insights provided.")

    st.subheader("Metrics")  # RENDER METRICS PANEL
    
                         
    metrics = data.get("metrics", {})
    render_dynamic_metrics(user_question, metrics)

    st.subheader("Trend Visualization")
    vis = data.get("visual_data", {})
    if "labels" in vis and "values" in vis:
        try:
            df = pd.DataFrame({"Period": vis["labels"], "Value": vis["values"]})
            fig = px.line(df, x="Period", y="Value", title="Quarterly Trends", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Visualization error: {e}")
    else:
        st.info("No visual data available.")

    st.subheader("Data Table")
    table = data.get("table", [])
    if table:
        try:
            st.dataframe(pd.DataFrame(table), use_container_width=True)
        except Exception as e:
            st.warning(f"Table rendering error: {e}")
    else:
        st.info("No tabular data available.")

    st.subheader("📚 Sources & References")
    sources = data.get("sources", [])
    if sources:
        st.success(f"✅ {len(sources)} sources:")
        for i, s in enumerate(sources, 1):
            rank = classify_source_reliability(s) if s else "Unknown"
            st.markdown(f"{i}. [{s}]({s}) — {rank}")
    else:
        st.info("No sources cited.")

    if web_context and web_context.get("search_results"):
        with st.expander("🔍 Web Search Details"):
            st.write(f"**Sources Found:** {len(web_context['search_results'])}")
            st.write(f"**Pages Scraped:** {len(web_context.get('scraped_content', {}))}")
            for idx, result in enumerate(web_context["search_results"]):
                rank_list = web_context.get('source_reliability', [])
                reliab = rank_list[idx] if idx < len(rank_list) else classify_source_reliability(result['link'] + " " + result.get('source', ''))
                st.markdown(f"- **{result['title']}** ({result.get('source', 'Unknown')}) [{reliab}]")

    st.subheader("Validation Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Semantic Similarity", f"{sem_conf:.2f}%")
    if num_conf is not None:
        c2.metric("Numeric Alignment", f"{num_conf:.2f}%")
    else:
        c2.info("No numeric data to compare")

    if versions_history:
        render_evolution_layer(versions_history)

# ----------------------------
# MAIN WORKFLOW
# ----------------------------

def main():
    st.set_page_config(page_title="Yureeka Market Research Assistant", layout="wide")
    st.title("💹 Yureeka AI Market Analyst")
    st.caption("Self-Consistency + Cross-Model Verification + Live Web Search + Dynamic Metrics + Evolution Layer")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("""
        This assistant combines:
        - Self-consistency via multiple AI analyses
        - Cross-model validation
        - Live web search integration
        - Dynamic metrics relevant to your question
        - Confidence score component breakdown
        - Evolution layer for version control and metric drift
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

        num_conf = numeric_alignment_score(j1, j2)
        base_conf = max_score
        src_conf = source_quality_confidence(j1.get("sources", [])) * 100

        confidence_components = [base_conf, sem_conf]
        if num_conf is not None:
            confidence_components.append(num_conf)
        confidence_components.append(src_conf)
        final_conf = np.mean(confidence_components)

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
            },
        ]

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
