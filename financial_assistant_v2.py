# =========================================================
# AI FINANCIAL RESEARCH ASSISTANT – HYBRID VERIFICATION v5.5
# WITH WEB SEARCH, DYNAMIC METRICS, CONFIDENCE BREAKDOWN, AND EVOLUTION LAYER
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

# STEP 1: CONFIGURATION
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
    st.error("Missing API keys. Set PERPLEXITY_API_KEY and GEMINI_API_KEY.")
    st.stop()

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# SOURCE RELIABILITY CLASSIFIER
def classify_source_reliability(source):
    source = source.lower() if isinstance(source, str) else ""
    high_sources = ["gov", "imf", "worldbank", "world bank", "central bank", "fed", "ecb", "bank of england"]
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
    return "⚠️ Medium"

def source_quality_confidence(sources):
    weights = {"✅ High": 1.0, "⚠️ Medium": 0.6, "❌ Low": 0.3}
    total, count = 0, 0
    for src in sources:
        rank = classify_source_reliability(src)
        total += weights.get(rank, 0.6)
        count += 1
    return total / count if count else 0.6

# PROMPT CONSTANTS
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
    "You are an AI analyst answering about finance, economics, and markets. "
    "Only include metrics relevant to the question, strictly following the JSON format:\n"
    f"{RESPONSE_TEMPLATE}"
)

@st.cache_resource
def load_models():
    cls = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    return cls, emb

domain_classifier, embedder = load_models()

# Web search and scraping functions (search_serpapi, scrape_url_scrapingdog, fetch_web_context)
# [Identical to earlier but with source defensive parsing]

# AI query functions with temperature 0.2 for stability: query_perplexity_with_context, query_gemini

# Self consistency and validation functions as before

# Dynamic metrics filtering and display based on user question
def filter_relevant_metrics(question, metrics):
    relevant_metrics = {}
    for m in metrics:
        res = domain_classifier(question, [m], multi_label=False)
        score = res['scores'][0] if 'scores' in res else 0
        if score > 0.5:
            relevant_metrics[m] = metrics[m]
    return relevant_metrics

def render_dynamic_metrics(question, metrics):
    if not metrics:
        st.info("No metrics available.")
        return
    filtered = filter_relevant_metrics(question, metrics)
    to_show = filtered if filtered else metrics
    cols = st.columns(len(to_show))
    for i, (k,v) in enumerate(to_show.items()):
        try:
            val = f"{float(v):.2f}"
        except:
            val = str(v)
        cols[i].metric(k, val)

# Evolution layer helpers: time_ago, display_metric_with_delta, render_evolution_layer
# [Same structured as before]

# Updated render_dashboard with dynamic metrics and confidence breakdown display,
# plus evolution layer rendering if versions_history is provided.
def render_dashboard(response, final_conf, sem_conf, num_conf, web_context=None, base_conf=None, src_conf=None, versions_history=None, user_question=""):
    if not response or not response.strip():
        st.error("Received empty response")
        return
    try:
        data = json.loads(response)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.code(response[:800])
        return

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

    st.header("📊 Financial Summary")
    st.write(data.get("summary", "No summary available."))

    st.subheader("Key Insights")
    insights = data.get("key_insights", [])
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No key insights provided.")

    st.subheader("Metrics")
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

# MAIN WORKFLOW
def main():
    st.set_page_config(page_title="Yureeka Market Research Assistant", layout="wide")
    st.title("💹 Yureeka AI Market Analyst")
    st.caption("Self-Consistency + Cross-Model Validation + Web Search + Evolution Layer + Dynamic Metrics")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        This assistant combines:
        - Self-consistency reasoning via multiple analyses
        - Cross-model validation
        - Live web search
        - Dynamic metrics tailored to question
        - Confidence score breakdown
        - Evolution layer for versioning and metric drift
        """)
    with col2:
        web_status = "✅ Enabled" if SERPAPI_KEY else "⚠️ Not configured"
        st.metric("Web Search", web_status)

    q = st.text_input("Enter your question about markets, finance, or economics:")
    use_web_search = st.checkbox("Enable live web search (recommended)", value=bool(SERPAPI_KEY), disabled=not SERPAPI_KEY)

    if st.button("Analyze") and q:
        web_context = {}
        if use_web_search:
            with st.spinner("Searching the web for latest info..."):
                web_context = fetch_web_context(q, num_sources=3)

        if web_context.get("search_results"):
            responses, scores = generate_self_consistent_responses_with_web(q, web_context, n=3)
        else:
            st.info("Using AI model knowledge only...")
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
            st.error("Could not determine primary response.")
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

        # Example version history (replace with actual stored versions)
        versions_history = [
            {
                "version": "V1 (Jul 10)",
                "timestamp": "2025-07-10T12:00:00",
                "metrics": j1.get("metrics", {}),
                "confidence": base_conf,
                "sources_freshness": 80,
                "change_reason": "Initial version"
            },
            {
                "version": "V2 (Aug 28)",
                "timestamp": "2025-08-28T15:30:00",
                "metrics": j1.get("metrics", {}),
                "confidence": base_conf * 0.98,
                "sources_freshness": 75,
                "change_reason": "Quarterly update"
            },
            {
                "version": "V3 (Nov 3)",
                "timestamp": datetime.now().isoformat(timespec='minutes'),
                "metrics": j1.get("metrics", {}),
                "confidence": final_conf,
                "sources_freshness": 78,
                "change_reason": "Latest analysis"
            },
        ]

        render_dashboard(
            chosen_primary, final_conf, sem_conf, num_conf, web_context,
            base_conf, src_conf, versions_history, user_question=q
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
