# =========================================================
# AI FINANCIAL RESEARCH ASSISTANT – HYBRID VERIFICATION v5.2
# DEBUGGED VERSION with Streamlit Cloud Support
# Combines:
#   - Self‑consistency reasoning (Perplexity Sonar)
#   - Cross‑model validation (Gemini 2.0 Flash)
# With comprehensive error handling and bug fixes.
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

# ----------------------------
# STEP 1: CONFIGURATION
# ----------------------------
# Try Streamlit secrets first, fallback to environment variables
try:
    PERPLEXITY_KEY = st.secrets["PERPLEXITY_API_KEY"]
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except:
    PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not PERPLEXITY_KEY or not GEMINI_KEY:
    st.error("Missing API keys. Please set PERPLEXITY_API_KEY and GEMINI_API_KEY in Streamlit secrets or environment variables.")
    st.stop()

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Configure Gemini properly
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

RESPONSE_TEMPLATE = """
You are a research assistant. Return ONLY valid JSON formatted as:
{
  "summary": "Brief summary of findings.",
  "key_insights": ["Insight 1", "Insight 2"],
  "metrics": {"GDP Growth (%)": number, "Inflation (%)": number, "Unemployment (%)": number},
  "visual_data": {"labels": ["Q1","Q2"], "values": [2.3,2.5]},
  "table": [{"Country": "US", "GDP": 25.5, "Inflation": 3.4}],
  "sources": ["https://imf.org", "https://reuters.com"],
  "confidence_score": 85
}
"""

SYSTEM_PROMPT = (
    "You are an AI research analyst that only answers topics related to business, finance, economics, or markets.\n"
    "Output strictly in the JSON structure below:\n"
    f"{RESPONSE_TEMPLATE}"
)

#SYSTEM_PROMPT = (

#"You are an AI research analyst that only answers topics related to business, finance, economics, or markets.\n"

#"Before producing any output, evaluate the user’s query:\n"
#"- If the query is clearly about business, finance, economics, or markets, proceed normally.\n"
#"- If the query is unrelated to these topics, respond with a short, polite JSON‑formatted message that declines the request and briefly reminds the user which subjects you cover.\n"

#"You must never attempt to answer or speculate on topics outside this approved list.\n"

#"Output strictly in the following JSON structure for all cases:\n"
#{
#  "status": "accepted" or "declined",
#  "category": "<identified domain if applicable>",
#  "response": f"{RESPONSE_TEMPLATE}"
#}
#)



# Cache model loading to avoid reloading on every Streamlit rerun
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
# STEP 2: CORE HELPERS
# ----------------------------
def is_finance_query(query: str, threshold=0.65):
    """Check if query is related to finance/economics"""
    try:
        r = domain_classifier(query, ALLOWED_TOPICS)
        return r["scores"][0] >= threshold
    except Exception as e:
        st.warning(f"Domain classification error: {e}")
        return True  # Allow query to proceed if classifier fails

def query_perplexity(query: str, temperature=0.7):
    """Query Perplexity API with proper error handling"""
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}", 
        "Content-Type": "application/json"
    }
    
    # Updated payload - Perplexity doesn't support system role in the same way
    payload = {
        "model": "sonar",
        "temperature": temperature,
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nUser Question: {query}"}
        ]
    }
    
    try:
        resp = requests.post(
            PERPLEXITY_URL, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        # Debug: Show response status and content if error
        if resp.status_code != 200:
            error_detail = resp.text
            st.error(f"Perplexity API Error {resp.status_code}: {error_detail}")
            raise Exception(f"Perplexity returned {resp.status_code}: {error_detail}")
        
        resp.raise_for_status()
        response_data = resp.json()
        
        # Debug: Check response structure
        if "choices" not in response_data:
            st.error(f"Unexpected response structure: {response_data}")
            raise Exception(f"No 'choices' in response: {response_data}")
        
        content = response_data["choices"][0]["message"]["content"]
        
        # Validate content is not empty
        if not content or not content.strip():
            raise Exception("Perplexity returned empty response")
        
        # Try to parse as JSON to validate format
        try:
            json.loads(content)
        except json.JSONDecodeError:
            # If not valid JSON, wrap it in a structured format
            st.warning("Perplexity returned non-JSON response, reformatting...")
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
    except requests.exceptions.Timeout:
        raise Exception("Perplexity API timeout after 30 seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Perplexity API request error: {e}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected Perplexity response structure: {e}")

def query_gemini(query: str):
    """Query Gemini API with proper error handling"""
    prompt = f"{SYSTEM_PROMPT}\n\nUser query: {query}"
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )
        content = response.text
        
        # Validate content is not empty
        if not content or not content.strip():
            raise Exception("Gemini returned empty response")
        
        # Try to parse as JSON to validate format
        try:
            json.loads(content)
        except json.JSONDecodeError:
            # If not valid JSON, wrap it in a structured format
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
        # Return valid JSON fallback
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
# STEP 3: SELF‑CONSISTENCY PROMPTING
# ----------------------------
def generate_self_consistent_responses(query, n=5):
    """Generate multiple responses and track alignment"""
    st.info(f"Generating {n} independent Perplexity analyst responses...")
    responses, scores = [], []
    success_count = 0
    
    for i in range(n):
        try:
            r = query_perplexity(query, temperature=0.8)
            responses.append(r)
            scores.append(parse_confidence(r))
            success_count += 1
        except Exception as e:
            st.warning(f"Attempt {i+1}/{n} failed: {e}")
            # Don't add empty responses - maintain alignment
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
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return 0.0

# ----------------------------
# STEP 4: VALIDATION FUNCTIONS
# ----------------------------
def semantic_similarity_score(a, b):
    """Calculate semantic similarity between two texts"""
    try:
        v1, v2 = embedder.encode([a, b])
        sim = util.cos_sim(v1, v2)
        # Extract scalar value properly from tensor
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
                
                # Handle zero values
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
# STEP 5: DASHBOARD RENDERER
# ----------------------------
def render_dashboard(response, final_conf, sem_conf, num_conf):
    """Render the financial dashboard with results"""
    
    # Validate response is not empty
    if not response or not response.strip():
        st.error("Received empty response from model")
        return
    
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON returned by model: {e}")
        st.write("**Raw response received:**")
        st.code(response[:1000], language="text")  # Show first 1000 chars
        st.info("💡 **Troubleshooting tips:**")
        st.markdown("""
        - The API may not be following the JSON format instructions
        - Try rephrasing your question to be more specific
        - Check API key validity and rate limits
        - The model might be returning plain text instead of JSON
        """)
        return

    # Confidence score display
    st.metric("Overall Confidence (%)", f"{final_conf:.1f}")
    
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

    # Sources
    st.subheader("Sources")
    sources = data.get("sources", [])
    if sources:
        for s in sources:
            st.markdown(f"- [{s}]({s})")
    else:
        st.info("No sources cited.")

    # Validation metrics
    st.subheader("Validation Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Semantic Similarity", f"{sem_conf:.2f}%")
    if num_conf is not None:
        col2.metric("Numeric Alignment", f"{num_conf:.2f}%")
    else:
        col2.info("No numeric data to compare")

# ----------------------------
# STEP 6: MAIN WORKFLOW
# ----------------------------
def main():
    st.set_page_config(
        page_title="Yureek Market Research Assistant", 
        layout="wide"
    )
    st.title("💹 Yureeka AI-assisted Market Analyst – Self‑Consistency + Cross‑Model Verification")
    
    st.markdown("""
    This assistant combines:
    - **Self-consistency reasoning** via multiple Perplexity analyses
    - **Cross-model validation** using Gemini 2.0 Flash
    - **Semantic and numeric alignment** scoring
    """)

    q = st.text_input("Enter your question about markets, finance, or economics:")
   # n_paths = st.slider(
   #     "Number of self‑consistent analysts (Perplexity)", 
   #     3, 10, 5
   # )

    if st.button("Analyze") and q:
        # Domain validation
       # if not is_finance_query(q):
       #     st.error("❌ Query not recognized as being relevant to finance, markets or business. Please reword your question.")
       #     return

        # --- Self‑Consistency Stage ---
        responses, scores = generate_self_consistent_responses(q, 3) # num paths = 3
        
        if not responses or not scores:
            st.error("Primary model failed to generate any valid responses.")
            return
        
        # Validate alignment
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

        # --- Independent Validation Stage ---
        st.info("Cross‑verifying via Gemini 2.0 Flash...")
        secondary_resp = query_gemini(q)

        # --- Scoring ---
        sem_conf = semantic_similarity_score(chosen_primary, secondary_resp)
        
        # Parse JSON safely
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
        confidence_components = [base_conf, sem_conf]
        if num_conf is not None:
            confidence_components.append(num_conf)
        
        final_conf = np.mean(confidence_components)

        # --- Display ---
        render_dashboard(chosen_primary, final_conf, sem_conf, num_conf)
        
        # Debug info (collapsible)
        with st.expander("🔍 Debug Information"):
            st.write("**Primary Response (Perplexity):**")
            st.code(chosen_primary, language="json")
            st.write("**Validation Response (Gemini):**")
            st.code(secondary_resp, language="json")
            st.write(f"**All Confidence Scores:** {scores}")
            st.write(f"**Selected Best Score:** {base_conf}")

# ----------------------------
# RUN STREAMLIT
# ----------------------------
if __name__ == "__main__":
    main()
