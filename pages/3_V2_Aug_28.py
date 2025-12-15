import streamlit as st
import json
import pandas as pd
import plotly.express as px
from collections import OrderedDict

# ======================================================================
# 1. JSON DATA FOR V1 (JULY 2025 ESTIMATE)
# ======================================================================
# This is the conservative, previous estimate data (V1)
V0_JULY_2024_JSON = {
  "version": "V0 (Initial Estimate - July 2024)",
  "timestamp": "2024-07-01T09:00:00.000000",
  "executive_summary": "The global electric vehicle (EV) market is in an early stage of rapid growth, with current projections placing the 2024 market valuation near USD 550–600 billion. Annual unit sales are projected to reach only 12 million in 2024. Growth is heavily dependent on overcoming policy uncertainty and scaling battery production. Forecasts are cautious, projecting a single-digit to low double-digit long-term CAGR.",
  "primary_metrics": {
    "metric_1": {
      "name": "Global market value (2024 estimate)",
      "value": 580.0,
      "unit": "$B"
    },
    "metric_2": {
      "name": "Global market value (2025 estimate, common range)",
      "value": 750.0,
      "unit": "$B"
    },
    "metric_3": {
      "name": "Global electric car sales",
      "value": 12,
      "unit": "million units"
    },
    "metric_4": {
      "name": "Global EV sales (H1 2024)",
      "value": 6,
      "unit": "million units"
    },
    "metric_5": {
      "name": "Projected CAGR (mid 2020s to 2030s)",
      "value": 9,
      "unit": "%"
    }
  },
  "key_findings": [
    "Market valuation for 2024 was set lower due to recession fears and macroeconomic uncertainty.",
    "Global unit sales in 2024 were conservatively projected at 12 million units, reflecting weak consumer adoption in North America.",
    "Battery supply chain bottlenecks and high raw material costs were the primary concerns affecting forecast growth.",
    "PHEVs were still seen as a major transitional technology, expected to maintain a large share of new sales.",
    "Confidence in achieving long-term double-digit growth remained tenuous without further policy interventions."
  ],
  "top_entities": [
    {
      "name": "China (market & manufacturers)",
      "share": "≈40% of global sales",
      "growth": "Moderate, facing slowing domestic demand"
    },
    {
      "name": "Europe (EU markets)",
      "share": "≈18% of new sales share",
      "growth": "Slow, due to initial subsidy phase-outs"
    },
    {
      "name": "United States",
      "share": "≈5% new sales share",
      "growth": "Very slow/stagnant"
    },
    {
      "name": "Established OEMs (Legacy brands)",
      "share": "Dominant in some regions, but struggling with EV rollout",
      "growth": "Slow EV volume expansion"
    }
  ],
  "trends_forecast": [
    {
      "trend": "PHEV sales will rise to provide range flexibility and mitigate charging infrastructure risks",
      "direction": "↑ (PHEV share)",
      "timeline": "2024-2026"
    },
    {
      "trend": "Focus on low-cost, smaller battery chemistries (LFP) to drive down vehicle prices",
      "direction": "↑ (adoption)",
      "timeline": "2024-2027"
    }
  ],
  "visualization_data": {
    "chart_labels": [
      "2023",
      "2024",
      "2025",
      "2026",
      "2027"
    ],
    "chart_values": [
      600,
      580,
      750,
      950,
      1150
    ],
    "chart_title": "Estimated Global EV Market Value (USD billions, illustrative, V0)",
    "chart_type": "line"
  },
  "sources": [
    "iea.org Global EV Outlook 2023",
    "marketsandmarkets.com 2024 Q1 report",
    "bnef.com EV market outlook, Q2 2024"
  ],
  "confidence": 65,
  "freshness": "Jul 2024",
  "action": {
    "recommendation": "Sell/Hold",
    "confidence": "Low",
    "rationale": "High policy and economic uncertainty limits near-term growth potential and presents downside risk for manufacturers."
  }
}

# ======================================================================
# 2. STREAMLIT RENDERING FUNCTIONS
# ======================================================================

def render_metrics(metrics_data):
    """Renders key metrics using columns."""
    st.header("Key Market Metrics")
    metrics_list = list(metrics_data.values())
    
    # Use columns to display metrics horizontally
    cols = st.columns(len(metrics_list))
    for i, metric in enumerate(metrics_list):
        with cols[i]:
            st.metric(
                label=metric['name'],
                value=f"{metric['value']:,.1f}",
                delta=metric['unit']
            )

def render_chart(viz_data):
    """Renders a Plotly line chart."""
    df_chart = pd.DataFrame({
        'Year': viz_data['chart_labels'],
        'Value ($B)': viz_data['chart_values']
    })
    
    fig = px.line(
        df_chart,
        x='Year',
        y='Value ($B)',
        title=viz_data['chart_title'],
        markers=True,
        height=400
    )
    fig.update_layout(
        title_font_size=18, 
        xaxis_title="Year", 
        yaxis_title="Market Value (USD Billions)"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_entity_table(entities_data):
    """Renders the top entities/regions table."""
    st.header("Top Entities & Regions")
    
    # Convert list of dictionaries to a DataFrame
    df_entities = pd.DataFrame(entities_data)
    # Rename columns for display
    df_entities.columns = ["Region/Entity", "Market Share", "Growth Trajectory"]
    
    st.dataframe(df_entities, hide_index=True, use_container_width=True)


def render_dashboard(data):
    """Main function to render the full dashboard."""
    st.title(f"Electric Vehicle Market Analysis: {data['version']}")
    
    st.markdown(f"**Date of Analysis:** {pd.to_datetime(data['timestamp']).strftime('%B %d, %Y')}")
    st.markdown(f"**Overall Confidence:** **{data['confidence']}%**")
    st.markdown(f"**Freshness:** {data['freshness']}")

    # --- Section 1: Executive Summary & Metrics ---
    st.subheader("Executive Summary")
    st.info(data['executive_summary'])
    
    render_metrics(data['primary_metrics'])
    
    st.markdown("---")

    # --- Section 2: Key Findings & Action ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Key Findings")
        for i, finding in enumerate(data['key_findings']):
            st.markdown(f"**{i+1}.** {finding}")

    with col2:
        st.header("Recommendation")
        st.success(f"**Recommendation:** {data['action']['recommendation']}")
        st.markdown(f"**Confidence:** {data['action']['confidence']}")
        st.markdown(f"**Rationale:** {data['action']['rationale']}")

    st.markdown("---")

    # --- Section 3: Visualization ---
    st.header("Market Forecast & Trends")
    render_chart(data['visualization_data'])

    # --- Section 4: Entities and Trends ---
    col3, col4 = st.columns(2)
    
    with col3:
        render_entity_table(data['top_entities'])

    with col4:
        st.header("Trends & Forecast")
        for trend in data['trends_forecast']:
            st.markdown(
                f"**{trend['trend']}** (`{trend['direction']}` for {trend['timeline']})"
            )

    # --- Section 5: Sources ---
    with st.expander("View Sources"):
        st.markdown("**Sources Used:**")
        for source in data['sources']:
            st.markdown(f"- {source}")

# ======================================================================
# 3. RUN STREAMLIT APP
# ======================================================================

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_dashboard(V0_JULY_2024_JSON)

# How to run this script:
# 1. Save the code above as a Python file (e.g., render_v1_json.py).
# 2. Open your terminal and navigate to the directory where you saved the file.
# 3. Run the command: streamlit run render_v1_json.py
