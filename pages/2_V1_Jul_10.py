import streamlit as st
import json
import pandas as pd
import plotly.express as px
from collections import OrderedDict

# ======================================================================
# 1. JSON DATA FOR V1 (JULY 2025 ESTIMATE)
# ======================================================================
# This is the conservative, previous estimate data (V1)
V1_JULY_2025_JSON = {
    "version": "V1 (Previous Estimate - July 2025)",
    "timestamp": "2025-07-01T09:00:00.000000",
    "executive_summary": "The global electric vehicle (EV) market continues to expand, though current projections place the 2024-2025 market valuation near USD 650–850 billion, with annual unit sales estimated around 15–16 million units. Growth is hampered by persistent charging infrastructure gaps and slowing consumer demand in certain markets. Forecasts suggest steady, but more moderate, double-digit CAGR.",
    "primary_metrics": {
        "metric_1": {
            "name": "Global market value (2024 estimate)",
            "value": 650.0,
            "unit": "$B"
        },
        "metric_2": {
            "name": "Global market value (2025 estimate, common range)",
            "value": 850.0,
            "unit": "$B"
        },
        "metric_3": {
            "name": "Global electric car sales",
            "value": 15,
            "unit": "million units"
        },
        "metric_4": {
            "name": "Global EV sales (H1 2025)",
            "value": 8,
            "unit": "million units"
        },
        "metric_5": {
            "name": "Projected CAGR (mid 2020s to 2030s)",
            "value": 10,
            "unit": "%"
        }
    },
    "key_findings": [
        "Market value estimates were clustered around the $650B to $850B range for 2024–2025, reflecting uncertainty in consumer demand.",
        "Global unit sales reached 15 million units in 2024, representing solid but decelerating year-on-year growth.",
        "Charging infrastructure gaps remained a significant barrier to widespread adoption outside of major metropolitan areas.",
        "Policy support was viewed as inconsistent, particularly in the United States, contributing to mixed growth forecasts.",
        "China was still the market leader but domestic competition was expected to slow profit margins."
    ],
    "top_entities": [
        {
            "name": "China (market & manufacturers)",
            "share": "≈45% of global sales",
            "growth": "High, but with saturation risks"
        },
        {
            "name": "Europe (EU markets)",
            "share": "≈20% of new sales share",
            "growth": "Moderate, subject to subsidy stability"
        },
        {
            "name": "United States",
            "share": "≈7% new sales share",
            "growth": "Slow, driven by light truck preference"
        },
        {
            "name": "Tesla / Established OEMs",
            "share": "Leading volume, but facing new competition",
            "growth": "Moderate volume expansion"
        }
    ],
    "trends_forecast": [
        {
            "trend": "Focus shifts to plug-in hybrids (PHEVs) as consumers prioritize range flexibility over pure BEVs",
            "direction": "↑ (PHEV share)",
            "timeline": "2025-2026"
        },
        {
            "trend": "Increased geopolitical focus on mineral supply chains, slowing battery cost declines",
            "direction": "→",
            "timeline": "2025-2027"
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
            650,
            850,
            1050,
            1250
        ],
        "chart_title": "Estimated Global EV Market Value (USD billions, illustrative, V1)",
        "chart_type": "line"
    },
    "sources": [
        "iea.org Global EV Outlook 2024",
        "marketsandmarkets.com 2024 Q2 report",
        "bnef.com EV market outlook, H1 2025"
    ],
    "confidence": 75,
    "freshness": "Jul 2025",
    "action": {
        "recommendation": "Hold",
        "confidence": "Medium-Low",
        "rationale": "Market growth is solid but faces near-term demand elasticity issues and structural infrastructure bottlenecks."
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
    render_dashboard(V1_JULY_2025_JSON)

# How to run this script:
# 1. Save the code above as a Python file (e.g., render_v1_json.py).
# 2. Open your terminal and navigate to the directory where you saved the file.
# 3. Run the command: streamlit run render_v1_json.py
