import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 V2 - Aug 28, 2025")

version_data = {
    "timestamp": "2025-08-28T15:30:00",
    "summary": "Q3 update: GDP growth accelerates to 2.4%. Inflation ticks up to 3.4%. Fed signals rate pause.",
    "metrics": {"GDP Growth (%)": 2.4, "Inflation (%)": 3.4, "Oil Price (USD)": 82.5},
    "confidence": 82.1,
    "sources_freshness": 75,
    "change_reason": "Quarterly update with Fed commentary"
}

st.markdown("**Summary Analysis**")
st.write(version_data["summary"])

st.subheader("Comparative Table")
table_data = [
    {"Metric": "GDP Growth", "V1 (Jul)": "2.1%", "V2 (Aug)": "2.4%"},
    {"Metric": "Inflation", "V1 (Jul)": "3.2%", "V2 (Aug)": "3.4%"},
    {"Metric": "Oil Price", "V1 (Jul)": "$75", "V2 (Aug)": "$82.5"}
]
st.table(pd.DataFrame(table_data))

st.caption(f"**Confidence:** {version_data['confidence']:.1f}% | **Freshness:** {version_data['sources_freshness']}%")

