import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.title("🕐 V1 - Jul 10, 2025")

# Demo data for this version
version_data = {
    "timestamp": "2025-07-10T12:00:00",
    "summary": "Initial market analysis showing stable GDP growth at 2.1%. Oil prices steady at $75/bbl.",
    "metrics": {"GDP Growth (%)": 2.1, "Inflation (%)": 3.2, "Oil Price (USD)": 75.0},
    "confidence": 78.5,
    "sources_freshness": 80,
    "change_reason": "Initial baseline analysis"
}

st.markdown("**Summary Analysis**")
st.write(version_data["summary"])

st.subheader("📊 Key Metrics")
cols = st.columns(3)
for i, (k, v) in enumerate(version_data["metrics"].items()):
    cols[i].metric(k, f"{v:.1f}")

st.subheader("Trend Chart")
demo_data = {"Q1": 1.8, "Q2": 2.0, "Q3": 2.1}
df = pd.DataFrame(list(demo_data.items()), columns=["Quarter", "GDP Growth"])
fig = px.line(df, x="Quarter", y="GDP Growth", markers=True, title="GDP Growth Evolution")
st.plotly_chart(fig, use_container_width=True)

st.caption(f"**Updated:** 4 months ago | Confidence: {version_data['confidence']:.1f}%")

