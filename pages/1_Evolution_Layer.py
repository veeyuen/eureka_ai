import streamlit as st

st.title("Evolution Layer – Debug")

st.write("If you see this page, multipage is working in the cloud.")

if "versions_history" in st.session_state:
    st.write("versions_history found in session_state.")
    st.json(st.session_state["versions_history"])
else:
    st.info("No versions_history in session_state yet. Run an analysis on the main page first.")

