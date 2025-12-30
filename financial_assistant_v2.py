def render_source_anchored_results(results: Dict, question: str):
    """
    Streamlit renderer for source-anchored evolution output.
    - Guards against missing/None fields (esp. stability_score)
    - Displays metric changes, source fetch diagnostics, and source stability
    """

    st.subheader("🧬 Source-Anchored Evolution Results")
    st.caption(question or "")

    if not results or not isinstance(results, dict):
        st.error("❌ No evolution results to display.")
        return

    # ---- Safe read of core fields ----
    sources_checked = results.get("sources_checked", 0) or 0
    sources_fetched = results.get("sources_fetched", 0) or 0

    stability = results.get("stability_score", 0.0)
    try:
        stability_val = float(stability) if stability is not None else 0.0
    except Exception:
        stability_val = 0.0

    summary = results.get("summary", {}) or {}
    inc = int(summary.get("metrics_increased", 0) or 0)
    dec = int(summary.get("metrics_decreased", 0) or 0)
    same = int(summary.get("metrics_unchanged", 0) or 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sources Checked", sources_checked)
    col2.metric("Sources Fetched", sources_fetched)
    col3.metric("Stability", f"{stability_val:.1f}%")
    col4.metric("Stable Metrics", same)

    st.markdown("---")

    # ---- Optional: Source stability diagnostics ----
    ss = results.get("source_stability")
    if isinstance(ss, dict):
        st.subheader("🧷 Source Stability Diagnostics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exact URL matches", int(ss.get("exact_url_matches", 0) or 0))
        c2.metric("Domain matches", int(ss.get("domain_matches", 0) or 0))
        c3.metric("Identical fingerprints", int(ss.get("identical_fingerprints", 0) or 0))
        c4.metric("Fingerprints compared", int(ss.get("fingerprints_compared", 0) or 0))
        st.markdown("---")

    # ---- Metric changes table ----
    st.subheader("📌 Metric Changes")

    rows = results.get("metric_changes", []) or []
    if not rows:
        st.info("No metric changes computed.")
    else:
        table_rows = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            cp = r.get("change_pct")
            try:
                cp_str = f"{float(cp):+.1f}%" if cp is not None else "N/A"
            except Exception:
                cp_str = "N/A"

            conf = r.get("confidence")
            try:
                conf_str = f"{float(conf):.0f}%"
            except Exception:
                conf_str = "N/A"

            table_rows.append({
                "Metric": r.get("metric", "N/A"),
                "Prev": r.get("previous_value", "N/A"),
                "Current": r.get("current_value", "N/A"),
                "Δ%": cp_str,
                "Confidence": conf_str,
                "Notes": r.get("notes", ""),
                "Source": r.get("matched_source", "") or "",
            })

        default_n = 12
        df = pd.DataFrame(table_rows)
        st.dataframe(df.head(default_n), hide_index=True, width="stretch")

        if len(df) > default_n:
            with st.expander(f"Show all metric changes ({len(df)})"):
                st.dataframe(df, hide_index=True, width="stretch")

    st.markdown("---")

    # ---- Per-source diagnostics ----
    st.subheader("🔗 Source Fetch Diagnostics")
    srs = results.get("source_results", []) or []
    if not srs:
        st.info("No per-source results available.")
        return

    diag_rows = []
    for sr in srs:
        if not isinstance(sr, dict):
            continue
        diag_rows.append({
            "Source (raw)": sr.get("url", ""),
            "Normalized": sr.get("normalized_url", ""),
            "Status": sr.get("status", ""),
            "Detail": sr.get("status_detail", ""),
            "Numbers Found": int(sr.get("numbers_found", 0) or 0),
            "Fingerprint": sr.get("fingerprint", "") or "",
        })

    st.dataframe(pd.DataFrame(diag_rows), hide_index=True, width="stretch")

    with st.expander("Show extracted numbers (first 25 per source)"):
        for sr in srs:
            if not isinstance(sr, dict):
                continue
            st.markdown(f"**{sr.get('normalized_url') or sr.get('url')}**")
            nums = sr.get("extracted_numbers", []) or []
            if not nums:
                st.caption("No extracted numbers.")
                st.markdown("---")
                continue
            show = []
            for n in nums[:25]:
                if isinstance(n, dict):
                    show.append({
                        "raw": n.get("raw", ""),
                        "value": n.get("value", ""),
                        "unit": n.get("unit", ""),
                        "context": (n.get("context", "") or "")[:240],
                    })
            st.dataframe(pd.DataFrame(show), hide_index=True, width="stretch")
            st.markdown("---")
