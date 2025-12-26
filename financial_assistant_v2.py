def render_dashboard(
    primary_json: str,
    final_conf: float,
    web_context: Dict,
    base_conf: float,
    user_question: str,
    veracity_scores: Optional[Dict] = None,
    source_reliability: Optional[List[str]] = None,
):
    """Render the analysis dashboard"""

    # Parse primary response
    try:
        data = json.loads(primary_json)
    except Exception as e:
        st.error(f"❌ Cannot render dashboard: {e}")
        st.code(primary_json[:1000])
        return

    # -------------------------
    # Small local helper: robust metric value formatting (range-aware)
    # -------------------------
    def _format_metric_value(m: Dict) -> str:
        if not isinstance(m, dict):
            return "N/A"

        # Prefer deterministic span/range if available
        span = None
        try:
            span = get_metric_value_span(m)
        except Exception:
            span = None

        # If the metric already has a merged range structure, prefer that
        rng = m.get("range") if isinstance(m, dict) else None
        unit = (m.get("unit") or "").strip()

        # Case 1: explicit "range" dict (from canonicalize_metrics merge)
        if isinstance(rng, dict) and rng.get("min") is not None and rng.get("max") is not None:
            try:
                vmin = float(rng["min"])
                vmax = float(rng["max"])
                if vmin != vmax:
                    return f"{rng['min']}–{rng['max']} {unit}".strip()
            except Exception:
                # fall through
                pass

        # Case 2: value_span from get_metric_value_span
        if isinstance(span, dict) and span.get("min") is not None and span.get("max") is not None:
            try:
                vmin = float(span["min"])
                vmax = float(span["max"])
                u = (span.get("unit") or unit or "").strip()
                if vmin != vmax:
                    return f"{span['min']}–{span['max']} {u}".strip()
                # equal bounds -> show mid if present
                mid = span.get("mid")
                if mid is not None:
                    return f"{mid} {u}".strip()
            except Exception:
                pass

        # Default
        return f"{m.get('value', 'N/A')} {unit}".strip()

    # Header
    st.header("📊 Yureeka Market Report")
    st.markdown(f"**Question:** {user_question}")

    # Confidence row
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Confidence", f"{final_conf:.1f}%")
    col2.metric("Base Model", f"{base_conf:.1f}%")
    if veracity_scores:
        col3.metric("Evidence", f"{veracity_scores.get('overall', 0):.1f}%")
    else:
        col3.metric("Evidence", "N/A")

    st.markdown("---")

    # Executive Summary
    st.subheader("📋 Executive Summary")
    st.markdown(f"**{data.get('executive_summary', 'No summary available')}**")

    st.markdown("---")

    # =========================
    # Key Metrics (template-driven when available)
    # =========================
    st.subheader("💰 Key Metrics")
    metrics = data.get("primary_metrics", {}) or {}

    # 3B: Pull question signals/category into the same table output
    question_category = data.get("question_category") or (data.get("question_profile", {}) or {}).get("category")
    question_signals = data.get("question_signals") or (data.get("question_profile", {}) or {}).get("signals", {})
    side_questions = data.get("side_questions") or (data.get("question_profile", {}) or {}).get("side_questions", [])

    expected_ids = data.get("expected_metric_ids") or (
        (data.get("question_signals") or {}).get("expected_metric_ids") or []
    )

    metric_rows: List[Dict[str, str]] = []

    # Prepend question-derived signals (if present)
    if question_category:
        metric_rows.append({"Metric": "Question Category", "Value": str(question_category)})

    if isinstance(question_signals, dict):
        years = question_signals.get("years")
        regions = question_signals.get("regions")
        if years:
            metric_rows.append({"Metric": "Question Years (detected)", "Value": ", ".join(map(str, years))})
        if regions:
            metric_rows.append({"Metric": "Regions (detected)", "Value": ", ".join(map(str, regions))})

    if side_questions:
        metric_rows.append({"Metric": "Side Question(s)", "Value": "; ".join(map(str, side_questions))})

    if isinstance(metrics, dict) and metrics:
        # Canonicalize to stabilize lookups
        try:
            canon = canonicalize_metrics(metrics)
        except Exception:
            canon = metrics  # fallback (worst-case)

        # Build lookup by base canonical id (strip year suffixes etc.)
        by_base: Dict[str, List[Dict]] = {}
        if isinstance(canon, dict):
            for cid, m in canon.items():
                base = re.sub(r'_\d{4}(?:_\d{4})*$', '', str(cid))
                by_base.setdefault(base, []).append(m)

        # If template expected_ids exists, render in that order
        if expected_ids and isinstance(by_base, dict):
            for base_id in expected_ids:
                candidates = by_base.get(base_id, [])
                if candidates:
                    # deterministic pick (canonicalize_metrics already sorts deterministically)
                    m = candidates[0]
                    metric_rows.append({
                        "Metric": m.get("name", base_id),
                        "Value": _format_metric_value(m)
                    })
                else:
                    # placeholder row (avoid hard dependency on METRIC_REGISTRY)
                    display = str(base_id).replace("_", " ").title()
                    metric_rows.append({"Metric": display, "Value": "N/A"})
        else:
            # No template → show first 6 canonicalized metrics
            if isinstance(canon, dict):
                for cid, m in list(canon.items())[:6]:
                    metric_rows.append({
                        "Metric": m.get("name", cid),
                        "Value": _format_metric_value(m)
                    })

    if metric_rows:
        st.table(pd.DataFrame(metric_rows))
    else:
        st.info("No metrics available")

    st.markdown("---")

    # Key Findings
    st.subheader("🔍 Key Findings")
    findings = data.get("key_findings", [])
    for i, finding in enumerate(findings[:8], 1):
        if finding:
            st.markdown(f"**{i}.** {finding}")

    st.markdown("---")

    # Top Entities
    entities = data.get("top_entities", [])
    if entities:
        st.subheader("🏢 Top Market Players")
        entity_data = []
        for ent in entities:
            if isinstance(ent, dict):
                entity_data.append({
                    "Entity": ent.get("name", "N/A"),
                    "Share": ent.get("share", "N/A"),
                    "Growth": ent.get("growth", "N/A")
                })
        if entity_data:
            st.dataframe(pd.DataFrame(entity_data), hide_index=True, use_container_width=True)

    # Trends Forecast
    trends = data.get("trends_forecast", [])
    if trends:
        st.subheader("📈 Trends & Forecast")
        trend_data = []
        for trend in trends:
            if isinstance(trend, dict):
                trend_data.append({
                    "Trend": trend.get("trend", "N/A"),
                    "Direction": trend.get("direction", "→"),
                    "Timeline": trend.get("timeline", "N/A")
                })
        if trend_data:
            st.table(pd.DataFrame(trend_data))

    st.markdown("---")

    # Visualization
    st.subheader("📊 Data Visualization")
    viz = data.get("visualization_data")

    if viz and isinstance(viz, dict):
        labels = viz.get("chart_labels", [])
        values = viz.get("chart_values", [])
        title = viz.get("chart_title", "Trend Analysis")
        chart_type = viz.get("chart_type", "line")

        if labels and values and len(labels) == len(values):
            try:
                numeric_values = [float(v) for v in values[:10]]

                # Detect axis labels
                x_label = viz.get("x_axis_label") or detect_x_label_dynamic(labels)
                y_label = viz.get("y_axis_label") or detect_y_label_dynamic(numeric_values)

                df_viz = pd.DataFrame({"x": labels[:10], "y": numeric_values})

                if chart_type == "bar":
                    fig = px.bar(df_viz, x="x", y="y", title=title)
                else:
                    fig = px.line(df_viz, x="x", y="y", title=title, markers=True)

                fig.update_layout(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    title_font_size=16,
                    font=dict(size=12),
                    xaxis=dict(tickangle=-45) if len(labels) > 5 else {}
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.info(f"⚠️ Chart rendering failed: {e}")
        else:
            st.info("📊 Visualization data incomplete or missing")
    else:
        st.info("📊 No visualization data available")

    # Comparison Bars
    comp = data.get("comparison_bars")
    if comp and isinstance(comp, dict):
        cats = comp.get("categories", [])
        vals = comp.get("values", [])
        if cats and vals and len(cats) == len(vals):
            try:
                df_comp = pd.DataFrame({"Category": cats, "Value": vals})
                fig = px.bar(
                    df_comp, x="Category", y="Value",
                    title=comp.get("title", "Comparison"), text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    st.markdown("---")

    # Sources
    st.subheader("🔗 Sources & Reliability")
    all_sources = data.get("sources", []) or (web_context.get("sources", []) if isinstance(web_context, dict) else [])

    if not all_sources:
        st.info("No sources found")
    else:
        st.success(f"📊 Found {len(all_sources)} sources")

    cols = st.columns(2)
    for i, src in enumerate(all_sources[:10], 1):
        col = cols[(i - 1) % 2]
        short_url = src[:60] + "..." if len(src) > 60 else src
        reliability = classify_source_reliability(str(src))
        col.markdown(
            f"**{i}.** [{short_url}]({src})<br><small>{reliability}</small>",
            unsafe_allow_html=True
        )

    # Metadata
    col_fresh, col_action = st.columns(2)
    with col_fresh:
        freshness = data.get("freshness", "Current")
        st.metric("Data Freshness", freshness)

    st.markdown("---")

    # Veracity Scores
    if veracity_scores:
        st.subheader("✅ Evidence Quality Scores")
        cols = st.columns(5)
        metrics_display = [
            ("Sources", "source_quality"),
            ("Numbers", "numeric_consistency"),
            ("Citations", "citation_density"),
            ("Consensus", "source_consensus"),
            ("Overall", "overall")
        ]
        for i, (label, key) in enumerate(metrics_display):
            cols[i].metric(label, f"{veracity_scores.get(key, 0):.0f}%")

    # Web Context
    if isinstance(web_context, dict) and web_context.get("search_results"):
        with st.expander("🌐 Web Search Details"):
            for i, result in enumerate(web_context["search_results"][:5]):
                st.markdown(f"**{i+1}. {result.get('title')}**")
                st.caption(f"{result.get('source')} - {result.get('date')}")
                st.write(result.get("snippet", ""))
                st.caption(f"[{result.get('link')}]({result.get('link')})")
                st.markdown("---")
