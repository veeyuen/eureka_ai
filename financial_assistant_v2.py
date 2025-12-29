with st.spinner("Generating interpretation..."):
    try:
        # Evidence-only: only include metrics that were actually found
        safe_changes = []
        for m in results.get("metric_changes", []) or []:
            if m.get("change_type") == "not_found":
                continue
            conf = float(m.get("match_confidence") or 0)
            if conf < 70:
                continue
            safe_changes.append(m)

        if not safe_changes:
            interpretation = (
                "Evidence coverage was limited in this run (sources blocked/unusable or metrics not confidently matched). "
                "No high-confidence metric changes could be validated, so interpretation is withheld."
            )
        else:
            changes_lines = []
            for m in safe_changes[:10]:
                changes_lines.append(
                    f"- {m.get('name')}: {m.get('previous_value')} → {m.get('current_value')} "
                    f"({m.get('change_type')}, conf={m.get('match_confidence')}%)"
                )

            changes_str = "\n".join(changes_lines)
            explanation_prompt = (
                f'Use ONLY the evidence below for "{evolution_query}". Do NOT introduce any other numbers.\n\n'
                f'{changes_str}\n\n'
                'Write a 2–3 sentence interpretation focused on what changed and what remained stable.\n'
                'If evidence is limited, say so explicitly.\n'
                'Return ONLY JSON: {"interpretation": "your text"}'
            )

            headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "sonar",
                "messages": [{"role": "user", "content": explanation_prompt}]
            }
            resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=20)
            explanation_data = parse_json_response(resp.json()["choices"][0]["message"]["content"], "Explanation")
            interpretation = (explanation_data.get("interpretation") or "").strip()

            if not interpretation:
                interpretation = "Interpretation could not be generated from the available evidence."
    except Exception:
        interpretation = "Unable to generate interpretation due to an error."
