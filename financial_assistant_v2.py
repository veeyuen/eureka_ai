# PATCH FIX2B10 (ADDITIVE): Inline canonical hydration for Evolution renderer
# Place this block INSIDE render_source_anchored_results(results, query)
# Immediately AFTER:
#     rows = results.get("metric_changes") or []

try:
    def _fix2b10_get_cm(res):
        if not isinstance(res, dict):
            return {}
        for path in (
            ("canonical_metrics",),
            ("results", "canonical_metrics"),
            ("output_debug", "canonical_metrics"),
            ("results", "output_debug", "canonical_metrics"),
        ):
            cur = res
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur.get(k)
                else:
                    ok = False
                    break
            if ok and isinstance(cur, dict) and cur:
                return cur
        return {}

    def _fix2b10_is_missing(v):
        s = str(v or "").strip().lower()
        return (not s) or (s in ("n/a", "na", "-", "—", "none", "null"))

    cm = _fix2b10_get_cm(results)
    hydrated = 0
    eligible = 0

    if isinstance(cm, dict) and cm and isinstance(rows, list) and rows:
        for r in rows:
            if not isinstance(r, dict):
                continue
            ck = str(r.get("canonical_key") or "").strip()
            if not ck:
                continue
            if _fix2b10_is_missing(r.get("current_value")):
                eligible += 1
                if ck in cm:
                    r["current_value"] = str(cm.get(ck))
                    if not str(r.get("match_stage") or "").strip():
                        r["match_stage"] = "canonical_hydrate"
                    st = str(r.get("status") or r.get("change_type") or "").strip().lower()
                    if st in ("", "not_found", "unit_mismatch", "missing", "unknown"):
                        r["status"] = "present_canonical"
                    hydrated += 1

    results.setdefault("output_debug", {})
    if isinstance(results.get("output_debug"), dict):
        results["output_debug"].setdefault("fix2b10", {})
        if isinstance(results["output_debug"].get("fix2b10"), dict):
            results["output_debug"]["fix2b10"].update({
                "code_version": CODE_VERSION if "CODE_VERSION" in globals() else "",
                "canonical_metrics_present": bool(isinstance(cm, dict) and bool(cm)),
                "rows_eligible_for_hydration": int(eligible),
                "rows_hydrated": int(hydrated),
            })
except Exception:
    pass
# END PATCH FIX2B10
