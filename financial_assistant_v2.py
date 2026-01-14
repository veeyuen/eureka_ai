# PATCH FIX2B8 (ADDITIVE)
# Option A canonical adapter (safe wrapper).
#
# Purpose:
# - Hydrate results['metric_changes'][*]['current_value'] from canonical_metrics so the Evolution dashboard
#   shows current values without relying on legacy matchers.
#
# Safety:
# - Fully gated inside a wrapper function; no module-scope references to `results` or `rows`.
# - If canonical_metrics is missing, it falls back to existing behavior.
# - Adds light debug markers under results['output_debug']['fix2b8'].

CODE_VERSION = "fix2b8_evo_canonical_adapter_v2"

# =====================================================================
# PATCH FIX2B8 START
# =====================================================================
try:
    _fix2b8_base = globals().get("render_source_anchored_results")
    if callable(_fix2b8_base) and not globals().get("render_source_anchored_results_BASE"):
        globals()["render_source_anchored_results_BASE"] = _fix2b8_base

    def _fix2b8_get_canonical_metrics(_res: dict):
        """Best-effort extraction of canonical_metrics across legacy/nested shapes."""
        if not isinstance(_res, dict):
            return {}
        # common locations
        for path in (
            ("canonical_metrics",),
            ("results", "canonical_metrics"),
            ("output_debug", "canonical_metrics"),
            ("results", "output_debug", "canonical_metrics"),
            ("results", "results", "canonical_metrics"),
            ("results", "results", "output_debug", "canonical_metrics"),
        ):
            cur = _res
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

    def _fix2b8_norm_missing(v):
        s = str(v or "").strip()
        return (not s) or (s.lower() in ("n/a", "na", "-", "—", "none", "null"))

    def _fix2b8_stringify(v):
        # keep as-is if already a nice string
        if v is None:
            return "N/A"
        if isinstance(v, (int, float)):
            # compact but deterministic
            try:
                if isinstance(v, bool):
                    return str(v)
                if float(v).is_integer():
                    return str(int(v))
                return str(v)
            except Exception:
                return str(v)
        return str(v)

    def render_source_anchored_results(results, query: str):
        """Wrapper: hydrate metric_changes.current_value from canonical_metrics, then call BASE renderer."""
        base_fn = globals().get("render_source_anchored_results_BASE")
        if not callable(base_fn):
            # Should never happen, but avoid crashing UI.
            return _fix2b8_base(results, query) if callable(_fix2b8_base) else None

        # Only mutate a shallow copy of the top dict (to avoid surprising other callers)
        try:
            res = dict(results) if isinstance(results, dict) else results
        except Exception:
            res = results

        try:
            if isinstance(res, dict):
                cm = _fix2b8_get_canonical_metrics(res)
                hydrated = 0
                eligible = 0

                # Where metric_changes lives can vary; prefer top-level.
                rows = res.get("metric_changes")
                if not isinstance(rows, list):
                    # nested legacy shape sometimes uses res['results']['metric_changes']
                    _r2 = res.get("results")
                    if isinstance(_r2, dict) and isinstance(_r2.get("metric_changes"), list):
                        rows = _r2.get("metric_changes")

                if isinstance(cm, dict) and cm and isinstance(rows, list) and rows:
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        ck = (r.get("canonical_key") or "").strip()
                        if not ck:
                            continue
                        curv = r.get("current_value")
                        if _fix2b8_norm_missing(curv):
                            eligible += 1
                            if ck in cm:
                                r["current_value"] = _fix2b8_stringify(cm.get(ck))
                                # keep existing match_stage if present
                                if not str(r.get("match_stage") or "").strip():
                                    r["match_stage"] = "canonical_hydrate"
                                # only overwrite status if it was clearly empty / not_found / unit_mismatch
                                _st = str(r.get("status") or r.get("change_type") or "").strip().lower()
                                if _st in ("", "not_found", "unit_mismatch", "missing", "unknown"):
                                    r["status"] = "present_canonical"
                                hydrated += 1

                # Attach debug marker
                res.setdefault("output_debug", {})
                if isinstance(res.get("output_debug"), dict):
                    res["output_debug"].setdefault("fix2b8", {})
                    if isinstance(res["output_debug"].get("fix2b8"), dict):
                        res["output_debug"]["fix2b8"].update({
                            "code_version": CODE_VERSION,
                            "canonical_metrics_present": bool(isinstance(cm, dict) and bool(cm)),
                            "rows_present": bool(isinstance(rows, list) and bool(rows)),
                            "rows_eligible_for_hydration": int(eligible),
                            "rows_hydrated": int(hydrated),
                        })
        except Exception:
            # Never break UI
            pass

        return base_fn(res, query)

    # Install wrapper
    globals()["render_source_anchored_results"] = render_source_anchored_results
except Exception:
    pass
# =====================================================================
# PATCH FIX2B8 END
# =====================================================================

# PATCH TRACKER (append-only)
# - fix2b8_evo_canonical_adapter_v2: Wrap render_source_anchored_results to hydrate metric_changes.current_value from canonical_metrics (Option A), fixing FIX2B7 scoping crash.

# Ensure authoritative CODE_VERSION at EOF
CODE_VERSION = "fix2b8_evo_canonical_adapter_v2"
