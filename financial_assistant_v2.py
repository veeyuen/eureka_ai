def compute_source_anchored_diff(previous_data: Dict) -> Dict:
    """
    Source-anchored evolution diff.

    Key upgrades (backward-compatible):
    - Uses baseline `evidence_records` / `metric_anchors` if present (from add_to_history enhancement).
    - Robust URL normalization to prevent MissingSchema failures on bare domains.
    - Produces stable `source_results` with extracted_numbers (or empty) for every source.
    - Always returns numeric stability_score (never None) to avoid Streamlit formatting errors.

    Expected output keys:
      status, sources_checked, sources_fetched, stability_score,
      summary, metric_changes, source_results
    """
    import math
    from datetime import datetime, timezone

    # -------------------------
    # small utilities (local)
    # -------------------------
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
        if not ts:
            return None
        try:
            ts2 = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _normalize_url(u: str) -> str:
        u = (u or "").strip()
        if not u:
            return ""
        if u.startswith("http://") or u.startswith("https://"):
            return u
        # Avoid MissingSchema: treat bare domains as https
        return "https://" + u.lstrip("/")

    def _safe_float(x) -> Optional[float]:
        try:
            if x is None or x == "":
                return None
            return float(str(x).replace(",", "").strip())
        except Exception:
            return None

    def _unit_norm(u: str) -> str:
        try:
            if "normalize_unit" in globals() and callable(globals()["normalize_unit"]):
                return normalize_unit(u)
        except Exception:
            pass
        return (u or "").strip()

    def _is_currencyish(s: str) -> bool:
        s = (s or "")
        return bool(re.search(r"(S\$|\$|USD|SGD|EUR|€|GBP|£)", s, flags=re.I))

    def _currency_token(s: str) -> str:
        t = (s or "")
        up = t.upper()
        if "S$" in up or "SGD" in up:
            return "SGD"
        if "€" in t or "EUR" in up:
            return "EUR"
        if "£" in t or "GBP" in up:
            return "GBP"
        if "$" in t or "USD" in up:
            return "USD"
        return ""

    def _compatible_currency(prev_raw: str, cand_raw: str, cand_ctx: str) -> bool:
        if not _is_currencyish(prev_raw):
            return True
        pc = _currency_token(prev_raw)
        cc = _currency_token(cand_raw) or _currency_token(cand_ctx)
        if pc and not cc:
            return False
        if pc and cc and pc != cc:
            return False
        return True

    def _unit_family(u: str) -> str:
        u = _unit_norm(u)
        if u in ("T", "B", "M", "K"):
            return "SCALE"
        if u == "%":
            return "PCT"
        return "OTHER"

    def _compatible_units(prev_unit: str, cand_unit: str, prev_raw: str, cand_raw: str, cand_ctx: str) -> bool:
        pu = _unit_norm(prev_unit)
        cu = _unit_norm(cand_unit)

        pf = _unit_family(pu)
        cf = _unit_family(cu)

        if pf == "PCT":
            return (cu == "%") or ("%" in (cand_raw or "")) or ("%" in (cand_ctx or ""))

        if pf == "SCALE":
            return cf == "SCALE"

        return True

    def _tokens(name: str) -> List[str]:
        n = (name or "").lower()
        toks = re.findall(r"[a-z0-9]+", n)
        stop = {"the", "and", "or", "of", "in", "to", "for", "by", "from", "with", "on", "at", "as"}
        out = [t for t in toks if len(t) > 3 and t not in stop]
        if "gdp" in toks:
            out += ["gross", "domestic", "product"]
        return list(dict.fromkeys(out))[:30]

    def _ctx_score(tokens: List[str], ctx: str) -> float:
        if not tokens:
            return 0.0
        c = (ctx or "").lower()
        hit = sum(1 for t in tokens if t in c)
        return hit / max(1, len(tokens))

    def _normalize_to_base(value: float, unit: str) -> float:
        """
        Normalize to a comparable base:
        - For SCALE: normalize into 'millions' equivalent (M=1, B=1000, T=1_000_000, K=0.001)
        - For %: unchanged
        - Otherwise: unchanged
        """
        u = (unit or "").strip().upper()
        if u == "T":
            return value * 1_000_000
        if u == "B":
            return value * 1_000
        if u == "M":
            return value * 1
        if u == "K":
            return value * 0.001
        return value

    def _pct_change(old: Optional[float], new: Optional[float]) -> Optional[float]:
        if old is None or new is None:
            return None
        if old == 0:
            return None
        return ((new - old) / abs(old)) * 100.0

    def _format_prev_raw(v, u) -> str:
        if v is None or v == "":
            return "N/A"
        u = (u or "").strip()
        if u == "%":
            return f"{v} %"
        return f"{v} {u}".strip() if u else str(v)

    # -------------------------
    # read baseline (previous)
    # -------------------------
    prev_primary = (previous_data or {}).get("primary_response", {}) or {}
    prev_metrics = prev_primary.get("primary_metrics", {}) or {}
    prev_sources = prev_primary.get("sources", []) or (previous_data or {}).get("web_sources", []) or []

    # Evidence cache from baseline (if add_to_history enhanced it)
    baseline_evidence_records = (previous_data or {}).get("evidence_records") or []
    baseline_metric_anchors = (previous_data or {}).get("metric_anchors") or []

    # If sources were stored as bare domains, normalize them (prevents MissingSchema)
    norm_sources = []
    for s in (prev_sources or []):
        if not s:
            continue
        ns = _normalize_url(str(s))
        if ns not in norm_sources:
            norm_sources.append(ns)

    # -------------------------
    # fetch + extract (or reuse baseline evidence)
    # -------------------------
    source_results: List[Dict] = []
    all_current_numbers: List[Dict] = []

    # helper: reuse baseline evidence if very recent (prevents “worked before, fails now” volatility)
    def _maybe_reuse_baseline(url: str) -> Optional[Dict]:
        # Try exact match first, then match by domain-ish substring
        if not baseline_evidence_records:
            return None

        candidates = []
        for rec in baseline_evidence_records:
            if not isinstance(rec, dict):
                continue
            rurl = _normalize_url(str(rec.get("url") or ""))
            if not rurl:
                continue
            if rurl == url:
                candidates.append(rec)
            else:
                # fallback: domain match
                try:
                    if re.sub(r"^https?://", "", rurl).split("/")[0] == re.sub(r"^https?://", "", url).split("/")[0]:
                        candidates.append(rec)
                except Exception:
                    pass

        if not candidates:
            return None

        # Choose the newest fetched_at
        newest = None
        newest_dt = None
        for rec in candidates:
            dt = _parse_iso(rec.get("fetched_at"))
            if dt and (newest_dt is None or dt > newest_dt):
                newest_dt = dt
                newest = rec

        # Reuse only if within 24 hours (tuneable)
        if newest and newest_dt:
            age_hours = (_now_utc() - newest_dt).total_seconds() / 3600.0
            if age_hours <= 24.0:
                return newest
        return None

    for url in norm_sources:
        status = "failed"
        detail = "unknown"
        fetched_at = _now_utc().isoformat()
        fingerprint = None
        extracted_numbers: List[Dict] = []

        # Prefer baseline reuse if recent
        reused = _maybe_reuse_baseline(url)
        if reused:
            status = "reused"
            detail = "baseline evidence_records (<=24h)"
            fetched_at = reused.get("fetched_at") or fetched_at
            fingerprint = reused.get("fingerprint")
            nums = reused.get("numbers") or []
            # normalize to the current expected shape: extracted_numbers
            for n in nums:
                if not isinstance(n, dict):
                    continue
                extracted_numbers.append({
                    "value": n.get("value"),
                    "unit": n.get("unit"),
                    "raw": (n.get("raw") or "").strip(),
                    "context_snippet": (n.get("context_snippet") or "").strip(),
                    "anchor_hash": n.get("anchor_hash"),
                })

        else:
            # Fetch fresh
            try:
                if "fetch_url_content_with_status" in globals() and callable(globals()["fetch_url_content_with_status"]):
                    fetched = fetch_url_content_with_status(url)
                else:
                    fetched = None

                if not fetched or not isinstance(fetched, dict):
                    status = "failed"
                    detail = "fetch_url_content_with_status unavailable"
                else:
                    status = fetched.get("status", "failed")
                    detail = fetched.get("status_detail", "")
                    fetched_at = fetched.get("fetched_at") or fetched_at
                    fingerprint = fetched.get("fingerprint")
                    text = fetched.get("text") or ""

                    # Extract numbers
                    if text and ("extract_numeric_candidates" in globals()) and callable(globals()["extract_numeric_candidates"]):
                        extracted_numbers = extract_numeric_candidates(text, source_url=url) or []
                    elif text and ("extract_numbers_from_text" in globals()) and callable(globals()["extract_numbers_from_text"]):
                        extracted_numbers = extract_numbers_from_text(text) or []
                    else:
                        extracted_numbers = []

                    # Ensure shape contains context_snippet + anchor_hash (optional)
                    clean = []
                    for n in (extracted_numbers or []):
                        if not isinstance(n, dict):
                            continue
                        raw = (n.get("raw") or "").strip()
                        ctx = (n.get("context_snippet") or n.get("context") or "").strip()
                        clean.append({
                            "value": n.get("value"),
                            "unit": n.get("unit"),
                            "raw": raw,
                            "context_snippet": ctx[:220],
                            "anchor_hash": n.get("anchor_hash"),
                        })
                    extracted_numbers = clean

            except Exception as e:
                status = "failed"
                detail = f"exception:{type(e).__name__}"
                extracted_numbers = []

        source_results.append({
            "url": url,
            "status": status,
            "status_detail": detail,
            "numbers_found": len(extracted_numbers or []),
            "fingerprint": fingerprint,
            "fetched_at": fetched_at,
            "extracted_numbers": extracted_numbers or [],
        })

        # Collect for matching
        for n in (extracted_numbers or []):
            if not isinstance(n, dict):
                continue
            all_current_numbers.append({
                "source_url": url,
                "fingerprint": fingerprint,
                "value": n.get("value"),
                "unit": n.get("unit"),
                "raw": n.get("raw"),
                "context_snippet": n.get("context_snippet") or "",
                "anchor_hash": n.get("anchor_hash"),
            })

    sources_checked = len(norm_sources)
    sources_fetched = sum(1 for r in source_results if r.get("status") in ("success", "reused"))

    # -------------------------
    # Match metrics: anchor-first, then fallback scorer
    # -------------------------
    # Build anchor lookup: metric_id -> anchor dict
    anchor_by_metric_id = {}
    for a in (baseline_metric_anchors or []):
        if isinstance(a, dict) and a.get("metric_id"):
            anchor_by_metric_id[str(a.get("metric_id"))] = a

    metric_changes: List[Dict] = []
    inc = dec = same = 0

    # Fallback: if primary_metrics is dict of dicts
    for mid, m in (prev_metrics or {}).items():
        if not isinstance(m, dict):
            continue

        metric_id = str(mid)
        metric_name = m.get("name") or metric_id
        prev_val = m.get("value")
        prev_unit = (m.get("unit") or "").strip()
        prev_raw = m.get("raw") or _format_prev_raw(prev_val, prev_unit)

        prev_num = _safe_float(prev_val)
        prev_num_norm = _normalize_to_base(prev_num, _unit_norm(prev_unit)) if prev_num is not None else None

        # 1) Anchor-first matching (best)
        chosen = None
        a = anchor_by_metric_id.get(metric_id)
        if a and isinstance(a, dict):
            a_hash = a.get("anchor_hash")
            a_url = _normalize_url(a.get("source_url") or "") if a.get("source_url") else ""
            if a_hash:
                for cand in all_current_numbers:
                    if cand.get("anchor_hash") == a_hash:
                        # If baseline anchor recorded a URL, prefer matches from same source_url
                        if a_url and _normalize_url(cand.get("source_url") or "") != a_url:
                            continue
                        chosen = cand
                        break

        # 2) Fallback matching (token + unit/currency guards)
        if chosen is None:
            toks = _tokens(metric_name)
            best = None
            best_score = -1.0

            for cand in all_current_numbers:
                cctx = cand.get("context_snippet") or ""
                craw = cand.get("raw") or ""
                cunit = (cand.get("unit") or "").strip()

                if "is_likely_junk_context" in globals() and callable(globals()["is_likely_junk_context"]):
                    try:
                        if is_likely_junk_context(cctx):
                            continue
                    except Exception:
                        pass

                if not _compatible_units(prev_unit, cunit, prev_raw, craw, cctx):
                    continue
                if not _compatible_currency(prev_raw, craw, cctx):
                    continue

                s_ctx = _ctx_score(toks, cctx)

                # numeric proximity bonus if both are numeric
                cval = _safe_float(cand.get("value"))
                cval_norm = _normalize_to_base(cval, _unit_norm(cunit)) if cval is not None else None
                bonus = 0.0
                if prev_num_norm is not None and cval_norm is not None:
                    denom = max(1.0, abs(prev_num_norm))
                    rel = abs(cval_norm - prev_num_norm) / denom
                    bonus = max(0.0, 0.35 - min(0.35, rel))  # up to +0.35

                score = s_ctx + bonus
                if score > best_score:
                    best_score = score
                    best = cand

            if best and best_score >= 0.20:
                chosen = best

        # Determine new values
        if chosen:
            curr_val = chosen.get("value")
            curr_unit = (chosen.get("unit") or "").strip()
            curr_raw = (chosen.get("raw") or "").strip() or _format_prev_raw(curr_val, curr_unit)

            curr_num = _safe_float(curr_val)
            curr_num_norm = _normalize_to_base(curr_num, _unit_norm(curr_unit)) if curr_num is not None else None

            pct = _pct_change(prev_num_norm, curr_num_norm)
            if pct is None:
                pct_str = "N/A"
            else:
                pct_str = f"{pct:+.1f}%"

            # classify
            if pct is None:
                direction = "Unknown"
                status_label = "Partial match"
            else:
                if abs(pct) < 0.05:
                    direction = "No change"
                    status_label = "Matched"
                    same += 1
                elif pct > 0:
                    direction = "Increase"
                    status_label = "Matched"
                    inc += 1
                else:
                    direction = "Decrease"
                    status_label = "Matched"
                    dec += 1

            metric_changes.append({
                "metric_id": metric_id,
                "metric_name": metric_name,
                "previous_value": prev_raw,
                "current_value": curr_raw,
                "pct_change": pct_str,
                "direction": direction,
                "match_status": status_label,
                "matched_source": chosen.get("source_url"),
                "matched_context": (chosen.get("context_snippet") or "")[:220],
                "matched_anchor_hash": chosen.get("anchor_hash"),
            })

        else:
            metric_changes.append({
                "metric_id": metric_id,
                "metric_name": metric_name,
                "previous_value": prev_raw,
                "current_value": "Not found",
                "pct_change": "N/A",
                "direction": "Unknown",
                "match_status": "Not found",
                "matched_source": None,
                "matched_context": None,
                "matched_anchor_hash": None,
            })

    total = max(1, len(metric_changes))
    # Stability: fraction that are "No change" OR at least "Matched"
    matched = sum(1 for r in metric_changes if r.get("match_status") in ("Matched",))
    unchanged = sum(1 for r in metric_changes if r.get("direction") == "No change")
    stability_score = float(round(((unchanged * 1.0) + (matched * 0.25)) / total * 100.0, 1))

    return {
        "status": "success",
        "sources_checked": sources_checked,
        "sources_fetched": sources_fetched,
        "stability_score": stability_score,
        "summary": {
            "metrics_increased": inc,
            "metrics_decreased": dec,
            "metrics_unchanged": same,
        },
        "metric_changes": metric_changes,
        "source_results": source_results,
    }
