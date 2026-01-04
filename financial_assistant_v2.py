# ===============================================================================
# YUREEKA AI RESEARCH ASSISTANT v7.41
# With Web Search, Evidence-Based Verification, Confidence Scoring
# SerpAPI Output with Evolution Layer Version
# Updated SerpAPI parameters for stable output
# Deterministic Output From LLM
# Deterministic Evolution Core Using Python Diff Engine
# Anchored Evolution Analysis Using JSON As Input Into Model
# Implementation of Source-Based Evolution
# Saving of JSON output Files into Google Sheets
# Canonical Metric Registry + Semantic Hashing of Findings
# Removal of Evolution Decisions from LLM
# Further Enhancements to Minimize Evolution Drift (Metric)
# Saving of Extraction Cache in JSON
# Prioritize High Quality Sources With Source Freshness Tracking
# Timestamps = Timezone Naive
# Improved Stability of Handling of Duplicate Canonicalized IDs
# Deterministic Main and Side Topic Extractor
# Range Aware Canonical Metrics
# Range + Source Attribution
# Proxy Labeler + Geo Tagging
# Improved Main Topic + Side Topic Extractor Using Deterministic-->NLP-->LLM layer
# Guardrails For Main + Side Topic Handling
# Numeric Consistency Scores
# Multi-Side Enumerations
# Dashboard Unit Presentation Fixes (Main + Evolution)
# Domain-Agnostic Question Profiling
# Baseline Caching Contains HTTP Validators + Numeric Data
# URL canonicalization
# Evolution Layer Leverage On New Analysis Pipeline to Minimise Volatility
# Canonicalization of Evolution Layer Metrics To Match Analysis Layer
# Fix URL/path Collapese Issue Causing + Tighten Evolution Extraction (Topic Gating)
# canonical-key-first matching
# Evolution Pipeline to Consume analysis upstream artifacts
# safety-net hard gates (minimal) before matching
# Tighten canonical identity + unit-family constraints
# Fingerprint freshness gating to evolution
# Fix SerpAPI access and fetching
# Keeps your snapshot-friendly scraped_meta (with extracted numbers + fingerprint fields)
# Safe fallback scraper when ScrapingDog is unavailable
# Prevent caching “empty results” from SerpAPI (no poisoned cache)
# Restoration of Range Estimates For Metrics
# Improved Junk Tagging and Rejection
# One Canononical Operator for Analysis + Evolution Layers
# Metric Aware Range Construction Everywhere
# Anchor Matching Correctness
# Unit Measure + Attribute Association e.g. M + units (sold)
# Enriched metric_schema_frozen (analysis side)
# ================================================================================

import io
import os
import re
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import base64
import hashlib
import numpy as np
import difflib
import gspread
import google.generativeai as genai
from pypdf import PdfReader
from pathlib import Path
from google.oauth2.service_account import Credentials
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from collections import Counter
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# =========================
# VERSION STAMP (ADDITIVE)
# =========================
CODE_VERSION = "financial_assistant_v7_41_FIX5_ANCHOR_DISPATCH_RETRY_DEBUG"
# =====================================================================
# PATCH FINAL (ADDITIVE): end-state single bump label (non-breaking)
# NOTE: We do not overwrite CODE_VERSION to avoid any legacy coupling.
# Consumers can prefer ENDSTATE_FINAL_VERSION when present.
# =====================================================================
ENDSTATE_FINAL_VERSION = "v7_41_endstate_final_1"
# =====================================================================

# =====================================================================
# PATCH ES2/ES8/ES9 (ADDITIVE): shared determinism helpers for drift=0
# - Deterministic sorting / tie-breaking helpers
# - Deterministic candidate index builder (anchor_hash -> best candidate)
# - Lightweight schema + universe hashing for convergence checks
# - One-button end-state validation harness (callable)
# NOTE: Additive only; existing logic remains intact.
# =====================================================================
import hashlib as _es_hashlib

def _es_hash_text(s: str) -> str:
    try:
        return _es_hashlib.sha256((s or "").encode("utf-8")).hexdigest()
    except Exception:
        return ""

def _es_stable_sort_key(v):
    """
    Deterministic sort key that never relies on Python's randomized hash().
    Keeps ordering stable across runs for mixed types.
    """
    try:
        if v is None:
            return (0, "")
        if isinstance(v, (int, float)):
            return (1, f"{v:.17g}")
        if isinstance(v, str):
            return (2, v)
        if isinstance(v, bytes):
            return (3, v.decode("utf-8", "ignore"))
        if isinstance(v, dict):
            items = sorted(((str(k), _es_stable_sort_key(vv)) for k, vv in v.items()), key=lambda x: x[0])
            return (4, str(items))
        if isinstance(v, (list, tuple, set)):
            lst = list(v)
            try:
                lst.sort(key=_es_stable_sort_key)
            except Exception:
                lst = sorted(lst, key=lambda x: str(x))
            return (5, str([_es_stable_sort_key(x) for x in lst]))
        return (9, str(v))
    except Exception:
        return (9, str(v))

def _es_sorted_pairs_from_sources_cache(baseline_sources_cache):
    pairs = []
    for sr in (baseline_sources_cache or []):
        if not isinstance(sr, dict):
            continue
        u = (sr.get("source_url") or sr.get("url") or "").strip()
        fp = (sr.get("source_fingerprint") or sr.get("fingerprint") or sr.get("content_fingerprint") or "").strip()
        if u and fp:
            pairs.append((u, fp))
    pairs.sort(key=lambda t: (t[0], t[1]))
    return pairs

def _es_compute_canonical_universe_hash(primary_metrics_canonical: dict, metric_schema_frozen: dict) -> str:
    try:
        keys = set()
        if isinstance(primary_metrics_canonical, dict):
            keys.update([str(k) for k in primary_metrics_canonical.keys()])
        if isinstance(metric_schema_frozen, dict):
            keys.update([str(k) for k in metric_schema_frozen.keys()])
        return _es_hash_text("|".join(sorted(keys)))
    except Exception:
        return ""

def _es_compute_schema_hash(metric_schema_frozen: dict) -> str:
    """
    Deterministic hash of schema fields that affect numeric comparisons.
    Keeps it lightweight: tolerances + units + scale hints only.
    """
    try:
        if not isinstance(metric_schema_frozen, dict):
            return ""
        rows = []
        for k in sorted(metric_schema_frozen.keys()):
            s = metric_schema_frozen.get(k) or {}
            if not isinstance(s, dict):
                continue
            abs_eps = s.get("abs_eps", s.get("ABS_EPS"))
            rel_eps = s.get("rel_eps", s.get("REL_EPS"))
            unit = s.get("unit") or s.get("units") or ""
            scale = s.get("scale") or s.get("magnitude") or ""
            rows.append(f"{k}::abs={abs_eps}::rel={rel_eps}::unit={unit}::scale={scale}")
        return _es_hash_text("|".join(rows))
    except Exception:
        return ""

def _es_build_candidate_index_deterministic(baseline_sources_cache):
    """
    Deterministically build anchor_hash -> candidate map.
    If multiple candidates share the same anchor_hash, choose the best by a stable
    tie-breaker that prefers:
      - higher anchor_confidence
      - longer context_snippet (more evidence)
      - stable context_hash / numeric value / unit
      - stable source_url
    """
    try:
        buckets = {}
        for sr in (baseline_sources_cache or []):
            if not isinstance(sr, dict):
                continue
            su = sr.get("source_url") or sr.get("url") or ""
            for cand in (sr.get("extracted_numbers") or []):
                if not isinstance(cand, dict):
                    continue
                ah = cand.get("anchor_hash") or cand.get("anchor") or ""
                if not ah:
                    continue
                c2 = dict(cand)
                if "source_url" not in c2:
                    c2["source_url"] = su
                buckets.setdefault(ah, []).append(c2)

        out = {}
        for ah in sorted(buckets.keys()):
            cands = buckets.get(ah) or []
            def _cand_key(c):
                try:
                    conf = c.get("anchor_confidence")
                    conf_key = -(float(conf) if conf is not None else 0.0)
                except Exception:
                    conf_key = 0.0
                ctx = (c.get("context_snippet") or c.get("context") or "")
                ctx_len = -len(str(ctx))
                ctx_hash = c.get("context_hash") or ""
                val = c.get("value")
                unit = c.get("unit") or ""
                su = c.get("source_url") or ""
                return (conf_key, ctx_len, str(ctx_hash), _es_stable_sort_key(val), str(unit), str(su))
            cands_sorted = sorted(cands, key=_cand_key)
            out[ah] = cands_sorted[0] if cands_sorted else None
        return out
    except Exception:
        return {}

def end_state_validation_harness(baseline_analysis: dict, evolution_output: dict, min_stability: float = 99.9) -> dict:
    """
    PATCH ES9 (ADDITIVE): one-button end-state validation (warn-only helper)
    Use this to assert drift=0 on identical inputs.

    Returns a dict with pass/fail booleans and diagnostic fields.
    This does NOT mutate inputs.
    """
    report = {
        "passed": False,
        "checks": {},
        "notes": [],
    }
    try:
        base_prev = baseline_analysis or {}
        evo = evolution_output or {}

        # Snapshot hash
        base_snap = base_prev.get("source_snapshot_hash") or base_prev.get("results", {}).get("source_snapshot_hash")
        evo_snap = evo.get("source_snapshot_hash")

        # Universe + schema hashes
        base_uni = base_prev.get("canonical_universe_hash") or base_prev.get("results", {}).get("canonical_universe_hash")
        base_sch = base_prev.get("schema_hash") or base_prev.get("results", {}).get("schema_hash")
        evo_uni = evo.get("canonical_universe_hash")
        evo_sch = evo.get("schema_hash")

        report["checks"]["snapshot_hash_match"] = bool(base_snap and evo_snap and base_snap == evo_snap)
        report["checks"]["canonical_universe_hash_match"] = bool(base_uni and evo_uni and base_uni == evo_uni)
        report["checks"]["schema_hash_match"] = bool(base_sch and evo_sch and base_sch == evo_sch)

        # Stability threshold (warn-only semantics: "passed" includes match + stability)
        try:
            st = float(evo.get("stability_score") or 0.0)
        except Exception:
            st = 0.0
        report["checks"]["stability_meets_threshold"] = bool(st + 1e-9 >= float(min_stability))

        # Drift suspicion flag (if your pipeline sets it)
        report["checks"]["drift_suspected_flag_false"] = (evo.get("drift_suspected") is False)

        # Final pass condition
        report["passed"] = (
            report["checks"]["snapshot_hash_match"]
            and report["checks"]["canonical_universe_hash_match"]
            and report["checks"]["schema_hash_match"]
            and report["checks"]["stability_meets_threshold"]
        )

        if not report["passed"]:
            report["notes"].append("If hashes match but stability is low, inspect candidate tie-breaks and ordering.")
    except Exception:
        report["notes"].append("Validation harness encountered an exception (non-fatal).")
    return report
# =====================================================================

            # =========================


# =========================================================
# GOOGLE SHEETS HISTORY STORAGE
# =========================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
MAX_HISTORY_ITEMS = 50

@st.cache_resource
def get_google_sheet():
    """Connect to Google Sheet (cached connection)"""
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=SCOPES
        )
        client = gspread.authorize(creds)

        # ===================== PATCH GS1 (ADDITIVE): prefer explicit History worksheet =====================
        # Why:
        # - Your spreadsheet contains multiple tabs (e.g., "New Analysis", "History", "HistoryFull", "Snapshots")
        # - sheet1 is often NOT "History", so get_history() reads the wrong tab and sees "no analyses"
        # Behavior:
        # - Default worksheet_title = "History" (override via secrets: google_sheets.history_worksheet)
        # - Fallback to sheet1 only if the worksheet doesn't exist
        spreadsheet_name = (
            st.secrets.get("google_sheets", {}).get("spreadsheet_name", "Yureeka_JSON")
        )
        ss = client.open(spreadsheet_name)

        worksheet_title = st.secrets.get("google_sheets", {}).get("history_worksheet", "History")
        try:
            sheet = ss.worksheet(worksheet_title)
        except Exception:
            sheet = ss.sheet1
        # =================== END PATCH GS1 (ADDITIVE) ===================

        # Ensure headers exist - handle response object
        try:
            headers = sheet.row_values(1)
            if not headers or len(headers) == 0 or headers[0] != "id":
                # update() returns a response object in newer gspread - ignore it
                _ = sheet.update('A1:E1', [["id", "timestamp", "question", "confidence", "data"]])
        except gspread.exceptions.APIError:
            _ = sheet.update('A1:E1', [["id", "timestamp", "question", "confidence", "data"]])
        except Exception:
            pass  # Headers probably already exist

        return sheet

    except gspread.exceptions.SpreadsheetNotFound:
        st.error("❌ Spreadsheet not found. Create 'Yureeka_JSON' (or your configured name) and share with service account.")
        return None
    except Exception as e:
        error_str = str(e)
        # Ignore Response [200] - it's actually success
        if "Response [200]" in error_str:
            # This means the connection worked, try to return the sheet anyway
            try:
                creds = Credentials.from_service_account_info(
                    dict(st.secrets["gcp_service_account"]),
                    scopes=SCOPES
                )
                client = gspread.authorize(creds)

                # ===================== PATCH GS1b (ADDITIVE): same worksheet selection in fallback =====================
                spreadsheet_name = st.secrets.get("google_sheets", {}).get("spreadsheet_name", "Yureeka_JSON")
                ss = client.open(spreadsheet_name)
                worksheet_title = st.secrets.get("google_sheets", {}).get("history_worksheet", "History")
                try:
                    return ss.worksheet(worksheet_title)
                except Exception:
                    return ss.sheet1
                # =================== END PATCH GS1b (ADDITIVE) ===================
            except:
                pass
        st.error(f"❌ Failed to connect to Google Sheets: {e}")
        return None

def generate_analysis_id() -> str:
    """Generate unique ID for analysis"""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]}"

def add_to_history(analysis: dict) -> bool:
    """
    Save analysis to Google Sheet (or session fallback).

    ADDITIVE end-state wiring:
      - If a baseline source cache exists, build & store:
          * evidence_records (structured, cached)
          * metric_anchors (baseline metrics anchored to evidence)
      - Prevent Google Sheets 50,000-char single-cell limit errors by shrinking only
        the JSON payload written into the single "analysis json" cell when necessary.

    Backward compatible:
      - Only adds keys; does not remove existing fields.
      - Never blocks saving if enrichment fails.
      - If Sheets unavailable, falls back to session_state.
    """
    import json
    import re
    import streamlit as st
    from datetime import datetime

    SHEETS_CELL_LIMIT = 50000

    # -----------------------
    # PATCH A1 (ADDITIVE): robustly locate baseline_sources_cache
    # - Added primary_response.baseline_sources_cache as extra fallback
    # -----------------------
    baseline_cache = (
        analysis.get("baseline_sources_cache")
        or (analysis.get("primary_response", {}) or {}).get("baseline_sources_cache")
        or (analysis.get("results", {}) or {}).get("baseline_sources_cache")
        or (analysis.get("results", {}) or {}).get("source_results")
    )

    # -----------------------
    # PATCH A2 (ADDITIVE): build evidence_records deterministically
    # -----------------------
    def _build_evidence_records_from_baseline_cache(baseline_cache_obj):
        records = []
        if not isinstance(baseline_cache_obj, list):
            return records

        # helper: safe sha1 fallback if needed
        def _sha1(s: str) -> str:
            try:
                import hashlib
                return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
            except Exception:
                return ""

        for sr in baseline_cache_obj:
            if not isinstance(sr, dict):
                continue
            url = sr.get("url") or ""
            fp = sr.get("fingerprint")
            fetched_at = sr.get("fetched_at")

            nums = sr.get("extracted_numbers") or []
            clean_nums = []

            if isinstance(nums, list):
                for n in nums:
                    if not isinstance(n, dict):
                        continue

                    # optional canonicalization hook
                    try:
                        fn = globals().get("canonicalize_numeric_candidate")
                        if callable(fn):
                            n = fn(dict(n))
                    except Exception:
                        n = dict(n)

                    raw = (n.get("raw") or "").strip()
                    ctx = (n.get("context_snippet") or n.get("context") or "").strip()
                    anchor_hash = n.get("anchor_hash") or _sha1(f"{url}|{raw}|{ctx[:240]}")

                    clean_nums.append({
                        "value": n.get("value"),
                        "unit": n.get("unit"),
                        "unit_tag": n.get("unit_tag"),
                        "unit_family": n.get("unit_family"),
                        "base_unit": n.get("base_unit"),
                        "multiplier_to_base": n.get("multiplier_to_base"),
                        "value_norm": n.get("value_norm"),

                        "raw": raw,
                        "context_snippet": ctx[:240],
                        "anchor_hash": anchor_hash,
                        "source_url": n.get("source_url") or url,

                        "start_idx": n.get("start_idx"),
                        "end_idx": n.get("end_idx"),

                        "is_junk": bool(n.get("is_junk")) if isinstance(n.get("is_junk"), bool) else False,
                        "junk_reason": n.get("junk_reason") or "",

                        "measure_kind": n.get("measure_kind"),
                        "measure_assoc": n.get("measure_assoc"),
                    })

            # stable ordering (prefer your helper if present)
            try:
                if "sort_snapshot_numbers" in globals() and callable(globals()["sort_snapshot_numbers"]):
                    clean_nums = sort_snapshot_numbers(clean_nums)
                else:
                    clean_nums = sorted(
                        clean_nums,
                        key=lambda x: (str(x.get("anchor_hash") or ""), str(x.get("raw") or ""))
                    )
            except Exception:
                pass

            records.append({
                "url": url,
                "fetched_at": fetched_at,
                "fingerprint": fp,
                "numbers": clean_nums,
            })

        # stable ordering (prefer helper if present)
        try:
            if "sort_evidence_records" in globals() and callable(globals()["sort_evidence_records"]):
                records = sort_evidence_records(records)
            else:
                records = sorted(records, key=lambda r: str(r.get("url") or ""))
        except Exception:
            pass

        return records

    # -----------------------
    # PATCH A3 (ADDITIVE): build metric_anchors deterministically (schema-first if present)
    # -----------------------
    def _build_metric_anchors(primary_metrics_canonical, metric_schema_frozen, evidence_records):
        """
        Deterministically anchor each canonical metric to the best matching extracted number
        from evidence_records.

        Returns:
        anchors: Dict[canonical_key -> anchor_dict]

        Notes:
        - Schema-first gating (unit_family/dimension) for drift stability.
        - Backward compatible: emits legacy fields metric_id/metric_name in each anchor.
        """
        import re

        anchors = {}
        if not isinstance(primary_metrics_canonical, dict) or not isinstance(evidence_records, list):
            return anchors

        def _tokenize(s: str):
            return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2]

        # PATCH A3.1 (ADDITIVE): tiny float helper for deterministic closeness scoring
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        # PATCH A3.9 (ADDITIVE): currency evidence helper
        # - Needed because many currency metrics appear as magnitude-tagged numbers (e.g., "40.7M")
        #   with currency implied in nearby context ("USD", "revenue", "$", etc.)
        def _has_currency_evidence(raw: str, ctx: str) -> bool:
            r = (raw or "")
            c = (ctx or "").lower()
            if any(s in r for s in ["$", "S$", "€", "£"]):
                return True
            if any(code in c for code in [" usd", "sgd", " eur", " gbp", " aud", " cad", " jpy", " cny", " rmb"]):
                return True
            strong_kw = [
                "revenue", "turnover", "valuation", "valued at", "market value", "market size",
                "sales value", "net profit", "operating profit", "gross profit",
                "ebitda", "earnings", "income", "capex", "opex"
            ]
            if any(k in c for k in strong_kw):
                return True
            return False
        # =========================

        # flatten candidates
        all_nums = []
        for rec in evidence_records:
            if not isinstance(rec, dict):
                continue
            for n in (rec.get("numbers") or []):
                if isinstance(n, dict):
                    all_nums.append(n)

        # PATCH A3.2 (ADDITIVE): normalize_unit_tag + unit_family hooks (if present)
        _norm_tag_fn = globals().get("normalize_unit_tag")
        _unit_family_fn = globals().get("unit_family")

        for ckey, m in primary_metrics_canonical.items():
            if not isinstance(m, dict):
                continue

            schema = (metric_schema_frozen or {}).get(ckey) if isinstance(metric_schema_frozen, dict) else None
            expected_family = (schema.get("unit_family") or "").lower().strip() if isinstance(schema, dict) else ""
            expected_unit = (schema.get("unit") or "").strip() if isinstance(schema, dict) else ""
            expected_dim = (schema.get("dimension") or "").lower().strip() if isinstance(schema, dict) else ""

            # tokens: schema keywords + metric name tokens
            toks = []
            if isinstance(schema, dict):
                for k in (schema.get("keywords") or []):
                    toks.extend(_tokenize(str(k)))
            toks.extend(_tokenize(m.get("name") or m.get("original_name") or ""))
            toks = list(dict.fromkeys(toks))[:40]

            best = None
            best_key = None

            # PATCH A3.3 (ADDITIVE): metric value reference for closeness bonus
            m_val = _to_float(m.get("value_norm") if m.get("value_norm") is not None else m.get("value"))

            # PATCH A3.4 (ADDITIVE): normalized expected tag (schema unit may be "M", "%", etc.)
            exp_tag = expected_unit
            try:
                if callable(_norm_tag_fn):
                    exp_tag = _norm_tag_fn(expected_unit)
            except Exception:
                pass

            # PATCH A3.10 (ADDITIVE): metric unit_tag (if available) to gate closeness bonus
            m_tag = (m.get("unit_tag") or "").strip()

            for cand in all_nums:
                if cand.get("is_junk") is True:
                    continue

                ctx = cand.get("context_snippet") or ""
                c_ut = (cand.get("unit_tag") or "").strip()
                c_fam = (cand.get("unit_family") or "").lower().strip()

                # =========================
                # PATCH A3.5 (ADDITIVE): derive candidate family if missing
                # - prevents leakage when unit_family wasn't populated upstream
                # =========================
                if not c_fam:
                    try:
                        if callable(_unit_family_fn):
                            c_fam = str(_unit_family_fn(c_ut or "") or "").lower().strip()
                    except Exception:
                        pass
                # =========================

                # =========================
                # PATCH A3.7 (ADDITIVE): prefer unit_tag matching (normalized) over raw unit matching
                # PATCH A3.11 (ADDITIVE): extend normalization fallback to raw/context
                # - helps older snapshots where unit_tag/unit may be empty but raw/context carries scale ("million", "%")
                # =========================
                cand_tag = c_ut
                try:
                    if callable(_norm_tag_fn):
                        cand_tag = _norm_tag_fn(c_ut or cand.get("unit") or cand.get("raw") or ctx)
                except Exception:
                    pass
                # =========================

                # =========================
                # PATCH A3.6 (FIX): schema-first family gate with currency exception
                # - Currency metrics often appear as magnitude candidates ("40.7M") + currency evidence in context.
                # - We allow cand_fam == "magnitude" for expected_family == "currency" ONLY when currency evidence exists.
                # =========================
                if expected_family in ("percent", "currency", "magnitude", "energy"):
                    if expected_family == "currency":
                        if c_fam not in ("currency", "magnitude"):
                            continue
                        if c_fam == "magnitude" and not _has_currency_evidence(cand.get("raw", ""), ctx):
                            continue
                    else:
                        if (c_fam or "") != expected_family:
                            continue
                # =========================

                # dimension/meaning gate using measure_kind when present (soft but helpful)
                mk = cand.get("measure_kind")
                if expected_dim == "percent" and mk and mk not in ("share_pct", "growth_pct", "percent_other"):
                    continue
                if expected_dim == "currency" and mk and mk == "count_units":
                    continue

                c_tokens = set(_tokenize(ctx))
                overlap = sum(1 for t in toks if t in c_tokens) if toks else 0
                score = overlap / max(1, len(toks))

                bonus = 0.0

                # =========================
                # PATCH A3.7 (ADDITIVE): tag-based unit bonus (stronger)
                # =========================
                if exp_tag and cand_tag and cand_tag == exp_tag:
                    bonus += 0.07
                # keep a small legacy bonus if exact unit string matches too
                if expected_unit and (str(cand.get("unit") or "").strip() == expected_unit):
                    bonus += 0.03
                # =========================

                # =========================
                # PATCH A3.8 (ADDITIVE): deterministic value closeness bonus (guarded)
                # - Only apply when units are comparable (tag match or both use value_norm).
                # - Prevents misleading closeness when one side is normalized and the other isn't.
                # =========================
                c_val = _to_float(cand.get("value_norm") if cand.get("value_norm") is not None else cand.get("value"))
                comparable = False
                if m_tag and cand_tag and m_tag == cand_tag:
                    comparable = True
                elif (m.get("value_norm") is not None) and (cand.get("value_norm") is not None):
                    comparable = True

                if comparable and m_val is not None and c_val is not None:
                    denom = max(1e-9, abs(m_val))
                    rel_err = abs(c_val - m_val) / denom
                    if rel_err <= 0.02:
                        bonus += 0.06
                    elif rel_err <= 0.10:
                        bonus += 0.03
                # =========================

                score = float(score + bonus)

                # stable tie-breaker
                key = (
                    score,
                    str(cand.get("source_url") or ""),
                    str(cand.get("anchor_hash") or ""),
                    str(cand.get("raw") or ""),
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best = cand

            if best and best_key and best_key[0] >= 0.10:
                anchors[ckey] = {
                    # =========================
                    # PATCH MA1 (ADDITIVE): legacy compat fields
                    # =========================
                    "metric_id": ckey,
                    "metric_name": (m.get("name") or m.get("original_name") or ckey),
                    # =========================

                    "canonical_key": ckey,
                    "anchor_hash": best.get("anchor_hash"),
                    "source_url": best.get("source_url"),
                    "raw": best.get("raw"),
                    "unit": best.get("unit"),
                    "unit_tag": best.get("unit_tag"),
                    "unit_family": best.get("unit_family"),
                    "base_unit": best.get("base_unit"),
                    "value": best.get("value"),
                    "value_norm": best.get("value_norm"),
                    "measure_kind": best.get("measure_kind"),
                    "measure_assoc": best.get("measure_assoc"),
                    "context_snippet": (best.get("context_snippet") or "")[:220],
                    "anchor_confidence": float(min(100.0, best_key[0] * 100.0)),

                    # =========================
                    # PATCH A3.12 (ADDITIVE): optional fingerprint passthrough (if present)
                    # - Useful later for evolution/debugging; harmless if missing.
                    # =========================
                    "fingerprint": best.get("fingerprint"),
                    # =========================
                }
            else:
                anchors[ckey] = {
                    # =========================
                    # PATCH MA1 (ADDITIVE): legacy compat fields
                    # =========================
                    "metric_id": ckey,
                    "metric_name": (m.get("name") or m.get("original_name") or ckey),
                    # =========================

                    "canonical_key": ckey,
                    "anchor_hash": None,
                    "source_url": None,
                    "raw": None,
                    "anchor_confidence": 0.0,

                    # PATCH A3.12 (ADDITIVE): keep key present for stable shape
                    "fingerprint": None,
                }

        # stable ordering (prefer helper if present)
        try:
            if "sort_metric_anchors" in globals() and callable(globals()["sort_metric_anchors"]):
                ordered = sort_metric_anchors(list(anchors.values()))
                anchors = {
                    a.get("canonical_key"): a
                    for a in ordered
                    if isinstance(a, dict) and a.get("canonical_key")
                }
        except Exception:
            pass

        return anchors

    # -----------------------
    # PATCH A4 (ADDITIVE): enrich analysis (never block saving)
    # -----------------------
    try:
        if isinstance(baseline_cache, list) and baseline_cache:
            evidence_records = _build_evidence_records_from_baseline_cache(baseline_cache)

            # =========================
            # PATCH A4.1 (ADDITIVE): evidence layer versioning (pipeline attribution)
            # - Use CODE_VERSION if available; else keep numeric fallback
            # =========================
            try:
                cv = globals().get("CODE_VERSION")
                analysis.setdefault("evidence_layer_version", cv or 1)
            except Exception:
                analysis.setdefault("evidence_layer_version", 1)
            analysis.setdefault("evidence_layer_schema_version", 1)
            # =========================

            # stash on analysis (additive)
            analysis["evidence_records"] = evidence_records

            # build anchors using canonical metrics + frozen schema if present
            primary_resp = analysis.get("primary_response") or {}
            if isinstance(primary_resp, dict):
                pmc = primary_resp.get("primary_metrics_canonical") or analysis.get("primary_metrics_canonical") or {}
                schema = primary_resp.get("metric_schema_frozen") or analysis.get("metric_schema_frozen") or {}
            else:
                pmc = analysis.get("primary_metrics_canonical") or {}
                schema = analysis.get("metric_schema_frozen") or {}

            metric_anchors = _build_metric_anchors(pmc, schema, evidence_records)
            analysis["metric_anchors"] = metric_anchors
    except Exception:
        pass

    # -----------------------
    # Existing Google Sheet save behavior (guarded)
    # -----------------------
    def _try_make_sheet_json(obj: dict) -> str:
        try:
            fn = globals().get("make_sheet_safe_json")
            if callable(fn):
                return fn(obj)
        except Exception:
            pass
        return json.dumps(obj, ensure_ascii=False, default=str)

    def _shrink_for_sheets(original: dict) -> dict:
        base_copy = dict(original)
        s = _try_make_sheet_json(base_copy)
        if isinstance(s, str) and len(s) <= SHEETS_CELL_LIMIT:
            return base_copy

        reduced = dict(base_copy)
        removed = []

        for k in [
            "evidence_records",
            "baseline_sources_cache",
            "metric_anchors",
            "source_results",
            "web_context",
            "scraped_meta",
            "raw_sources",
            "raw_text",
            "debug",
        ]:
            if k in reduced:
                reduced.pop(k, None)
                removed.append(k)

        reduced.setdefault("_sheet_write", {})
        if isinstance(reduced["_sheet_write"], dict):
            reduced["_sheet_write"]["truncated"] = True
            reduced["_sheet_write"]["removed_keys"] = removed[:50]

        s2 = _try_make_sheet_json(reduced)
        if isinstance(s2, str) and len(s2) <= SHEETS_CELL_LIMIT:
            return reduced

        return {
            "question": original.get("question"),
            "timestamp": original.get("timestamp"),
            "final_confidence": original.get("final_confidence"),
            "question_profile": original.get("question_profile"),
            "primary_response": original.get("primary_response") or {},
            "_sheet_write": {
                "truncated": True,
                "mode": "minimal_fallback",
                "note": "Full analysis too large for single Google Sheets cell (50k limit).",
            },
        }

    # Try Sheets
    try:
        sheet = get_google_sheet()
    except Exception:
        sheet = None

    if not sheet:
        if "analysis_history" not in st.session_state:
            st.session_state.analysis_history = []
        st.session_state.analysis_history.append(analysis)
        try:
            st.session_state["last_analysis"] = analysis
        except Exception:
            pass
        return False

    try:
        analysis_id = generate_analysis_id()


        # =====================================================================
        # PATCH ES1F (ADDITIVE): persist full snapshots + pointer for Sheets rows
        # - If full baseline_sources_cache exists (list-shaped), store it outside
        #   Sheets keyed by source_snapshot_hash, and attach pointer fields into
        #   analysis/results for deterministic evolution rehydration.
        # - Pure enrichment only (no refetch, no heuristics).
        # =====================================================================
        try:
            _bsc = None
            if isinstance(analysis, dict):
                _bsc = analysis.get("results", {}).get("baseline_sources_cache") or analysis.get("baseline_sources_cache")


            # =================================================================
            # PATCH SS6B (ADDITIVE): if snapshots were already summarized away,
            # rebuild minimal snapshot shape from evidence_records (deterministic).
            # This enables snapshot persistence even when baseline_sources_cache
            # is a summary dict in the main analysis object.
            # =================================================================
            try:
                if (not isinstance(_bsc, list)) and isinstance(analysis, dict):
                    _er = None
                    # prefer nested results evidence_records first
                    if isinstance(analysis.get("results"), dict):
                        _er = analysis["results"].get("evidence_records")
                    if _er is None:
                        _er = analysis.get("evidence_records")
                    _rebuilt = build_baseline_sources_cache_from_evidence_records(_er)
                    if isinstance(_rebuilt, list) and _rebuilt:
                        _bsc = _rebuilt
            except Exception:
                pass
            # =================================================================

            if isinstance(_bsc, list) and _bsc:
                _ssh = compute_source_snapshot_hash(_bsc)

                # =========================
                # PATCH A2 (ADD): also compute snapshot hash v2 for stronger identity
                # =========================
                _ssh_v2 = None
                try:
                    _ssh_v2 = compute_source_snapshot_hash_v2(_bsc)
                except Exception:
                    _ssh_v2 = None
                if _ssh:
                    # =============================================================
                    # PATCH SS4 (ADDITIVE): store snapshots to Snapshots worksheet when possible
                    # - Persists full baseline_sources_cache in a dedicated worksheet tab.
                    # - Falls back to local snapshot_store file if Sheets snapshot store unavailable.
                    # - Pointer ref stored as 'gsheet:Snapshots:<hash>' when successful.
                    # =============================================================
                    _gs_ref = ""
                    try:
                        _gs_ref = store_full_snapshots_to_sheet(_bsc, _ssh, worksheet_title="Snapshots")
                        # =========================
                        # PATCH A3 (ADD): mirror-write snapshots under v2 hash as well
                        # =========================
                        if _ssh_v2 and isinstance(_ssh_v2, str) and _ssh_v2 != _ssh:
                            try:
                                store_full_snapshots_to_sheet(_bsc, _ssh_v2, worksheet_title="Snapshots")
                            except Exception:
                                pass
                    except Exception:
                        _gs_ref = ""

                    _ref = store_full_snapshots_local(_bsc, _ssh)

                    analysis["source_snapshot_hash"] = analysis.get("source_snapshot_hash") or _ssh
                    analysis.setdefault("results", {})
                    if isinstance(analysis["results"], dict):
                        analysis["results"]["source_snapshot_hash"] = analysis["results"].get("source_snapshot_hash") or _ssh
                        # PATCH A4 (ADD): store v2 hash in results for downstream consumers
                        try:
                            if _ssh_v2:
                                analysis["results"]["source_snapshot_hash_v2"] = analysis["results"].get("source_snapshot_hash_v2") or _ssh_v2
                        except Exception:
                            pass

                    if _ref:
                        analysis["snapshot_store_ref"] = analysis.get("snapshot_store_ref") or _ref
                        if isinstance(analysis["results"], dict):
                            analysis["results"]["snapshot_store_ref"] = analysis["results"].get("snapshot_store_ref") or _ref
                            # PATCH A5 (ADD): v2 snapshot ref for convenience
                            try:
                                if _ssh_v2:
                                    analysis["results"]["snapshot_store_ref_v2"] = analysis["results"].get("snapshot_store_ref_v2") or f"gsheet:Snapshots:{_ssh_v2}"
                            except Exception:
                                pass
                    # =============================================================
                    # PATCH SS4B (ADDITIVE): prefer Sheets snapshot ref when available
                    # =============================================================
                    try:
                        if _gs_ref:
                            analysis["snapshot_store_ref"] = _gs_ref
                            if isinstance(analysis.get("results"), dict):
                                analysis["results"]["snapshot_store_ref"] = _gs_ref
                    except Exception:
                        pass

        except Exception:
            pass
        # =====================================================================

        payload_for_sheets = _shrink_for_sheets(analysis)
        payload_json = _try_make_sheet_json(payload_for_sheets)

        # =====================================================================
        # PATCH A5 (BUGFIX, REQUIRED): never write invalid JSON to Sheets
        # - Previous hard truncation produced non-JSON (prefix + random suffix),
        #   causing history loaders (json.loads) to skip the row entirely.
        # - This wrapper guarantees valid JSON even when we must truncate.
        # =====================================================================
        if isinstance(payload_json, str) and len(payload_json) > SHEETS_CELL_LIMIT:
            try:
                payload_json = json.dumps(
                    {
                        "_sheet_write": {
                            "truncated": True,
                            "mode": "hard_truncation_wrapper",
                            "note": "Payload exceeded Google Sheets single-cell limit; stored preview only.",
                        },
                        # keep a preview for debugging/UI; still parseable JSON
                        "preview": payload_json[: max(0, SHEETS_CELL_LIMIT - 600)],
                        "analysis_id": analysis_id,
                        "timestamp": analysis.get("timestamp", datetime.now().isoformat()),
                        "question": (analysis.get("question", "") or "")[:200],
                    },
                    ensure_ascii=False,
                    default=str,
                )
            except Exception:
                # ultra-safe fallback: still valid JSON
                payload_json = '{"_sheet_write":{"truncated":true,"mode":"hard_truncation_wrapper","note":"json.dumps failed"}}'
        # =====================================================================

        row = [
            analysis_id,
            analysis.get("timestamp", datetime.now().isoformat()),
            (analysis.get("question", "") or "")[:100],
            str(analysis.get("final_confidence", "")),
            payload_json,
        ]
        sheet.append_row(row, value_input_option="RAW")

        try:
            st.session_state["last_analysis"] = analysis
        except Exception:
            pass

        return True

    except Exception as e:
        st.warning(f"⚠️ Failed to save to Google Sheets: {e}")
        if "analysis_history" not in st.session_state:
            st.session_state.analysis_history = []
        st.session_state.analysis_history.append(analysis)
        try:
            st.session_state["last_analysis"] = analysis
        except Exception:
            pass
        return False


def normalize_unit_tag(unit_str: str) -> str:
    """
    Canonical unit tags used for drift=0 comparisons.
    """
    u = (unit_str or "").strip()
    if not u:
        return ""
    ul = u.lower().replace(" ", "")

    # energy units
    if ul == "twh":
        return "TWh"
    if ul == "gwh":
        return "GWh"
    if ul == "mwh":
        return "MWh"
    if ul == "kwh":
        return "kWh"
    if ul == "wh":
        return "Wh"

    # magnitudes
    if ul in ("t", "trillion", "tn"):
        return "T"
    if ul in ("b", "bn", "billion"):
        return "B"
    if ul in ("m", "mn", "mio", "million"):
        return "M"
    if ul in ("k", "thousand", "000"):
        return "K"

    # percent
    if ul in ("%", "pct", "percent"):
        return "%"

    return u


def unit_family(unit_tag: str) -> str:
    """
    Unit family classifier for gating.
    """
    ut = (unit_tag or "").strip()

    if ut in ("TWh", "GWh", "MWh", "kWh", "Wh"):
        return "energy"
    if ut == "%":
        return "percent"
    if ut in ("T", "B", "M", "K"):
        return "magnitude"

    return ""


def canonicalize_numeric_candidate(candidate: dict) -> dict:
    """
    Additive: attach canonical numeric fields to a candidate dict.
    Safe to call multiple times.
    """
    try:
        v = candidate.get("value")
        if v is None:
            return candidate
        v = float(v)
    except Exception:
        return candidate

    # Prefer unit_tag if already present
    ut = normalize_unit_tag(
        candidate.get("unit_tag") or candidate.get("unit") or ""
    )
    fam = unit_family(ut)

    base_unit = ut
    mult = 1.0

    if ut == "TWh":
        base_unit, mult = "Wh", 1e12
    elif ut == "GWh":
        base_unit, mult = "Wh", 1e9
    elif ut == "MWh":
        base_unit, mult = "Wh", 1e6
    elif ut == "kWh":
        base_unit, mult = "Wh", 1e3
    elif ut == "Wh":
        base_unit, mult = "Wh", 1.0

    candidate.setdefault("unit_tag", ut)
    candidate.setdefault("unit_family", fam)
    candidate.setdefault("base_unit", base_unit)
    candidate.setdefault("multiplier_to_base", mult)
    candidate.setdefault("value_norm", v * mult)

    return candidate


def rebuild_metrics_from_snapshots(
    prev_response: dict,
    baseline_sources_cache: list,
    web_context: dict = None
) -> dict:
    """
    Deterministic rebuild using cached snapshots only.
    If sources unchanged, rebuilt metrics converge with analysis.

    Behavior:
      1) Primary: anchor_hash match via prev_response.metric_anchors
      2) Fallback: schema-first deterministic selection when anchor missing
         using metric_schema_frozen + context match + deterministic tie-break.

    NOTE: Dead/unreachable legacy code previously below an early return has been removed
    (explicitly approved).
    """
    import re
    import hashlib

    # =========================
    # PATCH RMS0 (ADDITIVE): typing imports for Dict/Any/List used below
    # - Prevents NameError if typing symbols are not imported globally.
    # =========================
    from typing import Dict, Any, List
    # =========================

    prev_response = prev_response if isinstance(prev_response, dict) else {}

    # =========================
    # PATCH RMS0.1 (ADDITIVE): accept anchors stored under alternate keys
    # - Backward compatible: does not change existing behavior if metric_anchors exists.
    # =========================
    prev_anchors = (
        prev_response.get("metric_anchors")
        or prev_response.get("anchors")
        or {}
    )
    # =========================

    if not isinstance(prev_anchors, dict):
        prev_anchors = {}

    rebuilt: Dict[str, Any] = {}

    # ---------- schema + canonical lookup ----------
    metric_schema = prev_response.get("metric_schema_frozen") or {}
    if not isinstance(metric_schema, dict):
        metric_schema = {}

    # =========================
    # PATCH RB2 (ADDITIVE): ensure baseline_sources_cache is a full list (rehydrate from snapshot store if needed)
    # - Handles cases where history rows store only a summarized baseline_sources_cache, but full snapshots exist
    #   in the Snapshots sheet (referenced by snapshot_store_ref / source_snapshot_hash).
    # =========================
    try:
        if (not isinstance(baseline_sources_cache, list)) or (isinstance(baseline_sources_cache, dict) and baseline_sources_cache.get("_summary") is True):
            # Prefer already-rehydrated cache on prev_response["results"]["baseline_sources_cache"]
            _maybe = (prev_response.get("results", {}) or {}).get("baseline_sources_cache")
            if isinstance(_maybe, list) and _maybe:
                baseline_sources_cache = _maybe
            else:
                store_ref = prev_response.get("snapshot_store_ref") or (prev_response.get("results", {}) or {}).get("snapshot_store_ref")
                source_hash = prev_response.get("source_snapshot_hash") or (prev_response.get("results", {}) or {}).get("source_snapshot_hash")
                if (not store_ref) and source_hash:
                    store_ref = f"gsheet:Snapshots:{source_hash}"
                if isinstance(store_ref, str) and store_ref.startswith("gsheet:Snapshots:"):
                    _hash = store_ref.split(":")[-1]
                    _full = load_full_snapshots_from_sheet(_hash)
                    if isinstance(_full, list) and _full:
                        baseline_sources_cache = _full
    except Exception:
        pass

    prev_can = prev_response.get("primary_metrics_canonical") or {}
    if not isinstance(prev_can, dict):
        prev_can = {}

    # =========================
    # PATCH RMS0.2 (ADDITIVE): compute full metric key universe
    # - Important: some metrics may not have anchors yet; we still must rebuild them
    #   (otherwise evolution "misses" metrics and diffs become unstable).
    # =========================
    metric_key_universe = set()
    try:
        metric_key_universe.update(list(prev_can.keys()))
        metric_key_universe.update(list(prev_anchors.keys()))
    except Exception:
        metric_key_universe = set(prev_can.keys()) if isinstance(prev_can, dict) else set()
    # =========================

    # ---------- deterministic candidate id (tie-breaker) ----------
    def _candidate_id(c: dict) -> str:
        try:
            url = str(c.get("source_url") or c.get("url") or "")
            ah = str(c.get("anchor_hash") or "")
            vn = c.get("value_norm")
            bu = str(c.get("base_unit") or c.get("unit") or c.get("unit_tag") or "")
            mk = str(c.get("measure_kind") or "")
            vn_s = ""
            if vn is not None:
                try:
                    vn_s = f"{float(vn):.12g}"
                except Exception:
                    vn_s = str(vn)
            s = f"{url}|{ah}|{vn_s}|{bu}|{mk}"
            return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return ""

    # =====================================================================
    # PATCH RMS_E0 (ADDITIVE): small evidence extraction helper
    # - Ensures we consistently carry anchor/evidence fields onto rebuilt metrics.
    # - Purely additive; never affects selection logic.
    # =====================================================================
    def _extract_evidence_fields(c: dict) -> dict:
        if not isinstance(c, dict):
            return {}
        ctx = (c.get("context_snippet") or c.get("context") or "").strip()
        return {
            "raw": c.get("raw"),
            "candidate_id": c.get("candidate_id") or _candidate_id(c),
            "context_snippet": ctx[:240] if isinstance(ctx, str) else None,
            "measure_kind": c.get("measure_kind"),
            "measure_assoc": c.get("measure_assoc"),
            "start_idx": c.get("start_idx"),
            "end_idx": c.get("end_idx"),
            # optional passthroughs if upstream provides them
            "fingerprint": c.get("fingerprint"),
        }
    # =====================================================================

    # =====================================================================
    # PATCH RMS_E1 (ADDITIVE): anchor metadata getter
    # - Pull anchor_confidence (and any other safe fields) from prev_anchors entry.
    # - Helps diff/UI show confidence without recomputing.
    # =====================================================================
    def _anchor_meta(anchor_obj) -> dict:
        if isinstance(anchor_obj, dict):
            out = {}
            if anchor_obj.get("anchor_confidence") is not None:
                try:
                    out["anchor_confidence"] = float(anchor_obj.get("anchor_confidence"))
                except Exception:
                    pass
            # optional passthroughs if present
            if anchor_obj.get("source_url"):
                out["anchor_source_url"] = anchor_obj.get("source_url")
            if anchor_obj.get("raw"):
                out["anchor_raw"] = anchor_obj.get("raw")
            if anchor_obj.get("candidate_id"):
                out["anchor_candidate_id"] = anchor_obj.get("candidate_id")
            return out
        return {}
    # =====================================================================

    # ---------- collect candidates + anchor map ----------
    anchor_to_candidate: Dict[str, Dict[str, Any]] = {}
    all_candidates: List[Dict[str, Any]] = []

    for src in baseline_sources_cache or []:
        if not isinstance(src, dict):
            continue
        src_url = src.get("url") or src.get("source_url") or ""

        # =================================================================
        # PATCH RMS_E2 (ADDITIVE): capture source fingerprint on candidates
        # - Helps later debugging and “same source” proofs.
        # =================================================================
        src_fp = src.get("fingerprint")
        # =================================================================

        for c in (src.get("extracted_numbers") or []):
            if not isinstance(c, dict):
                continue

            # canonicalize if available (safe if repeated)
            try:
                c = canonicalize_numeric_candidate(dict(c))
            except Exception:
                c = dict(c)

            # ensure stable url carried through
            if not c.get("source_url"):
                c["source_url"] = src_url

            # =============================================================
            # PATCH RMS_E2 (ADDITIVE): attach fingerprint if missing
            # =============================================================
            if src_fp and not c.get("fingerprint"):
                c["fingerprint"] = src_fp
            # =============================================================

            ah = c.get("anchor_hash")
            if ah:
                if ah not in anchor_to_candidate:
                    anchor_to_candidate[ah] = c
                else:
                    old = anchor_to_candidate[ah]
                    if old.get("is_junk") and not c.get("is_junk"):
                        anchor_to_candidate[ah] = c

            all_candidates.append(c)

    # ---------- schema-first helpers ----------
    def _schema_for_key(metric_key: str) -> dict:
        d = metric_schema.get(metric_key)
        return d if isinstance(d, dict) else {}

    def _expected_from_schema(metric_key: str):
        d = _schema_for_key(metric_key)

        unit_family_s = str(d.get("unit_family") or "").strip().lower()
        dim_s = str(d.get("dimension") or "").strip().lower()
        unit_s = str(d.get("unit") or "").strip()
        name_l = str(d.get("name") or "").lower()

        expected_family = ""
        if unit_family_s in ("percent", "currency", "energy"):
            expected_family = unit_family_s
        else:
            ut = normalize_unit_tag(unit_s)
            if ut == "%":
                expected_family = "percent"
            elif ut in ("TWh", "GWh", "MWh", "kWh", "Wh"):
                expected_family = "energy"
            elif dim_s == "currency":
                expected_family = "currency"

        currencyish = (unit_family_s == "currency" or dim_s == "currency")

        expected_kind = None
        if expected_family == "percent":
            if any(k in name_l for k in ["growth", "cagr", "increase", "decrease", "yoy", "qoq", "mom", "rate"]):
                expected_kind = "growth_pct"
            else:
                expected_kind = "share_pct"
        if currencyish or expected_family == "currency":
            expected_kind = "money"
        if expected_kind is None and any(k in name_l for k in [
            "units", "unit sales", "vehicle sales", "vehicles sold", "sold",
            "deliveries", "shipments", "registrations", "volume"
        ]):
            expected_kind = "count_units"

        kw = d.get("keywords")
        schema_keywords = [str(x).strip() for x in kw] if isinstance(kw, list) else []
        schema_keywords = [x for x in schema_keywords if x]

        return expected_family, currencyish, expected_kind, schema_keywords, unit_s

    def _ctx_match_score(tokens: List[str], ctx: str) -> float:
        fn = globals().get("calculate_context_match")
        if callable(fn):
            try:
                return float(fn(tokens, ctx))
            except Exception:
                pass

        c = (ctx or "").lower()
        toks = [t.lower() for t in (tokens or []) if t and len(t) >= 2]
        if not toks:
            return 0.0
        hit = sum(1 for t in toks if t in c)
        return hit / max(1, len(toks))

    def _currency_evidence(raw: str, ctx: str) -> bool:
        r = (raw or "")
        c = (ctx or "").lower()
        if any(s in r for s in ["$", "S$", "€", "£"]):
            return True
        if any(code in c for code in [" usd", "sgd", " eur", " gbp", " aud", " cad", " jpy", " cny", " rmb"]):
            return True
        if any(k in c for k in ["revenue", "turnover", "valuation", "market size", "market value", "profit", "earnings", "ebitda"]):
            return True
        return False

    def _is_yearish_value(v) -> bool:
        try:
            iv = int(float(v))
            return 1900 <= iv <= 2099
        except Exception:
            return False

    # =========================
    # PATCH RMS_BASE (ADDITIVE): helper to overlay rebuilt fields onto prior canonical metric
    # - Keeps metric identity fields (name/canonical_key/dimension/etc.) stable for diffing.
    # - Only overwrites value-ish/source-ish fields with rebuilt candidate data.
    # =========================
    def _overlay_base(metric_key: str, patch: dict) -> dict:
        base = {}
        try:
            if isinstance(prev_can.get(metric_key), dict):
                base = dict(prev_can.get(metric_key) or {})
        except Exception:
            base = {}
        out = dict(base)
        try:
            if isinstance(patch, dict):
                out.update(patch)
        except Exception:
            pass
        return out
    # =========================

    # ---------- 1) primary rebuild by anchor ----------
    rebuilt_by_anchor = set()

    for metric_key, anchor in prev_anchors.items():
        ah = None
        if isinstance(anchor, dict):
            ah = anchor.get("anchor_hash") or anchor.get("anchor")
        elif isinstance(anchor, str):
            ah = anchor

        if ah and ah in anchor_to_candidate:
            c = anchor_to_candidate[ah]

            # =========================
            # PATCH RMS1 (ADDITIVE): overlay rebuilt candidate onto base canonical metric
            # - Keeps canonical identity fields intact for downstream diffs/UI.
            # =========================
            rebuilt[metric_key] = _overlay_base(metric_key, {
                "value": c.get("value"),
                "unit": c.get("unit"),
                "value_norm": c.get("value_norm"),
                "base_unit": c.get("base_unit"),
                "unit_tag": c.get("unit_tag"),
                "unit_family": c.get("unit_family"),
                "anchor_hash": ah,
                "source_url": c.get("source_url"),
                "context_snippet": (c.get("context_snippet") or c.get("context") or "")[:240],
                "measure_kind": c.get("measure_kind"),
                "measure_assoc": c.get("measure_assoc"),
                "rebuild_method": "anchor",

                # =============================================================
                # PATCH RMS_E3 (ADDITIVE): attach evidence + anchor metadata
                # - candidate_id used as stable ID for UI/debugging
                # - anchor_confidence helps diff/UI set match_confidence
                # =============================================================
                **_extract_evidence_fields(c),
                **_anchor_meta(anchor),
                # =============================================================
            })
            # =========================

            rebuilt_by_anchor.add(metric_key)

    # ---------- 2) fallback rebuild when anchor missing ----------
    # NOTE: existing loop only iterated prev_anchors.keys(); we keep it as-is,
    # and then add an extra additive loop to cover metrics without anchors. (PATCH RMS2)
    for metric_key in prev_anchors.keys():
        if metric_key in rebuilt_by_anchor:
            continue

        expected_family, currencyish, expected_kind, schema_keywords, schema_unit = _expected_from_schema(metric_key)

        # conservative fallback if schema is thin
        if not expected_family and metric_key in prev_can and isinstance(prev_can.get(metric_key), dict):
            pm = prev_can.get(metric_key) or {}
            ut = normalize_unit_tag(pm.get("unit") or schema_unit or "")
            if ut == "%":
                expected_family = "percent"
            elif ut in ("TWh", "GWh", "MWh", "kWh", "Wh"):
                expected_family = "energy"

        # tokens for context scoring
        tokens = []
        if schema_keywords:
            tokens = schema_keywords
        else:
            # fallback to build_metric_keywords(schema_name)
            schema_name = ""
            try:
                schema_name = str(_schema_for_key(metric_key).get("name") or "")
            except Exception:
                schema_name = ""
            fn_bmk = globals().get("build_metric_keywords")
            if callable(fn_bmk):
                try:
                    tokens = fn_bmk(schema_name or metric_key) or []
                except Exception:
                    tokens = []
            else:
                tokens = []

        best = None
        best_key = None
        best_score = -1.0

        for c in all_candidates:
            if not isinstance(c, dict):
                continue

            # fallback skips junk (anchor path already handled above)
            if c.get("is_junk") is True:
                continue

            ctx = (c.get("context") or c.get("context_snippet") or "").strip()
            if not ctx:
                continue

            # stop timeline years contaminating non-year metrics
            if expected_family not in ("percent", "energy") and not (currencyish or expected_family == "currency"):
                if (c.get("unit_tag") in ("", None)) and _is_yearish_value(c.get("value")):
                    continue

            cand_ut = c.get("unit_tag") or normalize_unit_tag(c.get("unit") or "")
            cand_fam = (c.get("unit_family") or unit_family(cand_ut) or "").strip().lower()
            mk = c.get("measure_kind")

            # unit-family gating
            if expected_family == "percent":
                if cand_fam != "percent" and cand_ut != "%":
                    continue
            elif expected_family == "energy":
                if cand_fam != "energy":
                    continue
            elif currencyish or expected_family == "currency":
                if cand_fam not in ("currency", "magnitude"):
                    continue
                if not _currency_evidence(c.get("raw", ""), ctx):
                    continue
                if mk == "count_units":
                    continue

            # measure-kind gating (only if candidate provides it)
            if expected_kind and mk and mk != expected_kind:
                continue

            # normalize value for ranking
            try:
                c2 = canonicalize_numeric_candidate(dict(c))
            except Exception:
                c2 = c

            val_norm = c2.get("value_norm")
            if val_norm is None:
                try:
                    val_norm = float(c2.get("value"))
                except Exception:
                    continue

            ctx_score = _ctx_match_score(tokens, ctx)
            if ctx_score <= 0.0:
                continue

            url = str(c2.get("source_url") or c2.get("url") or "")
            cid = c2.get("candidate_id") or _candidate_id({**c2, "value_norm": val_norm})

            # deterministic tie-break (max)
            key = (
                float(ctx_score),
                float(val_norm),
                url,
                str(cid),
            )

            if best_key is None or key > best_key:
                best_key = key
                best_score = float(ctx_score)
                best = {**c2, "value_norm": val_norm, "candidate_id": cid}

        if best:
            # =========================
            # PATCH RMS1 (ADDITIVE): overlay onto base canonical metric
            # =========================
            rebuilt[metric_key] = _overlay_base(metric_key, {
                "value": best.get("value"),
                "unit": best.get("unit") or best.get("unit_tag"),
                "value_norm": best.get("value_norm"),
                "base_unit": best.get("base_unit"),
                "unit_tag": best.get("unit_tag"),
                "unit_family": best.get("unit_family"),
                "anchor_hash": best.get("anchor_hash"),
                "source_url": best.get("source_url") or best.get("url"),
                "context_snippet": (best.get("context_snippet") or best.get("context") or "")[:240],
                "measure_kind": best.get("measure_kind"),
                "measure_assoc": best.get("measure_assoc"),
                "rebuild_method": "schema_fallback",
                "fallback_ctx_score": round(best_score, 6),
                "candidate_id": best.get("candidate_id"),

                # =============================================================
                # PATCH RMS_E4 (ADDITIVE): attach standardized evidence fields
                # - Ensures candidate_id/raw/context are always present when possible.
                # - Adds anchor_confidence derived from fallback_ctx_score.
                # =============================================================
                **_extract_evidence_fields(best),
                "anchor_confidence": float(min(100.0, max(0.0, best_score) * 100.0)) if best_score is not None else 0.0,
                # =============================================================
            })
            # =========================

    # =========================
    # PATCH RMS2 (ADDITIVE): ensure metrics without anchors are also rebuilt
    # - Your existing fallback loop only iterates prev_anchors.keys().
    # - This loop covers the remaining canonical metrics (prev_can keys) that are missing
    #   from prev_anchors, using the SAME schema-first logic (copied, not refactored).
    # - Additive: does not alter prior behavior for anchored metrics.
    # =========================
    for metric_key in (metric_key_universe or set()):
        if metric_key in rebuilt:
            continue

        expected_family, currencyish, expected_kind, schema_keywords, schema_unit = _expected_from_schema(metric_key)

        if not expected_family and metric_key in prev_can and isinstance(prev_can.get(metric_key), dict):
            pm = prev_can.get(metric_key) or {}
            ut = normalize_unit_tag(pm.get("unit") or schema_unit or "")
            if ut == "%":
                expected_family = "percent"
            elif ut in ("TWh", "GWh", "MWh", "kWh", "Wh"):
                expected_family = "energy"

        tokens = []
        if schema_keywords:
            tokens = schema_keywords
        else:
            schema_name = ""
            try:
                schema_name = str(_schema_for_key(metric_key).get("name") or "")
            except Exception:
                schema_name = ""
            fn_bmk = globals().get("build_metric_keywords")
            if callable(fn_bmk):
                try:
                    tokens = fn_bmk(schema_name or metric_key) or []
                except Exception:
                    tokens = []
            else:
                tokens = []

        best = None
        best_key = None
        best_score = -1.0

        for c in all_candidates:
            if not isinstance(c, dict):
                continue
            if c.get("is_junk") is True:
                continue

            ctx = (c.get("context") or c.get("context_snippet") or "").strip()
            if not ctx:
                continue

            if expected_family not in ("percent", "energy") and not (currencyish or expected_family == "currency"):
                if (c.get("unit_tag") in ("", None)) and _is_yearish_value(c.get("value")):
                    continue

            cand_ut = c.get("unit_tag") or normalize_unit_tag(c.get("unit") or "")
            cand_fam = (c.get("unit_family") or unit_family(cand_ut) or "").strip().lower()
            mk = c.get("measure_kind")

            if expected_family == "percent":
                if cand_fam != "percent" and cand_ut != "%":
                    continue
            elif expected_family == "energy":
                if cand_fam != "energy":
                    continue
            elif currencyish or expected_family == "currency":
                if cand_fam not in ("currency", "magnitude"):
                    continue
                if not _currency_evidence(c.get("raw", ""), ctx):
                    continue
                if mk == "count_units":
                    continue

            if expected_kind and mk and mk != expected_kind:
                continue

            try:
                c2 = canonicalize_numeric_candidate(dict(c))
            except Exception:
                c2 = c

            val_norm = c2.get("value_norm")
            if val_norm is None:
                try:
                    val_norm = float(c2.get("value"))
                except Exception:
                    continue

            ctx_score = _ctx_match_score(tokens, ctx)
            if ctx_score <= 0.0:
                continue

            url = str(c2.get("source_url") or c2.get("url") or "")
            cid = c2.get("candidate_id") or _candidate_id({**c2, "value_norm": val_norm})

            key = (
                float(ctx_score),
                float(val_norm),
                url,
                str(cid),
            )

            if best_key is None or key > best_key:
                best_key = key
                best_score = float(ctx_score)
                best = {**c2, "value_norm": val_norm, "candidate_id": cid}

        if best:
            rebuilt[metric_key] = _overlay_base(metric_key, {
                "value": best.get("value"),
                "unit": best.get("unit") or best.get("unit_tag"),
                "value_norm": best.get("value_norm"),
                "base_unit": best.get("base_unit"),
                "unit_tag": best.get("unit_tag"),
                "unit_family": best.get("unit_family"),
                "anchor_hash": best.get("anchor_hash"),
                "source_url": best.get("source_url") or best.get("url"),
                "context_snippet": (best.get("context_snippet") or best.get("context") or "")[:240],
                "measure_kind": best.get("measure_kind"),
                "measure_assoc": best.get("measure_assoc"),
                "rebuild_method": "schema_fallback_no_anchor",
                "fallback_ctx_score": round(best_score, 6),
                "candidate_id": best.get("candidate_id"),

                # =============================================================
                # PATCH RMS_E5 (ADDITIVE): attach standardized evidence fields
                # =============================================================
                **_extract_evidence_fields(best),
                "anchor_confidence": float(min(100.0, max(0.0, best_score) * 100.0)) if best_score is not None else 0.0,
                # =============================================================
            })
        else:
            # stable placeholder (do not fabricate)
            if isinstance(prev_can.get(metric_key), dict):
                rebuilt[metric_key] = _overlay_base(metric_key, {
                    "rebuild_method": "not_found_in_snapshots",

                    # =============================================================
                    # PATCH RMS_E6 (ADDITIVE): keep evidence fields present for stable shape
                    # =============================================================
                    "anchor_hash": None,
                    "source_url": None,
                    "context_snippet": None,
                    "raw": None,
                    "candidate_id": None,
                    "anchor_confidence": 0.0,
                    # =============================================================
                })
    # =========================

    # =====================================================================
    # PATCH RMS_FALLBACK1 (ADDITIVE): never return empty rebuild when we have a baseline universe
    # Why:
    #   - Source-anchored evolution is snapshot-gated; if snapshots exist but rebuild fails
    #     (missing anchors/schema mismatch/edge cases), returning {} causes evolution to hard-fail.
    #   - For determinism + drift-0 testing, we prefer a safe fallback that preserves the
    #     canonical metric universe from the previous analysis while emitting an explicit flag.
    #
    # Behavior:
    #   - If 'rebuilt' is empty/non-dict, fall back to prev_response['primary_metrics_canonical'].
    #   - Marks each metric with '_rebuild_fallback_used': True (additive field).
    #   - DOES NOT fabricate new values; it reuses previous canonical values only.
    # =====================================================================
    try:
        if not isinstance(rebuilt, dict) or not rebuilt:
            prev_universe = {}
            if isinstance(prev_response, dict):
                prev_universe = prev_response.get("primary_metrics_canonical") or {}
            if isinstance(prev_universe, dict) and prev_universe:
                rebuilt = {}
                for ck in sorted(prev_universe.keys()):
                    m = prev_universe.get(ck)
                    if isinstance(m, dict):
                        mm = dict(m)
                        mm["_rebuild_fallback_used"] = True
                        # Ensure ES7 fields exist (pure enrichment)
                        mm.setdefault("canonical_key", ck)
                        mm.setdefault("anchor_used", False)
                        mm.setdefault("anchor_confidence", 0.0)
                        rebuilt[ck] = mm
                # Add top-level marker (additive)
                try:
                    rebuilt["_rebuild_status"] = "fallback_prev_primary_metrics_canonical"
                except Exception:
                    pass
    except Exception:
        pass
    # =====================================================================

    return rebuilt




# =====================================================================
# PATCH RMS_MIN1 (ADDITIVE): Minimal schema-driven rebuild from snapshots
# ---------------------------------------------------------------------
# Goal:
#   - Provide a deterministic, evolution-safe metric rebuild that uses ONLY:
#       (a) baseline_sources_cache snapshots (and their extracted_numbers)
#       (b) frozen metric schema (metric_schema_frozen)
#   - No re-fetch, no LLM inference, no heuristic "best guess" beyond schema fields.
#
# Contract:
#   - Returns a dict shaped like primary_metrics_canonical:
#       { canonical_key: { ...metric fields... } }
#   - Deterministic tie-break ordering.
# =====================================================================

def rebuild_metrics_from_snapshots_schema_only(
    prev_response: dict,
    baseline_sources_cache: list,
    web_context: dict = None
) -> dict:
    """Schema-driven deterministic rebuild from cached snapshots only.

    This is intentionally minimal:
      - It does NOT attempt free-form metric discovery.
      - It ONLY populates metrics declared in the frozen schema.
      - Candidate selection is driven by schema fields (keywords + unit family/tag).
      - Deterministic sorting ensures stable output ordering.

    Returns:
      Dict[str, Dict] shaped like primary_metrics_canonical.
    """
    import re

    # -------------------------
    # Resolve frozen schema (supports multiple storage locations)
    # -------------------------
    schema = None
    try:
        if isinstance(prev_response, dict):
            if isinstance(prev_response.get("metric_schema_frozen"), dict):
                schema = prev_response.get("metric_schema_frozen")
            elif isinstance(prev_response.get("primary_response"), dict) and isinstance(prev_response["primary_response"].get("metric_schema_frozen"), dict):
                schema = prev_response["primary_response"].get("metric_schema_frozen")
            elif isinstance(prev_response.get("results"), dict) and isinstance(prev_response["results"].get("metric_schema_frozen"), dict):
                schema = prev_response["results"].get("metric_schema_frozen")
    except Exception:
        schema = None

    if not isinstance(schema, dict) or not schema:
        return {}

    # -------------------------
    # Collect candidates from snapshots (no re-fetch)
    # -------------------------
    candidates = []
    if isinstance(baseline_sources_cache, list):
        for src in baseline_sources_cache:
            if not isinstance(src, dict):
                continue
            nums = src.get("extracted_numbers")
            if not isinstance(nums, list) or not nums:
                continue
            for n in nums:
                if not isinstance(n, dict):
                    continue
                # Filter junk deterministically (schema-driven rebuild doesn't want nav chrome)
                if n.get("is_junk") is True:
                    continue
                # Normalize a few fields to ensure stable downstream access
                c = dict(n)
                if not c.get("source_url"):
                    c["source_url"] = src.get("url", "") or src.get("source_url", "") or ""
                candidates.append(c)

    # Deterministic candidate ordering (no set/dict iteration surprises)
    def _cand_sort_key(c: dict):
        return (
            str(c.get("source_url") or ""),
            str(c.get("anchor_hash") or ""),
            int(c.get("start_idx") or 0),
            str(c.get("raw") or ""),
            str(c.get("unit_tag") or ""),
            str(c.get("unit_family") or ""),
            float(c.get("value_norm") or c.get("value") or 0.0),
        )

    candidates.sort(key=_cand_sort_key)

    if not candidates:
        return {}

    # -------------------------
    # Deterministic schema-driven selection
    # -------------------------
    def _norm_text(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    out = {}

    for canonical_key in sorted(schema.keys()):
        spec = schema.get(canonical_key) or {}
        if not isinstance(spec, dict):
            continue

        spec_keywords = spec.get("keywords") or []
        if not isinstance(spec_keywords, list):
            spec_keywords = []
        spec_keywords_norm = [str(k).lower().strip() for k in spec_keywords if str(k).strip()]

        spec_unit_tag = str(spec.get("unit_tag") or spec.get("unit") or "").strip()
        spec_unit_family = str(spec.get("unit_family") or "").strip()

        # Score candidates by schema keyword hits, then filter by unit constraints if present.
        best = None
        best_key = None

        for c in candidates:
            ctx = _norm_text(c.get("context_snippet") or "")
            if not ctx:
                continue

            # Keyword hits: schema-driven (no external heuristics)
            hits = 0
            if spec_keywords_norm:
                for kw in spec_keywords_norm:
                    if kw and kw in ctx:
                        hits += 1

            if spec_keywords_norm and hits == 0:
                continue

            # Unit constraints (only if schema declares them)
            if spec_unit_family:
                if str(c.get("unit_family") or "").strip() != spec_unit_family:
                    # allow a unit_tag-only match when family is missing in candidate
                    if not (spec_unit_tag and str(c.get("unit_tag") or "").strip() == spec_unit_tag):
                        continue

            if spec_unit_tag:
                # if a tag is specified, prefer exact tag matches
                if str(c.get("unit_tag") or "").strip() != spec_unit_tag:
                    # allow family match when tag differs
                    if not (spec_unit_family and str(c.get("unit_family") or "").strip() == spec_unit_family):
                        continue

            # Deterministic tie-break:
            #   (-hits, then stable candidate identity tuple)
            tie = (-hits,) + _cand_sort_key(c)
            if best is None or tie < best_key:
                best = c
                best_key = tie

        if not isinstance(best, dict):
            continue

        # Emit a minimal canonical metric row (schema-driven, deterministic)
        metric = {
            "name": spec.get("name") or spec.get("canonical_id") or canonical_key,
            "value": best.get("value"),
            "unit": best.get("unit") or spec.get("unit") or "",
            "unit_tag": best.get("unit_tag") or spec.get("unit_tag") or "",
            "unit_family": best.get("unit_family") or spec.get("unit_family") or "",
            "base_unit": best.get("base_unit") or best.get("unit_tag") or spec.get("unit_tag") or "",
            "multiplier_to_base": best.get("multiplier_to_base") if best.get("multiplier_to_base") is not None else 1.0,
            "value_norm": best.get("value_norm") if best.get("value_norm") is not None else best.get("value"),
            "canonical_id": spec.get("canonical_id") or spec.get("canonical_key") or canonical_key,
            "canonical_key": canonical_key,
            "dimension": spec.get("dimension") or "",
            "original_name": spec.get("name") or "",
            "geo_scope": "unknown",
            "geo_name": "",
            "is_proxy": False,
            "proxy_type": "",
            "provenance": {
                "method": "schema_keyword_match",
                "best_candidate": {
                    "raw": best.get("raw"),
                    "source_url": best.get("source_url"),
                    "context_snippet": best.get("context_snippet"),
                    "anchor_hash": best.get("anchor_hash"),
                    "start_idx": best.get("start_idx"),
                    "end_idx": best.get("end_idx"),
                },
            },
        }

        out[canonical_key] = metric

    return out



# ===================== PATCH RMS_AWARE1 (ADDITIVE) =====================
def rebuild_metrics_from_snapshots_with_anchors(prev_response: dict, baseline_sources_cache, web_context=None) -> dict:
    """
    Anchor-aware deterministic rebuild (analysis-aligned):
      - Uses ONLY snapshots/cache + frozen schema + prior metric_anchors (if present)
      - No re-fetch
      - No heuristic matching outside anchor_hash + schema dimension checks
      - Deterministic ordering and selection

    Strategy:
      1) Load metric_anchors (canonical_key -> {anchor_hash, ...}) from prev_response (any common nesting).
      2) Flatten snapshot candidates (extracted_numbers) from baseline_sources_cache.
      3) For each canonical_key with an anchor_hash:
           pick candidate with matching anchor_hash (and compatible unit family if inferable).
      4) Build primary_metrics_canonical-like dict.

    Returns: dict {canonical_key: metric_obj}
    """
    import re

    if not isinstance(prev_response, dict):
        return {}

    # 1) Pull anchors from any common location
    metric_anchors = (
        prev_response.get("metric_anchors")
        or (prev_response.get("primary_response") or {}).get("metric_anchors")
        or (prev_response.get("results") or {}).get("metric_anchors")
    )
    if not isinstance(metric_anchors, dict) or not metric_anchors:
        return {}

    # 2) Pull frozen schema (for name/dimension hints; optional but preferred)
    metric_schema = (
        prev_response.get("metric_schema_frozen")
        or (prev_response.get("primary_response") or {}).get("metric_schema_frozen")
        or (prev_response.get("results") or {}).get("metric_schema_frozen")
        or {}
    )

    # Flatten candidates from baseline_sources_cache (list of source dicts with extracted_numbers)
    if isinstance(baseline_sources_cache, dict) and isinstance(baseline_sources_cache.get("snapshots"), list):
        sources = baseline_sources_cache.get("snapshots", [])
    elif isinstance(baseline_sources_cache, list):
        sources = baseline_sources_cache
    else:
        sources = []

    candidates = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        url = s.get("source_url") or s.get("url") or ""
        xs = s.get("extracted_numbers")
        if isinstance(xs, list) and xs:
            for c in xs:
                if not isinstance(c, dict):
                    continue
                if c.get("is_junk") is True:
                    continue
                c2 = dict(c)
                c2.setdefault("source_url", url)
                candidates.append(c2)

    # Deterministic sort key (stable across runs)
    def _cand_sort_key(c: dict):
        try:
            return (
                str(c.get("anchor_hash") or ""),
                str(c.get("source_url") or ""),
                int(c.get("start_idx") or 0),
                str(c.get("raw") or ""),
                str(c.get("unit") or ""),
                float(c.get("value_norm") or 0.0),
            )
        except Exception:
            return ("", "", 0, "", "", 0.0)

    candidates.sort(key=_cand_sort_key)

    # Unit family inference (lightweight; used only as a compatibility guard)
    def _unit_family(unit: str) -> str:
        u = (unit or "").strip().lower()
        if u in ("%", "percent", "percentage"):
            return "percent"
        if any(x in u for x in ("usd", "$", "eur", "gbp", "jpy", "cny", "aud", "sgd")):
            return "currency"
        if any(x in u for x in ("unit", "units", "vehicle", "vehicles", "kwh", "mwh", "gwh", "twh", "ton", "tons")):
            return "quantity"
        return ""

    rebuilt = {}

    # 3) Anchor_hash match first (no schema-free guessing)
    for canonical_key, a in metric_anchors.items():
        if not isinstance(a, dict):
            continue
        ah = a.get("anchor_hash") or a.get("anchor") or ""
        if not ah:
            continue

        sch = metric_schema.get(canonical_key) if isinstance(metric_schema, dict) else None
        name = (sch or {}).get("name") or a.get("name") or canonical_key
        expected_dim = ((sch or {}).get("dimension") or (sch or {}).get("unit_family") or "").strip().lower()

        best = None
        for c in candidates:
            if (c.get("anchor_hash") or "") != ah:
                continue
            # If we can infer unit family, enforce compatibility when expected_dim is given
            fam = _unit_family(c.get("unit") or "")
            if expected_dim and fam and expected_dim != fam:
                continue
            best = c
            break

        if not best:
            continue

        rebuilt[canonical_key] = {
            "canonical_key": canonical_key,
            "name": name,
            "value": best.get("value"),
            "unit": best.get("unit") or "",
            "value_norm": best.get("value_norm"),
            "source_url": best.get("source_url") or "",
            "anchor_hash": best.get("anchor_hash") or "",
            "evidence": [{
                "source_url": best.get("source_url") or "",
                "raw": best.get("raw") or "",
                "context_snippet": (best.get("context") or best.get("context_window") or "")[:400],
                "anchor_hash": best.get("anchor_hash") or "",
                "method": "anchor_hash_rebuild",
            }],
        }

    return rebuilt
# =================== END PATCH RMS_AWARE1 (ADDITIVE) ===================



def get_history(limit: int = MAX_HISTORY_ITEMS) -> List[Dict]:
    """Load analysis history from Google Sheet"""
    sheet = get_google_sheet()
    if not sheet:
        # Fallback to session state
        return st.session_state.get('analysis_history', [])

    try:
        # ============================================================
        # PATCH GH_KEY1 (ADDITIVE): Use the actual worksheet title as cache key
        # Why:
        # - Your sheet names are: 'Sheet1', 'Snapshots', 'HistoryFull'
        # - There is no worksheet called 'History'
        # - Using cache_key='History' can cache empty reads under the wrong key.
        # ============================================================
        _ws_title = getattr(sheet, "title", "") or "Sheet1"
        _cache_key = f"History::{_ws_title}"
        # ============================================================

        # Get all rows (skip header)
        values = []
        try:
            values = sheets_get_all_values_cached(sheet, cache_key=_cache_key)
        except Exception:
            values = []

        # ============================================================
        # PATCH GH_FALLBACK1 (ADDITIVE): One direct-read retry if cached read is empty
        # Why:
        # - If a prior transient read/429 produced an empty cached value,
        #   evolution may temporarily see no history even though rows exist.
        # ============================================================
        if not values or len(values) < 2:
            try:
                direct = sheet.get_all_values()
                if direct and len(direct) >= 2:
                    values = direct
            except Exception:
                pass
        # ============================================================

        all_rows = values[1:] if values and len(values) >= 2 else []

        # ============================================================
        # PATCH GH_RL1 (ADDITIVE): Rate-limit fallback for History reads
        # ============================================================
        try:
            if (not all_rows) and globals().get("_SHEETS_LAST_READ_ERROR"):
                if ("RESOURCE_EXHAUSTED" in str(_SHEETS_LAST_READ_ERROR)
                    or "Quota exceeded" in str(_SHEETS_LAST_READ_ERROR)
                    or "429" in str(_SHEETS_LAST_READ_ERROR)):
                    return st.session_state.get('analysis_history', [])
        except Exception:
            pass
        # ============================================================

        # Parse and return most recent
        history = []
        for row in all_rows[-limit:]:
            if len(row) >= 5:
                raw_cell = row[4]
                try:
                    data = json.loads(raw_cell)
                    data['_sheet_id'] = row[0]  # Keep track of sheet row ID

                    # (your existing GH2 / ES1G / GH1 / GH3 logic unchanged)
                    # ...
                    history.append(data)

                except json.JSONDecodeError:
                    # (your existing GH1 rescue logic unchanged)
                    continue

        # (your existing GH3 sort unchanged)
        return history

    except Exception as e:
        st.warning(f"⚠️ Failed to load from Google Sheets: {e}")
        return st.session_state.get('analysis_history', [])


def get_analysis_by_id(analysis_id: str) -> Optional[Dict]:
    """Get a specific analysis by ID"""
    sheet = get_google_sheet()
    if not sheet:
        return None

    try:
        # Find row with matching ID
        cell = sheet.find(analysis_id)
        if cell:
            row = sheet.row_values(cell.row)
            if len(row) >= 5:
                return json.loads(row[4])
    except Exception as e:
        st.warning(f"⚠️ Failed to find analysis: {e}")

    return None

def delete_from_history(analysis_id: str) -> bool:
    """Delete an analysis from history"""
    sheet = get_google_sheet()
    if not sheet:
        return False

    try:
        cell = sheet.find(analysis_id)
        if cell:
            sheet.delete_rows(cell.row)
            return True
    except Exception as e:
        st.warning(f"⚠️ Failed to delete: {e}")

    return False

def clear_history() -> bool:
    """Clear all history (keep headers)"""
    sheet = get_google_sheet()
    if not sheet:
        st.session_state.analysis_history = []
        return True

    try:
        # Get row count
        all_rows = sheets_get_all_values_cached(sheet, cache_key="History")
        if len(all_rows) > 1:
            # Delete all rows except header
            sheet.delete_rows(2, len(all_rows))
        return True
    except Exception as e:
        st.warning(f"⚠️ Failed to clear history: {e}")
        return False

def format_history_label(analysis: Dict) -> str:
    """Format a history item for dropdown display"""
    timestamp = analysis.get('timestamp', '')
    question = analysis.get('question', 'Unknown query')[:40]
    confidence = analysis.get('final_confidence', '')

    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now()
        delta = now - dt.replace(tzinfo=None)

        if delta.total_seconds() < 3600:
            time_str = f"{int(delta.total_seconds() / 60)}m ago"
        elif delta.total_seconds() < 86400:
            time_str = f"{int(delta.total_seconds() / 3600)}h ago"
        elif delta.days == 1:
            time_str = "Yesterday"
        elif delta.days < 7:
            time_str = f"{delta.days}d ago"
        else:
            time_str = dt.strftime("%b %d")
    except:
        time_str = timestamp[:10] if timestamp else "Unknown"

    conf_str = f" ({confidence:.0f}%)" if isinstance(confidence, (int, float)) else ""
    return f"{time_str}: {question}...{conf_str}"

def get_history_options() -> List[Tuple[str, int]]:
    """Get formatted history options for dropdown"""
    history = get_history()
    options = []
    for i, analysis in enumerate(reversed(history)):  # Most recent first
        label = format_history_label(analysis)
        actual_index = len(history) - 1 - i
        options.append((label, actual_index))
    return options

# =========================================================
# 1. CONFIGURATION & API KEY VALIDATION
# =========================================================

def load_api_keys():
    """Load and validate API keys from secrets or environment"""
    try:
        PERPLEXITY_KEY = st.secrets.get("PERPLEXITY_API_KEY") or os.getenv("PERPLEXITY_API_KEY", "")
        GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        SERPAPI_KEY = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY", "")
        SCRAPINGDOG_KEY = st.secrets.get("SCRAPINGDOG_KEY") or os.getenv("SCRAPINGDOG_KEY", "")
    except Exception:
        PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY", "")
        GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
        SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
        SCRAPINGDOG_KEY = os.getenv("SCRAPINGDOG_KEY", "")

    # Validate critical keys
    if not PERPLEXITY_KEY or len(PERPLEXITY_KEY) < 10:
        st.error("❌ PERPLEXITY_API_KEY is missing or invalid")
        st.stop()

    if not GEMINI_KEY or len(GEMINI_KEY) < 10:
        st.error("❌ GEMINI_API_KEY is missing or invalid")
        st.stop()

    return PERPLEXITY_KEY, GEMINI_KEY, SERPAPI_KEY, SCRAPINGDOG_KEY

PERPLEXITY_KEY, GEMINI_KEY, SERPAPI_KEY, SCRAPINGDOG_KEY = load_api_keys()
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Configure Gemini
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# =========================================================
# 2. PYDANTIC MODELS
# =========================================================

class MetricDetail(BaseModel):
    """Individual metric with name, value, and unit"""
    name: str = Field(..., description="Metric name")
    value: Union[float, int, str] = Field(..., description="Metric value")
    unit: str = Field(default="", description="Unit of measurement")
    model_config = ConfigDict(extra='ignore')

class TopEntityDetail(BaseModel):
    """Entity in top_entities list"""
    name: str = Field(..., description="Entity name")
    share: Optional[str] = Field(None, description="Market share")
    growth: Optional[str] = Field(None, description="Growth rate")
    model_config = ConfigDict(extra='ignore')

class TrendForecastDetail(BaseModel):
    """Trend forecast item"""
    trend: str = Field(..., description="Trend description")
    direction: Optional[str] = Field(None, description="Direction indicator")
    timeline: Optional[str] = Field(None, description="Timeline")
    model_config = ConfigDict(extra='ignore')

class VisualizationData(BaseModel):
    chart_labels: List[str] = Field(default_factory=list)
    chart_values: List[Union[float, int]] = Field(default_factory=list)
    chart_title: Optional[str] = Field("Trend Analysis")
    chart_type: Optional[str] = Field("line")
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    model_config = ConfigDict(extra='ignore')

class ComparisonBar(BaseModel):
    """Comparison bar chart data"""
    title: str = Field("Comparison", description="Chart title")
    categories: List[str] = Field(default_factory=list)
    values: List[Union[float, int]] = Field(default_factory=list)
    model_config = ConfigDict(extra='ignore')

class BenchmarkTable(BaseModel):
    """Benchmark table row"""
    category: str
    value_1: Union[float, int, str] = Field(default=0, description="Numeric value or string")
    value_2: Union[float, int, str] = Field(default=0, description="Numeric value or string")
    model_config = ConfigDict(extra='ignore')

class Action(BaseModel):
    """Investment/action recommendation"""
    recommendation: str = Field("Neutral", description="Buy/Hold/Sell/Neutral")
    confidence: str = Field("Medium", description="High/Medium/Low")
    rationale: str = Field("", description="Reasoning")
    model_config = ConfigDict(extra='ignore')

class LLMResponse(BaseModel):
    """Complete LLM response schema"""
    executive_summary: str = Field(..., description="High-level summary")
    primary_metrics: Dict[str, MetricDetail] = Field(default_factory=dict)
    key_findings: List[str] = Field(default_factory=list)
    top_entities: List[TopEntityDetail] = Field(default_factory=list)
    trends_forecast: List[TrendForecastDetail] = Field(default_factory=list)
    visualization_data: Optional[VisualizationData] = None
    comparison_bars: Optional[ComparisonBar] = None
    benchmark_table: Optional[List[BenchmarkTable]] = None
    sources: List[str] = Field(default_factory=list)
    confidence: Union[float, int] = Field(default=75)
    freshness: Optional[str] = Field(None)
    action: Optional[Action] = None
    model_config = ConfigDict(extra='ignore')

# =========================================================
# 3. PROMPTS
# =========================================================

RESPONSE_TEMPLATE = """
{
  "executive_summary": "3-4 sentence high-level answer",
  "primary_metrics": {
    "metric_1": {"name": "Key Metric 1", "value": 25.5, "unit": "%"},
    "metric_2": {"name": "Key Metric 2", "value": 623, "unit": "$B"}
  },
  "key_findings": [
    "Finding 1 with quantified impact",
    "Finding 2 explaining drivers"
  ],
  "top_entities": [
    {"name": "Entity 1", "share": "25%", "growth": "15%"}
  ],
  "trends_forecast": [
    {"trend": "Trend description", "direction": "↑", "timeline": "2025-2027"}
  ],
  "visualization_data": {
    "chart_labels": ["2023", "2024", "2025"],
    "chart_values": [100, 120, 145],
    "chart_title": "Market Growth",
    "chart_type": "line"
  },
  "comparison_bars": {
    "title": "Market Share",
    "categories": ["A", "B", "C"],
    "values": [45, 30, 25]
  },
  "benchmark_table": [
    {"category": "Company A", "value_1": 25.5, "value_2": 623}
  ],
  "sources": ["source1.com"],
  "confidence": 87,
  "freshness": "Dec 2024"
}
"""



SYSTEM_PROMPT = f"""You are a professional market research analyst.

CRITICAL RULES:
1. Return ONLY valid JSON. NO markdown, NO code blocks, NO extra text.
2. NO citation references like [1][2] inside strings.
3. Use double quotes for all keys and string values.
4. NO trailing commas in arrays or objects.
5. Escape internal quotes with backslash.
6. If the prompt includes "Query Structure", you MUST follow it:
   - Treat "MAIN QUESTION" as the primary topic and address it FIRST.
   - Treat "SIDE QUESTIONS" as secondary topics and address them AFTER the main topic.
   - Do NOT let a side question replace the main question just because it is more specific.
   - In executive_summary, clearly separate: "Main:" then "Side:" when side questions exist.


NUMERIC FIELD RULES (IMPORTANT):
- In benchmark_table: value_1 and value_2 MUST be numbers (never "N/A", "null", or text)
- If data unavailable, use 0 for benchmark_table values
- In primary_metrics: values can be numbers or strings with units (e.g., "25.5" or "25.5 billion")
- In top_entities: share and growth can be strings (e.g., "25%")

REQUIRED FIELDS (provide substantive data):

**executive_summary** - MUST be 4-6 complete sentences covering:
  • Sentence 1: Direct answer with specific quantitative data (market size, revenue, units, etc.)
  • Sentence 2: Major players or regional breakdown with percentages/numbers
  • Sentence 3: Key growth drivers or market dynamics
  • Sentence 4: Future outlook with projected CAGR, timeline, or target values
  • Sentence 5 (optional): Challenge, risk, or competitive dynamic

  BAD (too short): "The EV market is growing rapidly due to government policies."

  GOOD: "The global electric vehicle market reached 14.2 million units sold in 2023, representing 18% of total auto sales. China dominates with 60% market share, followed by Europe (25%) and North America (10%). Growth is driven by battery cost reductions (down 89% since 2010), expanding charging infrastructure, and stricter emission regulations in over 20 countries. The market is projected to grow at 21% CAGR through 2030, reaching 40 million units annually. However, supply chain constraints for lithium and cobalt remain key challenges."

- primary_metrics (3+ metrics with numbers)
- key_findings (3+ findings with quantitative details)
- top_entities (3+ companies/countries with market share %)
- trends_forecast (2+ trends with timelines)
- visualization_data (MUST have chart_labels and chart_values)
- benchmark_table (if included, value_1 and value_2 must be NUMBERS, not "N/A")

Even if web data is sparse, use your knowledge to provide complete, detailed analysis.

Output ONLY this JSON structure:
{RESPONSE_TEMPLATE}
"""

EVOLUTION_PROMPT_TEMPLATE = """You are a market research analyst performing an UPDATE ANALYSIS.

You have been given a PREVIOUS ANALYSIS from {time_ago}. Your task is to:
1. Search for CURRENT data on the same metrics and entities
2. Identify what has CHANGED vs what has STAYED THE SAME
3. Provide updated values where data has changed
4. Flag any metrics/entities that are no longer relevant or have new entries

PREVIOUS ANALYSIS:
==================
Question: {previous_question}
Timestamp: {previous_timestamp}

Previous Executive Summary:
{previous_summary}

Previous Key Metrics:
{previous_metrics}

Previous Top Entities:
{previous_entities}

Previous Key Findings:
{previous_findings}
==================

CRITICAL RULES:
1. Return ONLY valid JSON. NO markdown, NO code blocks.
2. For EACH metric, indicate if it INCREASED, DECREASED, or stayed UNCHANGED
3. Keep the SAME metric names as previous analysis for easy comparison
4. If a metric is no longer available, mark it as "discontinued"
5. If there's a NEW important metric, add it with status "new"


REQUIRED OUTPUT FORMAT:
{{
  "executive_summary": "Updated 4-6 sentence summary noting key changes since last analysis",
  "analysis_delta": {{
    "time_since_previous": "{time_ago}",
    "overall_trend": "improving/declining/stable",
    "major_changes": ["Change 1", "Change 2"],
    "data_freshness": "Q4 2024"
  }},
  "primary_metrics": {{
    "metric_key": {{
      "name": "Same metric name as before",
      "previous_value": 100,
      "current_value": 110,
      "unit": "$B",
      "change_pct": 10.0,
      "direction": "increased/decreased/unchanged",
      "status": "updated/discontinued/new"
    }}
  }},
  "key_findings": [
    "[UNCHANGED] Finding that remains true",
    "[UPDATED] Finding with new data",
    "[NEW] Completely new finding",
    "[REMOVED] Finding no longer relevant - reason"
  ],
  "top_entities": [
    {{
      "name": "Company A",
      "previous_share": "25%",
      "current_share": "27%",
      "previous_rank": 1,
      "current_rank": 1,
      "change": "increased",
      "status": "updated"
    }}
  ],
  "trends_forecast": [
    {{"trend": "Trend description", "direction": "↑", "timeline": "2025-2027", "confidence": "high/medium/low"}}
  ],
  "visualization_data": {{
    "chart_labels": ["Previous", "Current"],
    "chart_values": [100, 110],
    "chart_title": "Market Size Evolution"
  }},
  "sources": ["source1.com", "source2.com"],
  "confidence": 85,
  "freshness": "Dec 2024",
  "drift_summary": {{
    "metrics_changed": 2,
    "metrics_unchanged": 3,
    "entities_reshuffled": 1,
    "findings_updated": 4,
    "overall_stability_pct": 75
  }}
}}

NOW, search for CURRENT information to UPDATE the previous analysis.
Focus on finding CHANGES to the metrics and entities listed above.

User Question: {query}
"""

# =========================================================
# 4. MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner="🔧 Loading AI models...")
def load_models():
    """Load and cache sentence transformer and classifier"""
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return classifier, embedder
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

domain_classifier, embedder = load_models()

# =========================================================
# 5. JSON REPAIR FUNCTIONS
# =========================================================

def repair_llm_response(data: dict) -> dict:
    """
    Repair common LLM JSON structure issues:

    - Convert primary_metrics from list -> dict (stable keys)
    - Normalize MetricDetail fields so currency+unit do NOT get lost:
        "29.8 S$B" / "S$29.8B" / "S$29.8 billion" -> value=29.8, unit="S$B"
        "$204.7B" -> value=204.7, unit="$B"
        "9.8%" -> value=9.8, unit="%"
    - Ensure top_entities and trends_forecast are lists
    - Fix visualization_data legacy keys (labels/values)
    - Fix benchmark_table numeric values
    - Remove 'action' block entirely (no longer used)
    - Add minimal required fields if missing

    NOTE: This function is intentionally conservative: it normalizes obvious formatting
    without trying to "invent" missing values.
    """
    if not isinstance(data, dict):
        return {}

    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            return list(x.values())
        return []

    def _coerce_number(s: str):
        try:
            return float(str(s).replace(",", "").strip())
        except Exception:
            return None

    def _normalize_metric_item(item: dict) -> dict:
        """
        Normalize a single metric dict in-place-ish and return it.

        Goal: preserve currency + magnitude in `unit`, keep `value` numeric when possible.
        """
        if not isinstance(item, dict):
            return {"name": "N/A", "value": "N/A", "unit": ""}

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            name = "N/A"
        item["name"] = name

        raw_val = item.get("value")
        raw_unit = item.get("unit")

        unit = (raw_unit or "")
        if not isinstance(unit, str):
            unit = str(unit)

        # If already numeric and unit looks okay, keep as-is
        if isinstance(raw_val, (int, float)) and isinstance(unit, str):
            item["unit"] = unit.strip()
            return item

        # Try to parse string value forms like:
        # "S$29.8B", "29.8 S$B", "$ 204.7 billion", "9.8%", "12 percent"
        if isinstance(raw_val, str):
            txt = raw_val.strip()

            # Also allow unit to carry the number sometimes (rare but happens)
            # e.g. value="29.8", unit="S$B" is already fine.
            # But if unit is empty and txt contains unit, we extract.
            # Percent detection
            if re.search(r'(%|\bpercent\b)', txt, flags=re.I):
                num = _coerce_number(re.sub(r'[^0-9\.\-\,]+', '', txt))
                if num is not None:
                    item["value"] = num
                    item["unit"] = "%"
                    return item

            # Currency detection
            currency = ""
            # Normalize currency tokens in either value or unit
            combo = f"{txt} {unit}".strip()

            if re.search(r'\bSGD\b', combo, flags=re.I) or "S$" in combo.upper():
                currency = "S$"
            elif re.search(r'\bUSD\b', combo, flags=re.I) or "$" in combo:
                currency = "$"

            # Magnitude detection
            # Accept: T/B/M/K, or words
            mag = ""
            if re.search(r'\btrillion\b', combo, flags=re.I):
                mag = "T"
            elif re.search(r'\bbillion\b', combo, flags=re.I):
                mag = "B"
            elif re.search(r'\bmillion\b', combo, flags=re.I):
                mag = "M"
            elif re.search(r'\bthousand\b', combo, flags=re.I):
                mag = "K"
            else:
                m = re.search(r'([TBMK])\b', combo.replace(" ", ""), flags=re.I)
                if m:
                    mag = m.group(1).upper()

            # Extract numeric
            num = _coerce_number(re.sub(r'[^0-9\.\-\,]+', '', txt))
            if num is not None:
                # If unit was present and meaningful (and already includes %), keep it
                if unit.strip() == "%":
                    item["value"] = num
                    item["unit"] = "%"
                    return item

                # Build unit as currency+magnitude when any found
                # If neither found, keep existing unit (may be e.g. "years", "points")
                if currency or mag:
                    item["value"] = num
                    item["unit"] = f"{currency}{mag}".strip()
                    return item

                # No currency/mag detected: keep unit if provided; else blank
                item["value"] = num
                item["unit"] = unit.strip()
                return item

            # If we can’t parse into a number, at least preserve the original text
            item["value"] = txt
            item["unit"] = unit.strip()
            return item

        # Non-string, non-numeric (None, dict, list, etc.)
        if raw_val is None or raw_val == "":
            item["value"] = "N/A"
        else:
            item["value"] = str(raw_val)

        item["unit"] = unit.strip()
        return item

    # -------------------------
    # primary_metrics normalization
    # -------------------------
    metrics = data.get("primary_metrics")

    # list -> dict
    if isinstance(metrics, list):
        new_metrics = {}
        for i, item in enumerate(metrics):
            if not isinstance(item, dict):
                continue
            item = _normalize_metric_item(item)

            raw_name = item.get("name", f"metric_{i+1}")
            key = re.sub(r'[^a-z0-9_]', '', str(raw_name).lower().replace(" ", "_")).strip("_")
            if not key:
                key = f"metric_{i+1}"

            original_key = key
            j = 1
            while key in new_metrics:
                key = f"{original_key}_{j}"
                j += 1

            new_metrics[key] = item

        data["primary_metrics"] = new_metrics

    elif isinstance(metrics, dict):
        # Normalize each metric dict entry
        cleaned = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                cleaned[str(k)] = _normalize_metric_item(v)
            else:
                # If someone stored a scalar, wrap it
                cleaned[str(k)] = _normalize_metric_item({"name": str(k), "value": v, "unit": ""})
        data["primary_metrics"] = cleaned

    else:
        data["primary_metrics"] = {}

    # -------------------------
    # list-like fields
    # -------------------------
    data["top_entities"] = _to_list(data.get("top_entities"))
    data["trends_forecast"] = _to_list(data.get("trends_forecast"))
    data["key_findings"] = _to_list(data.get("key_findings"))

    # Ensure strings in key_findings
    data["key_findings"] = [str(x) for x in data["key_findings"] if x is not None and str(x).strip()]

    # -------------------------
    # visualization_data legacy keys
    # -------------------------
    if isinstance(data.get("visualization_data"), dict):
        viz = data["visualization_data"]
        if "labels" in viz and "chart_labels" not in viz:
            viz["chart_labels"] = viz.pop("labels")
        if "values" in viz and "chart_values" not in viz:
            viz["chart_values"] = viz.pop("values")

        # Coerce chart_labels/values types gently
        if "chart_labels" in viz and not isinstance(viz["chart_labels"], list):
            viz["chart_labels"] = [str(viz["chart_labels"])]
        if "chart_values" in viz and not isinstance(viz["chart_values"], list):
            viz["chart_values"] = [viz["chart_values"]]

    # -------------------------
    # benchmark_table numeric cleaning
    # -------------------------
    if isinstance(data.get("benchmark_table"), list):
        cleaned_table = []
        for row in data["benchmark_table"]:
            if not isinstance(row, dict):
                continue

            if "category" not in row:
                row["category"] = "Unknown"

            for key in ["value_1", "value_2"]:
                if key not in row:
                    row[key] = 0
                    continue

                val = row.get(key)
                if isinstance(val, str):
                    val_upper = val.upper().strip()
                    if val_upper in ["N/A", "NA", "NULL", "NONE", "", "-", "—"]:
                        row[key] = 0
                    else:
                        try:
                            cleaned = re.sub(r'[^\d.-]', '', val)
                            row[key] = float(cleaned) if '.' in cleaned else int(cleaned) if cleaned else 0
                        except Exception:
                            row[key] = 0
                elif isinstance(val, (int, float)):
                    pass
                else:
                    row[key] = 0

            cleaned_table.append(row)

        data["benchmark_table"] = cleaned_table

    # -------------------------
    # Remove action block entirely
    # -------------------------
    data.pop("action", None)

    # -------------------------
    # Minimal required top-level fields
    # -------------------------
    if not isinstance(data.get("executive_summary"), str) or not data.get("executive_summary", "").strip():
        data["executive_summary"] = "No executive summary provided."

    if not isinstance(data.get("sources"), list):
        data["sources"] = []

    if "confidence" not in data:
        data["confidence"] = 60

    if not isinstance(data.get("freshness"), str) or not data.get("freshness", "").strip():
        data["freshness"] = "Current"

    return data


def validate_numeric_fields(data: dict, context: str = "LLM Response") -> None:
    """
    Guardrail logger (and gentle coercer) for numeric lists used in charts/tables.

    We keep this lightweight: warn when strings appear where numbers are expected,
    and attempt to coerce when safe.
    """
    if not isinstance(data, dict):
        return

    # Check benchmark_table
    if "benchmark_table" in data and isinstance(data["benchmark_table"], list):
        for i, row in enumerate(data["benchmark_table"]):
            if isinstance(row, dict):
                for key in ["value_1", "value_2"]:
                    val = row.get(key)
                    if isinstance(val, str):
                        st.warning(
                            f"⚠️ {context}: benchmark_table[{i}].{key} is string: '{val}' (coercing to 0 if invalid)"
                        )
                        try:
                            cleaned = re.sub(r"[^\d\.\-]", "", val)
                            row[key] = float(cleaned) if cleaned else 0
                        except Exception:
                            row[key] = 0

    # Check visualization_data chart_values
    viz = data.get("visualization_data")
    if isinstance(viz, dict):
        vals = viz.get("chart_values")
        if isinstance(vals, list):
            new_vals = []
            for j, v in enumerate(vals):
                if isinstance(v, (int, float)):
                    new_vals.append(v)
                elif isinstance(v, str):
                    try:
                        cleaned = re.sub(r"[^\d\.\-]", "", v)
                        new_vals.append(float(cleaned) if cleaned else 0.0)
                        st.warning(f"⚠️ {context}: visualization_data.chart_values[{j}] is string: '{v}' (coerced)")
                    except Exception:
                        new_vals.append(0.0)
                else:
                    new_vals.append(0.0)
            viz["chart_values"] = new_vals


def preclean_json(raw: str) -> str:
    """
    Remove markdown fences and common citation markers before JSON parsing.
    Conservative: tries not to destroy legitimate JSON content.
    """
    if not raw or not isinstance(raw, str):
        return ""

    text = raw.strip()

    # Remove leading/trailing code fences (```json ... ```)
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)

    text = text.strip()

    # Remove common citation formats the model may append
    # [web:1], [1], (1) etc. (but avoid killing array syntax by being specific)
    text = re.sub(r'\[web:\d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!")\[\d+\](?!")', '', text)   # not inside quotes
    text = re.sub(r'(?<!")\(\d+\)(?!")', '', text)   # not inside quotes

    return text.strip()


def parse_json_safely(json_str: str, context: str = "LLM") -> dict:
    """
    Parse JSON with aggressive error recovery:
    1) Pre-clean markdown/citations
    2) Extract the *first* JSON object
    3) Repair common issues (unquoted keys, trailing commas, True/False/Null)
    4) Try parsing; if it fails, attempt a small set of pragmatic fixes
    """
    if json_str is None:
        return {}
    if not isinstance(json_str, str):
        json_str = str(json_str)

    if not json_str.strip():
        return {}

    cleaned = preclean_json(json_str)

    # Extract first JSON object (most LLM outputs are one object)
    match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if not match:
        st.warning(f"⚠️ No JSON object found in {context} response")
        return {}

    json_content = match.group(0)

    # Structural repairs
    try:
        # Fix unquoted keys: {key: -> {"key":
        json_content = re.sub(
            r'([\{\,]\s*)([a-zA-Z_][a-zA-Z0-9_\-]*)(\s*):',
            r'\1"\2"\3:',
            json_content
        )

        # Remove trailing commas
        json_content = re.sub(r',\s*([\]\}])', r'\1', json_content)

        # Fix boolean/null capitalization
        json_content = re.sub(r':\s*True\b', ': true', json_content)
        json_content = re.sub(r':\s*False\b', ': false', json_content)
        json_content = re.sub(r':\s*Null\b', ': null', json_content)

    except Exception as e:
        st.warning(f"⚠️ {context}: Regex repair failed: {e}")

    # Attempt parse with a few passes
    attempts = 0
    last_err = None

    while attempts < 6:
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            last_err = e
            msg = (e.msg or "").lower()

            # Pass 1: replace smart quotes
            if attempts == 0:
                json_content = (
                    json_content.replace("“", '"')
                                .replace("”", '"')
                                .replace("’", "'")
                )

            # Pass 2: single-quote keys/strings -> double quotes (limited)
            elif attempts == 1:
                # Only do this if it looks like single quotes dominate
                if json_content.count("'") > json_content.count('"'):
                    json_content = re.sub(r"\'", '"', json_content)

            # Pass 3: try removing control characters
            elif attempts == 2:
                json_content = re.sub(r"[\x00-\x1F\x7F]", "", json_content)

            # Pass 4: if unterminated string, try escaping a quote near the error
            elif "unterminated string" in msg or "unterminated" in msg:
                pos = e.pos
                # Try escaping a quote a bit before pos
                for i in range(pos - 1, max(0, pos - 200), -1):
                    if i < len(json_content) and json_content[i] == '"':
                        if i == 0 or json_content[i - 1] != "\\":
                            json_content = json_content[:i] + '\\"' + json_content[i + 1:]
                            break

            # Pass 5+: give up
            attempts += 1
            continue

    st.error(f"❌ Failed to parse JSON from {context}: {str(last_err)[:180] if last_err else 'unknown error'}")
    return {}




def parse_query_structure_safe(json_str: str, user_question: str) -> Dict:
    """
    Parse LLM-derived query structure with guaranteed deterministic fallback.
    Never raises, never returns empty dict.
    """
    parsed = parse_json_safely(json_str, context="LLM Query Structure")

    if isinstance(parsed, dict) and parsed:
        # Minimal schema validation
        if "main" in parsed or "category" in parsed:
            return parsed

    # 🔒 Deterministic fallback (NO LLM)
    return {
        "category": "unknown",
        "category_confidence": 0.0,
        "main": user_question,
        "side": []
    }


def extract_json_object(text: str) -> Optional[Dict]:
    """
    Best-effort extraction of the first JSON object from a string.
    Returns dict or None.
    """
    if not text or not isinstance(text, str):
        return None

    # Common cleanup
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Fast path
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Regex: first {...} block (non-greedy)
    try:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None
        candidate = m.group(0)
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


# =========================================================
# 6. WEB SEARCH FUNCTIONS
#   SERPAPI STABILITY CONFIGURATION
# =========================================================

# Fixed parameters to prevent geo/personalization variance

SERPAPI_STABILITY_CONFIG = {
    "gl": "us",                    # Fixed country
    "hl": "en",                    # Fixed language
    "google_domain": "google.com", # Fixed domain
    "nfpr": "1",                   # No auto-query correction
    "safe": "active",              # Consistent safe search
    "device": "desktop",           # Fixed device type
    "no_cache": "false",           # Allow Google caching (more stable)
}

# Preferred domains for consistent sourcing (sorted by priority)
PREFERRED_SOURCE_DOMAINS = [
    "statista.com", "reuters.com", "bloomberg.com", "imf.org", "wsj.com", "bcg.com", "opec.org",
    "worldbank.org", "mckinsey.com", "deloitte.com", "spglobal.com", "ft.com", "pwc.com", "semiconductors.org",
    "ft.com", "economist.com", "wsj.com", "forbes.com", "cnbc.com", "kpmg.com", "eia.org"
]

# Search results cache
_search_cache: Dict[str, Tuple[List[Dict], datetime]] = {}
SEARCH_CACHE_TTL_HOURS = 24

def get_search_cache_key(query: str) -> str:
    """Generate stable cache key for search query"""
    normalized = re.sub(r'\s+', ' ', query.lower().strip())
    normalized = re.sub(r'\b(today|current|latest|now|recent)\b', '', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()[:16]

def get_cached_search_results(query: str) -> Optional[List[Dict]]:
    """
    Get cached search results if still valid.

    IMPORTANT:
    - Never treat cached empty results as valid.
      Returning [] here "poisons" the pipeline for hours and makes SerpAPI look broken.
    """
    try:
        cache_key = get_search_cache_key(query)
        if cache_key in _search_cache:
            cached_results, cached_time = _search_cache[cache_key]
            if datetime.now() - cached_time < timedelta(hours=SEARCH_CACHE_TTL_HOURS):
                # ✅ Do not reuse empty cache entries
                if isinstance(cached_results, list) and len(cached_results) == 0:
                    return None
                return cached_results
            # expired
            del _search_cache[cache_key]
    except Exception:
        return None
    return None


def cache_search_results(query: str, results: List[Dict]):
    """
    Cache search results.

    IMPORTANT:
    - Do NOT cache empty lists
    - Do NOT cache lists that contain no usable URLs
      (prevents "poisoned cache" that makes SerpAPI appear broken)
    """
    try:
        if not isinstance(query, str) or not query.strip():
            return
        if not isinstance(results, list) or not results:
            return

        # Require at least one usable url/link
        has_url = False
        for r in results:
            if isinstance(r, dict):
                u = (r.get("link") or r.get("url") or "").strip()
                if u:
                    has_url = True
                    break
            elif isinstance(r, str) and r.strip():
                has_url = True
                break

        if not has_url:
            return

        cache_key = get_search_cache_key(query)
        _search_cache[cache_key] = (results, datetime.now())
    except Exception:
        return


# =========================================================
# LLM RESPONSE CACHE - Prevents variance on identical inputs
# =========================================================
_llm_cache: Dict[str, Tuple[str, datetime]] = {}
LLM_CACHE_TTL_HOURS = 24  # Cache LLM responses for 24 hours

def get_llm_cache_key(query: str, web_context: Dict) -> str:
    """Generate cache key from query + source URLs"""
    # Include source URLs so cache invalidates if sources change
    source_urls = sorted(web_context.get("sources", [])[:5])
    cache_input = f"{query.lower().strip()}|{'|'.join(source_urls)}"
    return hashlib.md5(cache_input.encode()).hexdigest()[:20]

def get_cached_llm_response(query: str, web_context: Dict) -> Optional[str]:
    """Get cached LLM response if still valid"""
    cache_key = get_llm_cache_key(query, web_context)
    if cache_key in _llm_cache:
        cached_response, cached_time = _llm_cache[cache_key]
        if datetime.now() - cached_time < timedelta(hours=LLM_CACHE_TTL_HOURS):
            return cached_response
        del _llm_cache[cache_key]
    return None

def cache_llm_response(query: str, web_context: Dict, response: str):
    """Cache LLM response"""
    cache_key = get_llm_cache_key(query, web_context)
    _llm_cache[cache_key] = (response, datetime.now())


def sort_results_deterministically(results: List[Dict]) -> List[Dict]:
    """Sort results for consistent ordering"""
    def sort_key(r):
        link = r.get("link", "").lower()
        # Priority: preferred domains first, then alphabetical
        priority = 999
        for i, domain in enumerate(PREFERRED_SOURCE_DOMAINS):
            if domain in link:
                priority = i
                break
        return (priority, link)
    return sorted(results, key=sort_key)


def classify_source_reliability(source: str) -> str:
    """Classify source as High/Medium/Low quality"""
    source = source.lower() if isinstance(source, str) else ""

    high = ["gov", "imf", "worldbank", "central bank", "fed", "ecb", "reuters", "spglobal", "economist", "mckinsey", "bcg", "cognitive market research",
            "financial times", "wsj", "oecd", "bloomberg", "tradingeconomics", "deloitte", "hsbc", "imarc", "booz allen", "bakerinstitute.org", "wef",
           "kpmg", "semiconductors.org", "eu", "iea", "world bank", "opec", "jpmorgan", "citibank", "goldmansachs", "j.p. morgan", "oecd",
           "world bank", "sec", "federalreserve", "bls", "bea"]
    medium = ["wikipedia", "forbes", "cnbc", "yahoo", "ceic", "statista", "trendforce", "digitimes", "idc", "gartner", "marketwatch", "fortune", "investopedia"]
    low = ["blog", "medium.com", "wordpress", "ad", "promo"]

    for h in high:
        if h in source:
            return "✅ High"
    for m in medium:
        if m in source:
            return "⚠️ Medium"
    for l in low:
        if l in source:
            return "❌ Low"

    return "⚠️ Medium"

def source_quality_score(sources: List[str]) -> float:
    """Calculate average source quality (0-100)"""
    if not sources:
        return 50.0  # Lower default when no sources

    weights = {"✅ High": 100, "⚠️ Medium": 60, "❌ Low": 30}
    scores = [weights.get(classify_source_reliability(s), 60) for s in sources]
    return sum(scores) / len(scores) if scores else 50.0

@st.cache_data(ttl=3600, show_spinner=False)
def search_serpapi(query: str, num_results: int = 10) -> List[Dict]:
    """Search Google via SerpAPI with stability controls"""
    if not SERPAPI_KEY:
        return []

    # Check cache first (this is the ONLY cache we use - removed @st.cache_data to avoid conflicts)
    cached = get_cached_search_results(query)
    if cached:
        st.info("📦 Using cached search results")
        return cached

    # Aggressive query normalization for consistent searches
    query_normalized = query.lower().strip()

    # Remove temporal words that cause variance
    query_normalized = re.sub(r'\b(latest|current|today|now|recent|new|upcoming|this year|this month)\b', '', query_normalized)

    # Normalize whitespace
    query_normalized = re.sub(r'\s+', ' ', query_normalized).strip()

    # Add year for consistency
    if not re.search(r'\b20\d{2}\b', query_normalized):
        query_normalized = f"{query_normalized} 2024"

    # Build search terms
    query_lower = query_normalized
    industry_kw = ["industry", "market", "sector", "size", "growth", "players"]

    if any(kw in query_lower for kw in industry_kw):
        search_terms = f"{query_normalized} market size growth statistics"
        tbm, tbs = "", ""  # Organic results (more stable than news)
    else:
        search_terms = f"{query_normalized} finance economics data"
        tbm, tbs = "", ""  # Use organic for stability

    params = {
        "engine": "google",
        "q": search_terms,
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "tbm": tbm,
        "tbs": tbs,
        **SERPAPI_STABILITY_CONFIG  # Add fixed location params
    }

    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []

        # Prefer organic results (more stable than news)
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "source": item.get("source", "")
            })

        # Fall back to news only if no organic results
        if not results:
            for item in data.get("news_results", [])[:num_results]:
                src = item.get("source", {})
                source_name = src.get("name", "") if isinstance(src, dict) else str(src)
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date": item.get("date", ""),
                    "source": source_name
                })

        # Sort deterministically
        results = sort_results_deterministically(results)
        results = results[:num_results]

        # Cache results
        if results:
            cache_search_results(query, results)

        return results

    except Exception as e:
        st.warning(f"⚠️ SerpAPI error: {e}")
        return []


def scrape_url(url: str) -> Optional[str]:
    """
    Scrape webpage content.

    Priority:
      1) ScrapingDog (if SCRAPINGDOG_KEY is present)
      2) Safe fallback: direct requests + BeautifulSoup visible-text extraction

    Returns:
      - Clean visible text (<= 3000 chars) or None
    """
    import re

    url_s = (url or "").strip()
    if not url_s:
        return None

    def _clean_html_to_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(html or "", "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
                try:
                    tag.decompose()
                except Exception:
                    pass
            txt = soup.get_text(separator="\n")
        except Exception:
            # fallback: strip tags
            txt = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html or "")
            txt = re.sub(r"(?is)<[^>]+>", " ", txt)
        # normalize whitespace
        lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
        out = "\n".join(lines)
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    def _direct_fetch(u: str) -> Optional[str]:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
            resp = requests.get(u, headers=headers, timeout=12, allow_redirects=True)
            if resp.status_code >= 400:
                return None

            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/pdf" in ctype:
                return None

            cleaned = _clean_html_to_text(resp.text or "")
            cleaned = cleaned.strip()
            if not cleaned:
                return None
            return cleaned[:3000]
        except Exception:
            return None

    # 1) ScrapingDog path (if configured)
    if globals().get("SCRAPINGDOG_KEY"):
        try:
            params = {"api_key": SCRAPINGDOG_KEY, "url": url_s, "dynamic": "false"}
            resp = requests.get("https://api.scrapingdog.com/scrape", params=params, timeout=15)
            if resp.status_code < 400:
                cleaned = _clean_html_to_text(resp.text or "").strip()
                if cleaned:
                    return cleaned[:3000]
        except Exception:
            pass  # fall through to direct fetch

    # 2) Safe fallback
    return _direct_fetch(url_s)


def fetch_web_context(
    query: str,
    num_sources: int = 3,
    *,
    fallback_mode: bool = False,
    fallback_urls: list = None,
    existing_snapshots: Any = None,   # <-- ADDITIVE
) -> dict:

    """
    Web context collector used by BOTH analysis + evolution.

    Enhancements:
    - Dashboard telemetry (sources found / HQ / admitted / scraped / success)
    - Keeps snapshot-friendly scraped_meta (fingerprint + extracted_numbers + numbers_found)
    - Uses scrape_url() which now has ScrapingDog + safe fallback scraper
    - Restores legacy contract: web_context["sources"] AND ["web_sources"]
    """
    import re
    from datetime import datetime, timezone

    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _is_probably_url(s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        t = s.strip()
        if " " in t:
            return False
        if re.match(r"^https?://", t, flags=re.I):
            return True
        if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}(/.*)?$", t, flags=re.I):
            return True
        return False

    def _normalize_url(s: str) -> str:
        t = (s or "").strip()
        if not t:
            return ""
        if re.match(r"^https?://", t, flags=re.I):
            return t
        if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}(/.*)?$", t, flags=re.I):
            return "https://" + t
        return ""


    out = {
        "query": query,
        "sources": [],        # ✅ legacy key many downstream blocks expect
        "web_sources": [],    # ✅ newer key used by evolution/snapshots
        "search_results": [],
        "scraped_meta": {},
        "scraped_content": {},
        "errors": [],
        "status": "ok",
        "status_detail": "",
        "fetched_at": _now_iso(),
        "debug_counts": {},   # ✅ telemetry for dashboard + JSON debugging
    }

    # ---- ADDITIVE: snapshot reuse lookup (Change #3) ----
    snap_lookup = {}
    if isinstance(existing_snapshots, dict):
        snap_lookup = existing_snapshots
    elif isinstance(existing_snapshots, list):
        for s in existing_snapshots:
            if isinstance(s, dict) and s.get("url"):
                snap_lookup[str(s.get("url")).strip()] = s

    extractor_fp = get_extractor_fingerprint()
    # ----------------------------------------------------


    q = (query or "").strip()
    if not q:
        out["status"] = "no_query"
        out["status_detail"] = "empty_query"
        return out

    # -----------------------------
    # 1) Search (SerpAPI) OR fallback_urls
    # -----------------------------
    search_results = []
    urls_raw = []

    if not fallback_mode:
        try:
            sr = search_serpapi(q, num_results=10) or []
            if isinstance(sr, list):
                search_results = sr
        except Exception as e:
            out["errors"].append(f"search_failed:{type(e).__name__}")
            search_results = []

        out["search_results"] = search_results

        # Extract urls from results
        for r in (search_results or []):
            if isinstance(r, dict):
                u = (r.get("link") or r.get("url") or "").strip()
                if _is_probably_url(u):
                    urls_raw.append(u)
            elif isinstance(r, str):
                if _is_probably_url(r):
                    urls_raw.append(r.strip())

    else:
        # Evolution fallback: use provided URLs
        if isinstance(fallback_urls, list):
            for u in fallback_urls:
                if isinstance(u, str) and _is_probably_url(u.strip()):
                    urls_raw.append(u.strip())

    # -----------------------------
    # 2) Compute "HQ" counts (like old version)
    # -----------------------------
    total_found = len(search_results) if not fallback_mode else len(urls_raw)
    hq_count = 0

    try:
        fn_rel = globals().get("classify_source_reliability")
        if callable(fn_rel) and not fallback_mode:
            for r in (search_results or []):
                if not isinstance(r, dict):
                    continue
                u = (r.get("link") or "").strip()
                if not u:
                    continue
                label = fn_rel(u) or ""
                if "✅" in str(label):
                    hq_count += 1
    except Exception:
        hq_count = 0

    # -----------------------------
    # 3) Sanitize + normalize + dedupe
    # -----------------------------
    normed = []
    seen = set()
    for u in (urls_raw or []):
        nu = _normalize_url(u)
        if not nu:
            continue
        if nu in seen:
            continue
        seen.add(nu)
        normed.append(nu)

    # admitted for scraping (top N)
    try:
        n = int(num_sources or 3)
    except Exception:
        n = 3
    n = max(1, min(12, n))
    admitted = normed[:n] if not fallback_mode else normed  # fallback_mode typically wants all

    out["sources"] = admitted
    out["web_sources"] = admitted

    # Telemetry before scrape
    out["debug_counts"].update({
        "total_found": int(total_found),
        "high_quality": int(hq_count),
        "admitted_for_scraping": int(len(admitted)),
        "fallback_mode": bool(fallback_mode),
    })

    # Dashboard info (restored)
    try:
        if not fallback_mode:
            st.info(
                f"🔍 Sources Found: **{out['debug_counts']['total_found']} total** | "
                f"**{out['debug_counts']['high_quality']} high-quality** | "
                f"Scraping **{out['debug_counts']['admitted_for_scraping']}**"
            )
        else:
            st.info(
                f"🧩 Fallback Sources: **{out['debug_counts']['admitted_for_scraping']}** (no SerpAPI search)"
            )
    except Exception:
        pass

    if not admitted:
        out["status"] = "no_sources"
        out["status_detail"] = "empty_sources_after_filter"
        return out

    # -----------------------------
    # 4) Scrape + extract numbers (snapshot-friendly scraped_meta)
    # -----------------------------
    fn_fp = globals().get("fingerprint_text")
    fn_extract = globals().get("extract_numbers_with_context") or globals().get("extract_numeric_candidates") or globals().get("extract_numbers_from_text")

    scraped_attempted = 0
    scraped_ok_text = 0
    scraped_ok_numbers = 0
    scraped_failed = 0

    # optional progress bar
    progress = None
    try:
        progress = st.progress(0)
    except Exception:
        progress = None

    for i, url in enumerate(admitted):
        scraped_attempted += 1

        meta = {
            "url": url,
            "fetched_at": _now_iso(),
            "status": "failed",
            "status_detail": "",
            "content_type": "",
            "content_len": 0,
            "clean_text_len": 0,
            "fingerprint": None,
            "numbers_found": 0,
            "extracted_numbers": [],
            "content": "",
            "clean_text": "",
        }

        try:
            text = scrape_url(url)  # ✅ ScrapingDog + fallback inside scrape_url
            if not text or not str(text).strip():
                meta["status"] = "failed"
                meta["status_detail"] = "failed:no_text"
                scraped_failed += 1
                out["scraped_meta"][url] = meta
            else:
                cleaned = str(text).strip()
                meta["status"] = "fetched"
                meta["status_detail"] = "success"
                meta["content"] = cleaned
                meta["clean_text"] = cleaned
                meta["content_len"] = len(cleaned)
                meta["clean_text_len"] = len(cleaned)

                # fingerprint
                try:
                    if callable(fn_fp):
                        meta["fingerprint"] = fn_fp(cleaned)
                    else:
                        meta["fingerprint"] = fingerprint_text(cleaned) if callable(globals().get("fingerprint_text")) else None
                except Exception:
                    meta["fingerprint"] = None

                # ---- ADDITIVE: reuse extracted_numbers when unchanged (Change #3) ----
                meta["extractor_fingerprint"] = extractor_fp
                prev = snap_lookup.get(url) if isinstance(snap_lookup, dict) else None
                if isinstance(prev, dict):
                    if prev.get("fingerprint") == meta.get("fingerprint") and prev.get("extractor_fingerprint") == extractor_fp:
                        prev_nums = prev.get("extracted_numbers")
                        if isinstance(prev_nums, list) and prev_nums:
                            meta["extracted_numbers"] = prev_nums
                            meta["numbers_found"] = len(prev_nums)
                            meta["reused_snapshot"] = True

                            out["scraped_meta"][url] = meta
                            out["scraped_content"][url] = cleaned

                            scraped_ok_text += 1
                            if meta["numbers_found"] > 0:
                                scraped_ok_numbers += 1

                            if progress:
                                try:
                                    progress.progress((i + 1) / max(1, len(admitted)))
                                except Exception:
                                    pass

                            continue
                meta["reused_snapshot"] = False
                # ---------------------------------------------------------------

                # numeric extraction (analysis-aligned if fn exists)
                nums = []
                try:
                    if callable(fn_extract):
                        nums = fn_extract(cleaned, url=url) if "url" in fn_extract.__code__.co_varnames else fn_extract(cleaned)
                except Exception:
                    nums = []

                if isinstance(nums, list):
                    meta["extracted_numbers"] = nums
                    meta["numbers_found"] = len(nums)

                    # ---- ADDITIVE: stable IDs + ordering (Change #2 / Part 1) ----
                    urlv = meta.get("url") or url
                    fpv = meta.get("fingerprint") or ""

                    for n in (meta["extracted_numbers"] or []):
                        if isinstance(n, dict):
                            if "extracted_number_id" not in n:
                                n["extracted_number_id"] = make_extracted_number_id(urlv, fpv, n)
                            if not n.get("source_url"):
                                n["source_url"] = urlv

                    meta["extracted_numbers"] = sort_snapshot_numbers(meta["extracted_numbers"])
                    meta["numbers_found"] = len(meta["extracted_numbers"])
                    # --------------------------------------------------------------

                out["scraped_meta"][url] = meta
                out["scraped_content"][url] = cleaned

                scraped_ok_text += 1
                if meta["numbers_found"] > 0:
                    scraped_ok_numbers += 1

        except Exception as e:
            meta["status"] = "failed"
            meta["status_detail"] = f"failed:exception:{type(e).__name__}"
            scraped_failed += 1
            out["scraped_meta"][url] = meta
            out["errors"].append(meta["status_detail"])

        if progress:
            try:
                progress.progress((i + 1) / max(1, len(admitted)))
            except Exception:
                pass

    out["debug_counts"].update({
        "scraped_attempted": int(scraped_attempted),
        "scraped_ok_text": int(scraped_ok_text),
        "scraped_ok_numbers": int(scraped_ok_numbers),
        "scraped_failed": int(scraped_failed),
    })

    # Dashboard scrape summary
    try:
        st.info(
            f"🧽 Scrape Results: **{out['debug_counts']['scraped_ok_text']} ok-text** | "
            f"**{out['debug_counts']['scraped_ok_numbers']} ok-numbers** | "
            f"**{out['debug_counts']['scraped_failed']} failed**"
        )
    except Exception:
        pass

    # status summarization
    if scraped_ok_text == 0:
        out["status"] = "failed"
        out["status_detail"] = "no_usable_text"
    elif scraped_ok_numbers == 0:
        out["status"] = "partial"
        out["status_detail"] = "text_ok_numbers_empty"
    else:
        out["status"] = "success"
        out["status_detail"] = "ok"

    return out



def fingerprint_text(text: str) -> str:
    """Stable short fingerprint for fetched content (for debugging + determinism checks)."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]

def unit_clean_first_letter(unit: str) -> str:
    """Normalize units to first letter (T/B/M/K/%), ignoring $ and spaces."""
    if not unit:
        return ""
    u = unit.replace("$", "").replace(" ", "").strip().upper()
    return u[0] if u else ""

# =========================================================
# 7. LLM QUERY FUNCTIONS
# =========================================================

def query_perplexity(query: str, web_context: Dict, query_structure: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Query Perplexity and return a validated JSON string (LLMResponse-compatible).
    Removes 'action' and excludes None fields from output JSON.
    """
    if not PERPLEXITY_KEY:
        st.error("❌ PERPLEXITY_KEY not set.")
        return None

    query_structure = query_structure or {}
    structure_txt = ""
    ordering_contract = ""

    try:
        structure_txt, ordering_contract = build_query_structure_prompt(query_structure)
    except Exception:
        structure_txt = ""
        ordering_contract = ""

    # Web context: show top sources + snippets
    sources = (web_context.get("sources", []) if isinstance(web_context, dict) else []) or []
    search_results = (web_context.get("search_results", []) if isinstance(web_context, dict) else []) or []
    search_count = int(web_context.get("search_count", len(search_results)) if isinstance(web_context, dict) else 0)

    context_section = "WEB CONTEXT:\n"
    for url in sources[:6]:
        content = (web_context.get("scraped_content", {}) or {}).get(url) if isinstance(web_context, dict) else None
        if content:
            context_section += f"\n{url}:\n{str(content)[:800]}...\n"
        else:
            context_section += f"\n{url}\n"

    enhanced_query = (
        f"{context_section}\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"User Question: {query}\n\n"
        f"{structure_txt}\n\n"
        f"{ordering_contract}\n"
        f"Web search returned {search_count} results.\n"
        f"Return ONLY valid JSON matching the template and include all required fields."
    )

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "temperature": 0.0,
        "max_tokens": 2400,
        "top_p": 1.0,
        "messages": [{"role": "user", "content": enhanced_query}]
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data:
            raise Exception("No 'choices' in Perplexity response")

        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise Exception("Empty Perplexity response")

        parsed = parse_json_safely(content, "Perplexity")
        if not parsed:
            return create_fallback_response(query, search_count, web_context)

        repaired = repair_llm_response(parsed)

        # Ensure action is removed even if present
        repaired.pop("action", None)

        validate_numeric_fields(repaired, "Perplexity")

        try:
            llm_obj = LLMResponse.model_validate(repaired)

            # Ensure action not present (belt + suspenders)
            if hasattr(llm_obj, "action"):
                llm_obj.action = None

            # Merge web sources
            if isinstance(web_context, dict) and web_context.get("sources"):
                existing = llm_obj.sources or []
                merged = list(dict.fromkeys(existing + web_context["sources"]))
                llm_obj.sources = merged[:10]
                llm_obj.freshness = "Current (web-enhanced)"

            result = llm_obj.model_dump_json(exclude_none=True)
            cache_llm_response(query, web_context, result)
            return result

        except ValidationError as e:
            st.warning(f"⚠️ Pydantic validation failed: {e}")
            return create_fallback_response(query, search_count, web_context)

    except Exception as e:
        st.error(f"❌ Perplexity API error: {e}")
        return create_fallback_response(query, search_count, web_context)


def query_perplexity_raw(prompt: str, max_tokens: int = 400, timeout: int = 30) -> str:
    """
    Raw Perplexity call that returns text only.
    IMPORTANT: Does NOT attempt to validate as LLMResponse.
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""

def create_fallback_response(query: str, search_count: int, web_context: Dict) -> str:
    """Create fallback response matching schema, excluding None fields and removing action."""
    fallback = LLMResponse(
        executive_summary=f"Analysis of '{query}' completed with {search_count} web sources. Schema validation used fallback structure.",
        primary_metrics={
            "sources": MetricDetail(name="Web Sources", value=search_count, unit="sources"),
            "quality": MetricDetail(name="Data Quality", value=70, unit="%")
        },
        key_findings=[
            f"Web search found {search_count} relevant sources.",
            "Primary model output required fallback due to format issues.",
            "Manual review of raw data recommended for accuracy."
        ],
        top_entities=[
            TopEntityDetail(name="Source 1", share="N/A", growth="N/A")
        ],
        trends_forecast=[
            TrendForecastDetail(trend="Schema validation used fallback", direction="⚠️", timeline="Now")
        ],
        visualization_data=VisualizationData(
            chart_labels=["Attempt"],
            chart_values=[search_count],
            chart_title="Search Results"
        ),
        sources=web_context.get("sources", []),
        confidence=60,
        freshness="Current (fallback)",
        action=None
    )

    return fallback.model_dump_json(exclude_none=True)


# =========================================================
# 7B. ANCHORED EVOLUTION QUERY
# =========================================================

def _ensure_metric_labels(metric_changes: list) -> list:
    """
    Backward/forward compatible label normalization:
    - guarantees a non-empty display label
    - adds aliases so different UIs render correctly: metric_name, metric, label
    """
    import re

    def _prettify(s: str) -> str:
        s = str(s or "").strip()
        if not s:
            return ""
        s = s.replace("__", " ").replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s[:120]

    out = []
    for row in (metric_changes or []):
        if not isinstance(row, dict):
            continue

        name = row.get("name")
        if isinstance(name, str):
            name = name.strip()
        else:
            name = ""

        # try to derive a label if name missing (canonical_key or metric_definition.name)
        if not name:
            md = row.get("metric_definition") if isinstance(row.get("metric_definition"), dict) else {}
            name = (md.get("name") or "").strip() if isinstance(md.get("name"), str) else ""
        if not name:
            ckey = row.get("canonical_key")
            name = _prettify(ckey) if ckey else "Unknown Metric"

        # write canonical label + aliases
        row["name"] = name
        row.setdefault("metric_name", name)
        row.setdefault("metric", name)
        row.setdefault("label", name)

        out.append(row)

    return out


def format_previous_metrics(metrics: Dict) -> str:
    """Format previous metrics for prompt"""
    if not metrics:
        return "No previous metrics available"

    lines = []
    for key, m in metrics.items():
        if isinstance(m, dict):
            lines.append(f"- {m.get('name', key)}: {m.get('value', 'N/A')} {m.get('unit', '')}")
    return "\n".join(lines) if lines else "No metrics"

def format_previous_entities(entities: List) -> str:
    """Format previous entities for prompt"""
    if not entities:
        return "No previous entities available"

    lines = []
    for i, e in enumerate(entities, 1):
        if isinstance(e, dict):
            lines.append(f"{i}. {e.get('name', 'Unknown')}: {e.get('share', 'N/A')} share, {e.get('growth', 'N/A')} growth")
    return "\n".join(lines) if lines else "No entities"

def format_previous_findings(findings: List) -> str:
    """Format previous findings for prompt"""
    if not findings:
        return "No previous findings available"

    lines = [f"- {f}" for f in findings if f]
    return "\n".join(lines) if lines else "No findings"

def calculate_time_ago(timestamp_str: str) -> str:
    """Calculate human-readable time difference"""
    try:
        prev_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        delta = datetime.now() - prev_time.replace(tzinfo=None)

        hours = delta.total_seconds() / 3600
        if hours < 24:
            return f"{hours:.1f} hours ago"
        elif hours < 168:  # 7 days
            return f"{hours/24:.1f} days ago"
        elif hours < 720:  # 30 days
            return f"{hours/168:.1f} weeks ago"
        else:
            return f"{hours/720:.1f} months ago"
    except:
        return "unknown time ago"

def query_perplexity_anchored(query: str, previous_data: Dict, web_context: Dict, temperature: float = 0.1) -> str:
    """
    Query Perplexity with previous analysis as anchor.
    This produces an evolution-aware response that tracks changes.
    """

    prev_response = previous_data.get("primary_response", {})
    prev_timestamp = previous_data.get("timestamp", "")
    prev_question = previous_data.get("question", query)

    time_ago = calculate_time_ago(prev_timestamp)

    # Build the anchored prompt
    anchored_prompt = EVOLUTION_PROMPT_TEMPLATE.format(
        time_ago=time_ago,
        previous_question=prev_question,
        previous_timestamp=prev_timestamp,
        previous_summary=prev_response.get("executive_summary", "No previous summary"),
        previous_metrics=format_previous_metrics(prev_response.get("primary_metrics", {})),
        previous_entities=format_previous_entities(prev_response.get("top_entities", [])),
        previous_findings=format_previous_findings(prev_response.get("key_findings", [])),
        query=query
    )

    # Add web context if available
    if web_context.get("summary"):
        anchored_prompt = f"CURRENT WEB RESEARCH:\n{web_context['summary']}\n\n{anchored_prompt}"

    # API request
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }


    payload = {
        "model": "sonar",
        "temperature": 0.0,      # DETERMINISTIC
        "max_tokens": 2500,
        "top_p": 1.0,            # DETERMINISTIC
        "messages": [{"role": "user", "content": anchored_prompt}]
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data:
            raise Exception("No choices in response")

        content = data["choices"][0]["message"]["content"]
        if not content:
            raise Exception("Empty response")

        # Parse JSON
        parsed = parse_json_safely(content, "Perplexity-Anchored")
        if not parsed:
            return create_anchored_fallback(query, previous_data, web_context)

        # Add sources from web context
        if web_context.get("sources"):
            existing = parsed.get("sources", [])
            parsed["sources"] = list(dict.fromkeys(existing + web_context["sources"]))[:10]

        return json.dumps(parsed)

    except Exception as e:
        st.error(f"❌ Anchored query error: {e}")
        return create_anchored_fallback(query, previous_data, web_context)

def create_anchored_fallback(query: str, previous_data: Dict, web_context: Dict) -> str:
    """Create fallback for anchored evolution query"""
    prev_response = previous_data.get("primary_response", {})

    fallback = {
        "executive_summary": f"Evolution analysis for '{query}' - model returned invalid format. Showing previous data.",
        "analysis_delta": {
            "time_since_previous": calculate_time_ago(previous_data.get("timestamp", "")),
            "overall_trend": "unknown",
            "major_changes": ["Unable to determine changes - API error"],
            "data_freshness": "Unknown"
        },
        "primary_metrics": prev_response.get("primary_metrics", {}),
        "key_findings": ["[UNCHANGED] " + f for f in prev_response.get("key_findings", [])[:3]],
        "top_entities": prev_response.get("top_entities", []),
        "trends_forecast": prev_response.get("trends_forecast", []),
        "sources": web_context.get("sources", []),
        "confidence": 50,
        "freshness": "Fallback",
        "drift_summary": {
            "metrics_changed": 0,
            "metrics_unchanged": len(prev_response.get("primary_metrics", {})),
            "entities_reshuffled": 0,
            "findings_updated": 0,
            "overall_stability_pct": 100
        }
    }
    return json.dumps(fallback)

# =========================================================
# 8. VALIDATION & SCORING
# =========================================================


def parse_number_with_unit(val_str: str) -> float:
    """
    Parse a numeric string into a comparable base scale.
    Returns a float in "millions" for currency/volume-like values.
    Percentages are returned as their numeric value (e.g., "9.8%" -> 9.8).

    Handles:
      - $58.3B, 58.3B, S$29.8B, 29.8 S$B, USD 21.18 B
      - 58.3 billion, 58.3 bn, 58.3 million, 58.3 mn, 570 thousand
      - 570,000 (interpreted as an absolute count -> converted to millions)
      - 9.8% (kept as 9.8)
    """
    if val_str is None:
        return 0.0

    s = str(val_str).strip()
    if not s:
        return 0.0

    s_low = s.lower()

    # If it's a percentage, return the raw percent number (not millions)
    if "%" in s_low:
        m = re.search(r'(-?\d+(?:\.\d+)?)', s_low)
        if not m:
            return 0.0
        try:
            return float(m.group(1))
        except Exception:
            return 0.0

    # Normalize: remove commas and common currency tokens/symbols
    # (keep letters because we need bn/mn/b/m/k detection)
    s_low = s_low.replace(",", " ")
    for token in ["s$", "usd", "sgd", "us$", "$", "€", "£", "aud", "cad"]:
        s_low = s_low.replace(token, " ")

    # Collapse whitespace
    s_low = re.sub(r"\s+", " ", s_low).strip()

    # Extract the first number
    m = re.search(r'(-?\d+(?:\.\d+)?)', s_low)
    if not m:
        return 0.0

    try:
        num = float(m.group(1))
    except Exception:
        return 0.0

    # Look at the remaining text after the number for unit words/suffix
    tail = s_low[m.end():].strip()

    # Decide multiplier (base = millions)
    # billions -> *1000, millions -> *1, thousands -> *0.001
    multiplier = 1.0

    # Word-based units
    if re.search(r'\b(trillion|tn)\b', tail):
        multiplier = 1_000_000.0  # trillion -> million
    elif re.search(r'\b(billion|bn)\b', tail):
        multiplier = 1000.0
    elif re.search(r'\b(million|mn)\b', tail):
        multiplier = 1.0
    elif re.search(r'\b(thousand|k)\b', tail):
        multiplier = 0.001
    else:
        # Suffix-style units (possibly with spaces), e.g. "29.8 b", "21.18 b", "58.3m"
        # We only look at the very first letter-ish token in tail.
        t0 = tail[:4].strip()  # enough to catch "b", "m", "k"
        if t0.startswith("b"):
            multiplier = 1000.0
        elif t0.startswith("m"):
            multiplier = 1.0
        elif t0.startswith("k"):
            multiplier = 0.001
        else:
            # No unit detected. If it's a big integer like 570000 (jobs, people),
            # interpret as an absolute count and convert to millions.
            # (570000 -> 0.57 million)
            if abs(num) >= 10000 and float(num).is_integer():
                multiplier = 1.0 / 1_000_000.0
            else:
                multiplier = 1.0

    return num * multiplier


def numeric_consistency_with_sources(primary_data: dict, web_context: dict) -> float:
    """Compare primary numbers vs source numbers"""
    primary_metrics = primary_data.get("primary_metrics", {})
    primary_numbers = []

    for metric in primary_metrics.values():
        if isinstance(metric, dict):
            val = metric.get("value")
            num = parse_number_with_unit(str(val))
            if num > 0:
                primary_numbers.append(num)

    if not primary_numbers:
        return 50.0  # Neutral when no metrics to compare

    # Extract source numbers with same parsing
    source_numbers = []
    search_results = web_context.get("search_results", [])

    for result in search_results:
        snippet = str(result.get("snippet", ""))
        # Match patterns like "$58.3B", "123M", "456 billion"
        patterns = [
            r'\$?(\d+(?:\.\d+)?)\s*([BbMmKk])',  # $58.3B
            r'(\d+(?:\.\d+)?)\s*(billion|million|thousand)',  # 58.3 billion
        ]

        for pattern in patterns:
            matches = re.findall(pattern, snippet, re.IGNORECASE)
            for num, unit in matches:
                source_numbers.append(parse_number_with_unit(f"{num}{unit[0].upper()}"))

    if not source_numbers:
        return 50.0  # Neutral when no source numbers found

    # Check agreement (within 25% tolerance)
    agreements = 0
    for p_num in primary_numbers:
        for s_num in source_numbers:
            if abs(p_num - s_num) / max(p_num, s_num, 1) < 0.25:
                agreements += 1
                break

    # Scale: 0 agreements = 30%, all agreements = 95%
    agreement_ratio = agreements / len(primary_numbers)
    agreement_pct = 30.0 + (agreement_ratio * 65.0)
    return min(agreement_pct, 95.0)

def numeric_consistency_with_sources_v2(primary_data: dict, web_context: dict) -> float:
    """
    Stable numeric consistency (0-100):
    - Evidence text: search_results snippets + web_context summary + scraped_content
    - Unit-aware parsing via parse_number_with_unit()
    - Range-aware (supports min/max if metric has a 'range' dict)
    - Downweights proxy metrics (is_proxy=True) so they don't tank the score
    """

    try:
        # Prefer canonical metrics if available (has is_proxy, range, etc.)
        metrics = primary_data.get("primary_metrics_canonical") or primary_data.get("primary_metrics") or {}
        if not isinstance(metrics, dict) or not metrics:
            return 50.0

        # -----------------------------
        # Build evidence text corpus
        # -----------------------------
        texts = []

        # 1) snippets
        sr = (web_context or {}).get("search_results") or []
        if isinstance(sr, list):
            for r in sr:
                if isinstance(r, dict):
                    snip = r.get("snippet", "")
                    if isinstance(snip, str) and snip.strip():
                        texts.append(snip)

        # 2) summary
        summary = (web_context or {}).get("summary") or ""
        if isinstance(summary, str) and summary.strip():
            texts.append(summary)

        # 3) scraped_content
        scraped = (web_context or {}).get("scraped_content") or {}
        if isinstance(scraped, dict):
            for _, content in scraped.items():
                if isinstance(content, str) and content.strip():
                    texts.append(content)

        evidence_text = "\n".join(texts)
        if not evidence_text.strip():
            return 45.0  # no evidence stored

        # -----------------------------
        # Extract numeric candidates from evidence text
        # -----------------------------
        # Keep this broad; parse_number_with_unit will normalize.
        patterns = [
            r'\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[BbMmKk]\b',                 # 29.8B, 570K, 1.2M
            r'\$?\s?\d+(?:\.\d+)?\s*(?:billion|million|thousand|bn|mn)\b',      # 29.8 billion, 29.8 bn
            r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b',                               # 570,000
            r'\b\d+(?:\.\d+)?\s*%\b',                                          # 9.8%
        ]

        evidence_numbers = []
        lowered = evidence_text.lower()

        for pat in patterns:
            for m in re.findall(pat, lowered, flags=re.IGNORECASE):
                n = parse_number_with_unit(str(m))
                if n and n > 0:
                    evidence_numbers.append(n)

        # If nothing extracted, don’t penalize too hard
        if not evidence_numbers:
            return 50.0

        # -----------------------------
        # Verify each metric against evidence numbers (tolerance match)
        # -----------------------------
        def _metric_candidates(m: dict) -> list:
            """Return list of candidate numeric values for a metric (range-aware)."""
            out = []
            if not isinstance(m, dict):
                return out

            # Range support: check min/max if present
            rng = m.get("range") if isinstance(m.get("range"), dict) else None
            if rng:
                if rng.get("min") is not None:
                    out.append(rng.get("min"))
                if rng.get("max") is not None:
                    out.append(rng.get("max"))

            # Also check main value
            if m.get("value") is not None:
                out.append(m.get("value"))

            return out

        def _parse_metric_num(val, unit_hint: str = "") -> float:
            # build a value+unit string so parse_number_with_unit has a chance
            if val is None:
                return 0.0
            s = str(val)
            if unit_hint and unit_hint.lower() not in s.lower():
                s = f"{s} {unit_hint}"
            return parse_number_with_unit(s)

        def _is_supported(target: float, evidence_nums: list, rel_tol: float = 0.25) -> bool:
            # same tolerance approach as v1 (25%)
            if not target or target <= 0:
                return False
            for e in evidence_nums:
                if e <= 0:
                    continue
                if abs(target - e) / max(target, e, 1) < rel_tol:
                    return True
            return False

        supported_w = 0.0
        total_w = 0.0

        for _, m in metrics.items():
            if not isinstance(m, dict):
                continue

            unit = str(m.get("unit") or "").strip()

            # proxy weighting
            is_proxy = bool(m.get("is_proxy"))
            w = 0.5 if is_proxy else 1.0

            cands = _metric_candidates(m)
            if not cands:
                continue

            # parse candidates into numeric values
            parsed_targets = []
            for c in cands:
                n = _parse_metric_num(c, unit_hint=unit)
                if n and n > 0:
                    parsed_targets.append(n)

            if not parsed_targets:
                continue

            total_w += w

            # supported if ANY candidate matches evidence
            if any(_is_supported(t, evidence_numbers, rel_tol=0.25) for t in parsed_targets):
                supported_w += w

        if total_w <= 0:
            return 50.0

        ratio = supported_w / total_w
        # Map: keep a soft floor so one miss doesn't tank the whole run
        score = 30.0 + (ratio * 65.0)  # same scale as v1 (30..95)
        return min(max(score, 20.0), 95.0)

    except Exception:
        return 45.0



def source_consensus(web_context: dict) -> float:
    """
    Calculate source consensus based on proportion of high-quality sources.
    Returns continuous score 0-100 based on quality distribution.
    """
    reliabilities = web_context.get("source_reliability", [])

    if not reliabilities:
        return 50.0  # Neutral when no sources

    total = len(reliabilities)
    high_count = sum(1 for r in reliabilities if "✅" in str(r))
    medium_count = sum(1 for r in reliabilities if "⚠️" in str(r))
    low_count = sum(1 for r in reliabilities if "❌" in str(r))

    # Weighted score: High=100, Medium=60, Low=30
    weighted_sum = (high_count * 100) + (medium_count * 60) + (low_count * 30)
    consensus_score = weighted_sum / total

    # Bonus for having multiple high-quality sources
    if high_count >= 3:
        consensus_score = min(100, consensus_score + 10)
    elif high_count >= 2:
        consensus_score = min(100, consensus_score + 5)

    return round(consensus_score, 1)

def evidence_based_veracity(primary_data: dict, web_context: dict) -> dict:
    """
    Evidence-driven veracity scoring.
    Returns breakdown of component scores and overall score (0-100).
    """
    breakdown = {}

    # 1. SOURCE QUALITY (35% weight)
    sources = primary_data.get("sources", [])
    src_score = source_quality_score(sources)
    breakdown["source_quality"] = round(src_score, 1)

    # 2. NUMERIC CONSISTENCY (30% weight)
    num_score = numeric_consistency_with_sources_v2(primary_data, web_context)
    breakdown["numeric_consistency"] = round(num_score, 1)

    # 3. CITATION DENSITY (20% weight)
    # FIXED: Higher score when sources support findings, not penalize detail
    sources_count = len(sources)
    findings_count = len(primary_data.get("key_findings", []))
    metrics_count = len(primary_data.get("primary_metrics", {}))

    # Total claims = findings + metrics
    total_claims = findings_count + metrics_count

    if total_claims == 0:
        citations_score = 40.0  # Low score for no claims
    else:
        # Ratio of sources to claims - ideal is ~0.5-1.0 sources per claim
        ratio = sources_count / total_claims
        if ratio >= 1.0:
            citations_score = 90.0  # Well-supported
        elif ratio >= 0.5:
            citations_score = 70.0 + (ratio - 0.5) * 40  # 70-90 range
        elif ratio >= 0.25:
            citations_score = 50.0 + (ratio - 0.25) * 80  # 50-70 range
        else:
            citations_score = ratio * 200  # 0-50 range

    breakdown["citation_density"] = round(min(citations_score, 95.0), 1)

    # 4. SOURCE CONSENSUS (15% weight)
    consensus_score = source_consensus(web_context)
    breakdown["source_consensus"] = round(consensus_score, 1)

    # Calculate weighted total
    total_score = (
        breakdown["source_quality"] * 0.35 +
        breakdown["numeric_consistency"] * 0.30 +
        breakdown["citation_density"] * 0.20 +
        breakdown["source_consensus"] * 0.15
    )

    breakdown["overall"] = round(total_score, 1)

    return breakdown

def calculate_final_confidence(
    base_conf: float,
    evidence_score: float
) -> float:
    """
    Calculate final confidence score.

    Formula balances model confidence with evidence quality:
    - Evidence has higher weight (65%) as it's more objective
    - Model confidence (35%) is adjusted by evidence quality

    This ensures:
    - High model + High evidence → High final (~85-90%)
    - High model + Low evidence → Medium final (~55-65%)
    - Low model + High evidence → Medium-High final (~70-80%)
    - Low model + Low evidence → Low final (~40-50%)
    """

    # Normalize inputs to 0-100 range
    base_conf = max(0, min(100, base_conf))
    evidence_score = max(0, min(100, evidence_score))

    # 1. EVIDENCE COMPONENT (65% weight) - Primary driver
    evidence_component = evidence_score * 0.65

    # 2. MODEL COMPONENT (35% weight) - Adjusted by evidence quality
    # When evidence is weak, model confidence is discounted
    evidence_multiplier = 0.5 + (evidence_score / 200)  # Range: 0.5 to 1.0
    model_component = base_conf * evidence_multiplier * 0.35

    final = evidence_component + model_component

    # Ensure result is in valid range
    return round(max(0, min(100, final)), 1)

# =========================================================
# 8A. DETERMINISTIC DIFF ENGINE
# Pure Python computation - no LLM variance
# =========================================================

@dataclass
class MetricDiff:
    """Single metric change record"""
    name: str
    old_value: Optional[float]
    new_value: Optional[float]
    old_raw: str  # Original string representation
    new_raw: str
    unit: str
    change_pct: Optional[float]
    change_type: str  # 'increased', 'decreased', 'unchanged', 'added', 'removed'

@dataclass
class EntityDiff:
    """Single entity ranking change record"""
    name: str
    old_rank: Optional[int]
    new_rank: Optional[int]
    old_share: Optional[str]
    new_share: Optional[str]
    rank_change: Optional[int]  # Positive = moved up
    change_type: str  # 'moved_up', 'moved_down', 'unchanged', 'added', 'removed'

@dataclass
class FindingDiff:
    """Single finding change record"""
    old_text: Optional[str]
    new_text: Optional[str]
    similarity: float  # 0-100
    change_type: str  # 'retained', 'modified', 'added', 'removed'

@dataclass
class EvolutionDiff:
    """Complete diff between two analyses"""
    old_timestamp: str
    new_timestamp: str
    time_delta_hours: Optional[float]
    metric_diffs: List[MetricDiff]
    entity_diffs: List[EntityDiff]
    finding_diffs: List[FindingDiff]
    stability_score: float  # 0-100
    summary_stats: Dict[str, int]

# =========================================================
# CANONICAL METRIC REGISTRY & SEMANTIC FINDING HASH
# Add this section after the dataclass definitions (around line 1587)
# =========================================================

# ------------------------------------
# CANONICAL METRIC REGISTRY
# Removes LLM control over metric identity
# ------------------------------------

# Metric type definitions with aliases
            # =========================
# PATCH MR1 (ADDITIVE): de-ambiguate "sales" so unit-sales doesn't map to Revenue
# - Remove standalone "sales" from Revenue aliases (too ambiguous)
# - Add money-explicit revenue phrases instead ("sales revenue", "sales value", etc.)
# - Add a couple of volume-style aliases under units_sold ("sales volume", "volume sales")
            # =========================

METRIC_REGISTRY = {
    # Market Size metrics
    "market_size": {
        "canonical_name": "Market Size",
        "aliases": [
            "market size", "market value", "market cap", "total market",
            "global market", "market valuation", "industry size",
            "total addressable market", "tam", "market worth"
        ],
        "unit_type": "currency",
        "category": "size"
    },
    "market_size_current": {
        "canonical_name": "Current Market Size",
        "aliases": [
            "2024 market size", "2025 market size", "current market",
            "present market size", "today market", "current year market",
            "market size 2024", "market size 2025"
        ],
        "unit_type": "currency",
        "category": "size"
    },
    "market_size_projected": {
        "canonical_name": "Projected Market Size",
        "aliases": [
            "projected market", "forecast market", "future market",
            "2026 market", "2027 market", "2028 market", "2029 market", "2030 market",
            "market projection", "expected market size", "estimated market"
        ],
        "unit_type": "currency",
        "category": "size"
    },

    # Growth metrics
    "cagr": {
        "canonical_name": "CAGR",
        "aliases": [
            "cagr", "compound annual growth", "compound growth rate",
            "annual growth rate", "growth rate", "yearly growth"
        ],
        "unit_type": "percentage",
        "category": "growth"
    },
    "yoy_growth": {
        "canonical_name": "YoY Growth",
        "aliases": [
            "yoy growth", "year over year", "year-over-year",
            "annual growth", "yearly growth rate", "growth percentage"
        ],
        "unit_type": "percentage",
        "category": "growth"
    },

    # Revenue metrics
    "revenue": {
        "canonical_name": "Revenue",
        "aliases": [
            "revenue",
            # =========================
            # PATCH MR1 (CHANGED): removed ambiguous standalone alias "sales"
            # =========================
            # "sales",
            # =========================
            "total revenue", "annual revenue",
            "yearly revenue", "gross revenue",

            # =========================
            # PATCH MR1 (ADDITIVE): money-explicit sales phrasing (revenue-like)
            # =========================
            "sales revenue",
            "revenue from sales",
            "sales value",
            "value of sales",
            "sales (value)",
            "turnover",  # common finance synonym
            # =========================
        ],
        "unit_type": "currency",
        "category": "financial"
    },

    # Market share
    "market_share": {
        "canonical_name": "Market Share",
        "aliases": [
            "market share", "share", "market portion", "market percentage",
            "share of market"
        ],
        "unit_type": "percentage",
        "category": "share"
    },

    # Volume metrics
    "units_sold": {
        "canonical_name": "Units Sold",
        "aliases": [
            "units sold", "unit sales", "volume", "units shipped",
            "shipments", "deliveries", "production volume",

            # =========================
            # PATCH MR1 (ADDITIVE): common unit-sales phrasing variants
            # =========================
            "sales volume",
            "volume sales",
            # =========================
        ],
        "unit_type": "count",
        "category": "volume"
    },

    # Pricing
    "average_price": {
        "canonical_name": "Average Price",
        "aliases": [
            "average price", "avg price", "mean price", "asp",
            "average selling price", "unit price"
        ],
        "unit_type": "currency",
        "category": "pricing"
    },

    # -------------------------
    # Country / Macro metrics
    # -------------------------
    "gdp": {
        "canonical_name": "GDP",
        "aliases": ["gdp", "gross domestic product", "economic output"],
        "unit_type": "currency",
        "category": "macro"
    },
    "gdp_per_capita": {
        "canonical_name": "GDP per Capita",
        "aliases": ["gdp per capita", "gdp/capita", "income per person", "per capita gdp"],
        "unit_type": "currency",
        "category": "macro"
    },
    "gdp_growth": {
        "canonical_name": "GDP Growth",
        "aliases": ["gdp growth", "economic growth", "growth rate of gdp", "real gdp growth"],
        "unit_type": "percentage",
        "category": "macro"
    },
    "population": {
        "canonical_name": "Population",
        "aliases": ["population", "population size", "number of people"],
        "unit_type": "count",
        "category": "macro"
    },
    "exports": {
        "canonical_name": "Exports",
        "aliases": ["exports", "export value", "total exports"],
        "unit_type": "currency",
        "category": "trade"
    },
    "imports": {
        "canonical_name": "Imports",
        "aliases": ["imports", "import value", "total imports"],
        "unit_type": "currency",
        "category": "trade"
    },
    "inflation": {
        "canonical_name": "Inflation",
        "aliases": ["inflation", "cpi", "consumer price index", "inflation rate"],
        "unit_type": "percentage",
        "category": "macro"
    },
    "interest_rate": {
        "canonical_name": "Interest Rate",
        "aliases": ["interest rate", "policy rate", "benchmark rate", "central bank rate"],
        "unit_type": "percentage",
        "category": "macro"
    }
}

            # =========================
# END PATCH MR1
            # =========================

# Year extraction pattern
YEAR_PATTERN = re.compile(r'(20\d{2})')

# ------------------------------------
# DETERMINISTIC QUESTION SIGNALS
# Drives metric table templates (no LLM)
# ------------------------------------

QUESTION_CATEGORY_TEMPLATES = {
    "country": [
        "gdp",
        "gdp_per_capita",
        "gdp_growth",
        "population",
        "exports",
        "imports",
        "inflation",
        "interest_rate",
    ],
    "industry": [
        "market_size_current",
        "market_size_projected",
        "cagr",
        "revenue",
        "market_share",
        "units_sold",
        "average_price",
    ],
}

def get_expected_metric_ids_for_category(category: str) -> List[str]:
    """
    Domain-agnostic mapping from a template/category string to expected metric IDs.

    Backward compatible:
      - accepts legacy categories like 'country', 'industry', 'company', 'generic'
      - also accepts template IDs like 'ENTITY_OVERVIEW_MARKET_LIGHT_V1', etc.

    NOTE:
    - This function returns a *default* set for a given template/category.
    - The profiler (classify_question_signals) can override/compose expected_metric_ids dynamically.
    """
    c_raw = (category or "unknown").strip()
    c = c_raw.lower().strip()

    # -------------------------
    # New generalized templates
    # -------------------------
    if c in {"entity_overview_country_light_v1", "entity_overview_country_v1"}:
        return [
            "population",
            "gdp_nominal",
            "gdp_per_capita",
            "gdp_growth",
            "inflation",
            "currency",
            "unemployment",
            "exports",
            "imports",
            "top_industries",
        ]

    if c in {"entity_overview_market_light_v1"}:
        return [
            "market_size_current",
            "market_size_projected",
            "cagr",
            "key_trends",
            "top_players",
        ]

    if c in {"entity_overview_market_heavy_v1"}:
        return [
            "market_size_current",
            "market_size_projected",
            "cagr",
            "key_trends",
            "top_players",
            "key_regions",
            "segments",
            "market_share",
            "revenue",
            "units_sold",
            "average_price",
        ]

    if c in {"entity_overview_company_light_v1", "entity_overview_company_v1"}:
        return [
            "revenue",
            "growth",
            "gross_margin",
            "operating_margin",
            "net_income",
            "market_cap",
            "valuation_multiple",
        ]

    if c in {"entity_overview_product_light_v1", "entity_overview_product_v1"}:
        return [
            "average_price",
            "units_sold",
            "market_share",
            "growth",
            "key_trends",
        ]

    if c in {"entity_overview_topic_v1", "generic_v1"}:
        return []

    # -------------------------
    # Legacy categories (still supported)
    # -------------------------
    if c == "country":
        return get_expected_metric_ids_for_category("ENTITY_OVERVIEW_COUNTRY_LIGHT_V1")

    if c == "industry":
        # legacy industry defaults to light market
        return get_expected_metric_ids_for_category("ENTITY_OVERVIEW_MARKET_LIGHT_V1")

    if c == "company":
        return get_expected_metric_ids_for_category("ENTITY_OVERVIEW_COMPANY_LIGHT_V1")

    if c == "generic":
        return []

    # fallback
    return []


def classify_question_signals(query: str) -> Dict[str, Any]:
    """
    Deterministically classify query and return:
      - category: high-level bucket used for templates (country | industry | company | generic)
      - expected_metric_ids: list[str]
      - signals: list[str] (debuggable reasons)
      - years: list[int]
      - regions: list[str]
      - intents: list[str] (market_size, growth_forecast, competitive_landscape, pricing, regulation, consumer_demand, supply_chain, investment, macro_outlook)
    """
    q_raw = (query or "").strip()
    q = q_raw.lower().strip()
    signals: List[str] = []

    if not q:
        return {
            "category": "generic",
            "expected_metric_ids": [],
            "signals": ["empty_query"],
            "years": [],
            "regions": [],
            "intents": []
        }

    # -------------------------
    # 1) Extract years (deterministic)
    # -------------------------
    years: List[int] = []
    try:
        year_matches = re.findall(r"\b(19|20)\d{2}\b", q_raw)
        # The regex above returns the first group; re-run with a non-capturing group to capture full year strings.
        year_matches_full = re.findall(r"\b(?:19|20)\d{2}\b", q_raw)
        years = sorted({int(y) for y in year_matches_full})
        if years:
            signals.append(f"years:{','.join(map(str, years[:8]))}")
    except Exception:
        years = []

    # -------------------------
    # 2) Extract regions/countries (best-effort deterministic; spaCy if available)
    # -------------------------
    regions: List[str] = []
    try:
        nlp = _try_spacy_nlp()
        if nlp:
            doc = nlp(q_raw)
            gpes = [ent.text.strip() for ent in getattr(doc, "ents", []) if ent.label_ in ("GPE", "LOC")]
            regions = []
            for g in gpes:
                if g and g.lower() not in [x.lower() for x in regions]:
                    regions.append(g)
            if regions:
                signals.append(f"regions_spacy:{','.join(regions[:6])}")
    except Exception:
        pass

    # Fallback: very lightweight region tokens
    if not regions:
        region_tokens = [
            "singapore", "malaysia", "indonesia", "thailand", "vietnam", "philippines",
            "china", "india", "japan", "korea", "australia",
            "usa", "united states", "europe", "uk", "united kingdom",
            "asean", "southeast asia", "sea", "global", "worldwide"
        ]
        hits = [t for t in region_tokens if t in q]
        if hits:
            # Keep original casing loosely (title-case single words)
            regions = [h.title() if " " not in h else h.upper() if h in ("usa", "uk") else h.title() for h in hits[:6]]
            signals.append(f"regions_kw:{','.join(hits[:6])}")

    # -------------------------
    # 3) Intent detection (domain-agnostic)
    # -------------------------
    intent_patterns: Dict[str, List[str]] = {
        "market_size": ["market size", "tam", "total addressable market", "how big", "size of the market", "market value"],
        "growth_forecast": ["cagr", "forecast", "projection", "by 20", "growth rate", "expected to", "outlook", "trend"],
        "competitive_landscape": ["key players", "competitors", "market share", "top companies", "leading players", "who are the players"],
        "pricing": ["pricing", "price", "asp", "average selling price", "cost", "margins"],
        "consumer_demand": ["demand", "users", "penetration", "adoption", "consumer", "customer", "behavior"],
        "supply_chain": ["supply", "capacity", "production", "manufacturing", "inventory", "shipment", "lead time"],
        "regulation": ["regulation", "policy", "law", "compliance", "tax", "tariff", "subsidy"],
        "investment": ["investment", "capex", "funding", "valuation", "roi", "profit", "ebitda"],
        "macro_outlook": ["gdp", "inflation", "interest rate", "policy rate", "exports", "imports", "currency", "exchange rate", "per capita"],
    }

    intents: List[str] = []
    for intent, pats in intent_patterns.items():
        if any(p in q for p in pats):
            intents.append(intent)

    # Small disambiguation: "by 2030" etc. strongly suggests forecast if years exist
    if years and "growth_forecast" not in intents and any(yr >= 2025 for yr in years):
        intents.append("growth_forecast")

    if intents:
        signals.append(f"intents:{','.join(intents[:10])}")

    # -------------------------
    # 4) Category decision (template driver)
    # -------------------------
    # Keep it coarse: country vs industry vs company vs generic
    country_kw = [
        "gdp", "per capita", "population", "exports", "imports",
        "inflation", "cpi", "interest rate", "policy rate", "central bank",
        "currency", "exchange rate"
    ]
    company_kw = ["revenue", "earnings", "profit", "ebitda", "guidance", "quarter", "fy", "10-k", "10q", "balance sheet"]
    industry_kw = [
        "market", "industry", "sector", "tam", "cagr", "market size", "market share",
        "key players", "competitors", "pricing", "forecast", "outlook"
    ]

    country_hits = [k for k in country_kw if k in q]
    company_hits = [k for k in company_kw if k in q]
    industry_hits = [k for k in industry_kw if k in q]

    # If macro intent is present, strongly bias to country
    if "macro_outlook" in intents and (regions or country_hits):
        category = "country"
        signals.append("category_rule:macro_outlook_bias_country")
    elif company_hits and not industry_hits:
        category = "company"
        signals.append(f"category_rule:company_keywords:{','.join(company_hits[:5])}")
    elif industry_hits and not country_hits:
        category = "industry"
        signals.append(f"category_rule:industry_keywords:{','.join(industry_hits[:5])}")
    elif industry_hits and country_hits:
        # tie-break: if market sizing/competitive signals exist -> industry; if macro_outlook -> country
        if "macro_outlook" in intents:
            category = "country"
            signals.append("category_rule:mixed_signals_macro_wins")
        else:
            category = "industry"
            signals.append("category_rule:mixed_signals_default_to_industry")
    else:
        category = "generic"
        signals.append("category_rule:no_template_keywords")

    # -------------------------
    # 5) Expected metric IDs (category + intent)
    # -------------------------
    expected_metric_ids: List[str] = []
    try:
        expected_metric_ids = get_expected_metric_ids_for_category(category) or []
    except Exception:
        expected_metric_ids = []

    # Add a few intent-driven metric IDs (only if your registry supports them)
    intent_metric_suggestions = {
        "market_size": ["market_size", "market_size_2024", "market_size_2025"],
        "growth_forecast": ["cagr", "forecast_period", "market_size_2030"],
        "competitive_landscape": ["market_share", "top_players"],
        "pricing": ["avg_price", "asp"],
        "consumer_demand": ["users", "penetration", "arpu"],
        "supply_chain": ["capacity", "shipments"],
        "investment": ["capex", "profit", "ebitda"],
        "macro_outlook": ["gdp", "inflation", "interest_rate", "exchange_rate"],
    }

    for intent in intents:
        for mid in intent_metric_suggestions.get(intent, []):
            if mid not in expected_metric_ids:
                expected_metric_ids.append(mid)

    return {
        "category": category,
        "expected_metric_ids": expected_metric_ids,
        "signals": signals,
        "years": years,
        "regions": regions,
        "intents": intents
    }


    def _contains_any(needle_list: List[str]) -> bool:
        return any(k in q for k in needle_list)

    # -------------------------
    # Determine intents
    # -------------------------
    intents: List[str] = []
    for intent, kws in intent_triggers.items():
        if _contains_any(kws):
            intents.append(intent)

    if intents:
        signals.append("intents:" + ",".join(sorted(set(intents))))

    # -------------------------
    # Determine entity_kind (best-effort heuristic)
    # -------------------------
    is_marketish = _contains_any(market_entity_kw) or any(i in intents for i in ["size", "growth", "forecast", "share", "segments", "players", "regions"])
    is_companyish = _contains_any(company_entity_kw) and not _contains_any(country_entity_kw)
    is_countryish = _contains_any(country_entity_kw) and not is_companyish
    is_productish = _contains_any(product_entity_kw) and not (is_marketish or is_countryish or is_companyish)

    if is_countryish:
        entity_kind = "country"
        signals.append("entity_kind:country")
    elif is_companyish:
        entity_kind = "company"
        signals.append("entity_kind:company")
    elif is_productish:
        entity_kind = "product"
        signals.append("entity_kind:product")
    elif is_marketish:
        entity_kind = "market"
        signals.append("entity_kind:market")
    else:
        entity_kind = "topic_general"
        signals.append("entity_kind:topic_general")

    # -------------------------
    # Determine scope
    # -------------------------
    is_comparative = _contains_any(comparative_kw)
    is_forecasty = _contains_any(forecast_kw) or bool(YEAR_PATTERN.findall(q_raw))

    # Broad overview should win when user explicitly asks for general explainer
    # BUT: if they also mention measurable intents (size/growth/forecast/etc.), treat as metrics_light.
    is_broad_phrase = _contains_any(broad_phrases)

    if is_comparative:
        scope = "comparative"
        signals.append("scope:comparative")
    elif is_forecasty and any(i in intents for i in ["forecast", "growth", "size"]):
        scope = "forecast_specific"
        signals.append("scope:forecast_specific")
    elif is_broad_phrase and not intents:
        scope = "broad_overview"
        signals.append("scope:broad_overview")
    else:
        # metrics light vs heavy
        heavy_asks = ["segments", "share", "volume", "regions", "players"]
        heavy_requested = any(i in intents for i in heavy_asks)
        if heavy_requested:
            scope = "metrics_heavy"
            signals.append("scope:metrics_heavy")
        else:
            scope = "metrics_light"
            signals.append("scope:metrics_light")

    # -------------------------
    # Map entity_kind -> category (backward compatible)
    # -------------------------
    if entity_kind == "country":
        category = "country"
    elif entity_kind == "company":
        category = "company"
    elif entity_kind in {"market", "product"}:
        category = "industry"
    else:
        category = "generic"

    # -------------------------
    # Choose generalized template + tiers
    # -------------------------
    # Tier meanings:
    #  1 = high extractability (size/growth/forecast)
    #  2 = medium (players/regions/basic segments)
    #  3 = low (granular channels, detailed splits) -> only if explicitly asked
    if category == "country":
        metric_template_id = "ENTITY_OVERVIEW_COUNTRY_LIGHT_V1" if scope != "metrics_heavy" else "ENTITY_OVERVIEW_COUNTRY_LIGHT_V1"
        metric_tiers_enabled = [1]
    elif category == "company":
        metric_template_id = "ENTITY_OVERVIEW_COMPANY_LIGHT_V1"
        metric_tiers_enabled = [1]
    elif category == "industry":
        if scope in {"metrics_heavy", "comparative"}:
            metric_template_id = "ENTITY_OVERVIEW_MARKET_HEAVY_V1"
            metric_tiers_enabled = [1, 2]
        else:
            metric_template_id = "ENTITY_OVERVIEW_MARKET_LIGHT_V1"
            metric_tiers_enabled = [1]
    else:
        metric_template_id = "ENTITY_OVERVIEW_TOPIC_V1"
        metric_tiers_enabled = []

    # -------------------------
    # Build expected_metric_ids dynamically from intents (domain-agnostic)
    # -------------------------
    # Slot -> metric id mapping (kept generic; avoids tourism specialization)
    # If you later add more canonical IDs, expand these mappings.
    market_slot_to_id = {
        "size_current": "market_size_current",
        "size_projected": "market_size_projected",
        "growth_cagr": "cagr",
        "growth_yoy": "growth",
        "share_key": "market_share",
        "volume_current": "units_sold",
        "price_avg": "average_price",
        "players_top": "top_players",
        "regions_key": "key_regions",
        "segments_basic": "segments",
        "trends": "key_trends",
        "revenue": "revenue",
    }

    company_slot_to_id = {
        "revenue": "revenue",
        "growth": "growth",
        "gross_margin": "gross_margin",
        "operating_margin": "operating_margin",
        "net_income": "net_income",
        "market_cap": "market_cap",
        "valuation_multiple": "valuation_multiple",
        "trends": "key_trends",
    }

    country_slot_to_id = {
        "population": "population",
        "gdp_nominal": "gdp_nominal",
        "gdp_per_capita": "gdp_per_capita",
        "gdp_growth": "gdp_growth",
        "inflation": "inflation",
        "currency": "currency",
        "unemployment": "unemployment",
        "exports": "exports",
        "imports": "imports",
        "top_industries": "top_industries",
        "trends": "key_trends",
    }

    # Determine slots from intents
    slots: List[str] = []
    if entity_kind == "country":
        # For countries: macro defaults if broad, otherwise macro intents
        if scope == "broad_overview":
            slots = ["population", "gdp_nominal", "gdp_per_capita", "gdp_growth", "inflation", "currency", "top_industries"]
        else:
            # If user asks for macro (or didn’t specify), still give a tight macro set
            slots = ["population", "gdp_nominal", "gdp_growth", "inflation", "currency"]
            if "macro" in intents:
                slots += ["unemployment", "exports", "imports"]

        mapper = country_slot_to_id

    elif entity_kind == "company":
        slots = ["revenue", "growth", "gross_margin", "operating_margin", "net_income", "market_cap", "valuation_multiple"]
        mapper = company_slot_to_id

    elif entity_kind in {"market", "product"}:
        # Tier 1 core (always when metrics_* scope)
        if scope == "broad_overview":
            slots = ["trends", "players_top"]
        else:
            slots = ["size_current", "growth_cagr"]
            if "forecast" in intents:
                slots.append("size_projected")
            if "trends" in intents:
                slots.append("trends")
            # Tier 2 (only when explicitly asked or heavy scope)
            if scope in {"metrics_heavy", "comparative"}:
                if "players" in intents:
                    slots.append("players_top")
                if "regions" in intents:
                    slots.append("regions_key")
                if "segments" in intents:
                    slots.append("segments_basic")
                if "share" in intents:
                    slots.append("share_key")
                if "volume" in intents:
                    slots.append("volume_current")
                if "price" in intents:
                    slots.append("price_avg")
            else:
                # metrics_light: include players/trends only if asked
                if "players" in intents:
                    slots.append("players_top")
                if "regions" in intents:
                    slots.append("regions_key")

        mapper = market_slot_to_id

    else:
        # topic_general
        slots = []
        mapper = {}

    expected_metric_ids = []
    for s in slots:
        mid = mapper.get(s)
        if mid:
            expected_metric_ids.append(mid)

    # If still empty but template provides defaults, use template defaults
    if not expected_metric_ids:
        expected_metric_ids = get_expected_metric_ids_for_category(metric_template_id)

    # De-dup while preserving order
    seen = set()
    expected_metric_ids = [x for x in expected_metric_ids if not (x in seen or seen.add(x))]

    # -------------------------
    # Preferred source classes (generic)
    # -------------------------
    if category == "country":
        preferred_source_classes = ["official_stats", "government", "reputable_org", "reference"]
    elif category == "company":
        preferred_source_classes = ["official_filings", "investor_relations", "reputable_org", "news"]
    elif category == "industry":
        preferred_source_classes = ["industry_association", "reputable_org", "official_stats", "news", "research_portal"]
    else:
        preferred_source_classes = ["reference", "official_stats", "reputable_org"]

    # Attach year detection signal
    years = sorted(set(YEAR_PATTERN.findall(q_raw))) if YEAR_PATTERN.findall(q_raw) else []
    if years:
        signals.append("years_detected:" + ",".join(years))

    return {
        "category": category,
        "expected_metric_ids": expected_metric_ids,
        "signals": signals,
        "entity_kind": entity_kind,
        "scope": scope,
        "metric_template_id": metric_template_id,
        "metric_tiers_enabled": metric_tiers_enabled,
        "preferred_source_classes": preferred_source_classes,
        "intents": sorted(set(intents)),
    }


def get_canonical_metric_id(metric_name: str) -> Tuple[str, str]:
    """
    Map a metric name to its canonical ID and display name.

    Returns:
        Tuple of (canonical_id, canonical_display_name)

    Example:
        "2024 Market Size" -> ("market_size_2024", "Market Size (2024)")
        "Global Market Value" -> ("market_size", "Market Size")
        "CAGR 2024-2030" -> ("cagr_2024_2030", "CAGR (2024-2030)")
    """
    import re

    if not metric_name:
        return ("unknown", "Unknown Metric")

    name_lower = metric_name.lower().strip()
    name_normalized = re.sub(r"[^\w\s]", " ", name_lower)
    name_normalized = re.sub(r"\s+", " ", name_normalized).strip()

    # Extract years
    years = YEAR_PATTERN.findall(metric_name)
    year_suffix = "_".join(sorted(years)) if years else ""

    # =========================
    # PATCH CM1 (ADDITIVE): intent signals to prevent "sales" -> "revenue" mis-maps
    # =========================
    name_words = set(name_normalized.split())

    # Explicit money intent (strong)
    money_tokens = {
        "revenue", "turnover", "valuation", "valued", "value", "market", "capex", "opex",
        "profit", "earnings", "ebitda", "income",
        "usd", "sgd", "eur", "gbp", "aud", "cad", "jpy", "cny", "rmb"
    }
    # Currency symbols appear in raw text sometimes
    has_currency_symbol = any(sym in metric_name for sym in ["$", "€", "£", "S$"])

    has_money_intent = bool(name_words & money_tokens) or has_currency_symbol

    # Explicit unit/count intent (strong)
    unit_tokens = {
        "unit", "units", "deliveries", "shipments", "registrations", "vehicles",
        "sold", "salesvolume", "volume", "pcs", "pieces"
    }
    # normalize joined token cases like "sales volume"
    joined = name_normalized.replace(" ", "")
    has_unit_intent = bool(name_words & unit_tokens) or any(t in joined for t in ["salesvolume", "unitsold", "vehiclesold"])
    # =========================

    # Find best matching registry entry
    best_match_id = None
    best_match_score = 0.0

    # =========================
    # PATCH CM2 (ADDITIVE): helper to identify revenue-like registry targets
    # =========================
    def _is_revenue_like(metric_id: str, config: dict) -> bool:
        mid = (metric_id or "").lower()
        cname = str((config or {}).get("canonical_name") or "").lower()
        # treat "market value" / "valuation" as currency-like too
        if any(k in cname for k in ["revenue", "market value", "valuation", "market size", "turnover"]):
            return True
        if any(k in mid for k in ["revenue", "market_value", "market_size", "valuation"]):
            return True
        return False
    # =========================

    for metric_id, config in METRIC_REGISTRY.items():
        for alias in config["aliases"]:
            # Remove years from alias for comparison
            alias_no_year = YEAR_PATTERN.sub("", alias).strip().lower()
            alias_no_year = re.sub(r"[^\w\s]", " ", alias_no_year)
            alias_no_year = re.sub(r"\s+", " ", alias_no_year).strip()

            name_no_year = YEAR_PATTERN.sub("", name_normalized).strip()

            # ---- base score from your existing logic ----
            score = 0.0

            # Exact match
            if alias_no_year == name_no_year and alias_no_year:
                score = 1.0

            # Containment match
            elif alias_no_year and (alias_no_year in name_no_year or name_no_year in alias_no_year):
                score = len(alias_no_year) / max(len(name_no_year), 1)

            # Word overlap match
            else:
                alias_words = set(alias_no_year.split())
                name_words_local = set(name_no_year.split())
                if alias_words and name_words_local:
                    overlap = len(alias_words & name_words_local) / len(alias_words | name_words_local)
                    score = max(score, overlap)

            # =========================
            # PATCH CM3 (ADDITIVE): disambiguation penalties/guards
            # - Block "sales" -> revenue when no money intent
            # - Block unit-intent -> revenue-like
            # - Require explicit money intent for revenue-like (soft guard, not hard stop)
            # =========================
            if score > 0.0:
                revenue_like = _is_revenue_like(metric_id, config)

                # If target is revenue-like but name has strong unit intent, penalize heavily
                if revenue_like and has_unit_intent and not has_money_intent:
                    score *= 0.20  # strong downweight

                # If target is revenue-like but name has NO money intent at all, penalize
                if revenue_like and not has_money_intent:
                    score *= 0.55  # moderate downweight

                # If name includes the word "sales" but no money intent, avoid mapping to revenue-like
                if revenue_like and ("sales" in name_no_year.split()) and not has_money_intent:
                    score *= 0.60

                # Conversely: if target is NOT revenue-like but name has money intent, slight penalty
                if (not revenue_like) and has_money_intent and ("sales" in name_no_year.split()) and not has_unit_intent:
                    score *= 0.85
            # =========================

            if score > best_match_score:
                best_match_id = metric_id
                best_match_score = score

            if best_match_score == 1.0:
                break

        if best_match_score == 1.0:
            break

    # Build canonical ID and display name
    if best_match_id and best_match_score > 0.4:
        config = METRIC_REGISTRY[best_match_id]
        canonical_base = best_match_id
        display_name = config["canonical_name"]

        if year_suffix:
            canonical_id = f"{canonical_base}_{year_suffix}"
            if len(years) == 1:
                display_name = f"{display_name} ({years[0]})"
            else:
                display_name = f"{display_name} ({'-'.join(years)})"
        else:
            canonical_id = canonical_base

        return (canonical_id, display_name)

    # Fallback: create ID from normalized name
    fallback_id = re.sub(r"\s+", "_", name_normalized)
    if year_suffix:
        fallback_id = f"{fallback_id}_{year_suffix}" if year_suffix not in fallback_id else fallback_id

    return (fallback_id, metric_name)

# ------------------------------------
# GEO + PROXY TAGGING (DETERMINISTIC)
# ------------------------------------

import re
from typing import Dict, Any, Tuple, List, Optional

REGION_KEYWORDS = {
    "APAC": ["apac", "asia pacific", "asia-pacific"],
    "SOUTHEAST_ASIA": ["southeast asia", "asean", "sea "],  # note space to reduce false matches
    "ASIA": ["asia"],
    "EUROPE": ["europe", "eu", "emea"],
    "NORTH_AMERICA": ["north america"],
    "LATAM": ["latin america", "latam"],
    "MIDDLE_EAST": ["middle east", "mena"],
    "AFRICA": ["africa"],
    "OCEANIA": ["oceania", "australia", "new zealand"],
}

GLOBAL_KEYWORDS = ["global", "worldwide", "world", "international", "across the world"]

# Minimal country map (expand deterministically over time)
COUNTRY_KEYWORDS = {
    "Singapore": ["singapore", "sg"],
    "United States": ["united states", "usa", "u.s.", "us"],
    "United Kingdom": ["united kingdom", "uk", "u.k.", "britain", "england"],
    "China": ["china", "prc"],
    "Japan": ["japan"],
    "India": ["india"],
    "Indonesia": ["indonesia"],
    "Malaysia": ["malaysia"],
    "Thailand": ["thailand"],
    "Vietnam": ["vietnam"],
    "Philippines": ["philippines"],
}

def infer_geo_scope(*texts: str) -> Dict[str, str]:
    """
    Deterministically infer geography from text.
    Returns {"geo_scope": "local|regional|global|unknown", "geo_name": "<name or ''>"}.
    Priority: country > region > global.
    """
    combined = " ".join([t for t in texts if isinstance(t, str) and t.strip()]).lower()
    if not combined:
        return {"geo_scope": "unknown", "geo_name": ""}

    # 1) Country/local (most specific)
    for country, kws in COUNTRY_KEYWORDS.items():
        for kw in kws:
            if kw in combined:
                return {"geo_scope": "local", "geo_name": country}

    # 2) Region
    for region_name, kws in REGION_KEYWORDS.items():
        for kw in kws:
            if kw in combined:
                pretty = region_name.replace("_", " ").title()
                return {"geo_scope": "regional", "geo_name": pretty}

    # 3) Global
    for kw in GLOBAL_KEYWORDS:
        if kw in combined:
            return {"geo_scope": "global", "geo_name": "Global"}

    return {"geo_scope": "unknown", "geo_name": ""}


# ---- Proxy labeling ----
# "Proxy" = adjacent metric that can help approximate the target but isn't the target definition.
# You can expand these sets deterministically.

PROXY_PATTERNS = [
    # (pattern, proxy_type, reason_template)
    (r"\bapparel\b|\bfashion\b|\bclothing\b", "adjacent_category", "Uses apparel/fashion as an adjacent proxy for streetwear."),
    (r"\bfootwear\b|\bsneaker\b|\bshoes\b", "subsegment", "Uses footwear/sneakers as a subsegment proxy for the broader market."),
    (r"\bresale\b|\bsecondary market\b", "channel_proxy", "Uses resale/secondary-market measures as a channel proxy."),
    (r"\be-?commerce\b|\bonline sales\b|\bsocial commerce\b", "channel_proxy", "Uses e-commerce indicators as a channel proxy."),
    (r"\btourism\b|\bvisitor\b|\btravel retail\b", "demand_driver", "Uses tourism indicators as a demand-driver proxy."),
    (r"\bsearch interest\b|\bgoogle trends\b|\bweb traffic\b", "interest_proxy", "Uses interest/attention measures as a proxy."),
]

# These are words that signal "core market size" style metrics (usually non-proxy if they match the user topic).
CORE_MARKET_PATTERNS = [
    r"\bmarket size\b",
    r"\bmarket value\b",
    r"\brevenue\b",
    r"\bsales\b",
    r"\bcagr\b",
    r"\bgrowth\b",
    r"\bprojected\b|\bforecast\b",
]

def infer_proxy_label(
    metric_name: str,
    question_text: str = "",
    category_hint: str = "",
    *extra_context: str
) -> Dict[str, Any]:
    """
    Deterministically label a metric as proxy/non-proxy.

    Returns fields:
      is_proxy: bool
      proxy_type: str
      proxy_reason: str
      proxy_confidence: float (0-1)
      proxy_target: str (best-guess target topic)
    """
    name = (metric_name or "").lower().strip()
    q = (question_text or "").lower().strip()
    ctx = " ".join([c for c in extra_context if isinstance(c, str)]).lower()

    combined = " ".join([name, q, ctx]).strip()

    # Default: not proxy
    out = {
        "is_proxy": False,
        "proxy_type": "",
        "proxy_reason": "",
        "proxy_confidence": 0.0,
        "proxy_target": ""
    }

    if not combined:
        return out

    # Best-effort target topic extraction (very light heuristic)
    # If you already have question signals elsewhere, you can pass them in category_hint/question_text.
    # Here we just keep a short phrase if present.
    proxy_target = ""
    if "streetwear" in q:
        proxy_target = "streetwear"
    elif "semiconductor" in q:
        proxy_target = "semiconductors"
    elif "battery" in q:
        proxy_target = "batteries"
    out["proxy_target"] = proxy_target

    # If metric name itself looks like core market patterns AND includes the target keyword, treat as non-proxy.
    # (prevents incorrectly labeling "Singapore streetwear market size" as proxy)
    core_like = any(re.search(p, name) for p in CORE_MARKET_PATTERNS)
    if core_like:
        # If it explicitly contains the topic keyword, strongly non-proxy
        if proxy_target and proxy_target in name:
            return out
        # If it says "streetwear market" in name, non-proxy even if target not detected
        if "streetwear" in name:
            return out

    # Detect proxies using patterns.
    for pat, ptype, reason in PROXY_PATTERNS:
        if re.search(pat, combined):
            out["is_proxy"] = True
            out["proxy_type"] = ptype
            out["proxy_reason"] = reason
            # Confidence: stronger if pattern appears in metric name; weaker if only in context.
            if re.search(pat, name):
                out["proxy_confidence"] = 0.9
            elif re.search(pat, ctx):
                out["proxy_confidence"] = 0.7
            else:
                out["proxy_confidence"] = 0.6
            return out

    return out


def merge_group_geo(group: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Choose the most frequent geo tag within a merged group deterministically.
    Returns (geo_scope, geo_name).
    """
    items = []
    for g in group:
        s = g.get("geo_scope", "unknown")
        n = g.get("geo_name", "")
        if s and s != "unknown":
            items.append((s, n))

    if not items:
        return "unknown", ""

    counts: Dict[str, int] = {}
    for s, n in items:
        k = f"{s}|{n}"
        counts[k] = counts.get(k, 0) + 1

    best_k = max(counts.items(), key=lambda kv: kv[1])[0]  # deterministic tie via insertion order after stable sort
    s, n = best_k.split("|", 1)
    return s, n


def merge_group_proxy(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge proxy labels for duplicates deterministically.
    If ANY member is proxy -> merged metric is proxy.
    Choose the highest-confidence proxy candidate.
    """
    best = None
    best_conf = -1.0

    for g in group:
        is_proxy = bool(g.get("is_proxy", False))
        conf = float(g.get("proxy_confidence", 0.0) or 0.0)
        if is_proxy and conf > best_conf:
            best_conf = conf
            best = g

    if best is None:
        return {
            "is_proxy": False,
            "proxy_type": "",
            "proxy_reason": "",
            "proxy_confidence": 0.0,
            "proxy_target": "",
        }

    return {
        "is_proxy": True,
        "proxy_type": best.get("proxy_type", ""),
        "proxy_reason": best.get("proxy_reason", ""),
        "proxy_confidence": float(best.get("proxy_confidence", 0.0) or 0.0),
        "proxy_target": best.get("proxy_target", ""),
    }

def canonicalize_metrics(
    metrics: Dict,
    merge_duplicates_to_range: bool = True,
    question_text: str = "",
    category_hint: str = ""
) -> Dict:
    """
    Convert metrics to canonical IDs, but NEVER merge across incompatible dimensions.

    Key fix:
      - Adds deterministic 'dimension' classification and incorporates it into canonical keys.
      - Prevents revenue vs unit-sales from merging just because the year matches.
      - Keeps your geo + proxy tagging behavior.

    Output:
      canonicalized[canonical_key] -> metric dict with:
        - canonical_id (base id)
        - canonical_key (dimension-safe id you should use everywhere downstream)
        - dimension (currency | unit_sales | percent | count | index | unknown)
        - name (dimension-corrected display name)
    """
    import re  # ========================= PATCH C0 (ADDITIVE): missing import =========================

    if not isinstance(metrics, dict):
        return {}

    # =========================
    # PATCH C1 (ADDITIVE): safe helpers for canonical numeric fields
    # - Prefer existing normalize_unit_tag/unit_family/canonicalize_numeric_candidate if present.
    # - Never breaks if those helpers are missing.
    # =========================
    def _safe_normalize_unit_tag(u: str) -> str:
        try:
            fn = globals().get("normalize_unit_tag")
            if callable(fn):
                return fn(u or "")
        except Exception:
            pass
        # minimal fallback (kept conservative)
        uu = (u or "").strip()
        ul = uu.lower().replace(" ", "")
        if ul in ("%", "pct", "percent"):
            return "%"
        if ul in ("twh",):
            return "TWh"
        if ul in ("gwh",):
            return "GWh"
        if ul in ("mwh",):
            return "MWh"
        if ul in ("kwh",):
            return "kWh"
        if ul in ("wh",):
            return "Wh"
        if ul in ("t", "trillion", "tn"):
            return "T"
        if ul in ("b", "bn", "billion"):
            return "B"
        if ul in ("m", "mn", "mio", "million"):
            return "M"
        if ul in ("k", "thousand", "000"):
            return "K"
        return uu

    def _safe_unit_family(unit_tag: str) -> str:
        try:
            fn = globals().get("unit_family")
            if callable(fn):
                return fn(unit_tag or "")
        except Exception:
            pass
        ut = (unit_tag or "").strip()
        if ut in ("TWh", "GWh", "MWh", "kWh", "Wh"):
            return "energy"
        if ut == "%":
            return "percent"
        if ut in ("T", "B", "M", "K"):
            return "magnitude"
        # currency not reliably derived here (handled elsewhere)
        return ""
    # =========================

    def infer_metric_dimension(metric_name: str, unit_raw: str) -> str:
        n = (metric_name or "").lower()
        u = (unit_raw or "").strip().lower()

        # Percent
        if "%" in u or "percent" in n or "share" in n or "cagr" in n:
            return "percent"

        # Currency signals
        currency_tokens = ["$", "s$", "usd", "sgd", "eur", "€", "gbp", "£", "jpy", "¥", "cny", "rmb", "aud", "cad"]
        if any(t in u for t in currency_tokens) or any(t in n for t in ["revenue", "market value", "valuation", "value (", "usd", "sgd", "eur"]):
            return "currency"

        # Unit sales / shipments
        unit_tokens = ["unit", "units", "sold", "sales volume", "shipments", "registrations", "deliveries", "vehicles", "pcs", "pieces", "volume"]
        if any(t in n for t in unit_tokens):
            return "unit_sales"

        # Pure counts
        if any(t in n for t in ["count", "number of", "install base", "installed base", "users", "subscribers"]) and "revenue" not in n:
            return "count"

        # Index / score
        if any(t in n for t in ["index", "score", "rating"]):
            return "index"

        return "unknown"

    def display_name_for_dimension(original_display: str, dim: str) -> str:
        if not original_display:
            return original_display

        od = original_display.strip()
        od_low = od.lower()

        if dim == "unit_sales":
            if "revenue" in od_low or "market value" in od_low or "valuation" in od_low:
                return re.sub(r"(?i)revenue|market value|valuation", "Unit Sales", od).strip()
            if od_low.startswith("sales"):
                return "Unit Sales" + od[len("Sales"):]
            if "sales" in od_low:
                return re.sub(r"(?i)sales", "Unit Sales", od).strip()
            return od

        if dim == "currency":
            if "unit sales" in od_low:
                return re.sub(r"(?i)unit sales", "Revenue", od).strip()
            return od

        if dim == "percent":
            if "unit sales" in od_low or "revenue" in od_low:
                return od
            return od

        return od

    candidates = []

    for key, metric in metrics.items():
        if not isinstance(metric, dict):
            continue

        original_name = metric.get("name", key)
        canonical_id, canonical_name = get_canonical_metric_id(original_name)

        # =========================
        # PATCH CM1 (ADDITIVE): registry-guided dimension hint
        # - If the canonical base metric is in METRIC_REGISTRY, use its unit_type
        #   as a strong prior for dimension classification.
        # - This reduces mislabel drift like "Revenue" being assigned as unit_sales
        #   (or vice-versa) purely from noisy LLM labels.
        #
        # NOTE (conflict fix, additive):
        # - Your prior code risked UnboundLocalError due to base_id scoping.
        # - We keep your legacy behavior, but guard it and define base_id upfront.
        # =========================

        registry_unit_type = ""

        # ---- PATCH CM1.A (ADDITIVE): define base_id upfront to prevent UnboundLocalError ----
        base_id = ""
        # -------------------------------------------------------------------------------

        try:
            # =========================
            # PATCH CM1.B (BUGFIX + ADDITIVE): registry base_id extraction
            # - canonical_id may contain underscores inside the base id (e.g., "market_size_2025")
            # - Find the LONGEST registry key that is a prefix of canonical_id.
            # =========================
            try:
                reg = globals().get("METRIC_REGISTRY")
                cid = str(canonical_id or "")
                if isinstance(reg, dict) and cid:
                    # choose the longest matching prefix key
                    for k in reg.keys():
                        ks = str(k)
                        if cid == ks or cid.startswith(ks + "_"):
                            if len(ks) > len(base_id):
                                base_id = ks

                    if base_id and isinstance(reg.get(base_id), dict):
                        registry_unit_type = (reg[base_id].get("unit_type") or "").strip().lower()
            except Exception:
                # keep safe defaults
                pass
            # =========================

            # -------------------------------------------------------------------
            # PATCH CM1.C (ADDITIVE): legacy code preserved, but guarded
            # - This block is redundant with CM1.B, but we keep it as requested.
            # - Guard prevents:
            #   (1) base_id undefined
            #   (2) overwriting registry_unit_type already computed above
            # -------------------------------------------------------------------
            if not registry_unit_type:
                reg = globals().get("METRIC_REGISTRY")
                if base_id and isinstance(reg, dict) and base_id in reg and isinstance(reg[base_id], dict):
                    registry_unit_type = (reg[base_id].get("unit_type") or "").strip().lower()
            # -------------------------------------------------------------------

        except Exception:
            registry_unit_type = ""

        # Map registry unit_type -> canonicalize_metrics dimension vocabulary
        # (keep it small + deterministic)
        if registry_unit_type:
            if registry_unit_type in ("currency",):
                registry_dim_hint = "currency"
            elif registry_unit_type in ("percentage", "percent"):
                registry_dim_hint = "percent"
            elif registry_unit_type in ("count",):
                # keep "unit_sales" vs "count" distinction:
                # registry says count; name-based inference decides "unit_sales" if it sees units/shipments/deliveries
                registry_dim_hint = "count"
            else:
                registry_dim_hint = ""
        else:
            registry_dim_hint = ""
        # =========================

        raw_unit = (metric.get("unit") or "").strip()

        # =========================
        # PATCH C2 (ADDITIVE): compute unit_tag/unit_family without changing existing unit behavior
        # - We keep your existing unit_norm logic for backwards compatibility.
        # - But we ALSO attach unit_tag + unit_family so downstream can gate deterministically.
        # =========================
        unit_tag = metric.get("unit_tag") or _safe_normalize_unit_tag(raw_unit)
        unit_family_tag = metric.get("unit_family") or _safe_unit_family(unit_tag)
        # =========================

        unit_norm = raw_unit.upper()  # keep original behavior (do not change)
        dim = infer_metric_dimension(str(original_name), raw_unit)

        # =========================
        # PATCH CM1 (ADDITIVE): apply registry hint as override / guardrail
        # - If registry says currency/percent, force that dimension.
        # - If registry says count, prevent accidental "currency"/"percent".
        # =========================
        if registry_dim_hint in ("currency", "percent"):
            dim = registry_dim_hint
        elif registry_dim_hint == "count":
            # Allow unit_sales if name clearly indicates it; else keep "count"
            if dim in ("currency", "percent"):
                dim = "count"
        # =========================

        canonical_key = f"{canonical_id}__{dim}"

        parsed_val = parse_to_float(metric.get("value"))
        value_for_sort = parsed_val if parsed_val is not None else str(metric.get("value", ""))

        stable_sort_key = (
            str(original_name).lower().strip(),
            dim,
            unit_norm,
            str(value_for_sort),
            str(key),
        )

        geo = infer_geo_scope(
            str(original_name),
            str(metric.get("context_snippet", "")),
            str(metric.get("source", "")),
            str(metric.get("source_url", "")),
        )

        proxy = infer_proxy_label(
            str(original_name),
            str(question_text),
            str(category_hint),
            str(metric.get("context_snippet", "")),
            str(metric.get("source", "")),
            str(metric.get("source_url", "")),
        )

        # =========================
        # PATCH C3 (ADDITIVE): canonicalize numeric fields on the candidate metric dict
        # - If canonicalize_numeric_candidate exists, it will attach:
        #   unit_tag/unit_family/base_unit/multiplier_to_base/value_norm
        # - If not, we attach minimal fields ourselves (still additive).
        # =========================
        metric_enriched = dict(metric)  # never mutate caller's dict
        try:
            fn_can = globals().get("canonicalize_numeric_candidate")
            if callable(fn_can):
                metric_enriched = fn_can(metric_enriched)
        except Exception:
            pass

        # Ensure minimal canonical fields exist (additive)
        metric_enriched.setdefault("unit_tag", unit_tag)
        metric_enriched.setdefault("unit_family", unit_family_tag)
        # =========================

        candidates.append({
            "canonical_id": canonical_id,
            "canonical_key": canonical_key,
            "canonical_name": display_name_for_dimension(canonical_name, dim),
            "original_name": original_name,

            # NOTE: store enriched metric
            "metric": metric_enriched,

            "unit": unit_norm,
            "parsed_val": parsed_val,
            "dimension": dim,
            "stable_sort_key": stable_sort_key,
            "geo_scope": geo["geo_scope"],
            "geo_name": geo["geo_name"],
            **proxy,
        })

    candidates.sort(key=lambda x: x["stable_sort_key"])

    grouped: Dict[str, List[Dict]] = {}
    for c in candidates:
        grouped.setdefault(c["canonical_key"], []).append(c)

    canonicalized: Dict[str, Dict] = {}

    for ckey, group in grouped.items():
        if len(group) == 1 or not merge_duplicates_to_range:
            g = group[0]
            m = g["metric"]

            # =========================
            # PATCH C4 (ADDITIVE): keep canonical numeric & semantic fields on output row
            # (only adds keys; does not remove/rename existing keys)
            # =========================
            out_row = {
                **m,
                "name": g["canonical_name"],
                "canonical_id": g["canonical_id"],
                "canonical_key": ckey,
                "dimension": g["dimension"],
                "original_name": g["original_name"],
                "geo_scope": g.get("geo_scope", "unknown"),
                "geo_name": g.get("geo_name", ""),
                "is_proxy": bool(g.get("is_proxy", False)),
                "proxy_type": g.get("proxy_type", ""),
                "proxy_reason": g.get("proxy_reason", ""),
                "proxy_confidence": float(g.get("proxy_confidence", 0.0) or 0.0),
                "proxy_target": g.get("proxy_target", ""),
            }
            # Ensure these exist if upstream provided them
            for k in ["anchor_hash", "source_url", "context_snippet", "measure_kind", "measure_assoc",
                      "unit_tag", "unit_family", "base_unit", "multiplier_to_base", "value_norm"]:
                if k in m and k not in out_row:
                    out_row[k] = m.get(k)
            canonicalized[ckey] = out_row
            # =========================
            continue

        # Merge duplicates within SAME dimension-safe canonical_key
        base = group[0]
        base_metric = dict(base["metric"])
        base_metric["name"] = base["canonical_name"]
        base_metric["canonical_id"] = base["canonical_id"]
        base_metric["canonical_key"] = ckey
        base_metric["dimension"] = base["dimension"]

        geo_scope, geo_name = merge_group_geo(group)
        base_metric["geo_scope"] = geo_scope
        base_metric["geo_name"] = geo_name

        merged_proxy = merge_group_proxy(group)
        base_metric.update(merged_proxy)

        vals = [g["parsed_val"] for g in group if g["parsed_val"] is not None]
        raw_vals = [str(g["metric"].get("value", "")) for g in group]
        orig_names = [g["original_name"] for g in group]

        units = [g["unit"] for g in group if g["unit"]]
        unit_base = units[0] if units else (base_metric.get("unit") or "")
        base_metric["unit"] = unit_base

        base_metric["original_names"] = orig_names
        base_metric["raw_values"] = raw_vals

        # =========================
        # PATCH C5 (ADDITIVE): optional canonical range using value_norm if present
        # - Keeps your existing "range" untouched.
        # - Adds "range_norm" when we can compute it.
        # =========================
        vals_norm = []
        for g in group:
            mm = g.get("metric") if isinstance(g, dict) else {}
            if isinstance(mm, dict) and mm.get("value_norm") is not None:
                try:
                    vals_norm.append(float(mm.get("value_norm")))
                except Exception:
                    pass
        # =========================

        if vals:
            vals_sorted = sorted(vals)
            vmin, vmax = vals_sorted[0], vals_sorted[-1]
            vmed = vals_sorted[len(vals_sorted) // 2]
            base_metric["value"] = vmed
            base_metric["range"] = {
                "min": vmin,
                "max": vmax,
                "candidates": vals_sorted,
                "n": len(vals_sorted),
            }
        else:
            base_metric["range"] = {"min": None, "max": None, "candidates": [], "n": 0}

        if len(vals_norm) >= 2:
            vn = sorted(vals_norm)
            base_metric["range_norm"] = {
                "min": vn[0],
                "max": vn[-1],
                "candidates": vn,
                "n": len(vn),
                "unit": base_metric.get("base_unit") or base_metric.get("unit") or "",
            }
        # =========================

        canonicalized[ckey] = base_metric

    return canonicalized



def freeze_metric_schema(canonical_metrics: Dict) -> Dict:
    """
    Lock metric identity + expected schema for future evolution.

    Key fix:
      - Stores canonical_key (dimension-safe)
      - Stores dimension + unit family
      - Keywords include dimension hints to improve later matching
    """
    frozen = {}
    if not isinstance(canonical_metrics, dict):
        return frozen

    # =========================
    # PATCH F1 (ADDITIVE): prefer shared normalize_unit_tag/unit_family helpers if present
    # This improves consistency with extractor + attribution gating.
    # Falls back safely to old heuristics.
    # =========================
    def _normalize_unit_safe(u: str) -> str:
        try:
            fn = globals().get("normalize_unit_tag")
            if callable(fn):
                return fn(u or "")
        except Exception:
            pass
        return (u or "").strip()

    def _unit_family_safe(unit_raw: str, dim_hint: str = "") -> str:
        # 1) dimension-first (strongest signal)
        d = (dim_hint or "").strip().lower()
        if d in ("percent", "pct"):
            return "percent"
        if d in ("currency",):
            return "currency"
        if d in ("energy",):
            return "energy"
        if d in ("unit_sales", "count"):
            # You’ve been treating M/B/T as “magnitude” for counts; keep aligned.
            return "magnitude"
        if d in ("index", "score"):
            return "index"

        # 2) if you already have a unit-family helper in the codebase, use it
        try:
            fn = globals().get("unit_family")
            if callable(fn):
                uf = fn(_normalize_unit_safe(unit_raw))
                if isinstance(uf, str) and uf.strip():
                    return uf.strip().lower()
        except Exception:
            pass

        # 3) fallback to old heuristic (your original logic)
        u = (unit_raw or "").strip().lower()
        if not u:
            return "unknown"
        if "%" in u:
            return "percent"
        if any(t in u for t in ["$", "s$", "usd", "sgd", "eur", "€", "gbp", "£", "jpy", "¥", "cny", "rmb"]):
            return "currency"
        if any(t in u for t in ["b", "bn", "billion", "m", "mn", "million", "k", "thousand", "t", "trillion"]):
            return "magnitude"
        return "other"
    # =========================

    for ckey, m in canonical_metrics.items():
        if not isinstance(m, dict):
            continue

        dim = (m.get("dimension") or "").strip() or "unknown"
        name = m.get("name")
        unit = (m.get("unit") or "").strip()

        # =========================
        # PATCH F2 (ADDITIVE): compute unit_family using dimension-first logic
        # =========================
        uf = _unit_family_safe(unit, dim_hint=dim)
        # =========================

        # Keywords: name + dimension token to prevent cross-dimension matches later
        kws = extract_context_keywords(name or "") or []
        if dim and dim not in kws:
            kws.append(dim)
        if uf and uf not in kws:
            kws.append(uf)

        # =========================
        # PATCH F3 (ADDITIVE): preserve schema unit more safely
        # - Keep your existing behavior in 'unit' (backward compatible),
        #   BUT also add 'unit_tag' which is the canonicalized unit used downstream.
        # - This avoids the "SGD -> S" schema corruption that breaks currency gating.
        # =========================
        unit_tag = _normalize_unit_safe(unit)
        # Keep existing 'unit' output to avoid breaking consumers:
        unit_out = unit_clean_first_letter(unit.upper())
        # =========================

        frozen[ckey] = {
            "canonical_key": ckey,
            "canonical_id": m.get("canonical_id") or ckey.split("__", 1)[0],
            "dimension": dim,
            "name": name,

            # Existing field kept exactly (backward compatible)
            "unit": unit_out,

            # =========================
            # PATCH F3 (ADDITIVE): extra stable fields (non-breaking additions)
            # =========================
            "unit_tag": unit_tag,          # e.g., "%", "M", "B", "TWh"
            "unit_family": uf,             # e.g., "currency", "percent", "magnitude"
            # =========================

            "keywords": kws[:30],
        }

    return frozen


# =========================================================
# RANGE + SOURCE ATTRIBUTION (DETERMINISTIC, NO LLM)
# =========================================================

def stable_json_hash(obj: Any) -> str:
    import hashlib, json
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def make_extracted_number_id(source_url: str, fingerprint: str, n: Dict) -> str:
    payload = {
        "url": source_url or "",
        "fp": fingerprint or "",
        "start": n.get("start_idx"),
        "end": n.get("end_idx"),
        "value": n.get("value"),
        "unit": normalize_unit(n.get("unit") or ""),
        "raw": n.get("raw") or "",
        "ctx": " ".join((n.get("context_snippet") or "").split())[:240],
    }
    return stable_json_hash(payload)

def sort_snapshot_numbers(numbers: List[Dict]) -> List[Dict]:
    """
    Deterministic ordering for extracted_numbers in snapshots.

    Backward compatible + robust:
      - Uses start/end idx when present
      - Avoids hard dependency on normalize_unit() (may not exist)
      - Falls back to normalize_unit_tag() if available
    """

    # =========================
    # PATCH SS1 (ADDITIVE): safe unit normalizer
    # - Prefer normalize_unit() if it exists
    # - Else fall back to normalize_unit_tag() if present
    # - Else just return stripped unit
    # =========================
    _norm_unit_fn = globals().get("normalize_unit")
    _norm_tag_fn = globals().get("normalize_unit_tag")

    def _safe_norm_unit(u: str) -> str:
        u = (u or "").strip()
        try:
            if callable(_norm_unit_fn):
                return str(_norm_unit_fn(u) or "")
        except Exception:
            pass
        try:
            if callable(_norm_tag_fn):
                # normalize_unit_tag expects tags / unit-ish strings; still better than raw
                return str(_norm_tag_fn(u) or "")
        except Exception:
            pass
        return u
    # =========================

    def k(n: Dict[str, Any]):
        n = n or {}
        return (
            n.get("start_idx") if isinstance(n.get("start_idx"), int) else 10**18,
            n.get("end_idx") if isinstance(n.get("end_idx"), int) else 10**18,

            # stable identity ordering
            str(n.get("anchor_hash") or ""),

            # unit + value
            _safe_norm_unit(str(n.get("unit") or "")),
            str(n.get("unit_tag") or ""),
            str(n.get("value_norm") if n.get("value_norm") is not None else n.get("value")),

            # final tie-breakers
            str(n.get("raw") or ""),
            str(n.get("context_snippet") or n.get("context") or "")[:80],
        )

    return sorted((numbers or []), key=k)

def sort_evidence_records(records: List[Dict]) -> List[Dict]:
    """
    Deterministic ordering for evidence_records.

    Backward compatible:
      - Uses url + fingerprint (as you had)
      - Adds fetched_at as tie-breaker if present (non-breaking)
    """

    # =========================
    # PATCH SE1 (ADDITIVE): add fetched_at tie-breaker (optional)
    # =========================
    def k(r: Dict[str, Any]):
        r = r or {}
        return (
            str(r.get("url") or ""),
            str(r.get("fingerprint") or ""),
            str(r.get("fetched_at") or ""),
        )
    # =========================

    return sorted((records or []), key=k)

def sort_metric_anchors(anchors: List[Dict]) -> List[Dict]:
    # =========================
    # PATCH MA2 (ADDITIVE): canonical-first stable sort
    # - Prefer canonical_key (new)
    # - Fall back to metric_id/metric_name (legacy)
    # =========================
    return sorted(
        (anchors or []),
        key=lambda a: (
            str((a or {}).get("canonical_key") or ""),
            str((a or {}).get("metric_id") or ""),
            str((a or {}).get("metric_name") or ""),
            str((a or {}).get("source_url") or ""),
        ),
    )


def normalize_unit(unit: str) -> str:
    """
    Deterministic unit normalizer used across analysis/evolution.

    Goals:
    - Preserve domain units like TWh/GWh/MWh/kWh (do NOT collapse to T/M/etc.)
    - Normalize magnitude suffixes case-insensitively: b/m/t/k -> B/M/T/K
    - Normalize percent consistently to "%"
    - Avoid clever heuristics; only normalize when confidently recognized
    """
    if not unit:
        return ""

    u0 = str(unit).strip()
    if not u0:
        return ""

    ul = u0.strip().lower().replace(" ", "")

    # --- Domain energy units (short-circuit, must be first) ---
    # Normalize casing to canonical display forms
    if "twh" == ul or ul.endswith("twh"):
        return "TWh"
    if "gwh" == ul or ul.endswith("gwh"):
        return "GWh"
    if "mwh" == ul or ul.endswith("mwh"):
        return "MWh"
    if "kwh" == ul or ul.endswith("kwh"):
        return "kWh"
    if ul == "wh":
        return "Wh"

    # --- Percent ---
    if ul in ("%", "percent", "pct"):
        return "%"

    # --- Currency prefixes/symbols (do not try to infer currency codes here) ---
    # Keep currency detection elsewhere; unit here is for magnitude tags.
    # If unit is literally "usd"/"$" etc, strip to empty.
    if ul in ("$", "usd", "sgd", "eur", "gbp", "aud", "cad", "jpy", "cny", "rmb"):
        return ""

    # --- Magnitude tags (case-insensitive) ---
    # IMPORTANT: handle single-letter forms used by extractor ("m", "b", "t", "k")
    if ul in ("trillion", "tn", "t"):
        return "T"
    if ul in ("billion", "bn", "b"):
        return "B"
    if ul in ("million", "mn", "mio", "m"):
        return "M"
    if ul in ("thousand", "k", "000"):
        return "K"

    # Unknown: return original trimmed (preserve domain-specific tokens)
    return u0.strip()




def to_billions(value: float, unit_tag: str) -> Optional[float]:
    """Convert T/B/M tagged values into billions. Leaves % unchanged (returns None for % here)."""
    try:
        v = float(value)
    except Exception:
        return None

    if unit_tag == "T":
        return v * 1000.0
    if unit_tag == "B":
        return v
    if unit_tag == "M":
        return v / 1000.0
    return None


def build_metric_keywords(metric_name: str) -> List[str]:
    """Reuse your existing keyword extractor, but ensure we always have something."""
    kws = extract_context_keywords(metric_name) or []
    # Add simple fallback tokens (deterministic)
    for t in re.findall(r"[a-zA-Z]{4,}", str(metric_name).lower()):
        if t not in kws:
            kws.append(t)
    return kws[:25]


def extract_numbers_from_scraped_sources(
    scraped_content: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Deterministically extract numeric candidates from all scraped source texts.
    Returns list of {url, value, unit_tag, raw, context}.
    """
    candidates: List[Dict[str, Any]] = []
    if not isinstance(scraped_content, dict):
        return candidates

    for url, content in scraped_content.items():
        if not content or not isinstance(content, str) or len(content) < 200:
            continue

        # =========================
        # PATCH 1 (ADDITIVE): pass source_url through (improves anchor stability)
        # =========================
        nums = extract_numbers_with_context(content, source_url=url)
        # =========================

        for n in nums:
            # =========================
            # PATCH 1 (ADDITIVE): prefer extractor-provided unit_tag if present; else normalize
            # =========================
            unit_tag = n.get("unit_tag")
            if not unit_tag:
                unit_tag = normalize_unit_tag(n.get("unit", ""))
            # =========================

            row = {
                "url": url,
                "value": n.get("value"),
                "unit_tag": unit_tag,
                "raw": n.get("raw", ""),
                "context": (n.get("context") or ""),
            }

            # =========================
            # PATCH 3 (ADDITIVE): preserve measure association tags if extractor provides them
            # =========================
            if "measure_kind" in n:
                row["measure_kind"] = n.get("measure_kind")
            if "measure_assoc" in n:
                row["measure_assoc"] = n.get("measure_assoc")
            # =========================

            # =========================
            # PATCH 1 (ADDITIVE): preserve extra fields if extractor provides them
            # (backwards compatible: we only add keys, never remove)
            # =========================
            for k in [
                "unit", "is_junk", "junk_reason", "anchor_hash",
                "start_idx", "end_idx", "context_snippet",
                "unit_family", "base_unit", "multiplier_to_base", "value_norm"
            ]:
                if k in n:
                    row[k] = n.get(k)
            # =========================

            # ============================================================
            # PATCH 9 (ADDITIVE): enforce canonical numeric fields uniformly
            # Why:
            #   - Some candidates may not carry unit_family/base_unit/value_norm yet
            #   - We want every candidate (analysis + evolution) to have the same
            #     canonical fields so diff + span logic is stable and drift-free.
            #
            # This is additive and safe to call multiple times.
            # ============================================================
            try:
                fn_can = globals().get("canonicalize_numeric_candidate")
                if callable(fn_can):
                    row = fn_can(row) or row
                else:
                    row = canonicalize_numeric_candidate(row) or row
            except Exception:
                pass

            # --- ADDITIVE: ensure canonical keys exist even if canonicalize failed ---
            row.setdefault("unit_family", unit_family(row.get("unit_tag", "") or ""))
            row.setdefault("base_unit", row.get("unit_tag", "") or "")
            row.setdefault("multiplier_to_base", 1.0)
            if row.get("value") is not None and row.get("value_norm") is None:
                try:
                    row["value_norm"] = float(row.get("value"))
                except Exception:
                    pass
            # ------------------------------------------------------------------------
            # ============================================================

            candidates.append(row)

    return candidates


def attribute_span_to_sources(
    metric_name: str,
    metric_unit: str,
    scraped_content: Dict[str, str],
    rel_tol: float = 0.08,
    # =========================
    # PATCH S1 (ADDITIVE): optional schema inputs (non-breaking)
    # - If provided, we enforce schema-first gating for drift stability.
    # - If not provided, we fall back to existing heuristic behavior.
    # =========================
    canonical_key: str = "",
    metric_schema: Dict[str, Any] = None,
    # =========================
) -> Dict[str, Any]:
    """
    Build a deterministic span (min/mid/max) for a metric, and attribute min/max to sources.
    Uses only scraped content + regex extractions (NO LLM).

    Schema-first behavior (when metric_schema/canonical_key provided):
      - Enforces unit_family and currency/count/percent gating from frozen schema
      - Uses measure_kind tags when available to avoid semantic leakage
      - Keeps deterministic tie-breaking
    """
    import re
    import hashlib

    unit_tag_hint = normalize_unit_tag(metric_unit)
    keywords = build_metric_keywords(metric_name)

    all_candidates = extract_numbers_from_scraped_sources(scraped_content)
    filtered: List[Dict[str, Any]] = []

    metric_l = (metric_name or "").lower()

    # =========================
    # PATCH S2 (ADDITIVE): resolve schema entry (if available)
    # =========================
    schema_entry = None
    if isinstance(metric_schema, dict) and canonical_key and isinstance(metric_schema.get(canonical_key), dict):
        schema_entry = metric_schema.get(canonical_key)
    # =========================

    # =========================
    # PATCH S3 (ADDITIVE): schema-derived expectations with safe fallbacks
    # =========================
    schema_unit_family = ""
    schema_dimension = ""
    schema_unit = ""
    if isinstance(schema_entry, dict):
        schema_unit_family = (schema_entry.get("unit_family") or "").strip().lower()
        schema_dimension = (schema_entry.get("dimension") or "").strip().lower()
        schema_unit = (schema_entry.get("unit") or "").strip()

    expected_family = ""
    if schema_unit_family in ("percent", "currency", "energy"):
        expected_family = schema_unit_family
    if not expected_family:
        ut = normalize_unit_tag(metric_unit)
        if ut == "%":
            expected_family = "percent"
        elif ut in ("TWh", "GWh", "MWh", "kWh", "Wh"):
            expected_family = "energy"
        else:
            expected_family = ""

    currencyish = False
    if schema_unit_family == "currency" or schema_dimension == "currency":
        currencyish = True
    if not currencyish:
        mu = (metric_unit or "").lower()
        if any(x in mu for x in ["usd", "sgd", "eur", "gbp", "$", "s$", "€", "£", "aud", "cad", "jpy", "cny", "rmb"]):
            currencyish = True
    if not currencyish and any(x in metric_l for x in ["revenue", "turnover", "valuation", "market value", "market size",
                                                       "profit", "earnings", "ebitda", "capex", "opex"]):
        currencyish = True
    # =========================

    # =========================
    # PATCH S4 (ADDITIVE): expected measure_kind (schema-first with fallback)
    # =========================
    expected_kind = None

    if expected_family == "percent":
        if any(k in metric_l for k in ["growth", "cagr", "increase", "decrease", "yoy", "qoq", "mom", "rate"]):
            expected_kind = "growth_pct"
        else:
            expected_kind = "share_pct"

    if currencyish:
        expected_kind = "money"

    if expected_kind is None and any(k in metric_l for k in [
        "units", "unit sales", "vehicle sales", "vehicles sold", "sold", "sales volume",
        "deliveries", "shipments", "registrations", "volume"
    ]):
        expected_kind = "count_units"
    # =========================

    # =========================
    # PATCH S5 (ADDITIVE): year-ish suppression helpers (unchanged behavior)
    # =========================
    metric_is_yearish = any(k in metric_l for k in ["year", "years", "fy", "fiscal", "calendar", "timeline", "target year"])

    def _looks_like_year_value(v) -> bool:
        try:
            iv = int(float(v))
            return 1900 <= iv <= 2099
        except Exception:
            return False

    def _ctx_has_year_range(ctx: str) -> bool:
        return bool(re.search(r"\b(19|20)\d{2}\s*(?:-|–|—|to)\s*(19|20)\d{2}\b", ctx or "", flags=re.I))
    # =========================

    # =========================
    # PATCH S6 (ADDITIVE): currency evidence check (used only when currencyish)
    # =========================
    def _has_currency_evidence(raw: str, ctx: str) -> bool:
        r = (raw or "")
        c = (ctx or "").lower()

        if any(s in r for s in ["$", "S$", "€", "£"]):
            return True
        if any(code in c for code in [" usd", "sgd", " eur", " gbp", " aud", " cad", " jpy", " cny", " rmb"]):
            return True

        strong_kw = [
            "revenue", "turnover", "valuation", "valued at", "market value", "market size",
            "sales value", "net profit", "operating profit", "gross profit",
            "ebitda", "earnings", "income", "capex", "opex"
        ]
        if any(k in c for k in strong_kw):
            return True
        return False
    # =========================

    # =========================================================================
    # PATCH S11 (ADDITIVE): deterministic candidate_id for tie-breaking
    # - Stable across runs, depends only on stable fields
    # - Used ONLY as final tie-breaker (won't change non-tie outcomes)
    # =========================================================================
    def _candidate_id(x: dict) -> str:
        try:
            url = str(x.get("url") or x.get("source_url") or "")
            ah = str(x.get("anchor_hash") or "")
            vn = x.get("value_norm")
            bu = str(x.get("base_unit") or x.get("unit") or x.get("unit_tag") or "")
            mk = str(x.get("measure_kind") or "")
            # normalize numeric string for stability
            vn_s = ""
            if vn is not None:
                try:
                    vn_s = f"{float(vn):.12g}"
                except Exception:
                    vn_s = str(vn)
            s = f"{url}|{ah}|{vn_s}|{bu}|{mk}"
            return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return ""
    # =========================================================================

    for c in all_candidates:
        ctx = c.get("context", "")
        if not ctx:
            continue

        if c.get("is_junk") is True:
            continue

        if not metric_is_yearish:
            if (c.get("unit_tag") in ("", None)) and _looks_like_year_value(c.get("value")):
                continue
            if _looks_like_year_value(c.get("value")) and _ctx_has_year_range(ctx):
                continue

        ctx_score = calculate_context_match(keywords, ctx)
        if ctx_score <= 0.0:
            continue

        cand_ut = c.get("unit_tag") or normalize_unit_tag(c.get("unit") or "")
        cand_fam = (c.get("unit_family") or unit_family(cand_ut) or "").strip().lower()

        if expected_family:
            if expected_family == "percent" and cand_fam != "percent":
                continue
            if expected_family == "currency":
                if cand_fam not in ("currency", "magnitude"):
                    continue
                if not _has_currency_evidence(c.get("raw", ""), ctx):
                    continue
            if expected_family == "energy" and cand_fam != "energy":
                continue

        if expected_kind:
            mk = c.get("measure_kind")
            if mk and mk != expected_kind:
                continue

        val_norm = None
        if expected_family == "percent" or unit_tag_hint == "%":
            if cand_ut != "%":
                continue
            val_norm = c.get("value")

        elif expected_family == "energy":
            val_norm = c.get("value_norm")
            if val_norm is None:
                val_norm = c.get("value")

        elif currencyish or expected_family == "currency":
            if c.get("measure_kind") == "count_units":
                continue
            if cand_ut not in ("T", "B", "M"):
                continue
            val_norm = to_billions(c.get("value"), cand_ut)
            if val_norm is None:
                continue

        else:
            try:
                val_norm = float(c.get("value"))
            except Exception:
                continue

        row = {
            **c,
            "unit_tag": cand_ut,
            "unit_family": cand_fam,
            "value_norm": val_norm,
            "ctx_score": float(ctx_score),
        }

        # =========================
        # PATCH S11 (ADDITIVE): attach candidate_id (safe extra field)
        # =========================
        row.setdefault("candidate_id", _candidate_id(row))
        # =========================

        filtered.append(row)

    if not filtered:
        return {
            "span": None,
            "source_attribution": None,
            "evidence": []
        }

    # Deterministic selection: value_norm then ctx_score then url then candidate_id
    # =========================================================================
    # PATCH S12 (ADDITIVE): candidate_id as final tie-breaker
    # =========================================================================
    def min_key(x):
        return (
            float(x["value_norm"]),
            -float(x["ctx_score"]),
            str(x.get("url", "")),
            str(x.get("candidate_id", "")),
        )

    def max_key(x):
        return (
            -float(x["value_norm"]),
            -float(x["ctx_score"]),
            str(x.get("url", "")),
            str(x.get("candidate_id", "")),
        )
    # =========================================================================

    min_item = sorted(filtered, key=min_key)[0]
    max_item = sorted(filtered, key=max_key)[0]

    vmin = float(min_item["value_norm"])
    vmax = float(max_item["value_norm"])
    vmid = (vmin + vmax) / 2.0

    if expected_family == "percent" or unit_tag_hint == "%":
        unit_out = "%"
    elif currencyish or expected_family == "currency":
        unit_out = "billion USD"
    elif expected_family == "energy":
        unit_out = "Wh"
    else:
        unit_out = metric_unit or (schema_unit or "")

    evidence = []
    for it in sorted(filtered, key=lambda x: (-float(x["ctx_score"]), str(x.get("url", "")), str(x.get("candidate_id", ""))))[:12]:
        evidence.append({
            "url": it.get("url"),
            "raw": it.get("raw"),
            "unit_tag": it.get("unit_tag"),
            "unit_family": it.get("unit_family"),
            "measure_kind": it.get("measure_kind"),
            "measure_assoc": it.get("measure_assoc"),
            "value_norm": it.get("value_norm"),
            "candidate_id": it.get("candidate_id"),  # PATCH S11: exposed for transparency
            "context_snippet": (it.get("context") or "")[:220],
            "context_score": round(float(it.get("ctx_score", 0.0)) * 100, 1),
        })

    return {
        "span": {
            "min": round(vmin, 4),
            "mid": round(vmid, 4),
            "max": round(vmax, 4),
            "unit": unit_out
        },
        "source_attribution": {
            "min": {
                "url": min_item.get("url"),
                "raw": min_item.get("raw"),
                "measure_kind": min_item.get("measure_kind"),
                "measure_assoc": min_item.get("measure_assoc"),
                "value_norm": min_item.get("value_norm"),
                "candidate_id": min_item.get("candidate_id"),  # PATCH S11
                "context_snippet": (min_item.get("context") or "")[:220],
                "context_score": round(float(min_item.get("ctx_score", 0.0)) * 100, 1),
            },
            "max": {
                "url": max_item.get("url"),
                "raw": max_item.get("raw"),
                "measure_kind": max_item.get("measure_kind"),
                "measure_assoc": max_item.get("measure_assoc"),
                "value_norm": max_item.get("value_norm"),
                "candidate_id": max_item.get("candidate_id"),  # PATCH S11
                "context_snippet": (max_item.get("context") or "")[:220],
                "context_score": round(float(max_item.get("ctx_score", 0.0)) * 100, 1),
            }
        },
        "evidence": evidence
    }



def add_range_and_source_attribution_to_canonical_metrics(
    canonical_metrics: Dict[str, Any],
    web_context: dict,
    # =========================
    # PATCH R1 (ADDITIVE): optional schema-first inputs
    # If provided, attribution uses frozen schema to avoid semantic/unit leakage.
    # =========================
    metric_schema: Dict[str, Any] = None,
    # =========================
) -> Dict[str, Any]:
    """
    Enrich canonical metrics with deterministic range + source attribution.

    IMPORTANT:
    - canonical_metrics is expected to be keyed by canonical_key (dimension-safe),
      i.e. the output of canonicalize_metrics().
    - Schema-first mode (recommended): pass metric_schema=metric_schema_frozen so
      attribute_span_to_sources() can enforce unit_family / measure_kind gates.
    - Backward compatible: if metric_schema not provided, attribution falls back
      to existing heuristic behavior inside attribute_span_to_sources().
    """
    enriched: Dict[str, Any] = {}
    if not isinstance(canonical_metrics, dict):
        return enriched

    scraped = (web_context or {}).get("scraped_content") or {}
    if not isinstance(scraped, dict):
        scraped = {}

    # =========================
    # PATCH R2 (ADDITIVE): resolve schema dict safely
    # =========================
    schema = metric_schema if isinstance(metric_schema, dict) else {}
    # =========================

    for ckey, m in canonical_metrics.items():
        if not isinstance(m, dict):
            continue

        metric_name = m.get("name") or m.get("original_name") or str(ckey)
        metric_unit = m.get("unit") or ""

        # =========================
        # PATCH R3 (BUGFIX): schema-first wiring (no undefined prev_response/ckey)
        # - canonical_key is the dict key (ckey)
        # - metric_schema is the frozen schema dict (if provided)
        # =========================
        span_pack = attribute_span_to_sources(
            metric_name=metric_name,
            metric_unit=metric_unit,
            scraped_content=scraped,
            canonical_key=str(ckey),
            metric_schema=schema,
        )
        # =========================

        mm = dict(m)

        # Preserve old behavior: only add keys (don’t remove anything)
        if isinstance(span_pack, dict):
            if span_pack.get("span") is not None:
                mm["source_span"] = span_pack.get("span")
            if span_pack.get("source_attribution") is not None:
                mm["source_attribution"] = span_pack.get("source_attribution")
            if span_pack.get("evidence") is not None:
                mm["evidence"] = span_pack.get("evidence")

        enriched[ckey] = mm

    return enriched





# ------------------------------------
# SEMANTIC FINDING HASH
# Removes wording-based churn from findings comparison
# ------------------------------------

# Semantic components to extract from findings
FINDING_PATTERNS = {
    # Growth/decline patterns
    "growth": [
        r'(?:grow(?:ing|th)?|increas(?:e|ing)|expand(?:ing)?|ris(?:e|ing)|up)\s*(?:by|at|of)?\s*(\d+(?:\.\d+)?)\s*%?',
        r'(\d+(?:\.\d+)?)\s*%?\s*(?:growth|increase|expansion|rise)',
    ],
    "decline": [
        r'(?:declin(?:e|ing)|decreas(?:e|ing)|fall(?:ing)?|drop(?:ping)?|down)\s*(?:by|at|of)?\s*(\d+(?:\.\d+)?)\s*%?',
        r'(\d+(?:\.\d+)?)\s*%?\s*(?:decline|decrease|drop|fall)',
    ],

    # Value patterns
    "value": [
        r'\$\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)?',
        r'(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)',
    ],

    # Ranking patterns
    "rank": [
        r'(?:lead(?:ing|er)?|top|first|largest|biggest|#1|number one)',
        r'(?:second|#2|runner.?up)',
        r'(?:third|#3)',
    ],

    # Trend patterns
    "trend_up": [
        r'(?:bullish|optimistic|positive|strong|robust|accelerat)',
    ],
    "trend_down": [
        r'(?:bearish|pessimistic|negative|weak|slow(?:ing)?|decelerat)',
    ],

    # Entity patterns (will be filled dynamically)
    "entities": []
}

# Common stop words to remove
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'also', 'now', 'new'
}


def extract_semantic_components(finding: str) -> Dict[str, Any]:
    """
    Extract semantic components from a finding.

    Example:
        "The market is growing at 15% annually" ->
        {
            "direction": "up",
            "percentage": 15.0,
            "subject": "market",
            "timeframe": "annual",
            "entities": [],
            "keywords": ["market", "growing", "annually"]
        }
    """
    if not finding:
        return {}

    finding_lower = finding.lower()
    components = {
        "direction": None,
        "percentage": None,
        "value": None,
        "value_unit": None,
        "subject": None,
        "timeframe": None,
        "entities": [],
        "keywords": []
    }

    # Extract direction
    for pattern in FINDING_PATTERNS["growth"]:
        match = re.search(pattern, finding_lower)
        if match:
            components["direction"] = "up"
            if match.groups():
                try:
                    components["percentage"] = float(match.group(1))
                except:
                    pass
            break

    if not components["direction"]:
        for pattern in FINDING_PATTERNS["decline"]:
            match = re.search(pattern, finding_lower)
            if match:
                components["direction"] = "down"
                if match.groups():
                    try:
                        components["percentage"] = float(match.group(1))
                    except:
                        pass
                break

    # Extract trend sentiment
    if not components["direction"]:
        for pattern in FINDING_PATTERNS["trend_up"]:
            if re.search(pattern, finding_lower):
                components["direction"] = "up"
                break
        for pattern in FINDING_PATTERNS["trend_down"]:
            if re.search(pattern, finding_lower):
                components["direction"] = "down"
                break

    # Extract value
    for pattern in FINDING_PATTERNS["value"]:
        match = re.search(pattern, finding_lower)
        if match:
            try:
                components["value"] = float(match.group(1))
                if len(match.groups()) > 1 and match.group(2):
                    components["value_unit"] = match.group(2)[0].upper()
            except:
                pass
            break

    # Extract timeframe
    timeframe_patterns = {
        "annual": r'\b(?:annual(?:ly)?|year(?:ly)?|per year|yoy|y-o-y)\b',
        "quarterly": r'\b(?:quarter(?:ly)?|q[1-4])\b',
        "monthly": r'\b(?:month(?:ly)?|per month)\b',
        "2024": r'\b2024\b',
        "2025": r'\b2025\b',
        "2026": r'\b2026\b',
        "2030": r'\b2030\b',
    }
    for tf_name, tf_pattern in timeframe_patterns.items():
        if re.search(tf_pattern, finding_lower):
            components["timeframe"] = tf_name
            break

    # Extract subject keywords
    words = re.findall(r'\b[a-z]{3,}\b', finding_lower)
    keywords = [w for w in words if w not in STOP_WORDS]
    components["keywords"] = keywords[:10]  # Limit to top 10

    # Identify likely subject
    subject_candidates = ["market", "industry", "sector", "segment", "revenue", "sales", "demand", "supply"]
    for word in keywords:
        if word in subject_candidates:
            components["subject"] = word
            break

    return components


def compute_semantic_hash(finding: str) -> str:
    """
    Compute a semantic hash for a finding that's invariant to wording changes.

    Two findings with the same meaning should produce the same hash.

    Example:
        "The market is growing at 15% annually" -> "up_15.0_market_annual"
        "Annual growth rate stands at 15%" -> "up_15.0_market_annual"
    """
    components = extract_semantic_components(finding)

    # Build hash components in consistent order
    hash_parts = []

    # Direction
    if components.get("direction"):
        hash_parts.append(components["direction"])

    # Percentage (rounded to avoid float issues)
    if components.get("percentage") is not None:
        hash_parts.append(f"{components['percentage']:.1f}")

    # Value with unit
    if components.get("value") is not None:
        val_str = f"{components['value']:.1f}"
        if components.get("value_unit"):
            val_str += components["value_unit"]
        hash_parts.append(val_str)

    # Subject
    if components.get("subject"):
        hash_parts.append(components["subject"])

    # Timeframe
    if components.get("timeframe"):
        hash_parts.append(components["timeframe"])

    # If we have enough components, use them for hash
    if len(hash_parts) >= 2:
        return "_".join(hash_parts)

    # Fallback: use sorted keywords
    keywords = sorted(components.get("keywords", []))[:5]
    if keywords:
        return "_".join(keywords)

    # Last resort: normalized text hash
    normalized = re.sub(r'\s+', ' ', finding.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def compute_semantic_finding_diffs(old_findings: List[str], new_findings: List[str]) -> List[FindingDiff]:
    """
    Compute finding diffs using semantic hashing instead of text similarity.

    This ensures that findings with the same meaning but different wording
    are recognized as the same finding.
    """
    diffs = []
    matched_new_indices = set()

    # Compute hashes for all findings
    old_hashes = [(f, compute_semantic_hash(f), extract_semantic_components(f)) for f in old_findings if f]
    new_hashes = [(f, compute_semantic_hash(f), extract_semantic_components(f)) for f in new_findings if f]

    # Match by semantic hash
    for old_text, old_hash, old_components in old_hashes:
        best_match_idx = None
        best_match_score = 0

        for i, (new_text, new_hash, new_components) in enumerate(new_hashes):
            if i in matched_new_indices:
                continue

            # Exact hash match = same finding
            if old_hash == new_hash:
                best_match_idx = i
                best_match_score = 100
                break

            # Component-based similarity
            score = compute_component_similarity(old_components, new_components)
            if score > best_match_score:
                best_match_score = score
                best_match_idx = i

        if best_match_idx is not None and best_match_score >= 60:
            matched_new_indices.add(best_match_idx)
            new_text = new_hashes[best_match_idx][0]

            if best_match_score >= 90:
                change_type = 'retained'
            else:
                change_type = 'modified'

            diffs.append(FindingDiff(
                old_text=old_text,
                new_text=new_text,
                similarity=best_match_score,
                change_type=change_type
            ))
        else:
            # Finding removed
            diffs.append(FindingDiff(
                old_text=old_text,
                new_text=None,
                similarity=0,
                change_type='removed'
            ))

    # Find added findings
    for i, (new_text, new_hash, new_components) in enumerate(new_hashes):
        if i not in matched_new_indices:
            diffs.append(FindingDiff(
                old_text=None,
                new_text=new_text,
                similarity=0,
                change_type='added'
            ))

    return diffs


def compute_component_similarity(comp1: Dict, comp2: Dict) -> float:
    """
    Compute similarity between two finding component dictionaries.
    Returns a score from 0-100.
    """
    if not comp1 or not comp2:
        return 0

    score = 0
    weights = {
        "direction": 25,
        "percentage": 25,
        "value": 20,
        "subject": 15,
        "timeframe": 10,
        "keywords": 5
    }

    # Direction match
    if comp1.get("direction") and comp2.get("direction"):
        if comp1["direction"] == comp2["direction"]:
            score += weights["direction"]
    elif not comp1.get("direction") and not comp2.get("direction"):
        score += weights["direction"] * 0.5  # Both neutral

    # Percentage match (within 2% tolerance)
    if comp1.get("percentage") is not None and comp2.get("percentage") is not None:
        diff = abs(comp1["percentage"] - comp2["percentage"])
        if diff <= 2:
            score += weights["percentage"]
        elif diff <= 5:
            score += weights["percentage"] * 0.5

    # Value match (within 10% tolerance)
    if comp1.get("value") is not None and comp2.get("value") is not None:
        v1, v2 = comp1["value"], comp2["value"]
        # Normalize by unit
        if comp1.get("value_unit") == comp2.get("value_unit"):
            if v1 > 0 and v2 > 0:
                ratio = min(v1, v2) / max(v1, v2)
                if ratio >= 0.9:
                    score += weights["value"]
                elif ratio >= 0.8:
                    score += weights["value"] * 0.5

    # Subject match
    if comp1.get("subject") and comp2.get("subject"):
        if comp1["subject"] == comp2["subject"]:
            score += weights["subject"]

    # Timeframe match
    if comp1.get("timeframe") and comp2.get("timeframe"):
        if comp1["timeframe"] == comp2["timeframe"]:
            score += weights["timeframe"]

    # Keyword overlap
    kw1 = set(comp1.get("keywords", []))
    kw2 = set(comp2.get("keywords", []))
    if kw1 and kw2:
        overlap = len(kw1 & kw2) / len(kw1 | kw2)
        score += weights["keywords"] * overlap

    return score


# ------------------------------------
# UPDATED METRIC DIFF COMPUTATION
# Using canonical IDs
# ------------------------------------

def compute_metric_diffs_canonical(old_metrics: Dict, new_metrics: Dict) -> List[MetricDiff]:
    """
    Compute metric diffs using canonical IDs for stable matching.
    Range-aware via get_metric_value_span + spans_overlap.
    """
    diffs: List[MetricDiff] = []

    old_canonical = canonicalize_metrics(old_metrics)
    new_canonical = canonicalize_metrics(new_metrics)

    matched_new_ids = set()

    # Match by canonical ID
    for old_id, old_m in old_canonical.items():
        old_name = old_m.get("name", old_id)

        old_span = get_metric_value_span(old_m)
        old_raw = str(old_m.get("value", ""))
        old_unit = old_span.get("unit") or old_m.get("unit", "")
        old_val = old_span.get("mid")

        # -------------------------
        # Direct canonical ID match
        # -------------------------
        if old_id in new_canonical:
            new_m = new_canonical[old_id]
            matched_new_ids.add(old_id)

            new_raw = str(new_m.get("value", ""))
            new_span = get_metric_value_span(new_m)
            new_val = new_span.get("mid")
            new_unit = new_span.get("unit") or new_m.get("unit", old_unit)

            if spans_overlap(old_span, new_span, rel_tol=0.05):
                change_pct = 0.0
                change_type = "unchanged"
            else:
                change_pct = compute_percent_change(old_val, new_val)
                if change_pct is None or abs(change_pct) < 0.5:
                    change_type = "unchanged"
                elif change_pct > 0:
                    change_type = "increased"
                else:
                    change_type = "decreased"

            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=new_val,
                old_raw=old_raw,
                new_raw=new_raw,
                unit=new_unit or old_unit,
                change_pct=change_pct,
                change_type=change_type
            ))
            continue  # important: don't fall into base-ID matching

        # -------------------------
        # Base ID match (strip years)
        # -------------------------
        base_id = re.sub(r'_\d{4}(?:_\d{4})*$', '', old_id)
        found = False

        for new_id, new_m in new_canonical.items():
            if new_id in matched_new_ids:
                continue

            new_base_id = re.sub(r'_\d{4}(?:_\d{4})*$', '', new_id)
            if base_id != new_base_id:
                continue

            matched_new_ids.add(new_id)
            found = True

            new_raw = str(new_m.get("value", ""))
            new_span = get_metric_value_span(new_m)
            new_val = new_span.get("mid")
            new_unit = new_span.get("unit") or new_m.get("unit", old_unit)

            if spans_overlap(old_span, new_span, rel_tol=0.05):
                change_pct = 0.0
                change_type = "unchanged"
            else:
                change_pct = compute_percent_change(old_val, new_val)
                if change_pct is None or abs(change_pct) < 0.5:
                    change_type = "unchanged"
                elif change_pct > 0:
                    change_type = "increased"
                else:
                    change_type = "decreased"

            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=new_val,
                old_raw=old_raw,
                new_raw=new_raw,
                unit=new_unit or old_unit,
                change_pct=change_pct,
                change_type=change_type
            ))
            break

        if not found:
            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=None,
                old_raw=old_raw,
                new_raw="",
                unit=old_unit,
                change_pct=None,
                change_type="removed"
            ))

    # Added metrics
    for new_id, new_m in new_canonical.items():
        if new_id in matched_new_ids:
            continue

        new_name = new_m.get("name", new_id)
        new_raw = str(new_m.get("value", ""))
        new_span = get_metric_value_span(new_m)
        new_val = new_span.get("mid")
        new_unit = new_span.get("unit") or new_m.get("unit", "")

        diffs.append(MetricDiff(
            name=new_name,
            old_value=None,
            new_value=new_val,
            old_raw="",
            new_raw=new_raw,
            unit=new_unit,
            change_pct=None,
            change_type="added"
        ))

    return diffs


# ------------------------------------
# NUMERIC PARSING (DETERMINISTIC)
# ------------------------------------

def parse_to_float(value: Any) -> Optional[float]:
    """
    Deterministically parse any value to float.
    Returns None if unparseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    # Clean string
    cleaned = value.strip().upper()
    cleaned = re.sub(r'[,$]', '', cleaned)

    # Handle empty/NA
    if cleaned in ['', 'N/A', 'NA', 'NULL', 'NONE', '-', '—']:
        return None

    # Extract multiplier
    multiplier = 1.0
    if 'TRILLION' in cleaned or cleaned.endswith('T'):
        multiplier = 1_000_000
        cleaned = re.sub(r'T(?:RILLION)?', '', cleaned)
    elif 'BILLION' in cleaned or cleaned.endswith('B'):
        multiplier = 1_000
        cleaned = re.sub(r'B(?:ILLION)?', '', cleaned)
    elif 'MILLION' in cleaned or cleaned.endswith('M'):
        multiplier = 1
        cleaned = re.sub(r'M(?:ILLION)?', '', cleaned)
    elif 'THOUSAND' in cleaned or cleaned.endswith('K'):
        multiplier = 0.001
        cleaned = re.sub(r'K(?:THOUSAND)?', '', cleaned)

    # Handle percentages
    if '%' in cleaned:
        cleaned = cleaned.replace('%', '')
        # Don't apply multiplier to percentages
        multiplier = 1.0

    try:
        return float(cleaned.strip()) * multiplier
    except (ValueError, TypeError):
        return None

def get_metric_value_span(metric: Dict) -> Dict[str, Any]:
    """
    Return a numeric span for a metric to support range-aware canonical metrics.

    Output:
      {
        "min": float|None,
        "max": float|None,
        "mid": float|None,   # representative value (median if range, else parsed value)
        "unit": str,         # normalized (upper/stripped), preserves %/$ units if present
        "is_range": bool
      }
    """
    if not isinstance(metric, dict):
        return {"min": None, "max": None, "mid": None, "unit": "", "is_range": False}

    unit = (metric.get("unit") or "").strip()

    # If metric already has a range object, prefer it
    r = metric.get("range")
    if isinstance(r, dict):
        vmin = r.get("min")
        vmax = r.get("max")
        # ensure numeric
        try:
            vmin = float(vmin) if vmin is not None else None
        except Exception:
            vmin = None
        try:
            vmax = float(vmax) if vmax is not None else None
        except Exception:
            vmax = None

        # Representative = median of candidates if provided, else midpoint of min/max
        candidates = r.get("candidates")
        nums = []
        if isinstance(candidates, list):
            for c in candidates:
                try:
                    nums.append(float(c))
                except Exception:
                    pass
        if nums:
            nums_sorted = sorted(nums)
            mid = nums_sorted[len(nums_sorted) // 2]
        else:
            mid = None
            if vmin is not None and vmax is not None:
                mid = (vmin + vmax) / 2.0
            elif vmin is not None:
                mid = vmin
            elif vmax is not None:
                mid = vmax

        return {
            "min": vmin,
            "max": vmax,
            "mid": mid,
            "unit": unit,
            "is_range": True
        }

    # Non-range metric: parse single numeric value
    val = parse_to_float(metric.get("value"))
    return {
        "min": val,
        "max": val,
        "mid": val,
        "unit": unit,
        "is_range": False
    }


def spans_overlap(a: Dict[str, Any], b: Dict[str, Any], rel_tol: float = 0.05) -> bool:
    """
    Decide whether two spans overlap "enough" to be considered stable.
    rel_tol provides a small widening to avoid false drift from rounding.
    """
    if not a or not b:
        return False
    a_min, a_max = a.get("min"), a.get("max")
    b_min, b_max = b.get("min"), b.get("max")

    if a_min is None or a_max is None or b_min is None or b_max is None:
        return False

    # Widen spans slightly
    a_pad = max(abs(a_max), abs(a_min), 1.0) * rel_tol
    b_pad = max(abs(b_max), abs(b_min), 1.0) * rel_tol

    a_min2, a_max2 = a_min - a_pad, a_max + a_pad
    b_min2, b_max2 = b_min - b_pad, b_max + b_pad

    return not (a_max2 < b_min2 or b_max2 < a_min2)


def compute_percent_change(old_val: Optional[float], new_val: Optional[float]) -> Optional[float]:
    """
    Compute percent change. Returns None if either value is None or old is 0.
    """
    if old_val is None or new_val is None:
        return None
    if old_val == 0:
        return None if new_val == 0 else float('inf')
    return round(((new_val - old_val) / abs(old_val)) * 100, 2)

# ------------------------------------
# NAME MATCHING (DETERMINISTIC)
# ------------------------------------

def normalize_name(name: str) -> str:
    """Normalize name for matching"""
    if not name:
        return ""
    n = name.lower().strip()
    n = re.sub(r'[^\w\s]', '', n)
    n = re.sub(r'\s+', ' ', n)
    return n

def name_similarity(name1: str, name2: str) -> float:
    """Compute similarity ratio between two names (0-1)"""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    # Check containment
    if n1 in n2 or n2 in n1:
        return 0.9
    # Sequence matcher
    return difflib.SequenceMatcher(None, n1, n2).ratio()

def find_best_match(name: str, candidates: List[str], threshold: float = 0.7) -> Optional[str]:
    """Find best matching name from candidates"""
    best_match = None
    best_score = threshold
    for candidate in candidates:
        score = name_similarity(name, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate
    return best_match

# =========================================================
# DETERMINISTIC QUERY STRUCTURE EXTRACTION
# - Classify query into a known category (country / industry / etc.)
# - Extract main question + "side questions" deterministically
# - Optional: spaCy dependency parse (if installed)
# - Optional: embedding similarity (if sentence-transformers/sklearn installed)
# =========================================================

SIDE_CONNECTOR_PATTERNS = [
    r"\bimpact of\b",
    r"\beffect of\b",
    r"\binfluence of\b",
    r"\brole of\b",
    r"\bdriven by\b",
    r"\bcaused by\b",
    r"\bdue to\b",
    r"\bincluding\b",
    r"\bincluding but not limited to\b",
    r"\bwith a focus on\b",
    r"\bespecially\b",
    r"\bnotably\b",
    r"\bplus\b",
    r"\bas well as\b",
    r"\band also\b",
    r"\bvs\b",
    r"\bversus\b",
]

QUESTION_CATEGORIES = {
    "country": {
        "signals": [
            "gdp", "gdp per capita", "population", "inflation", "interest rate",
            "exports", "imports", "trade balance", "currency", "fx", "central bank",
            "unemployment", "fiscal", "budget", "debt", "sovereign", "country"
        ],
        "template_sections": [
            "GDP & GDP per capita", "Growth rates", "Population & demographics",
            "Key industries", "Exports & imports", "Currency & FX trends",
            "Interest rates & inflation", "Risks & outlook"
        ],
    },
    "industry": {
        "signals": [
            "market size", "tam", "cagr", "industry", "sector", "market",
            "key players", "competitive landscape", "drivers", "challenges",
            "regulation", "technology trends", "forecast"
        ],
        "template_sections": [
            "Total Addressable Market (TAM) / Market size", "Growth rates (CAGR/YoY)",
            "Key players", "Key drivers", "Challenges & risks",
            "Technology trends", "Regulatory / environmental factors", "Outlook"
        ],
    },
    "company": {
        "signals": [
            "revenue", "earnings", "profit", "margins", "guidance",
            "business model", "segments", "customers", "competitors",
            "valuation", "multiple", "pe ratio", "cash flow"
        ],
        "template_sections": [
            "Business overview", "Revenue / profitability", "Segments",
            "Competitive position", "Key risks", "Guidance / outlook"
        ],
    },
    "unknown": {
        "signals": [],
        "template_sections": [],
    }
}

def _normalize_q(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def _cleanup_clause(text: str) -> str:
    t = _normalize_q(text)
    t = re.sub(r"^[,;:\-\s]+", "", t)
    t = re.sub(r"[,;:\-\s]+$", "", t)
    return t

def detect_query_category(query: str) -> Dict[str, Any]:
    """
    Deterministically classify query category using keyword signals.
    Returns: {"category": "...", "confidence": 0-1, "matched_signals": [...]}
    """
    q = (query or "").lower()
    best_cat = "unknown"
    best_hits = 0
    best_matched = []

    for cat, cfg in QUESTION_CATEGORIES.items():
        if cat == "unknown":
            continue
        matched = [s for s in cfg["signals"] if s in q]
        hits = len(matched)
        if hits > best_hits:
            best_hits = hits
            best_cat = cat
            best_matched = matched

    # simple confidence: saturate after ~6 hits
    conf = min(best_hits / 6.0, 1.0) if best_hits > 0 else 0.0
    return {"category": best_cat, "confidence": round(conf, 2), "matched_signals": best_matched[:8]}

# =========================================================
# 3A+. LAYERED QUERY STRUCTURE PARSER (Deterministic -> NLP -> Embeddings -> LLM fallback)
# =========================================================

_QUERY_SPLIT_PATTERNS = [
    r"\bas well as\b",
    r"\balong with\b",
    r"\bin addition to\b",
    r"\band\b",
    r"\bplus\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r",",
    r";",
]

_COUNTRY_OVERVIEW_SIGNALS = [
    "in general", "overview", "tell me about", "general", "profile", "facts about",
    "economy", "population", "gdp", "currency", "exports", "imports",
]

def _normalize_q(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q

def _split_clauses_deterministic(q: str) -> List[str]:
    """
    Deterministically split a question into ordered clauses.

    Supports:
    - comma/connector splits (",", "and", "as well as", "in addition to", etc.)
    - multi-side enumerations like:
        "in addition to: 1. X 2. Y"
        "including: (1) X (2) Y"
        "as well as: • X • Y"
    """
    if not isinstance(q, str):
        return []

    s = q.strip()
    if not s:
        return []

    # Normalize whitespace early (keep original casing if present; upstream may lowercase already)
    s = re.sub(r"\s+", " ", s).strip()

    # --- Step A: If there's an enumeration intro, split head vs tail ---
    # Examples: "in addition to:", "including:", "plus:", "as well as:"
    enum_intro = re.search(
        r"\b(in addition to|in addition|including|in addition to the following|as well as|plus)\b\s*:?\s*",
        s,
        flags=re.IGNORECASE,
    )

    head = s
    tail = ""

    if enum_intro:
        # Split at the FIRST occurrence of the enum phrase
        idx = enum_intro.start()
        # head is everything before the phrase if it exists, otherwise keep whole string
        # but we usually want "Tell me about X in general" to remain in head.
        head = s[:idx].strip().rstrip(",")
        tail = s[enum_intro.end():].strip()

        # If head is empty (e.g., query begins with "In addition to:"), treat everything as head
        if not head:
            head = s
            tail = ""

    clauses: List[str] = []

    # --- Step B: Split head using your existing connector patterns ---
    if head:
        parts = [head]
        for pat in _QUERY_SPLIT_PATTERNS:
            next_parts = []
            for p in parts:
                next_parts.extend(re.split(pat, p, flags=re.IGNORECASE))
            parts = next_parts

        for p in parts:
            p = p.strip(" ,;:.").strip()
            if p:
                clauses.append(p)

    # --- Step C: If tail exists, split as enumerated items/bullets ---
    if tail:
        # Split on "1.", "1)", "(1)", "•", "-", "*"
        # Keep it robust: find item starts, then slice.
        item_start = re.compile(r"(?:^|\s)(?:\(?\d+\)?[\.\)]|[•\-\*])\s+", flags=re.IGNORECASE)
        starts = [m.start() for m in item_start.finditer(tail)]

        if starts:
            # Build slices using detected starts
            spans = []
            for i, st0 in enumerate(starts):
                st = st0
                # Move start to the start of token (strip leading whitespace)
                while st < len(tail) and tail[st].isspace():
                    st += 1
                en = starts[i + 1] if i + 1 < len(starts) else len(tail)
                spans.append((st, en))

            for st, en in spans:
                item = tail[st:en].strip(" ,;:.").strip()
                # Remove the leading bullet/number token again (safety)
                item = re.sub(r"^\(?\d+\)?[\.\)]\s+", "", item)
                item = re.sub(r"^[•\-\*]\s+", "", item)
                item = item.strip(" ,;:.").strip()
                if item:
                    clauses.append(item)
        else:
            # If tail doesn't look enumerated, fall back to normal splitter on tail
            parts = [tail]
            for pat in _QUERY_SPLIT_PATTERNS:
                next_parts = []
                for p in parts:
                    next_parts.extend(re.split(pat, p, flags=re.IGNORECASE))
                parts = next_parts

            for p in parts:
                p = p.strip(" ,;:.").strip()
                if p:
                    clauses.append(p)

    # Final cleanup + dedupe while preserving order
    out: List[str] = []
    seen = set()
    for c in clauses:
        c2 = c.strip()
        if not c2:
            continue
        if c2.lower() in seen:
            continue
        seen.add(c2.lower())
        out.append(c2)

    return out



def _dedupe_clauses(clauses: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in clauses:
        c2 = c.strip().lower()
        if not c2 or c2 in seen:
            continue
        seen.add(c2)
        out.append(c.strip())
    return out

def _choose_main_and_side(clauses: List[str]) -> Tuple[str, List[str]]:
    """
    Pick 'main' as the first clause; side = remainder.
    Deterministic, stable across runs.
    """
    clauses = _dedupe_clauses(clauses)
    if not clauses:
        return "", []
    main = clauses[0]
    side = clauses[1:]
    return main, side

def _try_spacy_nlp():
    """
    Optional NLP layer. If spaCy is installed, use it; otherwise return None.
    """
    try:
        import spacy  # type: ignore
        # Avoid heavy model loading; prefer blank model with sentencizer if no model available.
        try:
            nlp = spacy.load("en_core_web_sm")  # common if installed
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        return None

def _nlp_refine_clauses(query: str, clauses: List[str]) -> Dict[str, Any]:
    """
    Use dependency/NER cues to:
      - detect country-overview questions
      - improve main-vs-side decision (coordination / 'as well as' patterns)
    Returns partial overrides: {"main":..., "side":[...], "hints":{...}}
    """
    nlp = _try_spacy_nlp()
    if not nlp:
        return {"hints": {"nlp_used": False}}

    doc = nlp(_normalize_q(query))
    # Named entities that look like places
    gpes = [ent.text for ent in getattr(doc, "ents", []) if ent.label_ in ("GPE", "LOC")]
    gpes_norm = [g.strip() for g in gpes if g and len(g.strip()) > 1]

    # Coordination hint: if query has "as well as" or "and", keep deterministic split,
    # but try to pick the more "general" clause as main when overview signals exist.
    overview_hit = any(sig in (query or "").lower() for sig in _COUNTRY_OVERVIEW_SIGNALS)
    hints = {
        "nlp_used": True,
        "gpe_entities": gpes_norm[:5],
        "overview_signal_hit": bool(overview_hit),
    }

    main, side = _choose_main_and_side(clauses)

    # If overview signals + place entity present, bias main to the overview clause
    if overview_hit and gpes_norm:
        # choose clause with strongest overview signal density
        def score_clause(c: str) -> int:
            c = c.lower()
            return sum(1 for sig in _COUNTRY_OVERVIEW_SIGNALS if sig in c)
        scored = sorted([(score_clause(c), c) for c in clauses], key=lambda x: (-x[0], x[1]))
        if scored and scored[0][0] > 0:
            main = scored[0][1]
            side = [c for c in clauses if c != main]

    return {"main": main, "side": side, "hints": hints}

def _embedding_category_vote(query: str) -> Dict[str, Any]:
    """
    Deterministic 'embedding-like' similarity using TF-IDF (no external model downloads).
    Produces a category suggestion + confidence based on similarity to category descriptors.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    except Exception:
        return {"category": "unknown", "confidence": 0.0, "method": "tfidf_unavailable"}

    q = _normalize_q(query).lower()
    if not q:
        return {"category": "unknown", "confidence": 0.0, "method": "tfidf_empty"}

    # Build deterministic descriptors from your registry
    cat_texts = []
    cat_names = []
    for cat, cfg in (QUESTION_CATEGORIES or {}).items():
        if not isinstance(cfg, dict) or cat == "unknown":
            continue
        signals = " ".join(cfg.get("signals", [])[:50])
        sections = " ".join((cfg.get("template_sections", []) or [])[:50])
        descriptor = f"{cat} {signals} {sections}".strip()
        if descriptor:
            cat_names.append(cat)
            cat_texts.append(descriptor)

    if not cat_texts:
        return {"category": "unknown", "confidence": 0.0, "method": "tfidf_no_registry"}

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)
    X = vec.fit_transform(cat_texts + [q])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()

    best_idx = int(sims.argmax()) if sims.size else 0
    best_sim = float(sims[best_idx]) if sims.size else 0.0
    best_cat = cat_names[best_idx] if cat_names else "unknown"

    # Map cosine similarity (~0-1) into a conservative confidence
    conf = max(0.0, min(best_sim / 0.35, 1.0))  # 0.35 sim ~= "high"
    return {"category": best_cat, "confidence": round(conf, 2), "method": "tfidf"}

def _llm_fallback_query_structure(query: str, web_context: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Last resort: ask LLM to output ONLY a small JSON query-structure object.
    Guardrail: do NOT let the LLM invent extra side questions unless the user explicitly enumerated them.
    This path must NOT validate against LLMResponse.
    """
    try:
        q = str(query or "").strip()
        if not q:
            return None

        # --- Detect explicit enumeration / list structure in the USER query ---
        # If the user wrote a list (1., 2), bullets, etc.), it's reasonable to accept multiple sides.
        enum_patterns = [
            r"(^|\n)\s*\d+\s*[\.\)]\s+",     # 1.  / 2)
            r"(^|\n)\s*[-•*]\s+",           # - item / • item
            r"(^|\n)\s*[a-zA-Z]\s*[\.\)]\s+"  # a) / b. etc.
        ]
        has_explicit_enumeration = any(re.search(p, q, flags=re.MULTILINE) for p in enum_patterns)

        # Deterministic baseline (what the system already extracted)
        # We use this to clamp LLM hallucinations.
        det_clauses = _split_clauses_deterministic(_normalize_q(q))
        det_main, det_side = _choose_main_and_side(det_clauses)
        det_side = _dedupe_clauses([s.strip() for s in (det_side or []) if isinstance(s, str) and s.strip()])

        prompt = (
            "Extract a query structure.\n"
            "Return ONLY valid JSON with keys:\n"
            "  category: one of [country, industry, company, finance, market, unknown]\n"
            "  category_confidence: number 0-1\n"
            "  main: string (the main question/topic)\n"
            "  side: array of strings (side questions)\n"
            "No extra keys, no commentary.\n\n"
            f"Query: {q}"
        )

        raw = query_perplexity_raw(prompt, max_tokens=250, timeout=30)

        # Parse
        if isinstance(raw, dict):
            parsed = raw
        else:
            if raw is None:
                raw = ""
            if not isinstance(raw, str):
                raw = str(raw)
            parsed = parse_json_safely(raw, "LLM Query Structure")

        if not isinstance(parsed, dict) or parsed.get("main") is None:
            return None

        # --- Clean/normalize fields ---
        llm_main = str(parsed.get("main") or "").strip()
        llm_side = parsed.get("side") if isinstance(parsed.get("side"), list) else []
        llm_side = [str(s).strip() for s in llm_side if s is not None and str(s).strip()]

        # Reject "invented" side items that look like generic outline bullets
        # (Only apply this rejection when the user did NOT explicitly enumerate a list.)
        def _looks_like_outline_item(s: str) -> bool:
            s2 = s.lower().strip()
            bad_starts = (
                "overview", "key", "key stats", "statistics", "major statistics",
                "policies", "infrastructure", "recent trends", "post-covid", "covid",
                "challenges", "opportunities", "drivers", "headwinds",
                "background", "introduction"
            )
            return any(s2.startswith(b) for b in bad_starts)

        # --- Guardrail policy ---
        # If user didn't enumerate, do NOT accept LLM expansion of side questions.
        if not has_explicit_enumeration:
            # Keep deterministic sides only. (You can allow 1 LLM side if deterministic found none.)
            final_side = det_side
            if not final_side and llm_side:
                # Allow at most one side item as a fallback, but avoid outline-like additions.
                cand = llm_side[0]
                final_side = [] if _looks_like_outline_item(cand) else [cand]
        else:
            # User enumerated: accept multiple sides, but still de-dupe and keep deterministic items first
            merged = []
            for s in (det_side + llm_side):
                s = str(s).strip()
                if not s:
                    continue
                if s not in merged:
                    merged.append(s)
            final_side = merged

        # If LLM main is empty or fragment-y, keep deterministic main
        bad_prefixes = ("as well as", "as well", "and ", "also ", "plus ", "as for ")
        if not llm_main or any(llm_main.lower().startswith(p) for p in bad_prefixes):
            llm_main = (det_main or "").strip()

        # Return only allowed keys
        out = {
            "category": parsed.get("category", "unknown") or "unknown",
            "category_confidence": parsed.get("category_confidence", 0.0),
            "main": llm_main,
            "side": final_side,
        }
        return out

    except Exception:
        return None


def _split_side_candidates(query: str) -> List[str]:
    """
    Deterministic splitting into clause candidates.
    We keep it conservative to avoid over-splitting.
    """
    q = _normalize_q(query)
    # Pull quoted strings as strong side-topic candidates
    quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", q)
    q_wo_quotes = re.sub(r"['\"][^'\"]{2,80}['\"]", " ", q)

    # Split on major separators
    parts = re.split(r"[;]|(?:\s+-\s+)|(?:\s+—\s+)", q_wo_quotes)
    parts = [p for p in (_cleanup_clause(x) for x in parts) if p]

    # Further split on side connectors
    connector_re = "(" + "|".join(SIDE_CONNECTOR_PATTERNS) + ")"
    expanded = []
    for p in parts:
        # break into chunks but keep connector words in-place by splitting into sentences first
        sub = re.split(r"\.\s+|\?\s+|\!\s+", p)
        for s in sub:
            s = _cleanup_clause(s)
            if not s:
                continue
            # if contains connector, split into [before, after...] using first connector
            m = re.search(connector_re, s.lower())
            if m:
                idx = m.start()
                before = _cleanup_clause(s[:idx])
                after = _cleanup_clause(s[idx:])
                if before:
                    expanded.append(before)
                if after:
                    expanded.append(after)
            else:
                expanded.append(s)

    # Add quoted items as standalone candidates (often side topics)
    for qitem in quoted:
        cleaned = _cleanup_clause(qitem)
        if cleaned:
            expanded.append(cleaned)

    # De-dupe while preserving order (deterministic)
    seen = set()
    out = []
    for x in expanded:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _extract_spacy_side_topics(query: str) -> List[str]:
    """
    Optional: use spaCy dependency parse if available.
    Extracts objects of 'impact/effect/role/influence' patterns.
    """
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load("en_core_web_sm")  # type: ignore
        except Exception:
            return []
    except Exception:
        return []

    doc = nlp(query)
    side = []

    triggers = {"impact", "effect", "influence", "role"}
    for token in doc:
        if token.lemma_.lower() in triggers:
            # Look for "of X" attached to trigger
            for child in token.children:
                if child.dep_ == "prep" and child.text.lower() == "of":
                    pobj = next((c for c in child.children if c.dep_ in ("pobj", "dobj", "obj")), None)
                    if pobj is not None:
                        # take subtree as phrase
                        phrase = " ".join(t.text for t in pobj.subtree)
                        phrase = _cleanup_clause(phrase)
                        if phrase and phrase.lower() not in (s.lower() for s in side):
                            side.append(phrase)
    return side[:5]

def _embedding_similarity(a: str, b: str) -> Optional[float]:
    """
    Optional: compute cosine similarity using:
      - sentence-transformers (preferred) OR
      - sklearn TF-IDF fallback
    Returns None if unavailable.
    """
    a = _normalize_q(a)
    b = _normalize_q(b)
    if not a or not b:
        return None

    # 1) sentence-transformers (if installed)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([a, b], normalize_embeddings=True)
        sim = float(np.dot(emb[0], emb[1]))
        return max(min(sim, 1.0), -1.0)
    except Exception:
        pass

    # 2) sklearn TF-IDF cosine similarity (deterministic)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform([a, b])
        sim = float(cosine_similarity(X[0], X[1])[0, 0])
        return max(min(sim, 1.0), -1.0)
    except Exception:
        return None

def extract_query_structure(query: str) -> Dict[str, Any]:
    """
    Layered query structure extraction:
      1) Deterministic clause split -> main/side
      2) Deterministic category from keyword signals (detect_query_category)
      3) Optional NLP refinement (spaCy if available)
      4) Deterministic similarity vote (TF-IDF)
      5) LLM fallback ONLY if confidence remains low
    """
    q = _normalize_q(query)
    clauses = _split_clauses_deterministic(q)
    main, side = _choose_main_and_side(clauses)

    # --- Layer 1: deterministic keyword category ---
    det_cat = detect_query_category(q)
    category = det_cat.get("category", "unknown")
    cat_conf = float(det_cat.get("confidence", 0.0))

    debug = {
        "deterministic": {
            "clauses": clauses,
            "main": main,
            "side": side,
            "category": category,
            "confidence": cat_conf,
            "matched_signals": det_cat.get("matched_signals", []),
        }
    }

    # --- Layer 2: NLP refinement (optional) ---
    nlp_out = _nlp_refine_clauses(q, clauses)
    if isinstance(nlp_out, dict):
        hints = nlp_out.get("hints", {})
        debug["nlp"] = hints or {"nlp_used": False}

        # Override main/side if NLP produced them (guard against fragment-y mains)
        nlp_main = (nlp_out.get("main") or "").strip()
        if nlp_main:
            bad_prefixes = ("as well as", "as well", "and ", "also ", "plus ", "as for ")
            if not any(nlp_main.lower().startswith(p) for p in bad_prefixes):
                main = nlp_main

        if isinstance(nlp_out.get("side"), list):
            side = nlp_out["side"]

        # If NLP detects a place + overview cue, bias to "country"
        gpes = (hints or {}).get("gpe_entities", []) if isinstance(hints, dict) else []
        overview_hit = (hints or {}).get("overview_signal_hit", False) if isinstance(hints, dict) else False
        if overview_hit and gpes and cat_conf < 0.45:
            category = "country"
            cat_conf = max(cat_conf, 0.55)

    # --- Layer 3: embedding-style category vote ---
    emb_vote = _embedding_category_vote(q)
    debug["similarity_vote"] = emb_vote

    emb_cat = emb_vote.get("category", "unknown")
    emb_conf = float(emb_vote.get("confidence", 0.0))

    if cat_conf < 0.40 and emb_cat != "unknown" and emb_conf >= 0.45:
        category = emb_cat
        cat_conf = max(cat_conf, min(0.75, emb_conf))

    # --- Layer 4: LLM fallback if still ambiguous ---
    if cat_conf < 0.30:
        llm = _llm_fallback_query_structure(q)
        debug["llm_fallback_used"] = bool(llm)

        if isinstance(llm, dict):
            category = llm.get("category", category) or category
            try:
                cat_conf = float(llm.get("category_confidence", cat_conf))
            except Exception:
                pass

            llm_main = (llm.get("main") or "").strip()
            llm_side = llm.get("side") if isinstance(llm.get("side"), list) else []

            det_main = (main or "").strip()
            det_side = side or []

            def _overview_score(s: str) -> int:
                if not s:
                    return 0
                s2 = s.lower()
                signals = [
                    "in general", "overview", "background", "basic facts",
                    "at a glance", "tell me about", "describe", "introduction"
                ]
                return sum(1 for sig in signals if sig in s2)

            def _is_bad_main(s: str) -> bool:
                if not s or len(s) < 8:
                    return True
                return s.lower().startswith(
                    ("as well as", "as well", "and ", "also ", "plus ", "as for ")
                )

            merged_side = []
            for s in det_side + llm_side:
                s = str(s).strip()
                if s and s not in merged_side:
                    merged_side.append(s)

            det_score = _overview_score(det_main)
            llm_score = _overview_score(llm_main)

            if llm_main and not _is_bad_main(llm_main):
                if not det_main or llm_score > det_score:
                    main = llm_main

            side = merged_side

    side = _dedupe_clauses([s.strip() for s in (side or []) if s.strip()])

    return {
        "category": category or "unknown",
        "category_confidence": round(max(0.0, min(cat_conf, 1.0)), 2),
        "main": (main or "").strip(),
        "side": side,
        "debug": debug,
    }


def format_query_structure_for_prompt(qs: Optional[Dict[str, Any]]) -> str:
    if not qs or not isinstance(qs, dict):
        return ""

    parts = []
    parts.append("STRUCTURED QUESTION (DETERMINISTIC):")
    parts.append(f"- Category: {qs.get('category','unknown')} (conf {qs.get('category_confidence','')})")
    parts.append(f"- Main (answer this FIRST): {qs.get('main','')}")
    side = qs.get("side") or []

    if side:
        parts.append("- Side questions (answer AFTER main, in this exact order):")
        for i, s in enumerate(side[:10], 1):
            parts.append(f"  {i}. {s}")

    tmpl = qs.get("template_sections") or []
    if tmpl:
        parts.append("- Recommended response sections (use as headings if helpful):")
        for t in tmpl[:10]:
            parts.append(f"  - {t}")

    # Hard behavioral instruction to the LLM (kept short and explicit)
    parts.append(
        "RESPONSE RULES:\n"
        "1) Start by answering the MAIN request with general context.\n"
        "2) Then answer EACH side question explicitly (label them).\n"
        "3) Metrics/findings can include both main + side, but do not ignore the main.\n"
        "4) If you provide tourism/industry metrics, ALSO provide basic country/overview facts when main is an overview."
    )

    return "\n".join(parts).strip()


# ------------------------------------
# METRIC DIFF COMPUTATION
# ------------------------------------

def compute_metric_diffs(old_metrics: Dict, new_metrics: Dict) -> List[MetricDiff]:
    """
    Compute deterministic diffs between metric dictionaries.
    Returns list of MetricDiff objects.
    """
    diffs = []
    matched_new_keys = set()

    # Build lookup for new metrics by normalized name
    new_by_name = {}
    for key, m in new_metrics.items():
        if isinstance(m, dict):
            name = m.get('name', key)
            new_by_name[normalize_name(name)] = (key, m)

    # Process old metrics
    for old_key, old_m in old_metrics.items():
        if not isinstance(old_m, dict):
            continue

        old_name = old_m.get('name', old_key)
        old_raw = str(old_m.get('value', ''))
        old_unit = old_m.get('unit', '')
        old_val = parse_to_float(old_m.get('value'))

        # Find matching new metric
        norm_name = normalize_name(old_name)
        match = new_by_name.get(norm_name)

        if not match:
            # Try fuzzy matching
            best = find_best_match(old_name, [m.get('name', k) for k, m in new_metrics.items() if isinstance(m, dict)])
            if best:
                for k, m in new_metrics.items():
                    if isinstance(m, dict) and m.get('name', k) == best:
                        match = (k, m)
                        break

        if match:
            new_key, new_m = match
            matched_new_keys.add(new_key)

            new_raw = str(new_m.get('value', ''))
            new_val = parse_to_float(new_m.get('value'))
            new_unit = new_m.get('unit', old_unit)

            change_pct = compute_percent_change(old_val, new_val)

            # Determine change type
            if change_pct is None:
                change_type = 'unchanged'
            elif abs(change_pct) < 0.5:  # Less than 0.5% change = unchanged
                change_type = 'unchanged'
            elif change_pct > 0:
                change_type = 'increased'
            else:
                change_type = 'decreased'

            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=new_val,
                old_raw=old_raw,
                new_raw=new_raw,
                unit=new_unit or old_unit,
                change_pct=change_pct,
                change_type=change_type
            ))
        else:
            # Metric was removed
            diffs.append(MetricDiff(
                name=old_name,
                old_value=old_val,
                new_value=None,
                old_raw=old_raw,
                new_raw='',
                unit=old_unit,
                change_pct=None,
                change_type='removed'
            ))

    # Find added metrics
    for new_key, new_m in new_metrics.items():
        if new_key in matched_new_keys:
            continue
        if not isinstance(new_m, dict):
            continue

        new_name = new_m.get('name', new_key)
        new_raw = str(new_m.get('value', ''))
        new_val = parse_to_float(new_m.get('value'))
        new_unit = new_m.get('unit', '')

        diffs.append(MetricDiff(
            name=new_name,
            old_value=None,
            new_value=new_val,
            old_raw='',
            new_raw=new_raw,
            unit=new_unit,
            change_pct=None,
            change_type='added'
        ))

    return diffs

# ------------------------------------
# ENTITY DIFF COMPUTATION
# ------------------------------------

def compute_entity_diffs(old_entities: List, new_entities: List) -> List[EntityDiff]:
    """
    Compute deterministic diffs between entity rankings.
    """
    diffs = []

    # Build lookups with ranks
    old_lookup = {}
    for i, e in enumerate(old_entities):
        if isinstance(e, dict):
            name = normalize_name(e.get('name', ''))
            old_lookup[name] = {
                'rank': i + 1,
                'share': e.get('share'),
                'original_name': e.get('name', '')
            }

    new_lookup = {}
    for i, e in enumerate(new_entities):
        if isinstance(e, dict):
            name = normalize_name(e.get('name', ''))
            new_lookup[name] = {
                'rank': i + 1,
                'share': e.get('share'),
                'original_name': e.get('name', '')
            }

    # All unique names
    all_names = set(old_lookup.keys()) | set(new_lookup.keys())

    for norm_name in all_names:
        old_data = old_lookup.get(norm_name)
        new_data = new_lookup.get(norm_name)

        if old_data and new_data:
            # Entity exists in both
            rank_change = old_data['rank'] - new_data['rank']  # Positive = moved up

            if rank_change > 0:
                change_type = 'moved_up'
            elif rank_change < 0:
                change_type = 'moved_down'
            else:
                change_type = 'unchanged'

            diffs.append(EntityDiff(
                name=new_data['original_name'],
                old_rank=old_data['rank'],
                new_rank=new_data['rank'],
                old_share=old_data['share'],
                new_share=new_data['share'],
                rank_change=rank_change,
                change_type=change_type
            ))
        elif old_data:
            # Entity removed
            diffs.append(EntityDiff(
                name=old_data['original_name'],
                old_rank=old_data['rank'],
                new_rank=None,
                old_share=old_data['share'],
                new_share=None,
                rank_change=None,
                change_type='removed'
            ))
        else:
            # Entity added
            diffs.append(EntityDiff(
                name=new_data['original_name'],
                old_rank=None,
                new_rank=new_data['rank'],
                old_share=None,
                new_share=new_data['share'],
                rank_change=None,
                change_type='added'
            ))

    # Sort by new rank (added entities at end)
    diffs.sort(key=lambda x: x.new_rank if x.new_rank else 999)
    return diffs

# ------------------------------------
# FINDING DIFF COMPUTATION
# ------------------------------------

def compute_finding_diffs(old_findings: List[str], new_findings: List[str]) -> List[FindingDiff]:
    """
    Compute deterministic diffs between findings using text similarity.
    """
    diffs = []
    matched_new_indices = set()

    # Match old findings to new
    for old_f in old_findings:
        if not old_f:
            continue

        best_match_idx = None
        best_similarity = 0.5  # Minimum threshold

        for i, new_f in enumerate(new_findings):
            if i in matched_new_indices or not new_f:
                continue

            sim = name_similarity(old_f, new_f)  # Reuse name similarity for text
            if sim > best_similarity:
                best_similarity = sim
                best_match_idx = i

        if best_match_idx is not None:
            matched_new_indices.add(best_match_idx)
            similarity_pct = round(best_similarity * 100, 1)

            if similarity_pct >= 90:
                change_type = 'retained'
            else:
                change_type = 'modified'

            diffs.append(FindingDiff(
                old_text=old_f,
                new_text=new_findings[best_match_idx],
                similarity=similarity_pct,
                change_type=change_type
            ))
        else:
            # Finding removed
            diffs.append(FindingDiff(
                old_text=old_f,
                new_text=None,
                similarity=0,
                change_type='removed'
            ))

    # Find added findings
    for i, new_f in enumerate(new_findings):
        if i in matched_new_indices or not new_f:
            continue

        diffs.append(FindingDiff(
            old_text=None,
            new_text=new_f,
            similarity=0,
            change_type='added'
        ))

    return diffs

# =========================================================
# 8C. DETERMINISTIC SOURCE EXTRACTION
# Extract metrics/entities directly from web snippets - NO LLM
# =========================================================

def extract_metrics_from_sources(web_context: Dict) -> Dict:
    """
    Extract numeric metrics directly from web search snippets.
    100% deterministic - no LLM involved.
    """
    extracted = {}
    search_results = web_context.get("search_results", [])

    # Patterns to match common metric formats
    patterns = [
        # Market size patterns
        (r'\$\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)\b', 'market_size'),
        (r'market\s+size[:\s]+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'market_size'),
        (r'valued\s+at\s+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'market_size'),
        (r'worth\s+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'market_size'),

        # Growth rate patterns
        (r'CAGR[:\s]+of?\s*(\d+(?:\.\d+)?)\s*%', 'cagr'),
        (r'(\d+(?:\.\d+)?)\s*%\s*CAGR', 'cagr'),
        (r'grow(?:th|ing)?\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%', 'growth_rate'),

        # Revenue patterns
        (r'revenue[:\s]+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'revenue'),

        # Year-specific values
        (r'(?:in\s+)?20\d{2}[:\s]+\$?\s*(\d+(?:\.\d+)?)\s*(trillion|billion|million|T|B|M)', 'year_value'),
    ]

    all_matches = []

    for result in search_results:
        snippet = result.get("snippet", "")
        title = result.get("title", "")
        source = result.get("source", "")
        text = f"{title} {snippet}".lower()

        for pattern, metric_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    value_str, unit = match[0], match[1] if len(match) > 1 else ''
                else:
                    value_str, unit = match, ''

                try:
                    value = float(value_str)

                    # Normalize unit
                    unit_lower = unit.lower() if unit else ''
                    if unit_lower in ['t', 'trillion']:
                        unit_normalized = 'T'
                        value_in_billions = value * 1000
                    elif unit_lower in ['b', 'billion']:
                        unit_normalized = 'B'
                        value_in_billions = value
                    elif unit_lower in ['m', 'million']:
                        unit_normalized = 'M'
                        value_in_billions = value / 1000
                    elif unit_lower == '%':
                        unit_normalized = '%'
                        value_in_billions = value  # Keep as-is for percentages
                    else:
                        unit_normalized = ''
                        value_in_billions = value

                    all_matches.append({
                        'type': metric_type,
                        'value': value,
                        'unit': unit_normalized,
                        'value_normalized': value_in_billions,
                        'source': source,
                        'raw': f"{value_str} {unit}".strip()
                    })
                except (ValueError, TypeError):
                    continue

    # Deduplicate and select best matches by type
    metrics_by_type = {}
    for match in all_matches:
        mtype = match['type']
        if mtype not in metrics_by_type:
            metrics_by_type[mtype] = []
        metrics_by_type[mtype].append(match)

    # For each type, take the most common value (mode) or median
    metric_counter = 0
    for mtype, matches in metrics_by_type.items():
        if not matches:
            continue

        # Group by similar values (within 10%)
        value_groups = []
        for m in matches:
            added = False
            for group in value_groups:
                if group and abs(m['value_normalized'] - group[0]['value_normalized']) / max(group[0]['value_normalized'], 0.001) < 0.1:
                    group.append(m)
                    added = True
                    break
            if not added:
                value_groups.append([m])

        # Take the largest group (most consensus)
        if value_groups:
            best_group = max(value_groups, key=len)
            representative = best_group[0]

            metric_counter += 1
            metric_key = f"extracted_{mtype}_{metric_counter}"

            # Map type to readable name
            type_names = {
                'market_size': 'Market Size',
                'cagr': 'CAGR',
                'growth_rate': 'Growth Rate',
                'revenue': 'Revenue',
                'year_value': 'Market Value'
            }

            extracted[metric_key] = {
                'name': type_names.get(mtype, mtype.replace('_', ' ').title()),
                'value': representative['value'],
                'unit': f"${representative['unit']}" if representative['unit'] in ['T', 'B', 'M'] else representative['unit'],
                'source_count': len(best_group),
                'sources': list(set(m['source'] for m in best_group))[:3]
            }

    return extracted


def extract_entities_from_sources(web_context: Dict) -> List[Dict]:
    """
    Extract company/entity names from web search snippets.
    100% deterministic - no LLM involved.
    """
    search_results = web_context.get("search_results", [])

    # Common market leaders that appear in financial contexts
    known_entities = [
        # Tech
        'apple', 'microsoft', 'google', 'alphabet', 'amazon', 'meta', 'facebook',
        'nvidia', 'tesla', 'intel', 'amd', 'qualcomm', 'broadcom', 'cisco',
        'ibm', 'oracle', 'salesforce', 'adobe', 'netflix', 'uber', 'airbnb',
        # Finance
        'jpmorgan', 'goldman sachs', 'morgan stanley', 'bank of america',
        'wells fargo', 'citigroup', 'blackrock', 'vanguard', 'fidelity',
        # Auto
        'toyota', 'volkswagen', 'ford', 'gm', 'general motors', 'honda',
        'bmw', 'mercedes', 'byd', 'nio', 'rivian', 'lucid',
        # Pharma
        'pfizer', 'johnson & johnson', 'roche', 'novartis', 'merck',
        'abbvie', 'eli lilly', 'astrazeneca', 'moderna', 'gilead',
        # Energy
        'exxon', 'chevron', 'shell', 'bp', 'totalenergies', 'conocophillips',
        # Consumer
        'walmart', 'costco', 'home depot', 'nike', 'starbucks', 'mcdonalds',
        'coca-cola', 'pepsi', 'procter & gamble', 'unilever',
        # Regions (for market share by region)
        'north america', 'europe', 'asia pacific', 'asia-pacific', 'apac',
        'china', 'united states', 'japan', 'germany', 'india', 'uk',
        'latin america', 'middle east', 'africa'
    ]

    entity_mentions = {}

    for result in search_results:
        snippet = result.get("snippet", "").lower()
        title = result.get("title", "").lower()
        text = f"{title} {snippet}"

        for entity in known_entities:
            if entity in text:
                # Try to extract market share if mentioned
                share_pattern = rf'{re.escape(entity)}[^.]*?(\d+(?:\.\d+)?)\s*%'
                share_match = re.search(share_pattern, text, re.IGNORECASE)

                share = None
                if share_match:
                    share = f"{share_match.group(1)}%"

                if entity not in entity_mentions:
                    entity_mentions[entity] = {'count': 0, 'shares': []}

                entity_mentions[entity]['count'] += 1
                if share:
                    entity_mentions[entity]['shares'].append(share)

    # Sort by mention count and build list
    sorted_entities = sorted(entity_mentions.items(), key=lambda x: x[1]['count'], reverse=True)

    entities = []
    for entity_name, data in sorted_entities[:10]:  # Top 10
        # Use most common share if available
        share = None
        if data['shares']:
            # Take the most common share value
            share_counts = Counter(data['shares'])
            share = share_counts.most_common(1)[0][0]

        entities.append({
            'name': entity_name.title(),
            'share': share,
            'growth': None,  # Can't reliably extract growth from snippets
            'mention_count': data['count']
        })

    return entities

# ------------------------------------
# STABILITY SCORE COMPUTATION
# ------------------------------------

def compute_stability_score(
    metric_diffs: List[MetricDiff],
    entity_diffs: List[EntityDiff],
    finding_diffs: List[FindingDiff]
) -> float:
    """
    Compute overall stability score (0-100).
    Higher = more stable (less change).
    """
    scores = []

    # Metric stability (40% weight)
    if metric_diffs:
        stable_metrics = sum(1 for m in metric_diffs if m.change_type == 'unchanged')
        small_change = sum(1 for m in metric_diffs if m.change_pct and abs(m.change_pct) < 10)
        metric_score = ((stable_metrics + small_change * 0.5) / len(metric_diffs)) * 100
        scores.append(('metrics', metric_score, 0.4))

    # Entity stability (35% weight)
    if entity_diffs:
        stable_entities = sum(1 for e in entity_diffs if e.change_type == 'unchanged')
        entity_score = (stable_entities / len(entity_diffs)) * 100
        scores.append(('entities', entity_score, 0.35))

    # Finding stability (25% weight)
    if finding_diffs:
        retained = sum(1 for f in finding_diffs if f.change_type in ['retained', 'modified'])
        finding_score = (retained / len(finding_diffs)) * 100
        scores.append(('findings', finding_score, 0.25))

    if not scores:
        return 100.0

    # Weighted average
    total_weight = sum(s[2] for s in scores)
    weighted_sum = sum(s[1] * s[2] for s in scores)
    return round(weighted_sum / total_weight, 1)

# ------------------------------------
# MAIN DIFF COMPUTATION
# ------------------------------------

def compute_evolution_diff(old_analysis: Dict, new_analysis: Dict) -> EvolutionDiff:
    """
    Main entry point: compute complete deterministic diff between two analyses.
    """
    old_response = old_analysis.get('primary_response', {})
    new_response = new_analysis.get('primary_response', {})

    # Timestamps
    old_ts = old_analysis.get('timestamp', '')
    new_ts = new_analysis.get('timestamp', '')

    # Calculate time delta
    time_delta = None
    try:
        old_dt = datetime.fromisoformat(old_ts.replace('Z', '+00:00'))
        new_dt = datetime.fromisoformat(new_ts.replace('Z', '+00:00'))
        time_delta = round((new_dt.replace(tzinfo=None) - old_dt.replace(tzinfo=None)).total_seconds() / 3600, 1)
    except:
        pass

    # Compute diffs using CANONICAL metric registry for stable matching
    metric_diffs = compute_metric_diffs_canonical(
        old_response.get('primary_metrics', {}),
        new_response.get('primary_metrics', {})
    )

    entity_diffs = compute_entity_diffs(
        old_response.get('top_entities', []),
        new_response.get('top_entities', [])
    )

    # Use SEMANTIC finding comparison (stable across wording changes)
    finding_diffs = compute_semantic_finding_diffs(
        old_response.get('key_findings', []),
        new_response.get('key_findings', [])
    )

    # Compute stability
    stability = compute_stability_score(metric_diffs, entity_diffs, finding_diffs)

    # Summary stats
    summary_stats = {
        'metrics_increased': sum(1 for m in metric_diffs if m.change_type == 'increased'),
        'metrics_decreased': sum(1 for m in metric_diffs if m.change_type == 'decreased'),
        'metrics_unchanged': sum(1 for m in metric_diffs if m.change_type == 'unchanged'),
        'metrics_added': sum(1 for m in metric_diffs if m.change_type == 'added'),
        'metrics_removed': sum(1 for m in metric_diffs if m.change_type == 'removed'),
        'entities_moved_up': sum(1 for e in entity_diffs if e.change_type == 'moved_up'),
        'entities_moved_down': sum(1 for e in entity_diffs if e.change_type == 'moved_down'),
        'entities_unchanged': sum(1 for e in entity_diffs if e.change_type == 'unchanged'),
        'entities_added': sum(1 for e in entity_diffs if e.change_type == 'added'),
        'entities_removed': sum(1 for e in entity_diffs if e.change_type == 'removed'),
        'findings_retained': sum(1 for f in finding_diffs if f.change_type == 'retained'),
        'findings_modified': sum(1 for f in finding_diffs if f.change_type == 'modified'),
        'findings_added': sum(1 for f in finding_diffs if f.change_type == 'added'),
        'findings_removed': sum(1 for f in finding_diffs if f.change_type == 'removed'),
    }

    return EvolutionDiff(
        old_timestamp=old_ts,
        new_timestamp=new_ts,
        time_delta_hours=time_delta,
        metric_diffs=metric_diffs,
        entity_diffs=entity_diffs,
        finding_diffs=finding_diffs,
        stability_score=stability,
        summary_stats=summary_stats
    )

# ------------------------------------
# LLM EXPLANATION (ONLY INTERPRETS DIFFS)
# ------------------------------------

def generate_diff_explanation_prompt(diff: EvolutionDiff, query: str) -> str:
    """
    Generate prompt for LLM to EXPLAIN computed diffs (not discover them).
    """
    # Build metric changes text
    metric_changes = []
    for m in diff.metric_diffs:
        if m.change_type == 'increased':
            metric_changes.append(f"- {m.name}: {m.old_raw} → {m.new_raw} ({m.change_pct:+.1f}%) INCREASED")
        elif m.change_type == 'decreased':
            metric_changes.append(f"- {m.name}: {m.old_raw} → {m.new_raw} ({m.change_pct:+.1f}%) DECREASED")
        elif m.change_type == 'added':
            metric_changes.append(f"- {m.name}: NEW metric added with value {m.new_raw}")
        elif m.change_type == 'removed':
            metric_changes.append(f"- {m.name}: REMOVED (was {m.old_raw})")

    # Build entity changes text
    entity_changes = []
    for e in diff.entity_diffs:
        if e.change_type == 'moved_up':
            entity_changes.append(f"- {e.name}: Rank {e.old_rank} → {e.new_rank} (moved UP {e.rank_change} positions)")
        elif e.change_type == 'moved_down':
            entity_changes.append(f"- {e.name}: Rank {e.old_rank} → {e.new_rank} (moved DOWN {abs(e.rank_change)} positions)")
        elif e.change_type == 'added':
            entity_changes.append(f"- {e.name}: NEW entrant at rank {e.new_rank}")
        elif e.change_type == 'removed':
            entity_changes.append(f"- {e.name}: DROPPED OUT (was rank {e.old_rank})")

    # Build findings changes text
    finding_changes = []
    for f in diff.finding_diffs:
        if f.change_type == 'added':
            finding_changes.append(f"- NEW: {f.new_text}")
        elif f.change_type == 'removed':
            finding_changes.append(f"- REMOVED: {f.old_text}")
        elif f.change_type == 'modified':
            finding_changes.append(f"- MODIFIED: '{f.old_text[:50]}...' → '{f.new_text[:50]}...'")

    prompt = f"""You are a market analyst explaining changes between two analysis snapshots.

    QUERY: {query}
    TIME ELAPSED: {diff.time_delta_hours:.1f} hours
    STABILITY SCORE: {diff.stability_score:.0f}%

    COMPUTED METRIC CHANGES:
    {chr(10).join(metric_changes) if metric_changes else "No significant metric changes"}

    COMPUTED ENTITY RANKING CHANGES:
    {chr(10).join(entity_changes) if entity_changes else "No ranking changes"}

    COMPUTED FINDING CHANGES:
    {chr(10).join(finding_changes) if finding_changes else "No finding changes"}

    SUMMARY STATS:
    - Metrics: {diff.summary_stats['metrics_increased']} increased, {diff.summary_stats['metrics_decreased']} decreased, {diff.summary_stats['metrics_unchanged']} unchanged
    - Entities: {diff.summary_stats['entities_moved_up']} moved up, {diff.summary_stats['entities_moved_down']} moved down
    - Findings: {diff.summary_stats['findings_added']} new, {diff.summary_stats['findings_removed']} removed

    YOUR TASK: Provide a 3-5 sentence executive interpretation of these changes.
    - What is the overall trend (improving/declining/stable)?
    - What are the most significant changes and why might they have occurred?
    - What should stakeholders pay attention to?

    Return ONLY a JSON object:
    {{
        "trend": "improving/declining/stable",
        "headline": "One sentence summary of key change",
        "interpretation": "3-5 sentence detailed interpretation",
        "watch_items": ["Item 1 to monitor", "Item 2 to monitor"]
    }}
    """
    return prompt

def get_llm_explanation(diff: EvolutionDiff, query: str) -> Dict:
    """
    Ask LLM to explain the computed diffs (not discover them).
    """
    prompt = generate_diff_explanation_prompt(diff, query)

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 500,
        "top_p": 1.0,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        parsed = parse_json_safely(content, "Explanation")
        if parsed:
            return parsed
    except Exception as e:
        st.warning(f"LLM explanation failed: {e}")

    # Fallback
    return {
        "trend": "stable" if diff.stability_score >= 70 else "changing",
        "headline": f"Analysis shows {diff.stability_score:.0f}% stability over {diff.time_delta_hours:.0f} hours",
        "interpretation": "Unable to generate detailed interpretation.",
        "watch_items": []
    }


# =========================================================
# 8B. EVOLUTION DASHBOARD RENDERING
# =========================================================

def render_evolution_results(diff: EvolutionDiff, explanation: Dict, query: str):
    """Render deterministic evolution results"""

    st.header("📈 Evolution Analysis")
    st.markdown(f"**Query:** {query}")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    if diff.time_delta_hours:
        if diff.time_delta_hours < 24:
            time_str = f"{diff.time_delta_hours:.1f}h"
        else:
            time_str = f"{diff.time_delta_hours/24:.1f}d"
        col1.metric("Time Elapsed", time_str)
    else:
        col1.metric("Time Elapsed", "Unknown")

    col2.metric("Stability", f"{diff.stability_score:.0f}%")

    trend = explanation.get('trend', 'stable')
    trend_icon = {'improving': '📈', 'declining': '📉', 'stable': '➡️'}.get(trend, '➡️')
    col3.metric("Trend", f"{trend_icon} {trend.title()}")

    # Stability indicator
    if diff.stability_score >= 80:
        col4.success("🟢 Highly Stable")
    elif diff.stability_score >= 60:
        col4.warning("🟡 Moderate Changes")
    else:
        col4.error("🔴 Significant Drift")

    # Headline
    st.info(f"**{explanation.get('headline', 'Analysis complete')}**")

    st.markdown("---")

    # Interpretation
    st.subheader("📋 Interpretation")
    st.markdown(explanation.get('interpretation', 'No interpretation available'))

    # Watch items
    watch_items = explanation.get('watch_items', [])
    if watch_items:
        st.markdown("**🔔 Watch Items:**")
        for item in watch_items:
            st.markdown(f"- {item}")

    st.markdown("---")

    # Metric Changes Table
    st.subheader("💰 Metric Changes")
    if diff.metric_diffs:
        metric_rows = []

        def _fmt_currency_first(raw: str, unit: str) -> str:
            """
            Formats evolution metrics as:
            - S$29.8B
            - $120M
            - 29.8%
            """
            raw = (raw or "").strip()
            unit = (unit or "").strip()

            if not raw or raw == "-":
                return "-"

            # If already currency-first, trust it
            if raw.startswith("S$") or raw.startswith("$"):
                return raw

            # Percent case
            if unit == "%":
                return f"{raw}%"

            # Detect currency from unit
            currency = ""
            scale = unit.replace(" ", "")

            if scale.upper().startswith("SGD"):
                currency = "S$"
                scale = scale[3:]
            elif scale.upper().startswith("USD"):
                currency = "$"
                scale = scale[3:]
            elif scale.startswith("S$"):
                currency = "S$"
                scale = scale[2:]
            elif scale.startswith("$"):
                currency = "$"
                scale = scale[1:]

            # Human-readable units
            if unit.lower().endswith("billion"):
                return f"{currency}{raw} billion".strip()
            if unit.lower().endswith("million"):
                return f"{currency}{raw} million".strip()

            # Compact units (B/M/K)
            if scale.upper() in {"B", "M", "K"}:
                return f"{currency}{raw}{scale}".strip()

            # Fallback
            return f"{currency}{raw} {unit}".strip()

        for m in diff.metric_diffs:
            icon = {
                'increased': '📈', 'decreased': '📉', 'unchanged': '➡️',
                'added': '🆕', 'removed': '❌'
            }.get(m.change_type, '•')

            change_str = f"{m.change_pct:+.1f}%" if m.change_pct is not None else "-"

            prev_raw = m.old_raw or "-"
            curr_raw = m.new_raw or "-"

            metric_rows.append({
                "": icon,
                "Metric": m.name,
                "Previous": _fmt_currency_first(prev_raw, getattr(m, "unit", "") or ""),
                "Current":  _fmt_currency_first(curr_raw, getattr(m, "unit", "") or ""),
                "Change": change_str,
                "Status": m.change_type.replace('_', ' ').title()
            })

        st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No metrics to compare")

    st.markdown("---")

    # Entity Changes Table
    st.subheader("🏢 Entity Ranking Changes")
    if diff.entity_diffs:
        entity_rows = []
        for e in diff.entity_diffs:
            icon = {
                'moved_up': '⬆️', 'moved_down': '⬇️', 'unchanged': '➡️',
                'added': '🆕', 'removed': '❌'
            }.get(e.change_type, '•')

            rank_str = f"{e.rank_change:+d}" if e.rank_change else "-"

            entity_rows.append({
                "": icon,
                "Entity": e.name,
                "Old Rank": e.old_rank or "-",
                "New Rank": e.new_rank or "-",
                "Rank Δ": rank_str,
                "Old Share": e.old_share or "-",
                "New Share": e.new_share or "-"
            })
        st.dataframe(pd.DataFrame(entity_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No entities to compare")

    st.markdown("---")

    # Finding Changes
    st.subheader("🔍 Finding Changes")
    if diff.finding_diffs:
        added = [f for f in diff.finding_diffs if f.change_type == 'added']
        removed = [f for f in diff.finding_diffs if f.change_type == 'removed']
        modified = [f for f in diff.finding_diffs if f.change_type == 'modified']

        if added:
            st.markdown("**🆕 New Findings:**")
            for f in added:
                st.success(f"• {f.new_text}")

        if removed:
            st.markdown("**❌ Removed Findings:**")
            for f in removed:
                st.error(f"• ~~{f.old_text}~~")

        if modified:
            st.markdown("**✏️ Modified Findings:**")
            for f in modified:
                st.warning(f"• {f.new_text} *(similarity: {f.similarity:.0f}%)*")
    else:
        st.info("No findings to compare")

    st.markdown("---")

    # Summary Stats
    st.subheader("📊 Change Summary")
    stats = diff.summary_stats

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Metrics:**")
        st.write(f"📈 {stats['metrics_increased']} increased")
        st.write(f"📉 {stats['metrics_decreased']} decreased")
        st.write(f"➡️ {stats['metrics_unchanged']} unchanged")

    with col2:
        st.markdown("**Entities:**")
        st.write(f"⬆️ {stats['entities_moved_up']} moved up")
        st.write(f"⬇️ {stats['entities_moved_down']} moved down")
        st.write(f"🆕 {stats['entities_added']} new")

    with col3:
        st.markdown("**Findings:**")
        st.write(f"✅ {stats['findings_retained']} retained")
        st.write(f"✏️ {stats['findings_modified']} modified")
        st.write(f"🆕 {stats['findings_added']} new")


# =========================================================
# 8D. SOURCE-ANCHORED EVOLUTION
# Re-fetch the SAME sources from previous analysis for true stability
# Enhanced fetch_url_content function to use scrapingdog as fallback
# =========================================================

def _extract_pdf_text_from_bytes(pdf_bytes: bytes, max_pages: int = 6, max_chars: int = 7000) -> Optional[str]:
    """
    Extract readable text from PDF bytes deterministically.
    Limits pages/chars for speed and consistent output.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for i, page in enumerate(reader.pages[:max_pages]):
            t = page.extract_text() or ""
            t = t.replace("\x00", " ").strip()
            if t:
                texts.append(t)
        joined = "\n".join(texts).strip()
        if len(joined) < 200:
            return None
        return joined[:max_chars]
    except Exception:
        return None



def fetch_url_content(url: str) -> Optional[str]:
    """Fetch content from a specific URL with ScrapingDog fallback"""

    def extract_text(html: str) -> Optional[str]:
        """Extract clean text from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        clean_text = ' '.join(line for line in lines if line)
        return clean_text[:5000] if len(clean_text) > 200 else None

    # Try 1: Direct request
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        if 'captcha' not in resp.text.lower():
            content = extract_text(resp.text)
            if content:
                return content
    except:
        pass

    # Try 2: ScrapingDog API
    if SCRAPINGDOG_KEY:
        try:
            api_url = "https://api.scrapingdog.com/scrape"
            params = {"api_key": SCRAPINGDOG_KEY, "url": url, "dynamic": "false"}
            resp = requests.get(api_url, params=params, timeout=30)
            if resp.status_code == 200:
                content = extract_text(resp.text)
                if content:
                    return content
        except:
            pass

    return None



def fetch_url_content_with_status(url: str, timeout: int = 25):
    """
    Fetch URL content and return (text, status_detail).

    status_detail:
      - "success"
      - "success_pdf"
      - "http_<code>"
      - "exception:<TypeName>"
      - "empty"
      - "success_scrapingdog"

    Hardened:
      - Uses browser-like headers for direct fetch
      - Falls back to ScrapingDog when blocked/empty and SCRAPINGDOG_KEY is available
      - Avoids returning binary garbage as "text"
    """
    import re
    import requests

    def _normalize_url(s: str) -> str:
        t = (s or "").strip()
        if not t:
            return ""
        if re.match(r"^https?://", t, flags=re.I):
            return t
        if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}(/.*)?$", t, flags=re.I):
            return "https://" + t
        return ""

    url = _normalize_url(url)
    if not url:
        return None, "empty"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    # ---------- 1) Direct fetch ----------
    try:
        resp = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)

        ct = (resp.headers.get("content-type", "") or "").lower()

        if resp.status_code >= 400:
            # If blocked, try ScrapingDog fallback (optional)
            if resp.status_code in (401, 403, 429) and globals().get("SCRAPINGDOG_KEY"):
                txt = _fetch_via_scrapingdog(url, timeout=timeout)
                if txt and txt.strip():
                    return txt, "success_scrapingdog"
            return None, f"http_{resp.status_code}"

        # PDF handling
        if "application/pdf" in ct or url.lower().endswith(".pdf"):
            try:
                import io
                import pdfplumber  # type: ignore
                with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                    out = []
                    for page in pdf.pages[:20]:
                        t = page.extract_text() or ""
                        if t.strip():
                            out.append(t)
                text = "\n".join(out).strip()
                if not text:
                    return None, "empty"
                return text, "success_pdf"
            except Exception as e:
                return None, f"exception:{type(e).__name__}"

        # Text/HTML
        text = resp.text or ""
        # If empty or suspiciously short, attempt ScrapingDog (optional)
        if (not text.strip() or len(text.strip()) < 300) and globals().get("SCRAPINGDOG_KEY"):
            txt = _fetch_via_scrapingdog(url, timeout=timeout)
            if txt and txt.strip():
                return txt, "success_scrapingdog"

        if not text.strip():
            return None, "empty"

        return text, "success"

    except Exception as e:
        # ScrapingDog as last resort for network-y issues
        try:
            if globals().get("SCRAPINGDOG_KEY"):
                txt = _fetch_via_scrapingdog(url, timeout=timeout)
                if txt and txt.strip():
                    return txt, "success_scrapingdog"
        except Exception:
            pass
        return None, f"exception:{type(e).__name__}"


def _fetch_via_scrapingdog(url: str, timeout: int = 25) -> str:
    """
    Internal helper used by fetch_url_content_with_status.
    Returns raw HTML text from ScrapingDog (or "" on failure).
    """
    import requests

    key = globals().get("SCRAPINGDOG_KEY")
    if not key:
        return ""

    params = {"api_key": key, "url": url, "dynamic": "false"}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get("https://api.scrapingdog.com/scrape", params=params, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            return ""
        return resp.text or ""
    except Exception:
        return ""

def get_extractor_fingerprint() -> str:
    """
    Bump this string whenever you change extraction or normalization behavior.
    Used to decide whether cached extracted_numbers are still valid.
    """
    return "extract_v2_normunits_2026-01-02"



def extract_numbers_from_text(text: str) -> List[Dict]:
    """
    Backward-compatible wrapper.

    v7_34 tightening:
    - Delegate to extract_numbers_with_context() so junk suppression is applied consistently.
    """
    try:
        return extract_numbers_with_context(text or "", source_url="", max_results=600) or []
    except Exception:
        return []


def _parse_iso_dt(ts: Optional[str]) -> Optional[datetime]:
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

def now_utc() -> datetime:
    """Timezone-aware UTC now (prevents naive/aware datetime bugs)."""
    return datetime.now(timezone.utc)


def _normalize_number_to_parse_base(value: float, unit: str) -> float:
    u = (unit or "").strip().upper()
    if u == "T":
        return value * 1_000_000
    if u == "B":
        return value * 1_000
    if u == "M":
        return value * 1
    if u == "K":
        return value * 0.001
    if u == "%":
        return value
    return value

def run_source_anchored_evolution(previous_data: dict, web_context: dict = None) -> dict:
    """
    Backward-compatible entrypoint used by the Streamlit Evolution UI.

    Enhancements:
      - Accept optional web_context so evolution can reuse same-run analysis upstream artifacts.
      - ALWAYS returns a dict with required keys (even on crash).
    """
    fn = globals().get("compute_source_anchored_diff")

    def _fail(msg: str) -> dict:
        return {
            "status": "failed",
            "message": msg,
            "sources_checked": 0,
            "sources_fetched": 0,
            "numbers_extracted_total": 0,
            "stability_score": 0.0,
            "summary": {
                "total_metrics": 0,
                "metrics_found": 0,
                "metrics_increased": 0,
                "metrics_decreased": 0,
                "metrics_unchanged": 0,
            },
            "metric_changes": [],
            "source_results": [],
            "interpretation": "Evolution failed.",
        }

    if not callable(fn):
        return _fail("compute_source_anchored_diff() is not defined, so source-anchored evolution cannot run.")

    try:
        # Support both old signature (previous_data) and new signature (previous_data, web_context)
        try:
            out = fn(previous_data, web_context=web_context)
        except TypeError:
            out = fn(previous_data)
    except Exception as e:
        return _fail(f"compute_source_anchored_diff crashed: {e}")

    if not isinstance(out, dict):
        return _fail("compute_source_anchored_diff returned a non-dict payload.")

    # Renderer-required defaults
    out.setdefault("status", "success")
    out.setdefault("message", "")
    out.setdefault("sources_checked", 0)
    out.setdefault("sources_fetched", 0)
    out.setdefault("numbers_extracted_total", 0)
    out.setdefault("stability_score", 0.0)
    out.setdefault("summary", {})
    out["summary"].setdefault("total_metrics", len(out.get("metric_changes") or []))
    out["summary"].setdefault("metrics_found", 0)
    out["summary"].setdefault("metrics_increased", 0)
    out["summary"].setdefault("metrics_decreased", 0)
    out["summary"].setdefault("metrics_unchanged", 0)
    out.setdefault("metric_changes", [])
    out.setdefault("source_results", [])
    out.setdefault("interpretation", "")

    return out


# =========================================================
# ROBUST EVOLUTION HELPERS (DETERMINISTIC)
# =========================================================

NON_DATA_CONTEXT_HINTS = [
    "table of contents", "cookie", "privacy", "terms", "copyright",
    "subscribe", "newsletter", "login", "sign in", "nav", "footer"
]


def _truncate_json_safely_for_sheets(json_str: str, max_chars: int = 45000) -> str:
    """
    PATCH TS1 (ADDITIVE): JSON-safe truncation wrapper
    - Ensures json.loads always succeeds for any returned value.
    - Stores a preview when oversized.
    """
    import json

    s = "" if json_str is None else str(json_str)
    if len(s) <= max_chars:
        return s

    preview_len = max(0, int(max_chars) - 700)
    wrapper = {
        "_sheets_safe": True,
        "_sheet_write": {
            "truncated": True,
            "mode": "json_wrapper",
            "note": "Payload exceeded cell limit; stored preview only.",
        },
        "preview": s[:preview_len],
    }
    try:
        return json.dumps(wrapper, ensure_ascii=False, default=str)
    except Exception:
        return '{"_sheets_safe":true,"_sheet_write":{"truncated":true,"mode":"json_wrapper","note":"json.dumps failed"}}'


def _truncate_for_sheets(s: str, max_chars: int = 45000) -> str:
    """Hard cap to stay under Google Sheets 50k/cell limit."""
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    head = s[: int(max_chars * 0.75)]
    tail = s[- int(max_chars * 0.20):]
    return head + "\n...\n[TRUNCATED FOR GOOGLE SHEETS]\n...\n" + tail



def _summarize_heavy_fields_for_sheets(obj: dict) -> dict:
    """
    Summarize fields that commonly exceed the per-cell limit while keeping debug utility.
    Only used for Sheets serialization; does NOT modify your in-memory analysis dict.
    """
    if not isinstance(obj, dict):
        return {"_type": str(type(obj)), "value": str(obj)[:500]}

    out = dict(obj)

    # Common bloat fields
    if "scraped_meta" in out:
        sm = out.get("scraped_meta")
        if isinstance(sm, dict):
            compact = {}
            for url, meta in list(sm.items())[:12]:
                if isinstance(meta, dict):
                    compact[url] = {
                        "status": meta.get("status"),
                        "status_detail": meta.get("status_detail"),
                        "numbers_found": meta.get("numbers_found"),
                        "fingerprint": meta.get("fingerprint"),
                        "clean_text_len": meta.get("clean_text_len"),
                    }
            out["scraped_meta"] = {"_summary": True, "count": len(sm), "sample": compact}
        else:
            out["scraped_meta"] = {"_summary": True, "type": str(type(sm))}

    for big_key in ("source_results", "baseline_sources_cache", "baseline_sources_cache_compact"):
        if big_key in out:
            sr = out.get(big_key)
            if isinstance(sr, list):
                sample = []
                for item in sr[:2]:
                    if isinstance(item, dict):
                        item2 = dict(item)
                        if isinstance(item2.get("extracted_numbers"), list):
                            item2["extracted_numbers"] = {"_summary": True, "count": len(item2["extracted_numbers"])}
                        sample.append(item2)
                out[big_key] = {"_summary": True, "count": len(sr), "sample": sample}
            else:
                out[big_key] = {"_summary": True, "type": str(type(sr))}

    # If you store full scraped_content anywhere, summarize it too
    if "scraped_content" in out:
        sc = out.get("scraped_content")
        if isinstance(sc, dict):
            out["scraped_content"] = {"_summary": True, "count": len(sc), "keys_sample": list(sc.keys())[:10]}
        else:
            out["scraped_content"] = {"_summary": True, "type": str(type(sc))}

    # =====================================================================
    # PATCH SS2 (ADDITIVE, REQUIRED): summarize nested heavy fields under out["results"]
    # Why:
    # - Your biggest payload is typically results.baseline_sources_cache (full snapshots)
    # - The previous summarizer only handled top-level keys, so Sheets payload still exceeded limits
    # - This keeps the saved JSON smaller AND keeps json.loads(get_history) working reliably
    # =====================================================================
    try:
        r = out.get("results")
        if isinstance(r, dict):
            r2 = dict(r)

            for big_key in ("baseline_sources_cache", "source_results"):
                if big_key in r2:
                    sr = r2.get(big_key)
                    if isinstance(sr, list):
                        sample = []
                        for item in sr[:2]:
                            if isinstance(item, dict):
                                item2 = dict(item)
                                if isinstance(item2.get("extracted_numbers"), list):
                                    item2["extracted_numbers"] = {
                                        "_summary": True,
                                        "count": len(item2["extracted_numbers"])
                                    }
                                sample.append(item2)
                        r2[big_key] = {"_summary": True, "count": len(sr), "sample": sample}
                    else:
                        r2[big_key] = {"_summary": True, "type": str(type(sr))}

            out["results"] = r2
    except Exception:
        pass
    # =====================================================================

    return out



def make_sheet_safe_json(obj: dict, max_chars: int = 45000) -> str:
    """
    Serialize sheet-safe JSON under the cell limit.

    NOTE / CONFLICT:
      - The prior implementation used _truncate_for_sheets() on the JSON string, which can produce
        invalid JSON (cut mid-string). Invalid JSON rows are skipped by get_history() (json.loads fails),
        so evolution can't pick them up.
      - This patch preserves summarization but replaces raw string truncation with a JSON wrapper
        that is ALWAYS valid JSON.

    Output behavior:
      - If JSON fits: returns full compact JSON string.
      - If too large: returns a valid JSON wrapper with a preview + metadata.
    """
    import json

    # Keep existing behavior: summarize heavy fields
    compact = _summarize_heavy_fields_for_sheets(obj if isinstance(obj, dict) else {"value": obj})
    if isinstance(compact, dict):
        compact["_sheets_safe"] = True

    # Try to serialize
    try:
        s = json.dumps(compact, ensure_ascii=False, default=str)
    except Exception:
        # ultra-safe fallback (still return valid JSON)
        try:
            s = json.dumps({"_sheets_safe": True, "_sheet_write": {"error": "json.dumps failed"}}, ensure_ascii=False)
        except Exception:
            return '{"_sheets_safe":true,"_sheet_write":{"error":"json.dumps failed"}}'

    # If it fits, return as-is
    if isinstance(s, str) and len(s) <= int(max_chars or 45000):
        return s

    # =========================
    # PATCH SS1 (BUGFIX, REQUIRED): valid JSON wrapper when oversized
    # - Never return mid-string truncations that break json.loads in get_history().
    # =========================
    try:
        preview_len = max(0, int(max_chars or 45000) - 700)  # leave room for wrapper fields
        wrapper = {
            "_sheets_safe": True,
            "_sheet_write": {
                "truncated": True,
                "mode": "sheets_safe_wrapper",
                "note": "Payload exceeded cell limit; stored preview only. Full snapshots must be stored separately if needed.",
            },
            # Keep a preview for UI/debugging
            "preview": s[:preview_len],
        }

        # Optional: carry minimal identity fields for convenience (additive)
        if isinstance(obj, dict):
            wrapper["question"] = (obj.get("question") or "")[:200]
            wrapper["timestamp"] = obj.get("timestamp")
            wrapper["code_version"] = obj.get("code_version") or (obj.get("primary_response") or {}).get("code_version")

            # =========================
            # PATCH SS1B (ADDITIVE, REQUIRED FOR SNAPSHOT REHYDRATION):
            # Carry snapshot pointers even when the payload is wrapped.
            # Without these fields, evolution cannot rehydrate full snapshots
            # from the Snapshots worksheet (or local fallback) and will fail
            # the snapshot gate with "No valid snapshots".
            # =========================
            try:
                _ssh = obj.get("source_snapshot_hash") or (obj.get("results") or {}).get("source_snapshot_hash")
                _ref = obj.get("snapshot_store_ref") or (obj.get("results") or {}).get("snapshot_store_ref")
                if _ssh:
                    wrapper["source_snapshot_hash"] = _ssh
                if _ref:
                    wrapper["snapshot_store_ref"] = _ref
            except Exception:
                pass

        return json.dumps(wrapper, ensure_ascii=False, default=str)
    except Exception:
        return '{"_sheets_safe":true,"_sheet_write":{"truncated":true,"mode":"sheets_safe_wrapper","note":"wrapper failed"}}'


# =====================================================================
# PATCH ES1D (ADDITIVE): external snapshot store (local file-based)
# Purpose:
#   - Store full baseline_sources_cache outside Google Sheets when rows
#     are too large (Sheets wrapper / preview mode).
#   - Allow deterministic rehydration for evolution (no refetch).
# =====================================================================
def _snapshot_store_dir() -> str:
    import os
    d = os.path.join(os.getcwd(), "snapshot_store")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d

def store_full_snapshots_local(baseline_sources_cache: list, source_snapshot_hash: str) -> str:
    """
    Store full snapshots deterministically by hash. Returns a store ref string (path).
    Additive-only helper.
    """
    import os, json
    if not source_snapshot_hash:
        return ""
    if not isinstance(baseline_sources_cache, list) or not baseline_sources_cache:
        return ""

    path = os.path.join(_snapshot_store_dir(), f"{source_snapshot_hash}.json")
    try:
        # write-once semantics (deterministic)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    except Exception:
        pass

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(baseline_sources_cache, f, ensure_ascii=False, default=str)
        return path
    except Exception:
        return ""

def load_full_snapshots_local(snapshot_store_ref: str) -> list:
    """
    Load full snapshots from a store ref string (path). Returns [] if not available.
    """
    import json, os
    try:
        if not snapshot_store_ref or not isinstance(snapshot_store_ref, str):
            return []
        if not os.path.exists(snapshot_store_ref):
            return []
        with open(snapshot_store_ref, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

# =====================================================================
# PATCH ES1E (ADDITIVE): deterministic source_snapshot_hash helper
# =====================================================================
def compute_source_snapshot_hash(baseline_sources_cache: list) -> str:
    import hashlib
    pairs = []
    for sr in (baseline_sources_cache or []):
        if not isinstance(sr, dict):
            continue
        u = (sr.get("source_url") or sr.get("url") or "").strip()
        fp = (sr.get("fingerprint") or sr.get("content_fingerprint") or "").strip()
        if u:
            pairs.append((u, fp))
    pairs.sort()
    sig = "|".join([f"{u}#{fp}" for (u, fp) in pairs])
    return hashlib.sha256(sig.encode("utf-8")).hexdigest() if sig else ""
# =====================================================================
# =====================================================================
# PATCH SS6 (ADDITIVE): build full baseline_sources_cache from evidence_records
# Why:
# - Sheets-safe summarization may replace baseline_sources_cache/extracted_numbers
#   with summary dicts. However, evidence_records often remains available and is
#   already deterministic, snapshot-derived data.
# - This helper reconstructs the minimal snapshot shape needed for
#   source-anchored evolution WITHOUT re-fetching or heuristic matching.
# =====================================================================

            # =========================
# PATCH A (ADD): Snapshot hash v2 (stable, content-weighted)
# - Keeps v1 compute_source_snapshot_hash() for backward compatibility.
# - v2 includes url + status + fingerprint + (anchor_hash,value_norm,unit_tag) tuples (bounded) for stronger identity.
            # =========================
def compute_source_snapshot_hash_v2(baseline_sources_cache: list, max_items_per_source: int = 120) -> str:
    import hashlib
    import json

    try:
        sources = baseline_sources_cache if isinstance(baseline_sources_cache, list) else []
        parts = []
        for s in sources:
            if not isinstance(s, dict):
                continue
            url = str(s.get("url") or "")
            status = str(s.get("status") or "")
            status_detail = str(s.get("status_detail") or "")
            fingerprint = str(s.get("fingerprint") or "")

            nums = s.get("extracted_numbers") or s.get("numbers") or []
            # Sometimes stored in summarized form
            if isinstance(nums, dict) and nums.get("_summary") and isinstance(nums.get("count"), int):
                # no details available; just use summary
                num_tuples = [("summary_count", int(nums.get("count")))]
            else:
                num_list = nums if isinstance(nums, list) else []
                num_tuples = []
                for n in num_list[: int(max_items_per_source or 120)]:
                    if not isinstance(n, dict):
                        continue
                    ah = str(n.get("anchor_hash") or "")
                    vn = n.get("value_norm")
                    ut = str(n.get("unit_tag") or n.get("unit") or "")
                    # Use JSON for float stability + None handling
                    num_tuples.append((ah, vn, ut))
                # Deterministic order
                num_tuples = sorted(num_tuples, key=lambda t: (t[0], str(t[1]), t[2]))

            parts.append({
                "url": url,
                "status": status,
                "status_detail": status_detail,
                "fingerprint": fingerprint,
                "nums": num_tuples,
            })

        # Deterministic ordering of sources
        parts = sorted(parts, key=lambda d: (d.get("url",""), d.get("fingerprint",""), d.get("status","")))

        payload = json.dumps(parts, ensure_ascii=False, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        # Ultra-safe fallback (still deterministic-ish)
        try:
            return hashlib.sha256(str(baseline_sources_cache).encode("utf-8")).hexdigest()
        except Exception:
            return "0"*64

def build_baseline_sources_cache_from_evidence_records(evidence_records):
    """Return a list-shaped baseline_sources_cache rebuilt from evidence_records, or [].

    Expected evidence_records shape (existing pipeline):
      [{"url":..., "fingerprint":..., "fetched_at":..., "numbers":[{...candidate...}, ...]}, ...]
    We map:
      source_url <- url
      extracted_numbers <- numbers
    """
    try:
        if not isinstance(evidence_records, list) or not evidence_records:
            return []
        rebuilt = []
        for rec in evidence_records:
            if not isinstance(rec, dict):
                continue
            url = (rec.get("url") or rec.get("source_url") or "").strip()
            nums = rec.get("numbers") or rec.get("extracted_numbers") or []
            if not url or not isinstance(nums, list):
                continue
            rebuilt.append({
                "source_url": url,
                "url": url,  # legacy compatibility
                "fingerprint": rec.get("fingerprint") or rec.get("content_fingerprint"),
                "fetched_at": rec.get("fetched_at"),
                "extracted_numbers": nums,
            })
        # Deterministic ordering
        rebuilt.sort(key=lambda d: (str(d.get("source_url") or ""), str(d.get("fingerprint") or "")))
        return rebuilt
    except Exception:
        return []
# =====================================================================


# =====================================================================

# =====================================================================
# PATCH SHEETS_CACHE1 (ADDITIVE): In-run Google Sheets read caching + rate-limit fallback
# Why:
# - Google Sheets consumer quota is very low (e.g., 60 reads/min/user). Evolution may trigger
#   multiple reads (History, HistoryFull, Snapshots) within a single run.
# - When quota is exceeded (429 RESOURCE_EXHAUSTED), we should:
#     (1) reuse cached reads within the same run, and
#     (2) fall back to last cached value (if any) rather than hard-failing.
# Notes:
# - Additive only. No behavior changes unless we would otherwise exceed quota.
# - Cache is in-memory per Python process; it resets when the app restarts.
# =====================================================================
_SHEETS_READ_CACHE = {}
_SHEETS_READ_CACHE_TTL_SEC = 55  # keep under the 60s/min quota window
_SHEETS_LAST_READ_ERROR = None

def _sheets_now_ts():
    import time
    return time.time()

def _sheets_cache_get(key: str):
    try:
        item = _SHEETS_READ_CACHE.get(key)
        if not item:
            return None
        ts, val = item
        if (_sheets_now_ts() - ts) > _SHEETS_READ_CACHE_TTL_SEC:
            return None
        return val
    except Exception:
        return None

def _sheets_cache_set(key: str, val):
    try:
        _SHEETS_READ_CACHE[key] = (_sheets_now_ts(), val)
    except Exception:
        pass

def _is_sheets_rate_limit_error(err: Exception) -> bool:
    s = ""
    try:
        s = str(err) or ""
    except Exception:
        s = ""
    # Common markers seen via gspread/googleapiclient:
    markers = ["RESOURCE_EXHAUSTED", "Quota exceeded", "RATE_LIMIT_EXCEEDED", "429"]
    return any(m in s for m in markers)

def sheets_get_all_values_cached(ws, cache_key: str):
    """
    Cached wrapper for ws.get_all_values() with rate-limit fallback.
    cache_key should be stable for the worksheet (e.g., 'Snapshots', 'HistoryFull', 'History').
    """
    global _SHEETS_LAST_READ_ERROR
    key = f"get_all_values:{cache_key}"
    cached = _sheets_cache_get(key)
    if cached is not None:
        return cached
    try:
        # === PATCH SHEETS_CACHE1 (CONFLICT FIX, MINIMAL): call the underlying worksheet read ===
        # Previous draft accidentally recursed into itself and referenced an undefined variable.
        # This is a direct execution conflict fix (no behavior change intended beyond correctness).
        values = ws.get_all_values() if ws else []
        _sheets_cache_set(key, values)
        return values
    except Exception as e:
        _SHEETS_LAST_READ_ERROR = str(e)
        # Rate-limit fallback: return last cached value if we have one, else empty list
        if _is_sheets_rate_limit_error(e):
            stale = _SHEETS_READ_CACHE.get(key)
            if stale and isinstance(stale, tuple) and len(stale) == 2:
                return stale[1]
            return []
        raise

# =====================================================================
# PATCH SS2 (ADDITIVE): Google Sheets snapshot store (separate worksheet)
# Purpose:
#   - Persist full baseline_sources_cache inside the same Spreadsheet
#     but in a dedicated worksheet (tab), chunked across rows.
#   - Enables deterministic rehydration for evolution without refetch.
# Notes:
#   - Write-once semantics by source_snapshot_hash.
#   - Chunking and reassembly are deterministic (part_index ordering).
# =====================================================================
def get_google_spreadsheet():
    """Connect to Google Spreadsheet (cached connection if available)."""
    try:
        # If get_google_sheet() exists and already opened the spreadsheet as sheet.sheet1,
        # we re-open to obtain the Spreadsheet handle (additive; avoids refactoring).
        import streamlit as st
        from google.oauth2.service_account import Credentials
        import gspread

        SCOPES = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=SCOPES
        )
        client = gspread.authorize(creds)
        spreadsheet_name = st.secrets.get("google_sheets", {}).get("spreadsheet_name", "Yureeka_JSON")
        return client.open(spreadsheet_name)
    except Exception:
        return None

def _ensure_snapshot_worksheet(spreadsheet, title: str = "Snapshots"):
    """Ensure a worksheet tab exists for snapshot storage."""
    try:
        if not spreadsheet:
            return None
        try:
            ws = spreadsheet.worksheet(title)
            return ws
        except Exception:
            # Create with a reasonable default size; Sheets can expand.
            ws = spreadsheet.add_worksheet(title=title, rows=2000, cols=8)
            try:
                ws.append_row(
                    ["source_snapshot_hash", "part_index", "total_parts", "payload_part", "created_at", "code_version", "fingerprints_sig", "sha256"],
                    value_input_option="RAW",
                )
            except Exception:
                pass
            return ws
    except Exception:
        return None

def store_full_snapshots_to_sheet(baseline_sources_cache: list, source_snapshot_hash: str, worksheet_title: str = "Snapshots", chunk_chars: int = 45000) -> str:
    """
    Store full snapshots to a dedicated worksheet tab in chunked rows.
    Returns a ref string like: 'gsheet:Snapshots:<hash>'
    """
    import json, hashlib
    if not source_snapshot_hash:
        return ""
    if not isinstance(baseline_sources_cache, list) or not baseline_sources_cache:
        return ""

    try:
        ss = get_google_spreadsheet()
        ws = _ensure_snapshot_worksheet(ss, worksheet_title) if ss else None
        if not ws:
            return ""

        # Write-once: if hash already present, do not write again.
        try:
            # Find any existing rows for this hash (skip header)
            existing = ws.findall(source_snapshot_hash)
            if existing:
                return f"gsheet:{worksheet_title}:{source_snapshot_hash}"
        except Exception:
            # best effort; continue to attempt write
            pass

        payload = json.dumps(baseline_sources_cache, ensure_ascii=False, default=str)
        sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        # deterministic chunking
        chunk_size = max(1000, int(chunk_chars or 45000))
        parts = [payload[i:i+chunk_size] for i in range(0, len(payload), chunk_size)]
        total = len(parts)

        # Optional fingerprints signature (stable)
        pairs = []
        for sr in baseline_sources_cache:
            if isinstance(sr, dict):
                u = (sr.get("source_url") or sr.get("url") or "").strip()
                fp = (sr.get("fingerprint") or sr.get("content_fingerprint") or "").strip()
                if u:
                    pairs.append((u, fp))
        pairs.sort()
        fingerprints_sig = "|".join([f"{u}#{fp}" for (u, fp) in pairs]) if pairs else ""

        from datetime import datetime
        created_at = datetime.utcnow().isoformat() + "Z"

        # Append rows in order (deterministic)
        code_version = ""
        try:
            # best effort: use global if exists
            code_version = globals().get("CODE_VERSION") or ""
        except Exception:
            code_version = ""

        # Use append_rows if available, else append_row in loop
        rows = []
        for idx, part in enumerate(parts):
            rows.append([source_snapshot_hash, idx, total, part, created_at, code_version, fingerprints_sig, sha])

        try:
            ws.append_rows(rows, value_input_option="RAW")
        except Exception:
            for r in rows:
                try:
                    ws.append_row(r, value_input_option="RAW")
                except Exception:
                    # partial failure: still return empty to avoid false pointer
                    return ""

        return f"gsheet:{worksheet_title}:{source_snapshot_hash}"
    except Exception:
        return ""

def load_full_snapshots_from_sheet(source_snapshot_hash: str, worksheet_title: str = "Snapshots") -> list:
    """Load and reassemble full snapshots list from a dedicated worksheet."""
    import json, hashlib
    if not source_snapshot_hash:
        return []
    try:
        ss = get_google_spreadsheet()
        ws = ss.worksheet(worksheet_title) if ss else None
        if not ws:
            return []

        # =====================================================================
        # PATCH SNAPLOAD1 (ADDITIVE): cache-safe snapshot read fallback
        # Why:
        # - If a prior read hit quota / partial failure and we cached [], evolution
        #   will permanently think "no snapshots exist" until cache clears.
        # Behavior:
        # - Try cached read first (fast)
        # - If empty/too small, do ONE direct read to bypass stale empty cache
        # =====================================================================
        values = []
        try:
            values = sheets_get_all_values_cached(ws, cache_key=worksheet_title)
        except Exception:
            values = []

        if not values or len(values) < 2:
            # Direct retry (best-effort)
            try:
                direct = ws.get_all_values()
                if direct and len(direct) >= 2:
                    values = direct
            except Exception:
                pass
        # =====================================================================
        # END PATCH SNAPLOAD1 (ADDITIVE)
        # =====================================================================

        if not values or len(values) < 2:
            return []

        header = values[0] or []
        # Expect at least: source_snapshot_hash, part_index, total_parts, payload_part
        try:
            col_h = header.index("source_snapshot_hash")
            col_i = header.index("part_index")
            col_t = header.index("total_parts")
            col_p = header.index("payload_part")
            col_sha = header.index("sha256") if "sha256" in header else None
        except Exception:
            # If headers are missing/misaligned, bail safely
            return []

        # Filter rows for this hash
        rows = []
        for r in values[1:]:
            try:
                if len(r) > col_h and r[col_h] == source_snapshot_hash:
                    rows.append(r)
            except Exception:
                continue

        if not rows:
            return []

        # Deterministic sort by part_index
        def _safe_int(x):
            try:
                return int(x)
            except Exception:
                return 0
        rows.sort(key=lambda r: _safe_int(r[col_i] if len(r) > col_i else 0))

        # Reassemble
        payload_parts = []
        for r in rows:
            if len(r) > col_p:
                payload_parts.append(r[col_p] or "")
        payload = "".join(payload_parts)

        # Optional integrity check
        try:
            if col_sha is not None and len(rows[0]) > col_sha:
                expected = rows[0][col_sha] or ""
                if expected:
                    actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                    if actual != expected:
                        return []
        except Exception:
            pass

        try:
            data = json.loads(payload)
            return data if isinstance(data, list) else []
        except Exception:
            return []
    except Exception:
        return []

# =====================================================================
# PATCH HF4 (ADDITIVE): HistoryFull payload rehydration support
# Why:
# - Evolution may receive a sheets-safe wrapper that omits primary_response,
#   metric_schema_frozen, metric_anchors, etc.
# - When wrapper includes full_store_ref ("gsheet:HistoryFull:<analysis_id>"),
#   we can deterministically load the full analysis payload (no re-fetch).
# Notes:
# - Additive only. Safe no-op if sheet/tab not present.
# =====================================================================
def load_full_history_payload_from_sheet(analysis_id: str, worksheet_title: str = "HistoryFull") -> dict:
    """Load and reassemble a full analysis payload dict from HistoryFull worksheet."""
    import json, hashlib
    if not analysis_id:
        return {}
    try:
        ss = get_google_spreadsheet()
        ws = ss.worksheet(worksheet_title) if ss else None
        if not ws:
            return {}

        values = sheets_get_all_values_cached(ws, cache_key=worksheet_title)
        if not values or len(values) < 2:
            return {}

        header = values[0]
        def _idx(name):
            try:
                return header.index(name)
            except Exception:
                return None

        i_id = _idx("analysis_id")
        i_part = _idx("part_index")
        i_total = _idx("total_parts")
        i_payload = _idx("payload_part")
        i_sha = _idx("sha256")

        if i_id is None or i_part is None or i_total is None or i_payload is None:
            return {}

        rows = []
        expected_sha = None
        for r in values[1:]:
            if not r or i_id >= len(r):
                continue
            if str(r[i_id]).strip() != str(analysis_id).strip():
                continue
            try:
                part_index = int(r[i_part]) if i_part < len(r) else 0
            except Exception:
                part_index = 0
            if i_sha is not None and i_sha < len(r):
                if r[i_sha]:
                    expected_sha = r[i_sha]
            payload_part = r[i_payload] if i_payload < len(r) else ""
            rows.append((part_index, payload_part))

        if not rows:
            return {}

        rows.sort(key=lambda x: x[0])
        payload = "".join([p for _, p in rows])

        if expected_sha:
            sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if sha != expected_sha:
                return {}

        data = json.loads(payload)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
# =====================================================================

# =====================================================================


def fingerprint_text(text: str) -> str:
    """Stable short fingerprint for fetched content (deterministic)."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]

def attach_source_snapshots_to_analysis(analysis: dict, web_context: dict) -> dict:
    """
    Attach stable source snapshots (from web_context.scraped_meta) into analysis.

    Enhancements (v7_34 patch):
    - Ensures scraped_meta.extracted_numbers is always list-like
    - Adds RANGE capture per canonical metric using admitted snapshots:
        primary_metrics_canonical[ckey]["value_range"] = {min,max,n,examples}
      This restores earlier "range vs point estimate" behavior in a compatible way.
    """
    import re
    from datetime import datetime, timezone

    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _fingerprint(text: str) -> str:
        try:
            fn = globals().get("fingerprint_text")
            if callable(fn):
                return fn(text)
        except Exception:
            pass
        try:
            import hashlib
            t = re.sub(r"\s+", " ", (text or "").strip().lower())
            return hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:12]
        except Exception:
            return ""

    # =========================================================================
    # PATCH N1 (ADDITIVE): stable anchor_hash fallback helper for snapshots
    # - Does NOT change existing behavior if anchor_hash already present.
    # =========================================================================
    def _sha1(s: str) -> str:
        try:
            import hashlib
            return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return ""
    # =========================================================================

    # =========================================================================
    # PATCH N2 (ADDITIVE): optional canonicalizer hook for snapshot numbers
    # - Ensures unit_tag/unit_family/base_unit/value_norm are present when possible.
    # - No behavior change if helper missing.
    # =========================================================================
    _canon_fn = globals().get("canonicalize_numeric_candidate")
    def _maybe_canonicalize(n: dict) -> dict:
        try:
            if callable(_canon_fn):
                return _canon_fn(dict(n))
        except Exception:
            pass
        return dict(n)
    # =========================================================================

    def _parse_num(value, unit_hint=""):
        try:
            fn = globals().get("parse_human_number")
            if callable(fn):
                return fn(str(value), unit_hint)
        except Exception:
            pass
        # fallback
        try:
            s = str(value).strip().replace(",", "")
            if not s:
                return None
            return float(re.findall(r"-?\d+(?:\.\d+)?", s)[0])
        except Exception:
            return None

    def _unit_family_from_metric(mdef: dict) -> str:
        # prefer metric schema
        uf = (mdef or {}).get("unit_family") or ""
        uf = str(uf).lower().strip()
        if uf in ("percent", "pct"):
            return "PCT"
        if uf in ("currency",):
            return "CUR"
        if uf in ("magnitude", "unit_sales", "other"):
            return "MAG"
        return "OTHER"

    def _cand_unit_family(cunit: str, craw: str) -> str:
        u = (cunit or "").strip()
        r = (craw or "")
        uu = u.upper()
        ru = r.upper()

        # Percent
        if uu == "%" or "%" in ru:
            return "PCT"

        # Energy
        if any(x in (u or "").lower() for x in ["twh", "gwh", "mwh", "kwh"]) or any(x in (r or "").lower() for x in ["twh", "gwh", "mwh", "kwh"]):
            return "ENERGY"

        # Currency (symbol/code presence)
        #if any(x in ru for x in ["$", "USD", "SGD", "EUR", "GBP", "S$"]) or uu in ("USD", "SGD", "EUR", "GBP"):
        #    return "CUR"

        if re.search(r"(\$|S\$|€|£)\s*\d", r) or any(x in ru for x in ["USD", "SGD", "EUR", "GBP"]) or uu in ("USD","SGD","EUR","GBP"):
            return "CUR"


        # Magnitude (case-insensitive)
        if uu in ("K", "M", "B", "T") or (u or "").lower() in ("k", "m", "b", "t"):
            return "MAG"

        return "OTHER"

    def _tokenize(s: str):
        return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2]

    def _safe_norm_unit_tag(x: str) -> str:
        try:
            fn = globals().get("normalize_unit_tag")
            if callable(fn):
                return fn(x or "")
        except Exception:
            pass
        return (x or "").strip()


    # -----------------------------
    # Build baseline_sources_cache from scraped_meta (snapshot-friendly)
    # -----------------------------
    baseline_sources_cache = []
    scraped_meta = (web_context or {}).get("scraped_meta") or {}
    if isinstance(scraped_meta, dict):
        for url, meta in scraped_meta.items():
            if not isinstance(meta, dict):
                continue
            nums = meta.get("extracted_numbers") or []
            if nums is None or not isinstance(nums, list):
                nums = []

            content = meta.get("content") or meta.get("clean_text") or (web_context.get("scraped_content", {}) or {}).get(url, "") or ""

            baseline_sources_cache.append({
                "url": url,
                "status": "fetched" if str(meta.get("status_detail", "")).startswith("success") or meta.get("status") == "fetched" else "failed",
                "status_detail": meta.get("status_detail") or meta.get("status") or "",
                "numbers_found": int(meta.get("numbers_found") or (len(nums) if isinstance(nums, list) else 0)),
                "fetched_at": meta.get("fetched_at") or _now_iso(),
                "fingerprint": meta.get("fingerprint") or _fingerprint(content),

                # =====================================================================
                # PATCH N1 (+ N2) (ADDITIVE): preserve full candidate record in snapshots
                # - This is critical for:
                #   * range gating (metric-aware)
                #   * schema-first attribution
                #   * evolution rebuild (anchor_hash + value_norm + unit_family)
                # - Backward compatible: only adds keys; existing keys unchanged.
                # =====================================================================
                "extracted_numbers": [
                    (lambda nn: {
                        "value": nn.get("value"),
                        "unit": nn.get("unit"),
                        "raw": nn.get("raw"),
                        "context_snippet": (nn.get("context_snippet") or nn.get("context") or "")[:240],

                        # keep existing anchor_hash if present; else stable fallback
                        "anchor_hash": (
                            nn.get("anchor_hash")
                            or _sha1(
                                f"{url}|{str(nn.get('raw') or '')}|{(nn.get('context_snippet') or nn.get('context') or '')[:240]}"
                            )
                        ),

                        "source_url": nn.get("source_url") or url,

                        # ---- Additive: junk tagging & deterministic offsets ----
                        "is_junk": nn.get("is_junk"),
                        "junk_reason": nn.get("junk_reason"),
                        "start_idx": nn.get("start_idx"),
                        "end_idx": nn.get("end_idx"),

                        # ---- Additive: normalized unit fields (if already present or canonicalized) ----
                        "unit_tag": nn.get("unit_tag"),
                        "unit_family": nn.get("unit_family"),
                        "base_unit": nn.get("base_unit"),
                        "multiplier_to_base": nn.get("multiplier_to_base"),
                        "value_norm": nn.get("value_norm"),

                        # ---- Additive: semantic association tags (if present) ----
                        "measure_kind": nn.get("measure_kind"),
                        "measure_assoc": nn.get("measure_assoc"),
                    })(_maybe_canonicalize(n))
                    for n in nums
                    if isinstance(n, dict)
                ]
                # =====================================================================
            })

    if baseline_sources_cache:

        # ---- ADDITIVE: stable ordering of snapshots (Change #2) ----
        for s in (baseline_sources_cache or []):
            if isinstance(s, dict) and isinstance(s.get("extracted_numbers"), list):

                # =========================================================================
                # PATCH N3 (ADDITIVE): guard sort_snapshot_numbers if not defined
                # =========================================================================
                try:
                    if "sort_snapshot_numbers" in globals() and callable(globals()["sort_snapshot_numbers"]):
                        s["extracted_numbers"] = sort_snapshot_numbers(s["extracted_numbers"])
                    else:
                        # safe fallback: anchor_hash then raw
                        s["extracted_numbers"] = sorted(
                            s["extracted_numbers"],
                            key=lambda x: (str((x or {}).get("anchor_hash") or ""), str((x or {}).get("raw") or ""))
                        )
                except Exception:
                    pass
                # =========================================================================

                s["numbers_found"] = len(s["extracted_numbers"])

        baseline_sources_cache = sorted(
            baseline_sources_cache,
            key=lambda x: str((x or {}).get("url") or "")
        )
        # -----------------------------------------------------------

        analysis["baseline_sources_cache"] = baseline_sources_cache
        analysis.setdefault("results", {})
        if isinstance(analysis["results"], dict):
            analysis["results"]["baseline_sources_cache"] = baseline_sources_cache


    # -----------------------------
    # RANGE capture for canonical metrics
    # -----------------------------
    pmc = analysis.get("primary_response", {}).get("primary_metrics_canonical") if isinstance(analysis.get("primary_response"), dict) else analysis.get("primary_metrics_canonical")
    schema = analysis.get("primary_response", {}).get("metric_schema_frozen") if isinstance(analysis.get("primary_response"), dict) else analysis.get("metric_schema_frozen")

    # Support both placements (your JSON seems to store these at top-level primary_response)
    if pmc is None and isinstance(analysis.get("primary_response"), dict):
        pmc = analysis["primary_response"].get("primary_metrics_canonical")
    if schema is None and isinstance(analysis.get("primary_response"), dict):
        schema = analysis["primary_response"].get("metric_schema_frozen")

    if isinstance(pmc, dict) and isinstance(schema, dict) and baseline_sources_cache:
        # flatten candidates
        all_cands = []
        for sr in baseline_sources_cache:
            for n in (sr.get("extracted_numbers") or []):
                if isinstance(n, dict):
                    all_cands.append(n)

        for ckey, m in pmc.items():
            if not isinstance(m, dict):
                continue
            mdef = schema.get(ckey) or {}
            uf = _unit_family_from_metric(mdef)
            keywords = mdef.get("keywords") or []

            kw_tokens = []
            for k in (keywords or []):
                kw_tokens.extend(_tokenize(str(k)))

            kw_tokens.extend(_tokenize(m.get("name") or m.get("original_name") or ""))
            kw_tokens = list(dict.fromkeys([t for t in kw_tokens if len(t) > 2]))[:40]

            vals = []
            examples = []

            for cand in all_cands:
                craw = str(cand.get("raw") or "")
                cunit = str(cand.get("unit") or "")
                ctx = str(cand.get("context_snippet") or cand.get("context") or "")

                # family gate
                cf = _cand_unit_family(cunit, craw)
                if uf == "PCT" and cf != "PCT":
                    continue
                if uf == "CUR" and cf != "CUR":
                    continue
                # MAG: allow MAG/OTHER but avoid CUR/PCT
                if uf == "MAG" and cf in ("CUR", "PCT"):
                    continue

                # NEW (additive): metric-aware magnitude gate
                if uf == "MAG":

                    cand_tag = _safe_norm_unit_tag(cunit or craw)
                    exp_tag = _safe_norm_unit_tag((mdef.get("unit") or "") or (m.get("unit") or ""))


                    if exp_tag in ("K", "M", "B", "T"):
                        if cand_tag != exp_tag:
                            continue
                    else:
                        if cand_tag not in ("K", "M", "B", "T"):
                            continue

                # token overlap gate
                c_tokens = set(_tokenize(ctx))
                if kw_tokens:
                    overlap = sum(1 for t in kw_tokens if t in c_tokens)
                    if overlap < max(1, min(3, len(kw_tokens) // 8)):
                        continue

                v = _parse_num(cand.get("value"), cunit) or _parse_num(craw, cunit)
                if v is None:
                    continue

                vals.append(float(v))
                if len(examples) < 5:
                    examples.append({
                        "raw": craw[:32],
                        "source_url": cand.get("source_url"),
                        "context_snippet": ctx[:180]
                    })

            if len(vals) >= 2:
                vmin = min(vals)
                vmax = max(vals)
                if abs(vmax - vmin) > max(1e-9, abs(vmin) * 0.02):
                    m["value_range"] = {
                        "min": vmin,
                        "max": vmax,
                        "n": len(vals),
                        "examples": examples,
                        "method": "snapshot_candidates"
                    }
                    try:
                        unit_disp = m.get("unit") or ""
                        m["value_range_display"] = f"{vmin:g}–{vmax:g} {unit_disp}".strip()
                    except Exception:
                        pass
    # =====================================================================
    # PATCH V1 (ADDITIVE): analysis & schema version stamping
    # - Pure metadata, NO logic impact
    # - Allows downstream drift attribution:
    #     * pipeline changes vs source changes
    # =====================================================================
    analysis.setdefault("analysis_pipeline_version", "v7_41_endstate_wip_1")
    analysis.setdefault("metric_identity_version", "canon_v2_dim_safe")
    analysis.setdefault("schema_freeze_version", 1)
    # =====================================================================

    # =========================
    # VERSION STAMP (ADDITIVE)
    # =========================
    analysis.setdefault("code_version", CODE_VERSION)
    # =========================


    return analysis



def normalize_unit(unit: str) -> str:
    """Normalize unit to one of: T/B/M/%/'' (deterministic)."""
    if not unit:
        return ""
    u = unit.strip().upper().replace("USD", "").replace("$", "").replace(" ", "")
    if u in ["TRILLION", "T"]:
        return "T"
    if u in ["BILLION", "B"]:
        return "B"
    if u in ["MILLION", "M"]:
        return "M"
    if u in ["PERCENT", "%"]:
        return "%"
    if u in ["K", "THOUSAND"]:
        return "K"
    return u

def normalize_currency_prefix(raw: str) -> bool:
    """True if looks like a currency number ($/USD)."""
    if not raw:
        return False
    s = raw.strip().upper()
    return s.startswith("$") or " USD" in s or s.startswith("USD")

def is_likely_junk_context(ctx: str) -> bool:
    """
    Returns True if a context snippet strongly indicates the number is coming from
    HTML/JS/CSS/asset junk (srcset resize params, scripts, svg path data, etc.)
    rather than real narrative/tabular data.
    """
    import re

    c = (ctx or "").strip()
    if not c:
        return True

    cl = c.lower()

    # Too much binary / garbled text (common when PDF bytes leak through)
    non_print = sum(1 for ch in c if ord(ch) < 9 or (13 < ord(ch) < 32))
    if non_print > 0:
        return True

    # Lots of replacement chars / unusual glyphs → decode garbage
    bad_glyphs = c.count("\ufffd")
    if bad_glyphs >= 1:
        return True

    # Very long uninterrupted “code-ish” context
    if len(c) > 260 and ("{" in c and "}" in c) and ("function" in cl or "var " in cl or "const " in cl):
        return True

    # Hard “asset / markup / script” indicators
    hard_hints = [
        "srcset=", "resize=", "quality=", "offsc", "offscreencanvas", "createelement(\"canvas\")",
        "willreadfrequently", "function(", "webpack", "window.", "document.", "var ", "const ",
        "<script", "</script", "<style", "</style", "text/javascript", "application/javascript",
        "og:image", "twitter:image", "meta property=", "content=\"width=device-width",
        "/wp-content/", ".jpg", ".jpeg", ".png", ".svg", ".webp", ".css", ".js", ".woff", ".woff2",
        "data:image", "base64,", "viewbox", "path d=", "d=\"m", "aria-label=", "class=\""
    ]
    if any(h in cl for h in hard_hints):
        return True

    # SVG path command patterns like "h4.16v-2.56"
    if re.search(r"(?:^|[^a-z0-9])[a-z]\d+(?:\.\d+)?[a-z]-?\d", cl):
        return True

    # Image resize query param like "...jpg?resize=770%2C513..."
    if re.search(r"resize=\d+%2c\d+", cl):
        return True

    # Phone / tracking / footer junk often has lots of separators and few letters
    letters = sum(1 for ch in c if ch.isalpha())
    if len(c) >= 120 and letters / max(1, len(c)) < 0.08:
        return True

    return False


def parse_human_number(value_str: str, unit: str) -> Optional[float]:
    """
    Parse number + unit into a comparable float scale.
    - For T/B/M: returns value in billions (B) to compare apples-to-apples.
    - For %: returns numeric percent.
    """
    if value_str is None:
        return None

    s = str(value_str).strip()
    if not s:
        return None

    # remove currency symbols/commas/space
    s = s.replace("$", "").replace(",", "").strip()

    # handle parentheses for negatives e.g. (12.3)
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()

    try:
        v = float(s)
    except Exception:
        return None

    u = normalize_unit(unit)

    # Normalize magnitudes into BILLIONS for currency-like units
    if u == "T":
        return v * 1000.0
    if u == "B":
        return v
    if u == "M":
        return v / 1000.0
    if u == "K":
        return v / 1_000_000.0

    # Percent: keep as percent number
    if u == "%":
        return v

    # Unknown unit: leave as-is (still useful for ratio filtering)
    return v

def build_prev_numbers(prev_metrics: Dict) -> Dict[str, Dict]:
    """
    Build previous metric lookup keyed by metric_name string.
    Stores:
      - parsed numeric value (for matching)
      - normalized unit (for gating)
      - raw display string INCLUDING currency/magnitude (for dashboards + evolution JSON)
      - raw_value/raw_unit for debugging
    """
    def _format_raw_display(value: Any, unit: str) -> str:
        v = "" if value is None else str(value).strip()
        u = (unit or "").strip()

        if not v:
            return ""

        # Currency prefix handling (SGD/USD keywords OR symbol prefixes)
        currency = ""
        u_nospace = u.replace(" ", "")

        if u_nospace.upper().startswith("SGD"):
            currency = "S$"
            u_tail = u_nospace[3:]
        elif u_nospace.upper().startswith("USD"):
            currency = "$"
            u_tail = u_nospace[3:]
        elif u_nospace.startswith("S$"):
            currency = "S$"
            u_tail = u_nospace[2:]
        elif u_nospace.startswith("$"):
            currency = "$"
            u_tail = u_nospace[1:]
        else:
            u_tail = u_nospace

        # Percent special case
        if u_tail == "%":
            return f"{v}%"

        # Word scales
        if "billion" in u.lower():
            return f"{currency}{v} billion".strip()
        if "million" in u.lower():
            return f"{currency}{v} million".strip()

        # Compact suffix (B/M/K/T)
        if u_tail.upper() in {"T", "B", "M", "K"}:
            return f"{currency}{v}{u_tail.upper()}".strip()

        # Fallback
        return f"{currency}{v} {u}".strip()

    prev_numbers: Dict[str, Dict] = {}
    for key, metric in (prev_metrics or {}).items():
        if not isinstance(metric, dict):
            continue

        metric_name = metric.get("name", key)
        raw_value = metric.get("value", "")
        raw_unit = metric.get("unit", "")

        val = parse_human_number(str(raw_value), raw_unit)
        if val is None:
            continue

        prev_numbers[metric_name] = {
            "value": val,
            "unit": normalize_unit(raw_unit),
            "raw": _format_raw_display(raw_value, raw_unit),   # ✅ now includes currency + unit
            "raw_value": raw_value,
            "raw_unit": raw_unit,
            "keywords": extract_context_keywords(metric_name),
        }

    return prev_numbers

def _extract_baseline_cache(previous_data: dict) -> list:
    """
    Pull prior source snapshots from any known places v7.x stores them.
    Returns a list of source_result-like dicts, or [].
    """
    pd = previous_data or {}
    pr = (pd.get("primary_response") or {}) if isinstance(pd.get("primary_response"), dict) else {}

    for obj in [
        pd.get("baseline_sources_cache"),
        (pd.get("results") or {}).get("baseline_sources_cache") if isinstance(pd.get("results"), dict) else None,
        (pd.get("results") or {}).get("source_results") if isinstance(pd.get("results"), dict) else None,
        pd.get("source_results"),
        pr.get("baseline_sources_cache"),
        (pr.get("results") or {}).get("source_results") if isinstance(pr.get("results"), dict) else None,
    ]:
        if isinstance(obj, list) and obj:
            return obj

    return []


def _extract_query_from_previous(previous_data: dict) -> str:
    """
    Try to recover the original user query/topic from the saved analysis object.
    v7.27 commonly uses 'question'.
    """
    pd = previous_data or {}
    if isinstance(pd.get("question"), str) and pd["question"].strip():
        return pd["question"].strip()

    pr = pd.get("primary_response") or {}
    if isinstance(pr, dict):
        if isinstance(pr.get("question"), str) and pr["question"].strip():
            return pr["question"].strip()
        if isinstance(pr.get("query"), str) and pr["query"].strip():
            return pr["query"].strip()

    meta = pd.get("meta") or {}
    if isinstance(meta, dict) and isinstance(meta.get("question"), str) and meta["question"].strip():
        return meta["question"].strip()

    return ""

def _build_source_snapshots_from_web_context(web_context: dict) -> list:
    """
    Convert fetch_web_context() output (scraped_meta) into evolution snapshots.

    Preferred inputs:
      - web_context["scraped_meta"][url]["extracted_numbers"] (analysis-aligned)

    Safety-net hard gates (small set):
      1) homepage-like URLs downweighted + tagged
      2) nav/chrome/junk context downweighted
      3) year-only suppression (e.g., raw == "2024" and no unit/context)
      4) light topic gate (requires minimal overlap with query tokens)
    """
    import hashlib
    from datetime import datetime
    from urllib.parse import urlparse
    import re

    def _sha1(s: str) -> str:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

    def _now() -> str:
        try:
            return datetime.utcnow().isoformat() + "+00:00"
        except Exception:
            return datetime.now().isoformat()

    def _is_homepage_url(u: str) -> bool:
        try:
            p = urlparse((u or "").strip())
            path = (p.path or "").strip()
            if path in ("", "/"):
                return True
            low = path.lower().rstrip("/")
            if low in ("/index", "/index.html", "/index.htm", "/home", "/default", "/default.aspx"):
                return True
            return False
        except Exception:
            return False



    def _tokenize(s: str) -> list:
        toks = re.findall(r"[a-z0-9]+", (s or "").lower())
        stop = {"the","and","or","of","in","to","for","by","from","with","on","at","as","a","an","is","are","this","that"}
        return [t for t in toks if len(t) >= 4 and t not in stop]

    def _looks_like_year_only(n: dict) -> bool:
        try:
            raw = str(n.get("raw") or "").strip()
            unit = str(n.get("unit") or "").strip()
            ctx = str(n.get("context") or n.get("context_snippet") or "").strip()
            # exactly 4 digits year and nothing else
            if re.fullmatch(r"(19|20)\d{2}", raw) and not unit:
                # if context is empty or super short, treat as junk
                if len(ctx) < 12:
                    return True
            return False
        except Exception:
            return False

    def _is_chrome_ctx(ctx: str) -> bool:
        if not ctx:
            return False
        low = ctx.lower()
        for h in globals().get("NON_DATA_CONTEXT_HINTS", []) or []:
            if h in low:
                return True
        return False

    if not isinstance(web_context, dict):
        return []

    scraped_meta = web_context.get("scraped_meta") or {}
    if not isinstance(scraped_meta, dict) or not scraped_meta:
        return []

    query = (web_context.get("query") or "")
    q_toks = set(_tokenize(query))

    out = []

    for url, meta in scraped_meta.items():
        if not isinstance(meta, dict):
            continue

        url_s = str(url or meta.get("url") or "").strip()
        if not url_s:
            continue

        extracted = meta.get("extracted_numbers") or []
        if not isinstance(extracted, list):
            extracted = []

        fp = meta.get("fingerprint") or meta.get("extract_hash") or meta.get("content_fingerprint")
        if fp and not isinstance(fp, str):
            fp = str(fp)
        if not fp and isinstance(meta.get("clean_text"), str):
            fp = _sha1(meta["clean_text"][:200000])

        status_detail = meta.get("status_detail") or meta.get("status") or ""
        fetched_ok = str(status_detail).startswith("success") or meta.get("status") == "fetched"

        is_homepage = _is_homepage_url(url_s)

        cleaned_numbers = []
        for n in extracted:
            if not isinstance(n, dict):
                continue

            # ---- Hard gate: year-only suppression ----
            if _looks_like_year_only(n):
                continue

            value = n.get("value")
            raw = n.get("raw")
            unit = n.get("unit")
            ctx = n.get("context") or n.get("context_snippet") or ""

            # normalize context
            ctx_s = ctx if isinstance(ctx, str) else ""
            ctx_s = ctx_s.strip()

            # ---- Hard gate: chrome/nav rejection (soft) ----
            chrome_ctx = _is_chrome_ctx(ctx_s)

            # ---- Light topic gate (soft): require some overlap with query tokens ----
            # This is intentionally mild: it *downweights* rather than drops everything.
            ctx_toks = set(_tokenize(ctx_s))
            tok_overlap = len(q_toks.intersection(ctx_toks)) if q_toks and ctx_toks else 0

            # quality scoring (small + interpretable)
            quality = 1.0
            reasons = []

            if is_homepage:
                quality *= 0.25
                reasons.append("homepage_like")

            if chrome_ctx:
                quality *= 0.40
                reasons.append("chrome_context")

            if q_toks and tok_overlap == 0:
                quality *= 0.55
                reasons.append("topic_miss")

            # cap/trim context snippet for JSON size
            ctx_snip = ctx_s[:240]

            cleaned_numbers.append({
                "value": value,
                "unit": unit,
                "raw": raw,
                "source_url": n.get("source_url") or url_s,
                "context_snippet": ctx_snip,
                "anchor_hash": n.get("anchor_hash") or _sha1(f"{url_s}|{ctx_snip}|{raw}|{unit}"),
                # Debug fields for tuning:
                "quality_score": round(float(quality), 3),
                "quality_reasons": reasons,
                "topic_overlap": tok_overlap,
            })

        out.append({
            "url": url_s,
            "status": "fetched_extracted" if cleaned_numbers else ("fetched" if fetched_ok else "failed"),
            "status_detail": status_detail,
            "numbers_found": len(cleaned_numbers),
            "fingerprint": fp or "",
            "fetched_at": meta.get("fetched_at") or _now(),
            "is_homepage_like": bool(is_homepage),
            "extracted_numbers": cleaned_numbers,
        })

    return out



def _build_source_snapshots_from_baseline_cache(baseline_cache: list) -> list:
    """
    Normalize prior cached source_results (from previous run) into a consistent schema.

    Tightening:
      - Detect domain-only/homepage URLs and label them (same as web_context snapshots)
      - Keep backward compatible fields; only add new fields.
    """
    from urllib.parse import urlparse

    def _is_homepage_url(u: str) -> bool:
        try:
            p = urlparse((u or "").strip())
            path = (p.path or "").strip()
            if path in ("", "/"):
                return True
            low = path.lower().rstrip("/")
            if low in ("/index", "/index.html", "/index.htm", "/home", "/default", "/default.aspx"):
                return True
            return False
        except Exception:
            return False

    out = []
    if not isinstance(baseline_cache, list):
        return out

    for sr in baseline_cache:
        if not isinstance(sr, dict):
            continue

        url = sr.get("url") or sr.get("source_url")
        if not url:
            continue
        url_s = str(url).strip()
        if not url_s:
            continue

        extracted = sr.get("extracted_numbers") or []
        if not isinstance(extracted, list):
            extracted = []

        cleaned = []
        for n in extracted:
            if not isinstance(n, dict):
                continue
            cleaned.append({
                "value": n.get("value"),
                "unit": n.get("unit"),
                "raw": n.get("raw"),
                "source_url": n.get("source_url") or url_s,
                "context": (n.get("context") or n.get("context_snippet") or "")[:220]
                if isinstance((n.get("context") or n.get("context_snippet")), str) else "",
            })

        fp = sr.get("fingerprint")
        if fp and not isinstance(fp, str):
            fp = str(fp)

        # --- homepage labeling (tightening #3) ---
        is_homepage = bool(sr.get("is_homepage")) or _is_homepage_url(url_s)
        quality_score = sr.get("quality_score")
        if quality_score is None:
            quality_score = 0.15 if is_homepage else 1.0

        skip_reason = sr.get("skip_reason") or ("homepage_url_low_signal" if is_homepage else "")

        host = sr.get("host") or ""
        path = sr.get("path") or ""
        if not host and not path:
            try:
                p = urlparse(url_s)
                host = p.netloc or ""
                path = p.path or ""
            except Exception:
                pass

        out.append({
            "url": url_s,
            "status": sr.get("status") or "",
            "status_detail": sr.get("status_detail") or "",
            "numbers_found": int(sr.get("numbers_found") or len(cleaned)),
            "fingerprint": fp,
            "fetched_at": sr.get("fetched_at"),
            "extracted_numbers": cleaned,

            # NEW debug fields (safe additions)
            "is_homepage": bool(is_homepage),
            "quality_score": float(quality_score),
            "skip_reason": skip_reason,
            "host": host,
            "path": path,
        })

    return out


def _merge_snapshots_prefer_cached_when_unchanged(current_snaps: list, cached_snaps: list) -> list:
    """
    Policy merge:
      - If current fingerprint matches cached fingerprint for same URL:
        reuse cached snapshot (even if live fetch worked)  ✅ point A
      - Else prefer current (fresh).
      - Add cached snapshots not present in current.
      - Also: if current numbers_found is 0 but cached has >0, reuse cached.
    """
    if not isinstance(current_snaps, list):
        current_snaps = []
    if not isinstance(cached_snaps, list):
        cached_snaps = []

    cached_by_url = {}
    for s in cached_snaps:
        if isinstance(s, dict) and s.get("url"):
            cached_by_url[str(s["url"])] = s

    merged = []
    seen = set()

    for cs in current_snaps:
        if not isinstance(cs, dict) or not cs.get("url"):
            continue
        url = str(cs["url"])
        seen.add(url)

        cached = cached_by_url.get(url)
        if not cached:
            merged.append(cs)
            continue

        cur_fp = cs.get("fingerprint")
        old_fp = cached.get("fingerprint")

        cur_nf = int(cs.get("numbers_found") or 0)
        old_nf = int(cached.get("numbers_found") or 0)

        # If current extraction is empty but cached had numbers, reuse cached.
        if cur_nf == 0 and old_nf > 0:
            merged.append(cached)
            continue

        # Fingerprint unchanged -> reuse cached even if live fetch worked.
        if cur_fp and old_fp and str(cur_fp) == str(old_fp):
            merged.append(cached)
        else:
            merged.append(cs)

    for url, cached in cached_by_url.items():
        if url not in seen:
            merged.append(cached)

    return merged


def _safe_parse_current_analysis(query: str, web_context: dict) -> dict:
    """
    Run the same analysis pipeline used in v7.27 to produce primary_response, but safely.
    Returns dict with at least {primary_response:{primary_metrics:{}}} or {} on failure.
    """
    import json
    qp = globals().get("query_perplexity")
    if not callable(qp):
        return {}

    try:
        raw = qp(query, web_context)
        if not raw or not isinstance(raw, str):
            return {}
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {}
        return {"primary_response": obj}
    except Exception:
        return {}


def diff_metrics_by_name(prev_response: dict, cur_response: dict):
    """
    Canonical-first diff with:
      - HARD STOP when prev canonical_key is missing in current (no name fallback)
      - Row-level metric_definition sourced from PREVIOUS (original new analysis) schema:
          prev_response['metric_schema_frozen'][canonical_key] (preferred)
          else prev_response['primary_metrics_canonical'][canonical_key]
      - Backward compatible: still returns 'name' (non-empty) and existing fields.

    Returns:
      metric_changes, unchanged, increased, decreased, found
    """
    import re

    # Defaults (used unless schema provides overrides)
    ABS_EPS = 1e-9
    REL_EPS = 0.0005

    def norm_name(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    def parse_num(v, unit=""):
        fn = globals().get("parse_human_number")
        if callable(fn):
            try:
                return fn(str(v), unit)
            except Exception:
                return None
        try:
            return float(str(v).replace(",", "").strip())
        except Exception:
            return None

    # =========================================================================
    # PATCH D1 (ADDITIVE): canonical numeric extractor
    # - Prefer value_norm/base_unit when present (analysis/evolution alignment)
    # - Fall back to existing parse_num(value, unit) when canonical fields missing
    # =========================================================================
    def get_canonical_value_and_unit(m: dict):
        """
        Returns: (val: float|None, unit: str)
        Priority:
          1) value_norm (float-like) + base_unit (if present)
          2) parse_num(value, unit)
        """
        m = m if isinstance(m, dict) else {}

        # 1) canonical path
        if m.get("value_norm") is not None:
            try:
                v = float(m.get("value_norm"))
                u = str(m.get("base_unit") or m.get("unit") or "").strip()
                return v, u
            except Exception:
                pass

        # 2) legacy parse path
        u = str(m.get("unit") or "").strip()
        v = parse_num(m.get("value"), u)
        return v, u
    # =========================================================================

    # =========================================================================
    # PATCH D0 (ADDITIVE): anchor helpers (drift=0 stability)
    # - If the SAME anchor_hash is present on both sides, treat metric as unchanged
    # - Pull prev anchor_hash from prev_response.metric_anchors when metric row lacks it
    # - Purely additive: does not remove or alter existing diff logic; only short-circuits
    #   to "unchanged" when anchors are identical.
    # =========================================================================
    def _get_anchor_hash_from_metric(m: dict):
        try:
            if isinstance(m, dict):
                ah = m.get("anchor_hash") or m.get("anchor") or m.get("anchorHash")
                return str(ah) if ah else None
        except Exception:
            pass
        return None

    def _get_prev_anchor_hash(prev_resp: dict, ckey: str, pm: dict):
        # 1) direct on metric row
        ah = _get_anchor_hash_from_metric(pm)
        if ah:
            return ah

        # 2) prev_response.metric_anchors[ckey].anchor_hash
        try:
            ma = (prev_resp or {}).get("metric_anchors")
            if isinstance(ma, dict):
                a = ma.get(ckey)
                if isinstance(a, dict):
                    ah2 = a.get("anchor_hash") or a.get("anchor")
                    if ah2:
                        return str(ah2)
        except Exception:
            pass

        return None

    def _get_cur_anchor_hash(cur_resp: dict, ckey: str, cm: dict):
        # 1) direct on metric row (evolution rebuild puts anchor_hash here)
        ah = _get_anchor_hash_from_metric(cm)
        if ah:
            return ah

        # 2) cur_response.metric_anchors[ckey].anchor_hash (if present)
        try:
            ma = (cur_resp or {}).get("metric_anchors")
            if isinstance(ma, dict):
                a = ma.get(ckey)
                if isinstance(a, dict):
                    ah2 = a.get("anchor_hash") or a.get("anchor")
                    if ah2:
                        return str(ah2)
        except Exception:
            pass

        return None
    # =========================================================================

    # =========================================================================
    # PATCH MA2 (ADDITIVE): wire metric_anchors into row fields
    # - Populate context_snippet/source_url from prev_response.metric_anchors[ckey] when available
    # - Keep existing anchor_same logic untouched; this is output enrichment only
    # =========================================================================
    def _get_prev_anchor_obj(prev_resp: dict, ckey: str):
        try:
            ma = (prev_resp or {}).get("metric_anchors")
            if isinstance(ma, dict):
                a = ma.get(ckey)
                return a if isinstance(a, dict) else {}
        except Exception:
            pass
        return {}

    def _anchor_meta(prev_resp: dict, cur_resp: dict, ckey: str, pm: dict, cm: dict):
        """
        Returns: (source_url, context_snippet, anchor_confidence)
        Priority:
          1) prev_response.metric_anchors[ckey] (baseline anchoring is authoritative)
          2) current metric row fields (if present)
          3) prev metric row fields (if present)
        """
        a = _get_prev_anchor_obj(prev_resp, ckey)

        src = a.get("source_url") or a.get("url")
        ctx = a.get("context_snippet") or a.get("context")
        conf = a.get("anchor_confidence")

        if not src:
            try:
                src = (cm or {}).get("source_url") or (cm or {}).get("url")
            except Exception:
                src = None
        if not ctx:
            try:
                ctx = (cm or {}).get("context_snippet") or (cm or {}).get("context")
            except Exception:
                ctx = None

        if not src:
            try:
                src = (pm or {}).get("source_url") or (pm or {}).get("url")
            except Exception:
                src = None
        if not ctx:
            try:
                ctx = (pm or {}).get("context_snippet") or (pm or {}).get("context")
            except Exception:
                ctx = None

        try:
            if isinstance(ctx, str):
                ctx = ctx.strip()[:220] or None
            else:
                ctx = None
        except Exception:
            ctx = None

        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None

        return src, ctx, conf
    # =========================================================================

    def prettify_ckey(ckey: str) -> str:
        ckey = str(ckey or "").strip()
        if not ckey:
            return "Unknown Metric"
        parts = ckey.split("__", 1)
        left = parts[0].replace("_", " ").strip()
        right = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
        left = " ".join(w.capitalize() for w in left.split())
        return f"{left} ({right})" if right else left

    def get_metric_definition(prev_resp: dict, ckey: str) -> dict:
        """
        Pull authoritative definition from the ORIGINAL analysis run (prev_response).
        """
        prev_resp = prev_resp if isinstance(prev_resp, dict) else {}

        schema = prev_resp.get("metric_schema_frozen")
        if isinstance(schema, dict):
            d = schema.get(ckey)
            if isinstance(d, dict) and d:
                out = dict(d)
                out.setdefault("canonical_key", ckey)
                return out

        prev_can = prev_resp.get("primary_metrics_canonical")
        if isinstance(prev_can, dict):
            d = prev_can.get(ckey)
            if isinstance(d, dict) and d:
                out = {
                    "canonical_key": ckey,
                    "canonical_id": d.get("canonical_id"),
                    "dimension": d.get("dimension"),
                    "name": d.get("name") or d.get("original_name"),
                    "unit": d.get("unit"),
                    "geo_scope": d.get("geo_scope"),
                    "geo_name": d.get("geo_name"),
                    "keywords": d.get("keywords"),
                }
                return {k: v for k, v in out.items() if v not in (None, "", [], {})}

        return {"canonical_key": ckey, "name": prettify_ckey(ckey)}

    def get_display_name(prev_resp: dict, prev_can_obj: dict, cur_can_obj: dict, ckey: str) -> str:
        schema = prev_resp.get("metric_schema_frozen")
        if isinstance(schema, dict):
            sm = schema.get(ckey)
            if isinstance(sm, dict):
                v = sm.get("name")
                if isinstance(v, str) and v.strip():
                    return v.strip()

        if isinstance(prev_can_obj, dict):
            for k in ("name", "original_name"):
                v = prev_can_obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        if isinstance(cur_can_obj, dict):
            for k in ("name", "original_name"):
                v = cur_can_obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        return prettify_ckey(ckey)

    # =========================================================================
    # PATCH D3 (ADDITIVE): schema-driven tolerances (optional)
    # - If schema provides abs_eps/rel_eps use them, else default.
    # - Safe: only affects comparisons when schema explicitly opts in.
    # =========================================================================
    def get_eps_for_metric(prev_resp: dict, ckey: str):
        ae = ABS_EPS
        re_ = REL_EPS
        try:
            schema = (prev_resp or {}).get("metric_schema_frozen")
            if isinstance(schema, dict):
                d = schema.get(ckey)
                if isinstance(d, dict):
                    if d.get("abs_eps") is not None:
                        try:
                            ae = float(d.get("abs_eps"))
                        except Exception:
                            pass
                    if d.get("rel_eps") is not None:
                        try:
                            re_ = float(d.get("rel_eps"))
                        except Exception:
                            pass
        except Exception:
            pass
        return ae, re_
    # =========================================================================

    prev_response = prev_response if isinstance(prev_response, dict) else {}
    cur_response = cur_response if isinstance(cur_response, dict) else {}

    prev_can = prev_response.get("primary_metrics_canonical")
    cur_can = cur_response.get("primary_metrics_canonical")

    # =========================
    # Path A: canonical-first
    # =========================
    if isinstance(prev_can, dict) and isinstance(cur_can, dict) and prev_can:
        metric_changes = []
        unchanged = increased = decreased = found = 0

        for ckey, pm in prev_can.items():
            pm = pm if isinstance(pm, dict) else {}
            cm = cur_can.get(ckey)
            cm = cm if isinstance(cm, dict) else {}

            display_name = get_display_name(prev_response, pm, cm, ckey)
            definition = get_metric_definition(prev_response, ckey)

            prev_raw = pm.get("raw") if pm.get("raw") is not None else pm.get("value")

            # ✅ HARD STOP: canonical key missing in current => not_found (no name fallback)
            if ckey not in cur_can or not isinstance(cur_can.get(ckey), dict):
                # =========================================================================
                # PATCH MA2 (ADDITIVE): fill row fields from metric_anchors where possible
                # =========================================================================
                _src, _ctx, _aconf = _anchor_meta(prev_response, cur_response, ckey, pm, {})
                # =========================================================================

                metric_changes.append({
                    "name": display_name,
                    "previous_value": prev_raw,
                    "current_value": "N/A",
                    "change_pct": None,
                    "change_type": "not_found",
                    "match_confidence": 0.0,

                    # PATCH MA2 (ADDITIVE): was None
                    "context_snippet": _ctx,
                    "source_url": _src,

                    "anchor_used": False,  # (kept) not applicable when current metric missing
                    "canonical_key": ckey,
                    "metric_definition": definition,

                    # PATCH MA2 (ADDITIVE): extra debug field (harmless if ignored)
                    "anchor_confidence": _aconf,
                })
                continue

            found += 1

            cur_raw = cm.get("raw") if cm.get("raw") is not None else cm.get("value")

            # =========================================================================
            # PATCH D0 (ADDITIVE): anchor-first unchanged shortcut
            # - If anchor_hash matches, mark unchanged regardless of value formatting/range
            # =========================================================================
            prev_ah = _get_prev_anchor_hash(prev_response, ckey, pm)
            cur_ah = _get_cur_anchor_hash(cur_response, ckey, cm)
            anchor_same = bool(prev_ah and cur_ah and str(prev_ah) == str(cur_ah))
            # =========================================================================

            # =========================================================================
            # PATCH D2 (ADDITIVE): use canonical values for diff when available
            # =========================================================================
            prev_val, prev_unit_cmp = get_canonical_value_and_unit(pm)
            cur_val, cur_unit_cmp = get_canonical_value_and_unit(cm)
            # =========================================================================

            # =========================================================================
            # PATCH D3 (ADDITIVE): metric-specific tolerances (schema overrides)
            # =========================================================================
            abs_eps, rel_eps = get_eps_for_metric(prev_response, ckey)
            # =========================================================================

            change_type = "unknown"
            change_pct = None

            # =========================================================================
            # PATCH D0 (ADDITIVE): apply anchor-first result
            # =========================================================================
            if anchor_same:
                change_type = "unchanged"
                change_pct = 0.0
                unchanged += 1
            elif prev_val is not None and cur_val is not None:
                if abs(prev_val - cur_val) <= max(abs_eps, abs(prev_val) * rel_eps):
                    change_type = "unchanged"
                    change_pct = 0.0
                    unchanged += 1
                elif cur_val > prev_val:
                    change_type = "increased"
                    change_pct = ((cur_val - prev_val) / max(abs_eps, abs(prev_val))) * 100.0
                    increased += 1
                else:
                    change_type = "decreased"
                    change_pct = ((cur_val - prev_val) / max(abs_eps, abs(prev_val))) * 100.0
                    decreased += 1
            # =========================================================================

            # =========================================================================
            # PATCH D4 (ADDITIVE): unit mismatch flag (debug only)
            # =========================================================================
            unit_mismatch = False
            try:
                if prev_unit_cmp and cur_unit_cmp and str(prev_unit_cmp) != str(cur_unit_cmp):
                    unit_mismatch = True
            except Exception:
                unit_mismatch = False
            # =========================================================================

            # =========================================================================
            # PATCH MA2 (ADDITIVE): fill row fields from metric_anchors where possible
            # =========================================================================
            _src, _ctx, _aconf = _anchor_meta(prev_response, cur_response, ckey, pm, cm)
            # =========================================================================

            metric_changes.append({
                "name": display_name,
                "previous_value": prev_raw,
                "current_value": cur_raw,
                "change_pct": change_pct,
                "change_type": change_type,
                "match_confidence": 92.0,

                # PATCH MA2 (ADDITIVE): was None
                "context_snippet": _ctx,
                "source_url": _src,

                # =========================================================================
                # PATCH D0 (ADDITIVE): mark anchor usage + expose hashes
                # =========================================================================
                "anchor_used": bool(anchor_same),
                "prev_anchor_hash": prev_ah,
                "cur_anchor_hash": cur_ah,
                # =========================================================================

                "canonical_key": ckey,
                "metric_definition": definition,

                # PATCH MA2 (ADDITIVE): extra debug field (harmless if ignored)
                "anchor_confidence": _aconf,

                # =========================================================================
                # PATCH D2 (ADDITIVE): expose canonical comparison basis for debugging/convergence
                # =========================================================================
                "prev_value_norm": prev_val,
                "cur_value_norm": cur_val,
                "prev_unit_cmp": prev_unit_cmp,
                "cur_unit_cmp": cur_unit_cmp,
                "unit_mismatch": bool(unit_mismatch),
                "abs_eps_used": abs_eps,
                "rel_eps_used": rel_eps,
                # =========================================================================
            })

        return metric_changes, unchanged, increased, decreased, found

    # =========================
    # Path B: legacy name fallback
    # =========================
    prev_metrics = prev_response.get("primary_metrics") or {}
    cur_metrics = cur_response.get("primary_metrics") or {}
    if not isinstance(prev_metrics, dict):
        prev_metrics = {}
    if not isinstance(cur_metrics, dict):
        cur_metrics = {}

    prev_index = {}
    for k, m in prev_metrics.items():
        if isinstance(m, dict):
            name = m.get("name") or k
            prev_index[norm_name(name)] = (name, m)

    cur_index = {}
    for k, m in cur_metrics.items():
        if isinstance(m, dict):
            name = m.get("name") or k
            cur_index[norm_name(name)] = (name, m)

    metric_changes = []
    unchanged = increased = decreased = found = 0

    for nk, (display_name, pm) in prev_index.items():
        prev_raw = pm.get("raw") if pm.get("raw") is not None else pm.get("value")

        if nk not in cur_index:
            metric_changes.append({
                "name": display_name or "Unknown Metric",
                "previous_value": prev_raw,
                "current_value": "N/A",
                "change_pct": None,
                "change_type": "not_found",
                "match_confidence": 0.0,
                "context_snippet": None,
                "source_url": None,
                "anchor_used": False,
            })
            continue

        found += 1
        _, cm = cur_index[nk]
        cur_raw = cm.get("raw") if cm.get("raw") is not None else cm.get("value")

        # =========================================================================
        # PATCH D2 (ADDITIVE): use canonical values when present (legacy path too)
        # =========================================================================
        prev_val, _prev_unit_cmp = get_canonical_value_and_unit(pm)
        cur_val, _cur_unit_cmp = get_canonical_value_and_unit(cm)
        # =========================================================================

        # =========================================================================
        # PATCH D0 (ADDITIVE): legacy-path anchor-first unchanged shortcut
        # - Only engages if both metric dicts carry anchor_hash (rare on legacy path)
        # =========================================================================
        prev_ah = _get_anchor_hash_from_metric(pm)
        cur_ah = _get_anchor_hash_from_metric(cm)
        anchor_same = bool(prev_ah and cur_ah and str(prev_ah) == str(cur_ah))
        # =========================================================================

        change_type = "unknown"
        change_pct = None

        # =========================================================================
        # PATCH D0 (ADDITIVE): apply anchor-first result
        # =========================================================================
        if anchor_same:
            change_type = "unchanged"
            change_pct = 0.0
            unchanged += 1
        elif prev_val is not None and cur_val is not None:
            if abs(prev_val - cur_val) <= max(ABS_EPS, abs(prev_val) * REL_EPS):
                change_type = "unchanged"
                change_pct = 0.0
                unchanged += 1
            elif cur_val > prev_val:
                change_type = "increased"
                change_pct = ((cur_val - prev_val) / max(ABS_EPS, abs(prev_val))) * 100.0
                increased += 1
            else:
                change_type = "decreased"
                change_pct = ((cur_val - prev_val) / max(ABS_EPS, abs(prev_val))) * 100.0
                decreased += 1
        # =========================================================================

        metric_changes.append({
            "name": display_name or "Unknown Metric",
            "previous_value": prev_raw,
            "current_value": cur_raw,
            "change_pct": change_pct,
            "change_type": change_type,
            "match_confidence": 80.0,
            "context_snippet": None,
            "source_url": None,

            # =========================================================================
            # PATCH D0 (ADDITIVE): anchor usage + expose hashes (legacy)
            # =========================================================================
            "anchor_used": bool(anchor_same),
            "prev_anchor_hash": prev_ah,
            "cur_anchor_hash": cur_ah,
            # =========================================================================

            # PATCH D2 (ADDITIVE): expose basis
            "prev_value_norm": prev_val,
            "cur_value_norm": cur_val,
        })

    return metric_changes, unchanged, increased, decreased, found



def _fallback_match_from_snapshots(prev_numbers: dict, snapshots: list, anchors_by_name: dict):
    """
    When current analysis is missing, fall back to cached extracted_numbers only.
    If there is no snapshot candidate, return not_found ✅.

    Tightening implemented:
      1) Reject obvious year mismatches:
         - If metric name or prev_raw includes a year (e.g., 2024), require candidate context to contain it.
         - Also reject candidates that are a bare year if metric is not a year metric.
      2) Unit-family gating:
         - percent vs currency vs magnitude vs other (GW/TWh/tons/etc)
      3) Domain/homepage handling:
         - Downweight homepage sources heavily unless anchored (or if no non-homepage pool exists)

    Debugging enhancements:
      - Each metric row includes match_debug with:
        method, pool sizes, required years, unit families, best score, reject counts, top alternatives (small).
    """
    import re

    ABS_EPS = 1e-9
    REL_EPS = 0.0005

    def norm_unit(u: str) -> str:
        fn = globals().get("normalize_unit")
        if callable(fn):
            try:
                return fn(u)
            except Exception:
                pass
        return (u or "").strip()

    def parse_num(v, unit=""):
        fn = globals().get("parse_human_number")
        if callable(fn):
            try:
                return fn(str(v), unit)
            except Exception:
                return None
        try:
            return float(str(v).replace(",", "").strip())
        except Exception:
            return None

    def metric_tokens(name: str):
        toks = re.findall(r"[a-z0-9]+", (name or "").lower())
        stop = {"the","and","or","of","in","to","for","by","from","with","on","at","as"}
        return [t for t in toks if len(t) > 3 and t not in stop][:24]

    def unit_family(unit: str, raw: str = "", ctx: str = "") -> str:
        u = (norm_unit(unit) or "").strip().upper()
        blob = f"{raw or ''} {ctx or ''}".upper()

        # percent
        if u == "%" or "%" in blob:
            return "percent"

        # currency
        if any(x in blob for x in ["USD", "SGD", "EUR", "GBP", "S$", "$", "€", "£"]):
            return "currency"
        if any(x in u for x in ["USD", "SGD", "EUR", "GBP"]) or u.startswith("$") or u.startswith("S$"):
            return "currency"

        # magnitude suffix
        if u in ("K", "M", "B", "T") or any(x in blob for x in [" BILLION", " MILLION", " TRILLION", " BN", " MN"]):
            return "magnitude"

        # otherwise: other units like GW, TWh, tons, units, etc
        return "other"

    def required_years(metric_name: str, prev_raw: str) -> list:
        years = set()
        for s in [metric_name or "", prev_raw or ""]:
            for y in re.findall(r"\b(19\d{2}|20\d{2})\b", str(s)):
                years.add(y)
        return sorted(years)

    def year_ok(req_years: list, ctx: str) -> bool:
        if not req_years:
            return True
        c = (ctx or "").lower()
        return any(y.lower() in c for y in req_years)

    def is_bare_year(raw: str, unit: str) -> bool:
        r = (raw or "").strip()
        if unit and norm_unit(unit) not in ("", None):
            # If there is a unit, don't treat as bare year
            return False
        return bool(re.match(r"^(19\d{2}|20\d{2})$", r))

    def ctx_score(tokens, ctx: str) -> float:
        c = (ctx or "").lower()
        if not tokens:
            return 0.0
        hit = sum(1 for t in tokens if t in c)
        return hit / max(1, len(tokens))

    # Flatten candidates from snapshots ONLY, keep snapshot metadata
    candidates = []
    for sr in (snapshots or []):
        if not isinstance(sr, dict):
            continue
        url = sr.get("url")
        if not url:
            continue
        is_home = bool(sr.get("is_homepage"))
        qs = sr.get("quality_score", 1.0)
        try:
            qs = float(qs)
        except Exception:
            qs = 1.0

        for n in (sr.get("extracted_numbers") or []):
            if not isinstance(n, dict):
                continue
            candidates.append({
                "url": url,
                "value": n.get("value"),
                "unit": norm_unit(n.get("unit") or ""),
                "raw": n.get("raw") or "",
                "context": n.get("context") or "",
                "is_homepage": is_home,
                "quality_score": qs,
            })

    # Pre-split pools for tightening #3
    non_home = [c for c in candidates if not c.get("is_homepage")]
    home = [c for c in candidates if c.get("is_homepage")]

    out_changes = []
    for metric_name, prev in (prev_numbers or {}).items():
        prev_raw = prev.get("raw") or prev.get("value") or "N/A"
        prev_unit = norm_unit(prev.get("unit") or "")
        prev_val = prev.get("value")
        toks = prev.get("keywords") or metric_tokens(metric_name)

        req_years = required_years(metric_name, str(prev_raw))
        prev_fam = unit_family(prev_unit, str(prev_raw), "")

        anchor = anchors_by_name.get(metric_name) or {}
        anchor_url = anchor.get("source_url") if isinstance(anchor, dict) else None

        # Pool policy:
        # - anchored: use anchor_url pool if exists
        # - else: use non-homepage pool when available; only fall back to homepage if necessary
        pool_policy = "non_home_preferred"
        pool = non_home if non_home else candidates
        if anchor_url:
            anchored_pool = [c for c in candidates if c.get("url") == anchor_url]
            if anchored_pool:
                pool = anchored_pool
                pool_policy = "anchored_url"
            else:
                pool_policy = "anchored_url_not_present"

        reject_counts = {"year_mismatch": 0, "unit_mismatch": 0, "bare_year_reject": 0}
        best = None
        best_score = -1e9
        top_alts = []  # store a few near-misses for debugging

        for c in pool:
            ctx = c.get("context", "") or ""
            raw = c.get("raw", "") or ""
            unit = c.get("unit", "") or ""

            # (1) year gating: if required years exist, require them in context
            if not year_ok(req_years, ctx):
                reject_counts["year_mismatch"] += 1
                continue

            # reject bare-year candidates unless the metric itself is a year metric
            # (prevents "2024" being selected as a value for percent/currency/etc)
            if is_bare_year(str(raw), unit) and prev_fam != "other":
                reject_counts["bare_year_reject"] += 1
                continue

            # (2) unit-family gating
            cand_fam = unit_family(unit, raw, ctx)
            if prev_fam != cand_fam:
                reject_counts["unit_mismatch"] += 1
                continue

            score = ctx_score(toks, ctx)

            # bonus for numeric closeness
            cv = parse_num(c.get("value"), unit) or parse_num(raw, unit)
            if prev_val is not None and cv is not None:
                if abs(prev_val - cv) <= max(ABS_EPS, abs(prev_val) * REL_EPS):
                    score += 0.25

            # (3) homepage penalty unless anchored
            if c.get("is_homepage") and not anchor_url:
                score -= 0.35

            # quality_score weighting
            try:
                score *= max(0.1, min(1.0, float(c.get("quality_score", 1.0))))
            except Exception:
                pass

            # keep top alternatives for debugging
            if len(top_alts) < 5:
                top_alts.append({
                    "raw": raw[:60],
                    "unit": unit,
                    "url": c.get("url"),
                    "score": float(score),
                    "is_homepage": bool(c.get("is_homepage")),
                    "ctx": (ctx or "")[:120],
                })

            if score > best_score:
                best_score = score
                best = c

        # sort alt candidates by score desc
        try:
            top_alts.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        except Exception:
            pass

        if not best:
            out_changes.append({
                "name": metric_name,
                "previous_value": prev_raw,
                "current_value": "N/A",
                "change_pct": None,
                "change_type": "not_found",
                "match_confidence": 0.0,
                "context_snippet": None,
                "source_url": None,
                "anchor_used": bool(anchor_url),

                # NEW debug payload
                "match_debug": {
                    "method": "snapshots_only",
                    "pool_policy": pool_policy,
                    "pool_size": int(len(pool)),
                    "req_years": req_years,
                    "prev_unit": prev_unit,
                    "prev_unit_family": prev_fam,
                    "reject_counts": reject_counts,
                    "top_alternatives": top_alts[:3],
                }
            })
            continue

        cur_raw = best.get("raw") or best.get("value")
        cv = parse_num(best.get("value"), best.get("unit")) or parse_num(cur_raw, best.get("unit"))

        change_type = "unknown"
        change_pct = None
        if prev_val is not None and cv is not None:
            if abs(prev_val - cv) <= max(ABS_EPS, abs(prev_val) * REL_EPS):
                change_type = "unchanged"
                change_pct = 0.0
            elif cv > prev_val:
                change_type = "increased"
                change_pct = ((cv - prev_val) / max(ABS_EPS, abs(prev_val))) * 100.0
            else:
                change_type = "decreased"
                change_pct = ((cv - prev_val) / max(ABS_EPS, abs(prev_val))) * 100.0

        conf = max(0.0, min(60.0, best_score * 60.0))

        out_changes.append({
            "name": metric_name,
            "previous_value": prev_raw,
            "current_value": cur_raw,
            "change_pct": change_pct,
            "change_type": change_type,
            "match_confidence": float(conf),
            "context_snippet": (best.get("context") or "")[:200] if isinstance(best.get("context"), str) else None,
            "source_url": best.get("url"),
            "anchor_used": bool(anchor_url),

            # NEW debug payload
            "match_debug": {
                "method": "snapshots_only",
                "pool_policy": pool_policy,
                "pool_size": int(len(pool)),
                "req_years": req_years,
                "prev_unit": prev_unit,
                "prev_unit_family": prev_fam,
                "best_unit": best.get("unit"),
                "best_unit_family": unit_family(best.get("unit") or "", best.get("raw") or "", best.get("context") or ""),
                "best_score": float(best_score),
                "best_is_homepage": bool(best.get("is_homepage")),
                "reject_counts": reject_counts,
                "top_alternatives": top_alts[:3],
            }
        })

    return out_changes


def compute_source_anchored_diff(previous_data: dict, web_context: dict = None) -> dict:
    """
    Tight source-anchored evolution:
      - Prefer snapshots from analysis (baseline_sources_cache)
      - Optionally reconstruct snapshots from web_context.scraped_meta
      - If no valid snapshots: return not_found (no heuristic junk)

    Always returns a dict.
    """
    import re
    from datetime import datetime, timezone

    def _now():
        return datetime.now(timezone.utc).isoformat()

    def _safe_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    def _fingerprint(text: str):
        try:
            fn = globals().get("fingerprint_text")
            if callable(fn):
                return fn(text or "")
        except Exception:
            pass
        try:
            return fingerprint_text(text or "")
        except Exception:
            return None

    # ---------- Pull baseline snapshots (VALID only) ----------
    snapshot_origin = "none"
    baseline_sources_cache = []

    try:
        if isinstance(previous_data, dict):
            # 1) results.baseline_sources_cache (preferred)
            r = previous_data.get("results")
            if isinstance(r, dict) and isinstance(r.get("baseline_sources_cache"), list):
                baseline_sources_cache = r.get("baseline_sources_cache") or []
                if baseline_sources_cache:
                    snapshot_origin = "analysis_results_cache"

            # 2) top-level baseline_sources_cache
            if not baseline_sources_cache and isinstance(previous_data.get("baseline_sources_cache"), list):
                baseline_sources_cache = previous_data.get("baseline_sources_cache") or []
                if baseline_sources_cache:
                    snapshot_origin = "analysis_top_level_cache"
    except Exception:
        baseline_sources_cache = []


    # =====================================================================
    # PATCH ES1B (ADDITIVE): broaden snapshot discovery (legacy storage shapes)
    # Why:
    # - Some callers persist only previous_data["primary_response"] or embed
    #   caches under different nesting. If snapshots exist there, evolution
    #   should still be able to use them without re-fetching.
    # Notes:
    # - Additive only: does not remove/replace existing preferred paths.
    # - Still snapshot-gated: we only accept FULL snapshot shapes (list of
    #   sources with extracted_numbers list). We do not fabricate numbers.
    # =====================================================================
    try:
        if (not baseline_sources_cache) and isinstance(previous_data, dict):
            pr = previous_data.get("primary_response") or {}
            if isinstance(pr, dict):
                # A) primary_response.results.baseline_sources_cache
                r2 = pr.get("results")
                if (not baseline_sources_cache) and isinstance(r2, dict) and isinstance(r2.get("baseline_sources_cache"), list):
                    baseline_sources_cache = r2.get("baseline_sources_cache") or []
                    if baseline_sources_cache:
                        snapshot_origin = "primary_response_results_cache"

                # B) primary_response.baseline_sources_cache
                if (not baseline_sources_cache) and isinstance(pr.get("baseline_sources_cache"), list):
                    baseline_sources_cache = pr.get("baseline_sources_cache") or []
                    if baseline_sources_cache:
                        snapshot_origin = "primary_response_top_level_cache"

                # C) primary_response.results.source_results (reconstruct minimal snapshot shape)
                #    We only use this if it already contains extracted_numbers lists.
                if (not baseline_sources_cache) and isinstance(r2, dict) and isinstance(r2.get("source_results"), list):
                    rebuilt_sr = []
                    for sr in (r2.get("source_results") or []):
                        if not isinstance(sr, dict):
                            continue
                        u = sr.get("source_url") or sr.get("url")
                        ex = sr.get("extracted_numbers")
                        if u and isinstance(ex, list) and ex:
                            rebuilt_sr.append({
                                "source_url": u,
                                "extracted_numbers": ex,
                                "clean_text": sr.get("clean_text") or sr.get("content") or "",
                                "fingerprint": sr.get("fingerprint"),
                                "fetched_at": sr.get("fetched_at"),
                            })
                    # Deterministic ordering
                    rebuilt_sr.sort(key=lambda d: (str(d.get("source_url") or ""), str(d.get("fingerprint") or "")))
                    if rebuilt_sr:
                        baseline_sources_cache = rebuilt_sr
                        snapshot_origin = "primary_response_source_results_rebuild"

        # D) As a last legacy fallback, some callers store caches under previous_data["results"]["source_results"]
        if (not baseline_sources_cache) and isinstance(previous_data, dict):
            r3 = previous_data.get("results")
            if isinstance(r3, dict) and isinstance(r3.get("source_results"), list):
                rebuilt_sr2 = []
                for sr in (r3.get("source_results") or []):
                    if not isinstance(sr, dict):
                        continue
                    u = sr.get("source_url") or sr.get("url")
                    ex = sr.get("extracted_numbers")
                    if u and isinstance(ex, list) and ex:
                        rebuilt_sr2.append({
                            "source_url": u,
                            "extracted_numbers": ex,
                            "clean_text": sr.get("clean_text") or sr.get("content") or "",
                            "fingerprint": sr.get("fingerprint"),
                            "fetched_at": sr.get("fetched_at"),
                        })
                rebuilt_sr2.sort(key=lambda d: (str(d.get("source_url") or ""), str(d.get("fingerprint") or "")))
                if rebuilt_sr2:
                    baseline_sources_cache = rebuilt_sr2
                    snapshot_origin = "analysis_source_results_rebuild"
    except Exception:
        pass


    # =====================================================================
    # PATCH SS6C (ADDITIVE): evidence_records fallback for snapshots (evolution-time)
    # If baseline_sources_cache is missing or summarized in previous_data, but
    # evidence_records are present, rebuild the minimal snapshot shape needed
    # for source-anchored evolution. No re-fetching; deterministic only.
    # =====================================================================
    try:
        if (not baseline_sources_cache) and isinstance(previous_data, dict):
            _er = None
            if isinstance(previous_data.get("results"), dict):
                _er = previous_data["results"].get("evidence_records")
            if _er is None:
                _er = previous_data.get("evidence_records")
            _rebuilt = build_baseline_sources_cache_from_evidence_records(_er)
            if isinstance(_rebuilt, list) and _rebuilt:
                baseline_sources_cache = _rebuilt
                try:
                    snapshot_origin = "evidence_records_rebuild"
                except Exception:
                    pass
    except Exception:
        pass
    # =====================================================================


    # =====================================================================
    # PATCH ES1C (ADDITIVE): validate snapshot shape & attach debug metadata
    # Why:
    # - Avoid "truthy but unusable" caches (e.g., summarized shapes).
    # - Provide actionable debug fields without changing failure message.
    # =====================================================================
    try:
        _raw_len = int(len(baseline_sources_cache)) if isinstance(baseline_sources_cache, list) else 0
        _kept = []
        if isinstance(baseline_sources_cache, list):
            for s in baseline_sources_cache:
                if not isinstance(s, dict):
                    continue
                u = s.get("source_url") or s.get("url")
                ex = s.get("extracted_numbers")
                # Require full snapshot shape: extracted_numbers must be a list (possibly empty is allowed, but we prefer list)
                if u and isinstance(ex, list):
                    _kept.append(s)
        # Deterministic ordering for downstream
        _kept.sort(key=lambda d: (str(d.get("source_url") or ""), str(d.get("fingerprint") or "")))
        baseline_sources_cache = _kept
        # Attach debug meta to output
        try:
            output["snapshot_debug"] = {
                "origin": snapshot_origin,
                "raw_count": _raw_len,
                "valid_count": int(len(baseline_sources_cache)),
                "example_urls": [x.get("source_url") for x in (baseline_sources_cache[:3] if isinstance(baseline_sources_cache, list) else [])],
                "prev_keys": sorted(list(previous_data.keys()))[:40] if isinstance(previous_data, dict) else [],
            }
        except Exception:
            pass
    except Exception:
        pass
    # =====================================================================

    # 3) reconstruct from web_context.scraped_meta (if provided)
    if (not baseline_sources_cache) and isinstance(web_context, dict):
        try:
            scraped_meta = web_context.get("scraped_meta") or {}
            rebuilt = []
            if isinstance(scraped_meta, dict):
                for url, meta in scraped_meta.items():
                    if not isinstance(meta, dict):
                        continue
                    content = meta.get("clean_text") or meta.get("content") or ""
                    fp = meta.get("fingerprint") or _fingerprint(content)
                    if not fp or len(content or "") < 800:
                        continue
                    nums = meta.get("extracted_numbers") or []
                    if not isinstance(nums, list):
                        nums = []

                    rebuilt.append({
                        "url": url,
                        "status": meta.get("status") or "fetched",
                        "status_detail": meta.get("status_detail") or "",
                        "numbers_found": _safe_int(meta.get("numbers_found"), default=len(nums)),
                        "fetched_at": meta.get("fetched_at") or _now(),
                        "fingerprint": fp,
                        "content_type": meta.get("content_type") or "",
                        "extracted_numbers": [
                            {
                                "value": n.get("value"),
                                "unit": n.get("unit"),
                                "raw": n.get("raw"),
                                "context_snippet": (n.get("context_snippet") or n.get("context") or "")[:200],

                                # =====================================================================
                                # PATCH (ADDITIVE): preserve analysis-aligned fields when present.
                                # =====================================================================
                                "anchor_hash": n.get("anchor_hash"),
                                "is_junk": n.get("is_junk"),
                                "junk_reason": n.get("junk_reason"),
                                "unit_tag": n.get("unit_tag"),
                                "unit_family": n.get("unit_family"),
                                "base_unit": n.get("base_unit"),
                                "multiplier_to_base": n.get("multiplier_to_base"),
                                "value_norm": n.get("value_norm"),
                                "start_idx": n.get("start_idx"),
                                "end_idx": n.get("end_idx"),
                                "source_url": n.get("source_url") or url,
                                # =====================================================================
                            }
                            for n in nums if isinstance(n, dict)
                        ]
                    })
            if rebuilt:
                baseline_sources_cache = rebuilt
                snapshot_origin = "web_context_scraped_meta"
        except Exception:
            pass

    # Also count invalid snapshots for debug (if present)
    invalid_count = 0
    try:
        if isinstance(previous_data, dict):
            r = previous_data.get("results")
            if isinstance(r, dict) and isinstance(r.get("baseline_sources_cache_invalid"), list):
                invalid_count = len(r.get("baseline_sources_cache_invalid") or [])
    except Exception:
        invalid_count = 0

    # ---------- Prepare stable default output ----------
    output = {
        "status": "success",
        "message": "",
        "sources_checked": 0,
        "sources_fetched": 0,
        "numbers_extracted_total": 0,
        "stability_score": 0.0,
        "summary": {
            "total_metrics": 0,
            "metrics_found": 0,
            "metrics_increased": 0,
            "metrics_decreased": 0,
            "metrics_unchanged": 0,
        },
        "metric_changes": [],
        "source_results": [],
        "interpretation": "",
        # debug
        "snapshot_origin": snapshot_origin,
        "valid_snapshot_count": len(baseline_sources_cache or []),
        "invalid_snapshot_count": int(invalid_count),
        "generated_at": _now(),
    }

    # =====================================================================
    # PATCH SS6 (ADDITIVE, REQUIRED): last-chance snapshot rehydration
    # Why:
    #   - When History rows are stored as Sheets-safe wrappers, baseline_sources_cache
    #     may be absent at runtime unless a caller went through get_history() rehydration.
    #   - Some UI paths may pass previous_data directly from the truncated wrapper row.
    # Determinism:
    #   - Only loads from existing snapshot stores (Sheets Snapshots tab or local file).
    #   - No re-fetching; no heuristic matching.
    # =====================================================================
    try:
        if not baseline_sources_cache and isinstance(previous_data, dict):
            _ref = previous_data.get("snapshot_store_ref") or (previous_data.get("results") or {}).get("snapshot_store_ref")
            _hash = previous_data.get("source_snapshot_hash") or (previous_data.get("results") or {}).get("source_snapshot_hash")

            # Prefer explicit pointer if provided
            if isinstance(_ref, str) and _ref.startswith("gsheet:"):
                parts = _ref.split(":")
                _ws_title = parts[1] if len(parts) > 1 and parts[1] else "Snapshots"
                _h = parts[2] if len(parts) > 2 else ""
                baseline_sources_cache = load_full_snapshots_from_sheet(_h, worksheet_title=_ws_title) if _h else []
                if baseline_sources_cache:
                    output["snapshot_origin"] = "sheet_snapshot_store_ref"

            # If no ref, try by hash (common when wrapper omitted pointer)
            if not baseline_sources_cache and isinstance(_hash, str) and _hash:
                baseline_sources_cache = load_full_snapshots_from_sheet(_hash, worksheet_title="Snapshots")
                if baseline_sources_cache:
                    output["snapshot_origin"] = "sheet_source_snapshot_hash"

            # Local fallback
            if not baseline_sources_cache and isinstance(_ref, str) and _ref and not _ref.startswith("gsheet:"):
                baseline_sources_cache = load_full_snapshots_local(_ref)
                if baseline_sources_cache:
                    output["snapshot_origin"] = "local_snapshot_store_ref"

            # Keep debug counts aligned
            if isinstance(baseline_sources_cache, list):
                output["valid_snapshot_count"] = len(baseline_sources_cache)
    except Exception:
        pass
    # =====================================================================

    # If no valid snapshots, return "not_found" (tight safety net)
    if not baseline_sources_cache:
        output["status"] = "failed"
        output["message"] = "No valid snapshots available for source-anchored evolution. (No re-fetch / no heuristic matching performed.)"
        output["interpretation"] = "Snapshot-gated: evolution refused to fabricate matches without valid cached source text."
        return output

    # =====================================================================
    # PATCH HF5 (ADDITIVE): rehydrate previous_data from HistoryFull if wrapper
    # Why:
    # - Some UI/Sheets paths provide a summarized wrapper that lacks primary_response,
    #   metric_schema_frozen, metric_anchors, etc.
    # - If a full_store_ref pointer exists, load the full payload deterministically.
    # =====================================================================
    _prev_rehydrated = False
    _prev_rehydrated_ref = ""
    try:
        if isinstance(previous_data, dict):
            # Determine if we are missing rebuild essentials
            _pr = previous_data.get("primary_response")
            _need = (not isinstance(_pr, dict)) or (not _pr) or (not isinstance(_pr.get("metric_schema_frozen"), dict))

            if _need:
                _ref = previous_data.get("full_store_ref")                     or (previous_data.get("results") or {}).get("full_store_ref")                     or (isinstance(_pr, dict) and _pr.get("full_store_ref"))                     or ""

                if isinstance(_ref, str) and _ref.startswith("gsheet:"):
                    parts = _ref.split(":")
                    _ws_title = parts[1] if len(parts) > 1 and parts[1] else "HistoryFull"
                    _aid = parts[2] if len(parts) > 2 else ""
                    full = load_full_history_payload_from_sheet(_aid, worksheet_title=_ws_title) if _aid else {}
                    if isinstance(full, dict) and full:
                        previous_data = full
                        _prev_rehydrated = True
                        _prev_rehydrated_ref = _ref
    except Exception:
        pass

    # Attach debug flags (harmless; helps diagnose missing schema)
    try:
        if _prev_rehydrated:
            output["previous_data_rehydrated"] = True
            output["previous_data_full_store_ref"] = _prev_rehydrated_ref
    except Exception:
        pass
    # =====================================================================

# ---------- Use your existing deterministic metric diff helper ----------
    # Pull baseline metrics from previous_data
    prev_response = (previous_data or {}).get("primary_response", {}) or {}
    # =====================================================================
    # PATCH HF6 (ADDITIVE): tolerate previous_data being the primary_response itself
    # Why:
    # - Some callers persist only the inner primary_response dict as "previous_data".
    # - In that case, previous_data won't have a "primary_response" key.
    # =====================================================================
    try:
        if (not isinstance(prev_response, dict) or not prev_response) and isinstance(previous_data, dict):
            if isinstance(previous_data.get("primary_metrics_canonical"), dict) or isinstance(previous_data.get("metric_schema_frozen"), dict):
                prev_response = previous_data
    except Exception:
        pass
    # =====================================================================

    prev_metrics = prev_response.get("primary_metrics_canonical") or prev_response.get("primary_metrics") or {}

    # =====================================================================
    # PATCH E1 (ADDITIVE): ensure schema is available inside prev_response
    # Why: rebuild_metrics_from_snapshots() fallback reads prev_response.metric_schema_frozen.
    # Some runs store metric_schema_frozen at analysis top-level or under results.
    # This patch copies it into prev_response *only if missing*.
    # =====================================================================
    try:
        if isinstance(prev_response, dict) and not isinstance(prev_response.get("metric_schema_frozen"), dict):
            schema_frozen = None

            # prefer top-level analysis object
            if isinstance(previous_data, dict) and isinstance(previous_data.get("metric_schema_frozen"), dict):
                schema_frozen = previous_data.get("metric_schema_frozen")

            # else try results.* (if you store it there)
            if schema_frozen is None:
                r = (previous_data or {}).get("results")
                if isinstance(r, dict) and isinstance(r.get("metric_schema_frozen"), dict):
                    schema_frozen = r.get("metric_schema_frozen")

            if isinstance(schema_frozen, dict) and schema_frozen:
                prev_response["metric_schema_frozen"] = schema_frozen
    except Exception:
        pass
    # =====================================================================

    # =====================================================================
    # PATCH E2 (ADDITIVE): ensure metric_anchors are available inside prev_response
    # Why:
    # - diff_metrics_by_name() can pull prev anchor_hash from prev_response.metric_anchors
    #   when the metric row itself doesn't carry anchor_hash.
    # - Some runs store metric_anchors at analysis top-level (previous_data) or under results.
    # This patch copies anchors into prev_response *only if missing*.
    # =====================================================================
    try:
        if isinstance(prev_response, dict) and not isinstance(prev_response.get("metric_anchors"), dict):
            anchors_src = None

            # prefer top-level analysis object
            if isinstance(previous_data, dict) and isinstance(previous_data.get("metric_anchors"), dict):
                anchors_src = previous_data.get("metric_anchors")

            # else try primary_response.metric_anchors (if different object)
            if anchors_src is None:
                pr = (previous_data or {}).get("primary_response")
                if isinstance(pr, dict) and isinstance(pr.get("metric_anchors"), dict):
                    anchors_src = pr.get("metric_anchors")

            # else try results.metric_anchors
            if anchors_src is None:
                r = (previous_data or {}).get("results")
                if isinstance(r, dict) and isinstance(r.get("metric_anchors"), dict):
                    anchors_src = r.get("metric_anchors")

            if isinstance(anchors_src, dict) and anchors_src:
                prev_response["metric_anchors"] = anchors_src
    except Exception:
        pass
    # =====================================================================

    # Build a minimal current metrics dict from snapshots:
    current_metrics = {}

    # =====================================================================
    # PATCH E3 (ADDITIVE): prefer metric_anchors to rebuild current_metrics
    # - If metric_anchors exist: resolve by anchor_hash -> candidate in CURRENT snapshots.
    # - Snapshot-only: if anchor_hash not found, we do NOT guess.
    # - Fully additive: if no anchors/hits, we fall back to rebuild_metrics_from_snapshots().
    # =====================================================================
    def _get_metric_anchors(prev: dict) -> dict:
        if not isinstance(prev, dict):
            return {}

        # 1) top-level
        a = prev.get("metric_anchors")
        if isinstance(a, dict) and a:
            return a

        # 2) under primary_response
        pr = prev.get("primary_response")
        if isinstance(pr, dict):
            a2 = pr.get("metric_anchors")
            if isinstance(a2, dict) and a2:
                return a2

        # 3) under results
        res = prev.get("results")
        if isinstance(res, dict):
            a3 = res.get("metric_anchors")
            if isinstance(a3, dict) and a3:
                return a3

        return {}

    def _canonicalize_candidate(n: dict) -> dict:
        try:
            fn = globals().get("canonicalize_numeric_candidate")
            if callable(fn):
                return fn(dict(n))
        except Exception:
            pass
        return dict(n)

    def _build_anchor_to_candidate_map(snapshots: list) -> dict:
        m = {}
        for sr in snapshots or []:
            if not isinstance(sr, dict):
                continue
            for n in (sr.get("extracted_numbers") or []):
                if not isinstance(n, dict):
                    continue
                nn = _canonicalize_candidate(n)
                ah = nn.get("anchor_hash")
                if not ah:
                    continue
                if ah not in m:
                    m[ah] = nn
                else:
                    # prefer non-junk if tie
                    old = m[ah]
                    if old.get("is_junk") is True and nn.get("is_junk") is not True:
                        m[ah] = nn
        return m

    try:
        metric_anchors = _get_metric_anchors(previous_data)
        anchor_to_candidate = _build_anchor_to_candidate_map(baseline_sources_cache)

        # =================================================================
        # PATCH ES2/ES8 (ADDITIVE): deterministic anchor_hash -> candidate tie-break
        # If multiple candidates share an anchor_hash, pick deterministically using
        # a stable sort key (confidence, context length, context_hash, value, unit, url).
        # =================================================================
        try:
            _det_map = _es_build_candidate_index_deterministic(baseline_sources_cache)
            if isinstance(_det_map, dict) and _det_map:
                anchor_to_candidate = _det_map
        except Exception:
            pass
        # =================================================================

        # =================================================================
        # PATCH ES8 (ADDITIVE): deterministic iteration order over metric_anchors
        # Ensure we iterate anchors in sorted canonical_key order for stable outputs.
        # =================================================================
        try:
            if isinstance(metric_anchors, dict):
                metric_anchors = dict(sorted(metric_anchors.items(), key=lambda kv: str(kv[0])))
        except Exception:
            pass
        # =================================================================


        if isinstance(metric_anchors, dict) and metric_anchors:
            for ckey, a in metric_anchors.items():
                if not isinstance(a, dict):
                    continue

                ah = a.get("anchor_hash") or a.get("anchor")
                if not ah:
                    continue

                cand = anchor_to_candidate.get(ah)
                if not isinstance(cand, dict):
                    continue

                # Optional: use baseline metric as template for stable identity fields
                base = prev_metrics.get(ckey) if isinstance(prev_metrics, dict) else None
                out_row = dict(base) if isinstance(base, dict) else {}

                out_row.update({
                    "canonical_key": ckey,
                    "anchor_hash": ah,
                    # =================================================================
                    # PATCH ES7 (ADDITIVE): evolution output normalization
                    # Ensure evolution output always includes anchor_used + anchor_confidence.
                    # =================================================================
                    "anchor_used": True,
                    "anchor_confidence": a.get("anchor_confidence"),
                    # =================================================================

                    "source_url": cand.get("source_url") or a.get("source_url"),
                    "raw": cand.get("raw"),
                    "value": cand.get("value"),
                    "unit": cand.get("unit"),
                    "unit_tag": cand.get("unit_tag"),
                    "unit_family": cand.get("unit_family"),
                    "base_unit": cand.get("base_unit"),
                    "multiplier_to_base": cand.get("multiplier_to_base"),
                    "value_norm": cand.get("value_norm"),
                    "measure_kind": cand.get("measure_kind"),
                    "measure_assoc": cand.get("measure_assoc"),
                    "context_snippet": cand.get("context_snippet") or cand.get("context") or "",

                    # =================================================================
                    # PATCH E3.1 (ADDITIVE): evidence/debug passthrough for diff/UI stability
                    # - candidate_id helps stable identity/debug (if available)
                    # - anchor_confidence lets diff emit match_confidence deterministically
                    # - fingerprint helps show source consistency (if present)
                    # =================================================================
                    "candidate_id": cand.get("candidate_id") or a.get("candidate_id"),
                    "anchor_confidence": a.get("anchor_confidence"),
                    "fingerprint": cand.get("fingerprint"),
                    # =================================================================
                })

                current_metrics[ckey] = out_row
    except Exception:
        pass
    # =====================================================================

    # =====================================================================
    # PATCH E4 (ADDITIVE): rebuild fallback only if anchors didn't produce metrics
    # - Prevents anchor-built current_metrics from being overwritten.
    # - Preserves existing behavior when anchors are missing or yield no hits.
    # =====================================================================
    # =====================================================================
    # PATCH BSC_NORM1 (ADDITIVE): baseline_sources_cache normalization
    # Purpose:
    #   - Some evolution paths populate "source_results" (current fetch) with extracted_numbers,
    #     but baseline_sources_cache may still be a snapshot-only list without extracted_numbers.
    #   - This normalization bridges the two WITHOUT refetching and WITHOUT inventing numbers.
    #
    # Behavior:
    #   - If baseline_sources_cache is list-shaped but has no extracted_numbers payload,
    #     and a local fetched source_results list exists with extracted_numbers,
    #     we rebuild a minimal baseline_sources_cache view from it (deterministic ordering).
    # =====================================================================
    try:
        _bsc_has_numbers = False
        for _sr in (baseline_sources_cache or []):
            if isinstance(_sr, dict) and isinstance(_sr.get("extracted_numbers"), list) and (_sr.get("extracted_numbers") or []):
                _bsc_has_numbers = True
                break

        if (not _bsc_has_numbers) and isinstance(baseline_sources_cache, list):
            # Try common local variable names used by different code paths.
            _fetched_sr = None
            try:
                _fetched_sr = locals().get("source_results")
            except Exception:
                _fetched_sr = None
            if not isinstance(_fetched_sr, list):
                try:
                    _fetched_sr = locals().get("current_source_results")
                except Exception:
                    _fetched_sr = _fetched_sr
            if not isinstance(_fetched_sr, list):
                try:
                    _fetched_sr = locals().get("fetched_source_results")
                except Exception:
                    _fetched_sr = _fetched_sr

            _rebuilt = []
            if isinstance(_fetched_sr, list):
                for _r in (_fetched_sr or []):
                    if not isinstance(_r, dict):
                        continue
                    _ex = _r.get("extracted_numbers")
                    if not isinstance(_ex, list) or not _ex:
                        continue
                    _u = _r.get("source_url") or _r.get("url")
                    _rebuilt.append({
                        "source_url": _u,
                        "extracted_numbers": _ex,
                        "clean_text": _r.get("clean_text") or _r.get("content") or "",
                        "fingerprint": _r.get("fingerprint"),
                        "fetched_at": _r.get("fetched_at"),
                    })
            _rebuilt.sort(key=lambda d: (str(d.get("source_url") or ""), str(d.get("fingerprint") or "")))

            if _rebuilt:
                baseline_sources_cache = _rebuilt
                snapshot_origin = "evolution_baseline_cache_normalized_from_source_results"
    except Exception:
        pass

    if not isinstance(current_metrics, dict) or not current_metrics:
        try:
            # =========================
            # PATCH RMS_MIN2 (ADDITIVE): prefer schema-only rebuild hook when available
            # =========================
            # ===================== PATCH RMS_DISPATCH1 (ADDITIVE) =====================
            # Prefer anchor-aware rebuild when anchors exist; otherwise schema-only; otherwise legacy.
            prev_response_for_dispatch = _coerce_prev_response_any(previous_data)
            anchors_for_dispatch = _get_metric_anchors_any(prev_response_for_dispatch)

            fn_rebuild = None
            if anchors_for_dispatch and callable(globals().get("rebuild_metrics_from_snapshots_with_anchors")):
                fn_rebuild = globals().get("rebuild_metrics_from_snapshots_with_anchors")
            elif callable(globals().get("rebuild_metrics_from_snapshots_schema_only")):
                fn_rebuild = globals().get("rebuild_metrics_from_snapshots_schema_only")
            else:
                fn_rebuild = globals().get("rebuild_metrics_from_snapshots")
# =================== END PATCH RMS_DISPATCH1 (ADDITIVE) ===================
            if callable(fn_rebuild):
                # =========================
                # PATCH (ADDITIVE): pass web_context through to rebuild hook
                # =========================
                current_metrics = fn_rebuild(prev_response, baseline_sources_cache, web_context=web_context)
                # =========================
        except Exception:
            current_metrics = {}
    # =====================================================================


    # =====================================================================
    # PATCH BSC_NORM_EARLY1 (ADDITIVE): early baseline_sources_cache normalization before guardrail
    # Purpose:
    #   - Some evolution paths carry extracted_numbers in output["source_results"] (current fetch),
    #     while baseline_sources_cache remains snapshot-only or numbers-empty.
    #   - The guardrail below checks rebuilt metrics; if baseline_sources_cache has no numeric payload,
    #     rebuild hooks may return empty and trigger a false "missing/empty" failure.
    #
    # Behavior:
    #   - If baseline_sources_cache contains no non-empty extracted_numbers,
    #     and output["source_results"] contains extracted_numbers,
    #     build a minimal baseline_sources_cache view from output["source_results"].
    #   - Deterministic ordering by (source_url, fingerprint) only.
    # =====================================================================
    try:
        _bsc_has_numbers2 = False
        for _sr2 in (baseline_sources_cache or []):
            if isinstance(_sr2, dict) and isinstance(_sr2.get("extracted_numbers"), list) and (_sr2.get("extracted_numbers") or []):
                _bsc_has_numbers2 = True
                break

        if not _bsc_has_numbers2:
            _out_sr = None
            try:
                _out_sr = output.get("source_results")
            except Exception:
                _out_sr = None

            if isinstance(_out_sr, list) and _out_sr:
                _rebuilt2 = []
                for _r2 in _out_sr:
                    if not isinstance(_r2, dict):
                        continue
                    _ex2 = _r2.get("extracted_numbers")
                    if not isinstance(_ex2, list) or not _ex2:
                        continue
                    _u2 = _r2.get("source_url") or _r2.get("url")
                    _rebuilt2.append({
                        "source_url": _u2,
                        "extracted_numbers": _ex2,
                        "clean_text": _r2.get("clean_text") or _r2.get("content") or "",
                        "fingerprint": _r2.get("fingerprint"),
                        "status": _r2.get("status"),
                        "status_detail": _r2.get("status_detail"),
                    })
                _rebuilt2.sort(key=lambda d: (str(d.get("source_url") or ""), str(d.get("fingerprint") or "")))
                if _rebuilt2:
                    baseline_sources_cache = _rebuilt2
                    try:
                        output["snapshot_origin"] = (output.get("snapshot_origin") or "") + "|baseline_cache_normalized_from_output_source_results"
                    except Exception:
                        pass
    except Exception:
        pass

    # ===================== PATCH RMS_WIRE1 (ADDITIVE) =====================
    # Prefer schema-only rebuild if available; fall back to legacy hook if present.
    fn_rebuild = globals().get("rebuild_metrics_from_snapshots_schema_only") or globals().get("rebuild_metrics_from_snapshots")

    # =====================================================================
    # PATCH RMS_DISPATCH4 (ADDITIVE): Prefer anchor-aware rebuild when metric_anchors exist
    # - Looks for anchors across nested paths via _get_metric_anchors_any (if present)
    # - Falls back to schema-only, then legacy
    # - Does NOT refactor existing logic; only overrides fn_rebuild if conditions met.
    # =====================================================================
    try:
        _anchors = None
        fn_get_anchors = globals().get("_get_metric_anchors_any")
        if callable(fn_get_anchors):
            _anchors = fn_get_anchors(previous_data)
        else:
            # minimal fallback
            _anchors = (previous_data or {}).get("metric_anchors") if isinstance(previous_data, dict) else None

        fn_anchor = globals().get("rebuild_metrics_from_snapshots_with_anchors")
        fn_schema = globals().get("rebuild_metrics_from_snapshots_schema_only")
        fn_legacy = globals().get("rebuild_metrics_from_snapshots")

        if isinstance(_anchors, dict) and _anchors and callable(fn_anchor):
            fn_rebuild = fn_anchor
        elif callable(fn_schema):
            fn_rebuild = fn_schema
        elif callable(fn_legacy):
            fn_rebuild = fn_legacy
    except Exception:
        pass
    # =====================================================================

    rebuilt_metrics = {}
    try:
        if callable(fn_rebuild):
            rebuilt_metrics = fn_rebuild(previous_data, baseline_sources_cache, web_context=web_context)
    except Exception:
        rebuilt_metrics = {}

    # =====================================================================
    # PATCH RMS_RETRY1 (ADDITIVE): If anchor-aware rebuild returns empty, retry schema-only
    # =====================================================================
    try:
        if (not isinstance(rebuilt_metrics, dict) or not rebuilt_metrics):
            fn_schema = globals().get("rebuild_metrics_from_snapshots_schema_only")
            if callable(fn_schema) and getattr(fn_rebuild, "__name__", "") == "rebuild_metrics_from_snapshots_with_anchors":
                rebuilt_metrics = fn_schema(previous_data, baseline_sources_cache, web_context=web_context) or {}
    except Exception:
        pass
    # =====================================================================

    # Optional debug visibility (safe additive fields)
    try:
        output.setdefault("debug", {})
        output["debug"]["rebuild_fn"] = getattr(fn_rebuild, "__name__", "None")
        try:
            # PATCH RMS_DEBUG1 (ADDITIVE): expose counts to diagnose empty rebuilds
            _anchors2 = None
            fn_get_anchors2 = globals().get("_get_metric_anchors_any")
            if callable(fn_get_anchors2):
                _anchors2 = fn_get_anchors2(previous_data)
            elif isinstance(previous_data, dict):
                _anchors2 = previous_data.get("metric_anchors")
            output["debug"]["anchor_count"] = len(_anchors2) if isinstance(_anchors2, dict) else 0

            _schema2 = None
            if isinstance(previous_data, dict):
                _schema2 = previous_data.get("metric_schema_frozen") or (previous_data.get("primary_response") or {}).get("metric_schema_frozen") or (previous_data.get("results") or {}).get("metric_schema_frozen")
            output["debug"]["schema_count"] = len(_schema2) if isinstance(_schema2, dict) else 0

            # candidate count from baseline_sources_cache extracted_numbers
            _cand_n = 0
            if isinstance(baseline_sources_cache, list):
                for _s in baseline_sources_cache:
                    if isinstance(_s, dict) and isinstance(_s.get("extracted_numbers"), list):
                        _cand_n += len(_s.get("extracted_numbers") or [])
            output["debug"]["snapshot_candidate_count"] = int(_cand_n)
        except Exception:
            pass

# ===================== PATCH RMS_DISPATCH3 (ADDITIVE) =====================
        # Loud warning if anchors exist but we did NOT select anchor-aware rebuild.
        try:
            if anchors_for_dispatch and getattr(fn_rebuild, "__name__", "") != "rebuild_metrics_from_snapshots_with_anchors":
                output["debug"]["rebuild_dispatch_warning"] = "metric_anchors present but anchor-aware rebuild not selected"
        except Exception:
            pass
# =================== END PATCH RMS_DISPATCH3 (ADDITIVE) ===================
        output["debug"]["rebuilt_metric_count"] = len(rebuilt_metrics) if isinstance(rebuilt_metrics, dict) else 0
    except Exception:
        pass

    # If rebuilt_metrics is non-empty, wire it into the expected place for downstream diff.
    if isinstance(rebuilt_metrics, dict) and rebuilt_metrics:
        output.setdefault("results", {})
        output["results"]["rebuilt_primary_metrics_canonical"] = rebuilt_metrics
    # =================== END PATCH RMS_WIRE1 (ADDITIVE) ===================

    # =====================================================================
    # PATCH RMS_WIRE2 (ADDITIVE): ensure guardrail sees rebuilt metrics
    # Why:
    # - Some evolution paths validate rebuild presence via `current_metrics` (not output['results']).
    # - If we successfully rebuilt metrics into `rebuilt_metrics` but did not populate `current_metrics`,
    #   the code would incorrectly fail with "rebuild missing/empty".
    # Behavior:
    # - If `current_metrics` is empty and `rebuilt_metrics` is a non-empty dict, set `current_metrics`
    #   from `rebuilt_metrics` (schema-only or anchor-aware).
    # =====================================================================
    try:
        if (not isinstance(current_metrics, dict) or not current_metrics) and isinstance(rebuilt_metrics, dict) and rebuilt_metrics:
            current_metrics = rebuilt_metrics
            output.setdefault("debug", {})
            output["debug"]["current_metrics_source"] = "rebuilt_metrics"
    except Exception:
        pass
    # =====================================================================




    # =====================================================================

    # If we cannot rebuild metrics, return a tight result that still exposes source_results for debugging
    if not isinstance(current_metrics, dict) or not current_metrics:
        output["status"] = "failed"
        output["message"] = "Valid snapshots exist, but no metric rebuild function is wired (rebuild_metrics_from_snapshots missing/empty)."
        output["source_results"] = baseline_sources_cache[:50]
        output["sources_checked"] = len(baseline_sources_cache)
        output["sources_fetched"] = len(baseline_sources_cache)
        output["interpretation"] = "Snapshot-ready but metric rebuild not implemented; add rebuild_metrics_from_snapshots() to converge evolution with analysis."
        return output

    # Diff using existing diff helper if present
    metric_changes = []
    try:
        fn_diff = globals().get("diff_metrics_by_name")
        if callable(fn_diff):
            # =====================================================================
            # PATCH E5 (ADDITIVE): pass metric_anchors into cur_response for diff
            # Why:
            # - diff_metrics_by_name() can also look at cur_response.metric_anchors for anchor_hash.
            # - Helps when rebuilt metric rows are missing anchor_hash (older snapshots / partial rebuild).
            # =====================================================================
            try:
                cur_resp_for_diff = {"primary_metrics_canonical": current_metrics}

                # carry schema too (harmless; helps display_name/definition if you ever use cur-side)
                if isinstance(prev_response, dict) and isinstance(prev_response.get("metric_schema_frozen"), dict):
                    cur_resp_for_diff["metric_schema_frozen"] = prev_response.get("metric_schema_frozen")

                # carry anchors (prefer the ones we extracted above)
                try:
                    _ma = None
                    if "metric_anchors" in locals() and isinstance(metric_anchors, dict) and metric_anchors:
                        _ma = metric_anchors
                    elif isinstance(prev_response, dict) and isinstance(prev_response.get("metric_anchors"), dict):
                        _ma = prev_response.get("metric_anchors")

                    if isinstance(_ma, dict) and _ma:
                        cur_resp_for_diff["metric_anchors"] = _ma
                except Exception:
                    pass

                metric_changes, unchanged, increased, decreased, found = fn_diff(prev_response, cur_resp_for_diff)
            except Exception:
                metric_changes, unchanged, increased, decreased, found = ([], 0, 0, 0, 0)
            # =====================================================================
        else:
            metric_changes, unchanged, increased, decreased, found = ([], 0, 0, 0, 0)
    except Exception:
        metric_changes, unchanged, increased, decreased, found = ([], 0, 0, 0, 0)

    output["metric_changes"] = metric_changes or []
    output["summary"]["total_metrics"] = len(output["metric_changes"])
    output["summary"]["metrics_found"] = int(found or 0)
    output["summary"]["metrics_increased"] = int(increased or 0)
    output["summary"]["metrics_decreased"] = int(decreased or 0)
    output["summary"]["metrics_unchanged"] = int(unchanged or 0)

    total = max(1, len(output["metric_changes"]))
    output["stability_score"] = (output["summary"]["metrics_unchanged"] / total) * 100.0
    # =====================================================================
    # PATCH ES8 (ADDITIVE): convergence hashes for drift=0 diagnostics
    # - source_snapshot_hash: sorted (url,fingerprint) hash
    # - canonical_universe_hash: hash of canonical_key universe
    # - schema_hash: hash of tolerance/unit-relevant schema fields
    # =====================================================================
    try:
        _pairs = _es_sorted_pairs_from_sources_cache(baseline_sources_cache)
        _sig = "|".join([f"{u}#{fp}" for (u, fp) in _pairs])
        output["source_snapshot_hash"] = _es_hash_text(_sig) if _sig else None
    except Exception:
        pass
    try:
        _pmc = prev_response.get("primary_metrics_canonical") if isinstance(prev_response, dict) else {}
        _msf = prev_response.get("metric_schema_frozen") if isinstance(prev_response, dict) else {}
        output["canonical_universe_hash"] = _es_compute_canonical_universe_hash(_pmc or {}, _msf or {})
        output["schema_hash"] = _es_compute_schema_hash(_msf or {})
    except Exception:
        pass
    # =====================================================================

    # =====================================================================
    # PATCH ES9 (ADDITIVE): stronger identical-input invariant (warn-only)
    # If snapshot + universe + schema hashes all match previous, any non-zero
    # changes indicate drift and should raise a loud warning flag.
    # =====================================================================
    try:
        _prev_snap = (previous_data or {}).get("source_snapshot_hash") or (previous_data or {}).get("results", {}).get("source_snapshot_hash")
        _prev_uni  = (previous_data or {}).get("canonical_universe_hash") or (previous_data or {}).get("results", {}).get("canonical_universe_hash")
        _prev_sch  = (previous_data or {}).get("schema_hash") or (previous_data or {}).get("results", {}).get("schema_hash")

        _cur_snap = output.get("source_snapshot_hash")
        _cur_uni  = output.get("canonical_universe_hash")
        _cur_sch  = output.get("schema_hash")

        _all_match = bool(_prev_snap and _prev_uni and _prev_sch and _cur_snap and _cur_uni and _cur_sch and
                          (_prev_snap == _cur_snap) and (_prev_uni == _cur_uni) and (_prev_sch == _cur_sch))

        if _all_match:
            _has_changes = bool(output.get("metric_changes"))
            if _has_changes:
                output.setdefault("warnings", [])
                output["warnings"].append({
                    "type": "identical_inputs_nonzero_changes",
                    "message": "Snapshots + schema + canonical universe hashes match previous, but metric_changes is non-empty (drift suspected).",
                })
                output["drift_suspected"] = True
            else:
                output["drift_suspected"] = False
    except Exception:
        pass
    # =====================================================================


    output["source_results"] = baseline_sources_cache[:50]
    output["sources_checked"] = len(baseline_sources_cache)
    output["sources_fetched"] = len(baseline_sources_cache)

    # =====================================================================
    # PATCH E6 (ADDITIVE): populate numbers_extracted_total for debug/telemetry
    # =====================================================================
    try:
        total_nums = 0
        for sr in baseline_sources_cache or []:
            if isinstance(sr, dict) and isinstance(sr.get("extracted_numbers"), list):
                total_nums += len(sr.get("extracted_numbers") or [])
        output["numbers_extracted_total"] = int(total_nums)
    except Exception:
        pass
    # =====================================================================

    output["message"] = "Source-anchored evolution completed (snapshot-gated, analysis-aligned)."
    output["interpretation"] = "Evolution used cached source snapshots only; no brute-force candidate harvesting."

    # =====================================================================
    # PATCH ES8 (ADDITIVE): deterministic ordering of evolution outputs
    # - Sort metric_changes rows by stable composite key
    # - Sort key lists for stable JSON output across identical inputs.
    # - Output-only ordering; does not change classification logic.
    # =====================================================================
    def _es_metric_row_sort_key(r):
        try:
            if not isinstance(r, dict):
                return (9, "")
            return (
                0,
                str(r.get("canonical_key") or r.get("metric_key") or r.get("name") or ""),
                str(r.get("anchor_hash") or ""),
                str(r.get("source_url") or ""),
                _es_stable_sort_key(r.get("value_norm") if r.get("value_norm") is not None else r.get("value")),
            )
        except Exception:
            return (9, str(r))

    try:
        if isinstance(output.get("metric_changes"), list):
            output["metric_changes"] = sorted(output["metric_changes"], key=_es_metric_row_sort_key)
    except Exception:
        pass
    try:
        for _k in ("unchanged", "increased", "decreased", "found"):
            if isinstance(output.get(_k), list):
                output[_k] = sorted([str(x) for x in output[_k]])
    except Exception:
        pass
    try:
        # Stable subset ordering for debug payload
        if isinstance(output.get("source_results"), list):
            output["source_results"] = sorted(
                output["source_results"],
                key=lambda sr: str(sr.get("source_url") or sr.get("url") or "")
                if isinstance(sr, dict) else str(sr)
            )
    except Exception:
        pass
    # =====================================================================


    return output


# ===================== PATCH RMS_CORE1 (ADDITIVE) =====================
def rebuild_metrics_from_snapshots_schema_only(prev_response: dict, baseline_sources_cache, web_context=None) -> dict:
    """
    Minimal deterministic rebuild:
      - Uses ONLY: baseline_sources_cache + frozen schema (or derives schema from canonical metrics)
      - No re-fetch
      - No heuristic name fallback outside schema fields
      - Deterministic selection + ordering

    Returns a dict shaped like primary_metrics_canonical (by canonical_key).
    """
    import re

    if not isinstance(prev_response, dict):
        return {}

    # 1) Obtain frozen schema (required contract for schema-driven rebuild)
    metric_schema = (
        prev_response.get("metric_schema_frozen")
        or (prev_response.get("primary_response") or {}).get("metric_schema_frozen")
        or (prev_response.get("results") or {}).get("metric_schema_frozen")
    )

    # If schema is missing but canonical metrics exist, derive schema deterministically
    if not metric_schema:
        try:
            canon = (
                prev_response.get("primary_metrics_canonical")
                or (prev_response.get("primary_response") or {}).get("primary_metrics_canonical")
                or (prev_response.get("results") or {}).get("primary_metrics_canonical")
            )
            fn_freeze = globals().get("freeze_metric_schema")
            if canon and callable(fn_freeze):
                metric_schema = fn_freeze(canon)
        except Exception:
            metric_schema = None

    if not isinstance(metric_schema, dict) or not metric_schema:
        return {}

    # 2) Flatten candidates (must come from snapshots/cache, no re-fetch)
    candidates = []
    if isinstance(baseline_sources_cache, dict) and isinstance(baseline_sources_cache.get("snapshots"), list):
        source_entries = baseline_sources_cache.get("snapshots", [])
    elif isinstance(baseline_sources_cache, list):
        source_entries = baseline_sources_cache
    else:
        source_entries = []

    # Candidate normalization helpers
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    def _unit_family_guess(unit: str) -> str:
        u = (unit or "").strip().lower()
        if u in ("%", "percent", "percentage"):
            return "percent"
        if any(x in u for x in ("usd", "$", "eur", "gbp", "jpy", "cny", "aud", "sgd")):
            return "currency"
        if any(x in u for x in ("unit", "units", "vehicle", "vehicles", "car", "cars", "kwh", "mwh", "gwh", "twh")):
            return "quantity"
        return ""

    # Prefer already-extracted numbers in snapshots; otherwise optionally extract from stored text if present.
    fn_extract = globals().get("extract_numbers_with_context")

    for s in source_entries:
        if not isinstance(s, dict):
            continue
        url = s.get("source_url") or s.get("url") or ""
        xs = s.get("extracted_numbers")
        if isinstance(xs, list) and xs:
            for c in xs:
                if isinstance(c, dict):
                    c2 = dict(c)
                    c2.setdefault("source_url", url)
                    candidates.append(c2)
            continue

        # Optional: if snapshot stores text, we can extract deterministically (no re-fetch).
        txt = s.get("text") or s.get("raw_text") or s.get("content_text") or ""
        if txt and callable(fn_extract):
            try:
                xs2 = fn_extract(txt, source_url=url)
                if isinstance(xs2, list):
                    for c in xs2:
                        if isinstance(c, dict):
                            c2 = dict(c)
                            c2.setdefault("source_url", url)
                            candidates.append(c2)
            except Exception:
                pass

    # Drop junk + enforce deterministic ordering
    def _cand_sort_key(c: dict):
        return (
            str(c.get("anchor_hash") or ""),
            str(c.get("source_url") or ""),
            int(c.get("start_idx") or 0),
            str(c.get("raw") or ""),
            str(c.get("unit") or ""),
            float(c.get("value_norm") or 0.0),
        )

    filtered = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if c.get("is_junk") is True:
            continue
        filtered.append(c)

    filtered.sort(key=_cand_sort_key)

    # 3) Schema-driven selection
    rebuilt = {}
    for canonical_key, sch in metric_schema.items():
        if not isinstance(sch, dict):
            continue

        # Schema tokens/keywords
        name = sch.get("name") or canonical_key
        keywords = sch.get("keywords") or sch.get("keyword_hints") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        kw_norm = [_norm(k) for k in keywords if k]

        # Expected family/dimension
        expected_dim = (sch.get("dimension") or sch.get("unit_family") or "").lower().strip()

        best = None
        best_score = None

        for c in filtered:
            ctx = _norm(c.get("context") or c.get("context_window") or "")
            raw = _norm(c.get("raw") or "")
            unit = c.get("unit") or ""
            fam = _unit_family_guess(unit)
            if expected_dim and fam and expected_dim not in (fam,):
                # strict family check when we can infer a family
                continue

            score = 0
            # keyword match only from schema-provided keywords
            for k in kw_norm:
                if k and (k in ctx or k in raw):
                    score += 10

            # Prefer candidates that have an anchor_hash (stability)
            if c.get("anchor_hash"):
                score += 1

            # Deterministic tie-breakers: earlier in list wins if equal score
            if best is None or score > best_score:
                best = c
                best_score = score

        if best is None or (best_score is not None and best_score <= 0):
            # No schema-consistent evidence found -> omit (or could mark proxy; keep minimal here)
            continue

        rebuilt[canonical_key] = {
            "canonical_key": canonical_key,
            "name": name,
            "value": best.get("value"),
            "unit": best.get("unit") or "",
            "value_norm": best.get("value_norm"),
            "source_url": best.get("source_url") or "",
            "anchor_hash": best.get("anchor_hash") or "",
            "evidence": [{
                "source_url": best.get("source_url") or "",
                "raw": best.get("raw") or "",
                "context_snippet": (best.get("context") or best.get("context_window") or "")[:400],
                "anchor_hash": best.get("anchor_hash") or "",
                "method": "schema_only_rebuild",
            }],
        }

    return rebuilt
# =================== END PATCH RMS_CORE1 (ADDITIVE) ===================





def extract_context_keywords(metric_name: str) -> List[str]:
    """
    General-purpose keyword extraction for matching metric names to page contexts.

    Goals:
    - Work for ANY topic (not tourism-specific)
    - Keep deterministic behavior
    - Extract years/quarters, key financial/stat terms, and meaningful tokens
    """
    if not metric_name:
        return []

    name = str(metric_name)
    n = name.lower()

    keywords: List[str] = []

    # Years (e.g., 2019, 2024)
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", name)
    keywords.extend(years)

    # Quarters / time buckets
    q = re.findall(r"\bq[1-4]\b", n)
    keywords.extend([x.upper() for x in q])

    # Common metric concepts (broad, cross-industry)
    concept_phrases = [
        "market size", "revenue", "sales", "turnover", "profit", "operating profit",
        "ebit", "ebitda", "net income", "gross margin", "margin",
        "growth", "yoy", "cagr", "share", "penetration",
        "forecast", "projected", "projection", "estimate", "expected",
        "actual", "baseline", "target",
        "volume", "units", "shipments", "users", "subscribers", "visitors",
        "price", "asp", "arpu", "aov",
        "inflation", "gdp", "unemployment", "interest rate"
    ]
    for p in concept_phrases:
        if p in n:
            keywords.append(p)

    # Units / scales that help matching
    unit_hints = ["trillion", "billion", "million", "thousand", "%", "percent"]
    for u in unit_hints:
        if u in n:
            keywords.append(u)

    # Tokenize remaining meaningful words
    tokens = re.findall(r"[a-z0-9]+", n)
    stop = {
        "the","and","or","of","in","to","for","by","from","with","on","at","as",
        "total","overall","average","avg","number","rate","value","amount",
        "annual","year","years","monthly","month","daily","day","quarter","quarters"
    }
    for t in tokens:
        if t in stop:
            continue
        if len(t) <= 2:
            continue
        keywords.append(t)

    # De-dup, keep stable ordering
    seen = set()
    out = []
    for k in keywords:
        if k and k not in seen:
            seen.add(k)
            out.append(k)

    return out[:30]

def extract_numbers_with_context(text, source_url: str = "", max_results: int = 350):
    """
    Extract numeric candidates with context windows (analysis-aligned, hardened).

    Fixes / tightening:
    - ALWAYS returns a list (never None)  ✅ critical for snapshots & evolution
    - Strips HTML tags/scripts/styles if HTML-like
    - Nav/chrome/junk rejection (analytics, cookie banners, menus, footers, etc.)
    - Suppress year-only candidates (e.g., "2024") unless clearly a metric
    - Suppress ID-like long integers, phone-like patterns, DOI/ISBN-like contexts
    - Captures currency + scale + percent + common magnitude suffixes
    - Adds anchor_hash for stable matching
    """
    import re
    import hashlib

    if not text or not str(text).strip():
        return []

    raw = str(text)

    # ---------- helpers ----------
    def _sha1(s: str) -> str:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

    def _normalize_unit(u: str) -> str:
        u = (u or "").strip()
        if not u:
            return ""
        ul = u.lower().replace(" ", "")

        # Energy units (must come before magnitude)
        if "twh" in ul:
            return "TWh"
        if "gwh" in ul:
            return "GWh"
        if "mwh" in ul:
            return "MWh"
        if "kwh" in ul:
            return "kWh"
        if ul == "wh":
            return "Wh"

        # Magnitudes (case-insensitive; fix: accept single-letter suffixes)
        if ul in ("bn", "billion", "b"):
            return "B"
        if ul in ("mn", "mio", "million", "m"):
            return "M"
        if ul in ("k", "thousand", "000"):
            return "K"
        if ul in ("trillion", "tn", "t"):
            return "T"

        if ul in ("pct", "percent", "%"):
            return "%"

        return u

    def _looks_html(s: str) -> bool:
        sl = s.lower()
        return ("<html" in sl) or ("<div" in sl) or ("<p" in sl) or ("<script" in sl) or ("</" in sl)

    def _html_to_text(s: str) -> str:
        # Prefer BeautifulSoup if available
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(s, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "header", "footer", "nav", "form"]):
                try:
                    tag.decompose()
                except Exception:
                    pass
            txt = soup.get_text(separator=" ", strip=True)
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt
        except Exception:
            # fallback: cheap strip
            s2 = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", s)
            s2 = re.sub(r"(?is)<[^>]+>", " ", s2)
            s2 = re.sub(r"\s+", " ", s2).strip()
            return s2

    def _is_phone_like(ctx: str, rawnum: str) -> bool:
        # strict phone pattern or phone keywords nearby
        if re.search(r"\b\d{3}-\d{3}-\d{4}\b", rawnum):
            return True
        c = (ctx or "").lower()
        if any(k in c for k in ["call", "phone", "tel:", "telephone", "contact us", "whatsapp"]):
            if re.search(r"\b\d{7,}\b", rawnum):
                return True
        return False

    def _is_id_like(val_str: str, ctx: str) -> bool:
        # very long digit strings typically IDs, unless explicitly monetary with symbols
        digits = re.sub(r"\D", "", val_str or "")
        if len(digits) >= 13:
            c = (ctx or "").lower()
            if any(k in c for k in ["isbn", "doi", "issn", "arxiv", "repec", "id:", "order", "invoice", "reference"]):
                return True
            # generic ID-like (too many digits)
            return True
        return False

    def _chrome_junk(ctx: str) -> bool:
        c = (ctx or "").lower()
        # common site chrome / analytics / cookie / nav junk
        bad = [
            "googleanalyticsobject", "gtag(", "googletagmanager", "analytics", "doubleclick",
            "cookie", "consent", "privacy", "terms", "copyright", "all rights reserved",
            "subscribe", "newsletter", "sign in", "login", "menu", "search", "breadcrumb",
            "share this", "follow us", "social media", "footer", "header", "nav", "sitemap"
        ]
        if any(b in c for b in bad):
            return True
        # css/js-like
        if any(b in c for b in ["function(", "var ", "const ", "let ", "webpack", "sourcemappingurl", ".css", "{", "};"]):
            return True
        # low alpha ratio
        if len(c) > 80:
            letters = sum(ch.isalpha() for ch in c)
            if letters / max(1, len(c)) < 0.18:
                return True
        return False

    def _year_only_suppression(num: float, unit: str, rawnum: str, ctx: str) -> bool:
        # suppress standalone 4-digit years like 2024 with no unit/currency
        if unit:
            return False
        s = (rawnum or "").strip()
        if re.fullmatch(r"\d{4}", s):
            year = int(s)
            if 1900 <= year <= 2099:
                c = (ctx or "").lower()
                allow_kw = ["cagr", "growth", "inflation", "gdp", "revenue", "market", "sales", "shipments", "capacity"]
                if not any(k in c for k in allow_kw):
                    return True
        return False

    # -------------------------------------------------------------------------
    # ADDITIVE (Patch A1): fix common "split year" artifact (e.g., "202 5" -> "2025")
    # Do this AFTER HTML->text and BEFORE regex extraction.
    # -------------------------------------------------------------------------

    # ---------- normalize to visible text ----------
    if _looks_html(raw):
        raw = _html_to_text(raw)

    # cap huge pages
    raw = raw[:250_000]

    # ---- ADDITIVE: fix common "split year" artifact (e.g., "202 5" -> "2025") ----
    raw = re.sub(r"\b((?:19|20)\d)\s+(\d)\b", r"\1\2", raw)
    # -----------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # PATCH A3 (ADDITIVE): year-range detector (tag-only, does NOT drop candidates)
    # -------------------------------------------------------------------------
    def _is_year_range_context(ctx: str) -> bool:
        return bool(re.search(r"\b(19|20)\d{2}\s*(?:-|–|—|to)\s*(19|20)\d{2}\b", ctx or "", flags=re.I))

    # -------------------------------------------------------------------------
    # ADDITIVE (Patch A2): non-destructive junk tagger
    # - We DO NOT filter here; we tag and downstream excludes by default.
    # -------------------------------------------------------------------------
    def _junk_tag(value: float, unit: str, raw_disp: str, ctx: str):
        """
        Non-destructive junk classifier.
        Returns (is_junk: bool, reason: str).
        """
        c = (ctx or "").lower()
        u = (unit or "").strip()

        # =========================
        # PATCH A3 (TAG ONLY): year-range endpoints are usually timeline metadata
        # =========================
        try:
            iv = int(float(value))
            if u == "" and 1900 <= iv <= 2099 and _is_year_range_context(ctx):
                return True, "year_range"
        except Exception:
            pass

        nav_hits = [
            "skip to content", "menu", "search", "login", "sign in", "sign up",
            "subscribe", "newsletter", "cookie", "privacy", "terms", "copyright",
            "all rights reserved", "back to top", "next", "previous", "page ",
            "home", "about", "contact", "sitemap", "breadcrumb"
        ]
        if any(h in c for h in nav_hits):
            try:
                if u == "" and abs(float(value)) <= 20:
                    return True, "nav_small_int"
            except Exception:
                pass

        if u == "":
            try:
                if abs(float(value)) <= 12:
                    if any(h in c for h in ["•", "–", "step", "chapter", "section", "item", "no."]):
                        return True, "enumeration_small_int"
            except Exception:
                pass

        if u == "":
            try:
                iv = int(abs(float(value)))
                if 190 <= iv <= 209:
                    if any(x in (raw_disp or "") for x in ["202", "203", "204", "205", "206", "207", "208", "209"]):
                        return True, "year_fragment_3digit"
            except Exception:
                pass

        return False, ""

    # -------------------------------------------------------------------------
    # PATCH M1 (ADDITIVE): semantic classifier for associations like "share" vs "units"
    # NOTE: moved OUTSIDE the loop for determinism + speed (no behavioral change).
    # Also emits a "measure_assoc" label that downstream can display easily.
    # -------------------------------------------------------------------------
    def _classify_measure(unit_tag: str, ctx: str):
        """
        Returns (measure_kind, measure_assoc):
          - measure_kind: stable internal tag (share_pct / growth_pct / count_units / money / etc.)
          - measure_assoc: human-meaning label ("share", "growth", "units", "money", "energy", etc.)
        """
        c = (ctx or "").lower()
        ut = (unit_tag or "").strip()

        if ut == "%":
            if any(k in c for k in ["market share", "share of", "share", "penetration", "portion", "contribution"]):
                return "share_pct", "share"
            if any(k in c for k in ["growth", "cagr", "increase", "decrease", "yoy", "mom", "qoq", "rate"]):
                return "growth_pct", "growth"
            return "percent_other", "percent"

        if ut in ("K", "M", "B", "T", ""):
            if any(k in c for k in ["units", "unit", "vehicles", "cars", "sold", "sales volume", "shipments", "deliveries", "registrations"]):
                return "count_units", "units"
            if any(k in c for k in ["revenue", "sales ($", "usd", "$", "market size", "valuation", "turnover"]):
                return "money", "money"
            return "magnitude_other", "magnitude"

        if ut in ("TWh", "GWh", "MWh", "kWh", "Wh"):
            return "energy", "energy"

        return "other", "other"
    # -------------------------------------------------------------------------

    # ---------- extraction pattern ----------
    # =========================
    # PATCH N1 (ADDITIVE, BUGFIX): currency tokens
    # - Fix US$ being parsed as S$ by matching US\$ first.
    # - Also accept "US$" as a single token (case-insensitive).
    # =========================
    pat = re.compile(
        r"(US\$|US\$(?!\w)|S\$|\$|USD|SGD|EUR|€|GBP|£)?\s*"
        # =========================
        # PATCH N2 (ADDITIVE, BUGFIX): avoid capturing negative year from year-range
        # - We'll still allow negatives generally, but we'll tag the special "2025-2030" case below.
        # (No behavior change for real negatives like -1.2% etc.)
        # =========================
        r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(?!\d)\s*"
        # =========================
        # PATCH N3 (ADDITIVE, BUGFIX): capture 'tn' magnitude explicitly
        # - Keep your A5 safeguard: single-letter magnitudes only match if NOT followed by a letter.
        # =========================
        r"(TWh|GWh|MWh|kWh|Wh|tn|(?:T|B|M|K)(?![A-Za-z])|trillion|billion|million|bn|mn|%|percent)?",
        flags=re.I
    )

    out = []
    for m in pat.finditer(raw):
        cur = (m.group(1) or "").strip()
        num_s = (m.group(2) or "").strip()
        unit_s = (m.group(3) or "").strip()

        if not num_s:
            continue

        start = max(0, m.start() - 160)
        end = min(len(raw), m.end() + 160)
        ctx = raw[start:end].replace("\n", " ")
        ctx = re.sub(r"\s+", " ", ctx).strip()
        ctx_store = ctx[:240]

        # numeric parse
        try:
            val = float(num_s.replace(",", ""))
        except Exception:
            continue

        # normalize unit
        unit = _normalize_unit(unit_s)

        raw_disp = f"{cur} {num_s}{unit_s}".strip()
        raw_num_only = (cur + num_s).strip()

        if _chrome_junk(ctx_store):
            continue
        if _is_phone_like(ctx_store, raw_disp):
            continue
        if _is_id_like(raw_disp, ctx_store):
            continue
        if _year_only_suppression(val, unit, num_s, ctx_store):
            continue

        # =========================
        # PATCH N2b (ADDITIVE, BUGFIX): tag the "negative year from range" case as junk
        # Example: "CAGR 2025-2030" producing "-2030"
        # - Do NOT drop here (keep non-destructive policy); just tag.
        # =========================
        neg_year_from_range = False
        try:
            if num_s.startswith("-"):
                iv = int(abs(float(val)))
                if 1900 <= iv <= 2099:
                    # look immediately behind the match for a digit (the "2025" in "2025-2030")
                    if m.start() > 0 and raw[m.start() - 1].isdigit():
                        neg_year_from_range = True
        except Exception:
            neg_year_from_range = False
        # =========================

        anchor_hash = _sha1(f"{source_url}|{raw_disp}|{ctx_store}")
        is_junk, junk_reason = _junk_tag(val, unit, raw_disp, ctx_store)

        # =========================
        # PATCH N2c (ADDITIVE): override junk tagging reason when we confidently detect this bug
        # =========================
        if neg_year_from_range:
            is_junk = True
            junk_reason = "year_range_negative_endpoint"
        # =========================

        # semantic association tags
        measure_kind, measure_assoc = _classify_measure(unit, ctx_store)

        out.append({
            "value": val,
            "unit": unit,
            "raw": raw_disp,
            "source_url": source_url,
            "context": ctx_store,
            "context_snippet": ctx_store,
            "anchor_hash": anchor_hash,

            "is_junk": bool(is_junk),
            "junk_reason": junk_reason,
            "start_idx": int(m.start()),
            "end_idx": int(m.end()),

            "measure_kind": measure_kind,
            "measure_assoc": measure_assoc,
        })

        if len(out) >= int(max_results or 350):
            break

    return out



def extract_numbers_with_context_pdf(text):
    """
    PDF-specialized extractor wrapper.

    Tightening changes (v7.29+):
    - Inherit the year-only rejection from extract_numbers_with_context().
    - Keep boilerplate filters; prefer metric/table-like contexts.
    """
    import re

    if not text:
        return []

    base = extract_numbers_with_context(text) or []

    def _bad_pdf_context(ctx):
        c = (ctx or "").lower()
        bad = [
            "issn", "isbn", "doi", "catalogue", "legal notice",
            "all rights reserved", "reproduction is authorised",
            "printed by", "manuscript completed", "©", "copyright",
            "table of contents"
        ]
        return any(b in c for b in bad)

    def _good_pdf_context(ctx):
        c = (ctx or "").lower()
        # Lightweight heuristic: "table-ish" or "metric-ish"
        good = [
            "market", "revenue", "sales", "capacity", "generation", "growth",
            "cagr", "forecast", "projection", "increase", "decrease",
            "percent", "%", "billion", "million", "trillion", "usd", "eur", "gbp", "sgd"
        ]
        return any(g in c for g in good)

    filtered = []
    for n in base:
        if not isinstance(n, dict):
            continue
        ctx = n.get("context") or ""
        if _bad_pdf_context(ctx):
            continue
        filtered.append(n)

    # Prefer contexts that look "metric-like"
    preferred = [n for n in filtered if _good_pdf_context(n.get("context") or "")]

    # If we filtered too aggressively, fall back safely
    if preferred:
        return preferred
    if filtered:
        return filtered
    return base


def calculate_context_match(keywords: List[str], context: str) -> float:
    """Calculate how well keywords match the context (deterministic)."""
    if not context:
        return 0.0

    context_lower = context.lower()

    # If no keywords, give a small baseline (we'll rely more on value_score)
    if not keywords:
        return 0.25

    # Year keywords MUST match if present
    year_keywords = [kw for kw in keywords if re.fullmatch(r"20\d{2}", kw)]
    if year_keywords:
        if not any(y in context_lower for y in year_keywords):
            return 0.0

    matches = sum(1 for kw in keywords if kw.lower() in context_lower)

    # Instead of hard "matches < 2 = reject", scale smoothly:
    match_ratio = matches / max(len(keywords), 1)

    # If nothing matches, reject
    if matches == 0:
        return 0.0

    # Score between 0.35 and 1.0 depending on ratio
    return 0.35 + (match_ratio * 0.65)


def render_source_anchored_results(results, query: str):
    """Render source-anchored evolution results (guarded + backward compatible + tuned debug UI)."""
    import math
    import re
    from collections import Counter, defaultdict

    st.header("📈 Source-Anchored Evolution Analysis")
    st.markdown(f"**Query:** {query}")

    if not isinstance(results, dict):
        st.error("❌ Evolution returned an invalid result payload (not a dict).")
        st.write(results)
        return

    status = (results.get("status") or "").strip().lower()
    message = results.get("message") or ""

    def _safe_int(x, default=0) -> int:
        try:
            if x is None:
                return default
            return int(x)
        except Exception:
            return default

    def _safe_float(x, default=0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _fmt_pct(x, default="—") -> str:
        try:
            if x is None:
                return default
            v = float(x)
            if math.isnan(v):
                return default
            return f"{v:.0f}%"
        except Exception:
            return default

    def _fmt_change_pct(x) -> str:
        try:
            if x is None:
                return "-"
            v = float(x)
            if math.isnan(v):
                return "-"
            return f"{v:+.1f}%"
        except Exception:
            return "-"

    def _short(u: str, n: int = 95) -> str:
        if not u:
            return ""
        return (u[:n] + "…") if len(u) > n else u

    if status != "success":
        st.error(f"❌ {message or 'Evolution failed'}")
        sr = results.get("source_results") or []
        if isinstance(sr, list) and sr:
            st.subheader("🔗 Source Verification")
            for src in sr:
                if not isinstance(src, dict):
                    continue
                u = _short((src.get("url") or ""), 90)
                st.error(f"❌ {u} - {src.get('status_detail', 'Unknown error')}")
        return

    sources_checked = _safe_int(results.get("sources_checked"), 0)
    sources_fetched = _safe_int(results.get("sources_fetched"), 0)
    stability = _safe_float(results.get("stability_score"), 0.0)
    summary = results.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}

    metrics_inc = _safe_int(summary.get("metrics_increased"), 0)
    metrics_dec = _safe_int(summary.get("metrics_decreased"), 0)
    metrics_unch = _safe_int(summary.get("metrics_unchanged"), 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sources Checked", sources_checked)
    col2.metric("Sources Fetched", sources_fetched)
    col3.metric("Stability", _fmt_pct(stability))
    if metrics_inc > metrics_dec:
        col4.success("📈 Trending Up")
    elif metrics_dec > metrics_inc:
        col4.error("📉 Trending Down")
    else:
        col4.info("➡️ Stable")

    if message:
        st.caption(message)

    st.markdown("---")

    # -------------------------
    # Source status
    # -------------------------
    st.subheader("🔗 Source Verification")
    src_results = results.get("source_results") or []
    if not isinstance(src_results, list):
        src_results = []

    # If everything failed, show breakdown
    if sources_checked > 0 and sources_fetched == 0 and src_results:
        reasons = []
        for s in src_results:
            if isinstance(s, dict):
                reasons.append((s.get("status_detail") or "unknown").split(":")[0])
        top = Counter(reasons).most_common(6)
        if top:
            st.warning("No sources were fetched successfully. Top failure types:")
            st.write({k: v for k, v in top})

    for src in src_results:
        if not isinstance(src, dict):
            continue
        url = src.get("url") or ""
        sstatus = src.get("status") or ""
        detail = src.get("status_detail") or ""
        ctype = src.get("content_type") or ""
        nfound = _safe_int(src.get("numbers_found"), 0)

        short = _short(url, 95)

        # show extra debug flags if present
        flags = []
        if src.get("snapshot_origin"):
            flags.append(f"origin={src.get('snapshot_origin')}")
        if src.get("is_homepage"):
            flags.append("homepage")
        if src.get("skip_reason"):
            flags.append(f"skip={src.get('skip_reason')}")
        if src.get("quality_score") is not None:
            try:
                flags.append(f"q={float(src.get('quality_score')):.2f}")
            except Exception:
                flags.append(f"q={src.get('quality_score')}")

        flag_txt = f" • {' • '.join(flags)}" if flags else ""

        if str(sstatus).startswith("fetched"):
            extra = f" ({nfound} nums)"
            if ctype:
                extra += f" • {ctype}"
            st.success(f"✅ {short}{extra}{flag_txt}")
        else:
            extra = f" - {detail}" if detail else ""
            if ctype:
                extra += f" • {ctype}"
            st.error(f"❌ {short}{extra}{flag_txt}")

    st.markdown("---")

    # -------------------------
    # Metric changes table
    # -------------------------
    st.subheader("💰 Metric Changes")
    rows = results.get("metric_changes") or []
    if not isinstance(rows, list) or not rows:
        st.info("No metric changes to display.")
        return

    table_rows = []
    for r in rows:
        if not isinstance(r, dict):
            continue

        metric_label = r.get("metric") or r.get("name") or ""
        status_label = r.get("status") or r.get("change_type") or ""

        table_rows.append({
            "Metric": metric_label,
            "Canonical Key": r.get("canonical_key", "") or "",
            "Match Stage": r.get("match_stage", "") or "",
            "Previous": r.get("previous_value", "") or "",
            "Current": r.get("current_value", "") or "",
            "Δ%": _fmt_change_pct(r.get("change_pct")),
            "Status": status_label,
            "Match": _fmt_pct(r.get("match_confidence")),
            "Score": ("" if r.get("match_score") is None else f"{_safe_float(r.get('match_score'), 0.0):.2f}"),
            "Anchor": "✅" if r.get("anchor_used") else "",
        })

    st.dataframe(table_rows, use_container_width=True)

    # -------------------------
    # Debug / tuning views
    # -------------------------
    # Aggregate rejection reasons across all metrics (quick tuning signal)
    agg_rej = Counter()
    for r in rows:
        if isinstance(r, dict) and isinstance(r.get("rejected_reason_counts"), dict):
            for k, v in r["rejected_reason_counts"].items():
                try:
                    agg_rej[k] += int(v or 0)
                except Exception:
                    pass

    if agg_rej:
        with st.expander("🧰 Tuning Summary (aggregate rejects across all metrics)"):
            st.write(dict(agg_rej.most_common(20)))

    # Full per-metric debug
    with st.expander("🧾 Per-metric match details (debug)"):
        for i, r in enumerate(rows, 1):
            if not isinstance(r, dict):
                continue

            metric_label = r.get("metric") or r.get("name") or f"metric_{i}"
            status_label = r.get("status") or r.get("change_type") or "unknown"

            canonical_key = r.get("canonical_key", "") or ""
            stage = r.get("match_stage", "") or ""
            conf = r.get("match_confidence", None)
            score = r.get("match_score", None)

            header = f"{i}. {metric_label} — {status_label}"
            meta_bits = []
            if canonical_key:
                meta_bits.append(f"ck={canonical_key}")
            if stage:
                meta_bits.append(f"stage={stage}")
            if conf is not None:
                meta_bits.append(f"conf={_fmt_pct(conf)}")
            if score is not None:
                try:
                    meta_bits.append(f"score={float(score):.2f}")
                except Exception:
                    meta_bits.append(f"score={score}")

            if meta_bits:
                header += f"  ({' • '.join(meta_bits)})"

            with st.expander(header):
                # Values
                st.write({
                    "previous_value": r.get("previous_value"),
                    "current_value": r.get("current_value"),
                    "change_pct": r.get("change_pct"),
                })

                # Candidate considered / rejects
                st.write("Candidates considered:", _safe_int(r.get("candidates_considered_count"), 0))

                rej = r.get("rejected_reason_counts")
                if isinstance(rej, dict) and rej:
                    # sort largest first
                    try:
                        rej_sorted = dict(sorted(((k, int(v or 0)) for k, v in rej.items()), key=lambda x: x[1], reverse=True))
                    except Exception:
                        rej_sorted = rej
                    st.write("Rejected reason counts:", rej_sorted)

                # Score breakdown (if present)
                sb = r.get("score_breakdown")
                if isinstance(sb, dict) and sb:
                    st.write("Score breakdown:", sb)

                # Matched candidate (new)
                mc = r.get("matched_candidate")
                if isinstance(mc, dict) and mc:
                    st.markdown("**Matched candidate**")
                    st.write({
                        "raw": mc.get("raw"),
                        "value": mc.get("value"),
                        "unit": mc.get("unit"),
                        "source_url": mc.get("source_url"),
                        "anchor_hash": mc.get("anchor_hash"),
                        "is_homepage": mc.get("is_homepage"),
                        "skip_reason": mc.get("skip_reason"),
                        "quality_score": mc.get("quality_score"),
                    })
                    ctx = mc.get("context_snippet")
                    if ctx:
                        st.write("Context:")
                        st.code(str(ctx))
                else:
                    # Backward-compatible fields
                    src = r.get("matched_source") or r.get("source_url")
                    ctx = r.get("matched_context") or r.get("context_snippet")
                    if src:
                        st.write("Source:", src)
                    if ctx:
                        st.write("Context:")
                        st.code(str(ctx))

                # Additional anchor hash compatibility
                if r.get("matched_anchor_hash"):
                    st.write("Matched Anchor Hash:", r.get("matched_anchor_hash"))

    st.markdown("---")


# =========================================================
# 9. DASHBOARD RENDERING
# =========================================================

def detect_x_label_dynamic(labels: list) -> str:
    """Enhanced X-axis detection with better region matching"""
    if not labels:
        return "Category"

    # Convert to lowercase for comparison
    label_texts = [str(l).lower().strip() for l in labels]
    all_text = ' '.join(label_texts)

    # 1. GEOGRAPHIC REGIONS (PRIORITY 1)
    region_keywords = [
        'north america', 'asia pacific', 'asia-pacific', 'apac', 'europe', 'emea',
        'latin america', 'latam', 'middle east', 'africa', 'oceania',
        'rest of world', 'row', 'china', 'usa', 'india', 'japan', 'germany'
    ]

    # Count how many labels contain region keywords
    region_matches = sum(
        1 for label in label_texts
        if any(keyword in label for keyword in region_keywords)
    )

    # If 40%+ of labels are regions → "Regions"
    if region_matches / len(labels) >= 0.4:
        return "Regions"

    # 2. YEARS (e.g., 2023, 2024, 2025)
    year_pattern = r'\b(19|20)\d{2}\b'
    year_count = sum(1 for label in label_texts if re.search(year_pattern, label))
    if year_count / len(labels) > 0.5:
        return "Years"

    # 3. QUARTERS (Q1, Q2, Q3, Q4)
    quarter_pattern = r'\bq[1-4]\b'
    quarter_count = sum(1 for label in label_texts if re.search(quarter_pattern, label, re.IGNORECASE))
    if quarter_count >= 2:
        return "Quarters"

    # 4. MONTHS
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_count = sum(1 for label in label_texts if any(month in label for month in months))
    if month_count >= 3:
        return "Months"

    # 5. COMPANIES (common suffixes)
    company_keywords = ['inc', 'corp', 'ltd', 'llc', 'gmbh', 'ag', 'sa', 'plc']
    company_count = sum(1 for label in label_texts if any(kw in label for kw in company_keywords))
    if company_count >= 2:
        return "Companies"

    # 6. PRODUCTS/SEGMENTS (if contains "segment", "product", "category")
    if any(word in all_text for word in ['segment', 'product line', 'category', 'type']):
        return "Segments"

    # Default
    return "Categories"

def detect_y_label_dynamic(values: list) -> str:
    """Fully dynamic Y-axis label based on magnitude + context"""
    if not values:
        return "Value"

    numeric_values = []
    for v in values:
        try:
            numeric_values.append(abs(float(v)))
        except (ValueError, TypeError):
            continue

    if not numeric_values:
        return "Value"

    avg_mag = np.mean(numeric_values)
    max_mag = max(numeric_values)

    # Non-overlapping ranges with clear boundaries
    # 1. BILLIONS (large market sizes)
    if max_mag > 100 or avg_mag > 50:
        return "USD B"

    # 2. MILLIONS (medium values)
    elif max_mag > 10 or avg_mag > 5:
        return "USD M"

    # 3. PERCENTAGES (typical 0-100 range, but also small decimals)
    elif max_mag <= 100 and avg_mag <= 50:
        # Check if values look like percentages (mostly 0-100)
        if all(0 <= v <= 100 for v in numeric_values):
            return "Percent %"
        else:
            return "USD K"

    # 4. Default
    else:
        return "Units"

# =========================================================
# 3A. QUESTION CATEGORIZATION + SIGNALS (DETERMINISTIC)
# =========================================================

def categorize_question_signals(query: str, qs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a question_profile used for structured reporting.

    IMPORTANT:
      - category must follow query_structure if provided (single source of truth).
      - signals can be rich, but must not contradict the chosen category.
    """
    qs = qs or {}
    q = (query or "").strip()

    # Prefer category/main/side from query_structure when available
    category = (qs.get("category") or "").strip() or "unknown"
    main_q = (qs.get("main") or "").strip() or q
    side_qs = qs.get("side") if isinstance(qs.get("side"), list) else []

    # Deterministic signals (richer classifier)
    base = classify_question_signals(q) or {}

    # Force category + expected_metric_ids to match query_structure category
    # (but preserve other extracted info like years/regions/intents)
    signals: Dict[str, Any] = {}
    signals["category"] = category

    # Carry over extracted fields
    signals["years"] = base.get("years", []) or []
    signals["regions"] = base.get("regions", []) or []
    signals["intents"] = base.get("intents", []) or []

    # Keep raw signals for debugging
    raw_hits = list(base.get("signals") or [])
    signals["raw_signals"] = raw_hits

    def _signal_consistent_with_category(sig: str, cat: str) -> bool:
        s = (sig or "").lower()
        c = (cat or "").lower()
        if not s:
            return False

        # If final category is country, drop industry/company category-rule strings
        if c == "country":
            if "industry_keywords" in s or "mixed_signals_default_to_industry" in s or "company_keywords" in s:
                return False

        # If final category is industry, drop explicit country-rule strings
        if c == "industry":
            if "macro_outlook_bias_country" in s or "country_keywords" in s:
                return False

        return True

    signals["signals"] = [s for s in raw_hits if _signal_consistent_with_category(s, category)]

    # Expected metric IDs: always determined by the final category, then lightly enriched by intents (optional)
    expected_metric_ids: List[str] = []
    try:
        expected_metric_ids = get_expected_metric_ids_for_category(category) or []
    except Exception:
        expected_metric_ids = []

    # Optional: enrich with intent-based suggestions (won't remove anything)
    intent_metric_suggestions = {
        "market_size": ["market_size", "market_size_2024", "market_size_2025"],
        "growth_forecast": ["cagr", "market_size_2030"],
        "competitive_landscape": ["market_share", "top_players"],
        "pricing": ["avg_price", "asp"],
        "consumer_demand": ["users", "penetration", "arpu"],
        "supply_chain": ["capacity", "shipments"],
        "investment": ["capex", "profit", "ebitda"],
        "macro_outlook": ["gdp", "inflation", "interest_rate", "exchange_rate"],
    }

    intents = signals.get("intents") or []
    for intent in intents:
        for mid in intent_metric_suggestions.get(intent, []):
            if mid not in expected_metric_ids:
                expected_metric_ids.append(mid)

    signals["expected_metric_ids"] = expected_metric_ids

    profile: Dict[str, Any] = {
        "category": category,
        "signals": signals,
        "main_question": main_q,
        "side_questions": side_qs,
    }

    # Keep debug for traceability
    if qs.get("debug") is not None:
        profile["debug_query_structure"] = qs.get("debug")

    return profile




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

    # -------------------------
    # Parse primary response
    # -------------------------

    # =========================
    # PATCH RD1 (ADDITIVE): safe preview helper
    # - Prevents slice errors when primary_json is dict/list/etc.
    # - Keeps original behavior for strings
    # =========================
    def _preview(x, limit: int = 1000) -> str:
        try:
            if isinstance(x, (dict, list)):
                s = json.dumps(x, ensure_ascii=False, indent=2, default=str)
            else:
                s = str(x)
        except Exception:
            s = repr(x)
        return s[:limit]
    # =========================

    try:
        # =========================
        # PATCH RD2 (ADDITIVE): accept dict/list directly
        # - If caller passes dict (primary_data), just use it
        # - If caller passes list, wrap it (keeps downstream dict access safe)
        # - Else try json.loads on string
        # =========================
        if isinstance(primary_json, dict):
            data = primary_json
        elif isinstance(primary_json, list):
            data = {"_list": primary_json}
        else:
            data = json.loads(primary_json)
        # =========================

    except Exception as e:
        st.error(f"❌ Cannot render dashboard: {e}")
        # =========================
        # PATCH RD1 (ADDITIVE): safe preview (no slicing crash)
        # =========================
        st.code(_preview(primary_json))
        # =========================
        return

    # -------------------------
    # Helper: metric value formatting (currency + compact units) + RANGE SUPPORT
    # -------------------------
    def _format_metric_value(m: Any) -> str:
        """
        Format metric values cleanly, with RANGE SUPPORT:
        - If value_range exists (min/max), show min–max using the same currency/unit rules
        - Otherwise show the point value as before
        """
        if not isinstance(m, dict):
            if m is None:
                return "N/A"
            return str(m)

        # -------------------------
        # Helper: format a single numeric endpoint (val+unit)
        # -------------------------
        def _format_point(val: Any, unit: str) -> str:
            if val is None or val == "":
                return "N/A"

            unit = (unit or "").strip()
            raw_val = str(val).strip()

            # Try parse numeric
            try:
                num = float(raw_val.replace(",", ""))
            except Exception:
                # If we can't parse as float, just glue value+unit neatly
                return f"{raw_val}{unit}".strip() if unit else raw_val

            # Normalize unit spacing
            unit = unit.replace(" ", "")
            currency_prefix = ""
            u_upper = unit.upper()

            # Common patterns: "S$B", "SGDB", "USD B", "$B"
            if u_upper.startswith("S$"):
                currency_prefix = "S$"
                unit = unit[2:]
            elif u_upper.startswith("SGD"):
                currency_prefix = "S$"
                unit = unit[3:]
            elif u_upper.startswith("USD"):
                currency_prefix = "$"
                unit = unit[3:]
            elif u_upper.startswith("$"):
                currency_prefix = "$"
                unit = unit[1:]

            unit = unit.strip()

            # Percent
            if unit == "%":
                return f"{num:.1f}%"

            # Compact units
            unit_upper = unit.upper()
            if unit_upper in ("B", "BILLION"):
                formatted = f"{num:.2f}".rstrip("0").rstrip(".") + "B"
                return f"{currency_prefix}{formatted}".strip()
            if unit_upper in ("M", "MILLION"):
                formatted = f"{num:.2f}".rstrip("0").rstrip(".") + "M"
                return f"{currency_prefix}{formatted}".strip()
            if unit_upper in ("K", "THOUSAND"):
                formatted = f"{num:.2f}".rstrip("0").rstrip(".") + "K"
                return f"{currency_prefix}{formatted}".strip()

            # Plain number formatting
            if abs(num) >= 1000:
                if float(num).is_integer():
                    formatted = f"{int(num):,}"
                else:
                    formatted = f"{num:,.2f}".rstrip("0").rstrip(".")
            else:
                formatted = f"{num:g}"

            # Unit glue
            if unit:
                formatted = f"{formatted} {unit}".strip()

            return f"{currency_prefix}{formatted}".strip()

        # -------------------------
        # RANGE: prefer value_range if present and meaningful
        # -------------------------
        unit = (m.get("unit") or "").strip()
        vr = m.get("value_range")

        if isinstance(vr, dict):
            vmin = vr.get("min")
            vmax = vr.get("max")
            if vmin is not None and vmax is not None:
                left = _format_point(vmin, unit)
                right = _format_point(vmax, unit)
                if left != "N/A" and right != "N/A" and left != right:
                    return f"{left}–{right}"

        # Precomputed range display (optional)
        vr_disp = m.get("value_range_display")
        if isinstance(vr_disp, str) and vr_disp.strip():
            return vr_disp.strip()

        # -------------------------
        # POINT VALUE fallback
        # -------------------------
        val = m.get("value")
        if val is None or val == "":
            return "N/A"

        return _format_point(val, unit)

    # -------------------------
    # Header + confidence row
    # -------------------------
    st.header("📊 Yureeka Market Report")
    st.markdown(f"**Question:** {user_question}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Confidence", f"{float(final_conf):.1f}%")
    col2.metric("Base Model", f"{float(base_conf):.1f}%")
    if isinstance(veracity_scores, dict):
        col3.metric("Evidence", f"{float(veracity_scores.get('overall', 0) or 0):.1f}%")
    else:
        col3.metric("Evidence", "N/A")

    st.markdown("---")

    # -------------------------
    # Executive Summary
    # -------------------------
    st.subheader("📋 Executive Summary")
    st.markdown(f"**{data.get('executive_summary', 'No summary available')}**")

    # Optional: expand summary if side-questions exist
    side_questions = data.get("side_questions") or (data.get("question_profile", {}) or {}).get("side_questions", [])
    if side_questions:
        st.markdown("")
        st.markdown("**Also addressed:**")
        for sq in side_questions[:6]:
            if sq:
                st.markdown(f"- {sq}")

    st.markdown("---")

    # -------------------------
    # Key Metrics
    # -------------------------
    st.subheader("💰 Key Metrics")
    metrics = data.get("primary_metrics", {}) or {}

    question_category = data.get("question_category") or (data.get("question_profile", {}) or {}).get("category")
    question_signals = data.get("question_signals") or (data.get("question_profile", {}) or {}).get("signals", {})
    expected_ids = data.get("expected_metric_ids") or ((data.get("question_signals") or {}).get("expected_metric_ids") or [])

    metric_rows: List[Dict[str, str]] = []

    if question_category:
        metric_rows.append({"Metric": "Question Category", "Value": str(question_category)})
    if isinstance(question_signals, dict) and question_signals:
        metric_rows.append({"Metric": "Signals", "Value": ", ".join([str(x) for x in (question_signals.get("signals") or [])][:10])})
    if expected_ids:
        metric_rows.append({"Metric": "Expected Metrics", "Value": ", ".join([str(x) for x in expected_ids][:10])})

    # Render primary metrics
    if isinstance(metrics, dict) and metrics:
        for _, m in metrics.items():
            if isinstance(m, dict):
                name = m.get("name") or "Metric"
                metric_rows.append({"Metric": str(name), "Value": _format_metric_value(m)})

    # Display metrics table
    if metric_rows:
        try:
            import pandas as pd  # optional dependency in your environment
            df_metrics = pd.DataFrame(metric_rows)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        except Exception:
            for r in metric_rows:
                st.write(f"**{r.get('Metric','')}**: {r.get('Value','')}")

    st.markdown("---")

    # -------------------------
    # Key Findings
    # -------------------------
    st.subheader("🧠 Key Findings")
    kf = data.get("key_findings") or []
    if isinstance(kf, list) and kf:
        for item in kf[:12]:
            if item:
                st.markdown(f"- {item}")
    else:
        st.info("No key findings available.")

    st.markdown("---")

    # -------------------------
    # Trends / Forecast
    # -------------------------
    st.subheader("📈 Trends & Forecast")
    tf = data.get("trends_forecast") or []
    if isinstance(tf, list) and tf:
        for t in tf[:12]:
            if isinstance(t, dict):
                trend = t.get("trend") or ""
                direction = t.get("direction") or ""
                timeline = t.get("timeline") or ""
                st.markdown(f"- **{trend}** {direction} ({timeline})")
            elif t:
                st.markdown(f"- {t}")
    else:
        st.info("No trends forecast available.")

    st.markdown("---")

    # -------------------------
    # Sources / Web Context summary
    # -------------------------
    st.subheader("🔎 Sources & Evidence")
    sources = data.get("sources") or data.get("web_sources") or []
    if isinstance(sources, list) and sources:
        with st.expander(f"Show sources ({len(sources)})"):
            for s in sources[:50]:
                if s:
                    st.markdown(f"- {s}")
            if len(sources) > 50:
                st.markdown(f"... (+{len(sources)-50} more)")

    # Web context debug counters if present
    if isinstance(web_context, dict):
        dbg = web_context.get("debug_counts") or {}
        if isinstance(dbg, dict) and dbg:
            with st.expander("Collector debug counts"):
                st.json(dbg)

    # Source reliability badges (if provided)
    if isinstance(source_reliability, list) and source_reliability:
        with st.expander("Source reliability"):
            for line in source_reliability[:80]:
                st.write(line)



def render_native_comparison(baseline: Dict, compare: Dict):
    """Render a clean comparison between two analyses"""

    st.header("📊 Analysis Comparison")

    # Time info
    baseline_time = baseline.get('timestamp', '')
    compare_time = compare.get('timestamp', '')

    try:
        baseline_dt = datetime.fromisoformat(baseline_time.replace('Z', '+00:00'))
        compare_dt = datetime.fromisoformat(compare_time.replace('Z', '+00:00'))
        delta = compare_dt - baseline_dt
        if delta.days > 0:
            delta_str = f"{delta.days}d {delta.seconds // 3600}h"
        else:
            delta_str = f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
    except:
        delta_str = "Unknown"

    # Overview row
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline", baseline_time[:16] if baseline_time else "N/A")
    col2.metric("Current", compare_time[:16] if compare_time else "N/A")
    col3.metric("Time Delta", delta_str)

    st.markdown("---")

    # Extract metrics
    baseline_metrics = baseline.get('primary_response', {}).get('primary_metrics', {})
    compare_metrics = compare.get('primary_response', {}).get('primary_metrics', {})

    # Build metric diff table
    st.subheader("💰 Metric Changes")

    diff_rows = []
    stability_count = 0
    total_count = 0

    # Canonicalize metrics for stable matching
    baseline_canonical = canonicalize_metrics(baseline_metrics)
    compare_canonical = canonicalize_metrics(compare_metrics)

    # Build lookup by canonical ID
    baseline_by_id = {}
    compare_by_id = {}

    for cid, m in baseline_canonical.items():
        baseline_by_id[cid] = m

    for cid, m in compare_canonical.items():
        compare_by_id[cid] = m

    all_ids = set(baseline_by_id.keys()).intersection(compare_by_id.keys())

    for cid in sorted(all_ids):
        baseline_m = baseline_by_id.get(cid)
        compare_m = compare_by_id.get(cid)

        # Use canonical name for display, fallback to original
        display_name = cid
        if baseline_m and baseline_m.get('name'):
            display_name = baseline_m['name']


        if baseline_m and compare_m:
            old_val = baseline_m.get('value', 'N/A')
            new_val = compare_m.get('value', 'N/A')
            unit = compare_m.get('unit', baseline_m.get('unit', ''))

            old_num = parse_to_float(old_val)
            new_num = parse_to_float(new_val)

            if old_num is not None and new_num is not None and old_num != 0:
                change_pct = ((new_num - old_num) / abs(old_num)) * 100

                if abs(change_pct) < 1:
                    icon, reason = "➡️", "No change"
                    stability_count += 1
                elif abs(change_pct) < 5:
                    icon, reason = "➡️", "Minor change"
                    stability_count += 1
                elif change_pct > 0:
                    icon, reason = "📈", "Increased"
                else:
                    icon, reason = "📉", "Decreased"

                delta_str = f"{change_pct:+.1f}%"
            else:
                icon, delta_str, reason = "➡️", "-", "Non-numeric"
                stability_count += 1

            diff_rows.append({
                '': icon,
                'Metric': display_name,
                'Old': _fmt_currency_first(str(old_val), str(unit)),
                'New': _fmt_currency_first(str(new_val), str(unit)),
                'Δ': delta_str,
                'Reason': reason
            })
            total_count += 1

        elif baseline_m:
            old_val = baseline_m.get('value', 'N/A')
            unit = baseline_m.get('unit', '')
            diff_rows.append({
                '': '❌',
                'Metric': display_name,
                'Old': f"{old_val} {unit}".strip(),
                'New': '-',
                'Δ': '-',
                'Reason': 'Removed'
            })
            total_count += 1
        else:
            new_val = compare_m.get('value', 'N/A')
            unit = compare_m.get('unit', '')
            diff_rows.append({
                '': '🆕',
                'Metric': display_name,
                'Old': '-',
                'New': f"{new_val} {unit}".strip(),
                'Δ': '-',
                'Reason': 'New'
            })
            total_count += 1

    if diff_rows:
        st.dataframe(pd.DataFrame(diff_rows), hide_index=True, use_container_width=True)

        # Show canonical ID mapping for debugging
        with st.expander("🔧 Canonical ID Mapping (Debug)"):
            st.write("**How metrics were matched:**")

            baseline_canonical = canonicalize_metrics(baseline_metrics)
            compare_canonical = canonicalize_metrics(compare_metrics)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Baseline Metrics:**")
                for cid, m in baseline_canonical.items():
                    original = m.get('original_name', 'N/A')
                    canonical = m.get('name', 'N/A')
                    st.caption(f"`{cid}`")
                    st.write(f"  {original} → {canonical}")

            with col2:
                st.write("**Current Metrics:**")
                for cid, m in compare_canonical.items():
                    original = m.get('original_name', 'N/A')
                    canonical = m.get('name', 'N/A')
                    st.caption(f"`{cid}`")
                    st.write(f"  {original} → {canonical}")
    else:
        st.info("No metrics to compare")

    # Stability score
    stability_pct = (stability_count / total_count * 100) if total_count > 0 else 100

    st.markdown("---")
    st.subheader("📊 Stability Score")

    col1, col2, col3 = st.columns(3)
    col1.metric("Stable Metrics", f"{stability_count}/{total_count}")
    col2.metric("Stability", f"{stability_pct:.0f}%")

    if stability_pct >= 80:
        col3.success("🟢 Highly Stable")
    elif stability_pct >= 60:
        col3.warning("🟡 Moderate Changes")
    else:
        col3.error("🔴 Significant Drift")

    # Confidence comparison
    st.markdown("---")
    st.subheader("🎯 Confidence Change")

    col1, col2, col3 = st.columns(3)
    baseline_conf = baseline.get('final_confidence', 0)
    compare_conf = compare.get('final_confidence', 0)
    conf_change = compare_conf - baseline_conf if isinstance(baseline_conf, (int, float)) and isinstance(compare_conf, (int, float)) else 0

    col1.metric("Baseline", f"{baseline_conf:.1f}%" if isinstance(baseline_conf, (int, float)) else "N/A")
    col2.metric("Current", f"{compare_conf:.1f}%" if isinstance(compare_conf, (int, float)) else "N/A")
    col3.metric("Change", f"{conf_change:+.1f}%")

    # Download comparison
    st.markdown("---")
    comparison_output = {
        "comparison_timestamp": datetime.now().isoformat(),
        "baseline": baseline,
        "current": compare,
        "stability_score": stability_pct,
        "metrics_compared": total_count,
        "metrics_stable": stability_count
    }

    st.download_button(
        label="💾 Download Comparison Report",
        data=json.dumps(comparison_output, indent=2, ensure_ascii=False).encode('utf-8'),
        file_name=f"yureeka_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# =========================================================
# 10. MAIN APPLICATION
# =========================================================

def main():
    st.set_page_config(
        page_title="Yureeka Market Report",
        page_icon="💹",
        layout="wide"
    )

    st.title("💹 Yureeka Market Intelligence")

    # Info section
    col_info, col_status = st.columns([3, 1])
    with col_info:
        st.markdown("""
        **Yureeka** provides AI-powered market research and analysis for finance,
        economics, and business questions.
        Powered by evidence-based verification and real-time web search.

        *Currently in prototype stage.*
        """)

    # Create tabs
    tab1, tab2 = st.tabs(["🔍 New Analysis", "📈 Evolution Analysis"])

    # =====================
    # TAB 1: NEW ANALYSIS
    # =====================
    with tab1:
        query = st.text_input(
            "Enter your question about markets, industries, finance, or economics:",
            placeholder="e.g., What is the size of the global EV battery market?"
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_web = st.checkbox(
                "Enable web search (recommended)",
                value=bool(SERPAPI_KEY),
                disabled=not SERPAPI_KEY
            )

        if st.button("🔍 Analyze", type="primary") and query:
            if len(query.strip()) < 5:
                st.error("❌ Please enter a question with at least 5 characters")
                return

            query = query.strip()[:500]

            query_structure = extract_query_structure(query) or {}
            question_profile = categorize_question_signals(query, qs=query_structure)
            question_signals = question_profile.get("signals", {}) or {}

            web_context = {}
            if use_web:
                with st.spinner("🌐 Searching the web..."):

                    # ---- ADDITIVE: pass existing snapshots for reuse (Change #3 wiring) ----
                    existing_snapshots = None

                    # If you have an analysis dict already in scope, reuse its cache
                    try:
                        if isinstance(locals().get("analysis"), dict):
                            existing_snapshots = (
                                analysis.get("baseline_sources_cache")
                                or (analysis.get("results", {}) or {}).get("baseline_sources_cache")
                                or (analysis.get("results", {}) or {}).get("source_results")
                            )
                    except Exception:
                        existing_snapshots = None

                    # Optional: if you keep a prior analysis in session_state, reuse it
                    try:
                        prev = st.session_state.get("last_analysis")
                        if existing_snapshots is None and isinstance(prev, dict):
                            existing_snapshots = (
                                prev.get("baseline_sources_cache")
                                or (prev.get("results", {}) or {}).get("baseline_sources_cache")
                                or (prev.get("results", {}) or {}).get("source_results")
                            )
                    except Exception:
                        pass

                    web_context = fetch_web_context(
                        query,
                        num_sources=3,
                        existing_snapshots=existing_snapshots,
                    )
                    # ----------------------------------------------------------------------

            if not web_context or not web_context.get("search_results"):
                st.info("💡 Using AI knowledge without web search")
                web_context = {
                    "search_results": [],
                    "scraped_content": {},
                    "summary": "",
                    "sources": [],
                    "source_reliability": []
                }

            with st.spinner("🤖 Analyzing query..."):
                primary_response = query_perplexity(query, web_context, query_structure=query_structure)

            if not primary_response:
                st.error("❌ Primary model failed to respond")
                return

            try:
                primary_data = json.loads(primary_response)
            except Exception as e:
                st.error(f"❌ Failed to parse primary response: {e}")
                st.code(primary_response[:1000])
                return

            with st.spinner("✅ Verifying evidence quality..."):
                veracity_scores = evidence_based_veracity(primary_data, web_context)

            base_conf = float(primary_data.get("confidence", 75))
            final_conf = calculate_final_confidence(base_conf, veracity_scores.get("overall", 0))

            # Optional: canonicalize + attribution + schema freeze (only if your codebase defines these)
            # Optional: canonicalize + attribution + schema freeze (only if your codebase defines these)
            try:
                # 1) canonicalize (unchanged)
                if primary_data.get("primary_metrics"):
                    primary_data["primary_metrics_canonical"] = canonicalize_metrics(
                        primary_data.get("primary_metrics", {}),
                        merge_duplicates_to_range=True,
                        question_text=query,
                        category_hint=str(primary_data.get("question_category", ""))
                    )

                # 2) freeze schema FIRST ✅ (so attribution can be schema-first)
                if primary_data.get("primary_metrics_canonical"):
                    primary_data["metric_schema_frozen"] = freeze_metric_schema(
                        primary_data["primary_metrics_canonical"]
                    )

                # 3) attribution using frozen schema ✅
                if primary_data.get("primary_metrics_canonical"):
                    primary_data["primary_metrics_canonical"] = add_range_and_source_attribution_to_canonical_metrics(
                        primary_data.get("primary_metrics_canonical", {}),
                        web_context,
                        metric_schema=(primary_data.get("metric_schema_frozen") or {}),
                    )

                # PATCH SV1/EG1 (ADDITIVE): validate frozen schema + enforce evidence gating (analysis-side)
                try:
                    fn = globals().get("apply_schema_validation_and_evidence_gating")
                    if callable(fn):
                        primary_data = fn(primary_data)
                except Exception:
                    pass

            except Exception:
                pass

            # Hash key findings (optional)
            try:
                if primary_data.get("key_findings"):
                    findings_with_hash = []
                    for finding in primary_data.get("key_findings", []):
                        if finding:
                            findings_with_hash.append({
                                "text": finding,
                                "semantic_hash": compute_semantic_hash(finding)
                            })
                    primary_data["key_findings_hashed"] = findings_with_hash
            except Exception:
                pass


            # Save baseline numeric cache if available (existing behavior)

            # Build output
            output = {
                "question": query,
                "question_profile": question_profile,
                "question_category": question_profile.get("category"),
                "question_signals": question_signals,
                "side_questions": question_profile.get("side_questions", []),
                "timestamp": now_utc().isoformat(),
                "primary_response": primary_data,
                "final_confidence": final_conf,
                "veracity_scores": veracity_scores,
                "web_sources": web_context.get("sources", []),
                "code_version": CODE_VERSION,
                }

            try:
                if isinstance(output.get("primary_response"), dict):
                    output["primary_response"]["code_version"] = CODE_VERSION
            except Exception:
                pass


            # ✅ NEW: attach analysis-aligned snapshots (from scraped_meta)
            # This is the stable cache evolution should reuse.
            try:
                output = attach_source_snapshots_to_analysis(output, web_context)
            except Exception:
                pass

            with st.spinner("💾 Saving to history..."):
                if add_to_history(output):
                    st.success("✅ Analysis saved to Google Sheets")
                else:
                    st.warning("⚠️ Saved to session only (Google Sheets unavailable)")

            json_bytes = json.dumps(output, indent=2, ensure_ascii=False).encode("utf-8")
            filename = f"yureeka_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            st.download_button(
                label="💾 Download Analysis JSON",
                data=json_bytes,
                file_name=filename,
                mime="application/json"
            )

            render_dashboard(
            primary_data,
            final_conf,
            web_context,
            base_conf,
            query,
            veracity_scores,
            web_context.get("source_reliability", [])
            )


            with st.expander("🔧 Debug Information"):
                st.write("**Confidence Breakdown:**")
                st.json({
                    "base_confidence": base_conf,
                    "evidence_score": veracity_scores.get("overall", 0),
                    "final_confidence": final_conf,
                    "veracity_breakdown": veracity_scores
                })
                st.write("**Primary Model Response:**")
                st.json(primary_data)

    # =====================
    # TAB 2: EVOLUTION ANALYSIS
    # =====================
    with tab2:
        st.markdown("""
        ### 📈 Track the evolution of key metrics over time using **deterministic source-anchored analysis**.

        **How it works:**
        - Select a baseline from your history (stored in Google Sheets)
        - Re-fetches the **exact same sources** from that analysis
        - Extracts current numbers using regex (no LLM variance)
        - Computes deterministic diffs with context-aware matching
        """)

        with st.sidebar:
            st.subheader("📚 History")

            if st.button("🔄 Refresh"):
                st.cache_resource.clear()
                st.rerun()

            sheet = get_google_sheet()
            if sheet:
                st.success("✅ Google Sheets connected")
            else:
                st.warning("⚠️ Using session storage")

        # ✅ FIX: your codebase uses get_history(), not load_history()
        history = get_history()

        if not history:
            st.info("📭 No previous analyses found. Run an analysis in the 'New Analysis' tab first.")
            return

        baseline_options = [
            f"{i+1}. {h.get('question', 'N/A')}  ({h.get('timestamp', '')})"
            for i, h in enumerate(history)
        ]
        baseline_choice = st.selectbox("Select baseline analysis:", baseline_options)
        baseline_idx = int(baseline_choice.split(".")[0]) - 1
        baseline_data = history[baseline_idx]

        compare_method = st.selectbox(
            "Comparison method:",
            [
                "source-anchored evolution (re-fetch same sources)",
                "another saved analysis (deterministic)",
                "fresh analysis (volatile)"
            ]
        )

        compare_data = None
        if "another saved analysis" in compare_method:
            compare_options = [
                f"{i+1}. {h.get('question', 'N/A')}  ({h.get('timestamp', '')})"
                for i, h in enumerate(history) if i != baseline_idx
            ]
            if compare_options:
                compare_choice = st.selectbox("Select comparison analysis:", compare_options)
                compare_idx = int(compare_choice.split(".")[0]) - 1
                compare_data = history[compare_idx]
            else:
                st.warning("No other saved analyses to compare with.")

        st.markdown("---")

        if st.button("🧬 Run Evolution Analysis", type="primary"):

            if "source-anchored evolution" in compare_method:
                evolution_query = baseline_data.get("question", "")
                if not evolution_query:
                    st.error("❌ No question found in baseline.")
                    return

                with st.spinner("🧬 Running source-anchored evolution..."):
                    try:
                        results = run_source_anchored_evolution(baseline_data)
                    except Exception as e:
                        st.error(f"❌ Evolution failed: {e}")
                        return

                interpretation = ""
                try:
                    if results and isinstance(results, dict):
                        interpretation = results.get("interpretation", "") or ""
                except Exception:
                    interpretation = ""

                evolution_output = {
                    "question": evolution_query,
                    "timestamp": datetime.now().isoformat(),
                    "analysis_type": "source_anchored",
                    "previous_timestamp": baseline_data.get("timestamp"),
                    "results": results,
                    "interpretation": {
                        "text": interpretation,
                        "authoritative": False,
                        "source": "llm_optional"
                    }
                }

                st.download_button(
                    label="💾 Download Evolution Report",
                    data=json.dumps(evolution_output, indent=2, ensure_ascii=False).encode("utf-8"),
                    file_name=f"yureeka_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

                # ✅ FIX: guarded renderer to avoid stability_score=None formatting crashes
                render_source_anchored_results(results, evolution_query)

            elif "another saved analysis" in compare_method:
                if compare_data:
                    st.success("✅ Comparing two saved analyses (deterministic)")
                    render_native_comparison(baseline_data, compare_data)
                else:
                    st.error("❌ Please select a comparison analysis")

            else:
                st.warning("⚠️ Running fresh analysis - results may vary")

                query = baseline_data.get("question", "")
                if not query:
                    st.error("❌ No query found")
                    return

                with st.spinner("🌐 Fetching current data..."):
                    # ---- ADDITIVE: pass existing snapshots for reuse (Change #3 wiring) ----
                    existing_snapshots = None

                    try:
                        prev = st.session_state.get("last_analysis")
                        if isinstance(prev, dict):
                            existing_snapshots = (
                                prev.get("baseline_sources_cache")
                                or (prev.get("results", {}) or {}).get("baseline_sources_cache")
                                or (prev.get("results", {}) or {}).get("source_results")
                            )
                    except Exception:
                        existing_snapshots = None

                    web_context = fetch_web_context(
                        query,
                        num_sources=3,
                        existing_snapshots=existing_snapshots,
                    )
                    # ----------------------------------------------------------------------


                if not web_context:
                    web_context = {
                        "search_results": [],
                        "scraped_content": {},
                        "summary": "",
                        "sources": [],
                        "source_reliability": []
                    }

                with st.spinner("🤖 Running analysis..."):
                    new_response = query_perplexity(query, web_context)

                if new_response:
                    try:
                        new_parsed = json.loads(new_response)
                        veracity = evidence_based_veracity(new_parsed, web_context)
                        base_conf = float(new_parsed.get("confidence", 75))
                        final_conf = calculate_final_confidence(base_conf, veracity.get("overall", 0))

                        compare_data = {
                            "question": query,
                            "timestamp": datetime.now().isoformat(),
                            "primary_response": new_parsed,
                            "final_confidence": final_conf,
                            "veracity_scores": veracity,
                            "web_sources": web_context.get("sources", [])
                        }

                        add_to_history(compare_data)
                        st.success("✅ Saved to history")

                        render_native_comparison(baseline_data, compare_data)
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
                else:
                    st.error("❌ Analysis failed")


# ======================================================================
# PATCH SV1/EG1 (ADDITIVE): Schema validation + Evidence gating (analysis)
# - Additive only: does not remove or refactor existing code.
# - Only applied in TAB 1 (New Analysis) via a small post-pass hook.
# - Does NOT alter evolution behavior (no changes to evolution functions).
# ======================================================================

def validate_metric_schema_frozen(metric_schema_frozen: dict) -> dict:
    """
    Validate frozen metric schema for internal consistency.
    Returns: {"ok": bool, "errors": [...], "warnings": [...], "by_key": {...}}
    """
    issues = {"ok": True, "errors": [], "warnings": [], "by_key": {}}

    def _add(kind: str, canonical_key: str, msg: str):
        issues["ok"] = issues["ok"] and (kind != "errors")
        issues[kind].append({"canonical_key": canonical_key, "message": msg})
        issues["by_key"].setdefault(canonical_key, {"errors": [], "warnings": []})
        issues["by_key"][canonical_key][kind].append(msg)

    if not isinstance(metric_schema_frozen, dict):
        _add("errors", "__schema__", "metric_schema_frozen missing or not a dict")
        return issues

    for canonical_key, spec in metric_schema_frozen.items():
        if not isinstance(spec, dict):
            _add("errors", canonical_key, "schema entry not a dict")
            continue

        dim = (spec.get("dimension") or spec.get("measure_kind") or "").lower().strip()
        unit = (spec.get("unit") or spec.get("unit_tag") or "").strip()
        unit_family = (spec.get("unit_family") or spec.get("unit_family_tag") or "").lower().strip()
        name = (spec.get("name") or canonical_key or "").lower()

        # Hard conflict: currency + percent
        if dim in ("currency", "revenue", "market_value", "value") and unit in ("%", "percent", "percentage"):
            _add("errors", canonical_key, "dimension=currency but unit is percent (%)")

        # Soft checks for percent metrics without percent unit
        if ("cagr" in name or dim in ("percent", "percentage", "growth_rate")) and unit and unit not in ("%", "percent", "percentage"):
            _add("warnings", canonical_key, f"percent-like metric but unit='{unit}' (expected '%')")

        # Common drift hazard: CAGR schema includes 'share'
        kw = " ".join([str(x) for x in (spec.get("keywords") or [])]).lower()
        if "cagr" in name and "share" in kw:
            _add("warnings", canonical_key, "CAGR schema keywords include 'share' (risk of mapping share% to CAGR)")

        # Unit family conflicts
        if dim == "currency" and unit_family and unit_family not in ("currency", "money"):
            _add("warnings", canonical_key, f"dimension=currency but unit_family='{unit_family}'")

    return issues


def _metric_evidence_list(metric: dict):
    ev = metric.get("evidence")
    if isinstance(ev, list):
        return ev
    return []


def _synthesize_evidence_from_examples(metric: dict, max_items: int = 5) -> list:
    """
    If metric has value_range.examples (from attribution pass), synthesize evidence records.
    This keeps JSON stable and makes evolution rebuild auditing possible.
    """
    examples = None
    vr = metric.get("value_range")
    if isinstance(vr, dict):
        examples = vr.get("examples")
    if not isinstance(examples, list) or not examples:
        return []

    # Try to use an existing anchor hash function if present
    anchor_fn = globals().get("compute_anchor_hash")
    out = []
    for ex in examples[:max_items]:
        if not isinstance(ex, dict):
            continue
        url = ex.get("source_url") or ex.get("url") or ""
        raw = ex.get("raw") or ""
        ctx = ex.get("context") or ex.get("context_window") or ex.get("snippet") or ""
        ah = ex.get("anchor_hash") or ""
        if not ah and callable(anchor_fn):
            try:
                ah = anchor_fn(url, ctx)
            except Exception:
                ah = ""
        out.append({
            "source_url": url,
            "raw": raw,
            "context_snippet": ctx[:500] if isinstance(ctx, str) else "",
            "anchor_hash": ah,
            "method": "value_range_examples",
        })
    return out


def ensure_metric_has_evidence(metric: dict) -> dict:
    """
    Evidence gating for a single metric:
    - If evidence already exists -> no change
    - Else synthesize from value_range.examples if available
    - Else mark as proxy (do not delete or zero the metric)
    """
    if not isinstance(metric, dict):
        return metric

    ev = _metric_evidence_list(metric)
    if ev:
        return metric

    synth = _synthesize_evidence_from_examples(metric)
    if synth:
        metric["evidence"] = synth
        return metric

    # No evidence at all: mark proxy (do not alter numeric payload)
    metric.setdefault("evidence", [])
    metric["is_proxy"] = True
    metric["proxy_type"] = "evidence_missing"
    metric["proxy_reason"] = "no_evidence_anchors_available"
    metric["proxy_confidence"] = float(metric.get("proxy_confidence") or 0.2)
    return metric


def enforce_evidence_gating(primary_metrics_canonical: dict) -> dict:
    """
    Apply evidence gating across canonical metrics.
    Returns the (mutated) dict for compatibility.
    """
    if not isinstance(primary_metrics_canonical, dict):
        return primary_metrics_canonical

    for k, m in list(primary_metrics_canonical.items()):
        if isinstance(m, dict):
            primary_metrics_canonical[k] = ensure_metric_has_evidence(m)

    return primary_metrics_canonical


def apply_schema_validation_and_evidence_gating(primary_data: dict) -> dict:
    """
    New Analysis post-pass hook:
    - validates metric_schema_frozen
    - evidence-gates primary_metrics_canonical
    - marks schema-conflict metrics as proxy (does not remove anything)
    """
    if not isinstance(primary_data, dict):
        return primary_data

    # Where schema is stored
    schema = (
        primary_data.get("metric_schema_frozen")
        or (primary_data.get("primary_response") or {}).get("metric_schema_frozen")
        or (primary_data.get("results") or {}).get("metric_schema_frozen")
        or {}
    )

    validation = validate_metric_schema_frozen(schema)
    primary_response = primary_data.setdefault("primary_response", {})
    primary_response["schema_validation"] = validation

    # Mark schema-conflict metrics as proxy (additive)
    pmc = primary_data.get("primary_metrics_canonical")
    if isinstance(pmc, dict) and validation.get("by_key"):
        for ck, iss in validation["by_key"].items():
            if ck in pmc and isinstance(pmc[ck], dict) and iss.get("errors"):
                pmc[ck]["is_proxy"] = True
                pmc[ck]["proxy_type"] = "schema_conflict"
                pmc[ck]["proxy_reason"] = "schema_validation_error"
                pmc[ck]["proxy_confidence"] = float(pmc[ck].get("proxy_confidence") or 0.15)
                pmc[ck]["schema_issues"] = {"errors": iss.get("errors", []), "warnings": iss.get("warnings", [])}

    # Evidence gating
    pmc2 = primary_data.get("primary_metrics_canonical")
    if isinstance(pmc2, dict):
        before = sum(1 for v in pmc2.values() if isinstance(v, dict) and _metric_evidence_list(v))
        enforce_evidence_gating(pmc2)
        after = sum(1 for v in pmc2.values() if isinstance(v, dict) and _metric_evidence_list(v))
        prox = sum(1 for v in pmc2.values() if isinstance(v, dict) and v.get("is_proxy"))
        primary_response["evidence_gating_summary"] = {
            "total_metrics": len(pmc2),
            "metrics_with_evidence_before": before,
            "metrics_with_evidence_after": after,
            "metrics_marked_proxy": prox,
        }

    return primary_data


if __name__ == "__main__":
    main()


# ===================== PATCH RMS_DISPATCH2 (ADDITIVE) =====================
def _get_metric_anchors_any(prev_response: dict) -> dict:
    """Best-effort retrieval of metric_anchors from any plausible location (additive helper)."""
    try:
        if not isinstance(prev_response, dict):
            return {}
        for path in (
            ("metric_anchors",),
            ("results", "metric_anchors"),
            ("primary_response", "metric_anchors"),
            ("primary_response", "results", "metric_anchors"),
        ):
            cur = prev_response
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, dict) and cur:
                return cur
        return {}
    except Exception:
        return {}

def _coerce_prev_response_any(previous_data):
    """Normalize previous_data into a dict-shaped 'prev_response' for rebuild dispatch (additive helper)."""
    try:
        return previous_data if isinstance(previous_data, dict) else {}
    except Exception:
        return {}
# =================== END PATCH RMS_DISPATCH2 (ADDITIVE) ===================
