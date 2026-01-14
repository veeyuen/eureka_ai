# fix2b0_unified_engine_poc_v1.py
# Yureeka — Minimal Unified Engine Proof-of-Life (Option 1)
#
# Purpose:
# - Provide a clean, standalone, deterministic "unified engine" path that:
#   (a) normalizes baseline + injected URLs,
#   (b) fetches text deterministically (including GitHub blob→raw),
#   (c) extracts a small set of EV metrics,
#   (d) binds to a frozen subset of schema keys,
#   (e) emits the Evolution-compatible output contract:
#       - metric_changes (with current_value populated when found)
#       - canonical_metrics (schema-keyed point estimates)
#       - source_results (per-URL fetch/extraction status)
#       - output_debug (traces)
#
# This file is intentionally independent of legacy wrappers.
#
# CODE_VERSION policy:
# - Update whenever a new file is issued.
CODE_VERSION = "fix2b0_unified_engine_poc_v1"

# PATCH TRACKER (append-only)
# - fix2b0_unified_engine_poc_v1: Minimal unified engine PoC (fetch+extract+bind+diff for a small schema subset)

from __future__ import annotations

import re
import json
import time
import math
import html
import hashlib
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from html.parser import HTMLParser


# -----------------------------
# URL normalization
# -----------------------------

_GITHUB_BLOB_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^#?]+)$", re.IGNORECASE)

def _strip_tracking(url: str) -> str:
    """Deterministic URL normalizer: strip common tracking params and fragments."""
    u = (url or "").strip()
    if not u:
        return ""
    # Remove fragment
    u = u.split("#", 1)[0]
    # Remove querystring entirely (safe for PoC; keeps identity stable)
    u = u.split("?", 1)[0]
    # Normalize scheme casing
    if u.startswith("http://"):
        # prefer https
        u = "https://" + u[len("http://"):]
    return u

def normalize_url(url: str) -> str:
    """Normalize URLs deterministically; convert GitHub blob to raw where possible."""
    u = _strip_tracking(url)
    if not u:
        return ""
    m = _GITHUB_BLOB_RE.match(u)
    if m:
        owner, repo, path = m.group(1), m.group(2), m.group(3)
        # GitHub raw host uses path without leading slash already in regex capture
        u = f"https://raw.githubusercontent.com/{owner}/{repo}/{path}"
    return u

def normalize_url_list(urls: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in (urls or []):
        nu = normalize_url(str(u or ""))
        if not nu or nu in seen:
            continue
        seen.add(nu)
        out.append(nu)
    return out


# -----------------------------
# Fetch + HTML→text
# -----------------------------

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: List[str] = []
    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)
    def get_text(self) -> str:
        return " ".join(self._parts)

def html_to_text(s: str) -> str:
    if not s:
        return ""
    # Fast path: strip scripts/styles crudely
    s2 = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", s)
    s2 = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", s2)
    p = _TextExtractor()
    try:
        p.feed(s2)
    except Exception:
        # fallback: regex tag strip
        s2 = re.sub(r"(?is)<[^>]+>", " ", s2)
        return re.sub(r"\s+", " ", html.unescape(s2)).strip()
    txt = html.unescape(p.get_text())
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def fetch_url(url: str, timeout_s: int = 20) -> Tuple[str, str, str]:
    """
    Deterministic fetch: returns (status, content_type, text)
    status: fetched_ok | fetched_empty | failed:<reason>
    """
    if not url:
        return ("failed:empty_url", "", "")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "YureekaPoC/1.0 (+https://example.invalid)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            raw = resp.read()
            # Attempt utf-8 decode; fallback latin-1
            try:
                body = raw.decode("utf-8", errors="replace")
            except Exception:
                body = raw.decode("latin-1", errors="replace")
            txt = html_to_text(body) if ("html" in ctype or body.lstrip().startswith("<")) else body.strip()
            if not txt.strip():
                return ("fetched_empty", ctype, "")
            return ("fetched_ok", ctype, txt)
    except urllib.error.HTTPError as e:
        return (f"failed:http_{getattr(e, 'code', 'unknown')}", "", "")
    except urllib.error.URLError as e:
        return (f"failed:url_{type(e).__name__}", "", "")
    except Exception as e:
        return (f"failed:{type(e).__name__}", "", "")


# -----------------------------
# Extraction (minimal)
# -----------------------------

@dataclass
class Candidate:
    canonical_key: str
    value_norm: float
    display: str
    unit_tag: str
    source_url: str
    context: str
    confidence: float

def _context_window(text: str, idx: int, win: int = 140) -> str:
    if not text:
        return ""
    a = max(0, idx - win)
    b = min(len(text), idx + win)
    return text[a:b].strip()

# Simple patterns geared to your PoC sentence and common phrasing
_RE_MILLION_UNITS_2024 = re.compile(
    r"(?i)(?:ev\s+sales|global\s+ev\s+sales|electric\s+vehicle\s+sales).*?(?:in\s+)?2024.*?(\d{1,3}(?:\.\d+)?)\s*(?:million|m)\s*(?:units|vehicles|cars)\b"
)
_RE_PCT_2024_SHARE = re.compile(
    r"(?i)(?:ev\s+share|share\s+of\s+global\s+light[-\s]?vehicle\s+sales|share\s+of\s+light[-\s]?vehicle\s+sales).*?(?:in\s+)?2024.*?(\d{1,3}(?:\.\d+)?)\s*%"
)
_RE_REVENUE_2024 = re.compile(
    r"(?i)(?:ev\s+revenue|global\s+new\s+ev\s+revenue|estimated\s+global\s+new\s+ev\s+revenue).*?(?:in\s+)?2024.*?(\d{1,3}(?:\.\d+)?)\s*(?:billion|bn)\b"
)
_RE_UNITS_2030 = re.compile(
    r"(?i)(?:projected|forecast|by)\s+2030.*?(?:ev\s+sales|global\s+ev\s+sales|units).*?(\d{1,3}(?:\.\d+)?)\s*(?:million|m)\s*(?:units|vehicles|cars)\b"
)

def extract_candidates(text: str, source_url: str) -> List[Candidate]:
    cands: List[Candidate] = []
    if not text:
        return cands

    # 2024 Global EV Sales (unit sales, in million units)
    for m in _RE_MILLION_UNITS_2024.finditer(text):
        v = float(m.group(1))
        idx = m.start(1)
        ctx = _context_window(text, idx)
        cands.append(Candidate(
            canonical_key="global_ev_sales_2024__unit_sales",
            value_norm=v,
            display=f"{v:g}",
            unit_tag="unit_sales_million",
            source_url=source_url,
            context=ctx,
            confidence=0.90,
        ))

    # 2024 EV share of light vehicle sales (%)
    for m in _RE_PCT_2024_SHARE.finditer(text):
        v = float(m.group(1))
        idx = m.start(1)
        ctx = _context_window(text, idx)
        cands.append(Candidate(
            canonical_key="ev_share_of_global_light_vehicle_sales_2024__percent",
            value_norm=v,
            display=f"{v:g}",
            unit_tag="percent",
            source_url=source_url,
            context=ctx,
            confidence=0.88,
        ))

    # 2024 Estimated global new EV revenue (billion currency units)
    for m in _RE_REVENUE_2024.finditer(text):
        v = float(m.group(1))
        idx = m.start(1)
        ctx = _context_window(text, idx)
        cands.append(Candidate(
            canonical_key="estimated_global_new_ev_revenue_2024__currency",
            value_norm=v,
            display=f"{v:g}",
            unit_tag="currency_billion",
            source_url=source_url,
            context=ctx,
            confidence=0.75,
        ))

    # 2030 projected EV sales/units (million units)
    for m in _RE_UNITS_2030.finditer(text):
        v = float(m.group(1))
        idx = m.start(1)
        ctx = _context_window(text, idx)
        cands.append(Candidate(
            canonical_key="projected_global_ev_sales_2030__unit_sales",
            value_norm=v,
            display=f"{v:g}",
            unit_tag="unit_sales_million",
            source_url=source_url,
            context=ctx,
            confidence=0.70,
        ))

    return cands


# -----------------------------
# Selection + binding (minimal)
# -----------------------------

FROZEN_KEYS = [
    "global_ev_sales_2024__unit_sales",
    "ev_share_of_global_light_vehicle_sales_2024__percent",
    "estimated_global_new_ev_revenue_2024__currency",
    "projected_global_ev_sales_2030__unit_sales",
]

def _pick_best(cands: List[Candidate]) -> Optional[Candidate]:
    if not cands:
        return None
    # deterministic: sort by confidence desc, then stable tie-breakers
    def k(c: Candidate):
        # prefer injected sources? In PoC, treat non-baseline as "injected" if URL not in baseline set, handled outside
        return (-float(c.confidence or 0.0), c.source_url, c.canonical_key, float(c.value_norm))
    return sorted(cands, key=k)[0]

def build_canonical_metrics(candidates: List[Candidate], baseline_urls_norm: List[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Return (canonical_metrics, debug_binding)"""
    by_key: Dict[str, List[Candidate]] = {}
    for c in candidates:
        by_key.setdefault(c.canonical_key, []).append(c)

    cm: Dict[str, float] = {}
    dbg: Dict[str, Any] = {"bound": {}, "missing": []}

    for key in FROZEN_KEYS:
        cands = by_key.get(key) or []
        if not cands:
            dbg["missing"].append(key)
            continue

        # Option A (injected wins): prefer candidates whose source_url not in baseline_urls_norm
        injected = [c for c in cands if (normalize_url(c.source_url) not in set(baseline_urls_norm))]
        chosen = _pick_best(injected) if injected else _pick_best(cands)
        if chosen:
            cm[key] = float(chosen.value_norm)
            dbg["bound"][key] = {
                "value_norm": chosen.value_norm,
                "unit_tag": chosen.unit_tag,
                "source_url": chosen.source_url,
                "confidence": chosen.confidence,
                "context": chosen.context[:240],
                "picked_from": "injected" if chosen in injected else "baseline_or_any",
                "candidates_count": len(cands),
                "injected_candidates_count": len(injected),
            }
    return cm, dbg


# -----------------------------
# Diff (minimal)
# -----------------------------

def compute_metric_changes(prev_canonical_metrics: Dict[str, Any], cur_canonical_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prev = prev_canonical_metrics or {}
    cur = cur_canonical_metrics or {}
    all_keys = sorted(set(FROZEN_KEYS) | set(prev.keys()) | set(cur.keys()))
    for k in all_keys:
        pv = prev.get(k)
        cv = cur.get(k)
        # normalize missing
        pv_num = float(pv) if isinstance(pv, (int, float)) else (float(pv) if isinstance(pv, str) and pv.strip().replace(".","",1).isdigit() else None)
        cv_num = float(cv) if isinstance(cv, (int, float)) else (float(cv) if isinstance(cv, str) and cv.strip().replace(".","",1).isdigit() else None)
        if cv_num is None:
            status = "not_found"
            current_value = "N/A"
        else:
            status = "ok"
            current_value = f"{cv_num:g}"
        previous_value = "" if pv_num is None else f"{pv_num:g}"
        change_pct = None
        if pv_num is not None and cv_num is not None and pv_num != 0:
            change_pct = (cv_num - pv_num) / pv_num * 100.0
        rows.append({
            "metric": k,
            "canonical_key": k,
            "match_stage": "poc_bind",
            "previous_value": previous_value,
            "current_value": current_value,
            "change_pct": change_pct,
            "status": status,
            "match_confidence": 1.0 if status == "ok" else 0.0,
            "anchor_used": True if status == "ok" else False,
        })
    return rows


# -----------------------------
# Public entrypoint
# -----------------------------

def run_unified_poc(
    question: str,
    baseline_urls: Optional[List[str]] = None,
    injected_urls: Optional[List[str]] = None,
    prev_snapshot: Optional[Dict[str, Any]] = None,
    timeout_s: int = 20,
) -> Dict[str, Any]:
    """
    Minimal unified engine PoC.

    Inputs:
      - question: str
      - baseline_urls: list[str]
      - injected_urls: list[str]
      - prev_snapshot: optional dict with {'canonical_metrics': {...}} or {'canonical_metrics': {...}} at top-level

    Output:
      dict with:
        status, message,
        canonical_metrics (schema-keyed dict),
        metric_changes (list of dicts),
        source_results (list),
        output_debug (dict)
    """
    t0 = time.time()
    baseline_urls_norm = normalize_url_list(baseline_urls or [])
    injected_urls_norm = normalize_url_list(injected_urls or [])

    urls = normalize_url_list((baseline_urls_norm or []) + (injected_urls_norm or []))

    source_results: List[Dict[str, Any]] = []
    all_candidates: List[Candidate] = []

    for u in urls:
        st, ctype, txt = fetch_url(u, timeout_s=timeout_s)
        cands = extract_candidates(txt, u) if (st == "fetched_ok" and txt) else []
        all_candidates.extend(cands)
        source_results.append({
            "url": u,
            "status": st if st.startswith("fetched") else "failed",
            "status_detail": "" if st.startswith("fetched") else st,
            "content_type": ctype,
            "text_len": int(len(txt or "")),
            "numbers_found": int(len(cands)),
            "is_injected": bool(u in set(injected_urls_norm)),
        })

    canonical_metrics, dbg_binding = build_canonical_metrics(all_candidates, baseline_urls_norm)

    # Prev canonical metrics extraction
    prev_cm = {}
    if isinstance(prev_snapshot, dict):
        if isinstance(prev_snapshot.get("canonical_metrics"), dict):
            prev_cm = prev_snapshot.get("canonical_metrics") or {}
        elif isinstance(prev_snapshot.get("results"), dict) and isinstance(prev_snapshot["results"].get("canonical_metrics"), dict):
            prev_cm = prev_snapshot["results"].get("canonical_metrics") or {}
        else:
            # accept direct schema map
            prev_cm = {k: prev_snapshot.get(k) for k in FROZEN_KEYS if k in prev_snapshot}

    metric_changes = compute_metric_changes(prev_cm, canonical_metrics)

    # Basic stability score: percent unchanged among found keys
    found = [k for k in FROZEN_KEYS if k in canonical_metrics]
    stability = 0.0
    if found:
        unchanged = 0
        for k in found:
            pv = prev_cm.get(k)
            cv = canonical_metrics.get(k)
            try:
                pvf = float(pv) if pv is not None else None
                cvf = float(cv) if cv is not None else None
                if pvf is not None and cvf is not None and abs(pvf - cvf) < 1e-9:
                    unchanged += 1
            except Exception:
                pass
        stability = (unchanged / max(1, len(found))) * 100.0

    out = {
        "status": "success",
        "message": "PoC unified engine completed",
        "code_version": CODE_VERSION,
        "sources_checked": int(len(urls)),
        "sources_fetched": int(sum(1 for s in source_results if str(s.get("status")).startswith("fetched") or s.get("status") == "fetched_ok")),
        "stability_score": stability,
        "canonical_metrics": canonical_metrics,
        "metric_changes": metric_changes,
        "source_results": source_results,
        "output_debug": {
            "question": question,
            "baseline_urls_norm": baseline_urls_norm,
            "injected_urls_norm": injected_urls_norm,
            "urls_universe": urls,
            "binding": dbg_binding,
            "candidates_total": int(len(all_candidates)),
            "runtime_ms": int((time.time() - t0) * 1000),
        },
    }
    return out


if __name__ == "__main__":
    # Simple CLI sanity
    demo = run_unified_poc(
        question="demo",
        baseline_urls=[],
        injected_urls=["https://github.com/veeyuen/injecton-test-page/blob/main/index.html"],
        prev_snapshot={"canonical_metrics": {}},
    )
    print(json.dumps(demo, indent=2)[:4000])
