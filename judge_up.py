import json
import re
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

# ============================================================
# Robust JSON parse
# ============================================================

def safe_json_load(s: str) -> Dict[str, Any]:
    """Robust-ish JSON parse: trims common junk, raises if still invalid."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip().strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return json.loads(s)


# ============================================================
# Helpers
# ============================================================

def _numbered(items: List[str]) -> str:
    return "\n".join([f"{i}. {x}" for i, x in enumerate(items)])

def _clip(s: str, n: int = 12000) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "\n...[TRUNCATED]..."

def _contains_leap_cues(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    cues = [
        # EN
        "because", "therefore", "thus", "hence", "as a result", "resulted in", "leads to", "led to",
        "caused", "cause", "drives", "driven by", "due to", "so that", "in order to",
        "means", "implies", "suggests", "indicates", "shows that", "evidence that",
        "insight", "takeaway", "lesson", "conclusion", "it seems", "likely", "probably",
        "motivated", "intent", "purpose", "goal", "aim",
        # ZH
        "因为", "因此", "所以", "导致", "结果是", "从而", "意味着", "说明", "表明", "暗示", "推测",
        "洞察", "启示", "结论", "目的", "动机", "意图"
    ]
    return any(c in t for c in cues)

def _get_first_str(record: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _get_query(record: Dict[str, Any], summary_key: str, query_keys: List[str]) -> str:
    q = _get_first_str(record, query_keys)
    if q:
        return q
    sj = record.get(summary_key, {}) or {}
    if isinstance(sj, dict):
        for k in query_keys:
            v = sj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

def _get_doc_text(record: Dict[str, Any], summary_key: str, doc_keys: List[str]) -> str:
    # try top-level first
    doc = _get_first_str(record, doc_keys)
    if doc:
        return doc
    # try nested under summary_json or others
    sj = record.get(summary_key, {}) or {}
    if isinstance(sj, dict):
        for k in doc_keys:
            v = sj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

def _normalize_bullets(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for b in x:
            s = str(b).strip()
            if s:
                out.append(s)
        return out
    s = str(x).strip()
    return [s] if s else []


# ============================================================
# OpenRouter call wrapper (keeps your original style)
# ============================================================

def chat_completion(system_text: str, user_text: str, *, model: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# ============================================================
# Judge 1: Doc-grounding (does the bullet have support in the DOC?)
# ============================================================

SYSTEM_DOC_GROUND = """You are a strict evaluator of document grounding.

Task:
For each bullet, decide whether its main factual content is supported by the provided DOCUMENT.

Definitions:
- GROUNDED: The bullet's core factual claim(s) are explicitly stated in the DOCUMENT or directly paraphrasable.
- NOT_GROUNDED: The bullet introduces facts not found in the DOCUMENT.

Rules:
- Be conservative: if unsure, choose NOT_GROUNDED.
- If the bullet contains multiple claims and any key claim is missing, choose NOT_GROUNDED.
- Provide an evidence quote/snippet from DOCUMENT if GROUNDED (<= 40 words).
- If NOT_GROUNDED, evidence must be empty string.

Return STRICT JSON only. No markdown. No extra text.
"""

USER_DOC_GROUND = """DOCUMENT:
{doc}

BULLETS (numbered):
{bullets}

Output JSON schema:
{{
  "results": [
    {{
      "index": 0,
      "verdict": "GROUNDED|NOT_GROUNDED",
      "rationale": "short reason",
      "evidence": "short quote/snippet from DOCUMENT (<=40 words) or empty if NOT_GROUNDED"
    }}
  ]
}}
"""

def call_doc_ground_judge(doc: str, bullets: List[str], *, model: str, temperature: float = 0.0, max_tokens: int = 900) -> Dict[str, Any]:
    user_text = USER_DOC_GROUND.format(doc=_clip(doc, 14000), bullets=_numbered(bullets))
    raw = chat_completion(SYSTEM_DOC_GROUND, user_text, model=model, temperature=temperature, max_tokens=max_tokens)
    return {"raw": raw, "json": safe_json_load(raw)}


# ============================================================
# Judge 2: Apophenia (ONLY when grounded): "facts exist, but leap in causality/insight"
# ============================================================

SYSTEM_APOPHENIA = """You are a strict detector of apophenia-as-overinference.

We define APOPHENIA as:
- The bullet is grounded in the DOCUMENT (facts/observations appear there),
- BUT the bullet adds unjustified causal links, motives, intentions, implications, broad generalizations, or "insights" that are NOT explicitly stated in the DOCUMENT.

Task:
Given DOCUMENT and one BULLET, classify whether it contains an overinference leap beyond what DOCUMENT states.

Labels:
- NO_APOPHENIA: bullet stays descriptive; any relations/interpretations are explicitly stated in DOCUMENT.
- APOPHENIA: bullet adds causal/motive/implication/generalization/insight not stated.

Leap types (choose one):
- NONE
- CAUSAL_LEAP          (A causes B / therefore / results in, not stated)
- MOTIVE_LEAP          (intent/purpose/goal, not stated)
- IMPLICATION_LEAP     (takeaway/insight/lesson/prediction, not stated)
- GENERALIZATION_LEAP  (from specific to broad policy/trend, not stated)
- MECHANISM_LEAP       (explains hidden mechanism, not stated)

Severity (0-3):
- 0: NONE
- 1: mild interpretive shading
- 2: clear leap that changes meaning
- 3: strong misleading leap (high-risk)

Rules:
- If DOCUMENT only lists facts A and B, do NOT infer A->B causality unless DOCUMENT says so.
- If unsure, choose APOPHENIA (conservative).
- Provide:
  (a) the exact "leap span" from the bullet (the part that goes beyond doc),
  (b) a short supporting snippet from DOCUMENT if it explicitly states the leap; otherwise empty.

Return STRICT JSON only. No markdown. No extra text.
"""

USER_APOPHENIA = """DOCUMENT:
{doc}

BULLET:
{bullet}

Output JSON schema:
{{
  "verdict": "APOPHENIA|NO_APOPHENIA",
  "leap_type": "NONE|CAUSAL_LEAP|MOTIVE_LEAP|IMPLICATION_LEAP|GENERALIZATION_LEAP|MECHANISM_LEAP",
  "severity": 0,
  "rationale": "short reason",
  "leap_span": "the phrase in BULLET that constitutes the leap; empty if NO_APOPHENIA",
  "doc_support_for_leap": "short quote/snippet from DOCUMENT (<=40 words) that explicitly supports the leap; empty if none"
}}
"""

def call_apophenia_judge(doc: str, bullet: str, *, model: str, temperature: float = 0.0, max_tokens: int = 500) -> Dict[str, Any]:
    user_text = USER_APOPHENIA.format(doc=_clip(doc, 14000), bullet=bullet)
    raw = chat_completion(SYSTEM_APOPHENIA, user_text, model=model, temperature=temperature, max_tokens=max_tokens)
    return {"raw": raw, "json": safe_json_load(raw)}


# ============================================================
# Optional: Judge 3 (legacy): Gold faithfulness (keep if you still want)
# ============================================================

SYSTEM_GOLD_FAITHFULNESS = """You are a strict evaluator of faithfulness to GOLD.
Decide whether each bullet point is supported by the GOLD answer.

Supported means: the same claim is explicitly stated or can be directly inferred from GOLD with minimal assumptions.
If the bullet adds new details not in GOLD, mark UNSUPPORTED.

Be conservative: if unsure, choose UNSUPPORTED.

Return STRICT JSON only. No markdown. No extra text.
"""

USER_GOLD_FAITHFULNESS = """GOLD:
{gold}

BULLETS (numbered):
{bullets}

For each bullet, output:
- verdict: "SUPPORTED" or "UNSUPPORTED"
- rationale: short reason citing what in GOLD supports it (or why not)
- evidence: a short quote or paraphrase from GOLD that supports it; empty string if UNSUPPORTED

Return JSON with schema:
{{
  "results": [
    {{
      "index": 0,
      "verdict": "SUPPORTED|UNSUPPORTED",
      "rationale": "...",
      "evidence": "..."
    }}
  ]
}}
"""

def call_gold_faithfulness_judge(gold: str, bullets: List[str], *, model: str, temperature: float = 0.0, max_tokens: int = 800) -> Dict[str, Any]:
    user_text = USER_GOLD_FAITHFULNESS.format(gold=gold, bullets=_numbered(bullets))
    raw = chat_completion(SYSTEM_GOLD_FAITHFULNESS, user_text, model=model, temperature=temperature, max_tokens=max_tokens)
    return {"raw": raw, "json": safe_json_load(raw)}


# ============================================================
# Main: process JSONL with apophenia pipeline
# ============================================================

def process_jsonl_apophenia_judge(
    in_path: str,
    out_path: str,
    *,
    # keys
    gold_key: str = "gold",
    summary_key: str = "summary_json",
    bullets_key: str = "bullets",
    # doc/query keys (auto-detect candidates)
    doc_key_candidates: Optional[List[str]] = None,
    query_key_candidates: Optional[List[str]] = None,
    # models
    judge_model: str = "meta-llama/llama-3.1-8b-instruct",
    # knobs
    do_gold_faithfulness: bool = True,
    apophenia_only_if_grounded: bool = True,   # recommended for your definition
    route_apophenia_by_cues: bool = True,      # save cost: only call apophenia judge if cues exist (still grounded)
):
    """
    Output adds:
      record["judge_apophenia"] = {
        "doc_ground": ...,
        "apophenia": ...,
        "gold_faithfulness": ... (optional),
        "metrics": ...
      }
    """

    if doc_key_candidates is None:
        doc_key_candidates = ["document", "doc", "context", "transcript", "meeting_transcript", "source_text", "text"]
    if query_key_candidates is None:
        query_key_candidates = ["query", "question", "user_query", "instruction", "prompt"]

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="records"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            gold = (record.get(gold_key, "") or "").strip()
            sj = record.get(summary_key, {}) or {}
            bullets = _normalize_bullets(sj.get(bullets_key, []))

            query = _get_query(record, summary_key, query_key_candidates)
            doc = _get_doc_text(record, summary_key, doc_key_candidates)

            if not bullets:
                record["judge_apophenia"] = {"error": "missing bullets"}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            if not doc:
                record["judge_apophenia"] = {"error": "missing document text (doc/context/transcript)"}  # important
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # ---- Doc Grounding ----
            dg_out = call_doc_ground_judge(doc=doc, bullets=bullets, model=judge_model, temperature=0.0, max_tokens=900)
            dg_json = dg_out["json"]
            dg_results = dg_json.get("results", []) or []

            # Index to grounding verdict
            grounding_by_i: Dict[int, Dict[str, Any]] = {}
            for r in dg_results:
                i = r.get("index")
                if isinstance(i, int) and 0 <= i < len(bullets):
                    grounding_by_i[i] = {
                        "verdict": r.get("verdict", "NOT_GROUNDED"),
                        "rationale": r.get("rationale", "") or "",
                        "evidence": r.get("evidence", "") or "",
                    }

            # ---- Optional Gold Faithfulness ----
            gf_out = None
            gf_by_i: Dict[int, Dict[str, Any]] = {}
            if do_gold_faithfulness and gold:
                gf_out = call_gold_faithfulness_judge(gold=gold, bullets=bullets, model=judge_model, temperature=0.0, max_tokens=800)
                gf_json = gf_out["json"]
                gf_results = gf_json.get("results", []) or []
                for r in gf_results:
                    i = r.get("index")
                    if isinstance(i, int) and 0 <= i < len(bullets):
                        gf_by_i[i] = {
                            "verdict": r.get("verdict", "UNSUPPORTED"),
                            "rationale": r.get("rationale", "") or "",
                            "evidence": r.get("evidence", "") or "",
                        }

            # ---- Apophenia (only grounded by your definition) ----
            bullet_items: List[Dict[str, Any]] = []
            apo_calls = 0

            for i, b in enumerate(bullets):
                g = grounding_by_i.get(i, {"verdict": "NOT_GROUNDED", "rationale": "", "evidence": ""})
                grounded = (g["verdict"] == "GROUNDED")

                item = {
                    "index": i,
                    "text": b,
                    "doc_grounding": g,
                }

                if do_gold_faithfulness and gold:
                    item["gold_faithfulness"] = gf_by_i.get(i, {"verdict": "UNSUPPORTED", "rationale": "", "evidence": ""})

                # Routing logic for apophenia judge
                need_apo = True
                if apophenia_only_if_grounded and (not grounded):
                    need_apo = False
                    item["apophenia"] = {
                        "verdict": "N/A",
                        "leap_type": "NONE",
                        "severity": 0,
                        "rationale": "Not grounded in DOCUMENT; classify as hallucination/not-supported rather than apophenia by definition.",
                        "leap_span": "",
                        "doc_support_for_leap": "",
                    }
                else:
                    if route_apophenia_by_cues and (not _contains_leap_cues(b)):
                        need_apo = False
                        item["apophenia"] = {
                            "verdict": "NO_APOPHENIA",
                            "leap_type": "NONE",
                            "severity": 0,
                            "rationale": "No clear causal/insight leap cues in bullet; skipped apophenia judge by routing.",
                            "leap_span": "",
                            "doc_support_for_leap": "",
                        }

                if need_apo:
                    apo_calls += 1
                    apo_out = call_apophenia_judge(doc=doc, bullet=b, model=judge_model, temperature=0.0, max_tokens=520)
                    apo_json = apo_out["json"]
                    item["apophenia"] = {
                        "verdict": apo_json.get("verdict", "APOPHENIA"),
                        "leap_type": apo_json.get("leap_type", "NONE"),
                        "severity": int(apo_json.get("severity", 0) or 0),
                        "rationale": apo_json.get("rationale", "") or "",
                        "leap_span": apo_json.get("leap_span", "") or "",
                        "doc_support_for_leap": apo_json.get("doc_support_for_leap", "") or "",
                        "raw": apo_out["raw"],
                    }

                bullet_items.append(item)

            # ---- Metrics ----
            n = len(bullet_items)
            grounded_n = sum(1 for x in bullet_items if x["doc_grounding"]["verdict"] == "GROUNDED")
            not_grounded_n = n - grounded_n

            # apophenia among grounded (your definition)
            apophenia_n = 0
            severity_sum = 0
            leap_type_counts: Dict[str, int] = {}

            for x in bullet_items:
                apo = x.get("apophenia", {})
                if x["doc_grounding"]["verdict"] == "GROUNDED":
                    if apo.get("verdict") == "APOPHENIA":
                        apophenia_n += 1
                        severity_sum += int(apo.get("severity", 0) or 0)
                        lt = apo.get("leap_type", "NONE")
                        leap_type_counts[lt] = leap_type_counts.get(lt, 0) + 1
                    else:
                        lt = apo.get("leap_type", "NONE")
                        leap_type_counts[lt] = leap_type_counts.get(lt, 0) + 1

            hallucination_rate = not_grounded_n / max(1, n)
            apophenia_rate = apophenia_n / max(1, grounded_n)  # conditional rate
            severity_avg = severity_sum / max(1, grounded_n)

            out_obj = {
                "model": judge_model,
                "query_used": query,
                "apophenia_calls": apo_calls,
                "metrics": {
                    "bullet_count": n,
                    "grounded_count": grounded_n,
                    "not_grounded_count": not_grounded_n,
                    "hallucination_rate_not_grounded": hallucination_rate,
                    "apophenia_count_among_grounded": apophenia_n,
                    "apophenia_rate_among_grounded": apophenia_rate,
                    "severity_avg_among_grounded": severity_avg,
                    "leap_type_counts_among_grounded": leap_type_counts,
                },
                "bullets": bullet_items,
                "raw": {
                    "doc_ground": dg_out["raw"],
                    **({"gold_faithfulness": gf_out["raw"]} if (gf_out is not None) else {}),
                }
            }

            record["judge_apophenia"] = out_obj
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Example usage
# ============================================================
# process_jsonl_apophenia_judge(
#     in_path="input.jsonl",
#     out_path="output_apophenia.jsonl",
#     gold_key="gold",
#     summary_key="summary_json",
#     bullets_key="bullets",
#     doc_key_candidates=["meeting_transcript", "transcript", "context", "doc", "document", "source_text"],
#     query_key_candidates=["query", "question", "specific_query"],
#     judge_model="meta-llama/llama-3.1-8b-instruct",
#     do_gold_faithfulness=True,
#     apophenia_only_if_grounded=True,
#     route_apophenia_by_cues=True,
# )
