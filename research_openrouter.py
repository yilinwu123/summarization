import os
import json
import pandas as pd
import argparse
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from openai import OpenAI


# ---------------------------
# OpenRouter (OpenAI-compatible) client
# ---------------------------


def chat_completion(system_text: str, user_text: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Single call to OpenRouter chat.completions; returns assistant content."""
    resp = client.chat.completions.create(
        model="qwen/qwen3.5-plus-02-15",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        #extra_body={"reasoning": {"enabled": True}},
    )
    return resp.choices[0].message.content or ""


def load_records(path):
    """
    Supports:
    - JSONL: each line is a JSON object
    - JSON: a single object or a list
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        # Try JSON (single object or list)
        if first in ["{", "["]:
            try:
                obj = json.load(f)
                if isinstance(obj, list):
                    return obj
                return [obj]
            except json.JSONDecodeError:
                # Fall back to JSONL
                f.seek(0)

        # JSONL
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON decode error at line {line_no}: {e}\n"
                    f"Line content: {line[:200]}..."
                )
    return records


def build_transcript_text(meeting_transcripts: List[Dict[str, str]]) -> str:
    parts = []
    for t in meeting_transcripts:
        speaker = (t.get("speaker") or "").strip()
        content = (t.get("content") or "").strip()
        if not content:
            continue

        if speaker:
            parts.append(f"{speaker}: {content}")
        else:
            parts.append(content)

    return "\n".join(parts)


def parse_spans(span_list: Any) -> List[Tuple[int, int]]:
    spans = []
    if not span_list:
        return spans

    for pair in span_list:
        if not pair or len(pair) != 2:
            continue
        try:
            s = int(pair[0])
            e = int(pair[1])
            if e > s:
                spans.append((s, e))
        except Exception:
            continue

    return spans


def slice_by_spans(
    text: str,
    spans: List[Tuple[int, int]],
    window: int = 200
) -> str:
    if not spans:
        return text

    intervals = []
    n = len(text)
    for s, e in spans:
        s2 = max(0, s - window)
        e2 = min(n, e + window)
        intervals.append((s2, e2))

    intervals.sort()
    merged = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    chunks = [text[s:e] for s, e in merged]

    return "\n...\n".join(chunks)


def safe_json_load(s: str) -> Optional[Dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


KEYPOINTS_PROMPT = """
Task: Extract key points from the meeting transcript in response to the user query.

Requirements:
1) Provide 5–8 key points, ordered by importance.
2) Each key point must include:
   - point: a concise statement (<= 30 words)
   - evidence_quote: a short verbatim quote from the transcript (<= 30 words)
3) Only use information from the provided transcript. Do NOT invent facts.
4) Output STRICT JSON only, with this format:

{
  "keypoints": [
    {"point": "...", "evidence_quote": "..."}
  ],
  "coverage_note": "If the query is only partially addressed, explain what's missing; otherwise write 'Fully covered.'"
}

User Query:
{query}

Meeting Transcript (context):
{context}
"""


SUMMARY_PROMPT = """
Task: Summarize the discussion relevant to the user query from the meeting transcript.

Requirements:
1) Bullet points you think that can be used for answering the query. (each <= 25 words).
4) Use ONLY the transcript; do NOT add assumptions.
5) Output STRICT JSON only, with this format:

{{
  "bullets": ["...", "..."]
}}

User Query:
{query}

Meeting Transcript (context):
{context}
"""


SYSTEM_INSTRUCTION = "System instruction: You are a careful and faithful meeting analysis assistant."


def run_keypoints(query: str, context: str) -> str:
    final_text = KEYPOINTS_PROMPT.format(query=query, context=context)
    return chat_completion(SYSTEM_INSTRUCTION, final_text, temperature=0.2, max_tokens=700)


def run_summary(query: str, context: str) -> str:
    final_text = SUMMARY_PROMPT.format(query=query, context=context)
    return chat_completion(SYSTEM_INSTRUCTION, final_text, temperature=0.2, max_tokens=700)


def main():
    # NOTE: keeping your original hard-coded paths; you can parameterize with argparse later.
    records = load_records(
        "/Users/yilin/Downloads/train.jsonl"
    )

    all_results = []

    for idx, rec in enumerate(tqdm(records[0:2], desc="records")):
        print(idx)

        meeting_id = rec.get("meeting_id") or f"meet_{idx}"
        transcript_text = build_transcript_text(
            rec.get("meeting_transcripts", [])
        )

        for item in rec.get("specific_query_list", []) or []:
            query = (item.get("query") or "").strip()
            gold = (item.get("answer") or "").strip() or None

            if not query:
                continue

            context = transcript_text

            # kp_out = run_keypoints(query, context)
            sm_out = run_summary(query, context)

            result = {
                "meeting_id": meeting_id,
                "query_type": "specific",
                "query": query,
                "gold": gold,
                # "keypoints_raw": kp_out,
                "summary_raw": sm_out,
                # "keypoints_json": safe_json_load(kp_out),
                "summary_json": safe_json_load(sm_out),
                # helpful metadata for reproducibility
            }

            all_results.append(result)

    save_path = (
        "/Users/yilin/Downloads/out.jsonl"
    )

    with open(save_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_results)} results to {save_path}")


if __name__ == "__main__":
    main()
