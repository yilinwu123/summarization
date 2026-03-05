import os
import json
import pandas as pd
import argparse
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)




def chat_completion(system_text: str, user_text: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Single call to OpenRouter chat.completions; returns assistant content."""
    resp = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        #extra_body={"reasoning": {"enabled": True}},
    )
    return resp.choices[0].message.content or ""

SYSTEM_JUDGE = """You are a strict evaluator of faithfulness.
Your job: decide whether each bullet point is supported by the GOLD answer.
Supported means: the same claim is explicitly stated or can be directly inferred from GOLD with minimal assumptions.
If the bullet adds new details not in GOLD, mark UNSUPPORTED.

Be conservative: if unsure, choose UNSUPPORTED.

Return STRICT JSON only. No markdown. No extra text.
"""

USER_TEMPLATE = """GOLD:
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

def safe_json_load(s: str) -> Dict[str, Any]:
    """Robust-ish JSON parse: trims common junk, raises if still invalid."""
    s = s.strip()
    # Sometimes models wrap with ```json ... ```
    if s.startswith("```"):
        s = s.strip("`")
        # try to remove leading 'json'
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return json.loads(s)

def call_judge(gold: str, bullets: List[str],
               temperature: float = 0.0,
               max_tokens: int = 800) -> Dict[str, Any]:
    numbered = "\n".join([f"{i}. {b}" for i, b in enumerate(bullets)])
    user_text = USER_TEMPLATE.format(gold=gold, bullets=numbered)

    
        
    resp = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_JUDGE},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content
    return {"judge_raw": raw, "judge_json": safe_json_load(raw)}
        

def process_jsonl(in_path: str, out_path: str,
                  gold_key: str = "gold",
                  summary_key: str = "summary_json",
                  bullets_key: str = "bullets"):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="records"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            gold = record.get(gold_key, "")
            summary_json = record.get(summary_key, {}) or {}
            bullets = summary_json.get(bullets_key, []) or []

            # 允许 bullets 不是 list 的情况
            if not isinstance(bullets, list):
                bullets = [str(bullets)]

            if not gold or not bullets:
                record["judge"] = {
                    #"model": JUDGE_MODEL,
                    "error": "missing gold or bullets"
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            judge_out = call_judge(gold=gold, bullets=bullets)
            judge_json = judge_out["judge_json"]

            # 统计支持率
            results = judge_json.get("results", [])
            supported = sum(1 for r in results if (r.get("verdict") == "SUPPORTED"))
            support_rate = supported / max(1, len(bullets))

            record["judge"] = {
                #"model": JUDGE_MODEL,
                "support_rate": support_rate,
                "supported_count": supported,
                "bullet_count": len(bullets),
                "results": results,
                "raw": judge_out["judge_raw"],
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # 你改成自己的文件名
    INPUT_JSONL = "/Users/yilin/Downloads/out.jsonl"
    OUTPUT_JSONL = "/Users/yilin/Downloads/results_judged.jsonl"
    process_jsonl(INPUT_JSONL, OUTPUT_JSONL)
    print(f"Done. Wrote: {OUTPUT_JSONL}")