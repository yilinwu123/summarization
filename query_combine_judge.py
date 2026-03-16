import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# =========================
# Config
# =========================
INPUT_JSON = "clusters.json"
OUTPUT_JSONL = "cluster_rule12_judged.jsonl"

MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0


# =========================
# Prompt
# =========================
SYSTEM_TEXT = """
You are a careful judge for constructing multi-document query-combination datasets.

Your task is to evaluate whether a cluster satisfies Rule 1 and Rule 2.

Definitions:

Rule 1: Answer-Document Alignment
- For each query, its provided answer must be supported by its own corresponding document.
- That means the answer should be grounded in the query's own meeting transcript.

Rule 2: Evidence Exclusivity
- For each query, the other documents in the cluster should NOT contain enough information to answer that query.
- In other words, each query should map to exactly one document for answering.

Important:
- Judge only based on the provided queries, answers, and meeting transcripts.
- Be strict.
- If another document contains enough evidence to answer a query, Rule 2 fails.
- If an answer is not supported by its own document, Rule 1 fails.
- Do not use outside knowledge.
- Return valid JSON only.
- Do not include markdown fences.
- Keep the reason under 40 words.
- overall_pass = rule1_pass AND rule2_pass.

Output JSON schema:
{
  "cluster_id": "string",
  "rule1_pass": true,
  "rule2_pass": true,
  "overall_pass": true,
  "violating_queries": ["string"],
  "reason": "string"
}
""".strip()

USER_TEMPLATE = """
Cluster ID:
{cluster_id}

Cluster Items:
{cluster_items}

Evaluate this cluster under Rule 1 and Rule 2.
Return JSON only.
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEXT),
        ("human", USER_TEMPLATE),
    ]
)


# =========================
# LLM
# =========================
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)

chain = prompt | llm


# =========================
# Helpers
# =========================
def load_clusters(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "clusters" in data:
        return data["clusters"]

    raise ValueError("Unsupported input JSON format. Expect a list or a dict with key 'clusters'.")


def normalize_cluster(cluster: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Normalize different possible cluster formats into:
    {
        "cluster_id": ...,
        "items": [
            {
                "query_id": ...,
                "query": ...,
                "answer": ...,
                "meeting_transcript": ...
            },
            ...
        ]
    }
    """
    cluster_id = cluster.get("cluster_id", f"cluster_{idx}")

    items = None
    for key in ["items", "queries", "examples", "members"]:
        if key in cluster:
            items = cluster[key]
            break

    if items is None:
        raise ValueError(
            f"Cluster {cluster_id} missing one of fields: items / queries / examples / members"
        )

    normalized_items = []
    for j, x in enumerate(items):
        query_id = x.get("query_id", x.get("id", f"{cluster_id}_q{j}"))
        query = x.get("query", "")
        answer = x.get("answer", x.get("gold_answer", ""))
        transcript = x.get("meeting_transcript", x.get("transcript", x.get("document", "")))

        normalized_items.append(
            {
                "query_id": query_id,
                "query": query,
                "answer": answer,
                "meeting_transcript": transcript,
            }
        )

    return {
        "cluster_id": cluster_id,
        "items": normalized_items,
    }


def build_cluster_text(cluster: Dict[str, Any]) -> str:
    parts = []
    for item in cluster["items"]:
        block = f"""
[Query ID] {item["query_id"]}
[Query]
{item["query"]}

[Answer]
{item["answer"]}

[Own Meeting Transcript]
{item["meeting_transcript"]}
""".strip()
        parts.append(block)

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(parts)


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)


def judge_one_cluster(cluster: Dict[str, Any]) -> Dict[str, Any]:
    cluster_items_text = build_cluster_text(cluster)

    response = chain.invoke(
        {
            "cluster_id": cluster["cluster_id"],
            "cluster_items": cluster_items_text,
        }
    )

    raw_text = response.content
    result = safe_parse_json(raw_text)

    # minimal post-check / cleanup
    result.setdefault("cluster_id", cluster["cluster_id"])
    result.setdefault("rule1_pass", False)
    result.setdefault("rule2_pass", False)
    result.setdefault("overall_pass", bool(result["rule1_pass"] and result["rule2_pass"]))
    result.setdefault("violating_queries", [])
    result.setdefault("reason", "")

    return result


def main():
    clusters_raw = load_clusters(INPUT_JSON)
    clusters = [normalize_cluster(c, i) for i, c in enumerate(clusters_raw)]

    output_path = Path(OUTPUT_JSONL)

    with output_path.open("w", encoding="utf-8") as f:
        for i, cluster in enumerate(clusters):
            try:
                result = judge_one_cluster(cluster)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    f"[OK] {i}: {cluster['cluster_id']} | "
                    f"overall_pass={result['overall_pass']}"
                )
            except Exception as e:
                error_obj = {
                    "cluster_id": cluster["cluster_id"],
                    "rule1_pass": False,
                    "rule2_pass": False,
                    "overall_pass": False,
                    "violating_queries": [],
                    "reason": f"ERROR: {str(e)}",
                }
                f.write(json.dumps(error_obj, ensure_ascii=False) + "\n")
                print(f"[ERROR] {i}: {cluster['cluster_id']} -> {e}")

    print(f"Done. Results saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
