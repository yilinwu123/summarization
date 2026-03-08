import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter


# ============================================================
# 0. 配置
# ============================================================

QMSUM_ROOT = "/Users/yilin/Downloads/QMSum"
OUTPUT_DIR = "/Users/yilin/Downloads/researchresults"

VALID_DOMAINS = {"Academic", "Committee", "Product"}

CACHE_LABEL_PATH = os.path.join(OUTPUT_DIR, "label_cache.json")
CACHE_PAIR_PATH = os.path.join(OUTPUT_DIR, "pair_cache.json")

USE_CACHE = True
SAVE_EVERY = 20

# 每个 query 在同组里最多比较多少个候选
MAX_CANDIDATES_PER_QUERY = 12

# cluster 至少来自多少个不同 meeting
MIN_MEETINGS_PER_CLUSTER = 2

# 只保留 size >= 2 的 cluster
MIN_CLUSTER_SIZE = 2

# 你的 LLM 封装
# 这里直接按你的结构写
from openai import OpenAI

client = OpenAI(
    api_key= "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",
    base_url="https://openrouter.ai/api/v1",
)

def chat_completion(
    system_text: str,
    user_text: str,
    *,
    #model: str = "meta-llama/llama-3.1-8b-instruct",
    model: str = "deepseek/deepseek-v3.2",
    temperature: float = 0.0,
    max_tokens: int = 800
) -> str:
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
# 1. 通用工具
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(records: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_cache(path: str) -> Dict[str, Any]:
    if USE_CACHE and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Any], path: str):
    if USE_CACHE:
        write_json(cache, path)


def jaccard_words(a: str, b: str) -> float:
    sa = set(normalize_key(a).split())
    sb = set(normalize_key(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def sentence_split(text: str) -> List[str]:
    text = str(text).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def dedup_sentences(sentences: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in sentences:
        k = normalize_key(s)
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out


# ============================================================
# 2. 读取 QMSum 文件夹
# ============================================================

def extract_transcript_text(meeting_json: Dict[str, Any]) -> str:
    for key in ["meeting_transcripts", "meeting_transcript", "transcript", "content"]:
        if key in meeting_json and isinstance(meeting_json[key], str):
            return meeting_json[key].strip()

    for key in ["meeting_transcripts", "meeting_transcript", "transcript", "utterances"]:
        if key in meeting_json and isinstance(meeting_json[key], list):
            lines = []
            for item in meeting_json[key]:
                if isinstance(item, str):
                    lines.append(item.strip())
                elif isinstance(item, dict):
                    speaker = safe_get(item, ["speaker", "participant", "role"], "")
                    text = safe_get(item, ["content", "text", "utterance", "sentence"], "")
                    speaker = str(speaker).strip()
                    text = str(text).strip()
                    if speaker and text:
                        lines.append(f"{speaker}: {text}")
                    elif text:
                        lines.append(text)
            return "\n".join(lines)

    return ""


def load_qmsum_meetings(root_dir: str) -> List[Dict[str, Any]]:
    meetings = []
    root = Path(root_dir)

    for domain_dir in root.iterdir():
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name
        if domain not in VALID_DOMAINS:
            continue

        for json_path in sorted(domain_dir.glob("*.json")):
            data = read_json(str(json_path))

            meeting_id = safe_get(
                data,
                ["meeting_id", "id", "doc_id", "file_id"],
                default=json_path.stem
            )

            transcript = extract_transcript_text(data)

            specific_queries = safe_get(
                data,
                ["specific_query_list", "specific_queries"],
                default=[]
            )

            general_queries = safe_get(
                data,
                ["general_query_list", "general_queries"],
                default=[]
            )

            meetings.append({
                "domain": domain,
                "meeting_id": meeting_id,
                "source_file": json_path.name,
                "source_path": str(json_path),
                "transcript": transcript,
                "specific_query_list": specific_queries,
                "general_query_list": general_queries
            })

    return meetings


# ============================================================
# 3. 展平 specific queries
# ============================================================

def extract_query_answer(qobj: Dict[str, Any]) -> Tuple[str, str]:
    query = safe_get(qobj, ["query", "question", "query_text"], "")
    answer = safe_get(qobj, ["answer", "summary", "gold", "reference"], "")
    return str(query).strip(), str(answer).strip()


def flatten_specific_queries(meetings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []

    for meeting in meetings:
        for i, qobj in enumerate(meeting["specific_query_list"]):
            query, answer = extract_query_answer(qobj)

            records.append({
                "domain": meeting["domain"],
                "meeting_id": meeting["meeting_id"],
                "source_file": meeting["source_file"],
                "source_path": meeting["source_path"],
                "query_id": f'{meeting["meeting_id"]}_sq_{i}',
                "query_idx_in_meeting": i,
                "query": query,
                "answer": answer,
                "transcript": meeting["transcript"]
            })

    return records


# ============================================================
# 4. LLM 打标签
# ============================================================

def build_label_prompt(query: str) -> str:
    return f"""
Task: Label the following meeting query for multi-document query summarization.

Return STRICT JSON only with these keys:
- intent
- topic
- entity

Definitions:
- intent: the main question type
- topic: the broader theme or subject
- entity: the most specific target being asked about

Allowed intent labels:
decision
discussion
problem
reason
plan
opinion
comparison
action_item
other

Guidelines:
1. topic should be broader than entity when possible.
2. entity should be concise and concrete.
3. Do not include explanation.
4. Output JSON only.

Examples:

Query: What did they decide about the remote control design?
{{
  "intent": "decision",
  "topic": "design",
  "entity": "remote control design"
}}

Query: What concerns were raised about the battery life?
{{
  "intent": "problem",
  "topic": "battery",
  "entity": "battery life"
}}

Query: Why did they reject the original proposal?
{{
  "intent": "reason",
  "topic": "proposal",
  "entity": "original proposal"
}}

Now label this query:

Query: {query}
""".strip()


def robust_json_parse(text: str, fallback: Dict[str, str]) -> Dict[str, str]:
    text = text.strip()

    try:
        obj = json.loads(text)
        return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj
        except Exception:
            pass

    return fallback


def sanitize_label_fields(label: Dict[str, Any], query: str) -> Dict[str, str]:
    allowed_intents = {
        "decision", "discussion", "problem", "reason",
        "plan", "opinion", "comparison", "action_item", "other"
    }

    intent = normalize_text(label.get("intent", "other"))
    if intent not in allowed_intents:
        intent = "other"

    topic = str(label.get("topic", "")).strip()
    entity = str(label.get("entity", "")).strip()

    if not topic:
        topic = query.strip().rstrip("?")
    if not entity:
        entity = topic

    return {
        "intent": intent,
        "topic": topic,
        "entity": entity
    }


def llm_label_query(query: str, max_retries: int = 5) -> Dict[str, str]:
    system_text = "You are a precise data labeling assistant. Return strict JSON only."
    user_text = build_label_prompt(query)

    fallback = {
        "intent": "other",
        "topic": query.strip().rstrip("?"),
        "entity": query.strip().rstrip("?")
    }

    for attempt in range(max_retries):
        try:
            text = chat_completion(
                system_text=system_text,
                user_text=user_text,
                temperature=0.0,
                max_tokens=300
            )
            label = robust_json_parse(text, fallback)
            return sanitize_label_fields(label, query)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[WARN] label failed: {query}")
                print(f"       error: {e}")
                return fallback
            time.sleep(2 * (attempt + 1))

    return fallback


def add_llm_labels(records: List[Dict[str, Any]], cache_path: str) -> List[Dict[str, Any]]:
    cache = load_cache(cache_path)
    labeled = []

    for idx, r in enumerate(records):
        query = r["query"]
        cache_key = normalize_key(query)

        if cache_key in cache:
            label = cache[cache_key]
        else:
            label = llm_label_query(query)
            cache[cache_key] = label

        rr = dict(r)
        rr["intent"] = label["intent"]
        rr["topic"] = label["topic"]
        rr["entity"] = label["entity"]
        labeled.append(rr)

        if (idx + 1) % SAVE_EVERY == 0:
            save_cache(cache, cache_path)
            print(f"[label] processed {idx + 1}/{len(records)}")

    save_cache(cache, cache_path)
    return labeled

# ============================================================
# 5. LLM group clustering within (domain, intent)
# ============================================================

MAX_GROUP_SIZE_FOR_CLUSTERING = 30   # 每次给 LLM 的 query 数量上限
ALLOWED_CLUSTER_SIZES = {2, 3, 4}


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def build_group_cluster_prompt(domain: str, intent: str, records: List[Dict[str, Any]]) -> str:
    """
    让 LLM 在同一个 (domain, intent) 组内直接做聚类
    """
    lines = []
    for r in records:
        lines.append(
            f'- query_id: {r["query_id"]}\n'
            f'  query: {r["query"]}\n'
            f'  topic: {r["topic"]}\n'
            f'  entity: {r["entity"]}\n'
            f'  meeting_id: {r["meeting_id"]}'
        )

    query_block = "\n".join(lines)

    return f"""
Task: Group the following meeting queries into mergeable clusters for multi-document query summarization.

Domain: {domain}
Intent: {intent}

Each cluster must satisfy ALL conditions:
1. All queries in the cluster have the same main intent.
2. They discuss the same or very similar topic/entity.
3. They can be naturally merged into one cross-meeting query.
4. Cluster size must be exactly 2, 3, or 4.
5. Prefer queries from different meetings.
6. Do NOT force every query into a cluster.
7. Exclude weak, vague, or unnatural clusters.

Return STRICT JSON only in this format:
{{
  "clusters": [
    {{
      "member_query_ids": ["...", "..."],
      "shared_topic": "...",
      "shared_entity": "...",
      "merged_query": "..."
    }}
  ]
}}

Queries:
{query_block}
""".strip()


def sanitize_cluster_output(obj: Dict[str, Any], valid_query_ids: set) -> Dict[str, Any]:
    raw_clusters = obj.get("clusters", [])
    clean_clusters = []
    seen_members = set()

    if not isinstance(raw_clusters, list):
        return {"clusters": []}

    for c in raw_clusters:
        if not isinstance(c, dict):
            continue

        member_query_ids = c.get("member_query_ids", [])
        if not isinstance(member_query_ids, list):
            continue

        member_query_ids = [str(x).strip() for x in member_query_ids if str(x).strip() in valid_query_ids]
        member_query_ids = list(dict.fromkeys(member_query_ids))  # 去重，保持顺序

        if len(member_query_ids) not in ALLOWED_CLUSTER_SIZES:
            continue

        # 避免完全重复 cluster
        member_key = tuple(sorted(member_query_ids))
        if member_key in seen_members:
            continue
        seen_members.add(member_key)

        shared_topic = str(c.get("shared_topic", "")).strip()
        shared_entity = str(c.get("shared_entity", "")).strip()
        merged_query = str(c.get("merged_query", "")).strip()

        clean_clusters.append({
            "member_query_ids": member_query_ids,
            "shared_topic": shared_topic,
            "shared_entity": shared_entity,
            "merged_query": merged_query
        })

    return {"clusters": clean_clusters}


def llm_cluster_group(domain: str, intent: str, records: List[Dict[str, Any]], max_retries: int = 5) -> Dict[str, Any]:
    system_text = (
        "You are a precise dataset construction assistant for multi-document query summarization. "
        "Return strict JSON only."
    )
    user_text = build_group_cluster_prompt(domain, intent, records)
    fallback = {"clusters": []}
    valid_query_ids = {r["query_id"] for r in records}

    for attempt in range(max_retries):
        try:
            text = chat_completion(
                system_text=system_text,
                user_text=user_text,
                temperature=0.0,
                max_tokens=1200
            )
            obj = robust_json_parse(text, fallback)
            return sanitize_cluster_output(obj, valid_query_ids)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[WARN] group clustering failed for domain={domain}, intent={intent}")
                print(f"       error: {e}")
                return fallback
            time.sleep(2 * (attempt + 1))

    return fallback


def cluster_queries_by_group_with_llm(records: List[Dict[str, Any]], cache_path: str) -> List[Dict[str, Any]]:
    """
    按 (domain, intent) 分组；如果组太大，则分 chunk 给 LLM 聚类。
    返回 cluster objects
    """
    cache = load_cache(cache_path)
    grouped = defaultdict(list)

    for r in records:
        grouped[(r["domain"], r["intent"])].append(r)

    all_clusters = []
    cluster_idx = 0

    for (domain, intent), group_records in grouped.items():
        # 按 meeting_id + query_id 排序，保证稳定
        group_records = sorted(group_records, key=lambda x: (x["meeting_id"], x["query_id"]))

        if len(group_records) < 2:
            continue

        # 分块，避免一次太长
        chunks = list(chunk_list(group_records, MAX_GROUP_SIZE_FOR_CLUSTERING))

        for chunk_id, chunk_records in enumerate(chunks):
            if len(chunk_records) < 2:
                continue

            cache_key = f"{domain} || {intent} || chunk_{chunk_id} || " + " | ".join(
                [r["query_id"] for r in chunk_records]
            )

            if cache_key in cache:
                result = cache[cache_key]
            else:
                result = llm_cluster_group(domain, intent, chunk_records)
                cache[cache_key] = result

            for c in result.get("clusters", []):
                member_ids = set(c["member_query_ids"])
                members = [r for r in chunk_records if r["query_id"] in member_ids]

                if len(members) not in ALLOWED_CLUSTER_SIZES:
                    continue

                meeting_ids = {m["meeting_id"] for m in members}
                if len(meeting_ids) < MIN_MEETINGS_PER_CLUSTER:
                    continue

                cluster_topic = c["shared_topic"].strip() if c["shared_topic"].strip() else members[0]["topic"]
                cluster_entity = c["shared_entity"].strip() if c["shared_entity"].strip() else members[0]["entity"]
                merged_query = c["merged_query"].strip()
                if not merged_query:
                    merged_query = default_merged_query(intent, cluster_entity, cluster_topic)

                all_clusters.append({
                    "cluster_id": f"cluster_{cluster_idx}",
                    "domain": domain,
                    "intent": intent,
                    "cluster_topic": cluster_topic,
                    "cluster_entity": cluster_entity,
                    "merged_query": merged_query,
                    "meeting_count": len(meeting_ids),
                    "size": len(members),
                    "members": members
                })
                cluster_idx += 1

        save_cache(cache, cache_path)

    return dedup_cluster_objects(all_clusters)


def dedup_cluster_objects(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去掉成员完全相同的重复 cluster
    """
    deduped = []
    seen = set()

    for c in clusters:
        key = tuple(sorted([m["query_id"] for m in c["members"]]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return deduped


# ============================================================
# 8. 构造 multi-doc sample
# ============================================================

def default_merged_query(intent: str, entity: str, topic: str) -> str:
    target = entity if entity else topic

    if intent == "decision":
        return f"What decisions were made about {target} across the meetings?"
    elif intent == "problem":
        return f"What problems or concerns were discussed about {target} across the meetings?"
    elif intent == "reason":
        return f"What reasons were given regarding {target} across the meetings?"
    elif intent == "plan":
        return f"What plans were discussed for {target} across the meetings?"
    elif intent == "action_item":
        return f"What action items were proposed regarding {target} across the meetings?"
    elif intent == "opinion":
        return f"What opinions were expressed about {target} across the meetings?"
    elif intent == "comparison":
        return f"How was {target} compared across the meetings?"
    elif intent == "discussion":
        return f"What was discussed about {target} across the meetings?"
    else:
        return f"How was {target} discussed across the meetings?"


def make_placeholder_merged_summary(cluster_members: List[Dict[str, Any]], max_sentences: int = 8) -> str:
    all_sents = []
    for r in cluster_members:
        all_sents.extend(sentence_split(r.get("answer", "")))
    all_sents = dedup_sentences(all_sents)
    return " ".join(all_sents[:max_sentences])


def build_multidoc_samples(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples = []

    for c in clusters:
        members = c["members"]

        meeting_map = {}
        for m in members:
            mid = m["meeting_id"]
            if mid not in meeting_map:
                meeting_map[mid] = {
                    "meeting_id": mid,
                    "source_file": m["source_file"],
                    "source_path": m["source_path"],
                    "transcript": m["transcript"]
                }

        samples.append({
            "sample_id": c["cluster_id"],
            "domain": c["domain"],
            "intent": c["intent"],
            "cluster_topic": c["cluster_topic"],
            "cluster_entity": c["cluster_entity"],
            "merged_query": c["merged_query"],
            "merged_summary_placeholder": make_placeholder_merged_summary(members),
            "source_meeting_ids": list(meeting_map.keys()),
            "source_meetings": list(meeting_map.values()),
            "source_queries": [
                {
                    "meeting_id": x["meeting_id"],
                    "query_id": x["query_id"],
                    "query": x["query"],
                    "answer": x["answer"],
                    "topic": x["topic"],
                    "entity": x["entity"]
                }
                for x in members
            ]
        })

    return samples


# ============================================================
# 9. 导出表格
# ============================================================

def write_csv(records: List[Dict[str, Any]], path: str):
    import csv
    if not records:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            pass
        return

    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def build_query_table(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in records:
        rows.append({
            "domain": r["domain"],
            "meeting_id": r["meeting_id"],
            "source_file": r["source_file"],
            "query_id": r["query_id"],
            "query": r["query"],
            "intent": r["intent"],
            "topic": r["topic"],
            "entity": r["entity"],
            "answer": r["answer"]
        })
    return rows


def build_pair_table(judged_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for p in judged_pairs:
        rows.append({
            "domain": p["domain"],
            "query_id_1": p["query_id_1"],
            "meeting_id_1": p["meeting_id_1"],
            "query_1": p["query_1"],
            "query_id_2": p["query_id_2"],
            "meeting_id_2": p["meeting_id_2"],
            "query_2": p["query_2"],
            "mergeable": p["mergeable"],
            "shared_intent": p["shared_intent"],
            "shared_topic": p["shared_topic"],
            "shared_entity": p["shared_entity"],
            "reason": p["reason"],
            "merged_query": p["merged_query"]
        })
    return rows


def build_cluster_table(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for c in clusters:
        for m in c["members"]:
            rows.append({
                "cluster_id": c["cluster_id"],
                "domain": c["domain"],
                "intent": c["intent"],
                "cluster_topic": c["cluster_topic"],
                "cluster_entity": c["cluster_entity"],
                "merged_query": c["merged_query"],
                "meeting_count": c["meeting_count"],
                "size": c["size"],
                "meeting_id": m["meeting_id"],
                "query_id": m["query_id"],
                "query": m["query"],
                "topic": m["topic"],
                "entity": m["entity"]
            })
    return rows


# ============================================================
# 10. 主流程
# ============================================================

def run_pipeline():
    ensure_dir(OUTPUT_DIR)

    print(f"[1] Loading meetings from: {QMSUM_ROOT}")
    meetings = load_qmsum_meetings(QMSUM_ROOT)
    print(f"    meetings loaded: {len(meetings)}")
    write_json(meetings, os.path.join(OUTPUT_DIR, "all_meetings_merged.json"))

    print("[2] Flattening specific queries")
    records = flatten_specific_queries(meetings)
    print(f"    specific queries: {len(records)}")

    print("[3] LLM labeling")
    labeled_records = add_llm_labels(records, CACHE_LABEL_PATH)
    write_jsonl(labeled_records, os.path.join(OUTPUT_DIR, "query_records_labeled.jsonl"))
    write_csv(build_query_table(labeled_records), os.path.join(OUTPUT_DIR, "query_table_labeled.csv"))

    print("[4] LLM clustering within each (domain, intent) group")
    clusters = cluster_queries_by_group_with_llm(
    labeled_records,
    cache_path=CACHE_PAIR_PATH   # 这里继续复用这个 cache 文件名也可以
    )
    print(f"    clusters: {len(clusters)}")

    write_json(clusters, os.path.join(OUTPUT_DIR, "clusters.json"))
    write_csv(build_cluster_table(clusters), os.path.join(OUTPUT_DIR, "cluster_table.csv"))

    print("[5] Building multi-document samples")
    samples = build_multidoc_samples(clusters)
    print(f"    multidoc samples: {len(samples)}")
    write_json(samples, os.path.join(OUTPUT_DIR, "multidoc_samples.json"))
    write_jsonl(samples, os.path.join(OUTPUT_DIR, "multidoc_samples.jsonl"))

   

    print("[6] Done")
    print(f"    outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()