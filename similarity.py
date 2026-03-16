import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer, CrossEncoder, util


# =========================================================
# 1. 配置
# =========================================================
INPUT_CSV = "/Users/yilin/Downloads/researchproject/query_table_labeled.csv"
OUTPUT_DIR = "/Users/yilin/Downloads/researchproject/"

BI_ENCODER_MODEL = "all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/stsb-roberta-large"

# 阈值，可按需要调整
LEXICAL_EXACT_THRESHOLD = 0.98
BI_ENCODER_LOW_THRESHOLD = 0.50
CROSS_EXACT_THRESHOLD = 0.90
CROSS_SIMILAR_THRESHOLD = 0.70

# 是否对 cross-encoder 做剪枝：
# True  = 只有 bi-encoder 分数 >= BI_ENCODER_LOW_THRESHOLD 才跑 cross-encoder
# False = 所有 pair 都跑 cross-encoder（更慢）
USE_CROSS_ENCODER_PRUNING = True

# cross-encoder batch size
CROSS_BATCH_SIZE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. 读取数据
# =========================================================
df = pd.read_csv(INPUT_CSV)

required_cols = ["query", "domain", "meeting_id", "source_file", "query_id"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"query_table.csv 缺少必要列: {missing_cols}")

# 去掉 query 为空的行
df = df.dropna(subset=["query"]).copy()
df["query"] = df["query"].astype(str).str.strip()
df = df[df["query"] != ""].reset_index(drop=True)

print(f"Loaded {len(df)} queries from {INPUT_CSV}")


# =========================================================
# 3. 加载模型
# =========================================================
print("Loading Bi-Encoder...")
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)

print("Loading Cross-Encoder...")
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)


# =========================================================
# 4. 一次性编码所有 query
# =========================================================
queries = df["query"].tolist()

print("Encoding all queries with Bi-Encoder...")
embeddings = bi_encoder.encode(
    queries,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Computing cosine similarity matrix...")
cosine_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()


# =========================================================
# 5. 构造所有 pair
# =========================================================
pair_indices = list(itertools.combinations(range(len(df)), 2))
print(f"Total pairs: {len(pair_indices)}")


# =========================================================
# 6. 第一阶段：计算 lexical / bi-encoder 分数，决定哪些需要 cross-encoder
# =========================================================
records = []
cross_needed_pairs = []
cross_needed_inputs = []

print("Stage 1: lexical + bi-encoder scoring...")
for i, j in tqdm(pair_indices):
    row_i = df.iloc[i]
    row_j = df.iloc[j]

    q_i = row_i["query"]
    q_j = row_j["query"]

    lexical_score = fuzz.token_sort_ratio(q_i.lower(), q_j.lower()) / 100.0
    bi_score = float(cosine_matrix[i, j])

    pair_id = f"{row_i['query_id']}_{row_j['query_id']}"

    same_meeting = row_i["meeting_id"] == row_j["meeting_id"]
    same_domain = row_i["domain"] == row_j["domain"]

    # 默认值
    cross_score = np.nan
    final_score = None
    relation = None

    # 先做一个 quick decision
    if lexical_score >= LEXICAL_EXACT_THRESHOLD:
        final_score = 1.0
        relation = "完全一样 (Exact Match)"
    elif bi_score < BI_ENCODER_LOW_THRESHOLD and USE_CROSS_ENCODER_PRUNING:
        final_score = bi_score
        relation = "相差很大 (Completely Different)"
    else:
        # 留待第二阶段 cross-encoder 计算
        cross_needed_pairs.append(len(records))
        cross_needed_inputs.append([q_i, q_j])

    rec = {
        "pair_id": pair_id,

        "query_a": q_i,
        "query_b": q_j,

        "meeting_id_a": row_i["meeting_id"],
        "query_id_a": row_i["query_id"],
        "domain_a": row_i["domain"],
        "source_file_a": row_i["source_file"],

        "meeting_id_b": row_j["meeting_id"],
        "query_id_b": row_j["query_id"],
        "domain_b": row_j["domain"],
        "source_file_b": row_j["source_file"],

        "same_meeting": same_meeting,
        "same_domain": same_domain,

        "lexical_score": lexical_score,
        "bi_encoder_score": bi_score,
        "cross_encoder_score": cross_score,
        "final_score": final_score,
        "relation": relation,
    }
    records.append(rec)

print(f"Pairs requiring Cross-Encoder: {len(cross_needed_inputs)}")


# =========================================================
# 7. 第二阶段：批量跑 cross-encoder
# =========================================================
if len(cross_needed_inputs) > 0:
    print("Stage 2: cross-encoder scoring...")
    all_cross_scores = []

    for start in tqdm(range(0, len(cross_needed_inputs), CROSS_BATCH_SIZE)):
        batch_pairs = cross_needed_inputs[start:start + CROSS_BATCH_SIZE]
        batch_scores = cross_encoder.predict(batch_pairs)
        batch_scores = np.array(batch_scores).reshape(-1)
        all_cross_scores.extend(batch_scores.tolist())

    # 回填结果
    for rec_idx, score in zip(cross_needed_pairs, all_cross_scores):
        score = float(score)
        records[rec_idx]["cross_encoder_score"] = score
        records[rec_idx]["final_score"] = score

        if score >= CROSS_EXACT_THRESHOLD:
            relation = "语义完全一样 (Exact Semantic Match)"
        elif score >= CROSS_SIMILAR_THRESHOLD:
            relation = "语义相似 (Similar)"
        else:
            relation = "相差很大 (Completely Different)"

        records[rec_idx]["relation"] = relation


# =========================================================
# 8. 保存 pair 级别结果
# =========================================================
pair_df = pd.DataFrame(records)

pair_csv = os.path.join(OUTPUT_DIR, "query_pair_similarity.csv")
pair_df.to_csv(pair_csv, index=False, encoding="utf-8-sig")
print(f"Saved pair similarity results to: {pair_csv}")


# =========================================================
# 9. 各种统计分布
# =========================================================

# 9.1 same_meeting / diff_meeting 总体数量
meeting_same_diff = (
    pair_df.groupby("same_meeting")
    .size()
    .reset_index(name="count")
)
meeting_same_diff["type"] = meeting_same_diff["same_meeting"].map({
    True: "same_meeting",
    False: "different_meeting"
})
meeting_same_diff = meeting_same_diff[["type", "count"]]

meeting_same_diff_csv = os.path.join(OUTPUT_DIR, "distribution_same_vs_different_meeting.csv")
meeting_same_diff.to_csv(meeting_same_diff_csv, index=False, encoding="utf-8-sig")

# 9.2 same_domain / diff_domain 总体数量
domain_same_diff = (
    pair_df.groupby("same_domain")
    .size()
    .reset_index(name="count")
)
domain_same_diff["type"] = domain_same_diff["same_domain"].map({
    True: "same_domain",
    False: "different_domain"
})
domain_same_diff = domain_same_diff[["type", "count"]]

domain_same_diff_csv = os.path.join(OUTPUT_DIR, "distribution_same_vs_different_domain.csv")
domain_same_diff.to_csv(domain_same_diff_csv, index=False, encoding="utf-8-sig")

# 9.3 每个 meeting 内部的 pair 数量
same_meeting_pairs = pair_df[pair_df["same_meeting"]].copy()
per_meeting_internal = (
    same_meeting_pairs.groupby("meeting_id_a")
    .size()
    .reset_index(name="pair_count")
    .rename(columns={"meeting_id_a": "meeting_id"})
    .sort_values("pair_count", ascending=False)
)

per_meeting_internal_csv = os.path.join(OUTPUT_DIR, "distribution_per_meeting_internal_pairs.csv")
per_meeting_internal.to_csv(per_meeting_internal_csv, index=False, encoding="utf-8-sig")

# 9.4 不同 meeting 之间的 pair 数量（meeting × meeting）
pair_df["meeting_pair"] = pair_df.apply(
    lambda x: " || ".join(sorted([str(x["meeting_id_a"]), str(x["meeting_id_b"])])),
    axis=1
)

meeting_by_meeting = (
    pair_df.groupby("meeting_pair")
    .size()
    .reset_index(name="pair_count")
    .sort_values("pair_count", ascending=False)
)

# 拆成两列
meeting_by_meeting[["meeting_1", "meeting_2"]] = meeting_by_meeting["meeting_pair"].str.split(" \\|\\| ", expand=True)
meeting_by_meeting = meeting_by_meeting[["meeting_1", "meeting_2", "pair_count"]]

meeting_by_meeting_csv = os.path.join(OUTPUT_DIR, "distribution_meeting_by_meeting.csv")
meeting_by_meeting.to_csv(meeting_by_meeting_csv, index=False, encoding="utf-8-sig")

# 9.5 每个 domain 内部的 pair 数量
same_domain_pairs = pair_df[pair_df["same_domain"]].copy()
per_domain_internal = (
    same_domain_pairs.groupby("domain_a")
    .size()
    .reset_index(name="pair_count")
    .rename(columns={"domain_a": "domain"})
    .sort_values("pair_count", ascending=False)
)

per_domain_internal_csv = os.path.join(OUTPUT_DIR, "distribution_per_domain_internal_pairs.csv")
per_domain_internal.to_csv(per_domain_internal_csv, index=False, encoding="utf-8-sig")

# 9.6 不同 domain 之间 pair 数量（domain × domain）
pair_df["domain_pair"] = pair_df.apply(
    lambda x: " || ".join(sorted([str(x["domain_a"]), str(x["domain_b"])])),
    axis=1
)

domain_by_domain = (
    pair_df.groupby("domain_pair")
    .size()
    .reset_index(name="pair_count")
    .sort_values("pair_count", ascending=False)
)

domain_by_domain[["domain_1", "domain_2"]] = domain_by_domain["domain_pair"].str.split(" \\|\\| ", expand=True)
domain_by_domain = domain_by_domain[["domain_1", "domain_2", "pair_count"]]

domain_by_domain_csv = os.path.join(OUTPUT_DIR, "distribution_domain_by_domain.csv")
domain_by_domain.to_csv(domain_by_domain_csv, index=False, encoding="utf-8-sig")


# =========================================================
# 10. 再做一个更直观的 summary
# =========================================================
summary_rows = []

summary_rows.append({
    "metric": "num_queries",
    "value": len(df)
})
summary_rows.append({
    "metric": "num_pairs",
    "value": len(pair_df)
})
summary_rows.append({
    "metric": "same_meeting_pairs",
    "value": int((pair_df["same_meeting"] == True).sum())
})
summary_rows.append({
    "metric": "different_meeting_pairs",
    "value": int((pair_df["same_meeting"] == False).sum())
})
summary_rows.append({
    "metric": "same_domain_pairs",
    "value": int((pair_df["same_domain"] == True).sum())
})
summary_rows.append({
    "metric": "different_domain_pairs",
    "value": int((pair_df["same_domain"] == False).sum())
})
summary_rows.append({
    "metric": "exact_match_pairs",
    "value": int((pair_df["relation"] == "完全一样 (Exact Match)").sum())
})
summary_rows.append({
    "metric": "exact_semantic_match_pairs",
    "value": int((pair_df["relation"] == "语义完全一样 (Exact Semantic Match)").sum())
})
summary_rows.append({
    "metric": "similar_pairs",
    "value": int((pair_df["relation"] == "语义相似 (Similar)").sum())
})
summary_rows.append({
    "metric": "completely_different_pairs",
    "value": int((pair_df["relation"] == "相差很大 (Completely Different)").sum())
})

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUT_DIR, "summary_stats.csv")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")


# =========================================================
# 11. 打印简要结果
# =========================================================
print("\n===== Summary =====")
print(summary_df.to_string(index=False))

print("\nSaved files:")
print(f"1. {pair_csv}")
print(f"2. {meeting_same_diff_csv}")
print(f"3. {domain_same_diff_csv}")
print(f"4. {per_meeting_internal_csv}")
print(f"5. {meeting_by_meeting_csv}")
print(f"6. {per_domain_internal_csv}")
print(f"7. {domain_by_domain_csv}")
print(f"8. {summary_csv}")


# =========================================================
# 12. 按相似度从高到低排序
# =========================================================

pair_df_sorted = pair_df.sort_values(
    by="final_score",
    ascending=False
)

sorted_csv = os.path.join(OUTPUT_DIR, "query_pair_similarity_sorted.csv")
pair_df_sorted.to_csv(sorted_csv, index=False, encoding="utf-8-sig")

print(f"Saved sorted similarity results to: {sorted_csv}")

# =========================================================
# 13. 相似度分布统计 (interval = 0.05)
# =========================================================

# 创建分箱
bins = np.arange(0, 1.05, 0.05)

pair_df["score_bin"] = pd.cut(
    pair_df["final_score"],
    bins=bins,
    include_lowest=True
)

distribution = (
    pair_df.groupby("score_bin")
    .size()
    .reset_index(name="count")
    .sort_values("score_bin", ascending=False)
)

distribution_csv = os.path.join(OUTPUT_DIR, "similarity_distribution_0.05_bins.csv")
distribution.to_csv(distribution_csv, index=False, encoding="utf-8-sig")

print(f"Saved similarity distribution to: {distribution_csv}")