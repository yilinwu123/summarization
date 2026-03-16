import os
import math
import pandas as pd
import numpy as np
from collections import defaultdict

# =========================================================
# 1. 配置
# =========================================================
INPUT_CSV = "/Users/yilin/Downloads/researchproject/query_pair_similarity_sorted.csv"
OUTPUT_DIR = "/Users/yilin/Downloads/researchproject/"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "aggregated_similarity_groups.csv")

WINDOW_SIZE = 50
MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 10
TARGET_PER_SIZE = 500

# 是否保留原始 pair 作为 size=2 group
INCLUDE_PAIR_GROUPS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. 读取 pair 数据
# =========================================================
df = pd.read_csv(INPUT_CSV)

required_cols = [
    "pair_id",
    "query_id_a", "query_id_b",
    "meeting_id_a", "meeting_id_b",
    "domain_a", "domain_b",
    "final_score"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"输入文件缺少必要字段: {missing}")

# 按分数从高到低确保排序
df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

print(f"Loaded {len(df)} pair rows from {INPUT_CSV}")


# =========================================================
# 3. 建立 query 节点属性表
# =========================================================
query_info = {}

for _, row in df.iterrows():
    qa = str(row["query_id_a"])
    qb = str(row["query_id_b"])

    if qa not in query_info:
        query_info[qa] = {
            "meeting_id": str(row["meeting_id_a"]),
            "domain": str(row["domain_a"])
        }

    if qb not in query_info:
        query_info[qb] = {
            "meeting_id": str(row["meeting_id_b"]),
            "domain": str(row["domain_b"])
        }

print(f"Unique queries found: {len(query_info)}")


# =========================================================
# 4. 建立图结构
#    neighbors[q1][q2] = score
# =========================================================
neighbors = defaultdict(dict)

for _, row in df.iterrows():
    qa = str(row["query_id_a"])
    qb = str(row["query_id_b"])
    score = float(row["final_score"])

    neighbors[qa][qb] = score
    neighbors[qb][qa] = score


# =========================================================
# 5. 工具函数
# =========================================================
def canonical_group_id(query_ids):
    """组id：按排序后的 query_id 拼接"""
    ids = sorted([str(x) for x in query_ids])
    return "_".join(ids)


def get_group_meta(query_ids, query_info):
    """判断 same_meeting / same_domain"""
    query_ids = list(query_ids)
    meetings = [query_info[q]["meeting_id"] for q in query_ids]
    domains = [query_info[q]["domain"] for q in query_ids]

    same_meeting = len(set(meetings)) == 1
    same_domain = len(set(domains)) == 1

    return meetings, domains, same_meeting, same_domain


def all_edges_in_group(query_ids, neighbors):
    """
    返回组内所有存在于图中的边及其分数
    edge 格式: (a, b, score), 其中 a < b
    """
    ids = sorted(query_ids)
    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if b in neighbors[a]:
                edges.append((a, b, float(neighbors[a][b])))
    return edges


def group_similarity_from_edges(query_ids, neighbors):
    """
    组相似度 = 参与获得该组的相似对相似值平均值
    这里用组内实际存在的所有pair边的平均值
    """
    edges = all_edges_in_group(query_ids, neighbors)
    if len(edges) == 0:
        return None
    return float(np.mean([e[2] for e in edges]))


def expand_seed_group(seed_queries, seed_edges, neighbors, max_size=10):
    """
    从 seed group 出发，基于间接关系逐步扩展
    策略：
    - 候选节点必须与当前组内至少一个节点相连
    - 优先选择与当前组连接强度最高的节点
    - 直到达到 max_size 或无法扩展
    返回多个前缀组：size=当前最小size ... max_size
    """
    current_group = set(seed_queries)
    current_edges = list(seed_edges)

    # 保存每个扩展阶段的组
    group_snapshots = []

    while len(current_group) < max_size:
        candidates = defaultdict(list)

        for q in current_group:
            for nb, score in neighbors[q].items():
                if nb not in current_group:
                    candidates[nb].append(score)

        if not candidates:
            break

        # 计算候选与当前组的综合连接强度
        # 先看平均连接分，再看连接数，再看最大边
        candidate_rank = []
        for nb, scores in candidates.items():
            avg_score = float(np.mean(scores))
            degree_to_group = len(scores)
            max_score = float(np.max(scores))
            candidate_rank.append((nb, avg_score, degree_to_group, max_score))

        # 排序：平均分高优先，连接数多优先，最大边高优先
        candidate_rank = sorted(
            candidate_rank,
            key=lambda x: (x[1], x[2], x[3], x[0]),
            reverse=True
        )

        best_nb = candidate_rank[0][0]
        current_group.add(best_nb)

        # 把新加入节点与当前组已有节点的边纳入
        current_ids = sorted(current_group)
        new_edges = []
        for q in current_ids:
            if q == best_nb:
                continue
            a, b = sorted([q, best_nb])
            if b in neighbors[a]:
                new_edges.append((a, b, float(neighbors[a][b])))

        current_edges.extend(new_edges)

        group_snapshots.append({
            "query_ids": sorted(current_group),
            "edges": list(current_edges)
        })

    return group_snapshots


def build_group_record(query_ids, neighbors, query_info):
    query_ids = sorted(query_ids)
    group_id = canonical_group_id(query_ids)
    group_size = len(query_ids)

    sim = group_similarity_from_edges(query_ids, neighbors)
    if sim is None:
        return None

    meetings, domains, same_meeting, same_domain = get_group_meta(query_ids, query_info)

    return {
        "group_id": group_id,
        "group_size": group_size,
        "group_similarity": sim,
        "member_query_ids": "|".join(query_ids),
        "member_meeting_ids": "|".join(meetings),
        "member_domains": "|".join(domains),
        "same_meeting": same_meeting,
        "same_domain": same_domain
    }


# =========================================================
# 6. 先加入原始 pair 作为 size=2 group
# =========================================================
all_group_records = []
seen_group_ids = set()

if INCLUDE_PAIR_GROUPS:
    print("Adding original pairs as size=2 groups...")
    for _, row in df.iterrows():
        query_ids = [str(row["query_id_a"]), str(row["query_id_b"])]
        record = build_group_record(query_ids, neighbors, query_info)
        if record is None:
            continue
        gid = record["group_id"]
        if gid not in seen_group_ids:
            all_group_records.append(record)
            seen_group_ids.add(gid)

print(f"Initial size=2 groups: {sum(1 for x in all_group_records if x['group_size']==2)}")


# =========================================================
# 7. 用滑动窗口生成 size=3~10 的候选组
# =========================================================
# 非重：不同 size>=3~10 的最终组之间不共享 query_id
# 这里 pair(size=2) 不参与“占用”，否则会极大限制后续组生成
# 如你想 pair 也占用，可改成 used_query_ids_final 初始化为所有pair节点
# =========================================================
used_query_ids_final = set()
selected_counts = {k: 0 for k in range(MIN_GROUP_SIZE, MAX_GROUP_SIZE + 1)}

candidate_groups_by_size = {k: [] for k in range(MIN_GROUP_SIZE, MAX_GROUP_SIZE + 1)}

print("Generating candidate groups from sliding windows...")

n = len(df)
for start in range(0, n):
    end = min(start + WINDOW_SIZE, n)
    window_df = df.iloc[start:end]

    # 用这个窗口内的边作为种子
    for _, row in window_df.iterrows():
        qa = str(row["query_id_a"])
        qb = str(row["query_id_b"])
        score = float(row["final_score"])

        seed_queries = [qa, qb]
        seed_edges = [(min(qa, qb), max(qa, qb), score)]

        expanded = expand_seed_group(
            seed_queries=seed_queries,
            seed_edges=seed_edges,
            neighbors=neighbors,
            max_size=MAX_GROUP_SIZE
        )

        for g in expanded:
            query_ids = sorted(g["query_ids"])
            size = len(query_ids)

            if size < MIN_GROUP_SIZE or size > MAX_GROUP_SIZE:
                continue

            gid = canonical_group_id(query_ids)
            if gid in seen_group_ids:
                continue

            # 先只存候选，不立刻选
            sim = group_similarity_from_edges(query_ids, neighbors)
            if sim is None:
                continue

            meetings, domains, same_meeting, same_domain = get_group_meta(query_ids, query_info)

            candidate_groups_by_size[size].append({
                "group_id": gid,
                "group_size": size,
                "group_similarity": sim,
                "member_query_ids": "|".join(query_ids),
                "member_meeting_ids": "|".join(meetings),
                "member_domains": "|".join(domains),
                "same_meeting": same_meeting,
                "same_domain": same_domain
            })


# =========================================================
# 8. 候选去重
# =========================================================
print("Deduplicating candidate groups...")

for size in range(MIN_GROUP_SIZE, MAX_GROUP_SIZE + 1):
    dedup = {}
    for rec in candidate_groups_by_size[size]:
        gid = rec["group_id"]
        # 若重复，保留相似度更高的
        if gid not in dedup or rec["group_similarity"] > dedup[gid]["group_similarity"]:
            dedup[gid] = rec
    candidate_groups_by_size[size] = list(dedup.values())

    # 按组相似度从高到低排序
    candidate_groups_by_size[size] = sorted(
        candidate_groups_by_size[size],
        key=lambda x: x["group_similarity"],
        reverse=True
    )

    print(f"size={size} unique candidates: {len(candidate_groups_by_size[size])}")


# =========================================================
# 9. 选择最终非重组
#    各 size 目标 500 个
# =========================================================
print("Selecting non-overlapping final groups...")

for size in range(MIN_GROUP_SIZE, MAX_GROUP_SIZE + 1):
    target = TARGET_PER_SIZE
    selected = 0

    for rec in candidate_groups_by_size[size]:
        if selected >= target:
            break

        qids = rec["member_query_ids"].split("|")

        # 非重要求：和之前已选 group 不共享 query_id
        if any(q in used_query_ids_final for q in qids):
            continue

        all_group_records.append(rec)
        seen_group_ids.add(rec["group_id"])
        used_query_ids_final.update(qids)

        selected += 1
        selected_counts[size] += 1

    print(f"size={size}: selected {selected}")


# =========================================================
# 10. 最终排序并保存
# =========================================================
result_df = pd.DataFrame(all_group_records)

# 排序：先按 group_size，再按 group_similarity 从高到低
result_df = result_df.sort_values(
    by=["group_size", "group_similarity"],
    ascending=[True, False]
).reset_index(drop=True)

result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\nSaved grouped results to: {OUTPUT_CSV}")


# =========================================================
# 11. 额外统计输出
# =========================================================
summary_rows = []

for size in range(2, MAX_GROUP_SIZE + 1):
    cnt = int((result_df["group_size"] == size).sum())
    summary_rows.append({
        "group_size": size,
        "count": cnt
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUT_DIR, "aggregated_group_size_distribution.csv")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

same_meeting_dist = (
    result_df.groupby(["group_size", "same_meeting"])
    .size()
    .reset_index(name="count")
)
same_meeting_csv = os.path.join(OUTPUT_DIR, "aggregated_same_meeting_distribution.csv")
same_meeting_dist.to_csv(same_meeting_csv, index=False, encoding="utf-8-sig")

same_domain_dist = (
    result_df.groupby(["group_size", "same_domain"])
    .size()
    .reset_index(name="count")
)
same_domain_csv = os.path.join(OUTPUT_DIR, "aggregated_same_domain_distribution.csv")
same_domain_dist.to_csv(same_domain_csv, index=False, encoding="utf-8-sig")

print("\nSelected counts for size 3-10:")
for size in range(MIN_GROUP_SIZE, MAX_GROUP_SIZE + 1):
    print(f"size={size}: {selected_counts[size]}")

print("\nAdditional files:")
print(summary_csv)
print(same_meeting_csv)
print(same_domain_csv)