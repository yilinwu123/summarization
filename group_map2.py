import json
import os
from pathlib import Path

# -----------------------------
# paths
# -----------------------------
CLUSTER_JSON = "/Users/yilin/Downloads/researchproject/clusters.json"
QMSUM_DIR = "/Users/yilin/Downloads/QMSum"
OUTPUT_JSON = "/Users/yilin/Downloads/researchproject/clusters_with_transcripts.json"

# -----------------------------
# load clusters
# -----------------------------
with open(CLUSTER_JSON, "r", encoding="utf-8") as f:
    clusters = json.load(f)

print(f"Loaded {len(clusters)} clusters")

# -----------------------------
# build transcript cache
# -----------------------------
transcript_cache = {}

def load_transcript(domain, meeting_id):
    key = f"{domain}/{meeting_id}"

    if key in transcript_cache:
        return transcript_cache[key]

    path = os.path.join(QMSUM_DIR, domain, f"{meeting_id}.json")

    if not os.path.exists(path):
        print(f"Warning: transcript file not found: {path}")
        transcript_cache[key] = None
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcript = data.get("meeting_transcripts", [])

    transcript_cache[key] = transcript
    return transcript


# -----------------------------
# attach transcripts
# -----------------------------
for cluster in clusters:

    for q in cluster["queries"]:

        meeting_id = q["meeting_id"]
        domain = q["domain"]

        transcript = load_transcript(domain, meeting_id)

        q["meeting_transcripts"] = transcript


print(f"Loaded {len(transcript_cache)} transcript files")

# -----------------------------
# save output
# -----------------------------
print([c["cluster_id"] for c in clusters])
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(clusters, f, indent=2, ensure_ascii=False)

print(f"Saved to {OUTPUT_JSON}")
