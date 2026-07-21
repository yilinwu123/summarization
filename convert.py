import pandas as pd
import json
 
def convert(parquet_path, json_path):
    df = pd.read_parquet(parquet_path)
    # numpy array -> list，方便 json 序列化
    if "options" in df.columns:
        df["options"] = df["options"].apply(lambda x: list(x))
    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"{parquet_path} -> {json_path}  ({len(records)} 条)")
 
if __name__ == "__main__":
    convert("test.parquet", "mmlu_pro_test.json")
    convert("validation.parquet", "mmlu_pro_validation.json")