import json
import os
from typing import List


def load_local_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_hf_dataset(dataset_name: str) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="test")
    return [dict(row) for row in ds]


def save_jsonl(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
