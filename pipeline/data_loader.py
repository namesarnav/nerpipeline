import json
import os
from typing import List
from datasets import load_dataset


def load_hf_dataset(dataset_name: str) -> List[dict]:
    ds = load_dataset(dataset_name, split="test")
    return [dict(row) for row in ds]


def save_jsonl(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
