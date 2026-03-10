#!/usr/bin/env python3
"""
Evaluation Script
* this is not the same as other evaluate in pipeline dir
Computes micro and macro F1 scores by comparing gold vs predicted spans.

Usage -> python evaluate.py --input <predictions.jsonl> --task <event|time>

Examples:
    python evaluate.py --input outputs/event_test__bert__predictions.jsonl --task event
    python evaluate.py --input outputs/time_test__t5__predictions.jsonl --task time
"""

import argparse
import json
import os
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NER predictions against gold labels")
    
    parser.add_argument("--input", type=str, required=True,
                        help="Path to predictions JSONL (output of run.py)")
    
    parser.add_argument("--task", type=str, required=True, choices=["event", "time"],
                        help="Task type: event or time")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save evaluation results as JSON")
    
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Lowercase + strip spans before comparing (default: True)")
    
    return parser.parse_args()


def normalize(span: str, do_normalize: bool) -> str:
    if do_normalize:
        return span.lower().strip()
    
    return span.strip()

def normalize_set(spans: List[str], do_normalize: bool) -> set:
    return {normalize(s, do_normalize) for s in spans if s.strip()}

def row_counts(gold: set, pred: set) -> Tuple[int, int, int]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return tp, fp, fn

def row_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1

def evaluate(rows: List[dict], gold_field: str, pred_field: str, do_normalize: bool) -> dict:
    # For micro -> accumulate TP/FP/FN globally
    total_tp, total_fp, total_fn = 0, 0, 0

    # For macro: collect per row F1
    per_row_f1 = []

    skipped = 0

    for row in rows:
        gold_raw = row.get(gold_field, [])
        pred_raw = row.get(pred_field, [])

        if gold_raw is None or pred_raw is None:
            skipped += 1
            continue

        gold = normalize_set(gold_raw, do_normalize)
        pred = normalize_set(pred_raw, do_normalize)

        tp, fp, fn = row_counts(gold, pred)
        total_tp  += tp
        total_fp  += fp
        total_fn  += fn

        _, _, f1 = row_f1(tp, fp, fn)
        per_row_f1.append(f1)

    # Micro F1 — global TP/FP/FN
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)

    # Macro F1 — average of per-row F1
    macro_f1 = sum(per_row_f1) / len(per_row_f1) if per_row_f1 else 0.0

    return {
        "num_rows":      len(rows),
        "skipped_rows":  skipped,
        "evaluated_rows": len(per_row_f1),
        "micro": {
            "precision": round(micro_p,  4),
            "recall":    round(micro_r,  4),
            "f1":        round(micro_f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "macro": {
            "f1": round(macro_f1, 4),
        },
    }


def print_results(results: dict, gold_field: str, pred_field: str):
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Gold field : {gold_field}")
    print(f"  Pred field : {pred_field}")
    print(f"  Rows       : {results['evaluated_rows']} evaluated / {results['num_rows']} total")
    if results["skipped_rows"] > 0:
        print(f"  Skipped    : {results['skipped_rows']} (missing gold or pred field)")
    print("-" * 50)
    m = results["micro"]
    print(f"  Micro Precision : {m['precision']:.4f}")
    print(f"  Micro Recall    : {m['recall']:.4f}")
    print(f"  Micro F1        : {m['f1']:.4f}   (TP={m['tp']}  FP={m['fp']}  FN={m['fn']})")
    print("-" * 50)
    print(f"  Macro F1        : {results['macro']['f1']:.4f}")
    print("=" * 50 + "\n")


def main():
    args = parse_args()

    gold_field = (
        "gold_event_expressions" if args.task == "event"
        else "gold_timex_expressions"
    )
    pred_field = (
        "post_processed_event_expressions" if args.task == "event"
        else "post_processed_timex_expressions"
    )

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} rows from {args.input}")

    results = evaluate(rows, gold_field, pred_field, do_normalize=args.normalize)
    print_results(results, gold_field, pred_field)

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
