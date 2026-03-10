"""
Evaluation: computes micro and macro F1 by comparing
predicted span strings against gold span strings.

Gold spans come from: event_expressions[i]["text"] or time_expressions[i]["text"]
Pred spans come from: the model's predicted list of strings

Matching is case-insensitive and whitespace-stripped.
"""

from typing import List, Tuple


def normalize(span: str) -> str:
    return span.lower().strip()


def normalize_set(spans: List[str]) -> set:
    return {normalize(s) for s in spans if s.strip()}


def row_counts(gold: set, pred: set) -> Tuple[int, int, int]:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return tp, fp, fn


def evaluate(
    gold_spans_all: List[List[str]],
    pred_spans_all: List[List[str]],
) -> dict:
    total_tp, total_fp, total_fn = 0, 0, 0
    per_row_f1 = []

    for gold_raw, pred_raw in zip(gold_spans_all, pred_spans_all):
        gold = normalize_set(gold_raw)
        pred = normalize_set(pred_raw)

        tp, fp, fn = row_counts(gold, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-row F1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_row_f1.append(f1)

    # Micro — global counts
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # Macro — average of per-row F1
    macro_f1 = sum(per_row_f1) / len(per_row_f1) if per_row_f1 else 0.0

    return {
        "total_rows": len(gold_spans_all),
        "micro_precision": round(micro_p,  4),
        "micro_recall":    round(micro_r,  4),
        "micro_f1":        round(micro_f1, 4),
        "macro_f1":        round(macro_f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def print_results(results: dict, logger):
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Rows evaluated : {results['total_rows']}")
    logger.info(f"  TP / FP / FN   : {results['tp']} / {results['fp']} / {results['fn']}")
    logger.info("-" * 50)
    logger.info(f"  Micro Precision : {results['micro_precision']:.4f}")
    logger.info(f"  Micro Recall    : {results['micro_recall']:.4f}")
    logger.info(f"  Micro F1        : {results['micro_f1']:.4f}")
    logger.info("-" * 50)
    logger.info(f"  Macro F1        : {results['macro_f1']:.4f}")
    logger.info("=" * 50)
