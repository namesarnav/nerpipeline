#!/usr/bin/env python3
"""
NER Inference + Evaluation Pipeline

Usage:
    python run.py --model <hf_model> --dataset <hf_dataset> --task <event|time>

To run : 

    python run.py \
        --model bert-base-uncased \
        --dataset conll2003 \
        --task event \
        --num_samples 400

"""

import argparse
import os
import json
from pipeline.loader import load_hf_dataset, save_jsonl
from pipeline.inference import NERInference
from pipeline.evaluate import evaluate, print_results
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--dataset",    type=str, required=True,
                        help="HuggingFace dataset name (test split is used)")
    parser.add_argument("--task",       type=str, required=True, choices=["event", "time"],
                        help="'event' or 'time'")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output JSONL path (default: outputs/<dataset>__<model>__predictions.jsonl)")
    parser.add_argument("--num_samples", type=int, default=400,
                        help="Maximum number of samples to process from test split (default: 400)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device",     type=str, default="auto")
    return parser.parse_args()


def resolve_output_path(args):
    if args.output:
        return args.output
    os.makedirs("outputs", exist_ok=True)
    dataset_slug = args.dataset.replace("/", "_")
    model_slug   = args.model.replace("/", "_")
    return os.path.join("outputs", f"{dataset_slug}__{model_slug}__predictions.jsonl")


def main():
    args   = parse_args()
    logger = setup_logger()

    # Field names to match evaluate.py expectations
    if args.task == "event":
        gold_field   = "gold_event_expressions"
        pred_field   = "post_processed_event_expressions"
        # For datasets that use different field names
        source_gold_field = "event_expressions"
    else:  # time
        gold_field   = "gold_timex_expressions"
        pred_field   = "post_processed_timex_expressions"
        source_gold_field = "time_expressions"

    logger.info("=" * 60)
    logger.info("NER INFERENCE PIPELINE")
    logger.info(f"  Model       : {args.model}")
    logger.info(f"  Dataset     : {args.dataset}  (split=test)")
    logger.info(f"  Task        : {args.task}")
    logger.info(f"  Max samples : {args.num_samples}")
    logger.info(f"  Gold field  : {gold_field}")
    logger.info(f"  Pred field  : {pred_field}")
    logger.info("=" * 60)

    # Load test split from HuggingFace
    logger.info(f"\nLoading dataset: {args.dataset}")
    rows = load_hf_dataset(args.dataset)
    
    # Handle datasets with fewer samples
    total_available = len(rows)
    num_samples = min(args.num_samples, total_available)
    
    if total_available < args.num_samples:
        logger.warning(f"Dataset only has {total_available} test samples, processing all instead of {args.num_samples}")
    
    rows = rows[:num_samples]
    logger.info(f"Processing {len(rows)} rows from test split.")

    # Load model
    inference = NERInference(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        logger=logger,
    )

    # Run inference
    texts = [row["text"] for row in rows]
    logger.info(f"\nRunning inference on {len(texts)} texts...")
    all_predicted_spans = inference.predict_batch(texts)

    # Build output rows with correct field names
    gold_spans_all = []
    output_rows    = []
    
    for row, predicted_spans in zip(rows, all_predicted_spans):
        # Extract gold spans from structured fields
        # Dataset has: time_expressions or event_expressions as list of dicts with "text" field
        gold_spans = []
        
        if source_gold_field in row and row[source_gold_field]:
            field_data = row[source_gold_field]
            if isinstance(field_data, list) and len(field_data) > 0:
                # If it's a list of dicts with 'text' field (your dataset format)
                if isinstance(field_data[0], dict):
                    gold_spans = [item.get("text", "").strip() for item in field_data if item.get("text")]
                else:
                    # If it's already a list of strings
                    gold_spans = [str(s).strip() for s in field_data if s]
        
        gold_spans_all.append(gold_spans)

        # Create output row with standardized field names
        output_row = {
            "text": row["text"],
            gold_field: gold_spans,
            pred_field: predicted_spans
        }
        output_rows.append(output_row)

    # Evaluate
    logger.info("\nEvaluating...")
    results = evaluate(gold_spans_all, all_predicted_spans)
    print_results(results, logger)

    # Save predictions
    output_path = resolve_output_path(args)
    save_jsonl(output_rows, output_path)
    
    # APPEND FINAL METRICS TO THE SAME FILE
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "FINAL_METRICS": {
                "total_rows": results.get("total_rows", len(output_rows)),
                "micro_precision": results.get("micro_precision", 0.0),
                "micro_recall": results.get("micro_recall", 0.0),
                "micro_f1": results.get("micro_f1", 0.0),
                "macro_f1": results.get("macro_f1", 0.0),
                "tp": results.get("tp", 0),
                "fp": results.get("fp", 0),
                "fn": results.get("fn", 0),
            }
        }) + "\n")
    
    # SAVE METRICS TO SEPARATE FILE
    metrics_file = output_path.replace(".jsonl", "_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_rows": results.get("total_rows", len(output_rows)),
            "micro_precision": results.get("micro_precision", 0.0),
            "micro_recall": results.get("micro_recall", 0.0),
            "micro_f1": results.get("micro_f1", 0.0),
            "macro_f1": results.get("macro_f1", 0.0),
            "tp": results.get("tp", 0),
            "fp": results.get("fp", 0),
            "fn": results.get("fn", 0),
        }, f, indent=2)
    
    logger.info(f"\nOutput saved → {output_path}")
    logger.info(f"  - Rows 1-{len(output_rows)}: Individual predictions")
    logger.info(f"  - Row {len(output_rows)+1}: FINAL_METRICS")
    logger.info(f"Metrics saved → {metrics_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()