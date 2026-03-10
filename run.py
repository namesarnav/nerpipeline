#!/usr/bin/env python3
"""
NER Inference + Evaluation Pipeline

Usage:
    python run.py --model <hf_model> --dataset <hf_dataset> --task <eventx|timex>

Example:
    python run.py \
        --model mdg-nlp/event-ner-bert-base-cased \
        --dataset mdg-nlp/eventx-recognition-perturbed \
        --task eventx
"""

import argparse
import os
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
    parser.add_argument("--task",       type=str, required=True, choices=["eventx", "timex"],
                        help="'eventx' or 'timex'")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output JSONL path (default: outputs/<dataset>__<model>__predictions.jsonl)")
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

    gold_field   = "event_expressions"          if args.task == "eventx" else "time_expressions"
    output_field = "predicted_event_expressions" if args.task == "eventx" else "predicted_time_expressions"

    logger.info("=" * 60)
    logger.info("NER INFERENCE PIPELINE")
    logger.info(f"  Model       : {args.model}")
    logger.info(f"  Dataset     : {args.dataset}  (split=test)")
    logger.info(f"  Task        : {args.task}")
    logger.info(f"  Gold field  : {gold_field}[].text")
    logger.info(f"  Output field: {output_field}")
    logger.info("=" * 60)

    # Load test split from HuggingFace
    logger.info(f"\nLoading dataset: {args.dataset}")
    rows = load_hf_dataset(args.dataset)
    logger.info(f"Loaded {len(rows)} rows from test split.")

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

    # Build output rows
    gold_spans_all = []
    output_rows    = []
    for row, predicted_spans in zip(rows, all_predicted_spans):
        gold_spans = [e["text"] for e in row.get(gold_field, []) if e.get("text")]
        gold_spans_all.append(gold_spans)

        out = dict(row)
        out[output_field] = predicted_spans
        output_rows.append(out)

    # Evaluate
    logger.info("\nEvaluating...")
    results = evaluate(gold_spans_all, all_predicted_spans)
    print_results(results, logger)

    # Save
    output_path = resolve_output_path(args)
    save_jsonl(output_rows, output_path)
    logger.info(f"\nOutput saved → {output_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()