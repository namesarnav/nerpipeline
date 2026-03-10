#!/usr/bin/env python3
"""
NER inference and eval

Just run the infer.sh file, easiest way. Read README . 
"""

import argparse
import os
from pipeline.loader import load_local_jsonl, load_hf_dataset, save_jsonl
from pipeline.inference import NERInference
from pipeline.evaluate import evaluate, print_results
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True,
                        help="Fine-tuned HuggingFace model name or local path")
    parser.add_argument("--input",      type=str, default=None,
                        help="Path to local JSONL file")
    parser.add_argument("--dataset",    type=str, default=None,
                        help="HuggingFace dataset name (uses test split)")
    parser.add_argument("--task",       type=str, required=True, choices=["event", "time"],
                        help="'event' or 'time'")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output JSONL path (default: outputs/<name>__<model>__predictions.jsonl)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device",     type=str, default="auto")
    return parser.parse_args()


def resolve_output_path(args):
    if args.output:
        return args.output
    os.makedirs("outputs", exist_ok=True)
    if args.input:
        stem = os.path.splitext(os.path.basename(args.input))[0]
    else:
        stem = args.dataset.replace("/", "_")
    model_slug = args.model.replace("/", "_").replace("-", "_")
    return os.path.join("outputs", f"{stem}__{model_slug}__predictions.jsonl")


def main():
    args   = parse_args()
    logger = setup_logger()

    if not args.input and not args.dataset:
        raise ValueError("Provide either --input (local file) or --dataset (HuggingFace name).")

    # Field names based on task
    gold_field   = "event_expressions"  if args.task == "event" else "time_expressions"
    output_field = "predicted_event_expressions" if args.task == "event" else "predicted_time_expressions"

    logger.info("=" * 60)
    logger.info("NER INFERENCE PIPELINE")
    logger.info(f"  Model       : {args.model}")
    logger.info(f"  Input       : {args.input or args.dataset}")
    logger.info(f"  Task        : {args.task}")
    logger.info(f"  Gold field  : {gold_field}[].text")
    logger.info(f"  Output field: {output_field}")
    logger.info("=" * 60)


    if args.input:
        logger.info(f"\nLoading local file: {args.input}")
        rows = load_local_jsonl(args.input)
    else:
        logger.info(f"\nLoading HuggingFace dataset: {args.dataset} (test split)")
        rows = load_hf_dataset(args.dataset)
    logger.info(f"Loaded {len(rows)} rows.")


    inference = NERInference(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        logger=logger,
    )


    texts = [row["text"] for row in rows]
    logger.info(f"\nRunning inference on {len(texts)} texts...")
    all_predicted_spans = inference.predict_batch(texts)

    output_rows = []
    gold_spans_all = []
    for row, predicted_spans in zip(rows, all_predicted_spans):
        gold_spans = [e["text"] for e in row.get(gold_field, []) if e.get("text")]
        gold_spans_all.append(gold_spans)

        out = dict(row)                      # all original fields preserved
        out[output_field] = predicted_spans  # predicted spans added
        output_rows.append(out)


    logger.info("\nEvaluating predictions against gold labels...")
    results = evaluate(gold_spans_all, all_predicted_spans)
    print_results(results, logger)

    output_path = resolve_output_path(args)
    save_jsonl(output_rows, output_path)
    logger.info(f"\nOutput saved → {output_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
