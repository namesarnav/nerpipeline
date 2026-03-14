#!/usr/bin/env python3
"""
Extract F1 Scores from metrics JSON files to CSV

Reads all *_metrics.json files from outputs/timex/metrics/ and outputs/eventx/metrics/
and creates a simple CSV with: Model, Dataset, F1 Micro, F1 Macro

Usage:
    python extract_metrics_to_csv.py
    python extract_metrics_to_csv.py --output results.csv
"""

import argparse
import json
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Extract F1 scores from metrics to CSV")
    parser.add_argument("--output", type=str, default="f1_scores.csv",
                        help="Output CSV file path (default: f1_scores.csv)")
    parser.add_argument("--outputs_dir", type=str, default="outputs",
                        help="Base outputs directory (default: outputs)")
    return parser.parse_args()


def extract_model_dataset_from_filename(filename):
    """
    Extract model and dataset from filename like:
    mdg-nlp_adv-timex-sentences-bae__mdg-nlp_gpt-2-timex-sentence-recognition__predictions_metrics.json
    """
    # Remove _predictions_metrics.json or _metrics.json
    name = filename.replace("_predictions_metrics.json", "").replace("_metrics.json", "")
    
    # Split by double underscore
    parts = name.split("__")
    
    if len(parts) >= 2:
        dataset = parts[0].replace("mdg-nlp_", "mdg-nlp/").replace("_", "-")
        model = parts[1].replace("mdg-nlp_", "mdg-nlp/").replace("_", "-")
        return dataset, model
    
    return "Unknown", "Unknown"


def read_metrics_file(filepath):
    """Read metrics JSON file and extract F1 scores"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        micro_f1 = data.get("micro_f1", 0.0)
        macro_f1 = data.get("macro_f1", 0.0)
        
        return micro_f1, macro_f1
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None


def collect_all_metrics(outputs_dir):
    """Collect metrics from both timex and eventx folders"""
    results = []
    
    # Check timex and eventx folders directly
    timex_dir = Path(outputs_dir) / "timex"
    eventx_dir = Path(outputs_dir) / "eventx"
    
    metrics_dirs = []
    if timex_dir.exists():
        metrics_dirs.append(("timex", timex_dir))
    if eventx_dir.exists():
        metrics_dirs.append(("eventx", eventx_dir))
    
    if not metrics_dirs:
        print(f"No timex or eventx directories found in {outputs_dir}")
        print(f"Expected: {outputs_dir}/timex/ or {outputs_dir}/eventx/")
        return results
    
    for task, task_dir in metrics_dirs:
        print(f"\nProcessing {task} metrics from: {task_dir}")
        
        # Find all *_metrics.json files
        metrics_files = list(task_dir.glob("*_metrics.json"))
        print(f"Found {len(metrics_files)} metrics files")
        
        for metrics_file in metrics_files:
            filename = metrics_file.name
            dataset, model = extract_model_dataset_from_filename(filename)
            
            micro_f1, macro_f1 = read_metrics_file(metrics_file)
            
            if micro_f1 is not None:
                results.append({
                    "Model": model,
                    "Dataset": dataset,
                    "Task": task,
                    "F1_Micro": micro_f1,
                    "F1_Macro": macro_f1
                })
                print(f"  ✓ {model} on {dataset}: Micro={micro_f1:.4f}, Macro={macro_f1:.4f}")
            else:
                print(f"  ✗ Failed to read: {filename}")
    
    return results


def save_to_csv(results, output_path):
    """Save results to CSV"""
    if not results:
        print("\nNo results to save!")
        return
    
    # Sort by Task, Dataset, Model
    results = sorted(results, key=lambda x: (x["Task"], x["Dataset"], x["Model"]))
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Dataset", "Task", "F1_Micro", "F1_Macro"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved {len(results)} results to: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("EXTRACTING F1 SCORES TO CSV")
    print("=" * 70)
    print(f"Outputs directory: {args.outputs_dir}")
    print(f"Output file: {args.output}")
    print("=" * 70)
    
    # Collect metrics
    results = collect_all_metrics(args.outputs_dir)
    
    if not results:
        print("\n❌ No metrics found!")
        print("\nMake sure:")
        print("  1. Metrics files exist in outputs/timex/ or outputs/eventx/")
        print("  2. Files are named *_metrics.json")
        return
    
    # Save to CSV
    save_to_csv(results, args.output)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {len(results)}")
    
    if results:
        avg_micro = sum(r["F1_Micro"] for r in results) / len(results)
        avg_macro = sum(r["F1_Macro"] for r in results) / len(results)
        max_micro = max(r["F1_Micro"] for r in results)
        min_micro = min(r["F1_Micro"] for r in results)
        
        print(f"Average Micro F1: {avg_micro:.4f}")
        print(f"Average Macro F1: {avg_macro:.4f}")
        print(f"Best Micro F1: {max_micro:.4f}")
        print(f"Worst Micro F1: {min_micro:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()