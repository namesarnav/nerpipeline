#!/bin/bash

# TIMEX Inference Pipeline
# Runs all TIMEX models on TIMEX datasets with error handling

set +e  # Don't exit on error - continue with next combination

# Configuration
NUM_SAMPLES=400
BATCH_SIZE=20
MAX_LENGTH=5000
DEVICE="auto"

# Create output directories
mkdir -p outputs/timex
mkdir -p results/timex
mkdir -p logs/timex

# Log file
LOG_FILE="logs/timex_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================" | tee -a "$LOG_FILE"
echo "TIMEX INFERENCE PIPELINE" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# TIMEX Models
TIMEX_MODELS=(
    "mdg-nlp/RoBERTa-timex-recognition-sentence"
    "mdg-nlp/T5-timex-recognition-sentence"
    "mdg-nlp/gpt-2-timex-sentence-recognition"
)

# TIMEX Datasets
TIMEX_DATASETS=(
    "mdg-nlp/timex-recognition-sentence-vocab-substitute"
    "mdg-nlp/timex-recognition-sentence-vocab-substituted"
    "mdg-nlp/timex-recognition-sentence-perturbed"
    "mdg-nlp/domain-timex-clinical-base"
    "mdg-nlp/domain-timex-clinical"
    "mdg-nlp/timex-compositional-sentence"
    "mdg-nlp/domain-timex-recognition-sentence-updated"
    "mdg-nlp/timex-recognition-sentence-updated"
    "mdg-nlp/timex-recognition-sentence"
    "mdg-nlp/timex-recognition-sentence-adversarial"
    "mdg-nlp/timex-recognition-sentence-original"
    "mdg-nlp/timex-recognition-document"
    "mdg-nlp/adv-timex-sentences-bert-attack"
    "mdg-nlp/adv-timex-sentences"
    "mdg-nlp/adv-timex-sentences-textfooler"
    "mdg-nlp/adv-timex-sentences-ner_clare"
    "mdg-nlp/adv-timex-sentences-morpheus"
    "mdg-nlp/adv-timex-sentences-deepword"
    "mdg-nlp/adv-timex-sentences-bae"
    "mdg-nlp/timex-recognition-sentence-private"
    "mdg-nlp/timex-recognition"
)

TOTAL_RUNS=$((${#TIMEX_MODELS[@]} * ${#TIMEX_DATASETS[@]}))
CURRENT_RUN=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

echo "Total combinations: $TOTAL_RUNS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track failures
FAILED_COMBINATIONS=()

for model in "${TIMEX_MODELS[@]}"; do
    for dataset in "${TIMEX_DATASETS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        echo "" | tee -a "$LOG_FILE"
        echo "======================================================" | tee -a "$LOG_FILE"
        echo "RUN [$CURRENT_RUN/$TOTAL_RUNS]" | tee -a "$LOG_FILE"
        echo "Model: $model" | tee -a "$LOG_FILE"
        echo "Dataset: $dataset" | tee -a "$LOG_FILE"
        echo "======================================================" | tee -a "$LOG_FILE"
        
        # Generate filenames
        MODEL_SLUG=$(echo "$model" | tr '/' '_')
        DATASET_SLUG=$(echo "$dataset" | tr '/' '_')
        OUTPUT_FILE="outputs/${DATASET_SLUG}__${MODEL_SLUG}__predictions.jsonl"
        RESULT_FILE="results/${DATASET_SLUG}__${MODEL_SLUG}__results.jsonl"
        
        # Flag to track this run's success
        RUN_SUCCESS=true
        
        # Run inference
        echo ">>> Running inference..." | tee -a "$LOG_FILE"
        if python run.py \
            --model "$model" \
            --dataset "$dataset" \
            --task time \
            --num_samples $NUM_SAMPLES \
            --batch_size $BATCH_SIZE \
            --max_length $MAX_LENGTH \
            --device $DEVICE \
            --output "$OUTPUT_FILE" 2>&1 | tee -a "$LOG_FILE"; then

            echo " ### Inference completed successfully ### " | tee -a "$LOG_FILE"
            
        else
            echo "✗ Inference FAILED - skipping to next combination" | tee -a "$LOG_FILE"
            RUN_SUCCESS=false
        fi
        
        # Update counters
        if [ "$RUN_SUCCESS" = true ]; then
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
            FAILED_COMBINATIONS+=("$model + $dataset")
        fi
        
        echo "Progress: $SUCCESSFUL_RUNS successful, $FAILED_RUNS failed" | tee -a "$LOG_FILE"
    done
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
echo "TIMEX PIPELINE COMPLETE" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "SUMMARY:" | tee -a "$LOG_FILE"
echo "  Total runs: $TOTAL_RUNS" | tee -a "$LOG_FILE"
echo "  Successful: $SUCCESSFUL_RUNS" | tee -a "$LOG_FILE"
echo "  Failed: $FAILED_RUNS" | tee -a "$LOG_FILE"
echo "  Success rate: $(awk "BEGIN {printf \"%.1f\", ($SUCCESSFUL_RUNS/$TOTAL_RUNS)*100}")%" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List failed combinations if any
if [ $FAILED_RUNS -gt 0 ]; then
    echo "FAILED COMBINATIONS:" | tee -a "$LOG_FILE"
    for combo in "${FAILED_COMBINATIONS[@]}"; do
        echo "  - $combo" | tee -a "$LOG_FILE"
    done
    echo "" | tee -a "$LOG_FILE"
fi

echo "Outputs: outputs/" | tee -a "$LOG_FILE"
echo "Results: results/" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"