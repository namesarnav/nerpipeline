#!/bin/bash

# eventx Inference Pipeline
# Runs all eventx models on eventx datasets with error handling

set +e  # Don't exit on error - continue with next combination

# Configuration
NUM_SAMPLES=80
BATCH_SIZE=16
MAX_LENGTH=5000
DEVICE="auto"

# Create output directories
mkdir -p outputs/eventx
mkdir -p results/eventx
mkdir -p logs/eventx

# Log file
LOG_FILE="logs/eventx_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================" | tee -a "$LOG_FILE"
echo "eventx INFERENCE PIPELINE" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# eventx Models
eventx_MODELS=(
    "mdg-nlp/T5-eventx-recognition-sentence"
)

eventx_DATASETS=(
    mdg-nlp/eventx-recognition-document # --> length 80
    # mdg-nlp/eventx-recognition-original # --> base
    # mdg-nlp/eventx-recognition-perturbed #  --> adversarial-prompt-based
    # mdg-nlp/domain-eventx-recognition-sentence-updated # --> eventx-domain
    # mdg-nlp/eventx-recognition-sentence-vocab-substituted-updated # --> eventx-domain-vocab
    # mdg-nlp/eventx-recognition-perturbed-gpt #-->  eventx-adversarial-prompt-based
)

TOTAL_RUNS=$((${#eventx_MODELS[@]} * ${#eventx_DATASETS[@]}))
CURRENT_RUN=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

echo "Total combinations: $TOTAL_RUNS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track failures
FAILED_COMBINATIONS=()

for model in "${eventx_MODELS[@]}"; do
    for dataset in "${eventx_DATASETS[@]}"; do
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
            --task event \
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
echo "eventx PIPELINE COMPLETE" | tee -a "$LOG_FILE"
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