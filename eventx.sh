#!/bin/bash

# EVENTX Inference Pipeline
# Runs all EVENTX models on EVENTX datasets with error handling

set +e  # Don't exit on error - continue with next combination

# Configuration
NUM_SAMPLES=400
BATCH_SIZE=16
MAX_LENGTH=512
DEVICE="auto"

# Create output directories
mkdir -p outputs
mkdir -p results
mkdir -p logs

# Log file
LOG_FILE="logs/eventx_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================" | tee -a "$LOG_FILE"
echo "EVENTX INFERENCE PIPELINE" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# EVENTX Models
EVENTX_MODELS=(
    "mdg-nlp/RoBERTa-eventx-recognition-sentence"
    "mdg-nlp/T5-eventx-sentence-recognition"
    "mdg-nlp/gpt-2-eventx-sentence-recognition"
)

# EVENTX Datasets
EVENTX_DATASETS=(
    "mdg-nlp/eventx-recognition-original"
    "mdg-nlp/eventx-recognition-sentence-vocab-substituted-updated"
    "mdg-nlp/domain-eventx-recognition"
    "mdg-nlp/eventx-recognition-sentence-vocab-substituted"
    "mdg-nlp/eventx-recognition-document"
    "mdg-nlp/adv_eventx_sentences"
    "mdg-nlp/domain-eventx-recognition-sentence-updated"
    "mdg-nlp/eventx-recognition-vocab-substituted"
    "mdg-nlp/eventx-recognition-perturbed-gpt"
    "mdg-nlp/eventx-recognition"
    "mdg-nlp/eventx-recognition-sentence-conll"
    "mdg-nlp/domain-eventx-clinical-base"
    "mdg-nlp/eventx-recognition-sentence"
    "mdg-nlp/eventx-recognition-perturbed"
    "mdg-nlp/domain-eventx-recognition-sentence"
    "mdg-nlp/domain-eventx-clinical"
)

TOTAL_RUNS=$((${#EVENTX_MODELS[@]} * ${#EVENTX_DATASETS[@]}))
CURRENT_RUN=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

echo "Total combinations: $TOTAL_RUNS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track failures
FAILED_COMBINATIONS=()

for model in "${EVENTX_MODELS[@]}"; do
    for dataset in "${EVENTX_DATASETS[@]}"; do
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
            
            echo "✓ Inference completed successfully" | tee -a "$LOG_FILE"
            
            # Run evaluation
            echo ">>> Running evaluation..." | tee -a "$LOG_FILE"
            if python evaluate.py \
                --input "$OUTPUT_FILE" \
                --task event \
                --output "$RESULT_FILE" 2>&1 | tee -a "$LOG_FILE"; then
                
                echo "✓ Evaluation completed successfully" | tee -a "$LOG_FILE"
                echo "  Predictions: $OUTPUT_FILE" | tee -a "$LOG_FILE"
                echo "  Results: $RESULT_FILE" | tee -a "$LOG_FILE"
                echo "  Metrics: ${RESULT_FILE/.jsonl/_metrics.json}" | tee -a "$LOG_FILE"
            else
                echo "✗ Evaluation FAILED - skipping" | tee -a "$LOG_FILE"
                RUN_SUCCESS=false
            fi
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
echo "EVENTX PIPELINE COMPLETE" | tee -a "$LOG_FILE"
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