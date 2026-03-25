#!/bin/bash

# Single Model Inference Script
# Run inference for one model on one dataset

# Configuration

#  "mdg-nlp/RoBERTa-timex-recognition-sentence"
#     "mdg-nlp/T5-timex-recognition-sentence"
#     "mdg-nlp/gpt-2-timex-sentence-recognition"

MODEL=mdg-nlp/T5-timex-recognition-sentence
DATASET=mdg-nlp/domain-timex-recognition-sentence-updated
TASK="time"               # event or time
NUM_SAMPLES=400
BATCH_SIZE=20
MAX_LENGTH=5000
DEVICE="auto"

echo "Running inference..."
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Task: $TASK"
echo ""

# Run inference
python run.py \
    --model $MODEL \
    --dataset $DATASET \
    --task $TASK \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --device $DEVICE

# Evaluate
echo ""
echo "Evaluating results..."

DATASET_SLUG=$(echo $DATASET | tr '/' '_')
MODEL_SLUG=$(echo $MODEL | tr '/' '_')
PREDICTIONS_FILE="outputs/${DATASET_SLUG}__${MODEL_SLUG}__predictions.jsonl"
RESULTS_FILE="results/${MODEL_SLUG}_${TASK}_results.jsonl"

mkdir -p results

python evaluate.py \
    --input $PREDICTIONS_FILE \
    --task $TASK \
    --output $RESULTS_FILE

echo ""
echo "Done!"
echo "Predictions: $PREDICTIONS_FILE"
echo "Results: $RESULTS_FILE"
echo "Metrics: ${RESULTS_FILE/.jsonl/_metrics.json}"