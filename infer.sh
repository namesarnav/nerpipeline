#!/bin/bash

python run.py \
        --model mdg-nlp/event-ner-bert-base-cased \   # replace this for model 
        --dataset mdg-nlp/eventx-recognition-perturbed \   # replace this for dataset
        --task eventx