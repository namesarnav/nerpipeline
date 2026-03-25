#!/bin/bash
cd outputs/
rm *metrics*
mv *timex-recognition-sentence-vocab-substituted-updated*.jsonl timex-domain-vocab-t5.jsonl
mv *timex-recognition-sentence-perturbed*.jsonl timex-adversarial-prompt-based-t5.jsonl
mv *timex-recognition-sentence-perturbed-gpt*.jsonl timex-prompt-based-t5.jsonl
mv *timex-recognition-sentence-original*.jsonl timex-base-t5.jsonl
mv *timex-recognition-sentence-adversarial*.jsonl timex-adversarial-t5.jsonl
mv *domain-timex-recognition-sentence-updated*.jsonl timex-domain-t5.jsonl
mv *timex-recognition-document*.jsonl timex-length-t5.jsonl
mv *.jsonl /Volumes/Github/ner/mdg/results/updated_outputs/fine-tuned-results/timex/t5/
