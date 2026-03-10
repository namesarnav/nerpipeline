# NER pipeline

Usage:
```bash
python run.py --model <hf_model> --dataset <hf_dataset> --task <eventx|timex>
```

Example : 
```bash
    python run.py \
        --model mdg-nlp/event-ner-bert-base-cased \
        --dataset mdg-nlp/eventx-recognition-perturbed \
        --task eventx
```