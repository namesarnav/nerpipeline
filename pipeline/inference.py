#!/usr/bin/env python3

import argparse
import json
import os
from datasets import load_dataset
from typing import List
import logging


# Import NERInference class
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    pipeline as hf_pipeline,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)


ENCODER_ONLY = {
    "bert", "roberta", "distilbert", "albert", "deberta",
    "electra", "xlm-roberta", "camembert", "ernie", "luke",
}
DECODER_ONLY = {
    "gpt2", "gpt_neo", "gpt_neox", "llama", "mistral", "falcon",
    "mpt", "opt", "bloom", "gemma", "phi", "qwen2", "mixtral",
}
ENC_DEC = {
    "t5", "bart", "pegasus", "mbart", "mt5", "longt5", "flan",
}


def detect_arch(model_name: str, config) -> str:
    name = model_name.lower()
    for a in ENCODER_ONLY:
        if a in name:
            return "encoder"
    for a in DECODER_ONLY:
        if a in name:
            return "decoder"
    for a in ENC_DEC:
        if a in name:
            return "enc_dec"

    mt = getattr(config, "model_type", "").lower()
    
    if mt in ENCODER_ONLY: return "encoder"
    if mt in DECODER_ONLY: return "decoder"
    if mt in ENC_DEC: return "enc_dec"
    
    if getattr(config, "decoder_start_token_id", None) is not None:
        return "enc_dec"

    return "encoder"


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():       return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def parse_generated_spans(text: str) -> List[str]:
    """
    Parse a flat list of spans from generated text.
    Handles: JSON list, comma-separated, pipe-separated, newline-separated.
    """
    import re, json as _json

    text = text.strip()

    try:
        parsed = _json.loads(text)
        if isinstance(parsed, list):
            return [str(s).strip() for s in parsed if str(s).strip()]
    except Exception:
        pass

    text = re.sub(r"^[A-Z_]+:\s*", "", text)

    if "|" in text:
        return [s.strip() for s in text.split("|") if s.strip()]
    if "," in text:
        return [s.strip() for s in text.split(",") if s.strip()]
    if "\n" in text:
        return [s.strip() for s in text.split("\n") if s.strip()]

    return [text] if text else []


class NERInference:
    def __init__(self, model_name: str, device: str, max_length: int,
                 batch_size: int, logger):
        self.model_name  = model_name
        self.max_length  = max_length
        self.batch_size  = batch_size
        self.logger      = logger
        self.device      = resolve_device(device)

        logger.info(f"\nLoading config: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name)
        self.arch   = detect_arch(model_name, self.config)
        logger.info(f"Architecture: {self.arch.upper()}")

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, add_prefix_space=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model...")
        self._load_model()
        logger.info(f"Ready on {self.device}.")

    def _load_model(self):
        if self.arch == "encoder":
            self._ner_pipeline = hf_pipeline(
                task="ner",
                model=self.model_name,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=self.device,
            )
            self.model = None

        elif self.arch == "enc_dec":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if str(self.device) != "cpu" else torch.float32,
            ).to(self.device)
            self._ner_pipeline = None

        else:  # decoder
            dtype = torch.float16 if str(self.device) != "cpu" else torch.float32
            try:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb,
                    device_map="auto",
                )
                self.logger.info("Loaded in 4-bit quantization.")
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto" if "cuda" in str(self.device) else None,
                )
                if "cuda" not in str(self.device):
                    self.model = self.model.to(self.device)
            self._ner_pipeline = None

    def predict_batch(self, texts: List[str]) -> List[List[str]]:
        all_spans = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            batch_num = i // self.batch_size + 1
            self.logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)")

            if self.arch == "encoder":
                spans = self._predict_encoder(batch)
            elif self.arch == "enc_dec":
                spans = self._predict_enc_dec(batch)
            else:
                spans = self._predict_decoder(batch)

            all_spans.extend(spans)
        return all_spans

    def _predict_encoder(self, texts: List[str]) -> List[List[str]]:
        results = self._ner_pipeline(texts, batch_size=len(texts))
        if results and not isinstance(results[0], list):
            results = [results]
        output = []
        for doc in results:
            spans = [ent["word"].strip() for ent in doc if ent.get("word", "").strip()]
            output.append(spans)
        return output

    def _predict_enc_dec(self, texts: List[str]) -> List[List[str]]:
        prompt = "Extract named entities: "
        inputs = self.tokenizer(
            [prompt + t for t in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
            )

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [parse_generated_spans(d) for d in decoded]

    def _predict_decoder(self, texts: List[str]) -> List[List[str]]:
        prompt_template = (
            "Extract all named entities from the text as a comma-separated list.\n"
            "Text: {text}\n"
            "Entities:"
        )
        prompts = [prompt_template.format(text=t) for t in texts]
        inputs  = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        decoded    = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [parse_generated_spans(d) for d in decoded]


def parse_args():
    parser = argparse.ArgumentParser(description="Run NER inference on test dataset")
    
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID (e.g., bert-base-uncased, roberta-base, t5-small)")
    
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="HuggingFace dataset ID (e.g., conll2003)")
    
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration (optional)")
    
    parser.add_argument("--task", type=str, required=True, choices=["event", "time"],
                        help="Task type: event or time")
    
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    
    parser.add_argument("--num_samples", type=int, default=400,
                        help="Maximum number of samples to process from test split (default: 400)")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference (default: 16)")
    
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, cpu, cuda, mps (default: auto)")
    
    return parser.parse_args()


def extract_entities_from_tags(tokens: List[str], tags: List[int], label_list: List[str]) -> List[str]:
    """
    Extract entity expressions from tokens and NER tags
    Handles BIO tagging scheme
    """
    entities = []
    current_entity = []
    
    for token, tag_id in zip(tokens, tags):
        if tag_id == -100 or tag_id >= len(label_list):
            continue
            
        label = label_list[tag_id]
        
        if label.startswith("B-") or (label != "O" and not label.startswith("I-")):
            # Begin new entity
            if current_entity:
                entities.append(" ".join(current_entity))
            current_entity = [token]
        elif label.startswith("I-") and current_entity:
            # Continue entity
            current_entity.append(token)
        else:
            # O tag or end of entity
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
    
    # Don't forget last entity
    if current_entity:
        entities.append(" ".join(current_entity))
    
    return entities


def run_inference(args):
    """Main inference function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load dataset
    logger.info(f"Loading dataset {args.dataset_id}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_id, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_id)
    
    # Get test split
    if "test" not in dataset:
        raise ValueError("Dataset does not have a 'test' split")
    
    test_data = dataset["test"]
    
    # Handle datasets with fewer samples than requested
    total_available = len(test_data)
    num_samples = min(args.num_samples, total_available)
    
    if total_available < args.num_samples:
        logger.warning(f"Dataset only has {total_available} test samples, processing all of them instead of {args.num_samples}")
    
    test_data = test_data.select(range(num_samples))
    
    logger.info(f"Processing {num_samples} samples from test split")
    
    # Get label list from dataset
    features = test_data.features
    if 'ner_tags' in features:
        label_list = features['ner_tags'].feature.names
    elif 'tags' in features:
        label_list = features['tags'].feature.names
    else:
        logger.warning("No NER tags found in dataset, using generic labels")
        label_list = ["O", "B-MISC", "I-MISC"]
    
    logger.info(f"Label list: {label_list}")
    
    # Initialize NER inference
    ner_model = NERInference(
        model_name=args.model_id,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        logger=logger
    )
    
    # Prepare texts and gold labels
    all_texts = []
    all_gold_entities = []
    
    logger.info("Extracting texts and gold labels...")
    for sample in test_data:
        tokens = sample.get("tokens", [])
        tags = sample.get("ner_tags", sample.get("tags", []))
        
        # Reconstruct text from tokens
        text = " ".join(tokens)
        all_texts.append(text)
        
        # Extract gold entities
        gold_entities = extract_entities_from_tags(tokens, tags, label_list)
        all_gold_entities.append(gold_entities)
    
    # Run predictions
    logger.info("Running inference...")
    all_predictions = ner_model.predict_batch(all_texts)
    
    # Prepare output
    if args.task == "event":
        gold_field = "gold_event_expressions"
        pred_field = "post_processed_event_expressions"
    else:  # time
        gold_field = "gold_timex_expressions"
        pred_field = "post_processed_timex_expressions"
    
    output_rows = []
    for text, gold, pred in zip(all_texts, all_gold_entities, all_predictions):
        output_row = {
            "text": text,
            gold_field: gold,
            pred_field: pred
        }
        output_rows.append(output_row)
    
    # Save results
    logger.info(f"Saving results to {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")
    
    logger.info(f"Done! Processed {len(output_rows)} samples")
    logger.info(f"Output saved to: {args.output}")


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()