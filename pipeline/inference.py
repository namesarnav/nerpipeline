"""
NERInference - autodetects model architecture and uses the correct head

- encoder like BERT, ROBERTA. uses huggingface ner pipeline which reads the saved
  head from config — works correctly for any fine-tuned encoder NER model

- encoder-decoder T5, BART ->  AutoModelForSeq2SeqLM + generation
- decoder GPT2, Llama, Mistral -> AutoModelForCausalLM + prompted generation
"""

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    pipeline as hf_pipeline,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from typing import List


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

    return "encoder"  # safest default for NER


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
            # Use HF ner pipeline reads the saved head from config.json directly
            # This is correct regardless of how the model was fine-tuned.
            self._ner_pipeline = hf_pipeline(
                task="ner",
                model=self.model_name,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",  # merges B/I tokens into word spans
                device=self.device,
            )
            self.model = None  # not needed directly

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
