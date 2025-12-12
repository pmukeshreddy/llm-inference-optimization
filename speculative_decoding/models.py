"""Model loading utilities"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

MODEL_PAIRS = {
    "tiny": {"draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "target": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
    "small": {"draft": "Qwen/Qwen2-0.5B-Instruct", "target": "Qwen/Qwen2-1.5B-Instruct"},
    "medium": {"draft": "Qwen/Qwen2-0.5B-Instruct", "target": "Qwen/Qwen2-7B-Instruct"},
    "llama": {"draft": "meta-llama/Llama-3.2-1B-Instruct", "target": "meta-llama/Llama-3.2-3B-Instruct"},
}


def get_quantization_config(quantization: Optional[str] = None) -> Optional[BitsAndBytesConfig]:
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model_pair(
    draft_model_name: str,
    target_model_name: str,
    device: str = "cuda",
    target_quantization: Optional[str] = "4bit",
    hf_token: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    
    logger.info(f"Loading draft: {draft_model_name}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        attn_implementation="sdpa"
    )
    draft_model.eval()
    
    logger.info(f"Loading target: {target_model_name}")
    target_quant = get_quantization_config(target_quantization)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        quantization_config=target_quant,
        torch_dtype=torch.float16 if target_quant is None else None,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        attn_implementation="sdpa"
    )
    target_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return draft_model, target_model, tokenizer


def load_preset(preset: str = "small", hf_token: Optional[str] = None, **kwargs):
    if preset not in MODEL_PAIRS:
        raise ValueError(f"Unknown preset: {preset}")
    pair = MODEL_PAIRS[preset]
    return load_model_pair(pair["draft"], pair["target"], hf_token=hf_token, **kwargs)
