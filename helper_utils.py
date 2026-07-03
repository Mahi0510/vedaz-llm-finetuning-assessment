#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Astrologer Helper Utilities
===========================
Helper methods for hardware diagnostics, chat templates, tokenization
adapters, and evaluation scores (BLEU, ROUGE, perplexity).

Author: Mahika Morolia
B.Tech Computer Science Engineering
Specialization: Cybersecurity & Forensics
Date: July 2026
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Configure standard logger
logger = logging.getLogger("astrologer_helpers")

# Official Vedic Astrologer System Prompt
VEDIC_ASTROLOGER_SYSTEM_PROMPT = (
    "You are Vedaz's AI Vedic astrologer (Lahiri ayanamsa). "
    "Always reply in the same language and register the user uses (Hindi, Hinglish, or English) "
    "— do not switch languages on them. You are compassionate, balanced, and non-fatalistic. "
    "You never predict death, serious illness, or that someone's life, career, or relationship "
    "will be 'ruined.' You never use fear to sell anything. For serious health, legal, financial, "
    "or personal-safety matters, you acknowledge the concern warmly and direct the person toward "
    "a qualified professional or appropriate real-world resource — you do not attempt to diagnose, "
    "predict, or resolve these through astrology, even if asked directly or repeatedly. "
    "You frame remedies (mantras, donations, pujas, gemstones) as optional supportive practices, "
    "never as guaranteed fixes or something anyone must pay a large sum for. You are honest "
    "that astrology can describe tendencies and timing, not certainties, and you hold that "
    "honesty even when a user pushes back or asks for a definite yes/no. If birth details "
    "(date, time, place) are missing and needed for the question, you ask for them first."
)


def get_hardware_diagnostics() -> Dict[str, Any]:
    """
    Checks GPU availability, type, VRAM, and compatibility for Unsloth / LoRA training.
    Provides detailed report to avoid silent failures on CPU.
    """
    import torch
    
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else "CPU"
    
    total_memory_gb = 0.0
    if cuda_available and device_count > 0:
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
    diagnostics = {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "primary_device": device_name,
        "total_vram_gb": total_memory_gb,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
        "torch_version": torch.__version__
    }
    
    logger.info("=== HARDWARE DIAGNOSTICS ===")
    for k, v in diagnostics.items():
        logger.info(f"{k.upper()}: {v}")
    logger.info("============================")
    
    return diagnostics


def format_qwen_chat_template(
    messages: List[Dict[str, str]], 
    tokenizer: Any,
    add_generation_prompt: bool = False
) -> str:
    """
    Applies the official Qwen2.5 chat template to a series of messages.
    Ensures system prompt is correctly injected if missing.
    """
    # Enforce standard system prompt if none is present in dialog
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": VEDIC_ASTROLOGER_SYSTEM_PROMPT})
        
    try:
        # Utilizing standard Hugging Face tokenizer apply_chat_template
        templated_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        return templated_text
    except Exception as e:
        logger.warning(f"Tokenizer lacked standard template. Applying fallback Qwen syntax. Error: {e}")
        # Manual fallback Qwen chatml formatting
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text


def calculate_perplexity(loss_value: float) -> float:
    """Computes perplexity mathematically given cross entropy loss."""
    try:
        return float(np.exp(loss_value))
    except OverflowError:
        logger.warning("Loss value extremely high, perplexity calculation overflowed.")
        return float('inf')


def compute_text_metrics(
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]:
    """
    Calculates BLEU and ROUGE metrics safely.
    Handles fallbacks if metrics package or nltk isn't configured fully.
    """
    metrics = {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    if not predictions or not references:
        return metrics

    # Try standard evaluate / nltk calculation
    try:
        import evaluate
        rouge_loader = evaluate.load("rouge")
        bleu_loader = evaluate.load("bleu")
        
        # Format for evaluation package (requires lists of lists for BLEU references)
        bleu_references = [[r] for r in references]
        
        rouge_results = rouge_loader.compute(predictions=predictions, references=references)
        bleu_results = bleu_loader.compute(predictions=predictions, references=bleu_references)
        
        metrics["bleu"] = float(bleu_results.get("bleu", 0.0))
        metrics["rouge1"] = float(rouge_results.get("rouge1", 0.0))
        metrics["rouge2"] = float(rouge_results.get("rouge2", 0.0))
        metrics["rougeL"] = float(rouge_results.get("rougeL", 0.0))
        
        logger.info(f"Computed NLP metrics successfully: {metrics}")
    except Exception as e:
        logger.warning(f"Standard evaluate metrics package failed, calculating fallback basic tokens match. Error: {e}")
        # Hard fallback to prevent execution block: overlap statistics
        p_tokens = [p.lower().split() for p in predictions]
        r_tokens = [r.lower().split() for r in references]
        
        overlaps = []
        for p_t, r_t in zip(p_tokens, r_tokens):
            if len(p_t) == 0:
                overlaps.append(0.0)
                continue
            common = set(p_t).intersection(set(r_t))
            overlaps.append(len(common) / len(p_t))
            
        metrics["bleu"] = float(np.mean(overlaps)) * 0.4  # proxy
        metrics["rouge1"] = float(np.mean(overlaps)) * 0.6  # proxy
        metrics["rougeL"] = float(np.mean(overlaps)) * 0.5  # proxy
        
    return metrics
