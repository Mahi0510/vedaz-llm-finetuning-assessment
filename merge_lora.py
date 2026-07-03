#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge LoRA Weights into Base Model
==================================
Loads base model and trained LoRA weights, fuses them into a single 16-bit
model, and exports the unified weights for high-throughput serving (e.g. vLLM).

Author: Mahika Morolia
B.Tech Computer Science Engineering
Specialization: Cybersecurity & Forensics
Date: July 2026
"""

import os
import sys
import argparse
import logging
import yaml
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("merge_lora.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("merge_pipeline")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Model weight merging pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.yaml",
        help="Path to YAML training configuration"
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    from helper_utils import get_hardware_diagnostics
    diagnostics = get_hardware_diagnostics()
    
    # Check if GPU is present for running actual Unsloth merge
    if not diagnostics["cuda_available"]:
        logger.warning("CUDA is not available. Simulating merge pipeline. Creating output configuration maps.")
        merged_dir = config["export"].get("merged_model_dir", "outputs/merged_qwen2.5-7b-vedic-astrologer")
        os.makedirs(merged_dir, exist_ok=True)
        
        # Save a config file indicating simulation export success
        with open(os.path.join(merged_dir, "merge_metadata.json"), "w") as mf:
            import json
            json.dump({
                "status": "SIMULATED_SUCCESS",
                "base_model": config["model"]["base_model_name"],
                "lora_weights": config["export"].get("lora_model_dir", "outputs/lora_weights"),
                "export_format": "pytorch_16bit"
            }, mf, indent=2)
            
        logger.info(f"Simulated merged model directory created at: {merged_dir}")
        return

    # Real GPU merging flow using Unsloth FastLanguageModel methods
    logger.info("Initializing Unsloth base model and adapters for fusion...")
    try:
        from unsloth import FastLanguageModel
        
        lora_weights = config["export"].get("lora_model_dir", "outputs/lora_weights")
        merged_dir = config["export"].get("merged_model_dir", "outputs/merged_qwen2.5-7b-vedic-astrologer")
        
        if not os.path.exists(lora_weights):
            logger.error(f"LoRA weights missing at {lora_weights}. Train the model first.")
            sys.exit(1)
            
        # 1. Load merged 16bit base model with LoRA adapters attached
        logger.info("Loading weights via from_pretrained...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_weights,
            max_seq_length=config["model"]["max_seq_length"],
            dtype=config["model"]["dtype"],
            load_in_4bit=config["model"]["load_in_4bit"]
        )
        
        # 2. Save merged model directly to disk in float16
        logger.info(f"Saving fully fused 16-bit float weights to: {merged_dir}")
        model.save_pretrained_merged(
            merged_dir, 
            tokenizer, 
            save_method="merged_16bit"
        )
        logger.info("Model adapters successfully fused into base weight maps.")
        
        # 3. Optional GGUF export (Unsloth feature)
        # Uncomment inside notebooks or scripts if deploying to local platforms (e.g. llama.cpp / Ollama)
        # logger.info("Generating standard GGUF Quantization...")
        # model.save_pretrained_merged(merged_dir, tokenizer, save_method="gguf")
        
    except Exception as e:
        logger.error(f"Failed to merge model adapters: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
