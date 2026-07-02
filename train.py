#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune Qwen2.5-7B-Instruct using Unsloth LoRA/QLoRA
=====================================================
Optimized training pipeline script for configuring, initializing, and
running responsible AI Vedic Astrologer model SFT alignment.

Author: Senior ML Engineer
Date: July 2026
"""

import os
import sys
import argparse
import logging
import yaml
from typing import Dict, Any

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("train_pipeline")


def parse_args() -> argparse.Namespace:
    """Parse command line parameters."""
    parser = argparse.ArgumentParser(description="Unsloth LoRA/QLoRA Fine-tuning script.")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.yaml",
        help="Path to YAML training configuration"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint folder to resume training from"
    )
    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Safely load and return YAML configuration."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"Failed to open config from {config_path}: {e}")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    
    # 1. Check Hardware Diagnostics
    from helper_utils import get_hardware_diagnostics, VEDIC_ASTROLOGER_SYSTEM_PROMPT
    diagnostics = get_hardware_diagnostics()
    
    if not diagnostics["cuda_available"]:
        logger.error("CUDA is NOT available. Unsloth requires a CUDA-enabled GPU (NVIDIA T4, V100, A100, RTX 3090/4090, etc.).")
        logger.warning("Aborting actual training. Writing configuration skeleton and exiting for safety.")
        sys.exit(1)

    # 2. Imports after validation to prevent long import times on CPU
    logger.info("Importing Unsloth and Hugging Face deep learning libraries...")
    try:
        import torch
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as e:
        logger.critical(f"Missing essential deep learning dependencies: {e}. Run pip install -r requirements.txt")
        sys.exit(1)

    # Load Hyperparameters from config.yaml
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    
    # 3. Load Model and Tokenizer via Unsloth FastLanguageModel
    logger.info(f"Loading Unsloth base model: {model_cfg['base_model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=model_cfg["load_in_4bit"],
        rope_scaling=model_cfg["rope_scaling"],
    )

    # 4. Inject LoRA adapters targeting projection layers
    logger.info("Injecting Parameter-Efficient LoRA matrices...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
        use_rslora=lora_cfg["use_rslora"],
        loftq_config=lora_cfg["loftq_config"],
    )

    # 5. Load Preprocessed Datasets
    logger.info("Loading training and validation partitions...")
    train_file = dataset_cfg["train_split_path"]
    eval_file = dataset_cfg["eval_split_path"]
    
    if not os.path.exists(train_file) or not os.path.exists(eval_file):
        logger.critical(f"Split dataset files missing! Run preprocess.py before training.")
        sys.exit(1)

    dataset_dict = load_dataset(
        "json", 
        data_files={"train": train_file, "eval": eval_file}
    )

    # 6. Apply Qwen2.5 Chat Template and Tokenization mapping
    logger.info("Formatting datasets with official ChatML sequence templates...")
    
    def apply_formatting_template(examples):
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            # Inject standard system prompt if none is set
            if conversation[0]["role"] != "system":
                conversation.insert(0, {"role": "system", "content": VEDIC_ASTROLOGER_SYSTEM_PROMPT})
            
            # Apply Tokenizer chat template natively
            templated = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(templated)
        return {"text": texts}

    dataset_formatted = dataset_dict.map(apply_formatting_template, batched=True)
    logger.info(f"Ingested training sequences count: {len(dataset_formatted['train'])}")

    # 7. Setup Training Arguments
    logger.info("Initializing Hugging Face TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_ratio=train_cfg["warmup_ratio"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        fp16=train_cfg["fp16"] and not train_cfg["bf16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        evaluation_strategy=train_cfg["evaluation_strategy"],
        eval_steps=train_cfg["eval_steps"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=train_cfg["seed"],
        report_to=train_cfg["report_to"],
        logging_dir=train_cfg["logging_dir"],
    )

    # 8. Wrap Model in Supervised Fine-Tuning SFTTrainer
    logger.info("Constructing SFTTrainer wrapper...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_formatted["train"],
        eval_dataset=dataset_formatted["eval"],
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        dataset_num_proc=2,
        packing=False,  # Can set true for packing multiple short conversations
        args=training_args,
    )

    # 9. Execute SFT Model training
    resume_checkpoint = args.resume or train_cfg.get("resume_from_checkpoint")
    if resume_checkpoint:
        logger.info(f"Resuming fine-tuning from existing checkpoint: {resume_checkpoint}")
    else:
        logger.info("Launching SFT training loop from scratch...")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info("Fine-tuning session successfully concluded.")
    logger.info(f"Training statistics: {trainer_stats}")

    # 10. Save LoRA Model Adapters
    lora_save_path = config["export"].get("lora_model_dir", "outputs/lora_weights")
    logger.info(f"Saving Parameter-Efficient LoRA adapters to: {lora_save_path}")
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)
    
    logger.info("Training pipeline completed with success.")


if __name__ == "__main__":
    main()
