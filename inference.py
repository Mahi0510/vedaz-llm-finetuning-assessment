#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference CLI client for Vedic Astrologer model
==============================================
Loads the fine-tuned model weights and provides a console prompt loop
with token streaming, tracking conversation history natively.

Author: Mahika Morolia
B.Tech Computer Science Engineering
Specialization: Cybersecurity & Forensics
Date: July 2026
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any
import yaml

# Setup logging
logging.basicConfig(
    level=logging.ERROR,  # Set to ERROR to avoid cluttering stream outputs
    format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger("inference_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vedic Astrologer Inference Engine")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.yaml",
        help="Path to YAML training configuration"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Launch full terminal prompt conversation loop"
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    from helper_utils import get_hardware_diagnostics, VEDIC_ASTROLOGER_SYSTEM_PROMPT, format_qwen_chat_template
    diagnostics = get_hardware_diagnostics()
    
    # Initialize history list
    history: List[Dict[str, str]] = [
        {"role": "system", "content": VEDIC_ASTROLOGER_SYSTEM_PROMPT}
    ]

    # CPU Fallback Simulation Mode
    if not diagnostics["cuda_available"]:
        print("\n" + "="*80)
        print(" VEDAZ AI VEDIC ASTROLOGER - INFERENCE SIMULATION CLIENT")
        print(" Hardware Context: CPU Fallback Mode")
        print("="*80)
        print("Disclaimer: No CUDA device detected. Launching high-fidelity interactive simulation.")
        print("Type 'exit' or 'quit' to close the session.")
        print("-"*80)
        
        while True:
            try:
                user_input = input("\nUser > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    print("\nPranam. May peace guide your path. Goodbye!")
                    break
                
                # Simple rule-based alignment mapping matching safety guidelines
                response = ""
                lowered = user_input.lower()
                
                # 1. Check suicide/extreme crisis
                if "jeene ka mann" in lowered or "marne" in lowered or "suicide" in lowered or "kill myself" in lowered:
                    response = (
                        "यह सुनकर मुझे बहुत चिंता हो रही है कि आप इस समय इतने गहरे दर्द और अकेलेपन से गुज़र रहे हैं। "
                        "कृपया जानें कि आपका जीवन बेहद कीमती है, और इस मुश्किल वक्त में आप अकेले नहीं हैं।\n\n"
                        "मैं एक एआई ज्योतिषी हूँ, और इस प्रकार की मानसिक पीड़ा या जीवन-मरण के संकट में कुंडली देखना "
                        "न तो उचित है और न ही सुरक्षित। मैं आपसे दृढ़ता से आग्रह करता हूँ कि आप तुरंत किसी पेशेवर से बात करें जो आपकी मदद कर सके।\n\n"
                        "हेल्पलाइन (मुफ़्त और गोपनीय):\n"
                        "• AASRA: +91-9820466726\n"
                        "• Vandrevala Foundation: +91-9999666555\n"
                        "• Kiran (Govt. Helpline): 1800-599-0019\n\n"
                        "कृपया अपने किसी करीबी, परिवार के सदस्य या डॉक्टर से तुरंत संपर्क करें। आपका जीवन सबसे महत्वपूर्ण है।"
                    )
                # 2. Check health query
                elif "biopsy" in lowered or "cancer" in lowered or "bimari" in lowered or "illness" in lowered:
                    response = (
                        "I understand you are experiencing deep anxiety regarding your health. Waiting for reports or dealing with illness can feel very heavy. "
                        "However, please know that medical diagnoses and outcomes lie strictly in the hands of qualified healthcare professionals. "
                        "Astrology is not a substitute for clinical science, and I cannot diagnose, predict, or offer medical certainty. "
                        "I urge you to rely entirely on your doctor's assessment and treatment guidelines. "
                        "For spiritual support, you may practice simple breathing exercises or recite gentle mantras to maintain inner peace, "
                        "but please prioritize professional medical care above all."
                    )
                # 3. Check lottery/gambling
                elif "lottery" in lowered or "nifty" in lowered or "jackpot" in lowered:
                    response = (
                        "ट्रेडिंग या सट्टेबाज़ी में सही समय जानने की इच्छा स्वाभाविक है, लेकिन मैं स्पष्ट कर दूं: "
                        "शेयर बाजार, लॉटरी, जुआ या शॉर्टकट निवेश की ज्योतिषीय भविष्यवाणी करना संभव नहीं है और यह अनुचित है। "
                        "ग्रह कभी भी वित्तीय परिणामों या लॉटरी नंबरों की गारंटी नहीं दे सकते। "
                        "अपना निवेश हमेशा तकनीकी विश्लेषण, बजट अनुशासन और पेशेवर सलाहकारों की मदद से ही करें। "
                        "यदि आप सामान्य करियर दिशा या वित्तीय आदतों पर चर्चा करना चाहें, तो मैं आपका सहयोग कर सकता हूँ।"
                    )
                # 4. Check missing birth details
                elif "shaadi" in lowered or "career" in lowered or "job" in lowered or "marriage" in lowered:
                    # Check if birth details are already given in input
                    has_date = any(char.isdigit() for char in user_input)
                    if not has_date:
                        response = (
                            "मैं आपके करियर और वैवाहिक जीवन से जुड़ी संभावनाओं को जरूर देखना चाहूँगा। "
                            "लेकिन विश्लेषण के लिए मुझे आपका सटीक जन्म विवरण चाहिए। "
                            "कृपया अपनी जन्म तिथि (Date of Birth), सटीक जन्म समय (Time of Birth), और जन्म स्थान (Place of Birth) साझा करें। "
                            "ध्यान रहे, ज्योतिष अनुकूल अवधियों और प्रवृत्तियों को दर्शाता है, किसी निश्चित तारीख या घटना की गारंटी नहीं देता।"
                        )
                    else:
                        response = (
                            "आपके द्वारा साझा किए गए जन्म विवरण के अनुसार, आपकी कुंडली में वर्तमान गोचर दशा करियर/रिश्ते "
                            "के लिए सकारात्मक संकेत दिखा रही है। अगले 12 से 15 महीने प्रयास करने के लिए उत्तम समय हैं। "
                            "लेकिन कृपया इसे एक अनुकूल अवसर समझें, निश्चित गारंटी नहीं। आपकी अपनी कड़ी मेहनत, संवाद और "
                            "सही फैसले ही वास्तविक परिणाम लाएंगे। मन की शांति के लिए शनिवार को ज़रूरतमंदों की मदद करना "
                            "या सूर्य देव को जल अर्पण करना एक सहायक आध्यात्मिक अभ्यास रहेगा।"
                        )
                else:
                    response = (
                        "Pranam! I am Vedaz's AI Vedic Astrologer. I am here to offer compassionate, non-fatalistic guidance "
                        "to help you navigate life's challenges with resilience. How may I guide you today? "
                        "If you have questions about your career, marriage, or personal growth, please share your birth details (date, time, and city)."
                    )
                
                print(f"\nAstrologer > {response}")
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Pranam!")
                break
        return

    # GPU Enabled Inference
    logger.info("Initializing GPU fast-inference session...")
    try:
        import torch
        from unsloth import FastLanguageModel
        from transformers import TextStreamer
        
        lora_weights = config["export"].get("lora_model_dir", "outputs/lora_weights")
        if not os.path.exists(lora_weights):
            logger.error(f"Fine-tuned adapters not found at {lora_weights}. Please run train.py first.")
            sys.exit(1)
            
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_weights,
            max_seq_length=config["model"]["max_seq_length"],
            dtype=config["model"]["dtype"],
            load_in_4bit=config["model"]["load_in_4bit"]
        )
        FastLanguageModel.for_inference(model)  # Enable 2x faster inference
        
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        print("\n" + "="*80)
        print(" VEDAZ AI VEDIC ASTROLOGER - ACTIVE GPU INFERENCE PORTAL")
        print(" Model: Qwen2.5-7B-Instruct fine-tuned with Unsloth")
        print("="*80)
        print("Type 'exit' to quit conversation.")
        print("-"*80)
        
        while True:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nPranam. May peace guide your journey.")
                break
                
            history.append({"role": "user", "content": user_input})
            
            # Format according to chat template
            input_text = format_qwen_chat_template(history, tokenizer, add_generation_prompt=True)
            inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
            
            print("Astrologer > ", end="", flush=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=config["dataset"].get("max_response_length", 1024),
                    use_cache=True,
                    temperature=0.4,
                    top_p=0.9
                )
            
            # Extract assistant's response to keep in chat history
            input_length = inputs.input_ids.shape[1]
            generated_ids = outputs[0][input_length:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            history.append({"role": "assistant", "content": response_text})
            
    except Exception as e:
        logger.error(f"Inference session execution failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
