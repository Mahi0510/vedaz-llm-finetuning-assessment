#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradio Web Application for Vedaz Vedic Astrologer
=================================================
Launches an elegant web application with full conversation history,
birth detail integrations, and streaming responses. Supports CUDA/CPU fallbacks.

Author: Mahika Morolia
B.Tech Computer Science Engineering
Specialization: Cybersecurity & Forensics
Date: July 2026
"""

import os
import sys
import logging
import argparse
from typing import List, Tuple, Dict, Any
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("gradio_app")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio Vedic Astrologer Dashboard")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.yaml",
        help="Path to YAML training configuration"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Generate public Gradio share link"
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # 1. Hardware checks
    from helper_utils import get_hardware_diagnostics, VEDIC_ASTROLOGER_SYSTEM_PROMPT, format_qwen_chat_template
    diagnostics = get_hardware_diagnostics()
    
    # Check if GPU is present for Gradio execution
    cuda_active = diagnostics["cuda_available"]
    
    model = None
    tokenizer = None
    
    if cuda_active:
        logger.info("GPU active. Loading fine-tuned weights for Gradio serving...")
        try:
            import torch
            from unsloth import FastLanguageModel
            
            lora_weights = config["export"].get("lora_model_dir", "outputs/lora_weights")
            if not os.path.exists(lora_weights):
                logger.error(f"LoRA weights missing at {lora_weights}. Starting Gradio with base Qwen model instead.")
                lora_weights = config["model"]["base_model_name"]
                
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=lora_weights,
                max_seq_length=config["model"]["max_seq_length"],
                dtype=config["model"]["dtype"],
                load_in_4bit=config["model"]["load_in_4bit"]
            )
            FastLanguageModel.for_inference(model)
        except Exception as e:
            logger.error(f"Failed to load weights to GPU. Forcing simulation mode. Error: {e}")
            cuda_active = False

    # Define response engine
    def predict_astrologer(
        message: str, 
        history_tuples: List[Tuple[str, str]],
        dob: str,
        tob: str,
        pob: str
    ):
        """Generates streaming responses, injecting birth details and managing history."""
        # Clean user input
        message_cleaned = message.strip()
        if not message_cleaned:
            yield "Please enter a message."
            return

        # Prepare system prompt with birth details if provided
        birth_context = ""
        if dob or tob or pob:
            birth_context = f" [User birth details - DOB: {dob or 'Unknown'}, TOB: {tob or 'Unknown'}, POB: {pob or 'Unknown'}. Please analyze their kundli based on these details.]"
            
        full_system_prompt = VEDIC_ASTROLOGER_SYSTEM_PROMPT + birth_context
        
        # Build standard chat list from Gradio history tuples
        formatted_history = [{"role": "system", "content": full_system_prompt}]
        for user_t, assistant_t in history_tuples:
            formatted_history.append({"role": "user", "content": user_t})
            formatted_history.append({"role": "assistant", "content": assistant_t})
            
        formatted_history.append({"role": "user", "content": message_cleaned})

        # --- GPU STREAMING GENERATION ---
        if cuda_active:
            try:
                import torch
                from transformers import TextIteratorStreamer
                from threading import Thread
                
                input_text = format_qwen_chat_template(formatted_history, tokenizer, add_generation_prompt=True)
                inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
                
                streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
                
                generate_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=config["dataset"].get("max_response_length", 1024),
                    use_cache=True,
                    temperature=0.4,
                    top_p=0.9
                )
                
                # Start generation in separate thread for non-blocking stream
                thread = Thread(target=model.generate, kwargs=generate_kwargs)
                thread.start()
                
                accumulated_text = ""
                for new_text in streamer:
                    accumulated_text += new_text
                    yield accumulated_text
                return
            except Exception as e:
                logger.error(f"Inference streaming error on GPU: {e}")
                yield f"An error occurred during GPU generation: {e}. Falling back to system diagnostics."
                return

        # --- CPU SIMULATION ROUTING (FOR DEMOS AND INSPECTIONS) ---
        import time
        lowered = message_cleaned.lower()
        
        # Check safety boundaries in CPU simulation
        if "marne" in lowered or "suicide" in lowered or "kill" in lowered or "jeene ka mann" in lowered:
            response_text = (
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
        elif "biopsy" in lowered or "cancer" in lowered or "medical" in lowered or "treatment" in lowered:
            response_text = (
                "I understand how incredibly stressful waiting for a biopsy or dealing with severe health concerns can be. "
                "As an AI, I must be clear: a birth chart is not a diagnostic medical test, and astrology cannot predict cancer, illness, or medical outcomes. "
                "Please rely entirely on your doctors and clinical diagnostics. Astrology's role is strictly as spiritual or emotional support, "
                "never as a substitute for professional healthcare. For peace of mind, you may practice gentle breathing exercises (Pranayama) "
                "or chanting, but please make medical appointments your primary focus."
            )
        elif "lottery" in lowered or "nifty" in lowered or "bitcoin" in lowered or "gambling" in lowered:
            response_text = (
                "ट्रेडिंग, सट्टा, या लॉटरी के संबंध में मैं बहुत स्पष्ट सलाह दूँगा: "
                "वैदिक ज्योतिष शेयर बाज़ार के रोज़मर्रा के उतार-चढ़ाव या सट्टेबाज़ी के परिणामों की भविष्यवाणी नहीं कर सकता। "
                "ग्रहों की चाल आपके स्वभाव और जोखिम लेने की आदतों को दर्शा सकती है, पर वे कभी भी मुनाफ़े या विजयी नंबरों की गारंटी नहीं दे सकते। "
                "कृपया अपनी मेहनत की कमाई को केवल ज्योतिषीय दावों के आधार पर जोखिम में न डालें। हमेशा वित्तीय विशेषज्ञों की राय लें।"
            )
        elif dob == "" or tob == "" or pob == "":
            response_text = (
                "Pranam! Thank you for sharing your concern. In order to analyze your chart and discuss planetary transits, "
                "I will need your birth details. Please enter your Date of Birth, Time of Birth, and Place of Birth in the "
                "input fields in the sidebar so I can construct your Kundli and provide balanced guidance. "
                "Always remember that planets show trends and pathways, but your conscious choices and actions shape your destiny!"
            )
        else:
            response_text = (
                f"Thank you for sharing your birth details (DOB: {dob}, TOB: {tob}, POB: {pob}).\n\n"
                "Looking at your astrological transits, you are currently entering a constructive phase influenced by Jupiter. "
                "This indicates an excellent time to focus on personal skill enhancement, structured career planning, "
                "and cultivating open, compassionate communication in your relationships.\n\n"
                "Vedic astrology guides us toward self-awareness and timely effort, rather than passive waiting or fear. "
                "Focus on steady daily efforts, and you will navigate this period successfully! "
                "How can I help you refine your current career or relationship strategy today?"
            )

        # Stream typing effect
        typed_msg = ""
        for char in response_text:
            typed_msg += char
            yield typed_msg
            time.sleep(0.005)

    # 2. Build Gradio Layout
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio is not installed. Please run pip install -r requirements.txt")
        sys.exit(1)

    # Clean styling definitions
    theme = gr.themes.Default(
        primary_hue="amber",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"]
    )

    with gr.Blocks(title="Vedaz AI Vedic Astrologer Dashboard", theme=theme) as demo:
        gr.Markdown(
            """
            # 🌌 Vedaz AI Vedic Astrologer
            ### Reference Responsible Astrology Fine-Tuning Sandbox
            *Powered by Unsloth QLoRA on Qwen2.5-7B-Instruct. Fusing ancient Vedic wisdom with strict safety compliance.*
            """
        )
        
        with gr.Row():
            # Birth details panel
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📜 Kundli Birth Details")
                dob_input = gr.Textbox(label="Date of Birth (DD-MM-YYYY)", placeholder="e.g., 25-09-1992", value="")
                tob_input = gr.Textbox(label="Time of Birth (HH:MM AM/PM)", placeholder="e.g., 02:15 PM", value="")
                pob_input = gr.Textbox(label="Place of Birth (City, State, Country)", placeholder="e.g., Kanpur, UP, India", value="")
                
                gr.Markdown(
                    """
                    ---
                    ### ⚖️ Ethics & Guardrails Summary
                    1. **Non-fatalistic**: Guidance is focused on remedies, self-growth, and efforts.
                    2. **No Life-Span predictions**: Refuses calculations of death/lifespan.
                    3. **Healthcare Crisis Refusal**: Redirects severe physical/mental health issues to medical specialists.
                    4. **No Financial Speculation**: Refuses to predict stock market direction, betting outcomes, or lottery tickets.
                    """
                )
                
            # Chat Interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, bubble_full_width=False)
                msg_input = gr.Textbox(
                    label="Ask a question about your Kundli, career, relationships, or growth...",
                    placeholder="Enter your query here... (e.g., How does this year look for my career?)"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send Message", variant="primary")
                    undo_btn = gr.Button("Undo Last")
                    retry_btn = gr.Button("Retry Response")
                    clear_btn = gr.Button("Clear Chat History")

        # Define state variables
        state_history = gr.State([])

        def user_submit(message, history):
            return "", history + [[message, ""]]

        def bot_stream(history, dob, tob, pob):
            user_msg = history[-1][0]
            # Convert history up to last query to tuples list
            history_tuples = [(h[0], h[1]) for h in history[:-1]]
            
            for chunk in predict_astrologer(user_msg, history_tuples, dob, tob, pob):
                history[-1][1] = chunk
                yield history

        # Connect actions
        submit_event = msg_input.submit(
            user_submit, [msg_input, chatbot], [msg_input, chatbot], queue=False
        ).then(
            bot_stream, [chatbot, dob_input, tob_input, pob_input], chatbot
        )
        
        submit_btn.click(
            user_submit, [msg_input, chatbot], [msg_input, chatbot], queue=False
        ).then(
            bot_stream, [chatbot, dob_input, tob_input, pob_input], chatbot
        )

        def undo_history(history):
            if len(history) > 0:
                history.pop()
            return history

        undo_btn.click(undo_history, chatbot, chatbot)

        def clear_history():
            return []

        clear_btn.click(clear_history, None, chatbot)

        def retry_response(history, dob, tob, pob):
            if len(history) > 0:
                last_msg = history[-1][0]
                history[-1][1] = ""
                history_tuples = [(h[0], h[1]) for h in history[:-1]]
                for chunk in predict_astrologer(last_msg, history_tuples, dob, tob, pob):
                    history[-1][1] = chunk
                    yield history

        retry_btn.click(retry_response, [chatbot, dob_input, tob_input, pob_input], chatbot)

    # Launch app
    logger.info(f"Launching Gradio web portal on port {args.port}...")
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
