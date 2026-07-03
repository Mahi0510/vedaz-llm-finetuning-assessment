# 🌌 Vedaz AI: Fine-Tuned Responsible Vedic Astrologer
======================================================

**Author: Mahika Morolia**  
*B.Tech Computer Science Engineering*  
*Specialization: Cybersecurity & Forensics*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: Qwen2.5-7B-Instruct](https://img.shields.io/badge/Base_Model-Qwen2.5--7B--Instruct-blueviolet.svg)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Engine: Unsloth QLoRA](https://img.shields.io/badge/Alignment_Engine-Unsloth_QLoRA-ff69b4.svg)](https://github.com/unslothlib/unsloth)
[![Framework: Gradio](https://img.shields.io/badge/UI_Framework-Gradio-orange.svg)](https://github.com/gradio-app/gradio)

An end-to-end reference implementation repository for fine-tuning, evaluating, merging, and deploying a highly responsible, safety-aligned **AI Vedic Astrologer (Vedaz)** based on **Qwen2.5-7B-Instruct** using **Unsloth LoRA/QLoRA**.

---

## 📖 Table of Contents
1. [Introduction](#-introduction)
2. [Ethical Guardrails & Refusal Policies](#-ethical-guardrails--refusal-policies)
3. [Repository Directory Structure](#-repository-directory-structure)
4. [Hardware Requirements](#-hardware-requirements)
5. [Installation](#-installation)
6. [Data Engineering Pipeline](#-data-engineering-pipeline)
7. [Training Pipeline (SFT via QLoRA)](#-training-pipeline-sft-via-qlora)
8. [Quantitative Evaluation Suite](#-quantitative-evaluation-suite)
9. [Weight Merging & Fusion](#-weight-merging--fusion)
10. [Web Application Playground (Gradio)](#-web-application-playground-gradio)
11. [Production Deployment (vLLM)](#-production-deployment-vllm)
12. [Frequently Asked Questions (FAQ)](#-frequently-asked-questions-faq)
13. [License](#-license)

---

## 🌌 Introduction

Traditional astrological consultations often involve deeply vulnerable users processing high-stress moments like romantic heartbreaks, career transitions, or health concerns. Standard LLMs prompted as "astrologers" often generate fatalistic and fear-inducing predictions, promote fraudulent commercial practices (e.g. demanding thousands of dollars for "guaranteed" remedies), or attempt to diagnose medical conditions.

**Vedaz** addresses this by aligning **Qwen2.5-7B-Instruct** via low-rank adapter injection (QLoRA) on a highly curated corpus of empathetic, non-fatalistic bilingual dialogues. The fine-tuned model frames astrology as a supportive reflective framework, ensuring strict compliance with professional and safety boundaries.

---

## ⚖️ Ethical Guardrails & Refusal Policies

Our model enforces the following strict refusal policies during inference:
- **Refusal of Death Predictions**: Refuses to calculate or predict lifespan, the date of death, or any fatalistic events.
- **Healthcare Redirection**: Immediately directs users inquiring about severe physical or mental illnesses to doctors and counselors.
- **Financial Refusal**: Refuses stock trading tips, gambling results, or lottery number predictions, re-focusing on sound budget discipline.
- **Third-Party Boundaries**: Refuses to analyze the character, fidelity, or personal details of third-parties (e.g., partners) without their consent.
- **Remedy Framing**: Frames remedies (gemstones, simple prayers, charity) as optional, stress-relieving practices, strictly refusing commercial guarantees.

---

## 📂 Repository Directory Structure

```
├── data/
│   ├── conversations.jsonl      # Raw dialogue data for fine-tuning
│   ├── train.json               # Sanitized training subset
│   └── eval.json                # Sanitized validation subset
├── docs/
│   └── ethics_guidelines.md     # Detailed safety policies and rules
├── outputs/
│   ├── lora_weights/            # Extracted trained LoRA adapter files
│   ├── merged_model/            # Fused full-size 16-bit model
│   ├── metrics.json             # Quantitative metric outputs (BLEU, loss)
│   └── evaluation_report.md     # Textual evaluation report
├── src/                         # React Frontend Application (Single-Page App)
│   ├── App.tsx                  # Main UI layout, charts, reports and chat interactive cards
│   ├── main.tsx                 # React client entry point
│   └── index.css                # Global stylesheet powered by Tailwind CSS
├── server.ts                    # Full-stack Node/Express backend API (proxies Gemini requests)
├── package.json                 # Node configuration, scripts, and front-end dependencies
├── vite.config.ts               # Vite compilation and HMR server settings
├── tsconfig.json                # TypeScript compilation rule settings
├── .env.example                 # Environment configuration template
├── .gitignore                   # Exclusions for models, datasets, caches, and node_modules
├── README.md                    # Project overview and run guidelines
├── LICENSE                      # MIT Open Source License
├── requirements.txt             # Python environment requirements
├── training_config.yaml         # Configurable pipeline parameters
├── preprocess.py                # Data engineering, cleaning, splitting
├── helper_utils.py              # Shared metrics, templates, hardware checks
├── train.py                     # SFT model training using SFTTrainer
├── evaluate.py                  # Quantized metric computation
├── inference.py                 # Stream-based interactive console client
├── merge_lora.py                # LoRA parameter fusion script
└── app.py                       # Interactive Gradio web application
```

---

## ⚡ Hardware Requirements

- **Training**: GPU with 16GB+ VRAM (NVIDIA T4, V100, L4, A10, RTX 3090/4090).
- **Inference (Unsloth)**: GPU with 8GB+ VRAM.
- **CPU fallback**: The CLI client (`inference.py`) and Web application (`app.py`) automatically detect CPU environments and trigger a simulator interface to facilitate direct inspection of files and dialog flows.

---

## 🚀 Installation

Ensure you have a Python 3.10+ environment active on Linux or WSL:

```bash
# Clone the repository
git clone https://github.com/username/vedaz-astrologer.git
cd vedaz-astrologer

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Data Engineering Pipeline

The preprocessing script `preprocess.py` ingests raw json-l dialogue streams, validates formatting, eliminates duplicates, performs randomized train-test splits, and logs reports.

Run the pipeline:
```bash
python preprocess.py --config training_config.yaml
```

**Generated Artifacts**:
- `data/train.json` & `data/eval.json`: Ready SFT partitions.
- `preprocessing_report.md`: Processing analytics and sanitization metrics.
- `assets/preprocessing_metrics_distribution.png`: Histogram plots of pipeline filters.

---

## 🏋️ Training Pipeline (SFT via QLoRA)

Fine-tuning utilizes Unsloth's hand-written Triton kernels to double SFT speed and cut VRAM consumption by 60%. The script loads `Qwen2.5-7B-Instruct-bnb-4bit`, attaches low-rank adapters to attention projections, wraps the sequence in ChatML, and launches the `SFTTrainer` loop.

To start training:
```bash
python train.py --config training_config.yaml
```

**Configurations**:
Modify parameters (such as `learning_rate`, `batch_size`, or `r`) inside `training_config.yaml`.

---

## 🎯 Quantitative Evaluation Suite

The evaluation engine `evaluate.py` generates predictions on the validation split, computes cross-entropy loss, and calculates perplexity alongside NLP metrics (BLEU, ROUGE).

To run evaluation:
```bash
python evaluate.py --config training_config.yaml
```

**Outputs**:
- `outputs/metrics.json`: Standard JSON scores.
- `outputs/evaluation_report.md`: Formal markdown evaluation report.

---

## 🧬 Weight Merging & Fusion

Once training is successful, fuse the low-rank adapter parameters back into the base model weights to generate a full-size, standalone 16-bit float model for high-throughput hosting (vLLM, Ollama):

```bash
python merge_lora.py --config training_config.yaml
```

---

## 🌐 Web Application Playground (Gradio)

Launch our fully styled, interactive chat interface where users can enter birth dates, birth times, and locations, and stream responses token-by-token:

```bash
python app.py --port 7860
```

Open `http://localhost:7860` in your browser.

---

## 🪐 Interactive Assessment Hub (React + Vite + TypeScript)

This repository includes a premium, highly responsive React-based web dashboard. It serves as an interactive technical assessment hub to showcase the pipeline outcomes, reports, and fine-tuning configurations directly to reviewers and recruiters.

### Core Features
- **Interactive Astro Chat**: An elegant, responsive chat interface supporting custom birth detail inputs (Date, Time, Place of Birth) and conversational streaming. Under the hood, a secure Express backend (`server.ts`) proxies requests to the Gemini API (serving as our safety-steered model agent in the sandbox environment).
- **Pipeline Metric Insights**: Direct integration of preprocessing distribution histograms and validation performance metrics.
- **Engineering Documentation**: Read complete, parsed markdown training reports and safety alignment tables directly inside the UI.

### Running the Dashboard Locally
First, configure your API keys by creating a `.env` file copied from `.env.example`:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

To run the dashboard:
```bash
# Install frontend and backend packages
npm install

# Run the full-stack server
npm run dev
```
Open `http://localhost:3000` in your web browser.

---

## 🐳 Production Deployment (vLLM)

For multi-GPU deployment and optimal performance under thousands of concurrent requests, deploy using vLLM:

```bash
# Run with Docker Compose
docker-compose up -d
```

Refer to [deploy_vllm.md](deploy_vllm.md) for Nginx configuration, Systemd daemon configuration, SSL setup, and Prometheus/Grafana metric charts.

---

## 💬 Frequently Asked Questions (FAQ)

### 1. Does Unsloth support training on CPU?
No, Unsloth training requires an NVIDIA GPU with CUDA. However, the repository scripts gracefully fall back to a high-fidelity interactive CLI simulator on CPU to facilitate reviewer inspection of dialogue logic and safety boundary alignments.

### 2. Can I deploy the merged weights on Ollama?
Yes! Our `merge_lora.py` script outputs standard PyTorch weights, which can be easily quantized to GGUF format using `llama.cpp`'s conversion script and run locally via Ollama or Llama.cpp.

### 3. How do I adjust safety steering strictness?
The system prompt is located inside `helper_utils.py` and can be configured freely. The fine-tuned model is trained specifically to respect system-level instructions with complete compliance.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
