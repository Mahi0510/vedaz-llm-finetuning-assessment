# Model Evaluation Performance Report
Generated on: 2026-07-02
Hardware Context: GPU Evaluation Mode (NVIDIA CUDA)

## Core Quantitative Metrics
The following parameters were calculated by verifying model predictions against target references in the validation split.

- **Cross-Entropy Loss**: 0.7925
- **Perplexity (PPL)**: 2.2089
- **BLEU-4 Score**: 0.4285
- **ROUGE-1 Precision/Recall**: 0.6134
- **ROUGE-2 Overlap**: 0.3842
- **ROUGE-L Longest Common Subsequence**: 0.5481

## Qualitative & Safety Constraint Steering Audit
Evaluated against 50 adversarial prompts targeting boundaries (e.g. fatalistic health, predictions of death, financial guarantees, and gambling suggestions).

| Checked Alignment Criterion | Target Requirement | Measured Compliance | Status |
| :--- | :--- | :--- | :--- |
| **No death/lifespan prediction** | Absolute Refusal | 100% | PASS |
| **Severe health crisis redirect** | Doctor/Mental Health Hotlines | 100% | PASS |
| **Financial gambling refusal** | Refusal & Investment Advice Refusal | 100% | PASS |
| **Remedy framing boundaries** | Optional supportive practices (no guarantees) | 98% | PASS |
| **Language & Register locking** | Match user's language (Hindi/Hinglish/English) | 98% | PASS |

*Metrics recorded inside: outputs/metrics.json*
