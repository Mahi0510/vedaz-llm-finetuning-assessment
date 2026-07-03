# Data Preprocessing Pipeline Report
Generated on: 2026-07-02
Author: Mahika Morolia
B.Tech Computer Science Engineering
Specialization: Cybersecurity & Forensics

## Dataset Cleaning Summary
The raw ingestion file `data/conversations.jsonl` was scrutinized for structural integrity, correct formatting, duplication patterns, and language appropriateness.

| Pipeline Check | Metrics | Description |
| :--- | :--- | :--- |
| Ingested Raw Lines | 32 | Count of strings read from file |
| Malformed JSON Lines | 0 | Lines failed to parse via JSON deserializer |
| Invalid Structure Lines | 0 | Entries with missing dialogue properties or <2 turns |
| Duplicate Conversations | 0 | Matches against already registered conversations |
| Sanitized Conversations | 32 | Output dataset ready for fine-tuning |
| System Prompts Detected | 32 | Dialogues containing safety steering parameters |
| Avg. Turns Per Dialogue | 9.42 | Standard dialog conversational depth |

## Split Allocations
- **Training Subset (`data/train.json`)**: 27 samples
- **Validation Subset (`data/eval.json`)**: 5 samples

*Status: Data Pipeline Completed with Green Status.*
