# Text-to-SQL with Multi-Agent Reflective Learning

## Overview
This project implements a multi-agent text-to-SQL system that converts natural language questions into SQLite-compatible SQL queries, evaluated on the Spider dataset [Yu et al., 2018]. It features a hierarchical architecture with style-specific SQL generation, a selector for choosing the best query, and reflective agents that iteratively improve performance by learning from past mistakes. The system includes dynamic prompt rewriting and reflection resets, logged for analysis, making it a robust platform for research into self-evolving AI systems, targeting top AI conferences like ACL and ICLR.

## Key Features
- **Multi-Agent SQL Generation**: Four `SQLPlanGeneratorAgent` instances generate SQL queries using distinct strategies:
  - **Default**: General-purpose SQL generation.
  - **Join-first**: Prioritizes explicit joins for multi-table queries.
  - **Subquery**: Emphasizes nested subqueries.
  - **Aggregation**: Focuses on aggregation logic (e.g., COUNT, SUM).
- **Selector Agent**: A `SelectorAgent` chooses the best SQL from candidates using an external prompt (`prompt/prompt_selector.txt`), informed by question intent, schema constraints, and past selector reflections.
- **Reflective Learning**:
  - `ReflectionAgent`: Generates general and style-specific insights for base agents, stored as JSON in `knowledge/reflection_<timestamp>.json`.
  - `SelectorReflectionAgent`: Reflects on selector choices, stored in `knowledge/selector/selector_reflection_<timestamp>.json`, with resets to empty (`{"general": []}`) after prompt rewrites to focus on new insights.
- **Prompt Rewriting**: A `SelectorAgentTeacher` rewrites the selector prompt based on accumulated reflections when the `"general"` list exceeds `--rewrite-selector-size` and the reflection file is at least `--min-reflection-age` examples old. Previous prompts are backed up as `prompt/prompt_selector_<timestamp>.txt`.
- **Reset Logging**: Prompt rewrite events are logged in `prompt/manifest.json` with metadata (e.g., timestamp, pre-reset reflection count, selector accuracy, prompt diff summary) for analysis and debugging.
- **Performance Tracking**: Logs per-style and selector accuracy (correct/total and percentage) to evaluate strategy effectiveness.
- **Configurability**: Command-line arguments (`--start`, `--n`, `--rewrite-selector-size`, `--min-reflection-age`) allow flexible experimentation on Spider dataset subsets.

## File Structure
- `runner.py`: Orchestrates dataset loading, agent execution, verification, reflection, selection, prompt rewriting, and statistics.
- `agents.py`: Defines agent classes (`SQLPlanGeneratorAgent`, `VerifierAgent`, `ReflectionAgent`, `SelectorAgent`, `SelectorReflectionAgent`, `SelectorAgentTeacher`).
- `utils.py`: Utility functions for loading Spider dataset, schemas, and extracting SQL from LLM responses.
- `knowledge/`: Stores base agent reflections as JSON (`reflection_<timestamp>.json`).
- `knowledge/selector/`: Stores selector reflections (`selector_reflection_<timestamp>.json`), reset to empty after prompt rewrites.
- `prompt/prompt_selector.txt`: External prompt for `SelectorAgent`, dynamically rewritten.
- `prompt/prompt_selector_<timestamp>.txt`: Backups of previous selector prompts.
- `prompt/manifest.json`: Logs metadata for prompt rewrite events.

## Usage Instructions
### Prerequisites
- Python 3.8+
- Dependencies: `openai`, `datasets`, `sqlite3`, `python-dotenv`
  ```bash
  pip install openai datasets python-dotenv
  ```
- Spider dataset downloaded to `./datasets/spider`.
- Environment variables: Set `UPSTAGE_API_KEY_0` and `UPSTAGE_API_BASE` in a `.env` file.

### Running the System
Run with default settings (start index 95, 5 examples, rewrite prompt after 10 reflections, min reflection age 5):
```bash
python runner.py
```
Custom settings:
```bash
python runner.py --start 100 --n 20 --rewrite-selector-size 8 --min-reflection-age 3
```
- `--start`: Starting index in Spider dev set.
- `--n`: Number of examples to process.
- `--rewrite-selector-size`: Threshold for number of selector reflection items to trigger prompt rewrite.
- `--min-reflection-age`: Minimum examples since last reflection reset to allow rewrite.

Output includes per-example logs, base/selector reflections, selector choices, rewritten prompts, and final statistics (per-style and selector accuracy, rewrite events, reflection file count).

## Research Context
This system is designed for research in text-to-SQL generation, emphasizing self-improving agents through reflective learning and dynamic prompt evolution. Key contributions include:
- Hierarchical architecture with style-specific generation and selection [Pan et al., 2023].
- Reflective feedback loops for base agents and selector, with periodic resets to focus learning [Shinn et al., 2023].
- Dynamic prompt rewriting with versioned backups and metadata logging for interpretability [Huang et al., 2024].
- Configurable reset conditions (`--min-reflection-age`) to balance knowledge retention and innovation.

Targeted for ACL/ICLR 2026 submissions, the system can explore prompt evolution, continual learning, and cross-domain generalization (e.g., BIRD, WikiSQL).

## Future Work
- **Prompt Rollback**: Implement automatic rollback to previous prompts if selector accuracy drops significantly, using `manifest.json` data.
- **Dynamic Rewrite Thresholds**: Adjust `--rewrite-selector-size` based on query complexity or performance trends.
- **Error Analysis**: Log error types (syntax vs. semantic) in reflections for finer-grained learning.
- **Semantic Deduplication**: Use embeddings (e.g., Sentence-BERT) to deduplicate reflections while preserving diversity [Reimers and Gurevych, 2019].
- **Multi-Domain Testing**: Evaluate on diverse datasets to test generalization.

## References
- [Huang et al., 2024] Huang, J., et al. “Internal Consistency and Self-Feedback in Large Language Models: A Survey.” *arXiv preprint arXiv:2407.07985*.
- [Pan et al., 2023] Pan, Y., et al. “MAGIC: Investigation of Large Language Models for Uncertainty-Aware Multi-Agent Collaboration.” *ICLR 2023*.
- [Pourreza et al., 2024] Pourreza, M., et al. “Distilling Text-to-SQL with Large Language Models.” *VLDB 2024*.
- [Reimers and Gurevych, 2019] Reimers, N., and Gurevych, I. “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.” *EMNLP 2019*.
- [Shinn et al., 2023] Shinn, N., et al. “Reflexion: Language Agents with Verbal Reinforcement Learning.” *NeurIPS 2023*.
- [Wang et al., 2020] Wang, B., et al. “RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers.” *ACL 2020*.
- [Yu et al., 2018] Yu, T., et al. “Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.” *EMNLP 2018*.