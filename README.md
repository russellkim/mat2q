# Text-to-SQL with Multi-Agent Reflective Learning

## Overview
This project implements a multi-agent text-to-SQL system that converts natural language questions into SQLite-compatible SQL queries, evaluated on the Spider dataset [Yu et al., 2018]. It features a hierarchical architecture with style-specific SQL generation, a selector for choosing the best query, and reflective agents that iteratively improve performance by learning from past mistakes. The system is designed for research purposes, targeting high-impact contributions in top AI conferences (e.g., ACL, ICLR).

## Key Features
- **Multi-Agent SQL Generation**: Four `SQLPlanGeneratorAgent` instances generate SQL queries using distinct strategies:
  - **Default**: General-purpose SQL generation.
  - **Join-first**: Prioritizes explicit joins for multi-table queries.
  - **Subquery**: Emphasizes nested subqueries.
  - **Aggregation**: Focuses on aggregation logic (e.g., COUNT, SUM).
- **Selector Agent**: A `SelectorAgent` chooses the best SQL from candidates based on question intent, schema constraints, and past selector reflections, using an LLM (e.g., solar-pro2).
- **Reflective Learning**:
  - `ReflectionAgent`: Generates general and style-specific insights for base agents, stored as JSON in `knowledge/reflection_<timestamp>.json`.
  - `SelectorReflectionAgent`: Reflects on selector choices, stored in `knowledge/selector/selector_reflection_<timestamp>.json`.
- **Knowledge Summarization**: A `KnowledgeSummarizerAgent` periodically summarizes reflections (every `--summary-interval` examples) into concise insights (`summary_<timestamp>.json`, `selector_summary_<timestamp>.json`), reducing prompt bloat and enhancing generalization.
- **Performance Tracking**: Logs per-style and selector accuracy (correct/total and percentage) to evaluate strategy effectiveness.
- **Configurability**: Command-line arguments (`--start`, `--n`, `--summary-interval`) allow flexible experimentation on Spider dataset subsets.

## File Structure
- `runner.py`: Main script orchestrating dataset loading, agent execution, verification, reflection, selection, and statistics.
- `agents.py`: Defines agent classes (`SQLPlanGeneratorAgent`, `VerifierAgent`, `ReflectionAgent`, `SelectorAgent`, `SelectorReflectionAgent`, `KnowledgeSummarizerAgent`).
- `utils.py`: Utility functions for loading Spider dataset, schemas, and extracting SQL from LLM responses.
- `knowledge/`: Stores base agent reflections and summaries as JSON.
- `knowledge/selector/`: Stores selector reflections and summaries.

## Usage Instructions
### Prerequisites
- Python 3.8+
- Dependencies: `openai`, `datasets`, `sqlite3`, `python-dotenv`
  ```bash
  pip install openai datasets python-dotenv
  ```
- Spider dataset downloaded to `./datasets/spider`
- Environment variables: Set `UPSTAGE_API_KEY_0` and `UPSTAGE_API_BASE` in a `.env` file.

### Running the System
Run with default settings (start index 95, 5 examples, summarize every 10 examples):
```bash
python runner.py
```
Custom settings:
```bash
python runner.py --start 100 --n 20 --summary-interval 5
```
- `--start`: Starting index in Spider dev set.
- `--n`: Number of examples to process.
- `--summary-interval`: Frequency of reflection summarization.

Output includes per-example logs, reflections, selector choices, and final statistics (per-style and selector accuracy).

## Research Context
This system is designed for research in text-to-SQL generation, emphasizing iterative self-improvement through reflective learning. Key contributions include:
- Hierarchical agent architecture with style-specific generation and selection [Pan et al., 2023].
- Reflective feedback loops for base agents and selector, enhancing robustness [Shinn et al., 2023].
- Periodic knowledge summarization to optimize prompt efficiency [Pourreza et al., 2024].

Targeted for ACL/ICLR 2026 submissions, the system can be extended to explore prompt rewriting, uncertainty estimation, or cross-domain generalization (e.g., BIRD, WikiSQL).

## Future Work
- **Dynamic Prompt Rewriting**: Enable agents to rewrite their prompts based on reflections, harmonizing insights for adaptive learning.
- **Uncertainty Estimation**: Incorporate LLM confidence scores or ensemble voting to improve selector decisions.
- **Error Analysis**: Log error types (syntax vs. semantic) to refine reflections.
- **Multi-Domain Testing**: Evaluate on diverse datasets to test generalization.

## References
- [Yu et al., 2018] Yu, T., et al. "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task." *EMNLP 2018*.
- [Pan et al., 2023] Pan, Y., et al. "MAGIC: Investigation of Large Language Models for Uncertainty-Aware Multi-Agent Collaboration." *ICLR 2023*.
- [Shinn et al., 2023] Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023*.
- [Pourreza et al., 2024] Pourreza, M., et al. "Distilling Text-to-SQL with Large Language Models." *VLDB 2024*.