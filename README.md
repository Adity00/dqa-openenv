---
title: DQA OpenEnv - Data Quality Assurance Environment
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - data-quality
  - rl-environment
---

# DQA-OpenEnv: Data Quality Assurance Environment

> A reinforcement learning environment where AI agents learn to clean
> real-world tabular datasets. Built for the Scaler x Meta PyTorch
> OpenEnv Hackathon.

## Environment Overview

DQA-OpenEnv simulates one of the most common and expensive tasks in
data engineering: cleaning dirty tabular datasets. An AI agent receives
a corrupted DataFrame and must apply structured data-cleaning actions
step-by-step to restore it to a known clean ground truth.

**Real-world utility:** Data professionals spend 60-80% of their time
cleaning data. An agent trained on DQA-OpenEnv learns transferable
skills applicable to real ETL pipelines.

## Action Space

The agent can take the following actions at each step:

| Action               | Description                  | Key Parameters                             |
| -------------------- | ---------------------------- | ------------------------------------------ |
| `fill_nulls`         | Fill missing values          | `strategy`: mean/median/mode/constant/drop |
| `drop_duplicates`    | Remove duplicate rows        | `subset`: column list (optional)           |
| `cast_type`          | Fix wrong data types         | `target_type`: int/float/str/bool          |
| `normalize_category` | Unify categorical variants   | `mapping`: {old: new} dict                 |
| `clip_outliers`      | Cap extreme values           | `method`: iqr/zscore                       |
| `drop_column`        | Remove a column              | (no parameters needed)                     |
| `filter_rows`        | Remove rows by condition     | `condition`: pandas query string           |
| `merge_categories`   | Consolidate rare categories  | `from_values`, `to_value`                  |
| `submit`             | Finalize and trigger grading | (ends the episode)                         |
| `noop`               | Do nothing (penalized)       | (no parameters needed)                     |

**Example action (JSON):**

```json
{
  "action_type": "fill_nulls",
  "column": "age",
  "parameters": { "strategy": "median" }
}
```

## Observation Space

After each action, the agent receives a `DQAObservation` containing:

| Field                | Type       | Description                                                        |
| -------------------- | ---------- | ------------------------------------------------------------------ |
| `task_description`   | str        | Plain English goal for this episode                                |
| `dataset_preview`    | list[dict] | First 10 rows of current dataset                                   |
| `column_stats`       | dict       | Per-column: null count, dtype, unique count, sample values         |
| `issue_hints`        | list[str]  | Auto-detected problems in plain English                            |
| `quality_scores`     | dict       | Live scores: completeness, consistency, uniqueness, validity (0-1) |
| `step_count`         | int        | Current step number                                                |
| `max_steps`          | int        | Episode budget                                                     |
| `action_history`     | list[str]  | Previous actions taken                                             |
| `last_action_result` | str        | Success/failure message                                            |
| `done`               | bool       | Whether episode has ended                                          |
| `reward`             | float      | Step reward (-1.0 to +1.0)                                         |

## Reward Function

Rewards are shaped to provide signal at every step (not just at the end):

- **Correct action**: `+0.1 to +0.3` (scaled by how much quality improved)
- **Wasted action** (on clean column): `-0.05`
- **Destructive action** (destroyed valid data): `-0.15 to -0.30`
- **noop**: `-0.02` per use
- **submit** (before max steps): `+0.05` efficiency bonus
- **Timeout** (hit max steps): `-0.10` penalty on final score

The step reward is the **delta** in overall quality score × 2.0.
This means agents receive immediate feedback on whether each action helped.

## Tasks

### Task 1: Customer Survey Formatting (Easy)

- **Dataset**: 200 rows, 6 columns (customer survey data)
- **Issues**: 3 columns with nulls (10-15%), wrong dtype on age, trailing space in column name
- **Max steps**: 20
- **Target score for GPT-3.5**: 0.75 - 0.90
- **Key challenge**: Identify which null strategy to use per column

### Task 2: HR Employee Records Cleaning (Medium)

- **Dataset**: ~620 rows, 8 columns (HR employee data)
- **Issues**: 120 duplicate rows, department name variants (Eng/engineering/ENGINEERING), mixed date formats, one column with 40% nulls (should be DROPPED not filled), salary nulls
- **Max steps**: 25
- **Target score for GPT-3.5**: 0.60 - 0.75
- **Key challenge**: Recognize that dropping performance_score is better than filling it

### Task 3: Financial Transactions Quality (Hard)

- **Dataset**: ~1080 rows, 12 columns (financial transactions)
- **Issues**: processing_fee nulls (fill with 0.0 NOT mean), amount outliers (CLIP not drop), merchant_category casing, negative purchase amounts (filter rows), status inconsistency, 80 duplicate rows
- **Max steps**: 30
- **Target score for GPT-3.5**: 0.45 - 0.65
- **Key challenge**: Order matters. Domain knowledge required (fees = 0, not mean).

## Grading

Episodes are graded across 4 quality dimensions:

| Dimension        | Description                                       |
| ---------------- | ------------------------------------------------- |
| **Completeness** | How well null values were handled vs ground truth |
| **Consistency**  | How well categorical values match clean values    |
| **Uniqueness**   | How well duplicates were removed                  |
| **Validity**     | How well values fall within expected ranges       |

Scores are weighted differently per task to reflect each task's character.
Final score: `0.0 to 1.0`. Grade: `A (≥0.85) / B (≥0.70) / C (≥0.55) / D (≥0.40) / F`.

## Baseline Scores

Baseline scores using `gpt-3.5-turbo` with 8 steps per task:

| Task     | Score     | Grade | Steps Used |
| -------- | --------- | ----- | ---------- |
| Easy     | ~0.85     | A/B   | 6-8        |
| Medium   | ~0.72     | B     | 7-8        |
| Hard     | ~0.58     | C     | 8          |
| **Mean** | **~0.72** | **B** | —          |

_Scores are reproducible with `TEMPERATURE=0.0` and fixed dataset seeds._

## Setup & Usage

### Requirements

- Python 3.10+
- Docker (for containerized deployment)
- OpenAI API key or HuggingFace token

### Local Installation

```bash
git clone https://github.com/Adity00/dqa-openenv.git
cd dqa-openenv
python -m venv venv
venv\Scripts\activate  # Windows
pip install openenv-core pandas numpy openai
```

### Run the Environment Server Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Baseline Inference

```bash
export OPENAI_API_KEY=your-key-here
export MODEL_NAME=gpt-3.5-turbo
python inference.py
```

### Docker

```bash
# Build
docker build -t dqa-openenv -f server/Dockerfile .

# Run
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your-key \
  -e MODEL_NAME=gpt-3.5-turbo \
  dqa-openenv
```

### Test the API

```bash
# Reset environment (easy task)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "fill_nulls", "column": "age", "parameters": {"strategy": "median"}}'

# Get state
curl http://localhost:7860/state
```

## Project Structure

```
dqa-openenv/
├── inference.py              # Baseline agent script (mandatory)
├── models.py                 # DQAAction, DQAObservation, DQAState
├── client.py                 # HTTP/WebSocket client
├── openenv.yaml              # OpenEnv spec metadata
├── pyproject.toml            # Package definition
└── server/
    ├── app.py                # FastAPI server entry point
    ├── dqa_openenv_environment.py  # Core environment logic
    ├── Dockerfile            # Container definition
    ├── requirements.txt      # Server dependencies
    ├── tasks/                # Task definitions (easy/medium/hard)
    ├── datasets/             # Dirty+clean DataFrame generators
    ├── graders/              # Deterministic scoring engine
    └── rewards/              # Delta reward calculator
```

## API Reference

| Endpoint  | Method | Description                                                  |
| --------- | ------ | ------------------------------------------------------------ |
| `/reset`  | POST   | Start new episode. Body: `{"task_id": "easy\|medium\|hard"}` |
| `/step`   | POST   | Execute action. Body: DQAAction JSON                         |
| `/state`  | GET    | Get current episode state                                    |
| `/health` | GET    | Container health check                                       |
| `/docs`   | GET    | Auto-generated OpenAPI docs                                  |
| `/ws`     | WS     | WebSocket for persistent sessions                            |

## Environment Variables

| Variable         | Required        | Default                     | Description                                   |
| ---------------- | --------------- | --------------------------- | --------------------------------------------- |
| `OPENAI_API_KEY` | Yes (inference) | —                           | OpenAI API key for LLM calls                  |
| `HF_TOKEN`       | Yes (inference) | —                           | HuggingFace token (alternative to OpenAI key) |
| `API_BASE_URL`   | No              | `https://api.openai.com/v1` | LLM API endpoint                              |
| `MODEL_NAME`     | No              | `gpt-3.5-turbo`             | Model to use for inference                    |

## Competition Info

- **Hackathon**: Scaler x Meta PyTorch OpenEnv Challenge
- **Submission**: https://huggingface.co/spaces/Adity00/dqa-openenv
- **GitHub**: https://github.com/Adity00/dqa-openenv
- **Deadline**: April 8, 2026
