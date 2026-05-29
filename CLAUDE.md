# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedGym Memory is a medical diagnosis agent framework with a memory-augmented system. It is a lightweight, self-contained fork from `rllm/examples/MedGym` that runs doctor-patient diagnostic dialogues with optional memory retrieval, guidance, and offline experience extraction. The codebase and all README comments are in Chinese.

## Running Experiments

The main entry point is `run_med_with_tool.py`. It requires several external services to be running (Doctor/Patient/Judge LLMs on vLLM, a retrieval server, and optionally a CXR grounding server). See `README.memory_agent.md` for full deployment instructions.

```bash
# Without memory (baseline)
python run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 10 --repeat_k 1 --no_cxr --parser_name qwen \
  --judge_model judge_agent --judge_base_url http://127.0.0.1:30002/v1

# With memory
python run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 10 --repeat_k 1 --no_cxr --parser_name qwen \
  --enable_memory --log_memory_trace \
  --judge_model judge_agent --judge_base_url http://127.0.0.1:30002/v1

# Run experiment suites (quick/main/full)
bash run_memory_experiments.sh
EXPERIMENT_SUITE=quick bash run_memory_experiments.sh
```

Cloud providers are supported via `--provider deepseek|qwen` with corresponding API key env vars (`DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`).

## Architecture

### Data Flow

```
run_med_with_tool.py
  → AgentExecutionEngine (rllm/engine/) — orchestrates async agent-env loops
    → MemoryWrappedMedicalAgent (memory_agent/wrapper.py) — wraps the base ToolAgent
      → Online pipeline (per turn):
          observation → update_case_state → build_memory_query → retrieve_multi_memory
          → apply_applicability_control → build_memory_guidance → inject into observation
      → Offline pipeline (on episode done):
          turn_records → distill_from_trajectory → extract_experiences → merge_experience
          → write to ExperienceMemoryStore / SkillMemoryStore
    → MedicalDialogueEnv (rllm/environments/medgym/) — simulates patient interaction
    → med_diagnosis_reward (rllm/rewards/) — result + process reward, supports LLM judge
```

### Key Modules

- **`memory_agent/wrapper.py`** (`MemoryWrappedMedicalAgent`): Central class that wraps the base doctor agent. Overrides `update_from_env` and `update_from_model` to intercept the agent-environment loop. Manages case state, triggers the online memory pipeline each turn, and runs offline memory extraction when an episode completes.

- **`memory_agent/online/`**: Per-turn memory pipeline. `case_updater.py` → `query_builder.py` → `retriever.py` → `applicability_controller.py` → `memory_guidance.py`. Each step has `rule` and `llm` modes. Trace logging is in `memory_trace.py`.

- **`memory_agent/offline/`**: Post-episode processing. `episode_distiller.py` distills trajectory into a structured episode; `experience_extractor.py` extracts experience cards (LLM or rule); `experience_merger.py` merges new experiences with existing ones; `skill_extractor.py` extracts action-sequence skills from correct episodes; `memory_writer.py` orchestrates the write pipeline.

- **`memory_agent/memory_store/`**: JSONL-backed persistent stores for experience, skill, and knowledge memory. All data lives under `memory_agent/memory_data/`.

- **`memory_agent/schemas/`**: Dataclasses for all structured data (`CaseState`, `MemoryQuery`, `ExperienceCard`, `SkillCard`, `MemoryGuidance`, `ApplicabilityResult`, `EpisodeFeedback`, `TurnRecord`, etc.). All inherit serialization via a common mixin.

- **`memory_agent/llm/`**: Lightweight OpenAI-compatible HTTP client (`LLMClient`) and embedding client (`EmbeddingClient`). No dependency on the `openai` SDK — uses `urllib` directly.

- **`memory_agent/utils/config.py`**: All tunable parameters — `RETRIEVAL_CONFIG` (scoring mode, top-k, thresholds), `MERGE_CONFIG`, `SKILL_CONFIG`, `MEMORY_ACTION_CONFIG` (tool→action mapping), `APPLICABILITY_CONFIG`, `TRACE_CONFIG`, `LLM_CONFIG`.

- **`rllm/`**: Minimal local runtime fork. Contains the agent base class, execution engine, MedGym environment, medical tools (ask_patient, diagnosis, retrieve, cxr, cxr_grounding, request_exam), and reward functions. This package is self-contained and does not depend on the full `rllm` codebase.

### Memory Retrieval Modes

Three scoring modes, selected via `--retrieval_mode`:
- **`fielded_bm25`** (default): Field-weighted BM25 over experience/skill text fields
- **`cosine`**: Token-level cosine similarity
- **`embedding`**: Dense embedding via an external embedding API (requires `--memory_embedding_model` and `--memory_embedding_base_url`)

Merge scoring follows retrieval by default (`--merge_scoring_mode same_as_retrieval`) for paired ablations.

### Important Behaviors

- **Fail-fast by default**: When `--allow_memory_fallback` is not set, any LLM failure or JSON parse error in the memory pipeline raises an exception rather than silently falling back to rule-based logic.
- **Memory write freeze**: `--disable_memory_write` freezes the store (read-only) for retrieval-only ablation runs.
- **Memory data location**: Defaults to `memory_agent/memory_data/`. Configurable via `--memory_root`. The `MEMORY_ROOT_DIRNAME` constant in `utils/config.py` resolves to the absolute path of this directory.
- **Environment variable `MEDGYM_RETRIEVAL_FALLBACK_SCORING`**: Set at startup by `run_med_with_tool.py` based on `--retrieval_mode`; the retriever reads it when embeddings are unavailable.
- **Output**: Run logs go to `logs/`, trajectories to `trajectories/`, memory traces to `memory_agent/memory_data/trace/`.

## Conventions

- Python source is a mix of English and Chinese comments. Module docstrings and README files are in Chinese.
- The `rllm/` package here is a lightweight local copy — changes should stay compatible with the parent `rllm` project at `/oral_llm/xiweidai/med_env/code/rllm`.
- `conda activate vllm_env` is the standard environment.
- When accessing local HTTP services, use `NO_PROXY=127.0.0.1,localhost` if an HTTP proxy is set in the environment.
