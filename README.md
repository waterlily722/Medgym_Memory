# MedGym Memory 轻量运行目录

这个目录是从原始 `rllm/examples/MedGym` 中拆出来的轻量版本，目标是专门用于：

- MedGym doctor agent 环境测试
- 有/无 memory 的诊断对照实验
- memory retrieval / merge / extraction 消融实验
- 轨迹、summary、memory trace 调试


## 目录包含什么

- `run_med_with_tool.py`：主运行入口，用于 doctor rollout、评估和日志保存。
- `memory_agent/`：记忆系统代码，包括：
  - Case Memory
  - Experience Memory
  - Skill Memory
  - Knowledge Memory
  - 在线检索、适用性判断、memory guidance
  - 离线 experience / skill 提取和 merge
- `rllm/`：MedGym 运行所需的最小本地 runtime 子集，包括：
  - `agents/`：base agent、tool agent、medical agent、system prompts
  - `engine/`：异步执行引擎、OpenAI-compatible rollout engine
  - `environments/`：base env、tool env、MedGym env
  - `tools/`：tool registry、multi-tool、medical tools
  - `rewards/`：医学诊断 reward 和 judge 调用逻辑
  - `utils/`：diagnose accuracy summary 和打印工具
- `prepare_med_data_bench.py`：bench 数据加载。
- 空的运行输出目录：
  - `logs/`
  - `trajectories/`
  - `memory_agent/memory_data/`


## 基本运行

进入新目录：

```bash
cd /oral_llm/xiweidai/med_env/code/medgym_memory
```

启动 retrieval server（同时用于 doctor 的 `retrieve` tool 和 `--retrieval_mode embedding`）：

```bash
cd /oral_llm/xiweidai/med_env/code/rllm

bash examples/search/retrieval/launch_server.sh \
  examples/search/guidelines/guidelines_index/e5_Flat.index \
  examples/search/guidelines/guidelines_index/corpus_passages.jsonl \
  8000 \
  INFO
```

确认 retrieval 和 embedding endpoint 都可用：

```bash
NO_PROXY=127.0.0.1,localhost curl -s http://127.0.0.1:8000/health

NO_PROXY=127.0.0.1,localhost curl -s http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"intfloat-e5-base-v2","input":["query: chest pain with fever"]}'
```

Memory embedding 会缓存到当前 `memory_root` 下的 `.embedding_cache/embeddings.jsonl`。
缓存按 `memory_id + memory_type + memory 文本 hash + embedding 模型配置` 命中，memory 内容不变时不会重复向量化；需要临时关闭缓存时可设置：

```bash
export MEDGYM_DISABLE_EMBEDDING_CACHE=1
```

不启用 memory：

```bash
python run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 200 \
  --repeat_k 1 \
  --no_cxr \
  --parser_name qwen \
  --trace_tag qwen_no_memory \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1
```

启用 memory，不注入 Case Memory（默认只注入检索到的 memory guidance）：

```bash
python run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 200 \
  --repeat_k 1 \
  --no_cxr \
  --parser_name qwen \
  --enable_memory \
  --trace_tag qwen_nocm \
  --log_memory_trace \
  --judge_model judge_agent \
  --execution_mode serial \
  --judge_base_url http://127.0.0.1:30002/v1
```

启用 memory，不注入 Case Memory，并使用 embedding 检索：

```bash
python run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 200 \
  --repeat_k 1 \
  --no_cxr \
  --parser_name qwen \
  --enable_memory \
  --log_memory_trace \
  --trace_tag qwen_nocm_embedding \
  --retrieval_mode embedding \
  --memory_embedding_model intfloat-e5-base-v2 \
  --memory_embedding_base_url http://127.0.0.1:8000/v1 \
  --execution_mode serial \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1
```

启用 memory + 注入 Case Memory（每轮将 compact case memory 暴露给 doctor agent）：

```bash
python run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 10 \
  --repeat_k 1 \
  --no_cxr \
  --parser_name qwen \
  --enable_memory \
  --log_memory_trace \ß
  --inject_case_memory \
  --trace_tag case_memory_exp \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1
```
  启动 CXR grounding server：

  conda activate vllm_env
  cd /oral_llm/xiweidai/med_env/code/medgym_memory
  bash start_grounding_server.sh

  然后跑 CXR 版本（去掉 --no_cxr）：

  python run_med_with_tool.py \
    --model doctor_agent \
    --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
    --base_url http://127.0.0.1:30000/v1 \
    --case_dir /oral_llm/xiweidai/med_env/bench \
    --max_cases 10 \
    --repeat_k 1 \
    --parser_name qwen \
    --enable_memory \
    --log_memory_trace \
    --judge_model judge_agent \
    --judge_base_url http://127.0.0.1:30002/v1

    
## Qwen / DeepSeek API

doctor model 和 judge model 都可以使用 OpenAI-compatible Chat Completions API。

当前已支持：

- 本地 vLLM
- DeepSeek
- 阿里云 Qwen / DashScope
- 其他兼容 `/v1/chat/completions` 的服务

### 使用阿里云 Qwen 作为 doctor

```bash
export DASHSCOPE_API_KEY="sk-..."

python run_med_with_tool.py \
  --provider qwen \
  --model qwen-plus \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 5 \
  --repeat_k 1 \
  --no_cxr \
  --parser_name qwen
```

### 只使用阿里云 Qwen 作为 judge

```bash
export DASHSCOPE_API_KEY="sk-..."

python run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 5 \
  --repeat_k 1 \
  --no_cxr \
  --parser_name qwen \
  --judge_provider qwen \
  --judge_model qwen-plus
```

### DeepSeek

```bash
export DEEPSEEK_API_KEY="sk-..."

python run_med_with_tool.py \
  --provider deepseek \
  --model deepseek-chat \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 5 \
  --repeat_k 1 \
  --no_cxr \
  --parser_name qwen
```

## API 默认配置

Qwen 默认配置：

- 中国站 endpoint：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 中国站默认模型：`qwen-plus`
- API key 环境变量：`DASHSCOPE_API_KEY` 或 `QWEN_API_KEY`
- 模型名环境变量：`QWEN_MODEL` 或 `DASHSCOPE_MODEL`

DeepSeek 默认配置：

- endpoint：`https://api.deepseek.com/v1`
- 默认模型：`deepseek-chat`
- API key 环境变量：`DEEPSEEK_API_KEY`

## 常用参数

| 参数 | 说明 |
|------|------|
| `--enable_memory` | 启用 memory wrapper |
| `--log_memory_trace` | 保存每个 case 的 memory trace |
| `--inject_case_memory` | 每轮将 compact CaseMemory 注入 doctor agent 的 observation（主诉、诊断目标、已获证据、prior summary） |
| `--trace_tag` | memory trace 和 trajectory 目录后缀，用于区分不同实验，默认无后缀 |
| `--disable_memory_write` | 冻结 memory store，只检索不写入 |
| `--retrieval_mode` | 在线检索方式：`fielded_bm25` / `cosine` / `embedding`，默认 `fielded_bm25` |
| `--merge_scoring_mode` | 离线 merge 候选召回方式，默认 `same_as_retrieval` |
| `--query_builder_mode` | CaseMemory 构建模式：`rule` / `llm`，默认 `llm` |
| `--applicability_mode` | 记忆适用性判断模式：`rule` / `llm` / `hybrid`，默认 `llm` |
| `--experience_extraction_mode` | 经验提取模式：`rule` / `llm`，默认 `llm` |
| `--experience_merge_mode` | 经验合并模式：`rule` / `llm`，默认 `llm` |
| `--judge_model` | 开启 LLM judge，用于判断最终诊断是否正确 |
| `--judge_provider` | judge provider：`auto` / `local` / `deepseek` / `qwen` / `custom` |
| `--memory_llm_model` | memory 系统专用 LLM，默认复用 doctor model |
| `--memory_root` | memory 数据目录，默认是 `memory_agent/memory_data` |
| `--trajectory_dir` | 完整 trajectory 输出目录，默认是 `trajectories/`，会覆盖外部 `RLLM_TRAJECTORY_DIR` |
| `--max_steps` | 每 case 最大 doctor-agent 轮次，默认 15 |
| `--execution_mode` | `parallel` / `serial`，serial 强制单线程 |

## 输出位置

默认输出到当前轻量目录下：

- run log：`logs/run_*.log`
- summary log：`logs/summary_*.log`
- trajectory：`trajectories/trajectory_*.json`（启用 `--trace_tag` 时输出到 `trajectories/<tag>/`）
- memory trace：`memory_agent/memory_data/trace/*.json`（启用 `--trace_tag` 时输出到 `memory_agent/memory_data/trace_<tag>/`）
- experience memory：`memory_agent/memory_data/experience_memory.jsonl`
- skill memory：`memory_agent/memory_data/skill_memory.jsonl`
- knowledge memory：`memory_agent/memory_data/knowledge_memory.jsonl`

## 说明

从 `code/medgym_memory` 目录运行时，Python 会优先使用本目录下的轻量 `rllm/` 包，因此不需要依赖原始大代码库里的训练模块。
