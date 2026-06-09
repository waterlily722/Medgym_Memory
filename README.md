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
  --max_cases 3470 \
  --repeat_k 5 \
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
  --max_cases 500 \
  --repeat_k 1 \
  --no_cxr \
  --parser_name qwen \
  --enable_memory \
  --log_memory_trace \
  --trace_tag qwen_nocm_embedding \
  --retrieval_mode embedding \
  --memory_embedding_model intfloat-e5-base-v2 \
  --memory_embedding_base_url http://127.0.0.1:8000/v1 \
  --memory_llm_model memory_llm \
  --memory_llm_base_url http://127.0.0.1:30003/v1 \
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
  --max_cases 200 \
  --repeat_k 1 \
  --no_cxr \
  --parser_name qwen \
  --enable_memory \
  --inject_case_memory \
  --log_memory_trace \
  --trace_tag qwen_embedding_case_memory \
  --retrieval_mode embedding \
  --memory_embedding_model intfloat-e5-base-v2 \
  --memory_embedding_base_url http://127.0.0.1:8000/v1 \
  --memory_llm_model memory_llm \
  --memory_llm_base_url http://127.0.0.1:30003/v1 \
  --execution_mode serial \
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
    --base_url http://127.0.0.1:30000/v1 \
    --case_dir /oral_llm/xiweidai/med_env/bench \
    --max_cases 401 \
    --repeat_k 5 \
    --parser_name qwen \
    --trace_tag qwen_no_memory_with_img \
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

DeepSeek doctor 使用原生 Function Calling。不要传入 `--parser_name qwen`；
程序会自动使用内部 `native` parser，并要求响应包含
`message.tool_calls`。

```bash
cd /oral_llm/xiweidai/med_env/code/medgym_memory
export DEEPSEEK_API_KEY="sk-..."
export RLLM_PATIENT_BASE_URL="https://api.deepseek.com/v1"
export RLLM_PATIENT_MODEL="deepseek-chat"
export RLLM_PATIENT_API_KEY="$DEEPSEEK_API_KEY"

# 先用 5 个 case 验证 doctor、patient/exam 和 judge 的连接
python run_med_with_tool.py \
  --provider deepseek \
  --model deepseek-chat \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 5 \
  --repeat_k 1 \
  --no_cxr \
  --execution_mode serial \
  --trace_tag ds_smoke_test \
  --judge_provider deepseek \
  --judge_model deepseek-chat
```

### DeepSeek 作为全部 LLM agent

DeepSeek doctor 使用官方 OpenAI-compatible Function Calling：请求发送
`tools`，响应读取 `message.tool_calls`，工具结果使用原始 `tool_call_id`
回传。DeepSeek provider 会自动使用内部 `native` parser，不经过 Qwen XML
tool-call parser。Patient、exam、judge 和 memory LLM 使用同一个 DeepSeek
Chat Completions endpoint。

```bash
cd /oral_llm/xiweidai/med_env/code/medgym_memory
export DEEPSEEK_API_KEY="sk-xxx"
export RLLM_PATIENT_BASE_URL="https://api.deepseek.com/v1"
export RLLM_PATIENT_MODEL="deepseek-chat"
export RLLM_PATIENT_API_KEY="$DEEPSEEK_API_KEY"
```

配置来源：

- doctor：`--provider deepseek`、`--model deepseek-chat` 和
  `DEEPSEEK_API_KEY`
- patient/exam：`RLLM_PATIENT_*`
- judge：`--judge_provider deepseek`，默认读取 `DEEPSEEK_API_KEY`
- memory LLM：`--memory_llm_*`；未显式设置时继承 doctor 配置
- embedding：本地 E5 服务，不属于 LLM agent

不用 memory 的 baseline：

```bash
python run_med_with_tool.py \
  --provider deepseek \
  --model deepseek-chat \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 5 \
  --repeat_k 1 \
  --max_steps 20 \
  --no_cxr \
  --execution_mode serial \
  --trace_tag ds_no_memory \
  --judge_provider deepseek \
  --judge_model deepseek-chat
```

使用 memory，但不注入 CaseMemory：

以下命令会运行 CaseMemory 构建、query、检索、experience/skill 提取和
experience merge，但不会把 CaseMemory 注入 doctor observation。保持命令中
没有 `--inject_case_memory` 即可。

```bash
# 另一个终端启动本地 memory embedding 服务
python -m vllm.entrypoints.openai.api_server \
  --model /oral_llm/xiweidai/med_env/models/intfloat-e5-base-v2 \
  --port 30010 \
  --convert embed \
  --runner pooling \
  --dtype auto

# DeepSeek doctor/patient/judge/memory + 本地 E5 memory retrieval
python run_med_with_tool.py \
  --provider deepseek \
  --model deepseek-chat \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 200 \
  --repeat_k 1 \
  --max_steps 15 \
  --no_cxr \
  --execution_mode serial \
  --enable_memory \
  --log_memory_trace \
  --trace_tag ds_memory_embedding_no_case_memory \
  --query_builder_mode llm \
  --applicability_mode llm \
  --experience_extraction_mode llm \
  --experience_merge_mode llm \
  --retrieval_mode embedding \
  --merge_scoring_mode embedding \
  --memory_embedding_model intfloat-e5-base-v2 \
  --memory_embedding_base_url http://127.0.0.1:30010/v1 \
  --memory_llm_model deepseek-chat \
  --memory_llm_base_url https://api.deepseek.com/v1 \
  --memory_llm_api_key "$DEEPSEEK_API_KEY" \
  --judge_provider deepseek \
  --judge_model deepseek-chat
```

上述命令默认 fail-fast：doctor 原生 tool call、patient、judge、memory LLM
或 embedding 任一路径失败都会中止，不会回退到 XML、rule、token cosine
或默认 patient answer。Qwen tokenizer 只用于本地 token 计数，不决定远程
DeepSeek doctor 模型。

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
| `--trace_tag` | 实验标签；用于区分 trajectory，并默认作为 memory store 子目录名 |
| `--memory_root_by_trace_tag` / `--no-memory-root-by-trace-tag` | 默认按 `trace_tag` 隔离整个 memory store；关闭后多个实验共享 `memory_root` |
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
- trajectory：`trajectories/trajectory_*.json`（设置 `--trace_tag TAG` 后输出到 `trajectories/TAG/`）
- memory store：设置 `--trace_tag TAG` 后，默认整体输出到 `memory_agent/memory_data/TAG/`
- memory trace：`memory_agent/memory_data/TAG/trace/*.json`
- experience memory：`memory_agent/memory_data/TAG/experience_memory.jsonl`
- skill memory：`memory_agent/memory_data/TAG/skill_memory.jsonl`
- knowledge memory：`memory_agent/memory_data/TAG/knowledge_memory.jsonl`
- embedding cache：`memory_agent/memory_data/TAG/.embedding_cache/embeddings.jsonl`

如果需要多个 `trace_tag` 检索和更新同一个共享 memory store，添加
`--no-memory-root-by-trace-tag`。此时 memory 文件继续写入 `--memory_root`
指定的根目录，memory trace 使用 `trace_<tag>/` 区分。

按 tag 隔离时不会自动复制根目录中已有的 memory。新 tag 第一次运行会从空
store 开始；如需继续使用根目录中的已有 memory，请使用
`--no-memory-root-by-trace-tag` 或显式准备对应 tag 子目录。

新的 run log 和 summary log 使用相同的实验时间戳，并在文件开头记录
`EXPERIMENT CONFIG`：完整脱敏命令、解析后的参数、实际生效的服务配置、
Git commit/dirty 状态、prompt 指纹，以及运行开始时 memory store 的行数和 SHA256。

使用 `--retrieval_mode embedding` 时，在线检索只使用 embedding cosine；
embedding 服务或向量生成失败会直接报错，不会回退到 token cosine/BM25。
当 `--experience_merge_mode llm` 存在 merge 候选时，LLM 服务失败或输出无效
也会直接报错，不会回退到 rule merge。
Experience merge 使用相似度筛选并排序候选，但只把前 5 条完整
ExperienceCard 传给 LLM，不向 LLM 暴露 embedding similarity score。

## 说明

从 `code/medgym_memory` 目录运行时，Python 会优先使用本目录下的轻量 `rllm/` 包，因此不需要依赖原始大代码库里的训练模块。
