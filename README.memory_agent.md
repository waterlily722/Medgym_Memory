# MedMemory — Medical Memory Agent

医学诊断任务的记忆增强系统。基于 rllm 框架，支持在线记忆检索与离线经验提取。

## 目录结构

```
memory_agent/
├── schemas/           # 数据模型（dataclass + SerializableMixin）
│   ├── case_state.py      # 当前病例状态
│   ├── experience_card.py # 经验记忆卡
│   ├── skill_card.py      # 技能记忆卡
│   ├── knowledge_item.py  # 知识条目
│   ├── memory_query.py    # 检索查询
│   ├── retrieval.py       # 检索结果
│   ├── applicability.py   # 适用性评估
│   ├── guidance.py        # 记忆引导
│   ├── turn_record.py     # 单步记录
│   └── episode.py         # 回合反馈
├── online/            # 在线推理 pipeline
│   ├── case_updater.py           # 病例状态更新
│   ├── query_builder.py          # 检索查询构建
│   ├── retriever.py              # 多源记忆检索（支持 embedding）
│   ├── applicability_controller.py # 记忆适用性判断
│   └── memory_guidance.py        # 引导生成
├── offline/           # 离线经验提取 pipeline
│   ├── episode_distiller.py      # 轨迹蒸馏
│   ├── experience_extractor.py   # 经验提取（LLM）
│   ├── experience_merger.py      # 经验合并
│   ├── memory_writer.py          # 记忆写入
│   └── skill_extractor.py        # 从诊断正确 episode 提取动作序列技能
├── memory_store/      # JSONL 持久化
├── llm/               # LLM / Embedding 客户端
├── utils/             # 工具函数（tokenizer, cosine similarity）
├── tests/             # 单元测试（76 tests）
└── wrapper.py         # MemoryWrappedMedicalAgent 主入口
```

## 环境准备

```bash
# 1. 激活 conda 环境
conda activate vllm_env

# 2. 安装 rllm（首次）
cd /oral_llm/xiweidai/med_env/code/rllm
pip install -e .

# 3. 验证导入
cd /oral_llm/xiweidai/med_env/code/rllm
PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm/examples/MedGym" \
  python -c "from memory_agent import MemoryWrappedMedicalAgent; print('memory_agent OK')"

# 4. （可选）MedGym 环境依赖 — 跑完整 benchmark 时需要
#    包含 decord / moviepy / opencv / scikit-learn 等，编译较慢
pip install -r /oral_llm/xiweidai/med_env/code/rllm/examples/MedGym/requirements.txt
```

## 运行测试

```bash
# 单元测试（93 个，无需 LLM/GPU）
cd /oral_llm/xiweidai/med_env/code/rllm/examples/MedGym
PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm" python -m unittest discover -s tests -v

# memory_agent 内部测试
PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm" python -c "
import sys; sys.path.insert(0, 'memory_agent/tests')
for m in ['test_schemas','test_scoring','test_parser','test_store','test_integration','test_e2e_pipeline']:
    __import__(m); print(f'{m}: OK')
print('All imports OK')
"
```

## 部署测试（需 GPU）

运行 `run_med_with_tool.py` 前，需要先启动 5 类服务：

1. Doctor LLM：`http://127.0.0.1:30000/v1`
2. Patient LLM：`http://127.0.0.1:30001/v1`
3. Judge LLM：`http://127.0.0.1:30002/v1`
4. Retrieval server：`http://127.0.0.1:8000`
5. CXR grounding server：`http://127.0.0.1:30050`

`run_med_with_tool.py` 只会连接这些服务，不会自动启动它们。看到 `Could not connect to retrieval server: [Errno 111] Connection refused` 时，通常表示 8000 端口的 retrieval server 没有启动，或者启动后异常退出。

### 终端 A — Doctor (port 30000)

```bash
conda activate vllm_env
export CUDA_VISIBLE_DEVICES=6,1
vllm serve /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name doctor_agent \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --max-model-len 64k \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

### 终端 B — Patient (port 30001)

```bash
conda activate vllm_env
export CUDA_VISIBLE_DEVICES=4,5
vllm serve /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name patient_agent \
  --host 0.0.0.0 \
  --port 30001 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code
```

### 终端 C — Judge (port 30002)

```bash
conda activate vllm_env
export CUDA_VISIBLE_DEVICES=2,3
vllm serve /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --served-model-name judge_agent \
  --host 0.0.0.0 \
  --port 30002 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.7 \
  --trust-remote-code
```

### 验证服务就绪

```bash
curl -s http://127.0.0.1:30000/v1/models | head
curl -s http://127.0.0.1:30001/v1/models | head
curl -s http://127.0.0.1:30002/v1/models | head
```

### 终端 D — Retrieval Server (port 8000)

Retrieval 依赖 guidelines corpus、FAISS index 和 E5 embedding 模型：

```text
corpus: /oral_llm/xiweidai/med_env/code/rllm/examples/search/guidelines/guidelines_index/corpus_passages.jsonl
index:  /oral_llm/xiweidai/med_env/code/rllm/examples/search/guidelines/guidelines_index/e5_Flat.index
model:  /oral_llm/xiweidai/med_env/models/intfloat-e5-base-v2
```

如果本地没有 E5 模型，先通过镜像下载：

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env

HF_ENDPOINT=https://hf-mirror.com hf download intfloat/e5-base-v2 \
  --repo-type model \
  --local-dir /oral_llm/xiweidai/med_env/models/intfloat-e5-base-v2
```

启动 retrieval server：

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/rllm

bash examples/search/retrieval/launch_server.sh \
  examples/search/guidelines/guidelines_index/e5_Flat.index \
  examples/search/guidelines/guidelines_index/corpus_passages.jsonl \
  8000 \
  INFO
```

`launch_server.sh` 会优先使用本地模型 `/oral_llm/xiweidai/med_env/models/intfloat-e5-base-v2`。如果想手动指定模型路径：

```bash
RETRIEVER_MODEL_PATH=/path/to/intfloat-e5-base-v2 \
bash examples/search/retrieval/launch_server.sh \
  examples/search/guidelines/guidelines_index/e5_Flat.index \
  examples/search/guidelines/guidelines_index/corpus_passages.jsonl \
  8000 \
  INFO
```

### 终端 E — CXR Grounding Server (port 30050)

无 CXR 模式加了 `--no_cxr` 时可以不启动它；有 CXR 模式必须启动。

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/rllm

bash examples/MedGym/start_grounding_server.sh
```

默认模型路径：

```text
/data/xuxiang/mimic-iv/models/grounding-dino-base
```

如需指定端口或模型目录：

```bash
PORT=30050 MODEL_DIR=/path/to/grounding-dino-base \
bash examples/MedGym/start_grounding_server.sh
```

### 后台启动方式

如果不想开多个终端，可以用 `tmux` 挂后台：

```bash
tmux new-session -d -s medgym_retrieval \
  'cd /oral_llm/xiweidai/med_env/code/rllm && conda run -n vllm_env bash examples/search/retrieval/launch_server.sh examples/search/guidelines/guidelines_index/e5_Flat.index examples/search/guidelines/guidelines_index/corpus_passages.jsonl 8000 INFO'

tmux new-session -d -s medgym_grounding \
  'cd /oral_llm/xiweidai/med_env/code/rllm && conda run -n vllm_env bash examples/MedGym/start_grounding_server.sh'
```

查看日志：

```bash
tmux attach -t medgym_retrieval
tmux attach -t medgym_grounding
```

停止服务：

```bash
tmux kill-session -t medgym_retrieval
tmux kill-session -t medgym_grounding
```

### 验证辅助服务

因为当前环境可能设置了 HTTP 代理，访问本机服务时建议显式绕过代理：

```bash
NO_PROXY=127.0.0.1,localhost curl -s http://127.0.0.1:8000/health
NO_PROXY=127.0.0.1,localhost curl -s http://127.0.0.1:30050/health

ss -ltnp | grep -E ':(8000|30050)\b'
```

### 运行评测

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/rllm
export PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm:/oral_llm/xiweidai/med_env/code/rllm/examples/MedGym"
export RLLM_PATIENT_BASE_URL="http://127.0.0.1:30001/v1"
export RLLM_PATIENT_MODEL="patient_agent"
export RLLM_PATIENT_API_KEY="None"
export RLLM_TRAJECTORY_DIR="/oral_llm/xiweidai/med_env/code/rllm/examples/MedGym/trajectories"
export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000"
export RLLM_GROUNDING_API_URL="http://127.0.0.1:30050/ground"
# export MEMORY_LLM_MODEL="your_model_name"
# export MEMORY_LLM_BASE_URL="https://your-api-host/v1"
# export MEMORY_LLM_API_KEY="your_api_key"


# 无 CXR 模式（默认 5 个 case）
python examples/MedGym/run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 50 \
  --no_cxr \
  --parser_name qwen \
  --enable_memory \
  --log_memory_trace \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1 

# 没有MEMORY
python examples/MedGym/run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 50 \
  --no_cxr \
  --parser_name qwen \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1 

export DEEPSEEK_API_KEY="your_deepseek_api_key"

python examples/MedGym/run_med_with_tool.py \
  --provider deepseek \
  --model deepseek-v4-flash \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --no_cxr \
  --max_cases 3 \
  --repeat_k 1



# 有 CXR 模式（去掉 --no_cxr 即可）
python examples/MedGym/run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 1 \
  --parser_name qwen \
  --judge_model judge_agent \
  --judge_base_url http://127.0.0.1:30002/v1 \
  --enable_memory \
  --log_memory_trace
```

## 记忆系统参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_memory` | off | 启用记忆系统 |
| `--memory_root` | `memory_agent/memory_data` | 记忆存储根目录；留空或传 `memory_data` 都会落到该目录 |
| `--query_builder_mode` | `llm` | 检索查询构建模式 (`rule` / `llm`) |
| `--applicability_mode` | `llm` | 适用性判断模式 (`rule` / `llm` / `hybrid`) |
| `--experience_extraction_mode` | `llm` | 经验提取模式 (`rule` / `llm`) |
| `--experience_merge_mode` | `llm` | 经验合并模式 (`rule` / `llm`) |
| `--log_memory_trace` | off | 记录每步记忆 trace 到文件 |
| `--disable_experience_memory` | off | 禁用经验记忆 |
| `--disable_skill_memory` | off | 禁用技能记忆 |
| `--disable_knowledge_memory` | off | 禁用知识记忆 |
| `--memory_llm_model` | 主 doctor 模型 | 记忆系统专用 LLM 模型名，默认复用 `--model` |
| `--memory_llm_base_url` | 主 doctor 地址 | 记忆系统专用 LLM 地址，默认复用 `--base_url` |
| `--allow_memory_fallback` | off | 默认 fail-fast：memory LLM 不可用、JSON 解析失败或 schema 不合法会直接报错；开启后才允许退回 rule 逻辑 |

### Memory Trace 输出

开启 `--log_memory_trace` 后，trace 会写入 `memory_agent/memory_data/trace/`：

- `<case_id>.json`：推荐阅读的聚合 memory trace，按 `turns[]` 保存 memory 构造链路摘要。它不同于 `trajectory_*.json`：trajectory 负责保存完整对话/工具轨迹，memory trace 只保存 `case_state`、`memory_query`、检索 hit 摘要、applicability 决策、guidance 和 selected action。
- 每个 turn 都有 `episode_id`、`case_id`、`turn_id`。`episode_id` 绑定病人 case，例如 `case_20002287_repeat_0`；`turn_id` 是该 episode 内的当前轮次，从 1 开始。
- `memory_query` 基于当前 turn 已进入 `CaseState` 的信息生成。也就是说，本轮 observation 会先更新 `CaseState`，再构造 query、检索 memory、生成 guidance。
- 每个 turn 的 `memory_debug` 用于调试 memory 构造链路，包含 `case_state_update`、`candidate_actions`、`query_builder`、`retrieval`、`applicability`、`guidance`。默认不保存完整 observation、LLM prompt、raw output，避免和 trajectory 重复；需要深调 LLM IO 时，可在 `memory_agent/utils/config.py` 的 `TRACE_CONFIG` 中打开 `include_llm_io` / `include_prompt_payload` / `include_observation_payload`。
- `<case_id>.jsonl`：兼容旧调试流程的逐步 memory snapshot 日志，默认关闭；需要时在 `TRACE_CONFIG.write_jsonl_snapshot` 中打开。


# baseline: no memory
python examples/MedGym/run_med_with_tool.py ... \
  --max_cases 50 \
  --no_cxr

# memory + cosine retrieve + cosine merge
python examples/MedGym/run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 50 \
  --no_cxr \
  --enable_memory \
  --retrieval_mode cosine \
  --disable_memory_write \
  --merge_scoring_mode same_as_retrieval \
  --log_memory_trace

# memory + BM25 retrieve + BM25 merge
python examples/MedGym/run_med_with_tool.py \
  --model doctor_agent \
  --tokenizer_path /oral_llm/xiweidai/med_env/models/Qwen3-VL-8B-Instruct \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 50 \
  --no_cxr \
  --enable_memory \
  --retrieval_mode fielded_bm25 \
    --disable_memory_write \
  --merge_scoring_mode same_as_retrieval \
  --log_memory_trace
