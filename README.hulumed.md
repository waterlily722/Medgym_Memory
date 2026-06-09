# Hulu-Med-14B 接入 MedGym Memory

使用 Hulu-Med-14B 模型替代 Qwen3-VL-8B 运行 doctor / patient / judge / memory 四个角色。

## 模型信息

```text
模型路径:  /data/xiweidai/models/Hulu-Med-14B
模型大小:  ~28GB (bf16)
最低需求:  2× RTX 3090 (24GB × 2)
服务脚本:  serve_hulumed_openai.py（基于 transformers + accelerate，device_map="auto" 自动跨卡）
```

> Hulu-Med-14B 不走 vLLM（自定义 modeling 无法被 vLLM 直接加载），使用 `serve_hulumed_openai.py` 启动 OpenAI 兼容 HTTP 服务。

## 环境准备

```bash
conda activate vllm_env

# 确认 accelerate 已安装（device_map="auto" 依赖）
python -c "import accelerate; print('accelerate', accelerate.__version__)"
```

## 启动服务

Hulu-Med-14B 约 28GB，需要 2 张 3090 做模型并行。以下以 7×3090 为例分配 GPU。

### 终端 A — Doctor (port 30000, GPU 0,1)

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/medgym_memory

python serve_hulumed_openai.py \
  --model_path /data/xiweidai/models/Hulu-Med-14B \
  --served_model_name hulumed-14b \
  --host 127.0.0.1 \
  --port 30000 \
  --gpus 0,1 \
  --dtype bfloat16
```

### 终端 B — Patient (port 30001, GPU 2,3)

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/medgym_memory

python serve_hulumed_openai.py \
  --model_path /data/xiweidai/models/Hulu-Med-14B \
  --served_model_name hulumed-14b \
  --host 127.0.0.1 \
  --port 30001 \
  --gpus 2,3 \
  --dtype bfloat16
```

### 终端 C — Judge (port 30002, GPU 4,5)

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/medgym_memory

python serve_hulumed_openai.py \
  --model_path /data/xiweidai/models/Hulu-Med-14B \
  --served_model_name hulumed-14b \
  --host 127.0.0.1 \
  --port 30002 \
  --gpus 4,5 \
  --dtype bfloat16
```

### 终端 D — Memory LLM (port 30003, GPU 6 + 复用一张)

> Memory 角色可以用 Doctor 的同一个服务（默认行为），也可以单独启一个实例。单独启动只需 1 张卡不够（28GB > 24GB），建议复用 GPU 6 和 Doctor/Patient/Judge 中的任一张卡。

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/medgym_memory

python serve_hulumed_openai.py \
  --model_path /data/xiweidai/models/Hulu-Med-14B \
  --served_model_name hulumed-14b \
  --host 127.0.0.1 \
  --port 30003 \
  --gpus 6,0 \
  --dtype bfloat16
```

> **不单独启动 Memory 的方案**：不加这个终端，memory 默认复用 Doctor 的 `--model` 和 `--base_url`，不需要额外参数。

### 终端 E — Retrieval Server (port 8000)

与 Qwen 方案共用，启动方式不变：

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/rllm

bash examples/search/retrieval/launch_server.sh \
  examples/search/guidelines/guidelines_index/e5_Flat.index \
  examples/search/guidelines/guidelines_index/corpus_passages.jsonl \
  8000 \
  INFO
```

### 验证服务就绪

```bash
NO_PROXY=127.0.0.1,localhost curl -s http://127.0.0.1:30000/v1/models
NO_PROXY=127.0.0.1,localhost curl -s http://127.0.0.1:30001/v1/models
NO_PROXY=127.0.0.1,localhost curl -s http://127.0.0.1:30002/v1/models
```

每个应返回：
```json
{"object":"list","data":[{"id":"hulumed-14b","object":"model","created":0,"owned_by":"local"}]}
```

### 后台启动方式（tmux）

```bash
# Doctor
tmux new-session -d -s hulumed_doctor \
  'cd /oral_llm/xiweidai/med_env/code/medgym_memory && conda run -n vllm_env \
   python serve_hulumed_openai.py --model_path /data/xiweidai/models/Hulu-Med-14B \
   --served_model_name hulumed-14b --port 30000 --gpus 0,1 --dtype bfloat16'

# Patient
tmux new-session -d -s hulumed_patient \
  'cd /oral_llm/xiweidai/med_env/code/medgym_memory && conda run -n vllm_env \
   python serve_hulumed_openai.py --model_path /data/xiweidai/models/Hulu-Med-14B \
   --served_model_name hulumed-14b --port 30001 --gpus 2,3 --dtype bfloat16'

# Judge
tmux new-session -d -s hulumed_judge \
  'cd /oral_llm/xiweidai/med_env/code/medgym_memory && conda run -n vllm_env \
   python serve_hulumed_openai.py --model_path /data/xiweidai/models/Hulu-Med-14B \
   --served_model_name hulumed-14b --port 30002 --gpus 4,5 --dtype bfloat16'

# Memory（可选，不启动则复用 Doctor）
tmux new-session -d -s hulumed_memory \
  'cd /oral_llm/xiweidai/med_env/code/medgym_memory && conda run -n vllm_env \
   python serve_hulumed_openai.py --model_path /data/xiweidai/models/Hulu-Med-14B \
   --served_model_name hulumed-14b --port 30003 --gpus 6,0 --dtype bfloat16'
```

停止：

```bash
tmux kill-session -t hulumed_doctor
tmux kill-session -t hulumed_patient
tmux kill-session -t hulumed_judge
tmux kill-session -t hulumed_memory
```

## 运行评测

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/medgym_memory
export PYTHONPATH="/oral_llm/xiweidai/med_env/code/rllm:/oral_llm/xiweidai/med_env/code/rllm/examples/MedGym"

# Patient 环境变量
export RLLM_PATIENT_MODEL="hulumed-14b"
export RLLM_PATIENT_BASE_URL="http://127.0.0.1:30001/v1"
export RLLM_PATIENT_API_KEY="None"

# Retrieval / Grounding
export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000"
export RLLM_GROUNDING_API_URL="http://127.0.0.1:30050/ground"
export RLLM_TRAJECTORY_DIR="/oral_llm/xiweidai/med_env/code/medgym_memory/trajectories"
```

### 无 Memory（baseline）

```bash
python run_med_with_tool.py \
  --provider hulumed \
  --model hulumed-14b \
  --tokenizer_path /data/xiweidai/models/Hulu-Med-14B \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 10 --repeat_k 1 --no_cxr --parser_name qwen \
  --judge_model hulumed-14b \
  --judge_base_url http://127.0.0.1:30002/v1
```

### 有 Memory（memory 复用 Doctor 实例）

```bash
python run_med_with_tool.py \
  --provider hulumed \
  --model hulumed-14b \
  --tokenizer_path /data/xiweidai/models/Hulu-Med-14B \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 10 --repeat_k 1 --no_cxr --parser_name qwen \
  --enable_memory --log_memory_trace \
  --judge_model hulumed-14b \
  --judge_base_url http://127.0.0.1:30002/v1
```

### 有 Memory（memory 单独实例 port 30003）

```bash
python run_med_with_tool.py \
  --provider hulumed \
  --model hulumed-14b \
  --tokenizer_path /data/xiweidai/models/Hulu-Med-14B \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 10 --repeat_k 1 --no_cxr --parser_name qwen \
  --enable_memory --log_memory_trace \
  --memory_llm_model hulumed-14b \
  --memory_llm_base_url http://127.0.0.1:30003/v1 \
  --judge_model hulumed-14b \
  --judge_base_url http://127.0.0.1:30002/v1
```

### 有 CXR 模式

去掉 `--no_cxr` 即可，需提前启动 CXR grounding server：

```bash
conda activate vllm_env
cd /oral_llm/xiweidai/med_env/code/rllm
bash examples/MedGym/start_grounding_server.sh
```

```bash
python run_med_with_tool.py \
  --provider hulumed \
  --model hulumed-14b \
  --tokenizer_path /data/xiweidai/models/Hulu-Med-14B \
  --base_url http://127.0.0.1:30000/v1 \
  --case_dir /oral_llm/xiweidai/med_env/bench \
  --max_cases 10 --repeat_k 1 --parser_name qwen \
  --enable_memory --log_memory_trace \
  --judge_model hulumed-14b \
  --judge_base_url http://127.0.0.1:30002/v1
```

## GPU 分配参考

| 角色 | 端口 | GPU | 显存占用 |
|------|------|-----|---------|
| Doctor | 30000 | 0, 1 | ~28GB / 48GB |
| Patient | 30001 | 2, 3 | ~28GB / 48GB |
| Judge | 30002 | 4, 5 | ~28GB / 48GB |
| Memory (可选) | 30003 | 6, 0 | ~28GB / 48GB |
| Retrieval | 8000 | CPU / 任意 | ~0.5GB |
| Grounding (可选) | 30050 | 任意 | ~2GB |

> 7×3090 共 168GB。Doctor + Patient + Judge 各占 2 卡 = 6 卡（~84GB），剩余 GPU 6 可与 Doctor 的 GPU 0 共同承载 Memory 实例。如不需要独立 Memory 实例，Memory 默认复用 Doctor 服务，无需额外资源。

## 与 Qwen 方案的差异

| 项目 | Qwen3-VL-8B | Hulu-Med-14B |
|------|-------------|-------------|
| 服务方式 | vLLM | transformers + custom HTTP server |
| 单卡需求 | 1× 3090 够 | 需 2× 3090 (device_map="auto") |
| tool call 格式 | Qwen XML 原生输出 | JSON → 服务端自动转换为 Qwen XML |
| 启动脚本 | `vllm serve ...` | `python serve_hulumed_openai.py ...` |
| `--provider` | `local` | `hulumed` |
| `--parser_name` | `qwen` | `qwen`（服务端已做格式适配） |
