# /home/xuxiang/virtual_env/code/rllm/examples/MedGym/test_doctor_dialog.py
import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

# from rllm.agents import ToolAgent
from rllm.agents.system_prompts import DOCTOR_SYSTEM_PROMPT, DOCTOR_SYSTEM_PROMPT_wo_IMG
from memory_agent.wrapper import MemoryWrappedMedicalAgent

from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.medgym.medgym_env import MedicalDialogueEnv
# from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.tools import register_med_tools
from rllm.tools.multi_tool import MultiTool
from rllm.utils.diagnose_acc import evaluate_doctor_results
# from prepare_med_data import prepare_med_data
from rllm.rewards.reward_fn import search_reward_fn
from rllm.rewards.med_diagnosis_reward import med_diagnosis_reward  # 结果+过程奖励：诊断包含 + ask_patient 置信度

from prepare_med_data_bench import prepare_med_data, SUBDIR_WITH_CXR, SUBDIR_NO_CXR
from memory_agent.utils.config import MEMORY_ROOT_DIRNAME


LOCAL_OPENAI_BASE_URL = "http://localhost:30000/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_DEFAULT_MODEL = "qwen-plus"
PATH_TAG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _redact_mapping(values: dict) -> dict:
    return {
        key: ("***" if "api_key" in key.lower() and value not in ("", None) else value)
        for key, value in values.items()
    }


def _redacted_command(argv: list[str]) -> str:
    redacted: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            redacted.append("***")
            hide_next = False
            continue
        redacted.append(token)
        hide_next = "api_key" in token.lower()
    return shlex.join(redacted)


def _safe_path_tag(value: str) -> str:
    tag = PATH_TAG_RE.sub("_", str(value or "").strip()).strip("._-")
    return tag[:80]


def _git_revision(repo_root: Path) -> dict[str, str | bool]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "", "dirty": False}


def _format_experiment_config(config: dict) -> str:
    return (
        "===== EXPERIMENT CONFIG =====\n"
        + json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n===== END EXPERIMENT CONFIG ====="
    )


def _file_snapshot(path: Path, count_lines: bool = False) -> dict[str, str | int]:
    if not path.is_file():
        return {"path": str(path), "exists": 0}
    digest = hashlib.sha256()
    line_count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            if count_lines:
                line_count += chunk.count(b"\n")
    snapshot: dict[str, str | int] = {
        "path": str(path),
        "exists": 1,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    if count_lines:
        snapshot["lines"] = line_count
    return snapshot


def resolve_doctor_endpoint(args):
    provider = args.provider.strip().lower()
    if provider == "deepseek":
        args.base_url = (
            args.base_url.strip()
            or os.getenv("DEEPSEEK_BASE_URL", "").strip()
            or DEEPSEEK_BASE_URL
        )
        args.api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY", "")
        args.model = (
            args.model.strip()
            or os.getenv("DEEPSEEK_MODEL", "").strip()
            or DEEPSEEK_DEFAULT_MODEL
        )
        if not args.api_key:
            raise ValueError("DeepSeek provider requires --api_key or DEEPSEEK_API_KEY.")
        return
    if provider == "qwen":
        args.base_url = (
            args.base_url.strip()
            or os.getenv("QWEN_BASE_URL", "").strip()
            or os.getenv("DASHSCOPE_BASE_URL", "").strip()
            or QWEN_BASE_URL
        )
        args.api_key = (
            args.api_key
            or os.getenv("DASHSCOPE_API_KEY", "")
            or os.getenv("QWEN_API_KEY", "")
        )
        args.model = (
            args.model.strip()
            or os.getenv("QWEN_MODEL", "").strip()
            or os.getenv("DASHSCOPE_MODEL", "").strip()
            or QWEN_DEFAULT_MODEL
        )
        if not args.api_key:
            raise ValueError("Qwen provider requires --api_key, DASHSCOPE_API_KEY, or QWEN_API_KEY.")
        return

    args.base_url = args.base_url.strip() or LOCAL_OPENAI_BASE_URL
    args.api_key = args.api_key or "None"
    args.model = args.model.strip()
    if not args.model:
        raise ValueError("--model is required unless --provider deepseek/qwen supplies a default.")


def resolve_judge_endpoint(args) -> tuple[str, str, str]:
    provider = args.judge_provider.strip().lower()
    if provider == "auto":
        if args.judge_base_url or args.judge_api_key:
            provider = "custom"
        else:
            provider = args.provider.strip().lower()

    if provider == "deepseek":
        base_url = (
            args.judge_base_url.strip()
            or os.getenv("DEEPSEEK_BASE_URL", "").strip()
            or DEEPSEEK_BASE_URL
        )
        api_key = args.judge_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        model = (
            args.judge_model.strip()
            or os.getenv("DEEPSEEK_JUDGE_MODEL", "").strip()
            or os.getenv("DEEPSEEK_MODEL", "").strip()
            or DEEPSEEK_DEFAULT_MODEL
        )
    elif provider == "qwen":
        base_url = (
            args.judge_base_url.strip()
            or os.getenv("QWEN_BASE_URL", "").strip()
            or os.getenv("DASHSCOPE_BASE_URL", "").strip()
            or QWEN_BASE_URL
        )
        api_key = (
            args.judge_api_key
            or os.getenv("DASHSCOPE_API_KEY", "")
            or os.getenv("QWEN_API_KEY", "")
        )
        model = (
            args.judge_model.strip()
            or os.getenv("QWEN_JUDGE_MODEL", "").strip()
            or os.getenv("QWEN_MODEL", "").strip()
            or os.getenv("DASHSCOPE_MODEL", "").strip()
            or QWEN_DEFAULT_MODEL
        )
    else:
        base_url = (args.judge_base_url or args.base_url).strip()
        api_key = args.judge_api_key if args.judge_api_key else args.api_key
        model = args.judge_model.strip()

    if not model:
        return "", "", ""
    if not base_url:
        raise ValueError("--judge_model is set but judge base URL is empty.")
    return model, base_url, api_key or "None"


def main():
    register_med_tools()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--provider",
        default="local",
        choices=["local", "deepseek", "qwen", "hulumed"],
        help="Doctor API provider. hulumed assumes a local OpenAI-compatible server.",
    )
    parser.add_argument("--model", default="", help="Doctor served model name. DeepSeek defaults to deepseek-chat; Qwen defaults to qwen-plus.")
    parser.add_argument("--tokenizer_path", required=True, help="Local tokenizer/model path for AutoTokenizer.")
    parser.add_argument("--base_url", default="", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api_key", default="", help="OpenAI-compatible API key.")
    parser.add_argument("--case_dir", required=True, help="Bench 根目录，例如 /data/xuxiang/mimic-iv/bench")
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--repeat_k", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=15, help="Maximum doctor-agent turns per case.")
    parser.add_argument("--no_cxr", action="store_true", help="使用无 CXR 数据（bench/without_img/hosp_only）；默认使用有 CXR（bench/with_img/ed_hosp）")
    parser.add_argument(
        "--execution_mode",
        default="parallel",
        choices=["parallel", "serial"],
        help="Run cases concurrently or one by one. serial forces --n_parallel_agents=1.",
    )
    parser.add_argument("--n_parallel_agents", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_response_length", type=int, default=8192)
    parser.add_argument("--parser_name", default="qwen")
    # Judge 模式：用 Qwen3-8B 等模型判断诊断是否与真实一致（不填则用“包含”规则）
    parser.add_argument("--judge_model", default="", help="Judge model name (e.g. Qwen/Qwen3-8B). If set, reward uses LLM judge instead of containment.")
    parser.add_argument(
        "--judge_provider",
        default="auto",
        choices=["auto", "local", "deepseek", "qwen", "hulumed", "custom"],
        help="Judge API provider. auto reuses --provider unless judge URL/key is explicitly set.",
    )
    parser.add_argument("--judge_base_url", default="", help="Judge API base URL (default: same as --base_url).")
    parser.add_argument("--judge_api_key", default="", help="Judge API key (default: same as --api_key).")
    parser.add_argument(
        "--allow_external_agent_fallback",
        action="store_true",
        help=(
            "Allow patient/judge API failures to degrade into default answers or "
            "incorrect judgments. Default is fail-fast."
        ),
    )
    parser.add_argument("--enable_memory", action="store_true")
    parser.add_argument("--memory_root", default="")
    parser.add_argument("--query_builder_mode", default="llm", choices=["rule", "llm"])
    parser.add_argument("--applicability_mode", default="llm", choices=["rule", "llm", "hybrid"])
    parser.add_argument("--experience_extraction_mode", default="llm", choices=["rule", "llm"])
    parser.add_argument("--experience_merge_mode", default="llm", choices=["rule", "llm"])
    parser.add_argument("--memory_top_k", type=int, default=3)
    parser.add_argument("--log_memory_trace", action="store_true")
    parser.add_argument(
        "--inject_case_memory",
        action="store_true",
        help="Expose compact CaseMemory to the doctor agent on each turn, in addition to selected memory guidance.",
    )
    parser.add_argument(
        "--trace_tag",
        default="",
        help=(
            "Experiment tag for trajectory output and, by default, the memory "
            "store subdirectory."
        ),
    )
    parser.add_argument(
        "--memory_root_by_trace_tag",
        "--memory-root-by-trace-tag",
        dest="memory_root_by_trace_tag",
        action="store_true",
        default=True,
        help=(
            "When memory is enabled and --trace_tag is set, store all memory "
            "files under <memory_root>/<trace_tag>/."
        ),
    )
    parser.add_argument(
        "--no-memory-root-by-trace-tag",
        dest="memory_root_by_trace_tag",
        action="store_false",
        help="Share one memory store across trace tags.",
    )
    parser.add_argument("--disable_experience_memory", action="store_true")
    parser.add_argument("--disable_skill_memory", action="store_true")
    parser.add_argument("--disable_knowledge_memory", action="store_true")
    parser.add_argument(
        "--allow_memory_fallback",
        action="store_true",
        help=(
            "Allow rule fallback for online query/applicability LLM failures. "
            "This never enables embedding-scoring or experience-merge fallback."
        ),
    )
    parser.add_argument("--memory_llm_model", default="")
    parser.add_argument("--memory_llm_base_url", default="")
    parser.add_argument("--memory_llm_api_key", default="")
    parser.add_argument(
        "--retrieval_mode",
        default="fielded_bm25",
        choices=["cosine", "fielded_bm25", "embedding"],
        help=(
            "Online memory retrieval scoring mode for controlled experiments. "
            "embedding requires --memory_embedding_model and --memory_embedding_base_url."
        ),
    )
    parser.add_argument("--memory_embedding_model", default="")
    parser.add_argument("--memory_embedding_base_url", default="")
    parser.add_argument("--memory_embedding_api_key", default="")
    parser.add_argument(
        "--merge_scoring_mode",
        default="same_as_retrieval",
        choices=["same_as_retrieval", "cosine", "fielded_bm25", "embedding"],
        help="Offline experience merge candidate/scoring mode for paired ablations.",
    )
    parser.add_argument(
        "--disable_memory_write",
        action="store_true",
        help="Freeze memory store during online retrieval ablations.",
    )
    parser.add_argument(
        "--summary_log_dir",
        default=str(Path(__file__).resolve().parent / "logs"),
        help="Directory for per-run evaluation summary logs. Set empty string to disable.",
    )
    parser.add_argument(
        "--run_log_dir",
        default=str(Path(__file__).resolve().parent / "logs"),
        help="Directory for realtime stdout/stderr run logs. Set empty string to disable.",
    )
    parser.add_argument(
        "--trajectory_dir",
        default=str(Path(__file__).resolve().parent / "trajectories"),
        help=(
            "Directory for full trajectory_*.json files. Defaults to this "
            "runtime directory and overrides any existing RLLM_TRAJECTORY_DIR. "
            "Set empty string to keep the external environment variable."
        ),
    )
    parser.add_argument(
        "--print_examples",
        type=int,
        default=-1,
        help="Number of eval examples to print. Use -1 for all, 0 for none.",
    )
    parser.add_argument(
        "--example_text_chars",
        type=int,
        default=0,
        help="Number of final-output chars per example. Use 0 for full text.",
    )
    args = parser.parse_args()
    if args.execution_mode == "serial":
        args.n_parallel_agents = 1
    resolve_doctor_endpoint(args)
    if not args.allow_external_agent_fallback:
        os.environ["RLLM_STRICT_PATIENT_ERRORS"] = "1"
        os.environ["RLLM_STRICT_JUDGE_ERRORS"] = "1"
    experiment_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    os.environ["MEDGYM_RETRIEVAL_FALLBACK_SCORING"] = (
        "fielded_bm25" if args.retrieval_mode == "fielded_bm25" else "cosine"
    )
    merge_scoring_mode = (
        args.merge_scoring_mode
        if args.merge_scoring_mode != "same_as_retrieval"
        else args.retrieval_mode
    )
    os.environ["MEDGYM_MERGE_SCORING"] = merge_scoring_mode

    if args.enable_memory and (
        args.retrieval_mode == "embedding" or merge_scoring_mode == "embedding"
    ):
        if not (args.memory_embedding_model and args.memory_embedding_base_url):
            raise ValueError(
                "Embedding retrieval/merge requires --memory_embedding_model "
                "and --memory_embedding_base_url."
            )

    run_log_file = None
    run_log_path = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if args.run_log_dir:
        mode = "memory" if args.enable_memory else "no_memory"
        cxr_mode = "no_cxr" if args.no_cxr else "with_cxr"
        retrieval_tag = (
            f"_{args.retrieval_mode}_merge-{merge_scoring_mode}"
            if args.enable_memory else ""
        )
        write_tag = "_frozen" if args.enable_memory and args.disable_memory_write else ""
        log_name = f"run_{experiment_id}_{mode}{retrieval_tag}{write_tag}_{cxr_mode}_n{args.max_cases}_k{args.repeat_k}.log"
        run_log_path = Path(args.run_log_dir) / log_name
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_file = run_log_path.open("w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, run_log_file)
        sys.stderr = Tee(sys.stderr, run_log_file)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    safe_trace_tag = _safe_path_tag(args.trace_tag)
    if args.trace_tag and not safe_trace_tag:
        raise ValueError("--trace_tag must contain at least one safe path character")

    if args.trajectory_dir:
        trajectory_dir = Path(args.trajectory_dir)
        if safe_trace_tag:
            trajectory_dir = trajectory_dir / safe_trace_tag
        os.environ["RLLM_TRAJECTORY_DIR"] = str(trajectory_dir)
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    else:
        os.environ.setdefault(
            "RLLM_TRAJECTORY_DIR",
            str(Path(__file__).resolve().parent / "trajectories"),
        )
    if "RETRIEVAL_SERVER_URL" not in os.environ:
        os.environ["RETRIEVAL_SERVER_URL"] = "http://127.0.0.1:8000"

    base_memory_root = args.memory_root.strip() if args.memory_root else ""
    if not base_memory_root or base_memory_root == "memory_data":
        base_memory_root = MEMORY_ROOT_DIRNAME
    memory_scope_tag = (
        safe_trace_tag
        if args.enable_memory and args.memory_root_by_trace_tag and safe_trace_tag
        else ""
    )
    memory_root = (
        str(Path(base_memory_root) / memory_scope_tag)
        if memory_scope_tag
        else base_memory_root
    )
    memory_trace_tag = "" if memory_scope_tag else safe_trace_tag

    memory_llm_model = (
        args.memory_llm_model
        or os.getenv("MEMORY_LLM_MODEL", "")
        or (args.model if args.enable_memory else "")
    )
    memory_llm_base_url = (
        args.memory_llm_base_url
        or os.getenv("MEMORY_LLM_BASE_URL", "")
        or (args.base_url if args.enable_memory else "")
    )
    memory_llm_api_key = (
        args.memory_llm_api_key
        or os.getenv("MEMORY_LLM_API_KEY", "")
        or (args.api_key if args.enable_memory else "")
    )
    if args.enable_memory and not (memory_llm_model and memory_llm_base_url):
        raise ValueError(
            "Memory is enabled and configured for LLM mode, but memory LLM is not available. "
            "Set --memory_llm_model and --memory_llm_base_url, or provide --model/--base_url."
        )
    effective_parser_name = "native" if args.provider == "deepseek" else args.parser_name

    subdir = SUBDIR_NO_CXR if args.no_cxr else SUBDIR_WITH_CXR
    tasks, cases, _ = prepare_med_data(
        args.case_dir,
        max_cases=args.max_cases,
        repeat_k=args.repeat_k,
        subdir=subdir,
    )

    # Judge 模式：把 judge 配置注入到每个 task，reward 里会读 task_info["judge_model_name"] 等
    judge_model = ""
    judge_base = ""
    judge_key = ""
    if args.judge_model or args.judge_provider != "auto":
        judge_model, judge_base, judge_key = resolve_judge_endpoint(args)
        if judge_model:
            for t in tasks:
                t["judge_model_name"] = judge_model
                t["judge_base_url"] = judge_base
                t["judge_api_key"] = judge_key

    repo_root = Path(__file__).resolve().parent
    memory_root_path = Path(memory_root)
    experiment_config = {
        "started_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "experiment_id": experiment_id,
        "command": _redacted_command([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
        "git": _git_revision(repo_root),
        "code_fingerprints": {
            "memory_prompts": _file_snapshot(repo_root / "memory_agent/llm/prompts.py"),
        },
        "memory_store_at_start": {
            name: _file_snapshot(memory_root_path / name, count_lines=True)
            for name in (
                "experience_memory.jsonl",
                "skill_memory.jsonl",
                "knowledge_memory.jsonl",
            )
        },
        "arguments": _redact_mapping(vars(args)),
        "effective": {
            "merge_scoring_mode": merge_scoring_mode,
            "doctor_tool_call_mode": (
                "native_function_calling" if args.provider == "deepseek" else "text_parser"
            ),
            "effective_parser_name": effective_parser_name,
            "n_parallel_agents": args.n_parallel_agents,
            "base_memory_root": base_memory_root,
            "memory_scope_tag": memory_scope_tag,
            "memory_root": memory_root,
            "memory_llm_model": memory_llm_model,
            "memory_llm_base_url": memory_llm_base_url,
            "judge_model": judge_model,
            "judge_base_url": judge_base,
            "patient_model": os.getenv("RLLM_PATIENT_MODEL", ""),
            "patient_base_url": os.getenv("RLLM_PATIENT_BASE_URL", ""),
            "strict_patient_errors": os.getenv("RLLM_STRICT_PATIENT_ERRORS", ""),
            "strict_judge_errors": os.getenv("RLLM_STRICT_JUDGE_ERRORS", ""),
            "retrieval_server_url": os.getenv("RETRIEVAL_SERVER_URL", ""),
            "trajectory_dir": os.getenv("RLLM_TRAJECTORY_DIR", ""),
            "run_log_path": str(run_log_path or ""),
        },
    }
    experiment_header = _format_experiment_config(experiment_config)
    print(experiment_header)

    # tasks, cases, _ = prepare_med_data(
    #     case_dir=args.case_dir,
    #     max_cases=args.max_cases,
    #     repeat_k=args.repeat_k,
    # )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.no_cxr:
        agent_args = {
            "tools": ["ask_patient", "diagnosis", "retrieve", "request_exam"],
            "parser_name": effective_parser_name,
            "system_prompt": DOCTOR_SYSTEM_PROMPT_wo_IMG,
            "enable_memory": args.enable_memory,
            "memory_root": memory_root,
            "query_builder_mode": args.query_builder_mode,
            "applicability_mode": args.applicability_mode,
            "experience_extraction_mode": args.experience_extraction_mode,
            "experience_merge_mode": args.experience_merge_mode,
            "memory_top_k": args.memory_top_k,
            "log_memory_trace": args.log_memory_trace,
            "inject_case_memory": args.inject_case_memory,
            "trace_tag": memory_trace_tag,
            "disable_experience_memory": args.disable_experience_memory,
            "disable_skill_memory": args.disable_skill_memory,
            "disable_knowledge_memory": args.disable_knowledge_memory,
            "disable_memory_write": args.disable_memory_write,
            "strict_memory_errors": not args.allow_memory_fallback,
            "memory_llm_model": memory_llm_model,
            "memory_llm_base_url": memory_llm_base_url,
            "memory_llm_api_key": memory_llm_api_key,
            "memory_embedding_model": args.memory_embedding_model,
            "memory_embedding_base_url": args.memory_embedding_base_url,
            "memory_embedding_api_key": args.memory_embedding_api_key,
            "retrieval_mode": args.retrieval_mode,
            "no_cxr": args.no_cxr,
            "native_tool_calls": args.provider == "deepseek",
        }

        env_args = {
            "tools": ["ask_patient", "diagnosis", "retrieve", "request_exam"],
            "reward_fn": med_diagnosis_reward,
            "max_steps": args.max_steps,
        }

    else:
        agent_args = {
            "tools": ["ask_patient", "diagnosis", "retrieve", "cxr", "request_exam", "cxr_grounding"],
            "parser_name": effective_parser_name,
            "system_prompt": DOCTOR_SYSTEM_PROMPT,
            "enable_memory": args.enable_memory,
            "memory_root": memory_root,
            "query_builder_mode": args.query_builder_mode,
            "applicability_mode": args.applicability_mode,
            "experience_extraction_mode": args.experience_extraction_mode,
            "experience_merge_mode": args.experience_merge_mode,
            "memory_top_k": args.memory_top_k,
            "log_memory_trace": args.log_memory_trace,
            "inject_case_memory": args.inject_case_memory,
            "trace_tag": memory_trace_tag,
            "disable_experience_memory": args.disable_experience_memory,
            "disable_skill_memory": args.disable_skill_memory,
            "disable_knowledge_memory": args.disable_knowledge_memory,
            "disable_memory_write": args.disable_memory_write,
            "strict_memory_errors": not args.allow_memory_fallback,
            "memory_llm_model": memory_llm_model,
            "memory_llm_base_url": memory_llm_base_url,
            "memory_llm_api_key": memory_llm_api_key,
            "memory_embedding_model": args.memory_embedding_model,
            "memory_embedding_base_url": args.memory_embedding_base_url,
            "memory_embedding_api_key": args.memory_embedding_api_key,
            "retrieval_mode": args.retrieval_mode,
            "no_cxr": args.no_cxr,
            "native_tool_calls": args.provider == "deepseek",
        }

        env_args = {
            "tools": ["ask_patient", "diagnosis", "retrieve", "cxr", "request_exam", "cxr_grounding"],
            "reward_fn": med_diagnosis_reward,
            "max_steps": args.max_steps,
            "context_injected_tool_names": ["cxr", "request_exam", "cxr_grounding"],
        }

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    native_tool_schemas = (
        MultiTool(tools=agent_args["tools"]).json
        if args.provider == "deepseek"
        else []
    )

    engine = AgentExecutionEngine(
        agent_class=MemoryWrappedMedicalAgent,
        agent_args=agent_args,
        env_class=MedicalDialogueEnv,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={
            "base_url": args.base_url,
            "api_key": args.api_key,
            "model": args.model,
            "use_chat_completions": args.provider in {"deepseek", "qwen", "hulumed"},
            "tools": native_tool_schemas,
        },
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_steps=args.max_steps,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        n_parallel_agents=args.n_parallel_agents,
    )

    results = asyncio.run(engine.execute_tasks(tasks))
    log_path = None
    if args.summary_log_dir:
        mode = "memory" if args.enable_memory else "no_memory"
        cxr_mode = "no_cxr" if args.no_cxr else "with_cxr"
        retrieval_tag = (
            f"_{args.retrieval_mode}_merge-{merge_scoring_mode}"
            if args.enable_memory else ""
        )
        write_tag = "_frozen" if args.enable_memory and args.disable_memory_write else ""
        log_name = f"summary_{experiment_id}_{mode}{retrieval_tag}{write_tag}_{cxr_mode}_n{args.max_cases}_k{args.repeat_k}.log"
        log_path = Path(args.summary_log_dir) / log_name
    evaluate_doctor_results(
        results,
        tasks,
        print_examples=args.print_examples,
        example_text_chars=args.example_text_chars,
        log_path=log_path,
        report_header=experiment_header,
    )
    if run_log_file:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        run_log_file.close()


if __name__ == "__main__":
    main()
