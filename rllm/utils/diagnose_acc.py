from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rllm.rewards.med_diagnosis_reward import result_reward, result_reward_judge


FINAL_JSON_RE = re.compile(r"\{.*\}", re.S)
FINAL_DIAGNOSIS_RE = re.compile(
    r"The\s+final\s+diagnosis\s+is\s*:\s*\\box(?:ed)?\{(.+?)\}\s*\.?\s*$",
    re.I | re.S,
)
BOXED_RE = re.compile(r"\\box(?:ed)?\{(.+?)\}", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _to_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _unwrap_action(action: Any) -> Any:
    if hasattr(action, "action"):
        return getattr(action, "action")
    return action


def _iter_steps(result: Any):
    """
    Yield Step-like objects from the common return shapes:
    Trajectory, trajectory dict, Episode-like object, or lists of those.
    """
    if result is None:
        return

    if isinstance(result, (list, tuple)):
        for item in result:
            yield from _iter_steps(item)
        return

    steps = _get(result, "steps")
    if isinstance(steps, list):
        for step in steps:
            yield step
        return

    trajectories = _get(result, "trajectories")
    if isinstance(trajectories, list):
        for traj in trajectories:
            yield from _iter_steps(traj)


def extract_final_answer_text(result: Any) -> str:
    """
    尽量从 trajectory/result 中提取最后的 assistant 输出文本。
    由于不同 agent/engine 的返回结构可能不同，这里用 best-effort。
    """
    # 常见字段
    for attr in ["final_answer", "answer", "output", "response", "text"]:
        if hasattr(result, attr):
            v = getattr(result, attr)
            if isinstance(v, str) and v.strip():
                return v.strip()

    steps = list(_iter_steps(result))
    if steps:
        last = steps[-1]
        for attr in ["model_response", "response", "assistant", "assistant_message", "output", "text", "observation"]:
            v = _get(last, attr)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return _to_str(result)


def count_tool_calls(
    result: Any,
    tool_name: str | None = None,
    exclude_tools: set[str] | None = None,
) -> int:
    """
    优先从 trajectory step.action 的结构化 tool call 计数。
    只有在没有结构化 step 时，才退回到字符串统计，避免把 system prompt
    或 tool schema 里的工具名也算进去。

    默认计数所有 tool call；可传 tool_name 只计一个工具，或传
    exclude_tools 排除部分工具。
    """
    exclude_tools = exclude_tools or set()

    def should_count(name: str) -> bool:
        if tool_name is not None:
            return name == tool_name
        return bool(name) and name not in exclude_tools

    steps = list(_iter_steps(result))
    if steps:
        n = 0
        for step in steps:
            action = _unwrap_action(_get(step, "action"))
            for call in _iter_tool_calls(action):
                if should_count(_tool_call_name(call)):
                    n += 1
        return n

    s = _to_str(result)
    parsed_count = sum(
        1
        for call in _parse_tool_calls_from_text(s)
        if should_count(_tool_call_name(call))
    )
    if parsed_count:
        return parsed_count
    if tool_name is not None:
        return s.count(tool_name)
    return 0


def parse_final_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a final-answer JSON object.

    Tool calls are also JSON blobs, so only objects with an explicit diagnosis
    field count as final JSON.
    """
    for m in re.finditer(r"\{.*?\}", text or "", re.S):
        obj = _parse_json_obj(m.group(0))
        if not obj:
            continue
        if "diagnosis" in obj or "final_diagnosis" in obj:
            return obj
    return None


def extract_boxed_diagnosis(text: str) -> Optional[str]:
    """
    doctor agent 当前通过 diagnosis tool 提交：
    The final diagnosis is: \\boxed{xxx}.
    这里兼容完整行和任意 \\boxed{xxx}。
    """
    if not text:
        return None
    text = text.split("</think>")[-1].strip()
    m = FINAL_DIAGNOSIS_RE.search(text)
    if m:
        return m.group(1).strip()
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for m in TOOL_CALL_RE.finditer(text or ""):
        obj = _parse_json_obj(m.group(1))
        if obj is not None:
            calls.append(obj)
    return calls


def _iter_tool_calls(action: Any):
    action = _unwrap_action(action)
    if isinstance(action, list):
        for item in action:
            yield from _iter_tool_calls(item)
    elif isinstance(action, dict):
        yield action
    elif isinstance(action, str):
        for call in _parse_tool_calls_from_text(action):
            yield call


def _tool_call_name(call: Dict[str, Any]) -> str:
    fn = call.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(call.get("name") or "")


def _tool_call_args(call: Dict[str, Any]) -> Dict[str, Any]:
    fn = call.get("function")
    args = fn.get("arguments") if isinstance(fn, dict) else call.get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        parsed = _parse_json_obj(args)
        return parsed or {}
    return {}


def _extract_diagnosis_from_args(args: Dict[str, Any]) -> str:
    for key in ("diagnosis", "final_diagnosis"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("final_response", "result", "response"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            boxed = extract_boxed_diagnosis(value)
            return (boxed or value).strip()

    return ""


def _extract_finish_diagnosis_from_args(args: Dict[str, Any]) -> str:
    for key in ("diagnosis", "final_diagnosis"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("final_response", "result", "response"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            boxed = extract_boxed_diagnosis(value)
            if boxed:
                return boxed.strip()
            obj = parse_final_json(value)
            if obj:
                return str(obj.get("diagnosis") or obj.get("final_diagnosis") or "").strip()

    return ""


def extract_final_diagnosis(result: Any, out_text: str = "") -> str:
    """
    Extract the canonical final diagnosis.

    The evaluator exposes one unified contract: final diagnosis text. It accepts
    old logs (JSON/plain boxed text) only as compatibility inputs and does not
    report separate source types.
    """
    final_json = parse_final_json(out_text)
    if isinstance(final_json, dict):
        pred = str(final_json.get("diagnosis", "")).strip()
        if pred:
            return pred

    steps = list(_iter_steps(result))
    for step in reversed(steps):
        action = _unwrap_action(_get(step, "action"))
        calls = list(_iter_tool_calls(action))
        for call in reversed(calls):
            name = _tool_call_name(call)
            if name == "diagnosis":
                pred = _extract_diagnosis_from_args(_tool_call_args(call))
                if pred:
                    return pred
            elif name == "finish":
                pred = _extract_finish_diagnosis_from_args(_tool_call_args(call))
                if pred:
                    return pred

        model_response = _get(step, "model_response")
        if isinstance(model_response, str):
            for call in reversed(_parse_tool_calls_from_text(model_response)):
                name = _tool_call_name(call)
                if name == "diagnosis":
                    pred = _extract_diagnosis_from_args(_tool_call_args(call))
                    if pred:
                        return pred
                elif name == "finish":
                    pred = _extract_finish_diagnosis_from_args(_tool_call_args(call))
                    if pred:
                        return pred

    for call in reversed(_parse_tool_calls_from_text(out_text)):
        name = _tool_call_name(call)
        if name == "diagnosis":
            pred = _extract_diagnosis_from_args(_tool_call_args(call))
            if pred:
                return pred
        elif name == "finish":
            pred = _extract_finish_diagnosis_from_args(_tool_call_args(call))
            if pred:
                return pred

    boxed = extract_boxed_diagnosis(out_text)
    if boxed:
        return boxed

    return ""


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 _\-]", "", s)  # 只保留字母数字和少量符号
    return s.strip()


def diagnosis_match_lenient(pred: str, gold: str) -> bool:
    """
    宽松匹配：gold 是 pred 的子串 或 pred 是 gold 的子串。
    """
    p = _norm(pred)
    g = _norm(gold)
    if not p or not g:
        return False
    return (g in p) or (p in g)


def diagnosis_correct_by_reward(
    pred: str,
    gold: str,
    task: Dict[str, Any],
) -> tuple[bool, str, Dict[str, Any]]:
    judge_model_name = str(task.get("judge_model_name") or "").strip()
    judge_base_url = str(task.get("judge_base_url") or "").strip()
    judge_api_key = task.get("judge_api_key") or task.get("api_key") or "None"

    if judge_model_name and judge_base_url:
        _, metadata = result_reward_judge(
            pred,
            gold,
            judge_model_name=judge_model_name,
            judge_base_url=judge_base_url,
            api_key=judge_api_key,
        )
        return bool(metadata.get("judge_consistent", False)), "judge", metadata

    _, metadata = result_reward(pred, gold)
    return bool(metadata.get("ground_truth_contained", False)), "containment", metadata


def evaluate_one(
    result: Any,
    task: Dict[str, Any],
    *,
    raw_text_chars: int = 800,
) -> Dict[str, Any]:
    out_text = extract_final_answer_text(result)
    tool_calls = count_tool_calls(result, exclude_tools={"ask_patient", "diagnosis"})
    final_json = parse_final_json(out_text)
    pred_diag = extract_final_diagnosis(result, out_text)

    gold = str(task.get("ground_truth", "")).strip()

    ok_tool = tool_calls >= 1
    ok_json = final_json is not None
    ok_final_answer = bool(pred_diag)
    ok_diag, eval_mode, reward_metadata = diagnosis_correct_by_reward(
        pred_diag,
        gold,
        task,
    ) if gold else (False, "none", {})

    if raw_text_chars <= 0:
        raw_text = out_text
    else:
        raw_text = out_text[-raw_text_chars:]

    return {
        "case_id": task.get("_case_group", task.get("case_id", "")),
        "repeat_id": task.get("_repeat_id", 0),
        "tool_calls": tool_calls,
        "pred_diagnosis": pred_diag,
        "ground_truth": gold,
        "ok_tool": ok_tool,
        "ok_json": ok_json,
        "ok_final_answer": ok_final_answer,
        "ok_diag": ok_diag,
        "diag_eval_mode": eval_mode,
        "diag_reward_metadata": reward_metadata,
        "raw_tail": raw_text,
    }


def compute_pass_at_k(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    假设同一 case_id 重复了 k 次，pass@k = 该 case 是否至少一次 ok_diag=True 的比例
    """
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)

    n_cases = len(by_case)
    passed = 0
    for cid, arr in by_case.items():
        if any(x["ok_diag"] for x in arr):
            passed += 1

    return {
        "n_cases": n_cases,
        "pass_at_k": (passed / n_cases) if n_cases else 0.0,
        "passed_cases": passed,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ok_tool = sum(1 for r in rows if r["ok_tool"])
    ok_json = sum(1 for r in rows if r["ok_json"])
    ok_final_answer = sum(1 for r in rows if r["ok_final_answer"])
    ok_diag = sum(1 for r in rows if r["ok_diag"])

    p_at_k = compute_pass_at_k(rows)

    return {
        "total_runs": n,
        "tool_called>=1": f"{ok_tool}/{n}",
        "has_final_json": f"{ok_json}/{n}",
        "has_final_answer": f"{ok_final_answer}/{n}",
        "diag_correct": f"{ok_diag}/{n}",
        "pass_at_k": p_at_k,
    }


def _format_evaluation_report(
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    print_examples: int = -1,
) -> str:
    lines: list[str] = []
    lines.append("\n===== SUMMARY =====")
    for k, v in summary.items():
        lines.append(f"{k}: {v}")

    if print_examples != 0:
        selected_rows = rows if print_examples < 0 else rows[:print_examples]
        label = "all" if print_examples < 0 else f"first {print_examples}"
        lines.append(f"\n===== EXAMPLES ({label}) =====")
        for i, r in enumerate(selected_rows):
            lines.append(f"\n--- example {i} ---")
            lines.append(f"case_id={r['case_id']} repeat={r['repeat_id']}")
            lines.append(
                f"tool_calls={r['tool_calls']} "
                f"has_json={r['ok_json']} has_final_answer={r['ok_final_answer']} "
                f"ok_diag={r['ok_diag']} eval={r.get('diag_eval_mode', '')}"
            )
            lines.append(f"gold={r['ground_truth']}")
            lines.append(f"pred={r['pred_diagnosis']}")
            lines.append(f"tail=\n{r['raw_tail']}")

    return "\n".join(lines)


def _write_evaluation_logs(
    payload: Dict[str, Any],
    report: str,
    log_path: str | Path,
) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report + "\n", encoding="utf-8")


def evaluate_doctor_results(
    results: List[Any],
    tasks: List[Dict[str, Any]],
    print_examples: int = -1,
    example_text_chars: int = 0,
    log_path: str | Path | None = None,
) -> Dict[str, Any]:
    rows = [
        evaluate_one(res, task, raw_text_chars=example_text_chars)
        for res, task in zip(results, tasks)
    ]
    summary = summarize(rows)
    payload = {"summary": summary, "rows": rows}

    report = _format_evaluation_report(summary, rows, print_examples=print_examples)
    print(report)

    if log_path:
        _write_evaluation_logs(payload, report, log_path)

    return payload
