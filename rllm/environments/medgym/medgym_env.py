# /data/xuxiang/mimic-iv/virtual_env/code/rllm/rllm/environments/medgym/medgym_env.py
from __future__ import annotations

import json
import multiprocessing as mp
import os
from typing import Any, Optional

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool


def _load_callable(path: str):
    """
    支持用字符串传 reward_fn（避免 pickling 问题）
    格式：
      - "pkg.module:func"
      - "pkg.module.func"
    """
    import importlib

    if ":" in path:
        module_path, name = path.split(":", 1)
    else:
        module_path, name = path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, name)


class MedicalDialogueEnv(BaseEnv):
    """
    - 主进程：reset/step/close 通过 Pipe 发命令
    - 子进程：真正执行工具、模拟对话、结算 reward
    """

    def __init__(
        self,
        task: dict | None = None,
        tools: list[str] | None = None,
        tool_map: dict[str, type[Tool]] | None = None,
        reward_fn: RewardFunction | str | None = None,
        max_steps: int = 10,
        ask_tool_name: str = "ask_patient",
        context_injected_tool_names: list[str] | None = None,
        parallel_tool_calls: bool = False,
        max_tool_workers: int = 4,
        timeout: Optional[float] = None,     # seconds
        start_method: str = "spawn",         # 建议默认 spawn，GPU/视觉工具更安全
    ):
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.timeout = timeout
        self._ctx = mp.get_context(start_method)

        self.parent_conn, child_conn = self._ctx.Pipe(duplex=True)

        # reward_fn 可能不可 pickling；允许传字符串路径
        reward_spec = reward_fn
        if reward_fn is None:
            reward_spec = None
        elif isinstance(reward_fn, str):
            reward_spec = reward_fn
        else:
            reward_spec = reward_fn

        if context_injected_tool_names is None:
            context_injected_tool_names = ["cxr"]
        config = {
            "task": task or {},
            "tools": tools,
            "tool_map": tool_map,
            "reward_fn": reward_spec,
            "max_steps": max_steps,
            "ask_tool_name": ask_tool_name,
            "context_injected_tool_names": list(context_injected_tool_names),
            "parallel_tool_calls": parallel_tool_calls,
            "max_tool_workers": max_tool_workers,
        }

        self.process = self._ctx.Process(target=self._worker, args=(child_conn, config))
        self.process.daemon = True
        self.process.start()


    @staticmethod
    def _worker(conn, config: dict):
        """
        子进程 worker：真正执行 reset/step/close 的逻辑
        """
        from rllm.tools import register_med_tools

        register_med_tools()

        task = config["task"]
        tools = config["tools"]
        tool_map = config["tool_map"]
        ask_tool_name = config["ask_tool_name"]
        context_injected_tool_names = set(config.get("context_injected_tool_names", ["cxr"]))
        parallel_tool_calls = config["parallel_tool_calls"]
        max_tool_workers = config["max_tool_workers"]
        max_steps = int(config["max_steps"])

        # init tools
        if tool_map is not None:
            tool_runner = MultiTool(tool_map=tool_map)
        elif tools is not None:
            tool_runner = MultiTool(tools=tools)
        else:
            tool_runner = MultiTool(tools=[])

        # init reward_fn
        reward_fn = config["reward_fn"]
        if reward_fn is None:
            reward_fn_impl = zero_reward
        elif isinstance(reward_fn, str):
            reward_fn_impl = _load_callable(reward_fn)
        else:
            reward_fn_impl = reward_fn

        def init_tool_context(t: dict) -> dict:
            """
            从 task 中初始化工具上下文：
            - 优先使用 task["context"]（prepare_med_data 产物一般在这里）
            - 兼容：task 顶层直接带 ehr / knowledge / knowbase
            - 确保 dialogue 是 list
            """
            ctx: dict = {}
            if isinstance(t.get("context"), dict):
                ctx.update(t["context"])

            if isinstance(t.get("ehr"), dict) and "ehr" not in ctx:
                ctx["ehr"] = t["ehr"]
            if t.get("knowledge") is not None and "knowledge" not in ctx and "knowbase" not in ctx:
                ctx["knowledge"] = t["knowledge"]
            if t.get("knowbase") is not None and "knowbase" not in ctx:
                ctx["knowbase"] = t["knowbase"]

            if not isinstance(ctx.get("dialogue"), list):
                ctx["dialogue"] = []
            return ctx

        tool = str(task.get("case_id", ""))  # default case_id
        tool_context: dict = init_tool_context(task)
        process_turns: list[dict] = []  # 用于过程奖励：每次 ask_patient 的 (dialogue_before, question, answer)

        step_count = 0

        def simulate_patient_reply(question: str) -> str:
            """
            fallback 用：优先从 task["patient_answers"] 里取
            """
            patient_answers = task.get("patient_answers", None)
            if isinstance(patient_answers, dict):
                if question in patient_answers:
                    return str(patient_answers[question])
                if "__default__" in patient_answers:
                    return str(patient_answers["__default__"])
            return "I’m not sure. Could you ask a more specific question?"

        def extract_finish_text(tool_calls: list[dict]) -> Optional[str]:
            for tc in tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") == "finish":
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            return ""
                    if isinstance(args, dict):
                        return str(args.get("response", ""))
                    return ""
            return None

        def extract_ask_question(tool_calls: list[dict]) -> Optional[str]:
            for tc in tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") == ask_tool_name:
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            return args
                    if isinstance(args, dict):
                        for k in ("question", "query", "text"):
                            if k in args:
                                return str(args[k])
                        return json.dumps(args, ensure_ascii=False)
            return None

        def execute_single_tool_call(tc: dict) -> tuple[str, str]:
            tool_name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", {})

            if isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except Exception:
                    tool_args = {"input": raw_args}
            elif isinstance(raw_args, dict):
                tool_args = raw_args
            else:
                tool_args = {}

            out = tool_runner(tool_name=tool_name, **tool_args)

            return str(tc.get("id", tool_name)), out.to_string()

        def execute_tool_calls(tool_calls: list[dict]) -> dict[str, str]:
            if (not parallel_tool_calls) or len(tool_calls) <= 1 or max_tool_workers <= 1:
                outputs = {}
                for tc in tool_calls:
                    i, s = execute_single_tool_call(tc)
                    outputs[i] = s
                return outputs

            from concurrent.futures import ThreadPoolExecutor, as_completed

            outputs = {}
            with ThreadPoolExecutor(max_workers=max_tool_workers) as ex:
                futs = [ex.submit(execute_single_tool_call, tc) for tc in tool_calls]
                for f in as_completed(futs):
                    i, s = f.result()
                    outputs[i] = s
            return outputs

        def _dialogue_to_text(dialogue: list) -> str:
            """将 dialogue list 转成纯文本，供过程奖励的 dialogue_before 使用。"""
            if not dialogue:
                return ""
            parts = []
            for item in dialogue:
                if isinstance(item, dict):
                    role = item.get("role", "")
                    content = item.get("content", "")
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        def finalize(action_text: str, raw_action: Any):
            task_for_reward = {**task, "process_turns": list(process_turns)}
            reward_out = reward_fn_impl(task_info=task_for_reward, action=action_text)
            return (
                {},  # next_obs
                reward_out.reward,
                True,
                {
                    "response": raw_action,
                    "metadata": reward_out.metadata,
                    "is_correct": reward_out.is_correct,
                },
            )

        try:
            while True:
                cmd, data = conn.recv()

                if cmd == "reset":
                    step_count = 0
                    tool_context = init_tool_context(task)
                    process_turns.clear()
                    conn.send((task, {}))

                elif cmd == "step":
                    action = data
                    step_count += 1
                    reached_max = step_count >= max_steps

                    if isinstance(action, str):
                        if reached_max:
                            conn.send(finalize(action_text=action, raw_action=action))
                        else:
                            reply = simulate_patient_reply(action)
                            conn.send(
                                (
                                    {"question": reply},
                                    0.0,
                                    False,
                                    {"response": action, "metadata": {"turn_type": "patient_dialogue"}},
                                )
                            )
                        continue

                    # normalize dict -> list
                    if isinstance(action, dict):
                        action = [action]

                    tool_calls = action if isinstance(action, list) else []

                    # finish
                    finish_text = extract_finish_text(tool_calls)
                    if finish_text is not None:
                        conn.send(finalize(action_text=finish_text, raw_action=tool_calls))
                        continue

                    if reached_max:
                        # max steps 强制结束
                        try:
                            fallback = json.dumps(tool_calls, ensure_ascii=False)
                        except Exception:
                            fallback = str(tool_calls)
                        conn.send(finalize(action_text=fallback, raw_action=tool_calls))
                        continue

                    ask_q = extract_ask_question(tool_calls)
                    if ask_q is not None:
                        try:
                            out = tool_runner(
                                tool_name=ask_tool_name,
                                question=ask_q,
                                case_id=task.get("case_id", "") or "",
                                context=tool_context,  # ✅ 注入 ehr/knowledge/dialogue
                            )

                            reply = ""
                            new_ctx = None

                            if hasattr(out, "output") and isinstance(out.output, dict):
                                reply = str(out.output.get("answer", ""))
                                new_ctx = out.output.get("context", None)

                            if not reply:
                                reply = out.to_string()

                            # 过程奖励：记录本轮 ask 前的对话与本轮问答
                            dialogue_before = _dialogue_to_text(tool_context.get("dialogue") or [])
                            process_turns.append({
                                "dialogue_before": dialogue_before,
                                "question": ask_q or "",
                                "answer": reply,
                            })

                            if isinstance(new_ctx, dict):
                                tool_context = new_ctx
                            else:
                                if not isinstance(tool_context.get("dialogue"), list):
                                    tool_context["dialogue"] = []

                        except Exception as e:
                            if str(os.getenv("RLLM_STRICT_PATIENT_ERRORS", "")).strip().lower() in {
                                "1", "true", "yes", "on",
                            }:
                                raise
                            reply = f"ERROR calling {ask_tool_name}: {type(e).__name__}: {e}"

                        conn.send(
                            (
                                {"question": reply},
                                0.0,
                                False,
                                {
                                    "response": tool_calls,
                                    "metadata": {
                                        "turn_type": "patient_dialogue",
                                        "ask_tool": ask_tool_name,
                                        "context_keys": sorted(list(tool_context.keys())),
                                    },
                                },
                            )
                        )
                        continue

                    # 对 cxr 等需要 context 的工具注入 tool_context 和 case_id（与 ask_patient 一致）
                    for tc in tool_calls:
                        name = (tc.get("function") or {}).get("name")
                        if name not in context_injected_tool_names:
                            continue
                        raw = (tc.get("function") or {}).get("arguments", {})
                        if isinstance(raw, str):
                            try:
                                raw = json.loads(raw)
                            except Exception:
                                raw = {}
                        if not isinstance(raw, dict):
                            raw = {}
                        raw["context"] = tool_context
                        if not raw.get("case_id"):
                            raw["case_id"] = task.get("case_id", "") or ""
                        if tc.get("function") is not None:
                            tc["function"]["arguments"] = raw

                    outputs = execute_tool_calls(tool_calls)
                    conn.send(
                        (
                            {"tool_outputs": outputs},
                            0.0,
                            False,
                            {"response": tool_calls, "metadata": {"turn_type": "tool"}},
                        )
                    )

                elif cmd == "close":
                    conn.close()
                    break

        except EOFError:
            try:
                conn.close()
            except Exception:
                pass

    # ---------------------------
    # Parent side API
    # ---------------------------

    def reset(self):
        self.parent_conn.send(("reset", None))
        if self.timeout is not None and (not self.parent_conn.poll(self.timeout)):
            raise TimeoutError(f"Timeout after {self.timeout} seconds waiting for reset().")
        return self.parent_conn.recv()

    def step(self, action):
        self.parent_conn.send(("step", action))
        if self.timeout is not None and (not self.parent_conn.poll(self.timeout)):
            raise TimeoutError(f"Timeout after {self.timeout} seconds waiting for step().")
        return self.parent_conn.recv()

    def close(self):
        try:
            self.parent_conn.send(("close", None))
        except Exception:
            pass

        self.process.join(60 * 2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

    @staticmethod
    def from_dict(env_args: dict) -> "MedicalDialogueEnv":
        """
        兼容框架：用 dict 创建 env。
        约定：
          - env_args 里可以包含 tools/tool_map/reward_fn/max_steps 等控制参数
          - env_args 剩下的作为 task 内容
        """
        tools = env_args.pop("tools", None)
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)

        ask_tool_name = env_args.pop("ask_tool_name", "ask_patient")
        context_injected_tool_names = env_args.pop("context_injected_tool_names", None)
        parallel_tool_calls = env_args.pop("parallel_tool_calls", False)
        max_tool_workers = env_args.pop("max_tool_workers", 4)

        timeout = env_args.pop("timeout", None)
        start_method = env_args.pop("start_method", "spawn")

        return MedicalDialogueEnv(
            task=env_args,
            tools=tools,
            tool_map=tool_map,
            reward_fn=reward_fn,
            max_steps=max_steps,
            ask_tool_name=ask_tool_name,
            context_injected_tool_names=context_injected_tool_names,
            parallel_tool_calls=parallel_tool_calls,
            max_tool_workers=max_tool_workers,
            timeout=timeout,
            start_method=start_method,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
