# /home/xuxiang/virtual_env/code/rllm/examples/MedGym/prepare_med_data.py
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cases_from_dir(case_dir: str, max_cases: int = 10) -> List[Dict[str, Any]]:

    paths = sorted(glob.glob(os.path.join(case_dir, "*.json")))
    if max_cases > 0:
        paths = paths[:max_cases]

    cases: List[Dict[str, Any]] = []
    for p in paths:
        obj = load_json(p)
        case_id = os.path.splitext(os.path.basename(p))[0]

        if isinstance(obj, dict) and "ehr" in obj and isinstance(obj["ehr"], dict):
            ehr = obj["ehr"]
            knowledge = obj.get("knowledge", None)
        else:
            ehr = obj if isinstance(obj, dict) else {}
            knowledge = None

        gold = ""
        if isinstance(ehr, dict):
            gold = str(ehr.get("Final_Result", "") or "")

        cases.append(
            {
                "case_id": case_id,
                "ehr": ehr,
                "knowledge": knowledge,
                "gold_diagnosis": gold,
                "ehr_path": p,
            }
        )

    return cases


def build_tasks(
    cases: List[Dict[str, Any]],
    repeat_k: int = 1,
) -> List[Dict[str, Any]]:

    tasks: List[Dict[str, Any]] = []
    repeat_k = max(1, int(repeat_k))

    for rep in range(repeat_k):
        for c in cases:
            info = c["ehr"]["Patient_info"]
            chief = c["ehr"]["History"]["Chief_Complaint"]
            tasks.append(
                {
                    "case_id": c["case_id"],
                    "context": {
                        "ehr": c.get("ehr", {}),
                        "knowbase": c.get("knowledge", None),
                    },
                    "question": (
                        f"I am a {info['age']}-year-old {info['gender']} patient. "
                        f"I haven't been feeling well recently. "
                        f"My main issue is: {chief}."
                    ),
                    "ground_truth": c.get("gold_diagnosis", ""),
                    "data_source": "medical",
                    "_case_group": c["case_id"],
                    "_repeat_id": rep,
                }
            )
    return tasks

def prepare_med_data(
    case_dir: str,
    max_cases: int = 10,
    repeat_k: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Any | None]:
    """
    返回：
    - tasks
    - cases
    - knowbase：这里不再使用全局 knowbase，返回 None（每个 case 的 knowledge 已经放进 task.context["knowbase"]）
    """
    cases = load_cases_from_dir(case_dir, max_cases=max_cases)
    tasks = build_tasks(cases, repeat_k=repeat_k)
    return tasks, cases, None
