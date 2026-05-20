from __future__ import annotations

from pathlib import Path


MEMORY_ROOT_DIRNAME = str(Path(__file__).resolve().parents[1] / "memory_data")

RETRIEVAL_CONFIG = {
    # Online retrieval fallback scoring when dense embeddings are unavailable.
    # Options:
    # - "cosine": previous token-cosine scoring over the full memory text.
    # - "fielded_bm25": field-weighted BM25 scoring for experience/skill.
    "fallback_scoring": "fielded_bm25",
    "positive_experience_top_k": 5,
    "negative_experience_top_k": 3,
    "skill_top_k": 3,
    "knowledge_top_k": 3,
    # Retrieval thresholds. Negative memory uses a higher threshold because it can
    # discourage or block actions downstream.
    "positive_experience_min_score": 0.18,
    "negative_experience_min_score": 0.25,
    "skill_min_score": 0.20,
    "knowledge_min_score": 0.12,
}

MERGE_CONFIG = {
    # Offline merge candidate recall / rule similarity scoring.
    # Keep this aligned with RETRIEVAL_CONFIG["fallback_scoring"] for paired
    # retrieve+merge ablation runs.
    "candidate_scoring": "fielded_bm25",
    "semantic_threshold": 0.80,
    "action_threshold": 0.75,
    "boundary_threshold": 0.50,
    "candidate_top_k": 20,
}

SKILL_CONFIG = {
    "min_support_count": 5,
    "min_unique_cases": 3,
    "min_success_rate": 0.75,
    "max_unsafe_rate": 0.05,
}

MEMORY_ACTION_CONFIG = {
    "tool_to_action": {
        "ask_patient": "ASK",
        "request_exam": "REQUEST_LAB",
        "retrieve": "REQUEST_LAB",
        "cxr": "REVIEW_IMAGE",
        "cxr_grounding": "REVIEW_IMAGE",
        "diagnosis": "FINALIZE_DIAGNOSIS",
    },
    "default_actions": [
        "ASK",
        "REQUEST_LAB",
        "REVIEW_IMAGE",
        "UPDATE_HYPOTHESIS",
        "FINALIZE_DIAGNOSIS",
    ],
    "always_include_actions": ["UPDATE_HYPOTHESIS"],
    "finalize_action": "FINALIZE_DIAGNOSIS",
}


# Applicability heuristics and thresholds used by the online controller
APPLICABILITY_CONFIG = {
    "reusable_memory_score": 0.30,
    "unsafe_block_score": 0.35,
    "skill_apply_score": 0.40,
    # Memory applicability should select reusable retrieved memories. Rule-only
    # action guards are off by default so guidance is driven by memory content,
    # not hard-coded safety blocks.
    "enable_rule_action_guards": False,
    "hard_block_finalize_on_high_risk": False,
    "hard_block_finalize_missing_info_min": 0,
    "image_unreviewed_warning": False,
    "hard_block_reason": "Configured safety rule blocked this action.",
    "risk_warning_text": {
        "high_finalize_risk": "finalize_risk is high",
        "missing_info": "multiple critical missing information slots remain unresolved",
        "image_unreviewed": "image modality is available but not yet reviewed",
    },
}

TRACE_CONFIG = {
    # Compact trace is for memory debugging. Full turn details remain in trajectory_*.json.
    "format": "compact",
    "write_jsonl_snapshot": False,
    "include_full_turn": False,
    "include_observation_payload": False,
    "include_llm_io": False,
    "include_prompt_payload": False,
    "include_applicability_debug": False,
    "max_text_chars": 0,
    "max_hits_per_type": 5,
}

LLM_CONFIG = {
    "experience_extraction_max_output_tokens": 2000,
    "experience_extraction_max_turns": 15,
    "experience_extraction_max_text_chars": 1600,
}
