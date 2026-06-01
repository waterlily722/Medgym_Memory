from __future__ import annotations

QUERY_BUILDER_SCHEMA = {
    "required": ["query_text", "skill_query_text"],
    "list_fields": [],
    "dict_fields": [],
}

CASE_MEMORY_SCHEMA = {
    "required": [
        "case_id",
        "turn_id",
        "chief_complaint",
        "diagnosis_goal",
        "efficient_turn_information",
        "ineffective_turn_information",
        "prior_information_summary",
    ],
    "list_fields": ["efficient_turn_information", "ineffective_turn_information"],
    "dict_fields": [],
}

APPLICABILITY_SCHEMA = {
    "required": [
        "memory_id",
        "memory_type",
        "decision",
        "reason",
        "action_bias",
        "blocked_actions",
    ],
    "list_fields": ["blocked_actions"],
    "dict_fields": ["action_bias"],
    "enum_fields": {
        "memory_type": ["experience", "skill", "knowledge"],
        "decision": ["apply", "ignore"],
    },
}

EXPERIENCE_EXTRACTION_SCHEMA = {
    "required": ["experiences"],
    "list_fields": ["experiences"],
    "dict_fields": [],
}

EXPERIENCE_MERGE_SCHEMA = {
    "required": [
        "merge_decision",
        "target_memory_ids",
        "reason",
        "merged_experience",
    ],
    "list_fields": ["target_memory_ids"],
    "dict_fields": ["merged_experience"],
    "enum_fields": {
        "merge_decision": ["insert_new", "merge"],
    },
}

SKILL_SCHEMA = {
    "required": [
        "memory_id",
        "memory_type",
        "skill_text",
        "procedure",
        "tags",
        "confidence",
        "support_count",
        "source",
    ],
    "list_fields": [
        "procedure",
        "tags",
    ],
    "dict_fields": ["source"],
    "enum_fields": {
        "memory_type": ["skill"],
    },
    "range_fields": {
        "confidence": {"min": 0.0, "max": 1.0},
        "support_count": {"min": 1, "max": 999999},
    },
}

SKILL_EXTRACTION_SCHEMA = {
    "required": ["skills"],
    "list_fields": ["skills"],
    "dict_fields": [],
}
