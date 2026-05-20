from __future__ import annotations

from typing import Any

from ..schemas import KnowledgeItem


def unwrap_osce_examination(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    osce = payload.get("OSCE_Examination")
    if isinstance(osce, dict):
        return osce
    return payload


def nested_get(payload: dict[str, Any] | None, path: list[str], default: Any = None) -> Any:
    current: Any = payload if isinstance(payload, dict) else {}
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def diagnosis_to_text(diagnosis: Any) -> str:
    if isinstance(diagnosis, dict):
        for key in ("icd_title", "title", "name", "icd_code"):
            value = diagnosis.get(key)
            if value:
                return str(value)
        return str(diagnosis)
    if diagnosis is None:
        return ""
    return str(diagnosis)


def extract_gold_diagnosis(payload: dict[str, Any] | None) -> str:
    osce = unwrap_osce_examination(payload)
    for key in ("Correct_Diagnosis", "Principal_Diagnosis", "Final_Result"):
        text = diagnosis_to_text(osce.get(key))
        if text:
            return text
    return ""


def knowledge_items_from_payload(payload: dict[str, Any] | None, case_id: str = "") -> list[KnowledgeItem]:
    osce = unwrap_osce_examination(payload)
    knowledge = osce.get("knowledge")
    items: list[KnowledgeItem] = []
    if not isinstance(knowledge, dict):
        return items
    principal = knowledge.get("principal_diagnosis") or {}
    matched = principal.get("matched_knowledge") or []
    if not isinstance(matched, list):
        return items

    for index, item in enumerate(matched):
        if not isinstance(item, dict):
            continue
        # Build minimal KnowledgeItem according to current schema
        title = str(item.get("name") or item.get("icd_title") or item.get("ICD-10") or f"knowledge_{index}")
        content_parts = [item.get("introduction", ""), item.get("signs_and_symptoms", ""), item.get("diagnosis", ""), item.get("prognosis", "")]
        content = "\n".join(str(part) for part in content_parts if part)
        disease_tags = [str(item.get(field)) for field in ("ICD-10", "ICD-10-CM", "ICD-9", "ICD-9-CM") if item.get(field)]
        # Consolidate tags into a single list for the minimal schema
        tags = disease_tags
        if case_id:
            tags.append(f"case:{case_id}")
        items.append(
            KnowledgeItem(
                memory_id=f"kn_{case_id}_{index}" if case_id else f"kn_{index}",
                content=(title + "\n" + content)[:4000],
                tags=tags,
                source=str(item.get("_source") or item.get("source_md") or "wiki"),
                confidence=0.9,
            )
        )
    return items
