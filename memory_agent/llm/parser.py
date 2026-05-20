from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_text(raw: str) -> str:
    raw = (raw or "").strip()

    block_match = JSON_BLOCK_RE.search(raw)
    if block_match:
        raw = block_match.group(1).strip()

    if raw.startswith("{") and raw.endswith("}"):
        return raw

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]

    return "{}"


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _apply_defaults(parsed: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fallback or {})
    merged.update(parsed or {})
    return merged


def _validate_and_repair(
    parsed: dict[str, Any],
    schema: dict[str, Any],
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    repaired = _apply_defaults(parsed, fallback)
    errors: list[str] = []

    for field in schema.get("required", []):
        if field not in repaired:
            repaired[field] = fallback.get(field)
            errors.append(f"missing:{field}")

    for field in schema.get("list_fields", []):
        repaired[field] = _coerce_list(repaired.get(field))

    for field in schema.get("dict_fields", []):
        repaired[field] = _coerce_dict(repaired.get(field))

    enum_fields = schema.get("enum_fields", {}) or {}
    for field, allowed in enum_fields.items():
        if field in repaired and repaired[field] not in allowed:
            errors.append(f"enum:{field}")
            repaired[field] = fallback.get(field, allowed[0] if allowed else "")

    range_fields = schema.get("range_fields", {}) or {}
    for field, bounds in range_fields.items():
        if field not in repaired:
            continue
        try:
            value = float(repaired[field])
            min_value = bounds.get("min")
            max_value = bounds.get("max")

            if min_value is not None:
                value = max(float(min_value), value)
            if max_value is not None:
                value = min(float(max_value), value)

            if isinstance(repaired[field], int):
                repaired[field] = int(value)
            else:
                repaired[field] = value
        except Exception:
            errors.append(f"range:{field}")
            repaired[field] = fallback.get(field)

    return repaired, errors


def parse_validate_repair(
    raw_output: str,
    schema: dict[str, Any],
    fallback: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, list[str]]:
    """
    Return:
      parsed_dict, ok, errors

    This is intentionally simple. Nested validation is handled in the caller
    when needed, especially for {"experiences": [...]}.
    """
    fallback = fallback or {}

    try:
        parsed = json.loads(_extract_json_text(raw_output))
        if not isinstance(parsed, dict):
            parsed = {}
    except Exception:
        parsed = {}

    repaired, errors = _validate_and_repair(parsed, schema, fallback)
    ok = len(errors) == 0

    return repaired, ok, errors