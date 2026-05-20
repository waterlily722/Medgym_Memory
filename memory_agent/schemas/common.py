from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Any


class OutcomeType(str, Enum):
    """Canonical outcome types for ExperienceCard."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    UNSAFE = "unsafe"

    @classmethod
    def _missing_(cls, value: object) -> "OutcomeType":
        if isinstance(value, str):
            normalized = value.lower().replace(" ", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        return cls.PARTIAL_SUCCESS


def _convert(value: Any) -> Any:
    if is_dataclass(value):
        return _convert(asdict(value))

    if isinstance(value, dict):
        return {str(key): _convert(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_convert(item) for item in value]

    if isinstance(value, tuple):
        return [_convert(item) for item in value]

    return value


class SerializableMixin:
    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} must be a dataclass")
        return _convert(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        if data is None:
            raise ValueError(f"{cls.__name__}.from_dict received None")
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__}.from_dict expected dict, got {type(data)}")
        values = dict(data)
        if is_dataclass(cls):
            allowed = {field.name for field in fields(cls)}
            values = {key: value for key, value in values.items() if key in allowed}
        return cls(**values)
