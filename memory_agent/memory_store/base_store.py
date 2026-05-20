from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class JsonMemoryStore:
    def __init__(
        self,
        root_dir: str | Path,
        filename: str,
        item_factory: Callable[[dict[str, Any]], Any],
        strict_json: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.path = self.root_dir / filename
        self.item_factory = item_factory
        self.strict_json = strict_json

        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def list_all(self) -> list[Any]:
        return [self.item_factory(row) for row in self._read_raw()]

    def append(self, item: Any) -> Any:
        row = self._to_row(item)
        memory_id = self._memory_id(row)

        if self.find_by_id(memory_id) is not None:
            logger.warning("Duplicate memory_id in append: %s", memory_id)
            raise ValueError(f"memory_id already exists: {memory_id}")

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        return item

    def upsert(self, item: Any) -> Any:
        row = self._to_row(item)
        memory_id = self._memory_id(row)

        rows = self._read_raw()
        replaced = False

        for index, existing in enumerate(rows):
            if str(existing.get("memory_id") or "") == memory_id:
                rows[index] = row
                replaced = True
                break

        if not replaced:
            rows.append(row)

        self._write_raw(rows)
        return item

    def find_by_id(self, memory_id: str) -> Any | None:
        memory_id = str(memory_id or "")
        if not memory_id:
            return None

        for row in self._read_raw():
            if str(row.get("memory_id") or "") == memory_id:
                return self.item_factory(row)

        return None

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def _read_raw(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        if not self.path.exists():
            return rows

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    if self.strict_json:
                        raise ValueError(
                            f"Invalid JSON in {self.path} at line {line_number}"
                        ) from exc
                    logger.warning(
                        "Skipped invalid JSON in %s at line %d", self.path, line_number
                    )
                    continue

                if not isinstance(value, dict):
                    if self.strict_json:
                        raise ValueError(
                            f"Expected JSON object in {self.path} at line {line_number}"
                        )
                    logger.warning(
                        "Skipped non-dict JSON in %s at line %d", self.path, line_number
                    )
                    continue

                rows.append(value)

        return rows

    def _write_raw(self, rows: list[dict[str, Any]]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _to_row(self, item: Any) -> dict[str, Any]:
        if hasattr(item, "to_dict"):
            row = item.to_dict()
        elif isinstance(item, dict):
            row = dict(item)
        else:
            raise TypeError(f"Unsupported memory item type: {type(item)}")

        self._memory_id(row)
        return row

    def _memory_id(self, row: dict[str, Any]) -> str:
        memory_id = str(row.get("memory_id") or "")
        if not memory_id:
            raise ValueError("memory item must contain non-empty memory_id")
        return memory_id