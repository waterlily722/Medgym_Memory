from __future__ import annotations

from datetime import datetime
from typing import Any

from ..schemas import DistilledEpisode, EpisodeFeedback, TurnRecord


def _to_dict_record(record: Any) -> dict[str, Any] | None:
    if isinstance(record, TurnRecord):
        return record.to_dict()
    if isinstance(record, dict):
        return record
    return None


def _turn_records_from_trajectory(trajectory: Any) -> list[dict[str, Any]]:
    """
    Expected source:
      trajectory.info["memory_agent"]["turn_records"]

    Also accepts:
      dict with memory_agent.turn_records
      list[TurnRecord|dict]
    """
    if isinstance(trajectory, list):
        records = trajectory
    elif isinstance(trajectory, dict):
        info = trajectory.get("info", trajectory)
        records = ((info.get("memory_agent") or {}).get("turn_records") or [])
    else:
        info = getattr(trajectory, "info", {}) or {}
        records = ((info.get("memory_agent") or {}).get("turn_records") or [])

    cleaned: list[dict[str, Any]] = []
    for record in records:
        item = _to_dict_record(record)
        if item is not None:
            cleaned.append(item)
    return cleaned


def distill_from_trajectory(
    trajectory: Any,
    feedback: EpisodeFeedback | dict[str, Any],
) -> DistilledEpisode:
    """
    Compress raw trajectory into the minimal package needed by offline extraction.

    This function should not extract ExperienceCards and should not mine skills.
    """
    feedback = (
        feedback
        if isinstance(feedback, EpisodeFeedback)
        else EpisodeFeedback.from_dict(feedback)
    )
    turn_records = _turn_records_from_trajectory(trajectory)

    summary = (
        f"timestamp={datetime.utcnow().isoformat()}; "
        f"turn_count={len(turn_records)}; "
        f"total_reward={feedback.total_reward}; "
        f"success={feedback.success}; "
        f"final_diagnosis={feedback.final_diagnosis}; "
        f"gold_diagnosis={feedback.gold_diagnosis}; "
        f"summary={feedback.summary}"
    )

    return DistilledEpisode(
        episode_id=feedback.episode_id,
        case_id=feedback.case_id,
        turn_records=turn_records,
        feedback=feedback.to_dict(),
        summary=summary,
    )