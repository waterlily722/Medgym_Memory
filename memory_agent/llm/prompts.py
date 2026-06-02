from __future__ import annotations

import json
from typing import Any

from .schemas import (
    APPLICABILITY_SCHEMA,
    CASE_MEMORY_SCHEMA,
    EXPERIENCE_EXTRACTION_SCHEMA,
    EXPERIENCE_MERGE_SCHEMA,
    QUERY_BUILDER_SCHEMA,
    SKILL_EXTRACTION_SCHEMA,
)

STRICT_JSON_RULES = """
Return exactly one valid JSON object.
Do not use markdown fences, bullets, commentary, or free text outside JSON.
Follow the schema exactly. Keep every required field.
If a field is unsupported, keep the field and use an empty value.
Use only information explicitly provided in the input.
Do not invent case ids, episode ids, turn ids, diagnoses, tests, findings, image findings, patient details, or source information.
""".strip()


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def query_builder_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are building retrieval queries for a clinical memory system used by a diagnostic doctor agent.

{STRICT_JSON_RULES}

Task:
Create two compact retrieval queries from CaseMemory.

Input case_memory contains:
- chief_complaint
- diagnostic_strategy
- efficient_turn_information
- ineffective_turn_information
- prior_information_summary

1. query_text:
   Query for EXPERIENCE memory.
   EXPERIENCE memory stores case-derived medical diagnostic insights from previous patients.

   Describe the current clinical evidence state:
   known symptoms, signs, positive findings, negative findings, exam/lab/imaging results,
   relevant history, relevant demographics, and unresolved diagnostic uncertainty if stated.

   The goal is to retrieve similar medical diagnostic insights.
   Do not write a next-step plan or final diagnosis.

2. skill_query_text:
   Query for SKILL memory.
   SKILL memory stores reusable diagnostic decision workflows.

   Describe the current decision point:
   presenting problem, current uncertainty, evidence already gathered,
   missing or unreviewed evidence, ineffective actions already tried,
   and what kind of workflow guidance is needed.

   The goal is to retrieve useful action-selection guidance.
   Do not invent a plan beyond the case memory.

Rules:
- Use only explicit CaseMemory information.
- Keep both queries concise, grounded, and retrieval-friendly.
- query_text targets diagnostic experience; skill_query_text targets decision workflow.

Schema:
{_dump(QUERY_BUILDER_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def case_memory_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical case-memory extractor for a diagnostic doctor agent.

{STRICT_JSON_RULES}

Task:
Update CaseMemory from case metadata and turn records.

Input contains:
- case_id, turn_id, chief_complaint
- initial_information
- current_turn_information
- acquired_turn_information: prior formatted turns excluding initial_information and current_turn_information

Core task:
1. Classify each current_turn_information record as effective or ineffective.
2. Update diagnostic_strategy and prior_information_summary.
3. Return CaseMemory only.

Current-turn classification:
- Use current_turn_information for the factual content of the current turn.
- Effective: adds useful positive/negative evidence, exam/lab/imaging/tool evidence, or medical knowledge that changes diagnostic uncertainty or next action.
- Ineffective: generic, unavailable, unknown without clinical signal, off-topic, redundant, or no-result response.
- Each current record must appear in exactly one list.

CaseMemory fields:
1. diagnostic_strategy:
   Write a concise clinician reasoning state:
   known evidence; leading uncertainty or differential direction; most useful missing evidence; whether diagnosis should not yet be finalized.

2. efficient_turn_information:
   Raw ledger of initial_information plus effective records.
   Copy records exactly. Do not summarize or infer.

3. ineffective_turn_information:
   Raw ledger of ineffective records.
   Copy records exactly. Do not summarize or infer.

4. prior_information_summary:
   Concise clinical handoff summary of all gathered patient facts.
   Include positive and negative symptoms, exam findings, labs/imaging, history, medications, allergies, family/social history when available.
   State what was asked or checked and what the answer was.
   Do not mention whether turns were effective.

Rules:
- Use only provided case metadata and turn records.
- Keep chief_complaint exactly equal to input.chief_complaint.
- Do not invent diagnoses, tests, findings, or patient details.
- Raw ledger fields must copy records exactly.

Schema:
{_dump(CASE_MEMORY_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def applicability_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical memory applicability judge for a diagnostic doctor agent.

{STRICT_JSON_RULES}

Task:
Decide whether one retrieved memory is relevant and useful for the doctor's current diagnostic reasoning or next action.

A memory is useful only if it matches the CURRENT DECISION POINT.
Do not apply a memory only because symptoms, disease labels, or organ systems overlap.

Judge match by:
- clinical trigger
- current uncertainty
- evidence already reviewed
- missing or unreviewed evidence
- intended lesson or workflow in the memory
- apply/do-not-use conditions
- action timing

Decision values:
- apply: the memory closely matches the current case state or decision point and is useful for diagnostic reasoning or next action.
- ignore: the memory is irrelevant, too generic, too case-specific, weakly matched, contradicted by current evidence, or premature.

Clinical rules:
- Experiences are case-derived medical diagnostic insights. Apply them only when the current case has a similar clinical pattern, evidence state, or diagnostic implication.
- Skill memory is workflow guidance; apply only when trigger, boundary, and timing match.
- When uncertain, choose ignore.

Schema:
{_dump(APPLICABILITY_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def experience_extraction_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical experience extractor for a medical memory system.

{STRICT_JSON_RULES}

Task:
Extract up to payload.max_experiences reusable ExperienceCards from this episode.

Input:
- case_context
- diagnostic_trajectory: effective interactions with action, response, reward, turn_reward, importance, and evidence_records
- ineffective_interactions: low-value or no-result interactions
- episode_outcome

Experience memory stores medical diagnostic insights learned from previous cases.
It is not an action policy or workflow.

Extract experiences about:
- symptom/finding patterns associated with diagnosis
- discriminative clues between similar conditions
- important negative findings
- lab, imaging, exam, or tool findings
- missed or misleading evidence in failed cases

Good experience:
"Finger swelling with redness, pus, and nail-biting history supports paronychia; nail-biting is a useful discriminative clue."

Bad experience:
"Ask targeted questions."  # generic workflow advice

Rules:
- Each experience must be grounded in specific case evidence.
- Prefer clinically discriminative or commonly missed clues.
- Do not use gold diagnosis to add facts absent from the trajectory.
- Use episode_outcome.success only to choose outcome_type.
- For failed episodes, return at most 2 high-confidence negative experiences.
- Skip generic, duplicate, or weakly supported lessons.

Confidence:
- 0.9-1.0: specific, directly supported, clinically important, reusable.
- 0.7-0.89: useful but indirectly supported.
- Below 0.7: do not output.

Write each ExperienceCard:
- text: concise medical diagnostic insight with clinical pattern, key evidence, and diagnostic implication.
- outcome_type: "positive" or "negative".
- confidence: per rules above.
- support_count: 1 unless otherwise stated.
- source: preserve only provided ids.

Schema:
{_dump(EXPERIENCE_EXTRACTION_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def experience_merge_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical experience-memory curator.

{STRICT_JSON_RULES}

Task:
Decide whether a new ExperienceCard should be merged with an existing memory or inserted as separate memory.

Allowed decisions:
- merge
- insert_new

Goal:
Keep the memory library compact without losing important medical distinctions.

Use merge only when all are true:
1. Same medical insight.
2. Same or compatible evidence pattern.
3. Same diagnostic implication.
4. Compatible outcome direction.
5. Compatible boundary or patient context.
6. Same retrieval purpose.

Use insert_new when:
- the medical insight differs
- the evidence pattern differs clinically
- the diagnostic implication differs
- the boundary differs or is unclear
- one memory is an exception or contrast
- merging would lose an important symptom, finding, test, modality, or nuance
- you are uncertain

Merge rules:
- Preserve the selected existing memory_id.
- Generalize only when supported by both memories.
- Keep concrete clinical evidence and diagnostic implication.
- Do not turn the merged memory into generic advice.
- Combine only provided sources.

Output format:
{{
  "merge_decision": "merge|insert_new",
  "target_memory_ids": [],
  "reason": "...",
  "merged_experience": {{
    "memory_id": "...",
    "memory_type": "experience",
    "text": "...",
    "outcome_type": "positive|negative",
    "confidence": 0.0,
    "support_count": 1,
    "source": {{
      "case_ids": [],
      "episode_ids": [],
      "turn_ids": []
    }}
  }}
}}

Rules:
- If merge_decision is "merge", merged_experience is the final merged card.
- If merge_decision is "insert_new", merged_experience is the new card unchanged or lightly cleaned.
- Never output discard or conflict.

Schema:
{_dump(EXPERIENCE_MERGE_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def skill_extraction_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical skill miner for a self-improving medical diagnostic agent.

{STRICT_JSON_RULES}

Task:
Extract up to payload.max_skills candidate SkillCards from one successful diagnostic episode.

Input:
- case_context
- diagnostic_trajectory: effective interactions with action, response, reward, turn_reward, importance, and evidence_records
- ineffective_interactions: low-value or no-result interactions
- episode_outcome

Skill memory stores reusable local diagnostic decision patterns.
It is not a medical fact, isolated clue, or symptom-diagnosis association.

A skill answers:
- when this local decision pattern applies
- what action or short action sequence is useful
- why it helps reduce diagnostic uncertainty
- when it should not be applied

Important:
A skill extracted from one episode should be treated as low-support skill evidence.
Its support_count is 1 unless otherwise stated.

Each skill should include:
1. trigger: clinical pattern or uncertainty that activates the skill
2. procedure: 1-3 diagnostic actions with rationale
3. boundary: when this skill should not apply

Do not extract:
- isolated medical facts or single clues
- symptom-diagnosis associations
- "consider diagnosis X"
- generic advice such as "ask more questions" or "order tests"
- actions contradicted by the episode outcome
- skills without a clear trigger or decision logic

Generalization:
- Remove incidental case details.
- Keep demographics or exposures only if they affect the decision.
- Keep clinical trigger, action logic, evidence threshold, and boundary.

Confidence:
- 0.9-1.0: local decision pattern is visible, evidence-supported, clinically coherent, and outcome-supported.
- 0.7-0.89: useful but partially inferred.
- Below 0.7: do not output.

Output format:
{{
  "skills": [
    {{
      "skill_text": "When [clinical trigger/uncertainty], do [local diagnostic action or short sequence] because [rationale]. Do not apply when [boundary].",
      "procedure": [
        {{"action_type": "...", "action_label": "..."}}
      ],
      "tags": [],
      "confidence": 0.0,
      "support_count": 1,
      "source": {{
        "case_ids": [],
        "episode_ids": [],
        "turn_ids": []
      }}
    }}
  ]
}}

Schema:
{_dump(SKILL_EXTRACTION_SCHEMA)}

Input:
{_dump(payload)}
""".strip()