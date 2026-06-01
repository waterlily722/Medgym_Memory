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
Do not invent source ids, case ids, turn ids, diagnoses, test results, or image findings.
""".strip()


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def query_builder_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are building retrieval queries for a clinical memory system.

{STRICT_JSON_RULES}

Task:
Create two compact retrieval queries from CaseMemory:

The input contains case_memory with:
- chief_complaint: the patient's presenting complaint
- diagnosis_goal: current diagnostic strategy
- efficient_turn_information: evidence-bearing information gathered so far
- ineffective_turn_information: attempted turns that did not add useful evidence
- prior_information_summary: summary of earlier evidence

1. query_text:
   Retrieval query for EXPERIENCE memory. Describe only KNOWN clinical facts,
   such as symptoms, signs, labs, imaging, PMH, medications, demographics, and
   important negative evidence. It is not a diagnostic plan, action guide, or
   next-step recommendation.

2. skill_query_text:
   Retrieval query for SKILL memory. Describe the current decision point:
   presenting problem, uncertainty state, evidence already gathered, failed or
   low-value actions already attempted, and the type of next workflow guidance
   needed. Keep it grounded in observed facts.

Rules:
- Do not invent diagnoses, test results, hidden labels, or patient details.
- If information is sparse, generate short queries from what is known.
- query_text should match similar diagnostic evidence.
- skill_query_text should match reusable action/workflow guidance.

Schema:
{_dump(QUERY_BUILDER_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def case_memory_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical case-memory extractor for a doctor agent.

{STRICT_JSON_RULES}

Task:
Update CaseMemory from compact case metadata and formatted turn records.

The input contains:
- case_id, turn_id, chief_complaint
- initial_information: initially exposed presenting information
- current_turn_information: only the newly exposed information for this turn
- acquired_turn_information: all formatted non-initial turns, including the
  current turn

Core task:
1. Classify each current_turn_information record as effective or ineffective.
2. Update diagnosis_goal and prior_information_summary using both the full
   history and the current turn.
3. Return CaseMemory only; do not add commentary.

Current-turn classification:
- Use current_turn_information only when deciding whether the current turn is
  effective. Do not use acquired_turn_information to rescue, reinterpret, or
  downgrade the current turn.
- Effective means the turn adds useful positive/negative clinical evidence,
  exam/lab/imaging/tool evidence, or retrieved medical knowledge that changes
  the differential diagnosis, confidence, or best next action.
- Ineffective means the turn is generic, unavailable, unknown without clinical
  signal, off-topic, redundant, or a no-result/no-knowledge-base response.
- Each current_turn_information record must appear in exactly one list:
  efficient_turn_information if effective, otherwise ineffective_turn_information.

CaseMemory should contain:
1. diagnosis_goal:
   A concise strategic objective: current uncertainty, useful evidence already
   gathered, failed/low-value actions already attempted, and the next evidence
   route that should be prioritized. Use acquired_turn_information for history
   and current_turn_information for the latest progress.

2. efficient_turn_information:
   A raw ledger of initial_information plus clinically effective records.
   Copy full formatted records exactly. Do not summarize, rewrite, deduplicate,
   reorder, add final diagnoses, or add inferred facts.

3. ineffective_turn_information:
   A raw ledger of clinically ineffective records. Copy full formatted records
   exactly. Do not summarize, rewrite, deduplicate, reorder, add final
   diagnoses, or add inferred facts.

4. prior_information_summary:
   A compact clinical summary of useful prior evidence and ineffective/no-result
   interactions. Include enough detail to prevent repeated questions or repeated
   low-value tool calls. Use acquired_turn_information for the overall summary
   and current_turn_information for the latest attempted action/result. Write a
   clinical summary, not a field dump.

Rules:
- Use only case_id, turn_id, chief_complaint, initial_information,
  current_turn_information, and acquired_turn_information.
- Do not invent diagnoses, missing tests, hidden labels, source ids, or patient
  details.
- Do not create summaries inside efficient_turn_information or
  ineffective_turn_information; those fields are raw ledgers.
- Do not include hidden benchmark labels, judge settings, internal metadata, or
  technical source paths.
- Do not include patient identifiers.
- Keep chief_complaint exactly equal to input.chief_complaint.
- Return exactly the schema below.

Schema:
{_dump(CASE_MEMORY_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def applicability_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical memory applicability judge for a doctor agent.

{STRICT_JSON_RULES}

Task:
Select whether one retrieved memory is reusable for the doctor's current next action.

A memory is useful only if it matches the CURRENT DECISION POINT.
Do not apply a memory merely because the symptom, disease label, or organ system overlaps.

Compare the retrieved memory with the current case using:
1. clinical trigger
2. uncertainty state
3. evidence already reviewed
4. missing or unreviewed evidence
5. intended next action
6. apply/do-not-use conditions described in the memory text

Decision values:
- apply:
  The memory is reusable now: it closely matches the current trigger, uncertainty state, apply conditions, and action timing.
  It should be included as active guidance.

- ignore:
  The memory is not reusable now: it is irrelevant, too generic, too case-specific,
  only partially matched, contradicted by current evidence, or depends on unavailable evidence.

Clinical memory rules:
1. Skills are workflow reminders, not final answers. Apply a skill only when its clinical situation and embedded boundary/risk
   conditions match strongly.
2. Experiences are local decision lessons. Apply them only when the current decision point is similar.

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
Extract up to payload.max_experiences reusable ExperienceCards from this clinical episode.

Experience memory stores action-level diagnostic lessons, not workflows. Capture
high-value evidence fragments and reasoning lessons that should help in a
similar future case: discriminative symptoms, meaningful negative evidence,
labs/imaging/exam interpretation, missed clues, differential-diagnosis cues,
low-value evidence gathering, and premature-closure risks.

Use final_case_memory.efficient_turn_information as the main source of
evidence-bearing turns. Use final_case_memory.ineffective_turn_information only
for cautionary lessons about low-value, unavailable, redundant, or misleading
actions.

Use episode_outcome.success to choose the experience polarity:
- success=true: extract positive lessons from turns/fragments that materially
  supported the correct diagnostic path.
- success=false: extract negative/cautionary lessons about what evidence was
  missed, misread, over-weighted, not sought, or gathered at low value.
- Failed episodes must not produce "consider the gold diagnosis" memories.
  Name the specific missed evidence and the differential it would distinguish.
  Return at most 1 negative experience unless there are clearly separate gaps
  such as one lab gap and one imaging gap.

Selection gate before writing a card:
- Prefer turns with positive delta, positive turn_reward, high importance, or a
  clear clinical role in the final useful evidence chain.
- Include a low/negative/zero-value turn only if it teaches a reusable caution.
- Keep a lesson only if it changes a differential, next action, rule-in/rule-out
  reasoning, high-risk miss, or premature-closure risk.
- Skip isolated disease labels, generic advice, duplicate facts, and patient
  identifiers.

Write each ExperienceCard:
- memory_id: use a provided id only; otherwise "".
- memory_type: always "experience".
- text: one concise reusable lesson in natural English. Include the clinical
  context, uncertainty state, key evidence or missed evidence, affected
  differential, discriminative value, and boundary/risk when supported.
- outcome_type: "positive" for successful reusable lessons, "negative" for
  failed/unsafe/missed/cautionary lessons.
- confidence: estimate reliability and reusability from 0 to 1.
- support_count: 1 unless the input states otherwise.
- source: preserve only provided case_ids, episode_ids, and turn_ids; do not
  invent provenance.

Output format:
{{
  "experiences": [
    {{
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
  ]
}}

Schema:
{_dump(EXPERIENCE_EXTRACTION_SCHEMA)}

Input:
{_dump(payload)}
""".strip()


def experience_merge_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are a clinical memory-library curator.

{STRICT_JSON_RULES}

Task:
Decide whether a new ExperienceCard should be merged with an existing memory or inserted as a separate memory.

Allowed decisions:
- merge
- insert_new

Goal:
Keep the memory library compact and retrieval-friendly without losing clinically important distinctions.

Use merge when all are true:
1. Same clinical decision point: The doctor is facing the same kind of uncertainty or next-step pressure.
2. Same or compatible evidence state.
3. Same or clinically inseparable action lesson.
4. Compatible outcome direction:The memories should teach the same lesson.
5. Compatible boundary: The apply/do-not-use conditions should not contradict each other.
6. Same retrieval purpose: A future doctor would want to retrieve them for the same reason.

Use insert_new when:
- the action lessons differ;
- the evidence states differ in a clinically meaningful way;
- the boundaries are not clearly compatible;
- one memory may be a true exception to the other;
- merging would lose an important warning, modality, or decision nuance;
- you are uncertain whether they are the same reusable lesson.

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

Rules for merged_experience:
- If merge_decision is "merge", merged_experience must contain the final merged card.
- For merge, preserve the selected existing memory_id in merged_experience.memory_id.
- If merge_decision is "insert_new", merged_experience should be the new memory unchanged or lightly cleaned.
- Never output discard.
- Never output conflict.

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
Extract episode-level SkillCards from one correctly diagnosed long-horizon clinical episode.

Skill memory stores a reusable decision program: under a certain diagnostic information background, what ordered actions should a doctor agent take to reduce uncertainty and reach a supported final diagnosis.
The skill_text should read like a general clinical indication plus workflow rationale, not like a summary of the source patient.

- Skill memory should capture reusable decision/action sequences under a
  diagnostic information background.
- Experience memory should capture high-value medical evidence and diagnostic
  reasoning lessons.
- Do not store isolated medical facts, single clues, or disease labels as skills.
- Skill memory must be more abstract than a single patient. Generalize away
  case-specific demographics and incidental facts unless they are essential
  clinical eligibility criteria for the workflow.
- Define the applicable indication in broad clinical terms: symptom cluster,
  syndrome, screening context, acuity, missing evidence state, and decision
  pressure. Avoid copying a full patient profile.

Use these signals:
- final_case_memory.efficient_turn_information: concise evidence-bearing turns that remained useful at the end of the case.
- final_case_memory.ineffective_turn_information: low-value/no-result turns to avoid, unless a listed action was necessary setup for the successful workflow.
- turn_value_signals: objective turn-ablation style signals. Prefer turns with positive delta, positive turn_reward, or high importance. A zero-reward turn may be included only when it is a necessary setup step in the successful sequence.

Useful skills often involve:
- ask targeted questions
- request focused labs, imaging, or exams
- retrieve focused knowledge at the right time
- update or broaden hypotheses
- delay finalization until evidence was sufficient
- finalize only when the evidence state became adequate

Do not extract:
- skills that only say "consider diagnosis X"
- one-step obvious actions without reusable workflow value
- actions contradicted by the episode outcome
- noisy actions from ineffective turns unless they are explicitly needed as a setup step in the final successful workflow
- exact ages, exact dates, sex, race, patient-specific social details, or one-case identifiers in skill_text, action_label, or tags.


Output format:
{{
  "skills": [
    {{
      "memory_id": "...",
      "memory_type": "skill",
      "skill_text": "For [general clinical indication pattern], use this workflow to [decision goal] while [boundary/risk]. Do not include exact patient age or incidental case details.",
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
