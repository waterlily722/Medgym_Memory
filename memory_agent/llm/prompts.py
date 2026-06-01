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
You are building a retrieval query for a clinical experience memory store.

{STRICT_JSON_RULES}

Task:
Create one compact query_text that describes the KNOWN clinical facts from the
current case, so that relevant past experiences can be retrieved by text similarity.

The input contains case_memory with:
- chief_complaint: the patient's presenting complaint
- diagnosis_goal: current diagnostic strategy
- efficient_turn_information: evidence-bearing information gathered so far
- prior_information_summary: summary of earlier evidence

CRITICAL: query_text is a RETRIEVAL QUERY for matching similar past experiences.
It is NOT a diagnostic plan, action guide, or next-step recommendation.

Rules:
- Only include KNOWN clinical facts: symptoms, signs, lab/imaging findings,
  PMH, medications, demographics.
- If clinical information is very sparse, generate a minimal query from what IS known.
  A short factual query is better than a speculative one.

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
Create a compact CaseMemory from CaseState.

CaseState is a faithful ledger of information already exposed to the doctor agent.
CaseState.current_turn contains only the newly exposed information for this turn.
CaseState.acquired_information contains all turns.
Do not add facts that are not in CaseState. Do not infer diagnoses, missing tests, risk labels, or hidden patient information.

CaseMemory should contain:
1. chief_complaint:
   Copy the chief complaint from CaseState. Keep it concise.

2. diagnosis_goal:
   Describe the current diagnostic objective, what has already been attempted,
   and what kind of next evidence should be prioritized.
   If prior patient questions or tools produced no useful information, mention
   that repeated questioning should be avoided and another evidence route or
   reasoning step should be considered.

3. efficient_turn_information:
   Copy rule_efficient_turn_information exactly.
   Do not summarize, rewrite, deduplicate, reorder, add final diagnosis, or add
   inferred facts.
   This list should remain a rule-extracted ledger of initial information and
   full effective doctor-patient/tool interaction records.

4. prior_information_summary:
   Actively summarize older useful information and ineffective/no-result interactions.
   Include enough detail to prevent repeated questions or repeated low-value tool calls.
   If current_turn is ineffective, do not add it to efficient_turn_information,
   but do update diagnosis_goal and prior_information_summary to record that
   the question/tool was attempted and did not provide evidence.
   Do not preserve raw source information. Write a clinical summary, not a field dump.

Rules:
- Use only CaseState.current_turn and CaseState.acquired_information.
- Do not create a new summary inside efficient_turn_information.
- Do not include hidden benchmark labels, judge settings, internal metadata, or
  technical source paths.
- Do not include patient identifiers.
- Keep efficient_turn_information exactly equal to rule_efficient_turn_information.
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
Extract reusable ExperienceCards from the clinical episode trace. Return no more than payload.max_experiences experiences.

The goal is to capture high-value medical diagnostic experience: discriminative clinical cues, missing evidence that mattered, evidence interpretation, and reasoning lessons that a doctor should remember in a similar future case.
Experience memory focuses on medical diagnostic knowledge and high-value evidence fragments: symptoms, negative evidence, tests, imaging/labs, differential-diagnosis cues, missed clues, and reasoning pitfalls.

Use episode_outcome.success to choose the experience polarity:
- If success=true, extract positive experiences from turns/fragments that materially supported the correct diagnostic path.
- If success=false, use the provided gold_diagnosis to understand what was missed, then extract negative/cautionary experiences about missed discriminative evidence, wrong direction, low-value information gathering, or premature closure.
- For failed episodes, do not create memories that simply say "consider the gold diagnosis". Extract the evidence lesson: what discriminative information was missing, misread, over-weighted, or should have been sought.

STRICT rules for failed episodes (negative experiences):
- Each negative experience MUST name the specific evidence missed AND the
  differential it would distinguish. 
- Maximum 1 negative experience per failed episode unless there are clearly
  distinct evidence gaps (e.g., missed lab AND missed imaging).

Use two signals before writing any ExperienceCard:

1. Objective turn-ablation signal:
   The trace may contain turn_importance fields such as conf_before, conf_after, delta, turn_reward, and importance.
   - Positive/high-value turns often have positive delta or high importance.
   - Negative/cautionary turns may have negative delta, zero value despite cost, or occur before a wrong/premature final diagnosis.
   - These signals are not perfect; use them as evidence, not as the only rule.

2. Clinical value judgment over effective turns:
   Mentally score each evidence-bearing turn for whether it:
   - provides discriminative information for the differential diagnosis;
   - changes the best next clinical action;
   - rules in/rules out a plausible diagnosis;
   - prevents a high-risk miss or premature closure;
   - exposes a misleading, redundant, or low-value interaction.
   Prefer turns/fragments with both objective support and clinical value.


ExperienceCard fields:

1. memory_id:
   Use the provided id if available.
   Do not invent source ids.
   If the schema expects an empty value, use "".

2. memory_type:
   Always use "experience".

3. text:
   Write one reusable, semi-structured clinical experience in concise English.
   The structure is a guide, not a rigid template. Make the final text natural,
   compact, and clinically reusable.

   Include the following elements when supported by the episode:
   - applicable disease type / syndrome / symptom context
   - clinical uncertainty state
   - missing key information
   - main differential diagnosis affected by the missing information
   - high-value diagnostic information, missed clue, or evidence interpretation
   - what discriminative value this information has
   - recommended evidence-gathering direction only when needed to explain the lesson
   - boundary: when not applicable or what risk exists

4. outcome_type:
   Use:
   - positive: successful reusable local action
   - negative: failed, unsafe, missed, or cautionary local lesson

5. confidence:
   Estimate how reliable and reusable this experience is.

6. support_count:
   Use 1 for a newly extracted single-turn or single-episode experience unless input states otherwise.

7. source:
   Preserve provided provenance only.
   Use a compact dict, for example:
   {{
     "case_ids": [],
     "episode_ids": [],
     "turn_ids": []
   }}
   Do not invent ids.

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
