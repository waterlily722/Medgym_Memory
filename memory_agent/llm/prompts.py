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
- current_turn_information: formatted records newly exposed in this turn
- acquired_turn_information: all formatted non-initial turn records, including current_turn_information

Core task:
1. Classify each current_turn_information record as effective or ineffective.
2. Update diagnostic_strategy and prior_information_summary.
3. Return CaseMemory only.

Current-turn classification:
- Use current_turn_information only to decide whether the current turn is effective.
- Do not use acquired_turn_information to rescue, reinterpret, or downgrade the current turn.
- Effective: adds useful positive/negative evidence, exam/lab/imaging/tool evidence, or information that changes diagnostic uncertainty, evidence interpretation, next action, or readiness to finalize.
- Ineffective: generic, unavailable, unknown without clinical signal, off-topic, redundant, repeated, or no-result response.
- Each current_turn_information record must appear in exactly one list:
  efficient_turn_information if effective, otherwise ineffective_turn_information.

CaseMemory fields:

1. diagnostic_strategy:
   Write a concise clinician reasoning state for the next doctor-agent decision.
   Include:
   - current clinical problem
   - key evidence already gathered
   - current diagnostic uncertainty
   - most useful missing information
   - whether existing evidence is enough or not enough to finalize diagnosis
   - any management-relevant risk only if explicitly supported

   This field should help the doctor agent decide what to ask, check, interpret, or finalize next.
   Do not invent diagnoses, tests, risks, or management plans.

2. efficient_turn_information:
   Raw ledger of initial_information plus clinically effective records from acquired_turn_information.
   Copy full formatted records exactly. Do not summarize, rewrite, deduplicate, reorder, or infer.

3. ineffective_turn_information:
   Raw ledger of clinically ineffective records from acquired_turn_information.
   Copy full formatted records exactly. Do not summarize, rewrite, deduplicate, reorder, or infer.

4. prior_information_summary:
   Write a compact clinical handoff summary of all gathered patient facts.
   Include available:
   - history and symptoms
   - relevant positive and negative findings
   - exam, lab, imaging, or tool results
   - interpretation of important evidence when directly supported
   - unresolved missing information
   - low-value or no-result attempts, if present

   The summary should preserve what was asked or checked and what the answer was.
   Do not label items as effective or ineffective in this field.
   Do not recommend treatment unless explicitly supported by the input.

Rules:
- Use only provided case metadata and turn records.
- Keep chief_complaint exactly equal to input.chief_complaint.
- Do not invent diagnoses, tests, findings, risks, treatments, or patient details.
- Raw ledger fields must copy records exactly.
- Keep the memory compact but useful for the next diagnostic turn.

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

Experience memory stores reusable CLINICAL REASONING PATTERNS — how to think through a diagnostic scenario, not just what evidence points to what diagnosis.

It is NOT an action policy, tool-usage rule, or workflow.
It IS a transferable reasoning chain that helps a doctor think better in similar situations.

Each experience should encode a clinical reasoning chain covering (as applicable):
- Trigger context: in what clinical scenario this reasoning applies
- Factors to consider: which differentials, comorbidities, or risk factors matter
- What to verify or cross-check: which indicators, tests, or findings confirm or refute the reasoning
- What to rule out: which critical risks or mimics must be excluded
- How to interpret the combined evidence: the diagnostic implication
- Boundary: when NOT to apply this reasoning pattern

Good experience:
"When a patient presents with acute chest pain and initial ECG is normal, do not prematurely dismiss cardiac etiology — serial troponins over 6-12 hours should be checked to rule out evolving NSTEMI, especially in elderly, diabetic, or female patients where presentations are frequently atypical. Consider aortic dissection if pain is described as tearing and blood pressure is asymmetric between arms. Do not apply this reasoning when ECG already shows clear STEMI pattern or when a trauma history fully explains the pain."

Bad experience:
"Consider cardiac causes for chest pain."   ← too vague, no reasoning chain
"Order troponin for chest pain."             ← this is a tool/action instruction (→ skill), not a reasoning pattern
"Chest pain with normal ECG is not cardiac." ← overgeneralized, wrong conclusion

Rules:
- Each experience must be grounded in explicit episode evidence.
- Write one concise clinical reasoning insight as a coherent paragraph, not labeled sections.
- Cover as many reasoning dimensions as the evidence supports: consider, verify, rule-out, interpret, boundary.
- Prefer reasoning patterns that are discriminative, commonly missed, or counter-intuitive.
- Do not use gold diagnosis to add facts absent from the trajectory.
- Do not extract tool/action instructions (those belong in skill memory).
- Do not extract pure symptom→diagnosis associations without reasoning depth.
- Do not output diagnosis-only or overgeneralized statements.
- Use episode_outcome.success only to choose outcome_type.
- For failed episodes, return at most 2 high-confidence negative experiences focusing on what reasoning went wrong and what should have been considered instead.
- Skip generic, duplicate, weakly supported, or non-reusable lessons.

Confidence:
- 0.9-1.0: directly supported, clinically important, reusable, with clear reasoning chain and boundary.
- 0.7-0.89: useful but partially inferred.
- Below 0.7: do not output.

Write each ExperienceCard:
- text: one complete clinical reasoning insight.
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

Use merge only when both memories have:
- same medical insight
- compatible evidence pattern
- same diagnostic implication
- compatible outcome direction
- compatible boundary or patient context
- same retrieval purpose

Use insert_new when:
- the medical insight differs
- the evidence pattern differs clinically
- the diagnostic implication differs
- the boundary differs or is unclear
- one memory is an exception or contrast
- merging would lose an important symptom, finding, test, modality, negative evidence, or nuance
- you are uncertain

Merge rules:
- Preserve the selected existing memory_id.
- Generalize only when supported by both memories.
- Keep concrete clinical evidence, diagnostic interpretation, and applicability boundary.
- Keep merged text as one concise clinical insight, not labeled fragments.
- Do not turn the merged memory into generic advice.
- Combine only provided sources.

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

Skill memory stores reusable DECISION-TOOL patterns — which tool or action to use, in what order, to resolve a specific diagnostic decision point.

Skill vs Experience boundary:
- Experience = HOW TO THINK: clinical reasoning pattern, what to consider/verify/exclude in a scenario
- Skill = HOW TO ACT: which tool/action to use, in what order, with what expected outcome, to resolve a decision point

A skill is NOT a medical fact, clinical reasoning insight, or symptom-diagnosis association.
A skill IS a concrete, tool-bound action strategy tied to a decision context.

A skill answers:
- when (trigger): at what decision point this action pattern applies
- what (procedure): which tool/action to use, with what specific parameters or focus
- why (rationale): what information the action is expected to yield and how it resolves the uncertainty
- when not (boundary): when this action pattern should NOT be applied

Important:
- Do not invent action_type values.
- procedure.action_type must be copied exactly from action/tool names present in the episode input.
- If a useful reasoning pattern has no corresponding available action/tool, skip it — it belongs in experience memory, not skill memory.
- Always output tags as [].
- support_count is 1 unless otherwise stated.

Each skill should include:
1. trigger: clinical decision point or diagnostic uncertainty that activates this skill
2. procedure: 1-3 available actions with concrete action_label — specify the focus or parameters of the action, not just the tool name
3. rationale: what information the action yields and how it advances the diagnostic process
4. boundary: when this action pattern is inappropriate, redundant, or contradicted by evidence already gathered

Good skill:
"When initial history suggests infection but the source is unclear and localizing symptoms are absent, use ask_patient to probe for urinary symptoms (dysuria, frequency, urgency) and recent fever pattern before ordering labs, because UTI is a common occult infection source in elderly patients and clarifying symptoms first avoids unnecessary broad workup. Do not apply if urinary symptoms have already been explicitly ruled out or if a clear alternative source has been identified."

Bad skill:
"Ask about urinary symptoms."          ← no trigger context, no rationale, no boundary
"Order tests to find the infection."    ← too vague, no specific tool/action
"Consider UTI in elderly patients."     ← this is experience (reasoning), not skill (action)

Do not extract:
- clinical reasoning patterns or "what to consider" insights (→ experience memory)
- isolated medical facts or symptom-diagnosis associations
- "consider diagnosis X"
- generic advice such as "ask more questions" or "order tests"
- actions contradicted by the episode outcome
- skills without clear trigger, action logic, and boundary
- action_type values not present in the episode input

Generalization:
- Remove incidental case details.
- Keep decision context, tool focus, evidence threshold, and boundary only if clinically relevant.
- Prefer local reusable action patterns over broad disease rules.

Confidence:
- 0.9-1.0: visible, evidence-supported, outcome-confirmed, with clear trigger/action/boundary.
- 0.7-0.89: useful but partially inferred.
- Below 0.7: do not output.

Rules:
- Do not normalize or invent action_type.
- action_label must be concrete and clinically useful.
- procedure should specify the FOCUS of the action (what to ask, what to look up, what to check), not just the tool name.

Schema:
{_dump(SKILL_EXTRACTION_SCHEMA)}

Input:
{_dump(payload)}
""".strip()