from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from ..llm import EmbeddingClient
from ..memory_store import ExperienceMemoryStore, KnowledgeMemoryStore, SkillMemoryStore
from ..schemas import MemoryQuery, MemoryRetrievalResult, OutcomeType, RetrievalHit
from ..utils.config import MEMORY_ROOT_DIRNAME, RETRIEVAL_CONFIG
from ..utils.scoring import (
    bm25_similarity,
    cosine_similarity as token_cosine,
    flatten_payload,
    tag_overlap_score,
    tokenize,
)

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_ROOT = Path(MEMORY_ROOT_DIRNAME)

# ------------------------------------------------------------------
# Clinical topic filter: skip memories whose topic is clearly unrelated
# ------------------------------------------------------------------

# Organ system / specialty clusters (lowercase tokens)
_TOPIC_CLUSTERS: list[tuple[set[str], str]] = [
    ({"cardiac", "heart", "aortic", "coronary", "myocardial", "valve", "arrhythmia",
      "atrial", "ventricular", "tachycardia", "fibrillation", "hypertension",
      "hypertensive", "chest pain", "ecg", "echocardiogram"}, "cardiovascular"),
    ({"pulmonary", "lung", "pneumonia", "asthma", "copd", "respiratory", "bronch",
      "dyspnea", "wheez", "cough", "pleural", "pneumothorax", "tb", "tuberculosis"}, "respiratory"),
    ({"neurologic", "brain", "stroke", "seizure", "epilepsy", "headache", "migraine",
      "brain", "cerebr", "hemorrhage", "aneurysm", "meningitis", "dementia",
      "alzheimer", "parkinson", "numbness", "weakness", "paralysis", "nerve",
      "neuropathy", "sciatica", "radiculopathy"}, "neurological"),
    ({"gastrointestin", "abdom", "stomach", "liver", "hepat", "pancrea",
      "bowel", "colon", "rectal", "gastric", "duoden", "intestin", "diverticul",
      "appendic", "cholecyst", "biliary", "nausea", "vomit", "diarrhea",
      "melena", "hematemesis"}, "gastrointestinal"),
    ({"renal", "kidney", "nephro", "dialysis", "urinary", "bladder", "ureter",
      "prostate", "ut", "cystitis", "pyelonephritis", "hematuria", "proteinuria"}, "urology"),
    ({"orthoped", "fracture", "bone", "joint", "spine", "spinal", "vertebr",
      "femur", "hip", "knee", "ankle", "shoulder", "osteoporosis", "arthritis",
      "osteoarthritis", "spondylo", "disc", "ligament", "tendon", "musculoskeletal"}, "orthopedic"),
    ({"psychiatr", "depress", "anxiety", "schizo", "bipolar", "mental",
      "psychosis", "suicid", "mood", "panic", "ptsd", "insomnia"}, "psychiatric"),
    ({"obstetr", "pregnan", "gestation", "maternal", "fetal", "placenta",
      "labor", "delivery", "cesarean", "preeclamp", "ectopic", "membrane",
      "amniotic"}, "obstetric"),
    ({"gynecol", "uterus", "ovarian", "cervical", "vagin", "endometri",
      "pelvic mass", "fibroid", "menstrual", "pcos"}, "gynecologic"),
    ({"dermatol", "skin", "rash", "cellulitis", "abscess", "wound", "ulcer",
      "urticaria", "eczema", "psoriasis", "melanoma"}, "dermatologic"),
    ({"hematol", "anemia", "leukemia", "lymphoma", "thrombocytopenia",
      "coagul", "bleeding", "bruising", "platelet", "itp", "neutropenia"}, "hematologic"),
    ({"endocrin", "diabetes", "thyroid", "adrenal", "pituitary", "hormone",
      "hyperglycemia", "hypoglycemia", "insulin", "cortisol"}, "endocrine"),
    ({"infect", "sepsis", "bacteri", "viral", "fungal", "hiv", "abscess",
      "cellulitis", "meningitis", "pneumonia"}, "infectious"),
    ({"oncolog", "cancer", "tumor", "malignan", "neoplasm", "carcinoma",
      "adenocarcinoma", "metasta", "chemotherapy", "chemo"}, "oncologic"),
]

# Tags that indicate a generic "broad differential" topic — always match
_GENERIC_TAGS = {"broad_differential", "premature_closure", "targeted_history",
                "low_value_interaction", "diagnostic_anchoring", "missing_lab",
                "missing_history", "missing_imaging", "unsafe_finalization",
                "positive", "negative"}


def _infer_topics(text: str, tags: list[str] | None = None) -> set[str]:
    """Infer clinical topic cluster(s) from text."""
    combined = (text or "").lower()
    topics: set[str] = set()
    for keywords, label in _TOPIC_CLUSTERS:
        for kw in keywords:
            if kw in combined:
                topics.add(label)
                break
    return topics


def _topics_overlap(q_topics: set[str], m_topics: set[str], tags: list[str] | None = None) -> bool:
    """Return True if query and memory share at least one topic."""
    if not q_topics or not m_topics:
        return True  # can't determine topic → don't filter
    return bool(q_topics & m_topics)


def _root(root_dir: str | None) -> Path:
    root = Path(root_dir) if root_dir else DEFAULT_MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _join(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values if value)


def experience_to_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(payload.get("text") or ""),
            f"Outcome type: {payload.get('outcome_type', '')}",
            f"Tags: {_join(payload.get('tags') or [])}",
        ]
    )


def skill_to_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Skill: {payload.get('skill_text', '')}",
            f"Procedure steps: {flatten_payload(payload.get('procedure', []))}",
            f"Tags: {_join(payload.get('tags') or [])}",
        ]
    )


def knowledge_to_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(payload.get("content", "")),
            f"Tags: {_join(payload.get('tags', []))}",
            f"Source: {payload.get('source', '')}",
        ]
    )


def memory_to_text(memory_type: str, payload: dict[str, Any]) -> str:
    if memory_type == "experience":
        return experience_to_text(payload)
    if memory_type == "skill":
        return skill_to_text(payload)
    if memory_type == "knowledge":
        return knowledge_to_text(payload)
    return flatten_payload(payload)


def _fallback_scoring_mode() -> str:
    return str(
        os.environ.get("MEDGYM_RETRIEVAL_FALLBACK_SCORING")
        or RETRIEVAL_CONFIG.get("fallback_scoring")
        or "cosine"
    ).strip().lower()


def _embedding_cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two dense embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(v * v for v in vec_a))
    norm_b = math.sqrt(sum(v * v for v in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embedding_cache_enabled() -> bool:
    value = str(os.environ.get("MEDGYM_DISABLE_EMBEDDING_CACHE", "")).strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _embedding_cache_path(root_dir: str | None) -> Path:
    root = _root(root_dir)
    return root / ".embedding_cache" / "embeddings.jsonl"


def _embedding_cache_namespace(embedding_client: EmbeddingClient) -> str:
    payload = {
        "model": embedding_client.model,
        "base_url": embedding_client.base_url,
        "dimensions": embedding_client.dimensions,
        "input_type": "passage",
        "version": 1,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_cache_key(
    namespace: str,
    memory_type: str,
    memory_id: str,
    text_hash: str,
) -> str:
    raw = "\0".join([namespace, memory_type, memory_id, text_hash])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_embedding_cache(path: Path, namespace: str) -> dict[str, list[float]]:
    if not path.exists():
        return {}

    cache: dict[str, list[float]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipped invalid embedding cache JSON in %s at line %d",
                        path,
                        line_number,
                    )
                    continue

                if row.get("namespace") != namespace:
                    continue
                key = str(row.get("key") or "")
                vector = row.get("embedding")
                if key and isinstance(vector, list):
                    cache[key] = vector
    except OSError as exc:
        logger.warning("Could not read embedding cache %s: %s", path, exc)
        return {}

    return cache


def _append_embedding_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Could not write embedding cache %s: %s", path, exc)


def _precompute_embeddings(
    items: list[Any],
    memory_type: str,
    embedding_client: EmbeddingClient | None,
    root_dir: str | None = None,
) -> dict[str, list[float]]:
    """Batch-embed all items. Returns {memory_id: vector}."""
    if embedding_client is None or not embedding_client.available():
        return {}

    texts: list[str] = []
    id_order: list[str] = []
    for item in items:
        payload = item.to_dict() if hasattr(item, "to_dict") else item
        mid = str(payload.get("memory_id") or "")
        text = memory_to_text(memory_type, payload)
        texts.append(text)
        id_order.append(mid)

    if not texts:
        return {}

    cache_enabled = _embedding_cache_enabled()
    namespace = _embedding_cache_namespace(embedding_client)
    cache_path = _embedding_cache_path(root_dir)
    cache = _load_embedding_cache(cache_path, namespace) if cache_enabled else {}

    result: dict[str, list[float]] = {}
    miss_texts: list[str] = []
    miss_ids: list[str] = []
    miss_keys: list[str] = []
    miss_hashes: list[str] = []

    for mid, text in zip(id_order, texts, strict=False):
        text_hash = _sha256_text(text)
        key = _embedding_cache_key(namespace, memory_type, mid, text_hash)
        cached_vector = cache.get(key)
        if cached_vector is not None:
            result[mid] = cached_vector
            continue
        miss_texts.append(text)
        miss_ids.append(mid)
        miss_keys.append(key)
        miss_hashes.append(text_hash)

    if not miss_texts:
        logger.debug(
            "Embedding cache hit for all %d %s memory items",
            len(result),
            memory_type,
        )
        return result

    vectors = embedding_client.embed_documents(miss_texts)
    if vectors is None or len(vectors) != len(miss_ids):
        logger.warning(
            "Embedding batch returned %d vectors for %d items; falling back",
            len(vectors) if vectors else 0, len(miss_ids),
        )
        return result

    cache_rows: list[dict[str, Any]] = []
    for mid, key, text_hash, vector in zip(
        miss_ids,
        miss_keys,
        miss_hashes,
        vectors,
        strict=False,
    ):
        result[mid] = vector
        cache_rows.append(
            {
                "namespace": namespace,
                "key": key,
                "memory_type": memory_type,
                "memory_id": mid,
                "text_hash": text_hash,
                "model": embedding_client.model,
                "base_url": embedding_client.base_url,
                "embedding": vector,
            }
        )

    if cache_enabled:
        _append_embedding_cache(cache_path, cache_rows)
        logger.debug(
            "Embedding cache %s: %d hits, %d misses for %s memory",
            cache_path,
            len(result) - len(miss_ids),
            len(miss_ids),
            memory_type,
        )

    return result


def _score_memory(
    query: MemoryQuery,
    memory_type: str,
    payload: dict[str, Any],
    query_embedding: list[float] | None = None,
    memory_embedding: list[float] | None = None,
) -> float:
    """
    Score a single memory against the query.

    Priority:
    1. If both embeddings available → embedding cosine similarity.
    2. Else use configured fallback scoring:
       - cosine: previous token-cosine over full memory text.
       - fielded_bm25: field-weighted BM25 for experience/skill.
    """
    if query_embedding is not None and memory_embedding is not None:
        score = _embedding_cosine(query_embedding, memory_embedding)
    elif _fallback_scoring_mode() != "fielded_bm25":
        text = memory_to_text(memory_type, payload)
        score = token_cosine(query.query_text, text)
    elif memory_type == "experience":
        tags = payload.get("tags") or []
        text = experience_to_text(payload)
        score = (
            0.85 * bm25_similarity(query.query_text, text)
            + 0.10 * bm25_similarity(query.query_text, str(payload.get("outcome_type") or ""))
            + 0.05 * tag_overlap_score(tokenize(query.query_text), tags if isinstance(tags, list) else [])
        )
    elif memory_type == "skill":
        tags = payload.get("tags") or []
        score = (
            0.65 * bm25_similarity(query.query_text, str(payload.get("skill_text") or ""))
            + 0.25 * bm25_similarity(query.query_text, flatten_payload(payload.get("procedure") or []))
            + 0.10 * tag_overlap_score(tokenize(query.query_text), tags if isinstance(tags, list) else [])
        )
    else:
        text = memory_to_text(memory_type, payload)
        score = token_cosine(query.query_text, text)
    return max(0.0, min(1.0, score))


def _build_hit(
    memory_id: str,
    memory_type: str,
    content: dict[str, Any],
    score: float,
) -> RetrievalHit:
    return RetrievalHit(
        memory_id=memory_id,
        memory_type=memory_type,
        content=content,
        score=round(score, 4),
    )


def _threshold(name: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    return float(RETRIEVAL_CONFIG.get(name, 0.0) or 0.0)


def _experience_hits(
    query: MemoryQuery,
    root_dir: str | None,
    positive_min_score: float,
    negative_min_score: float,
    embedding_vectors: dict[str, list[float]] | None = None,
    query_embedding: list[float] | None = None,
) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    store = ExperienceMemoryStore(_root(root_dir))
    all_cards = store.list_all()

    vectors = embedding_vectors or {}
    pos_top_k = int(RETRIEVAL_CONFIG.get("positive_experience_top_k", 5))
    neg_top_k = int(RETRIEVAL_CONFIG.get("negative_experience_top_k", 3))

    positive: list[RetrievalHit] = []
    negative: list[RetrievalHit] = []

    # Pre-compute query topics once
    q_topics = _infer_topics(query.query_text or "")

    for card in all_cards:
        payload = card.to_dict()
        mid = card.memory_id
        mem_vec = vectors.get(mid)

        # Clinical topic filter: skip memories clearly unrelated to the query
        mem_tags = payload.get("tags") or []
        mem_text = experience_to_text(payload)
        m_topics = _infer_topics(mem_text, mem_tags)
        if not _topics_overlap(q_topics, m_topics, mem_tags):
            continue

        score = _score_memory(
            query, "experience", payload,
            query_embedding=query_embedding,
            memory_embedding=mem_vec,
        )
        tags = {str(tag).lower() for tag in (payload.get("tags") or [])}
        if "negative" in tags:
            is_negative = True
        elif "positive" in tags:
            is_negative = False
        else:
            is_negative = card.outcome_type == OutcomeType.NEGATIVE.value

        if is_negative and score < negative_min_score:
            continue
        if not is_negative and score < positive_min_score:
            continue

        hit = _build_hit(mid, "experience", payload, score)
        if is_negative:
            negative.append(hit)
        else:
            positive.append(hit)

    positive.sort(key=lambda h: h.score, reverse=True)
    negative.sort(key=lambda h: h.score, reverse=True)
    return positive[:pos_top_k], negative[:neg_top_k]


def _skill_hits(
    query: MemoryQuery,
    root_dir: str | None,
    min_score: float,
    embedding_vectors: dict[str, list[float]] | None = None,
    query_embedding: list[float] | None = None,
) -> list[RetrievalHit]:
    store = SkillMemoryStore(_root(root_dir))
    all_cards = store.list_all()
    vectors = embedding_vectors or {}
    top_k = int(RETRIEVAL_CONFIG.get("skill_top_k", 3))
    hits: list[RetrievalHit] = []

    for card in all_cards:
        payload = card.to_dict()
        mid = card.memory_id
        mem_vec = vectors.get(mid)
        score = _score_memory(
            query, "skill", payload,
            query_embedding=query_embedding,
            memory_embedding=mem_vec,
        )
        if score >= min_score:
            hits.append(_build_hit(mid, "skill", payload, score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def _knowledge_hits(
    query: MemoryQuery,
    root_dir: str | None,
    min_score: float,
    embedding_vectors: dict[str, list[float]] | None = None,
    query_embedding: list[float] | None = None,
) -> list[RetrievalHit]:
    store = KnowledgeMemoryStore(_root(root_dir))
    all_items = store.list_all()
    vectors = embedding_vectors or {}
    top_k = int(RETRIEVAL_CONFIG.get("knowledge_top_k", 3))
    hits: list[RetrievalHit] = []

    for item in all_items:
        payload = item.to_dict()
        mid = item.memory_id
        mem_vec = vectors.get(mid)
        score = _score_memory(
            query, "knowledge", payload,
            query_embedding=query_embedding,
            memory_embedding=mem_vec,
        )
        if score >= min_score:
            hits.append(_build_hit(mid, "knowledge", payload, score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def retrieve_multi_memory(
    memory_query: MemoryQuery,
    root_dir: str | None = None,
    disable_experience_memory: bool = False,
    disable_skill_memory: bool = False,
    disable_knowledge_memory: bool = False,
    positive_experience_min_score: float | None = None,
    negative_experience_min_score: float | None = None,
    skill_min_score: float | None = None,
    knowledge_min_score: float | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> MemoryRetrievalResult:
    """
    Retrieve relevant memories for a query.

    When ``embedding_client`` is provided and available, all stored items are
    batch-embedded once and scored via embedding cosine similarity. Otherwise
    falls back to token-based Bag-of-Words cosine similarity.
    """
    # ── Pre-compute embeddings for all stores ──────────────────────────
    query_embedding: list[float] | None = None
    skill_query_embedding: list[float] | None = None
    exp_vectors: dict[str, list[float]] = {}
    skill_vectors: dict[str, list[float]] = {}
    kn_vectors: dict[str, list[float]] = {}

    if embedding_client is not None and embedding_client.available():
        # Embed experience query
        query_embedding = embedding_client.embed_query(memory_query.query_text)
        # Embed skill query (separate)
        skill_query_text = memory_query.skill_query_text or memory_query.query_text
        if skill_query_text and skill_query_text != memory_query.query_text:
            skill_query_embedding = embedding_client.embed_query(skill_query_text)
        else:
            skill_query_embedding = query_embedding

        if query_embedding and not disable_experience_memory:
            store = ExperienceMemoryStore(_root(root_dir))
            exp_vectors = _precompute_embeddings(
                store.list_all(), "experience", embedding_client, root_dir=root_dir
            ) or {}

        if (query_embedding or skill_query_embedding) and not disable_skill_memory:
            store = SkillMemoryStore(_root(root_dir))
            skill_vectors = _precompute_embeddings(
                store.list_all(), "skill", embedding_client, root_dir=root_dir
            ) or {}

        if query_embedding and not disable_knowledge_memory:
            store = KnowledgeMemoryStore(_root(root_dir))
            kn_vectors = _precompute_embeddings(
                store.list_all(), "knowledge", embedding_client, root_dir=root_dir
            ) or {}

        if not query_embedding and not skill_query_embedding:
            logger.warning("Query embedding failed; falling back to token scoring")
            query_embedding = None

    # ── Retrieve ───────────────────────────────────────────────────────
    positive: list[RetrievalHit] = []
    negative: list[RetrievalHit] = []

    if not disable_experience_memory:
        positive, negative = _experience_hits(
            memory_query, root_dir,
            _threshold("positive_experience_min_score", positive_experience_min_score),
            _threshold("negative_experience_min_score", negative_experience_min_score),
            embedding_vectors=exp_vectors or None,
            query_embedding=query_embedding,
        )

    # Build a skill-specific query using skill_query_text
    skill_query = memory_query
    if memory_query.skill_query_text and memory_query.skill_query_text != memory_query.query_text:
        skill_query = MemoryQuery(
            case_id=memory_query.case_id,
            turn_id=memory_query.turn_id,
            query_text=memory_query.skill_query_text,
            skill_query_text=memory_query.skill_query_text,
        )

    return MemoryRetrievalResult(
        positive_experience_hits=positive,
        negative_experience_hits=negative,
        skill_hits=[]
        if disable_skill_memory
        else _skill_hits(
            skill_query, root_dir,
            _threshold("skill_min_score", skill_min_score),
            embedding_vectors=skill_vectors or None,
            query_embedding=skill_query_embedding or query_embedding,
        ),
        knowledge_hits=[]
        if disable_knowledge_memory
        else _knowledge_hits(
            memory_query, root_dir,
            _threshold("knowledge_min_score", knowledge_min_score),
            embedding_vectors=kn_vectors or None,
            query_embedding=query_embedding,
        ),
    )
