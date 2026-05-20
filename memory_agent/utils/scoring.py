from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# English alphanumeric runs OR individual CJK characters (each char = 1 token)
TOKEN_RE = re.compile(
    r"[a-z0-9_]+|[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]"
)


def tokenize(text: str) -> list[str]:
    """Tokenize text into English word stems and individual CJK characters.

    English: lowercase alphanumeric runs (e.g. "chest" → "chest").
    CJK: each Chinese character becomes its own token (e.g. "胸痛" → ["胸", "痛"]).
    Punctuation and whitespace are discarded automatically.
    """
    return TOKEN_RE.findall((text or "").lower())


def flatten_payload(payload: Any) -> str:
    if payload is None:
        return ""

    if isinstance(payload, dict):
        return " ".join(
            f"{key} {flatten_payload(value)}"
            for key, value in payload.items()
        )

    if isinstance(payload, list):
        return " ".join(flatten_payload(item) for item in payload)

    return str(payload)


def cosine_similarity(text_a: str, text_b: str) -> float:
    vec_a = Counter(tokenize(text_a))
    vec_b = Counter(tokenize(text_b))

    if not vec_a or not vec_b:
        return 0.0

    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[token] * vec_b[token] for token in common)

    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    query_tokens = tokenize(query)
    doc_tokens = [tokenize(doc) for doc in documents]
    if not query_tokens or not doc_tokens:
        return [0.0 for _ in documents]

    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        for token in set(tokens):
            doc_freq[token] += 1

    n_docs = len(doc_tokens)
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(doc_lengths) / n_docs if n_docs else 0.0
    if avgdl <= 0.0:
        return [0.0 for _ in documents]

    query_terms = set(query_tokens)
    scores: list[float] = []
    for tokens, doc_len in zip(doc_tokens, doc_lengths, strict=False):
        if not tokens:
            scores.append(0.0)
            continue
        tf = Counter(tokens)
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = freq + k1 * (1.0 - b + b * doc_len / avgdl)
            score += idf * (freq * (k1 + 1.0)) / denom
        scores.append(score)
    return scores


def bm25_similarity(query: str, document: str) -> float:
    score = bm25_scores(query, [document])[0]
    self_score = bm25_scores(query, [query])[0]
    if self_score <= 0.0:
        return 0.0
    return max(0.0, min(1.0, score / self_score))


def tag_overlap_score(left: list[Any], right: list[Any]) -> float:
    left_set = {str(item).strip().lower() for item in left if str(item).strip()}
    right_set = {str(item).strip().lower() for item in right if str(item).strip()}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)
