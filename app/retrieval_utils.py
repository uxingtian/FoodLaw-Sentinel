from __future__ import annotations

import math
import re
from functools import lru_cache

import jieba
import numpy as np


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")


@lru_cache(maxsize=8192)
def tokenize_cached(text: str) -> tuple[str, ...]:
    normalized = text.lower()
    tokens: list[str] = []
    for token in jieba.lcut(normalized):
        token = token.strip()
        if token and TOKEN_PATTERN.search(token):
            tokens.append(token)
    return tuple(tokens)


def tokenize(text: str) -> list[str]:
    return list(tokenize_cached(text))


def normalize_scores(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return np.zeros_like(values) if max_value <= 0 else np.ones_like(values)
    return (values - min_value) / (max_value - min_value)
