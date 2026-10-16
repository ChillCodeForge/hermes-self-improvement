from __future__ import annotations

import re


_TASK_TOKENS = re.compile(r"[^\W_]+", re.UNICODE)
_MIN_NEAR_DUPLICATE_TOKENS = 8
_NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.85


def task_tokens(task: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TASK_TOKENS.findall(task))


def are_near_duplicate_tasks(left: str, right: str) -> bool:
    left_tokens = task_tokens(left)
    right_tokens = task_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens == right_tokens:
        return True
    if min(len(left_tokens), len(right_tokens)) < _MIN_NEAR_DUPLICATE_TOKENS:
        return False

    left_words = set(left_tokens)
    right_words = set(right_tokens)
    similarity = len(left_words & right_words) / len(left_words | right_words)
    return similarity >= _NEAR_DUPLICATE_JACCARD_THRESHOLD
