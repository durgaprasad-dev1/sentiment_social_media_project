"""Shared text preprocessing for training and the Streamlit app."""

import re
from typing import Any

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: Any) -> str:
    """Lightly normalize a social-media post without removing sentiment cues.

    URLs are removed, usernames are normalized to @user, hashtags are retained
    as words, and repeated whitespace is collapsed. Negation words and
    punctuation are intentionally not stripped aggressively.
    """
    if text is None:
        return ""
    value = str(text).strip()
    value = _URL_RE.sub(" URL ", value)
    value = _MENTION_RE.sub(" @user ", value)
    value = value.replace("#", "")
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip()
