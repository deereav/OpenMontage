"""Place-name filter for stock clips.

Stock providers (Wikimedia, archive.org, ...) put the shooting location in
the title/description/categories that end up in ``source_tags``. A semantic
query like "red rock canyon at sunset" happily returns Wadi Rum or Sedona,
so place-specific edits need a textual check on top of CLIP ranking.

Spec shape (``filters.location`` in corpus_builder, ``location`` in
clip_search)::

    {
        "include_terms": ["utah", "zion", "moab"],   # the place we want
        "exclude_terms": ["jordan", "arizona"],       # look-alike places
        "require_include": false,                     # strict mode
    }

Rules:
- A clip whose text names an exclude term is rejected, unless it also names
  an include term ("the border of Utah and Arizona" stays).
- With ``require_include`` the clip must name an include term; clips with
  no descriptive text are rejected too, since nothing vouches for them.
"""

from __future__ import annotations

import re
from typing import Any


def _compile(terms: Any) -> re.Pattern[str] | None:
    words = [str(t).strip() for t in (terms or []) if str(t).strip()]
    if not words:
        return None
    alternation = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def check_location(text: str | None, spec: dict[str, Any] | None) -> tuple[bool, str]:
    """Return ``(keep, reason)`` for one clip's descriptive text."""
    if not spec:
        return True, ""
    include = _compile(spec.get("include_terms"))
    exclude = _compile(spec.get("exclude_terms"))
    text = text or ""

    inc_hit = include.search(text) if include else None
    if inc_hit:
        return True, f"names {inc_hit.group(0)!r}"
    if spec.get("require_include") and include:
        return False, "names none of the include_terms"
    exc_hit = exclude.search(text) if exclude else None
    if exc_hit:
        return False, f"names excluded place {exc_hit.group(0)!r}"
    return True, ""
