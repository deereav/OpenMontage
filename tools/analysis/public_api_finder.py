"""Search the public-apis catalog for free APIs to use as data sources.

Wraps the community list at https://github.com/public-apis/public-apis
(~1,400 APIs across ~50 categories). The README is downloaded once, cached
under ~/.openmontage/public_apis/, and refreshed after a week. Searching is a
local keyword match over name, category, and description, so the tool works
offline once the cache exists.

This finds candidate APIs only — it does not call them. Entries are
community-maintained and can be stale; verify an API's docs before relying on it.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)

README_URL = "https://raw.githubusercontent.com/public-apis/public-apis/master/README.md"
CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600

AUTH_VALUES = ["any", "none", "apiKey", "OAuth", "X-Mashape-Key", "User-Agent"]

_ROW_RE = re.compile(r"^\|\s*\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)\s*\|(?P<rest>.*)$")


def default_cache_dir() -> Path:
    """Honors OPENMONTAGE_PUBLIC_APIS_CACHE_DIR, else ~/.openmontage/public_apis."""
    override = os.environ.get("OPENMONTAGE_PUBLIC_APIS_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".openmontage" / "public_apis"


def _strip_tracking(url: str) -> str:
    """Drop utm_* referral parameters the list adds to sponsored links."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")]
    return urlunsplit(parts._replace(query=urlencode(query)))


def parse_readme(text: str) -> list[dict[str, str]]:
    """Parse the README's category tables into API entries.

    Rows look like ``| [Name](url) | Description | Auth | HTTPS | CORS |``
    under ``### Category`` headings. The sponsored block at the top uses a
    3-column table and is skipped, as is anything that is not a 5-column row.
    """
    entries: list[dict[str, str]] = []
    category: Optional[str] = None
    for line in text.splitlines():
        if line.startswith("### "):
            category = line[4:].strip()
            continue
        if category is None:
            continue
        match = _ROW_RE.match(line.strip())
        if not match:
            continue
        cells = [c.strip() for c in match.group("rest").split("|")]
        # Some rows end with an extra "|", leaving empty trailing cells.
        while cells and not cells[-1]:
            cells.pop()
        if len(cells) != 4:
            continue
        description, auth, https, cors = cells
        auth = auth.strip("`")
        entries.append(
            {
                "name": match.group("name").strip(),
                "url": _strip_tracking(match.group("url").strip()),
                "description": description,
                "category": category,
                "auth": "none" if auth.lower() == "no" else auth,
                "https": https,
                "cors": cors,
            }
        )
    return entries


class PublicApiFinder(BaseTool):
    name = "public_api_finder"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "api_discovery"
    provider = "public-apis"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["python:requests"]
    install_instructions = "pip install requests"
    agent_skills = []

    capabilities = ["search_apis", "list_api_categories"]

    best_for = [
        "finding a free API to source data for a data-driven video (weather, finance, sports, space, etc.)",
        "checking which APIs in a topic need no auth key",
    ]

    not_good_for = [
        "calling the APIs it finds (it only returns catalog entries)",
        "guaranteed-current information — the list is community-maintained and some entries are dead",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords matched against API name, category, and description (e.g. 'earthquake', 'stock prices')",
            },
            "category": {
                "type": "string",
                "description": "Case-insensitive substring of a category name, e.g. 'weather' or 'finance'",
            },
            "auth": {
                "type": "string",
                "enum": AUTH_VALUES,
                "default": "any",
                "description": "'none' returns only APIs that need no key",
            },
            "https_only": {"type": "boolean", "default": False},
            "cors_only": {
                "type": "boolean",
                "default": False,
                "description": "Only APIs that allow browser (CORS) calls — useful for HyperFrames/Remotion pages",
            },
            "limit": {"type": "integer", "default": 15, "minimum": 1, "maximum": 200},
            "list_categories": {
                "type": "boolean",
                "default": False,
                "description": "Return category names with API counts instead of searching",
            },
            "refresh": {
                "type": "boolean",
                "default": False,
                "description": "Re-download the catalog even if the cache is fresh",
            },
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "total_matches": {"type": "integer"},
            "categories": {"type": "array"},
            "catalog": {"type": "object"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=1, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["ConnectionError", "Timeout"])
    idempotency_key_fields = ["query", "category", "auth", "https_only", "cors_only", "limit"]
    side_effects = ["caches the public-apis README under ~/.openmontage/public_apis/"]
    user_visible_verification = [
        "Open the returned API's docs link and confirm it is still live before building on it",
    ]

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or default_cache_dir()

    def _load_catalog(self, refresh: bool) -> tuple[list[dict[str, str]], dict[str, Any]]:
        readme_path = self.cache_dir / "README.md"
        meta_path = self.cache_dir / "meta.json"

        age = None
        if readme_path.exists():
            age = time.time() - readme_path.stat().st_mtime

        fetch_error = None
        if refresh or age is None or age > CACHE_MAX_AGE_SECONDS:
            try:
                import requests

                resp = requests.get(README_URL, timeout=30)
                resp.raise_for_status()
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                readme_path.write_text(resp.text, encoding="utf-8")
                meta_path.write_text(
                    json.dumps({"source": README_URL, "fetched_at": time.time()}),
                    encoding="utf-8",
                )
                age = 0.0
            except Exception as exc:
                if not readme_path.exists():
                    raise
                # Offline or GitHub hiccup: fall back to the stale copy.
                fetch_error = f"{type(exc).__name__}: {exc}"

        entries = parse_readme(readme_path.read_text(encoding="utf-8"))
        catalog = {
            "source": README_URL,
            "cache_path": str(readme_path),
            "age_hours": round((age or 0.0) / 3600, 1),
            "stale": fetch_error is not None,
            "refresh_error": fetch_error,
            "total_apis": len(entries),
        }
        return entries, catalog

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        try:
            entries, catalog = self._load_catalog(bool(inputs.get("refresh", False)))
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Could not download the public-apis catalog and no cache exists: {exc}",
            )

        if not entries:
            return ToolResult(
                success=False,
                error="Parsed 0 APIs from the public-apis README; its format may have changed.",
            )

        if inputs.get("list_categories"):
            counts: dict[str, int] = {}
            for entry in entries:
                counts[entry["category"]] = counts.get(entry["category"], 0) + 1
            categories = [{"category": c, "count": n} for c, n in sorted(counts.items())]
            return ToolResult(
                success=True,
                data={"categories": categories, "catalog": catalog},
                duration_seconds=round(time.time() - start, 2),
            )

        category = (inputs.get("category") or "").lower()
        auth = inputs.get("auth", "any")
        https_only = bool(inputs.get("https_only", False))
        cors_only = bool(inputs.get("cors_only", False))
        tokens = [t for t in re.split(r"\W+", (inputs.get("query") or "").lower()) if t]
        # Crude singularization so "earthquakes" still hits "Earthquake".
        tokens = [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in tokens]
        # Match at word starts so "stock" hits "Stocks" but not "BirkinBagStock".
        patterns = [re.compile(r"\b" + re.escape(t)) for t in tokens]
        limit = int(inputs.get("limit", 15))

        scored = []
        for entry in entries:
            if category and category not in entry["category"].lower():
                continue
            if auth != "any" and entry["auth"].lower() != auth.lower():
                continue
            if https_only and entry["https"].lower() != "yes":
                continue
            if cors_only and entry["cors"].lower() != "yes":
                continue

            score = 0
            matched = 0
            name = entry["name"].lower()
            cat = entry["category"].lower()
            desc = entry["description"].lower()
            for pattern in patterns:
                hit = (
                    3 * bool(pattern.search(name))
                    + 2 * bool(pattern.search(cat))
                    + bool(pattern.search(desc))
                )
                score += hit
                matched += hit > 0
            if patterns and score == 0:
                continue
            scored.append((matched, score, entry))

        # Entries matching more of the query words rank first; field weights break ties.
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]["name"].lower()))
        results = [entry for _, _, entry in scored[:limit]]

        return ToolResult(
            success=True,
            data={"results": results, "total_matches": len(scored), "catalog": catalog},
            duration_seconds=round(time.time() - start, 2),
        )
