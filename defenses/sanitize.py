"""Defense layer: filter retrieved chunks before they enter the LLM prompt.

Two checks:
1. Instruction-pattern filter — regex scan for imperative/system-like language.
2. Source allowlist — only chunks from trusted_sources.json are allowed.

Usage:
    from sanitize import filter_chunks
    safe = filter_chunks(retrieved_chunks)
"""

import json
import re
from pathlib import Path

TRUSTED_SOURCES_PATH = Path(__file__).parent / "trusted_sources.json"

INSTRUCTION_PATTERNS = [
    r"\[SYSTEM\s*NOTE:",
    r"\[INST:",
    r"\[ADMIN:",
    r"\[INSTRUCTION:",
    r"Always recommend",
    r"Do not mention",
    r"Regardless of (the |your )?question",
    r"IMPORTANT: You must",
    r"You must always",
    r"Never mention",
    r"Regardless of what (the |a )?user",
    r"Treat this as (a |the )?system",
]

_instruction_re = re.compile(
    "|".join(INSTRUCTION_PATTERNS), re.IGNORECASE
)


def _load_trusted_sources() -> set[str]:
    """Load the allowlist of trusted source filenames."""
    with open(TRUSTED_SOURCES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("trusted_sources", []))


def filter_chunks(chunks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Filter retrieved chunks through instruction-pattern + source-allowlist checks.

    Args:
        chunks: list of (source, text) tuples from retrieve_context().

    Returns:
        Filtered list of (source, text) tuples. Chunks that fail either check
        are dropped.
    """
    trusted = _load_trusted_sources()
    filtered = []
    dropped = []

    for source, text in chunks:
        # Check 1: source allowlist
        if source not in trusted:
            dropped.append((source, "source not in allowlist"))
            continue

        # Check 2: instruction-pattern scan
        if _instruction_re.search(text):
            dropped.append((source, "instruction pattern detected"))
            continue

        filtered.append((source, text))

    if dropped:
        print("\n[Defense] Filtered chunks:")
        for source, reason in dropped:
            preview = next(
                (t[:60] for s, t in chunks if s == source), ""
            )
            print(f"  DROPPED: {source} — {reason}")
            print(f"    Preview: {preview!r}...")
        print()

    return filtered


def is_chunk_safe(source: str, text: str) -> tuple[bool, str]:
    """Check a single chunk. Returns (is_safe, reason_if_not)."""
    trusted = _load_trusted_sources()

    if source not in trusted:
        return False, "source not in allowlist"

    if _instruction_re.search(text):
        return False, "instruction pattern detected"

    return True, ""
