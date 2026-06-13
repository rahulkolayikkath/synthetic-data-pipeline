"""sentence_generation.prompting — prompt construction & output parsing.

Builds the per-cell instruction prompt and splits a completion into candidate
sentences, stripping list scaffolding.
"""
from __future__ import annotations

import re

SENTENCE_TYPE_GUIDE = {
    "declarative":   "plain statements of fact or description",
    "interrogative": "questions (each must end with a question mark)",
    "imperative":    "commands, requests, or instructions",
    "exclamatory":   "exclamations expressing strong emotion (each ends with '!')",
}


def build_prompt(lang_name: str, script_name: str, topic: str, stype: str, n: int) -> str:
    return f"""You are generating natural, everyday {lang_name} sentences for a speech dataset.

Write exactly {n} {stype} sentences ({SENTENCE_TYPE_GUIDE[stype]}) about: "{topic}".

Strict rules:
- Write ONLY in {lang_name}, using the {script_name} script.
- Each sentence must be at min 5 words and at most 20 words.
- Write every number, date, currency amount, unit, and abbreviation as WORDS in {lang_name}. Do NOT use any digits.
- Use natural, everyday phrasing. Make all sentences distinct from one another.
- Output one sentence per line. No numbering, no bullets, no quotation marks, no English, no extra commentary."""


_BULLET_RE = re.compile(r'^\s*[\-\*•]\s*')
_NUMBERING_RE = re.compile(r'^\s*\d+[\.\)]\s*')


def parse_output(text: str):
    """Split a completion into candidate sentences, stripping list scaffolding."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        s = _BULLET_RE.sub('', s)
        s = _NUMBERING_RE.sub('', s)
        s = s.strip().strip('"').strip('“”').strip()
        if s:
            out.append(s)
    return out
