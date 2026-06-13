"""sentence_generation.validation

The source of truth for sentence quality. 
Oder of the validation gates are as follows:
degeneracy -> number normalization -> script ratio -> language-ID -> dedup. 
Every rejection returns a reason string (which becomes the QC-yield table).

"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter

import numpy as np

SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),   # Hindi, Marathi
    "Malayalam":  (0x0D00, 0x0D7F),
    "Tamil":      (0x0B80, 0x0BFF),
    "Bengali":    (0x0980, 0x09FF),
    "Telugu":     (0x0C00, 0x0C7F),
    "Kannada":    (0x0C80, 0x0CFF),
    "Gujarati":   (0x0A80, 0x0AFF),
    "Gurmukhi":   (0x0A00, 0x0A7F),   # Punjabi
    "Odia":       (0x0B00, 0x0B7F),
}

_CURRENCY_RE = re.compile(r'[₹$€£¥%]')


def script_ratio(text: str, script_name: str) -> float:
    lo, hi = SCRIPT_RANGES[script_name]
    letters = [c for c in text if unicodedata.category(c).startswith('L')]
    if not letters:
        return 0.0
    return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)


def has_unnormalized_numbers(text: str) -> bool:
    # \d matches Unicode decimal digits, incl. Devanagari ०-९, Malayalam ൦-൯, etc.
    return bool(re.search(r'\d', text)) or bool(_CURRENCY_RE.search(text))


def is_degenerate(text: str, cfg):
    """Return (is_degenerate, reason). Pure Python; no model needed."""
    words = text.split()
    n = len(words)
    if n < cfg.min_words:
        return True, "too_short"
    if n > cfg.max_words:
        return True, "too_long"
    if len(set(words)) / n < cfg.min_type_token_ratio:
        return True, "low_diversity"
    for a, b in zip(words, words[1:]):
        if a == b:
            return True, "repeated_word"
    if n >= 6:
        tris = [tuple(words[i:i + 3]) for i in range(n - 2)]
        if max(Counter(tris).values()) > 1:
            return True, "repeated_ngram"
    return False, None


def norm_key(text: str) -> str:
    """Normalized form for exact-dup hashing: NFC, collapse whitespace, drop punctuation."""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r'\s+', ' ', t).strip()
    return ''.join(c for c in t if not unicodedata.category(c).startswith('P'))


class LanguageIdentifier:
    """Confirms the *language* (not just the script). fastText lid.176 returns ISO codes
    (hi, ml, ta, bn, mr, ...) that match our language keys."""

    def __init__(self, backend: str = "fasttext", model_dir: str = None):
        self.backend = backend
        if backend == "fasttext":
            import fasttext
            import urllib.request
            path = os.path.join(model_dir or os.getcwd(), "lid.176.bin")
            if not os.path.exists(path):
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", path)
            self.model = fasttext.load_model(path)
        elif backend == "indiclid":
            # Upgrade path: AI4Bharat IndicLID — better on Indic + romanized text and on
            # languages that share a script. https://github.com/AI4Bharat/IndicLID
            raise NotImplementedError("Plug in IndicLID here (see notebook final cell).")
        else:
            raise ValueError(f"unknown langid backend: {backend}")

    def predict(self, text: str):
        labels, probs = self.model.predict(text.replace("\n", " "), k=1)
        return labels[0].replace("__label__", ""), float(probs[0])


class Deduper:
    """Per-language exact + semantic dedup. `is_dup` only checks; `add` commits an accepted
    sentence. Embeddings are L2-normalized so cosine == dot product."""

    def __init__(self, embed_model, threshold: float):
        self.embed = embed_model
        self.threshold = threshold
        self.exact = {}    # lang -> set of normalized keys
        self.embs = {}     # lang -> np.ndarray (N, D), normalized

    def _emb(self, text: str) -> np.ndarray:
        v = self.embed.encode([text], normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)            # (1, D)

    def is_dup(self, text: str, lang: str):
        if norm_key(text) in self.exact.get(lang, set()):
            return True, "exact_duplicate"
        M = self.embs.get(lang)
        if M is not None and len(M):
            if float((M @ self._emb(text)[0]).max()) >= self.threshold:
                return True, "near_duplicate"
        return False, None

    def add(self, text: str, lang: str):
        self.exact.setdefault(lang, set()).add(norm_key(text))
        v = self._emb(text)
        self.embs[lang] = v if self.embs.get(lang) is None else np.vstack([self.embs[lang], v])


def validate_sentence(text, lang_code, script_name, deduper, langid, cfg):
    """Return (ok, reason)"""
    if not text or len(text) < 2:
        return False, "empty"

    deg, why = is_degenerate(text, cfg)
    if deg:
        return False, why

    if cfg.reject_unnormalized_numbers and has_unnormalized_numbers(text):
        return False, "unnormalized_number"

    if script_ratio(text, script_name) < cfg.script_min_ratio:
        return False, "wrong_script"

    lab, conf = langid.predict(text)
    if lab != lang_code or conf < cfg.langid_min_conf:
        return False, f"langid_mismatch:{lab}"

    dup, why = deduper.is_dup(text, lang_code)
    if dup:
        return False, why

    return True, "accepted"
