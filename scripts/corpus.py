"""Shared aligned records with stable IDs across all notebooks (Python 3.10+)."""

import html
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": "train", "validation": "tst2012", "test": "tst2013"}


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", html.unescape(text)).split())


def records(split):
    base = ROOT / "data" / "raw" / "iwslt15"
    stem = SPLITS[split]
    with (base / f"{stem}.en").open(encoding="utf8") as en, (base / f"{stem}.vi").open(encoding="utf8") as vi:
        for row, (source, target) in enumerate(zip(en, vi, strict=True), 1):
            source, target = normalize(source), normalize(target)
            if split == "train" and (not source or not target):
                continue
            yield {"id": f"{stem}-{row}", "en": source, "vi": target}
