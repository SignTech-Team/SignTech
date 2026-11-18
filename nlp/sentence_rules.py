"""Yalnızca açıkça tanımlanmış işaret dizilerine şeffaf Türkçe öneri üretir.

Bu bir üretken dil modeli değildir. İşaretlenmeyen kelimeleri varsaymaz;
çekim eklerini/zaman bilgisini çıkaran kurallar varsayılan olarak kapalıdır.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


_SLOT = re.compile(r"^\{([a-z_]+)\}$")


@dataclass(frozen=True)
class Match:
    rule_id: str
    text: str
    inference: bool
    note: str
    count: int


class SentenceRuleEngine:
    def __init__(self, vocab: set[str], path: str | Path | None = None):
        rules_path = Path(path) if path else Path(__file__).with_name("sentence_rules.json")
        self.rules: list[dict[str, Any]] = json.loads(rules_path.read_text(encoding="utf-8"))
        for rule in self.rules:
            if not rule.get("pattern") or not rule.get("template"):
                raise ValueError(f"Geçersiz cümle kuralı: {rule}")
            for item in rule["pattern"]:
                match = _SLOT.fullmatch(item)
                if match:
                    options = rule.get("slots", {}).get(match.group(1), [])
                    if not options or any(word not in vocab for word in options):
                        raise ValueError(f"Kuraldaki slot sözlükte yok: {rule['id']}")
                elif item not in vocab:
                    raise ValueError(f"Kuraldaki işaret sözlükte yok: {item}")
        self.max_rule_length = max(len(rule["pattern"]) for rule in self.rules)

    @staticmethod
    def _match(rule: dict[str, Any], words: tuple[str, ...]) -> Match | None:
        if len(rule["pattern"]) != len(words):
            return None
        values = {}
        for wanted, actual in zip(rule["pattern"], words):
            slot = _SLOT.fullmatch(wanted)
            if slot:
                key = slot.group(1)
                if actual not in rule.get("slots", {}).get(key, []):
                    return None
                if key in values and values[key] != actual:
                    return None
                values[key] = actual
            elif wanted != actual:
                return None
        return Match(
            rule_id=rule["id"],
            text=rule["template"].format_map(values),
            inference=bool(rule.get("inference", False)),
            note=rule.get("note", ""),
            count=len(words),
        )

    def propose(self, words: list[str], *, allow_inference: bool = False) -> dict[str, Any]:
        """Tüm diziyi kapsayan kurallar varsa metin üretir; yoksa None döndürür.

        Birden çok cümle için dinamik programlama ile az varsayım yapan ve
        daha az parçaya bölünen yolu seçer. Ham işaret dizisi değişmez.
        """
        tokens = tuple(words)
        if not tokens:
            return {"sentence": None, "applied_rules": [], "inference_used": False, "notes": []}

        @lru_cache(None)
        def solve(start: int):
            if start == len(tokens):
                return ()
            candidates = []
            for rule in self.rules:
                if rule.get("inference", False) and not allow_inference:
                    continue
                length = len(rule["pattern"])
                if start + length > len(tokens):
                    continue
                hit = self._match(rule, tokens[start:start + length])
                if hit is None:
                    continue
                rest = solve(start + length)
                if rest is not None:
                    candidates.append((hit,) + rest)
            if not candidates:
                return None
            return min(candidates, key=lambda parts: (sum(p.inference for p in parts), len(parts)))

        matches = solve(0)
        if matches is None:
            return {"sentence": None, "applied_rules": [], "inference_used": False, "notes": []}
        return {
            "sentence": " ".join(m.text for m in matches),
            "applied_rules": [m.rule_id for m in matches],
            "inference_used": any(m.inference for m in matches),
            "notes": list(dict.fromkeys(m.note for m in matches if m.note)),
        }
