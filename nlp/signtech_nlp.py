"""SignTech: modelden gelen kelime tahminlerini kararlı metne dönüştürür.

Bu modül model eğitmez, kamerayı açmaz ve yeni işaret tahmini yapmaz.
Tahminleri filtreler; ham kelimeleri korur ve yalnızca bilinen kurallara
uyduğunda ayrı bir Türkçe cümle önerisi sunar.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sentence_rules import SentenceRuleEngine


class SignTechNLP:
    """Güven skoru + tekrarlayan kararlı tahminlerden ham metin oluşturur.

    Parametreler:
      stable_windows: Bir kelimeyi kabul etmek için kaç *tahmin penceresi* art arda aynı olmalı.
      threshold: 0-1 arası minimum confidence (örn. 0.70 = %70).
      release_windows: Yeni aynı kelimeyi kabul edebilmek için kaç boş/düşük güvenli
                       pencere görülmeli. Ayrı bir işaret de kilidi açar.

    NOT: stable_windows video *karesi* değil, model çıktısı sayısıdır.
    """

    def __init__(
        self,
        csv_path: str | Path | None = None,
        *,
        stable_windows: int = 3,
        threshold: float = 0.70,
        release_windows: int = 3,
        allow_inference: bool = False,
        rules_path: str | Path | None = None,
    ) -> None:
        if stable_windows < 1 or release_windows < 1:
            raise ValueError("Pencere sayıları en az 1 olmalı")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold 0 ile 1 arasında olmalı")
        path = Path(csv_path) if csv_path else Path(__file__).with_name("SignList_helpers.csv")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        self.id_to_word = {int(row["ClassId"]): row["SignId_Turkish"].strip() for row in records}
        self.word_lookup = {name.casefold(): name for name in self.id_to_word.values()}
        if len(self.id_to_word) != len(self.word_lookup):
            raise ValueError("Sözlükte mükerrer etiket bulundu")
        self.stable_windows = stable_windows
        self.threshold = threshold
        self.release_windows = release_windows
        self.allow_inference = allow_inference
        self.sentence_engine = SentenceRuleEngine(set(self.id_to_word.values()), rules_path)
        self.reset()

    def reset(self) -> None:
        self.words: list[str] = []
        self.candidate: str | None = None
        self.streak = 0
        self.locked_word: str | None = None
        self.empty_streak = 0

    def undo(self) -> dict[str, Any]:
        if self.words:
            self.words.pop()
        # Ekrandaki sonuç değişse de aynı sürekli işareti tekrar yazmamak için
        # locked_word korunur; yeni bir hareket veya boşluk gerekir.
        return self.state(status="last_word_removed")

    def _canonical_word(self, word_or_id: str | int | None) -> str | None:
        if word_or_id is None:
            return None
        if isinstance(word_or_id, bool):
            return None
        if isinstance(word_or_id, int):
            return self.id_to_word.get(word_or_id)
        if not isinstance(word_or_id, str):
            return None
        return self.word_lookup.get(word_or_id.strip().casefold())

    def add_confirmed(self, word_or_id: str | int) -> dict[str, Any]:
        """Halil'in modeline ihtiyaç duymadan NLP kısmını elle test etmek için."""
        word = self._canonical_word(word_or_id)
        if word is None:
            return self.state(status="unknown_word")
        self.words.append(word)
        return self.state(status="accepted", added_word=word)

    def update(
        self,
        word_or_id: str | int | None,
        confidence: float = 0.0,
        *,
        hands_visible: bool = True,
    ) -> dict[str, Any]:
        """Bir model tahmin penceresinin çıktısını işler (güven skoru 0..1)."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence 0 ile 1 arasında olmalı; %85 için 0.85 gir")
        word = self._canonical_word(word_or_id)
        if not hands_visible or word_or_id is None or confidence < self.threshold:
            self.candidate = None
            self.streak = 0
            self.empty_streak += 1
            if self.empty_streak >= self.release_windows:
                self.locked_word = None
            return self.state(status="no_confident_sign")
        if word is None:
            self.candidate = None
            self.streak = 0
            return self.state(status="unknown_word")

        self.empty_streak = 0
        if word == self.candidate:
            self.streak += 1
        else:
            self.candidate = word
            self.streak = 1

        if self.streak >= self.stable_windows and word != self.locked_word:
            self.words.append(word)
            self.locked_word = word
            return self.state(status="accepted", added_word=word)
        return self.state(status="waiting_for_stability")

    def state(self, *, status: str = "ready", added_word: str | None = None) -> dict[str, Any]:
        raw_text = " ".join(self.words)
        proposal = self.sentence_engine.propose(self.words, allow_inference=self.allow_inference)
        sentence = proposal["sentence"]
        # Öneri yoksa sadece ham metni noktalayarak gösteririz.
        readable = raw_text + ("." if raw_text else "")
        return {
            "status": status,
            "added_word": added_word,
            "raw_words": list(self.words),
            "raw_text": raw_text,
            "readable_text": readable,
            "suggested_sentence": sentence,
            "suggestion_is_rule_based": sentence is not None,
            "inference_used": proposal["inference_used"],
            "applied_rules": proposal["applied_rules"],
            "suggestion_notes": proposal["notes"],
            "candidate": self.candidate,
            "stable_count": self.streak,
        }
