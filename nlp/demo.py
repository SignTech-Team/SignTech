"""Python 3.10+: python demo.py"""
from signtech_nlp import SignTechNLP


def show(title, engine, words):
    engine.reset()
    for word in words:
        state = engine.add_confirmed(word)
    print(f"\n{title}")
    print("Ham işaretler :", state["raw_text"])
    print("Cümle önerisi:", state["suggested_sentence"] or "Kural yok (ham metin korunur)")
    print("Varsayım içeriyor mu?:", "Evet" if state["inference_used"] else "Hayır")
    if state["suggestion_notes"]:
        print("Açıklama      :", " | ".join(state["suggestion_notes"]))


safe = SignTechNLP(stable_windows=3, threshold=0.70)
show("1) Üç işaretli güvenli kural", safe, ["Hava", "Çok", "Güzel"])
show("2) Şablon kuralı", safe, ["Anne", "Hasta"])
show("3) İki cümle önerisi", safe, ["Hava", "Güzel", "Baba", "Hasta"])
show("4) Anlam uydurmayan varsayılan davranış", safe, ["Anne", "Ev", "Gitmek"])

contextual = SignTechNLP(allow_inference=True)
show("5) Açıkça izin verilen çekim önerisi (temkinli kullanılmalı)", contextual, ["Anne", "Ev", "Gitmek"])

print("\n6) Canlı tahmin taklidi / 3 kararlı pencere")
safe.reset()
for i, label in enumerate(["Hava", "Hava", "Hava", "Çok", "Çok", "Çok", "Güzel", "Güzel", "Güzel"], 1):
    state = safe.update(label, 0.9)
    print(f"{i:02d} {label:6} {state['status']:24} -> {state['raw_text']}")
print("SONUÇ:", safe.state()["suggested_sentence"])
