# SignTech NLP V2 — Kural tabanlı cümle önerisi

Halil'in modelinin çıktılarından gelen sınıf ID'si veya Türkçe etiketleri işler. Kerem'in backend'inden **bağımsız** çalışır; gerçek zamanlı model bağlantısı bu pakette yoktur. **Bu bir üretken dil modeli değildir.**

## Yenilikler (V1 → V2)

- `sentence_rules.json` içinde genişletilebilir iki ve üç işaretli şablonlar.
- Belirli kişi özneleri için `{person}` kalıpları: `Anne Hasta` → `Anne hasta.` gibi.
- İki ayrı kalıbın birleşiminde `Hava Güzel Baba Hasta` → `Hava güzel. Baba hasta.`
- Anlam/zaman değişimi içeren çekim önerileri **varsayılan olarak kapalı**; `allow_inference=True` ile denenebilir ve `inference_used` / `suggestion_notes` bilgileriyle açıkça işaretlenir.
- V1'deki `update`, `add_confirmed`, `undo`, `reset`, `state` arayüzleri korunmuştur.
- `raw_text` ham işaretleri korur. Bir diziyi tamamen açıklayan kural bulunamazsa `suggested_sentence` = `None`.
- `confidence` 0 ile 1 arasında; `stable_windows` modelin çıktı penceresi sayısıdır, video karesi değil.

## Windows / VS Code'da çalıştırma

Daha önceki gibi `SignTechhhh` klasörüne çıkar. Terminal:

```powershell
cd "C:\Users\admin\Desktop\SignTechhhh"
.\.venv\Scripts\python.exe .\SignTech_NLP_V2\demo.py
.\.venv\Scripts\python.exe -m unittest discover -s .\SignTech_NLP_V2 -p "test_nlp.py" -v
```

Hiçbir harici pip paketi gerektirmez, Python 3.10+ yeterlidir.

## Entegrasyon sözleşmesi (Kerem için)

```python
from signtech_nlp import SignTechNLP

# Her kamera oturumu için ayrı nesne yaratılmalı; global tek nesne YAPMAYIN.
engine = SignTechNLP(stable_windows=3, threshold=0.70)
result = engine.update(175, 0.90, hands_visible=True)  # Hava ID'si

# result alanları: raw_text, suggested_sentence, applied_rules,
# inference_used, suggestion_notes, status, added_word, ...
```

`update` çağrısından önce Halil'in modelinin çıktısı doğru biçimde çözülmüş olmalı. Modelin `argmax` indeksini doğrudan sınıf ID'si saymayın; `label_classes.npy` sırası ile eşleyin. Her kullanıcı/oturum için ayrı `SignTechNLP` nesnesi tutulmalıdır.

**İşaret dili ve Türkçe farklı dilbilgisine sahiptir.** Bu nedenle eklenen kurallar yalnızca örnek şablonlardır; Türk İşaret Dili bilen kişiler tarafından doğrulanmadan üretim çevirisi olarak sunulmamalıdır. Gerçek işaret/kelime dizileriyle değerlendirme şarttır.

## GitHub'a ekipçe aktarma (önerilen)

1. Ortak **SignTech** reposunu klonlayın veya mevcut klonu açın. `SignTechhhh` otomatik olarak o repo olmayabilir.
2. Bu klasörün yalnızca kaynak dosyalarını (Python, JSON, CSV, README, test) ortak repo altında `nlp/` klasörüne kopyalayın. `.venv`, `__pycache__`, büyük checkpoint'ler veya gizli anahtarları eklemeyin.
3. Repo kökünde yeni bir branch oluşturun:

```powershell
git switch main
git pull origin main
git switch -c feature/nlp-v2
git add nlp/
git commit -m "feat: add SignTech NLP V2 rule engine and tests"
git push -u origin feature/nlp-v2
```

4. GitHub üzerinden `feature/nlp-v2` → `main` **Pull Request** açın; Kerem/Halil gözden geçirsin. Merge sonrası diğer üyeler güncel `main` branch'ini çekebilir.
5. Ortak reponun ana dalı `main` değilse komutlardaki dal adını uyarlayın. İlgili branch zaten varsa yeni bir ad kullanın.
