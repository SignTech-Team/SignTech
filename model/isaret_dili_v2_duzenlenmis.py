# -*- coding: utf-8 -*-
# Google Colab / IPython export. Not a standalone Python application.

# %%
# @title 🗑️ Eski Kamera Kayıtlarını Otomatik Temizle

KAMERA_TEMIZLE_AKTIF = True  # @param {type:"boolean"}

import os
import glob
import shutil

if KAMERA_TEMIZLE_AKTIF:

    klasorler = glob.glob("/content/v2_kamera_*")
    silinen = 0

    for klasor in klasorler:

        if os.path.isdir(klasor) and not os.path.islink(klasor):

            shutil.rmtree(klasor)
            silinen += 1

    if silinen > 0:
        print(f"🗑️ {silinen} kamera kayıt klasörü silindi.")
    else:
        print("✅ Temizlenecek eski kamera kaydı bulunamadı.")

    print("🔒 Eğitim verileri ve model dosyaları korundu.")

else:
    print("⏸️ Otomatik kamera temizliği kapalı.")

# %%
# @title 📦 V2 Kütüphane Kurulumu (Tek Hücre)

KUTUPHANELERI_KUR = True  # @param {type:"boolean"}

if not KUTUPHANELERI_KUR:
    print("⏭️ Kütüphane kurulumu atlandı.")
else:
    get_ipython().run_line_magic('pip', 'install --no-cache-dir "tensorflow==2.17.1" "mediapipe==0.10.21" "numpy==1.26.4" "pandas==2.2.3" "protobuf==4.25.8"')

    print("✅ Kurulum tamamlandı.")
    print("⚠️ Çalışma zamanını yeniden başlat.")

# %%
import mediapipe as mp

print("MediaPipe:", mp.__version__)
print("Solutions:", hasattr(mp, "solutions"))

# %%
# @title 🔗 GOOGLE DRIVE BAĞLANTISI

from google.colab import drive
import os

drive.mount("/content/drive")

MODEL_DIR = "/content/drive/MyDrive/AUTSL_Modelim"

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError("Model klasörü bulunamadı!")

print("✅ Google Drive bağlantısı hazır.")

# %%
# @title 🔑 Kaggle API Bağlantısı
import os
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
print("Kaggle API bilgileri yüklendi.")

# %%
import os
import glob

target_dir = "/content/autsl_data/AUTSL/train"

# 1. Kontrol: Veri zaten inmiş ve açılmış mı?
if os.path.exists(target_dir):
    print("✅ Veri seti zaten mevcut, indirme adımı atlandı!")
else:
    print("⏳ Veri seti bulunamadı, Kaggle'dan indiriliyor ve açılıyor...")

    # Klasörü oluştur
    get_ipython().system('mkdir -p /content/autsl_data')

    # Kaggle'dan doğrudan indir
    get_ipython().system('kaggle datasets download -d ngphmng/autsl-dataset -p /content/autsl_data/')

    # ZIP'i aç ve temizle
    get_ipython().system('unzip -q /content/autsl_data/*.zip -d /content/autsl_data/')
    get_ipython().system('rm -f /content/autsl_data/*.zip')

    print("✅ İndirme ve çıkarma başarıyla tamamlandı!")

# 2. İçerik Doğrulama
dosyalar = glob.glob("/content/autsl_data/**", recursive=False)
print("📂 autsl_data İçeriği:", [os.path.basename(x) for x in dosyalar])

# %% [markdown]
# V2 MODEL EGITIMI

# %%
import cv2
import numpy as np
import mediapipe as mp
import os

# --- V2 HİBRİT PİPELİNE SINIFI (219 Koordinat) ---
class SignLanguagePipelineV2:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh

        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def extract_landmarks(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Eller (126 koordinat)
        hands_results = self.hands.process(image_rgb)
        left_hand = np.zeros(21 * 3)
        right_hand = np.zeros(21 * 3)
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[idx].classification[0].label
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                if handedness == 'Left': left_hand = coords
                else: right_hand = coords
        hands_vector = np.concatenate([left_hand, right_hand])

        # 2. Gövde / Pose (33 koordinat: Seçilen 11 kritik nokta)
        pose_results = self.pose.process(image_rgb)
        pose_vector = np.zeros(11 * 3)
        if pose_results.pose_landmarks:
            selected_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
            pose_coords = [coord for idx in selected_indices for lm in [pose_results.pose_landmarks.landmark[idx]] for coord in [lm.x, lm.y, lm.z]]
            pose_vector = np.array(pose_coords)

        # 3. Yüz Mimikleri (60 koordinat: Seçilen 20 kritik nokta)
        face_results = self.face.process(image_rgb)
        face_vector = np.zeros(20 * 3)
        if face_results.multi_face_landmarks:
            face_indices = [13, 14, 17, 61, 291, 0, 267, 37, 267, 270, 409, 292, 308, 324, 318, 402, 317, 14, 87, 178]
            face_coords = [coord for idx in face_indices[:20] for lm in [face_results.multi_face_landmarks[0].landmark[idx]] for coord in [lm.x, lm.y, lm.z]]
            face_vector = np.array(face_coords)

        # Toplam: 126 + 33 + 60 = 219 koordinat
        return np.concatenate([hands_vector, pose_vector, face_vector])

print("V2 Pipeline sınıfı hazırlandı. Dosyalar '_v2' uzantısıyla izole bir şekilde kaydedilecek.")

# %%
# @title 🚀 V2 TEMİZ VERİ İŞLEME (SIFIRDAN + DRIVE + 100'LÜK KAYIT)

V2_ISLEME_AKTIF = False  # @param {type:"boolean"}
SADECE_ILK_PAKET_TEST = False  # @param {type:"boolean"}

if not V2_ISLEME_AKTIF:
    print("⏸️ V2 veri işleme kapalı.")
    print("✅ Mevcut dosyalara dokunulmadı.")

else:
    import os
    import glob
    import json
    import hashlib
    import shutil
    import cv2
    import numpy as np
    import pandas as pd
    import mediapipe as mp

    from tqdm.notebook import tqdm
    from google.colab import drive

    # ==================================================
    # 1. AYARLAR
    # ==================================================

    BASE_DIR = "/content/autsl_data/AUTSL"
    VIDEO_DIR = os.path.join(BASE_DIR, "train")
    CSV_PATH = os.path.join(BASE_DIR, "train.csv")

    COLAB_DIR = "/content/V2_Temiz_Baslangic"

    DRIVE_DIR = (
        "/content/drive/MyDrive/AUTSL_Modelim/"
        "V2_Temiz_Baslangic"
    )

    MAX_FRAMES = 30
    FEATURE_DIM = 219
    PAKET_BOYUTU = 100

    # ==================================================
    # 2. GOOGLE DRIVE BAĞLANTISI
    # ==================================================

    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")

    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError("Google Drive bağlanamadı!")

    os.makedirs(COLAB_DIR, exist_ok=True)
    os.makedirs(DRIVE_DIR, exist_ok=True)

    # ==================================================
    # 3. VERİ SETİ KONTROLÜ
    # ==================================================

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"train.csv bulunamadı: {CSV_PATH}"
        )

    all_videos = sorted(
        glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    )

    if len(all_videos) == 0:
        raise RuntimeError("AUTSL videoları bulunamadı!")

    df = pd.read_csv(CSV_PATH, header=None)

    video_map = dict(zip(
        df.iloc[:, 0].astype(str),
        df.iloc[:, 1]
    ))

    def etiket_bul(video_path):
        isim = os.path.basename(video_path)
        kok = os.path.splitext(isim)[0]

        if isim in video_map:
            return int(video_map[isim])

        if kok in video_map:
            return int(video_map[kok])

        raise ValueError(
            f"Video için etiket bulunamadı: {isim}"
        )

    # İşleme başlamadan tüm etiketleri kontrol et.
    labels_all = np.array(
        [etiket_bul(v) for v in all_videos],
        dtype=np.int64
    )

    TOPLAM = len(all_videos)

    print("📹 Toplam video:", TOPLAM)
    print("🏷️ Etiket sayısı:", len(labels_all))
    print("📁 Colab klasörü:", COLAB_DIR)
    print("☁️ Drive klasörü:", DRIVE_DIR)

    if SADECE_ILK_PAKET_TEST:
        print("🧪 TEST MODU: En fazla ilk 100 video kontrol edilecek.")
        islenecek_paket_baslangiclari = [0]
    else:
        print("🚀 TAM MOD: Bütün paketler kontrol edilecek.")
        islenecek_paket_baslangiclari = range(
            0, TOPLAM, PAKET_BOYUTU
        )

    # ==================================================
    # 4. DOSYA HASH FONKSİYONU
    # ==================================================

    def sha256_hesapla(path):
        h = hashlib.sha256()

        with open(path, "rb") as f:
            for blok in iter(
                lambda: f.read(1024 * 1024),
                b""
            ):
                h.update(blok)

        return h.hexdigest()

    # ==================================================
    # 5. PAKET DOĞRULAMA
    # ==================================================

    def paket_dogrula(paket_dir, bas, son):

        if not os.path.isdir(paket_dir):
            return False

        tamam = os.path.join(paket_dir, "TAMAMLANDI")
        x_path = os.path.join(paket_dir, "X.npy")
        y_path = os.path.join(paket_dir, "y.npy")
        meta_path = os.path.join(paket_dir, "bilgi.json")

        gerekli = [
            tamam,
            x_path,
            y_path,
            meta_path
        ]

        if not all(os.path.isfile(p) for p in gerekli):
            return False

        try:
            X = np.load(
                x_path,
                mmap_mode="r",
                allow_pickle=False
            )

            y = np.load(
                y_path,
                allow_pickle=False
            )

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            adet = son - bas

            if X.shape != (adet, MAX_FRAMES, FEATURE_DIM):
                return False

            if X.dtype != np.float32:
                return False

            if y.shape != (adet,):
                return False

            if not np.array_equal(y, labels_all[bas:son]):
                return False

            if meta.get("baslangic") != bas:
                return False

            if meta.get("bitis") != son:
                return False

            videolar_beklenen = [
                os.path.basename(p)
                for p in all_videos[bas:son]
            ]

            if meta.get("videolar") != videolar_beklenen:
                return False

            if sha256_hesapla(x_path) != meta.get("X_sha256"):
                return False

            if sha256_hesapla(y_path) != meta.get("y_sha256"):
                return False

            return True

        except Exception:
            return False

    # ==================================================
    # 6. V2 LANDMARK ÇIKARMA
    # ==================================================

    if "SignLanguagePipelineV2" not in globals():
        raise RuntimeError(
            "Önce V2 HİBRİT PİPELİNE SINIFI kodunu çalıştır."
        )

    def video_isle(video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            return (
                np.zeros(
                    (MAX_FRAMES, FEATURE_DIM),
                    dtype=np.float32
                ),
                "acilamadi"
            )

        toplam_kare = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if toplam_kare <= 0:
            cap.release()
            return (
                np.zeros(
                    (MAX_FRAMES, FEATURE_DIM),
                    dtype=np.float32
                ),
                "kare_yok"
            )

        # Videoların birbirinin takip durumunu
        # etkilememesi için ayrı pipeline oluştur.
        pipeline = SignLanguagePipelineV2()

        secilen = set(
            np.linspace(
                0,
                toplam_kare - 1,
                min(MAX_FRAMES, toplam_kare),
                dtype=int
            ).tolist()
        )

        sequence = []

        try:
            for frame_idx in range(toplam_kare):

                ret, frame = cap.read()

                if not ret:
                    break

                if frame_idx not in secilen:
                    continue

                landmarks = pipeline.extract_landmarks(frame)

                landmarks = np.asarray(
                    landmarks,
                    dtype=np.float32
                )

                if landmarks.shape != (FEATURE_DIM,):
                    raise ValueError(
                        f"Landmark boyutu hatalı: {landmarks.shape}"
                    )

                if not np.isfinite(landmarks).all():
                    raise ValueError(
                        "NaN veya sonsuz landmark değeri."
                    )

                sequence.append(landmarks)

        finally:
            cap.release()
            pipeline.hands.close()
            pipeline.pose.close()
            pipeline.face.close()

        if not sequence:
            return (
                np.zeros(
                    (MAX_FRAMES, FEATURE_DIM),
                    dtype=np.float32
                ),
                "kare_okunamadi"
            )

        while len(sequence) < MAX_FRAMES:
            sequence.append(
                np.zeros(FEATURE_DIM, dtype=np.float32)
            )

        veri = np.asarray(
            sequence[:MAX_FRAMES],
            dtype=np.float32
        )

        if np.all(veri == 0):
            durum = "landmark_bulunamadi"
        else:
            durum = "basarili"

        return veri, durum

    # ==================================================
    # 7. PAKET KAYDETME
    # ==================================================

    def paket_olustur(bas, son, paket_dir):

        adet = son - bas

        X_paket = np.zeros(
            (adet, MAX_FRAMES, FEATURE_DIM),
            dtype=np.float32
        )

        y_paket = labels_all[bas:son].copy()

        durumlar = []
        hatalar = []

        for yerel_idx, genel_idx in enumerate(
            tqdm(
                range(bas, son),
                desc=f"Videolar {bas}-{son}"
            )
        ):

            video_path = all_videos[genel_idx]

            try:
                veri, durum = video_isle(video_path)

                X_paket[yerel_idx] = veri

            except Exception as e:
                durum = "isleme_hatasi"

                hatalar.append({
                    "indeks": int(genel_idx),
                    "video": os.path.basename(video_path),
                    "hata": str(e)
                })

            durumlar.append(durum)

        x_path = os.path.join(paket_dir, "X.npy")
        y_path = os.path.join(paket_dir, "y.npy")
        meta_path = os.path.join(paket_dir, "bilgi.json")

        np.save(x_path, X_paket)
        np.save(y_path, y_paket)

        meta = {
            "surum": "V2_Temiz_Baslangic",
            "baslangic": int(bas),
            "bitis": int(son),
            "video_sayisi": int(adet),

            "videolar": [
                os.path.basename(v)
                for v in all_videos[bas:son]
            ],

            "durumlar": durumlar,
            "hatalar": hatalar,

            "X_sha256": sha256_hesapla(x_path),
            "y_sha256": sha256_hesapla(y_path)
        }

        with open(
            meta_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                meta,
                f,
                ensure_ascii=False,
                indent=2
            )

        with open(
            os.path.join(paket_dir, "TAMAMLANDI"),
            "w",
            encoding="utf-8"
        ) as f:
            f.write("OK")

        return durumlar

    # ==================================================
    # 8. ANA İŞLEME DÖNGÜSÜ
    # ==================================================

    print("\n🚀 V2 işleme kontrolü başlıyor...")

    for bas in islenecek_paket_baslangiclari:

        son = min(
            bas + PAKET_BOYUTU,
            TOPLAM
        )

        paket_adi = f"paket_{bas:05d}_{son:05d}"

        colab_paket = os.path.join(
            COLAB_DIR,
            paket_adi
        )

        drive_paket = os.path.join(
            DRIVE_DIR,
            paket_adi
        )

        # ----------------------------------------------
        # A. DRIVE'DA TAMAMLANMIŞ PAKET VAR MI?
        # ----------------------------------------------

        if paket_dogrula(drive_paket, bas, son):
            print(
                f"✅ Drive'da mevcut, yeniden işlenmedi: "
                f"{paket_adi}"
            )
            continue

        # Drive'da bozuk veya yarım paket varsa dur.
        if os.path.exists(drive_paket):
            raise RuntimeError(
                f"⚠️ Drive'da eksik/bozuk paket var: {paket_adi}\n"
                "Dosyaların üzerine yazılmadı."
            )

        # ----------------------------------------------
        # B. COLAB'DA TAMAMLANMIŞ PAKET VAR MI?
        # ----------------------------------------------

        if not paket_dogrula(colab_paket, bas, son):

            if os.path.exists(colab_paket):
                raise RuntimeError(
                    f"⚠️ Colab'da eksik/bozuk paket var: {paket_adi}\n"
                    "Dosyaların üzerine yazılmadı."
                )

            gecici_colab = colab_paket + ".tmp"

            if os.path.exists(gecici_colab):
                raise RuntimeError(
                    f"Önceden kalan geçici paket var: "
                    f"{gecici_colab}"
                )

            os.makedirs(gecici_colab)

            print(f"\n📹 İşleniyor: {bas} - {son}")

            durumlar = paket_olustur(
                bas,
                son,
                gecici_colab
            )

            # Geçici paketi doğrula.
            if not paket_dogrula(
                gecici_colab,
                bas,
                son
            ):
                raise RuntimeError(
                    "Oluşturulan paket doğrulanamadı!"
                )

            # Tamamlanan paketi asıl yerine taşı.
            os.rename(
                gecici_colab,
                colab_paket
            )

            print(
                f"💾 Colab'a kaydedildi: "
                f"{paket_adi}"
            )

            print(
                "✅ Başarılı video:",
                durumlar.count("basarili")
            )

            print(
                "⚠️ Sorunlu video:",
                len(durumlar) -
                durumlar.count("basarili")
            )

        # ----------------------------------------------
        # C. GOOGLE DRIVE'A KOPYALA
        # ----------------------------------------------

        if not paket_dogrula(
            colab_paket,
            bas,
            son
        ):
            raise RuntimeError(
                f"Colab paketi doğrulanamadı: {paket_adi}"
            )

        gecici_drive = drive_paket + ".tmp"

        if os.path.exists(gecici_drive):
            raise RuntimeError(
                f"Drive'da geçici paket var: {gecici_drive}"
            )

        shutil.copytree(
            colab_paket,
            gecici_drive
        )

        if not paket_dogrula(
            gecici_drive,
            bas,
            son
        ):
            raise RuntimeError(
                "Drive'a kopyalanan paket doğrulanamadı!"
            )

        os.rename(
            gecici_drive,
            drive_paket
        )

        print(
            f"☁️ Drive'a kaydedildi: {paket_adi}"
        )

    # ==================================================
    # 9. SONUÇ
    # ==================================================

    print("\n" + "=" * 50)
    print("✅ V2 İŞLEME OTURUMU TAMAMLANDI")
    print("=" * 50)

    print("📁 Colab:", COLAB_DIR)
    print("☁️ Drive:", DRIVE_DIR)

    if SADECE_ILK_PAKET_TEST:
        print(
            "🧪 İlk 100 video test modu tamamlandı."
        )
        print(
            "Devam etmek için "
            "SADECE_ILK_PAKET_TEST = False yap."
        )
    else:
        print(
            "✅ Hedeflenen paketlerin kontrolü tamamlandı."
        )

# %%
# @title 🧠 V2 Transformer Model Mimarisi (219 Özellik)

import torch
import torch.nn as nn

class SignTransformerV2(nn.Module):

    def __init__(self, num_classes, input_dim=219):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, 64)

        encoder = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder,
            num_layers=2
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

print("✅ V2 Transformer mimarisi hazır.")

# %%
# @title 🧠 V2 TRANSFORMER MODEL EĞİTİMİ (KALDIĞI YERDEN)

EGITIM_AKTIF = False  # @param {type:"boolean"}
EPOCH_SAYISI = 100  # @param {type:"integer"}
BATCH_SIZE = 32  # @param {type:"integer"}

if not EGITIM_AKTIF:
    print("⏸️ V2 model eğitimi atlandı.")
    print("✅ Mevcut model ve veri dosyaları korundu.")

else:
    import os
    import glob
    import json
    import hashlib
    import shutil
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score

    from torch.utils.data import Dataset, DataLoader, random_split
    from sklearn.preprocessing import LabelEncoder

    # ==================================================
    # 1. AYARLAR
    # ==================================================

    BASE_DIR = "/content/autsl_data/AUTSL"

    VIDEO_DIR = os.path.join(BASE_DIR, "train")
    CSV_PATH = os.path.join(BASE_DIR, "train.csv")

    DRIVE_DIR = (
        "/content/drive/MyDrive/AUTSL_Modelim/"
        "V2_Temiz_Baslangic"
    )

    MODEL_COLAB = (
        "/content/sign_language_transformer_v2_temiz.pth"
    )

    MODEL_DRIVE = (
        "/content/drive/MyDrive/AUTSL_Modelim/"
        "sign_language_transformer_v2_temiz.pth"
    )

    MAX_FRAMES = 30
    FEATURE_DIM = 219
    PAKET_BOYUTU = 100

    if EPOCH_SAYISI <= 0 or BATCH_SIZE <= 0:
        raise ValueError(
            "Epoch sayısı ve batch size pozitif olmalı."
        )

    if "SignTransformerV2" not in globals():
        raise RuntimeError(
            "Önce V2 Transformer Model Mimarisi "
            "hücresini çalıştır."
        )

    if not os.path.isdir(DRIVE_DIR):
        raise FileNotFoundError(
            "Temiz V2 kayıt klasörü bulunamadı."
        )

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            "AUTSL train.csv bulunamadı."
        )

    # ==================================================
    # 2. ORİJİNAL VİDEO SIRASI
    # ==================================================

    videolar = sorted(
        glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    )

    df = pd.read_csv(CSV_PATH, header=None)

    video_map = dict(zip(
        df.iloc[:, 0].astype(str),
        df.iloc[:, 1]
    ))

    def etiket_bul(path):
        isim = os.path.basename(path)
        kok = os.path.splitext(isim)[0]

        label = video_map.get(
            isim,
            video_map.get(kok)
        )

        if label is None:
            raise ValueError(
                f"Etiket bulunamadı: {isim}"
            )

        return int(label)

    beklenen_y = np.asarray(
        [etiket_bul(v) for v in videolar],
        dtype=np.int64
    )

    TOPLAM = len(videolar)

    if TOPLAM == 0:
        raise RuntimeError("AUTSL videoları bulunamadı.")

    print("📹 Toplam video:", TOPLAM)

    # ==================================================
    # 3. HASH FONKSİYONU
    # ==================================================

    def sha256_hesapla(path):

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for blok in iter(
                lambda: f.read(1024 * 1024),
                b""
            ):
                h.update(blok)

        return h.hexdigest()

    # ==================================================
    # 4. TÜM PAKETLERİ DOĞRULA
    # ==================================================

    paket_bilgileri = []

    for bas in range(0, TOPLAM, PAKET_BOYUTU):

        son = min(bas + PAKET_BOYUTU, TOPLAM)

        paket_adi = f"paket_{bas:05d}_{son:05d}"

        paket_dir = os.path.join(
            DRIVE_DIR,
            paket_adi
        )

        x_path = os.path.join(paket_dir, "X.npy")
        y_path = os.path.join(paket_dir, "y.npy")
        meta_path = os.path.join(paket_dir, "bilgi.json")

        tamam_path = os.path.join(
            paket_dir,
            "TAMAMLANDI"
        )

        gerekli = [
            x_path,
            y_path,
            meta_path,
            tamam_path
        ]

        if not all(os.path.isfile(p) for p in gerekli):
            raise RuntimeError(
                f"❌ Eksik paket: {paket_adi}\n"
                "Bütün videolar işlenmeden eğitim başlamaz."
            )

        X_paket = np.load(
            x_path,
            mmap_mode="r",
            allow_pickle=False
        )

        y_paket = np.load(
            y_path,
            allow_pickle=False
        )

        with open(
            meta_path,
            "r",
            encoding="utf-8"
        ) as f:
            meta = json.load(f)

        adet = son - bas

        if X_paket.shape != (
            adet,
            MAX_FRAMES,
            FEATURE_DIM
        ):
            raise ValueError(
                f"X boyutu hatalı: {paket_adi}"
            )

        if X_paket.dtype != np.float32:
            raise ValueError(
                f"X veri tipi hatalı: {paket_adi}"
            )

        if y_paket.shape != (adet,):
            raise ValueError(
                f"y boyutu hatalı: {paket_adi}"
            )

        if not np.array_equal(
            y_paket,
            beklenen_y[bas:son]
        ):
            raise ValueError(
                f"Etiket sırası hatalı: {paket_adi}"
            )

        beklenen_isimler = [
            os.path.basename(p)
            for p in videolar[bas:son]
        ]

        if meta.get("videolar") != beklenen_isimler:
            raise ValueError(
                f"Video sırası hatalı: {paket_adi}"
            )

        if (
            meta.get("baslangic") != bas
            or meta.get("bitis") != son
        ):
            raise ValueError(
                f"Paket indeksleri hatalı: {paket_adi}"
            )

        if sha256_hesapla(x_path) != meta.get("X_sha256"):
            raise ValueError(
                f"X hash uyuşmuyor: {paket_adi}"
            )

        if sha256_hesapla(y_path) != meta.get("y_sha256"):
            raise ValueError(
                f"y hash uyuşmuyor: {paket_adi}"
            )

        if len(meta.get("durumlar", [])) != adet:
            raise ValueError(
                f"Video durumları eksik: {paket_adi}"
            )

        paket_bilgileri.append({
            "x_path": x_path,
            "y_path": y_path,
            "durumlar": meta["durumlar"]
        })

    print("✅ Bütün paketler doğrulandı.")
    print("📦 Paket sayısı:", len(paket_bilgileri))

    # ==================================================
    # 5. GEÇERLİ VERİLERİ BİRLEŞTİR
    # ==================================================

    X_list = []
    y_list = []

    atlanan = 0

    for paket in paket_bilgileri:

        X_paket = np.load(
            paket["x_path"],
            mmap_mode="r",
            allow_pickle=False
        )

        y_paket = np.load(
            paket["y_path"],
            allow_pickle=False
        )

        for i, durum in enumerate(paket["durumlar"]):

            lm = X_paket[i]

            if durum != "basarili":
                atlanan += 1
                continue

            if not np.isfinite(lm).all():
                atlanan += 1
                continue

            if np.all(lm == 0):
                atlanan += 1
                continue

            X_list.append(
                np.asarray(lm, dtype=np.float32)
            )

            y_list.append(int(y_paket[i]))

    if not X_list:
        raise RuntimeError(
            "Eğitim için kullanılabilir veri bulunamadı."
        )

    X = np.stack(X_list).astype(
        np.float32,
        copy=False
    )

    y = np.asarray(y_list, dtype=np.int64)

    del X_list, y_list

    print("\n📊 EĞİTİM VERİLERİ")
    print("✅ Kullanılabilir video:", len(X))
    print("⚠️ Atlanan video:", atlanan)
    print("📐 X boyutu:", X.shape)

    # ==================================================
    # 6. ETİKET KODLAMA
    # ==================================================

    le = LabelEncoder()

    y_encoded = le.fit_transform(y)

    print("🏷️ Sınıf sayısı:", len(le.classes_))

    if len(le.classes_) < 2:
        raise RuntimeError(
            "Eğitim için en az iki sınıf gerekli."
        )

    # ==================================================
    # 7. DATASET
    # ==================================================

    class V2Dataset(Dataset):

        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):

            features = np.array(
                self.X[idx],
                dtype=np.float32,
                copy=True
            )

            return (
                torch.from_numpy(features),
                torch.tensor(
                    int(self.y[idx]),
                    dtype=torch.long
                )
            )

    dataset = V2Dataset(
        X,
        y_encoded
    )

    # ==================================================
    # 8. EĞİTİM / DOĞRULAMA
    # ==================================================

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ==================================================
    # 9. MODEL HAZIRLIĞI
    # ==================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("🖥️ Eğitim cihazı:", device)

    model_v2 = SignTransformerV2(
        num_classes=len(le.classes_),
        input_dim=FEATURE_DIM
    ).to(device)

    # Orijinal model korunur; yeni dosyalar ayrı tutulur.
    ESKI_MODEL = MODEL_DRIVE
    YENI_MODEL = MODEL_DRIVE.replace(".pth", "_devam_eniyi.pth")
    YENI_COLAB = MODEL_COLAB.replace(".pth", "_devam_eniyi.pth")
    CHECKPOINT = MODEL_DRIVE.replace(".pth", "_devam_checkpoint.pth")
    SABIR = 15

    optimizer = torch.optim.Adam(model_v2.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # Aynı veriler ve aynı %90/%10 bölme ile devam et.
    veri_hash = hashlib.sha256()
    veri_hash.update(memoryview(X).cast("B"))
    veri_hash.update(y_encoded.tobytes())
    veri_imza = veri_hash.hexdigest()

    def guvenli_kaydet(payload, hedef):
        gecici = hedef + ".tmp"
        torch.save(payload, gecici)
        os.replace(gecici, hedef)

    def model_kontrol(kayit):
        if kayit["classes"] != le.classes_.tolist():
            raise ValueError("Model ve veri sınıfları uyuşmuyor.")
        if kayit["input_dim"] != FEATURE_DIM or kayit["max_frames"] != MAX_FRAMES:
            raise ValueError("Model giriş boyutu uyuşmuyor.")

    if os.path.isfile(CHECKPOINT):
        ck = torch.load(CHECKPOINT, map_location=device, weights_only=True)
        model_kontrol(ck)
        if ck["veri_imza"] != veri_imza:
            raise ValueError("Veri değişmiş; checkpoint ile devam edilemez.")
        model_v2.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        baslangic = ck["epoch"]
        best_acc = ck["best_accuracy"]
        bekleme = ck["bekleme"]
        history = ck["history"]
        if ck.get("en_iyi_yeni") and not os.path.isfile(YENI_MODEL):
            raise FileNotFoundError("Yeni en iyi model dosyası eksik.")
        print(f"🔄 Checkpoint yüklendi: {baslangic}. epoch tamamlanmış.")
    else:
        if not os.path.isfile(ESKI_MODEL):
            raise FileNotFoundError("20 epoch eğitilmiş orijinal model bulunamadı.")
        if os.path.isfile(YENI_MODEL):
            raise RuntimeError("Yeni model var fakat checkpoint yok; dosyaları kontrol edin.")
        ck = torch.load(ESKI_MODEL, map_location=device, weights_only=True)
        model_kontrol(ck)
        model_v2.load_state_dict(ck["model_state_dict"])
        baslangic = 20  # Kullanıcının tamamladığı ilk eğitim.
        best_acc = float(ck["best_accuracy"])
        bekleme = 0
        history = []
        print(f"🔄 Orijinal model yüklendi. En iyi doğrulama: %{100*best_acc:.2f}")
        print("ℹ️ Eski optimizer kaydedilmediği için yeni Adam ile devam ediliyor.")

    for epoch in range(baslangic, EPOCH_SAYISI):
        if bekleme >= SABIR:
            print("⏹️ Early stopping sınırına ulaşıldı.")
            break
        model_v2.train()
        correct = total = 0
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            output = model_v2(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
            correct += (output.argmax(1) == by).sum().item()
            total += len(by)

        model_v2.eval()
        val_loss = 0.0
        val_correct = val_total = 0
        tahminler, gercekler = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                output = model_v2(bx)
                val_loss += criterion(output, by).item() * len(by)
                pred = output.argmax(1)
                val_correct += (pred == by).sum().item()
                val_total += len(by)
                tahminler.extend(pred.cpu().tolist())
                gercekler.extend(by.cpu().tolist())

        accuracy = val_correct / val_total
        train_acc = correct / total
        macro_f1 = float(f1_score(gercekler, tahminler, average="macro",
                                 labels=list(range(len(le.classes_))), zero_division=0))
        scheduler.step(accuracy)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch+1, "train_acc": train_acc,
                        "val_acc": accuracy, "train_loss": train_loss/total,
                        "val_loss": val_loss/val_total, "macro_f1": macro_f1, "lr": lr})
        print(f"Epoch {epoch+1}/{EPOCH_SAYISI} | Eğitim: %{100*train_acc:.2f} | "
              f"Doğrulama: %{100*accuracy:.2f} | F1: {macro_f1:.4f} | "
              f"Loss: {train_loss/total:.4f}/{val_loss/val_total:.4f} | LR: {lr:.6g}")

        if accuracy > best_acc:
            best_acc = accuracy
            bekleme = 0
            payload = {"model_state_dict": model_v2.state_dict(),
                       "classes": le.classes_.tolist(), "input_dim": FEATURE_DIM,
                       "max_frames": MAX_FRAMES, "best_accuracy": best_acc,
                       "epoch": epoch+1}
            guvenli_kaydet(payload, YENI_COLAB)
            guvenli_kaydet(payload, YENI_MODEL)
            print("💾 Yeni en iyi model kaydedildi; orijinal model korundu.")
        else:
            bekleme += 1

        guvenli_kaydet({"model_state_dict": model_v2.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "classes": le.classes_.tolist(), "input_dim": FEATURE_DIM,
                        "max_frames": MAX_FRAMES, "epoch": epoch+1,
                        "best_accuracy": best_acc, "bekleme": bekleme,
                        "history": history, "veri_imza": veri_imza,
                        "en_iyi_yeni": os.path.isfile(YENI_MODEL)}, CHECKPOINT)
        if bekleme >= SABIR:
            print("⏹️ 15 epoch boyunca iyileşme olmadı; eğitim durduruldu.")
            break

    SECILEN_MODEL = YENI_MODEL if os.path.isfile(YENI_MODEL) else ESKI_MODEL
    # Test hücreleri model_v2 nesnesini kullanıyorsa en iyi ağırlıkları yükle.
    en_iyi = torch.load(SECILEN_MODEL, map_location=device, weights_only=True)
    model_v2.load_state_dict(en_iyi["model_state_dict"])
    model_v2.eval()
    print(f"🏆 En iyi doğrulama: %{100*float(en_iyi['best_accuracy']):.2f}")
    print("📁 Testte kullanılacak model:", SECILEN_MODEL)
    print("☁️ Devam checkpoint:", CHECKPOINT)

    if history:
        ep = [h["epoch"] for h in history]
        plt.figure(figsize=(9, 4))
        plt.plot(ep, [100*h["train_acc"] for h in history], label="Eğitim")
        plt.plot(ep, [100*h["val_acc"] for h in history], label="Doğrulama")
        plt.xlabel("Epoch"); plt.ylabel("Doğruluk (%)"); plt.legend(); plt.grid(); plt.show()
        plt.figure(figsize=(9, 4))
        plt.plot(ep, [h["train_loss"] for h in history], label="Eğitim loss")
        plt.plot(ep, [h["val_loss"] for h in history], label="Doğrulama loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(); plt.show()

# %%
# @title 🎬 V2 ÖRNEK VİDEO TESTİ (GÜVENLİ)

V2_TEST_AKTIF = True  # @param {type:"boolean"}

if not V2_TEST_AKTIF:
    print("⏭️ V2 video testi atlandı.")

else:
    import os
    import glob
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import subprocess
    import base64
    import tempfile

    from IPython.display import display, HTML

    # ================================================
    # 1. MODEL VE PIPELINE KONTROLÜ
    # ================================================

    if "SignTransformerV2" not in globals():
        raise RuntimeError(
            "Önce V2 Transformer Model Mimarisi hücresini çalıştır."
        )

    if "SignLanguagePipelineV2" not in globals():
        raise RuntimeError(
            "Önce V2 Pipeline sınıfını çalıştır."
        )

    model_yollari = [
        "/content/drive/MyDrive/AUTSL_Modelim/"
        "sign_language_transformer_v2_temiz_devam_eniyi.pth",

        "/content/"
        "sign_language_transformer_v2_temiz_devam_eniyi.pth"
    ]

    MODEL_PATH = next(
        (p for p in model_yollari if os.path.isfile(p)),
        None
    )

    if MODEL_PATH is None:
        raise FileNotFoundError(
            "Yeni V2 model dosyası bulunamadı. "
            "Google Drive bağlantısını kontrol et."
        )

    print("📁 Model bulundu:", MODEL_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    classes_v2 = np.asarray(checkpoint["classes"])

    FEATURE_DIM = int(checkpoint["input_dim"])
    MAX_FRAMES = int(checkpoint["max_frames"])

    if FEATURE_DIM != 219 or MAX_FRAMES != 30:
        raise ValueError(
            "Model giriş boyutları V2 ile uyumlu değil."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    test_model = SignTransformerV2(
        num_classes=len(classes_v2),
        input_dim=FEATURE_DIM
    ).to(device)

    test_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_model.eval()

    print("✅ V2 modeli hazır.")
    print("🏷️ Sınıf sayısı:", len(classes_v2))
    print("🖥️ Cihaz:", device)

    if "best_accuracy" in checkpoint:
        print(
            f"🏆 Model doğrulama başarısı: "
            f"%{checkpoint['best_accuracy'] * 100:.2f}"
        )

    if "epoch" in checkpoint:
        print("📊 Model epoch:", checkpoint["epoch"])

    # ================================================
    # 2. PIPELINE
    # ================================================

    test_pipeline = SignLanguagePipelineV2()

    # ================================================
    # 3. DOĞRU AUTSL TÜRKÇE KELİME LİSTESİ
    # ================================================

    id_to_word = {}

    kelime_yollari = [
        "/content/drive/MyDrive/AUTSL_Modelim/"
        "SignList_AUTSL_Dogru.csv",

        "/content/SignList_AUTSL_Dogru.csv"
    ]

    word_path = next(
        (p for p in kelime_yollari if os.path.isfile(p)),
        None
    )

    if word_path is None:
        raise FileNotFoundError(
            "❌ Doğru AUTSL kelime listesi bulunamadı. "
            "Önce doğru kelime listesi oluşturma hücresini çalıştır."
        )

    df_words = pd.read_csv(
        word_path,
        encoding="utf-8-sig"
    )

    # Gerekli sütunları kontrol et.

    gerekli_sutunlar = {"ClassId", "TR", "EN"}

    if not gerekli_sutunlar.issubset(df_words.columns):
        raise ValueError(
            "❌ AUTSL kelime listesi sütunları hatalı."
        )

    df_words["ClassId"] = pd.to_numeric(
        df_words["ClassId"],
        errors="raise"
    )

    if not (df_words["ClassId"] % 1 == 0).all():
        raise ValueError(
            "❌ Geçersiz sınıf numarası bulundu."
        )

    df_words["ClassId"] = df_words["ClassId"].astype(int)

    # 226 sınıf kontrolü.

    if len(df_words) != 226:
        raise ValueError(
            f"❌ 226 sınıf bekleniyor. "
            f"Bulunan: {len(df_words)}"
        )

    if not df_words["ClassId"].is_unique:
        raise ValueError(
            "❌ Tekrarlanan sınıf numarası bulundu."
        )

    if set(df_words["ClassId"]) != set(range(226)):
        raise ValueError(
            "❌ Sınıf numaraları 0-225 aralığıyla uyuşmuyor."
        )

    if (
        df_words["TR"].isna().any()
        or df_words["TR"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "❌ Boş Türkçe kelime bulundu."
        )

    # Model sınıfları ile sözlüğü karşılaştır.

    model_siniflari = set(map(int, classes_v2))
    sozluk_siniflari = set(df_words["ClassId"])

    if (
        len(classes_v2) != 226
        or model_siniflari != sozluk_siniflari
    ):
        raise ValueError(
            "❌ Model sınıfları ile AUTSL sözlüğü uyuşmuyor."
        )

    # Türkçe kelime eşleştirmesini oluştur.

    id_to_word = dict(
        zip(
            df_words["ClassId"],
            df_words["TR"].astype(str).str.strip()
        )
    )

    # Önceden tespit ettiğimiz sözlük hatasını kontrol et.

    if id_to_word.get(222, "").casefold() != "yorgun":
        raise ValueError(
            "❌ Yanlış AUTSL kelime listesi kullanılıyor. "
            "222 numaralı sınıf 'yorgun' olmalı."
        )

    print("📖 Doğru AUTSL kelime listesi yüklendi.")
    print("📁 Kelime listesi:", word_path)
    print("🏷️ Toplam kelime:", len(id_to_word))
    print("✅ Model sınıfları sözlükle uyumlu.")
    print("✅ 222 numaralı sınıf:", id_to_word[222])

    def kelime_bul(class_id):

        try:
            return id_to_word.get(
                int(class_id),
                f"Sınıf {class_id}"
            )

        except (ValueError, TypeError):
            return str(class_id)

    # ================================================
    # 4. VİDEODAN LANDMARK ÇIKARMA
    # ================================================

    def video_ozellik_cikar(video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            cap.release()

            raise ValueError(
                f"Video açılamadı: {video_path}"
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:

            cap.release()

            raise ValueError(
                "Video kare sayısı okunamadı."
            )

        indices = np.linspace(
            0,
            total_frames - 1,
            MAX_FRAMES,
            dtype=int
        )

        frames = []

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frames.append(frame)

        finally:

            cap.release()

        if not frames:

            raise ValueError(
                "Videoda okunabilir kare yok."
            )

        sequence = []

        for idx in indices:

            idx = min(idx, len(frames) - 1)

            landmark = test_pipeline.extract_landmarks(
                frames[idx]
            )

            landmark = np.asarray(
                landmark,
                dtype=np.float32
            ).reshape(-1)

            if landmark.size != FEATURE_DIM:

                raise ValueError(
                    f"{FEATURE_DIM} özellik bekleniyor: "
                    f"{landmark.size}"
                )

            sequence.append(landmark)

        features = np.asarray(
            sequence,
            dtype=np.float32
        )

        if features.shape != (MAX_FRAMES, FEATURE_DIM):

            raise ValueError(
                f"Geçersiz özellik boyutu: {features.shape}"
            )

        if not np.isfinite(features).all():

            raise ValueError(
                "Landmark verisinde NaN veya Inf bulundu."
            )

        if np.all(features == 0):

            raise ValueError(
                "Videoda geçerli landmark bulunamadı."
            )

        return features

    # ================================================
    # 5. VİDEO GÖSTERME
    # ================================================

    def video_goster(video_path):

        with tempfile.TemporaryDirectory() as temp_dir:

            output_path = os.path.join(
                temp_dir,
                "video.mp4"
            )

            sonuc = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "error",
                    "-i", video_path,
                    "-an",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    output_path
                ],
                capture_output=True
            )

            if (
                sonuc.returncode != 0
                or not os.path.isfile(output_path)
            ):

                print("⚠️ Video görüntülenemedi.")
                return

            with open(output_path, "rb") as f:

                video_base64 = base64.b64encode(
                    f.read()
                ).decode("utf-8")

            display(HTML(f"""
            <video width="500" controls>
                <source
                    src="data:video/mp4;base64,{video_base64}"
                    type="video/mp4">
            </video>
            """))

    # ================================================
    # 6. V2 TAHMİN FONKSİYONU
    # ================================================

    def v2_tahmin(video_path, gercek_id=None):

        print("\n🔍 Video analiz ediliyor...")

        features = video_ozellik_cikar(video_path)

        print("📐 Özellik boyutu:", features.shape)

        tensor = torch.from_numpy(
            features
        ).unsqueeze(0).to(device)

        with torch.inference_mode():

            output = test_model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(3, len(classes_v2))
        )

        print("\n" + "=" * 50)
        print("🎯 V2 TAHMİN SONUÇLARI")
        print("=" * 50)

        for sira, (prob, idx) in enumerate(
            zip(top_probs, top_indices),
            start=1
        ):

            class_id = int(classes_v2[idx.item()])

            print(
                f"{sira}. {kelime_bul(class_id)} "
                f"— %{prob.item() * 100:.2f} "
                f"(ID: {class_id})"
            )

        tahmin_id = int(
            classes_v2[top_indices[0].item()]
        )

        if gercek_id is not None:

            gercek_id = int(gercek_id)

            print(
                "\n📌 Gerçek işaret:",
                kelime_bul(gercek_id),
                f"(ID: {gercek_id})"
            )

            print(
                "🤖 Model tahmini:",
                kelime_bul(tahmin_id),
                f"(ID: {tahmin_id})"
            )

            if tahmin_id == gercek_id:

                print("✅ DOĞRU TAHMİN!")

            else:

                print("❌ YANLIŞ TAHMİN!")

            top3_ids = [
                int(classes_v2[i.item()])
                for i in top_indices
            ]

            if gercek_id in top3_ids:

                print("✅ Gerçek işaret Top-3 içinde.")

            else:

                print("❌ Gerçek işaret Top-3 içinde değil.")

        print("=" * 50)

        video_goster(video_path)

    # ================================================
    # 7. ÖRNEK VİDEO TESTİ
    # ================================================

    base_dir = "/content/autsl_data/AUTSL"

    videolar = sorted(
        glob.glob(
            os.path.join(base_dir, "train", "*.mp4")
        )
    )

    if not videolar:

        print("⚠️ AUTSL videoları bulunamadı.")

    else:

        print("📹 Bulunan video sayısı:", len(videolar))

        # Her çalıştırmada rastgele video seç.

        rng = np.random.default_rng()

        secilen_video = None

        for idx in rng.permutation(len(videolar)):

            yol = videolar[idx]

            cap = cv2.VideoCapture(yol)

            ret, _ = cap.read()

            cap.release()

            if ret:

                secilen_video = yol
                break

        if secilen_video is None:

            print("❌ Okunabilir video bulunamadı.")

        else:

            print(
                "📹 Seçilen video:",
                os.path.basename(secilen_video)
            )

            gercek_id = None

            csv_path = os.path.join(
                base_dir,
                "train.csv"
            )

            if os.path.isfile(csv_path):

                df = pd.read_csv(
                    csv_path,
                    header=None
                )

                video_map = dict(
                    zip(
                        df[0].astype(str),
                        df[1]
                    )
                )

                isim = os.path.basename(
                    secilen_video
                )

                gercek_id = video_map.get(
                    isim,
                    video_map.get(
                        os.path.splitext(isim)[0]
                    )
                )

            print(
                "ℹ️ Bu örnek train klasöründen seçildi."
            )

            v2_tahmin(
                secilen_video,
                gercek_id
            )

# %%
# @title 📷 V2 KAMERA İLE İŞARET DİLİ TESTİ (GÜVENLİ)

KAMERA_TEST_AKTIF = False  # @param {type:"boolean"}
KAYIT_SURESI = 2  # @param {type:"integer"}
TEST_KELIMESI = ""  # @param {type:"string"}

if not KAMERA_TEST_AKTIF:

    print("⏸️ V2 kamera testi atlandı.")
    print("✅ Kamera açılmadı.")
    print("✅ Mevcut kayıtlar korundu.")

else:
    import os
    import base64
    import subprocess
    import tempfile
    import unicodedata
    import numpy as np
    import pandas as pd

    from IPython.display import display, Javascript
    from google.colab.output import eval_js

    # ================================================
    # 1. MODEL KONTROLÜ
    # ================================================

    if "v2_tahmin" not in globals():
        raise RuntimeError(
            "❌ V2 tahmin fonksiyonu hazır değil!\n"
            "Önce '🎬 V2 ÖRNEK VİDEO TESTİ (GÜVENLİ)' "
            "hücresini çalıştırmalısın."
        )

    if "test_model" not in globals():
        raise RuntimeError(
            "❌ V2 modeli yüklenmemiş!"
        )

    if "classes_v2" not in globals():
        raise RuntimeError(
            "❌ V2 model sınıfları bulunamadı."
        )

    if "MODEL_PATH" not in globals():
        raise RuntimeError(
            "❌ Model dosya yolu bulunamadı."
        )

    if os.path.basename(MODEL_PATH) != (
        "sign_language_transformer_v2_temiz_devam_eniyi.pth"
    ):
        raise RuntimeError(
            "❌ Eski veya yanlış V2 modeli kullanılıyor.\n"
            "Önce güncel örnek video testi hücresini çalıştır."
        )

    if KAYIT_SURESI < 1 or KAYIT_SURESI > 30:
        raise ValueError(
            "Kayıt süresi 1 ile 30 saniye arasında olmalı."
        )

    if FEATURE_DIM != 219 or MAX_FRAMES != 30:
        raise ValueError(
            "❌ Model giriş boyutları V2 ile uyumlu değil."
        )

    test_model.eval()

    print("✅ Güncel V2 modeli hazır.")
    print("📁 Model:", os.path.basename(MODEL_PATH))
    print("🏷️ Sınıf sayısı:", len(classes_v2))
    print("📐 Özellik sayısı:", FEATURE_DIM)
    print("🎞️ Kare sayısı:", MAX_FRAMES)

    # ================================================
    # 2. DOĞRU AUTSL KELİME LİSTESİ
    # ================================================

    kelime_yollari = [
        "/content/drive/MyDrive/AUTSL_Modelim/"
        "SignList_AUTSL_Dogru.csv",

        "/content/SignList_AUTSL_Dogru.csv"
    ]

    word_path = next(
        (p for p in kelime_yollari if os.path.isfile(p)),
        None
    )

    if word_path is None:
        raise FileNotFoundError(
            "❌ Doğru AUTSL kelime listesi bulunamadı."
        )

    df_words = pd.read_csv(
        word_path,
        encoding="utf-8-sig"
    )

    gerekli_sutunlar = {"ClassId", "TR", "EN"}

    if not gerekli_sutunlar.issubset(df_words.columns):
        raise ValueError(
            "❌ Kelime listesi sütunları hatalı."
        )

    df_words["ClassId"] = pd.to_numeric(
        df_words["ClassId"],
        errors="raise"
    )

    if not (df_words["ClassId"] % 1 == 0).all():
        raise ValueError(
            "❌ Geçersiz sınıf numarası bulundu."
        )

    df_words["ClassId"] = df_words["ClassId"].astype(int)

    if len(df_words) != 226:
        raise ValueError(
            "❌ Kelime listesinde 226 sınıf bulunmalı."
        )

    if not df_words["ClassId"].is_unique:
        raise ValueError(
            "❌ Tekrarlanan sınıf numarası bulundu."
        )

    if set(df_words["ClassId"]) != set(range(226)):
        raise ValueError(
            "❌ AUTSL sınıf numaraları hatalı."
        )

    if (
        df_words["TR"].isna().any()
        or df_words["TR"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "❌ Kelime listesinde boş kelime bulundu."
        )

    if (
        len(classes_v2) != 226
        or set(map(int, classes_v2)) != set(df_words["ClassId"])
    ):
        raise ValueError(
            "❌ Model sınıfları ile kelime listesi uyuşmuyor."
        )

    # Güncel kelime sözlüğünü oluştur.

    id_to_word = dict(
        zip(
            df_words["ClassId"],
            df_words["TR"].astype(str).str.strip()
        )
    )

    if id_to_word.get(222, "").casefold() != "yorgun":
        raise ValueError(
            "❌ Yanlış AUTSL kelime listesi kullanılıyor."
        )

    print("📖 Doğru AUTSL kelime listesi yüklendi.")
    print("✅ Toplam kelime:", len(id_to_word))
    print("✅ 222 numaralı sınıf:", id_to_word[222])

    # ================================================
    # 3. TEST EDİLECEK KELİME
    # ================================================

    def kelime_normalize(kelime):

        kelime = str(kelime).strip().casefold()

        kelime = kelime.replace("ı", "i")
        kelime = kelime.replace("_", " ")

        kelime = unicodedata.normalize(
            "NFKD",
            kelime
        )

        kelime = "".join(
            karakter
            for karakter in kelime
            if not unicodedata.combining(karakter)
        )

        return " ".join(kelime.split())

    gercek_id = None

    if TEST_KELIMESI.strip():

        aranan_kelime = kelime_normalize(
            TEST_KELIMESI
        )

        eslesen_siniflar = [
            class_id
            for class_id, kelime in id_to_word.items()
            if kelime_normalize(kelime) == aranan_kelime
        ]

        if not eslesen_siniflar:
            raise ValueError(
                f"❌ '{TEST_KELIMESI}' kelimesi "
                "AUTSL sözlüğünde bulunamadı.\n"
                "Modelin desteklediği 226 kelimeden "
                "birini seçmelisin."
            )

        if len(eslesen_siniflar) > 1:
            raise ValueError(
                "❌ Birden fazla sınıf eşleşti: "
                f"{eslesen_siniflar}"
            )

        gercek_id = eslesen_siniflar[0]

        print(
            "\n🎯 Test edilecek kelime:",
            id_to_word[gercek_id]
        )

        print("🏷️ Gerçek sınıf ID:", gercek_id)

    else:

        print("\n🎯 Serbest tahmin modu aktif.")
        print("ℹ️ Model en olası 3 kelimeyi gösterecek.")

    print(f"📷 Kamera kayıt süresi: {KAYIT_SURESI} saniye")

    # ================================================
    # 4. KAMERA KAYDI
    # ================================================

    javascript = r"""
    async function recordSign(seconds) {

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: {ideal: 640},
                height: {ideal: 480},
                facingMode: "user"
            },
            audio: false
        });

        const container = document.createElement("div");

        container.style.textAlign = "center";
        container.style.padding = "15px";

        const status = document.createElement("h3");

        status.style.fontFamily = "Arial";
        status.innerText = "Kamera hazırlanıyor...";

        const video = document.createElement("video");

        video.srcObject = stream;
        video.autoplay = true;
        video.muted = true;
        video.playsInline = true;
        video.style.width = "450px";
        video.style.maxWidth = "100%";
        video.style.borderRadius = "12px";

        container.appendChild(status);
        container.appendChild(video);

        document.body.appendChild(container);

        const sleep = ms => new Promise(
            resolve => setTimeout(resolve, ms)
        );

        try {

            await video.play();

            // 3 saniyelik hazırlık süresi.

            for (let i = 3; i > 0; i--) {

                status.innerText = "Hazırlan: " + i;
                status.style.color = "#e67e22";

                await sleep(1000);
            }

            const recorder = new MediaRecorder(stream);
            const chunks = [];

            recorder.ondataavailable = event => {

                if (event.data.size > 0) {
                    chunks.push(event.data);
                }
            };

            const stopped = new Promise((resolve, reject) => {

                recorder.onstop = resolve;
                recorder.onerror = reject;
            });

            recorder.start();

            // Kayıt sırasında geri sayım.

            for (let i = seconds; i > 0; i--) {

                status.innerText =
                    "🔴 KAYIT — Kalan süre: " + i + " sn";

                status.style.color = "#e74c3c";

                await sleep(1000);
            }

            recorder.stop();

            await stopped;

            status.innerText = "Kayıt tamamlandı!";
            status.style.color = "#27ae60";

            const blob = new Blob(chunks, {
                type: recorder.mimeType
            });

            if (blob.size === 0) {
                throw new Error("Kamera kaydı boş.");
            }

            return await new Promise((resolve, reject) => {

                const reader = new FileReader();

                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;

                reader.readAsDataURL(blob);
            });

        } finally {

            stream.getTracks().forEach(
                track => track.stop()
            );

            container.remove();
        }
    }
    """

    display(Javascript(javascript))

    print("\n📷 Kamera izni isteniyor...")
    print("⏳ 3 saniyelik hazırlık süresi başlayacak.")
    print("✋ Kayıt başladığında işaretini yap.")

    result = eval_js(
        f"recordSign({KAYIT_SURESI})"
    )

    # ================================================
    # 5. VİDEOYU KAYDET
    # ================================================

    if not result or "," not in result:
        raise RuntimeError(
            "❌ Kamera kaydı alınamadı."
        )

    video_bytes = base64.b64decode(
        result.split(",", 1)[1]
    )

    if len(video_bytes) == 0:
        raise RuntimeError(
            "❌ Kamera kaydı boş."
        )

    # Her test için ayrı klasör oluştur.
    # Önceki kayıtların üzerine yazılmaz.

    test_dir = tempfile.mkdtemp(
        prefix="v2_kamera_",
        dir="/content"
    )

    webm_path = os.path.join(
        test_dir,
        "kamera.webm"
    )

    mp4_path = os.path.join(
        test_dir,
        "kamera.mp4"
    )

    with open(webm_path, "wb") as f:
        f.write(video_bytes)

    print("✅ Kamera kaydı tamamlandı.")

    # ================================================
    # 6. MP4 DÖNÜŞÜMÜ
    # ================================================

    sonuc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", webm_path,
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            mp4_path
        ],
        capture_output=True,
        text=True
    )

    if sonuc.returncode != 0:
        raise RuntimeError(
            "❌ Video MP4 formatına dönüştürülemedi.\n"
            + sonuc.stderr[-1000:]
        )

    if (
        not os.path.isfile(mp4_path)
        or os.path.getsize(mp4_path) == 0
    ):
        raise FileNotFoundError(
            "❌ Geçerli MP4 video oluşturulamadı."
        )

    print("✅ MP4 dönüşümü tamamlandı.")
    print("📁 Video:", mp4_path)

    # ================================================
    # 7. V2 MODEL TAHMİNİ
    # ================================================

    print("\n🧠 V2 Transformer tahmin yapıyor...")

    v2_tahmin(
        mp4_path,
        gercek_id=gercek_id
    )

    print("\n✅ V2 kamera testi tamamlandı.")
    print("✅ Kamera kapatıldı.")
    print("✅ Önceki kamera kayıtları korundu.")

# %%
# @title 🔍 AUTSL KELİMEYE GÖRE VİDEO BULMA VE OYNATMA

ARANAN_KELIME = "tesekkur"  # @param {type:"string"}
VIDEO_SAYISI = 1  # @param {type:"integer"}
TAM_ESLESME = True  # @param {type:"boolean"}

import os
import pandas as pd
import subprocess
import tempfile
import unicodedata

from IPython.display import Video, display

BASE_DIR = "/content/autsl_data/AUTSL"

# ================================================
# 1. DOĞRU AUTSL KELİME LİSTESİNİ BUL
# ================================================

kelime_yollari = [
    "/content/drive/MyDrive/AUTSL_Modelim/"
    "SignList_AUTSL_Dogru.csv",

    "/content/SignList_AUTSL_Dogru.csv"
]

word_path = next(
    (p for p in kelime_yollari if os.path.isfile(p)),
    None
)

if word_path is None:
    raise FileNotFoundError(
        "❌ Doğru AUTSL kelime listesi bulunamadı."
    )

df_words = pd.read_csv(
    word_path,
    encoding="utf-8-sig"
)

# Gerekli sütunları kontrol et.

gerekli_sutunlar = {"ClassId", "TR", "EN"}

if not gerekli_sutunlar.issubset(df_words.columns):
    raise ValueError(
        "❌ AUTSL kelime listesi sütunları hatalı."
    )

df_words["ClassId"] = pd.to_numeric(
    df_words["ClassId"],
    errors="raise"
)

if not (df_words["ClassId"] % 1 == 0).all():
    raise ValueError(
        "❌ Geçersiz sınıf numarası bulundu."
    )

df_words["ClassId"] = df_words["ClassId"].astype(int)

if len(df_words) != 226:
    raise ValueError(
        "❌ Kelime listesinde 226 sınıf bulunmalı."
    )

if not df_words["ClassId"].is_unique:
    raise ValueError(
        "❌ Tekrarlanan sınıf numarası bulundu."
    )

if set(df_words["ClassId"]) != set(range(226)):
    raise ValueError(
        "❌ AUTSL sınıf numaraları hatalı."
    )

if (
    df_words["TR"].isna().any()
    or df_words["TR"].astype(str).str.strip().eq("").any()
):
    raise ValueError(
        "❌ Kelime listesinde boş kelime bulundu."
    )

id_to_word = dict(
    zip(
        df_words["ClassId"],
        df_words["TR"].astype(str).str.strip()
    )
)

# Önceki hatalı sözlüğün kullanılmasını engelle.

if id_to_word.get(222, "").casefold() != "yorgun":
    raise ValueError(
        "❌ Yanlış AUTSL kelime listesi kullanılıyor."
    )

print("📖 Doğru AUTSL kelime listesi yüklendi.")
print("📁 Sözlük:", word_path)
print("🏷️ Toplam sınıf:", len(df_words))
print("✅ 222 numaralı sınıf:", id_to_word[222])

# ================================================
# 2. TÜRKÇE KARAKTER NORMALİZASYONU
# ================================================

def kelime_normalize(kelime):

    kelime = str(kelime).strip().casefold()

    kelime = kelime.replace("ı", "i")
    kelime = kelime.replace("_", " ")

    kelime = unicodedata.normalize(
        "NFKD",
        kelime
    )

    kelime = "".join(
        karakter
        for karakter in kelime
        if not unicodedata.combining(karakter)
    )

    return " ".join(kelime.split())

# ================================================
# 3. KELİMENİN SINIF NUMARASINI BUL
# ================================================

if VIDEO_SAYISI < 1:
    raise ValueError(
        "Video sayısı en az 1 olmalı."
    )

aranan = kelime_normalize(ARANAN_KELIME)

if not aranan:
    raise ValueError(
        "❌ Aranacak kelime boş bırakılamaz."
    )

kelimeler = df_words["TR"].apply(
    kelime_normalize
)

if TAM_ESLESME:

    eslesen = df_words[
        kelimeler == aranan
    ]

else:

    eslesen = df_words[
        kelimeler.str.contains(
            aranan,
            regex=False,
            na=False
        )
    ]

print("\n🔍 Aranan kelime:", ARANAN_KELIME)
print("📊 Eşleşen sınıf sayısı:", len(eslesen))

# ================================================
# 4. VİDEO OYNATMA FONKSİYONU
# ================================================

def video_oynat(video_path):

    with tempfile.TemporaryDirectory() as temp_dir:

        output_path = os.path.join(
            temp_dir,
            "oynat.mp4"
        )

        sonuc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-i", video_path,
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ],
            capture_output=True,
            text=True
        )

        if sonuc.returncode != 0:

            print("❌ Video dönüştürülemedi.")
            print(sonuc.stderr[-1000:])

            return

        if (
            not os.path.isfile(output_path)
            or os.path.getsize(output_path) == 0
        ):

            print("❌ Geçerli video oluşturulamadı.")

            return

        display(
            Video(
                output_path,
                embed=True,
                width=450,
                mimetype="video/mp4"
            )
        )

# ================================================
# 5. ORİJİNAL AUTSL VİDEOLARINI BUL
# ================================================

if eslesen.empty:

    print("\n❌ Kelime AUTSL veri setinde bulunamadı.")

    print(
        "ℹ️ Model yalnızca AUTSL veri setindeki "
        "226 işareti tanımaktadır."
    )

    print(
        "\n📖 Modelin desteklediği kelimeler:"
    )

    with pd.option_context("display.max_rows", 226):

        display(
            df_words[["ClassId", "TR", "EN"]]
        )

else:

    print("\n✅ Eşleşen kelimeler:")

    display(
        eslesen[["ClassId", "TR", "EN"]]
    )

    csv_path = os.path.join(
        BASE_DIR,
        "train.csv"
    )

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            "❌ AUTSL train.csv dosyası bulunamadı."
        )

    train_dir = os.path.join(
        BASE_DIR,
        "train"
    )

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            "❌ AUTSL train video klasörü bulunamadı."
        )

    df = pd.read_csv(
        csv_path,
        header=None
    )

    df.columns = ["video", "label"]

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    toplam_oynatilan = 0

    # ================================================
    # 6. EŞLEŞEN SINIFLARIN VİDEOLARINI OYNAT
    # ================================================

    for _, row in eslesen.iterrows():

        class_id = int(row["ClassId"])
        kelime = str(row["TR"])
        ingilizce = str(row["EN"])

        videolar = df[
            df["label"] == class_id
        ]

        print("\n" + "=" * 50)
        print("🔍 Türkçe kelime:", kelime)
        print("🌍 İngilizce kelime:", ingilizce)
        print("🏷️ Sınıf ID:", class_id)
        print("📹 Toplam video:", len(videolar))
        print("=" * 50)

        if videolar.empty:

            print("⚠️ Bu sınıfa ait video bulunamadı.")

            continue

        ornekler = videolar.sample(
            n=min(VIDEO_SAYISI, len(videolar)),
            random_state=42
        )

        for sira, (_, video_row) in enumerate(
            ornekler.iterrows(),
            start=1
        ):

            video_adi = os.path.basename(
                str(video_row["video"]).strip()
            )

            if not video_adi.lower().endswith(".mp4"):

                video_adi += ".mp4"

            video_path = os.path.join(
                train_dir,
                video_adi
            )

            print(f"\n🎬 Video {sira}: {video_adi}")
            print("📌 Gerçek etiket:", kelime)
            print("🏷️ Gerçek sınıf ID:", class_id)

            if os.path.isfile(video_path):

                video_oynat(video_path)

                toplam_oynatilan += 1

            else:

                print("⚠️ Video bulunamadı.")

    print("\n" + "=" * 50)
    print("✅ Video incelemesi tamamlandı.")
    print("📹 Oynatma denemesi:", toplam_oynatilan)
    print("=" * 50)

# %%
# @title 🔴 V2 CANLI İŞARET DİLİ TANIMA SİSTEMİ

CANLI_TEST_AKTIF = False  # @param {type:"boolean"}
GUVEN_ESIGI = 0.70  # @param {type:"number"}
HEDEF_FPS = 10  # @param {type:"integer"}
TAHMIN_ARALIGI = 5  # @param {type:"integer"}
NOKTALARI_GOSTER = True  # @param {type:"boolean"}

if not CANLI_TEST_AKTIF:

    print("⏸️ Canlı kamera testi atlandı.")

else:

    import cv2
    import numpy as np
    import torch
    import base64
    import json
    import time

    from collections import deque
    from IPython.display import display, Javascript
    from google.colab.output import eval_js

    # ================================================
    # 1. MODEL KONTROLLERİ
    # ================================================

    gerekli_degiskenler = [
        "test_model",
        "classes_v2",
        "id_to_word",
        "SignLanguagePipelineV2"
    ]

    for degisken in gerekli_degiskenler:

        if degisken not in globals():

            raise RuntimeError(
                f"❌ {degisken} bulunamadı.\n"
                "Önce güncel V2 örnek video testi "
                "hücresini çalıştır."
            )

    if len(classes_v2) != 226:
        raise ValueError("❌ Model sınıf sayısı hatalı.")

    if set(map(int, classes_v2)) != set(range(226)):
        raise ValueError("❌ Model sınıf ID'leri hatalı.")

    if set(id_to_word) != set(range(226)):
        raise ValueError("❌ Kelime sözlüğü eksik.")

    if id_to_word.get(222, "").strip().casefold() != "yorgun":
        raise ValueError("❌ Yanlış AUTSL sözlüğü kullanılıyor.")

    if FEATURE_DIM != 219 or MAX_FRAMES != 30:
        raise ValueError("❌ V2 giriş boyutları hatalı.")

    if not 0 <= GUVEN_ESIGI <= 1:
        raise ValueError("Güven eşiği 0-1 arasında olmalı.")

    if HEDEF_FPS < 1 or TAHMIN_ARALIGI < 1:
        raise ValueError("FPS ve tahmin aralığı pozitif olmalı.")

    device = next(test_model.parameters()).device

    test_model.eval()

    print("✅ V2 modeli hazır.")
    print("✅ Doğru AUTSL sözlüğü hazır.")
    print("🏷️ Sınıf sayısı:", len(classes_v2))
    print("🖥️ Cihaz:", device)

    # ================================================
    # 2. CANLI PİPELİNE VE KAYAN PENCERE
    # ================================================

    live_pipeline = SignLanguagePipelineV2()

    frame_buffer = deque(maxlen=30)
    time_buffer = deque(maxlen=30)

    frame_count = 0

    son_kelime = "Hazırlanıyor..."
    son_guven = 0.0
    son_top3 = []

    # ================================================
    # 3. JAVASCRIPT KAMERA ARAYÜZÜ
    # ================================================

    javascript = r"""
    window.v2Live = {};

    window.v2Live.start = async function() {

        const state = window.v2Live;

        state.running = true;

        const container = document.createElement("div");

        container.style.cssText = `
            max-width: 720px;
            margin: 15px auto;
            padding: 20px;
            background: #111827;
            color: white;
            border-radius: 16px;
            font-family: Arial, sans-serif;
        `;

        container.innerHTML = `
            <h2 style="text-align:center; margin-top:0;">
                🔴 V2 CANLI İŞARET DİLİ TANIMA
            </h2>

            <div style="position:relative; width:100%;">

                <video id="v2-video"
                    autoplay muted playsinline
                    style="
                        width:100%;
                        display:block;
                        border-radius:12px;
                    ">
                </video>

                <canvas id="v2-overlay"
                    style="
                        position:absolute;
                        top:0;
                        left:0;
                        width:100%;
                        height:100%;
                        pointer-events:none;
                    ">
                </canvas>

            </div>

            <div id="v2-status"
                style="
                    text-align:center;
                    color:#94a3b8;
                    margin:12px;
                ">
                Kamera hazırlanıyor...
            </div>

            <div style="
                background:#1f2937;
                border-radius:12px;
                padding:18px;
                text-align:center;
            ">

                <div style="color:#94a3b8;">
                    ALGILANAN İŞARET
                </div>

                <div id="v2-word"
                    style="
                        font-size:36px;
                        font-weight:bold;
                        color:#4ade80;
                        margin:10px;
                    ">
                    Bekleniyor...
                </div>

                <div id="v2-confidence"
                    style="font-size:22px;">
                    %0
                </div>

            </div>

            <h3>En olası 3 tahmin</h3>

            <div id="v2-top3">
                Henüz tahmin yapılmadı.
            </div>

            <button id="v2-stop"
                style="
                    width:100%;
                    margin-top:20px;
                    padding:14px;
                    background:#dc2626;
                    color:white;
                    border:none;
                    border-radius:10px;
                    font-size:17px;
                    cursor:pointer;
                ">
                ⏹ KAMERAYI DURDUR
            </button>
        `;

        document.body.appendChild(container);

        state.container = container;

        state.video = container.querySelector("#v2-video");
        state.overlay = container.querySelector("#v2-overlay");
        state.status = container.querySelector("#v2-status");
        state.word = container.querySelector("#v2-word");
        state.confidence = container.querySelector("#v2-confidence");
        state.top3 = container.querySelector("#v2-top3");

        state.captureCanvas = document.createElement("canvas");

        container.querySelector("#v2-stop").onclick = () => {
            state.stop();
        };

        try {

            state.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: {ideal: 640},
                    height: {ideal: 480},
                    facingMode: "user"
                },
                audio: false
            });

            state.video.srcObject = state.stream;

            await state.video.play();

            state.status.textContent =
                "🟢 Kamera aktif — İşaret yapabilirsin.";

            return true;

        } catch (error) {

            state.stop();
            throw error;
        }
    };

    // ============================================
    // KAMERADAN TEK KARE AL
    // ============================================

    window.v2Live.capture = function() {

        const state = window.v2Live;

        if (!state.running) {
            return null;
        }

        const video = state.video;
        const canvas = state.captureCanvas;

        if (!video.videoWidth || !video.videoHeight) {
            return "";
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");

        ctx.drawImage(video, 0, 0);

        return canvas.toDataURL("image/jpeg", 0.85);
    };

    // ============================================
    // TAHMİN VE NOKTALARI EKRANA YAZ
    // ============================================

    window.v2Live.update = function(data) {

        const state = window.v2Live;

        if (!state.running) {
            return;
        }

        state.word.textContent = data.word;

        state.word.style.color =
            data.accepted ? "#4ade80" : "#fbbf24";

        state.confidence.textContent =
            "%" + data.confidence.toFixed(1);

        state.status.textContent = data.status;

        state.top3.replaceChildren();

        for (const item of data.top3) {

            const row = document.createElement("div");

            row.style.marginBottom = "12px";

            const label = document.createElement("div");

            label.style.marginBottom = "5px";

            label.textContent =
                item.word + " — %" + item.prob.toFixed(1);

            const track = document.createElement("div");

            track.style.cssText = `
                height:10px;
                background:#374151;
                border-radius:8px;
                overflow:hidden;
            `;

            const bar = document.createElement("div");

            bar.style.cssText = `
                height:100%;
                background:#22c55e;
                border-radius:8px;
            `;

            bar.style.width = item.prob + "%";

            track.appendChild(bar);
            row.appendChild(label);
            row.appendChild(track);

            state.top3.appendChild(row);
        }

        // MediaPipe noktalarını çiz.

        const canvas = state.overlay;

        canvas.width = state.video.videoWidth;
        canvas.height = state.video.videoHeight;

        const ctx = canvas.getContext("2d");

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!data.show_points) {
            return;
        }

        function drawPoints(points, color, radius) {

            ctx.fillStyle = color;

            for (const point of points) {

                const x = point[0];
                const y = point[1];

                if (x === 0 && y === 0) {
                    continue;
                }

                ctx.beginPath();

                ctx.arc(
                    x * canvas.width,
                    y * canvas.height,
                    radius,
                    0,
                    Math.PI * 2
                );

                ctx.fill();
            }
        }

        drawPoints(data.left_hand, "#22c55e", 3);
        drawPoints(data.right_hand, "#38bdf8", 3);
        drawPoints(data.pose, "#facc15", 4);
        drawPoints(data.face, "#f472b6", 2);
    };

    // ============================================
    // KAMERAYI KAPAT
    // ============================================

    window.v2Live.stop = function() {

        const state = window.v2Live;

        state.running = false;

        if (state.stream) {

            state.stream.getTracks().forEach(
                track => track.stop()
            );
        }

        if (state.video) {
            state.video.srcObject = null;
        }

        if (state.status) {

            state.status.textContent =
                "⏹ Kamera durduruldu.";
        }

        if (state.overlay) {

            const ctx = state.overlay.getContext("2d");

            ctx.clearRect(
                0, 0,
                state.overlay.width,
                state.overlay.height
            );
        }
    };
    """

    display(Javascript(javascript))

    # ================================================
    # 4. KAYAN PENCERE İLE CANLI TAHMİN
    # ================================================

    print("\n📷 Kamera izni bekleniyor...")

    try:

        eval_js("window.v2Live.start()")

        while True:

            baslangic = time.monotonic()

            # Kameradan görüntü al.

            image_data = eval_js(
                "window.v2Live.capture()"
            )

            if image_data is None:

                print("⏹️ Kamera kullanıcı tarafından durduruldu.")
                break

            if not image_data:

                time.sleep(0.1)
                continue

            image_bytes = base64.b64decode(
                image_data.split(",", 1)[1]
            )

            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            # ========================================
            # 5. MEDIAPIPE ÖZELLİKLERİNİ ÇIKAR
            # ========================================

            landmarks = np.asarray(
                live_pipeline.extract_landmarks(frame),
                dtype=np.float32
            ).reshape(-1)

            if landmarks.shape != (219,):

                raise ValueError(
                    f"❌ Hatalı özellik boyutu: {landmarks.shape}"
                )

            if not np.isfinite(landmarks).all():
                continue

            left_hand = landmarks[0:63].reshape(21, 3)

            right_hand = landmarks[63:126].reshape(21, 3)

            pose = landmarks[126:159].reshape(11, 3)

            face = landmarks[159:219].reshape(20, 3)

            el_algilandi = bool(
                np.any(landmarks[:126] != 0)
            )

            frame_buffer.append(landmarks)
            time_buffer.append(time.monotonic())

            frame_count += 1

            # ========================================
            # 6. TRANSFORMER TAHMİNİ
            # ========================================

            if not el_algilandi:

                son_kelime = "El algılanmadı"
                son_guven = 0.0
                son_top3 = []

            elif len(frame_buffer) < 30:

                son_kelime = "Hazırlanıyor..."
                son_guven = 0.0
                son_top3 = []

            elif frame_count % TAHMIN_ARALIGI == 0:

                sequence = np.stack(
                    list(frame_buffer)
                )

                input_tensor = torch.from_numpy(
                    sequence
                ).unsqueeze(0).to(device)

                with torch.inference_mode():

                    outputs = test_model(input_tensor)

                    probabilities = torch.softmax(
                        outputs,
                        dim=1
                    )

                    top_probs, top_indices = torch.topk(
                        probabilities,
                        k=3,
                        dim=1
                    )

                son_top3 = []

                for prob, idx in zip(
                    top_probs[0],
                    top_indices[0]
                ):

                    class_id = int(
                        classes_v2[idx.item()]
                    )

                    kelime = id_to_word.get(
                        class_id,
                        f"Sınıf {class_id}"
                    )

                    son_top3.append({
                        "word": kelime.replace("_", " "),
                        "prob": float(prob.item() * 100)
                    })

                son_guven = son_top3[0]["prob"]

                if son_guven >= GUVEN_ESIGI * 100:

                    son_kelime = son_top3[0]["word"]

                else:

                    son_kelime = "Belirsiz işaret"

            # ========================================
            # 7. CANLI EKRANI GÜNCELLE
            # ========================================

            pencere_suresi = (
                time_buffer[-1] - time_buffer[0]
                if len(time_buffer) > 1
                else 0.0
            )

            status = (
                f"🟢 Canlı | "
                f"Kare: {len(frame_buffer)}/30 | "
                f"Pencere: {pencere_suresi:.1f} sn"
            )

            ekran_verisi = {

                "word": son_kelime,

                "confidence": son_guven,

                "accepted": (
                    el_algilandi
                    and son_guven >= GUVEN_ESIGI * 100
                ),

                "status": status,

                "top3": son_top3,

                "show_points": NOKTALARI_GOSTER,

                "left_hand": left_hand[:, :2].tolist(),

                "right_hand": right_hand[:, :2].tolist(),

                "pose": pose[:, :2].tolist(),

                "face": face[:, :2].tolist()
            }

            eval_js(
                "window.v2Live.update("
                + json.dumps(ekran_verisi, allow_nan=False)
                + ")"
            )

            # Hedef FPS değerini aşma.

            gecen_sure = time.monotonic() - baslangic

            bekleme = max(
                0,
                (1 / HEDEF_FPS) - gecen_sure
            )

            time.sleep(bekleme)

    except KeyboardInterrupt:

        print("\n⏹️ Canlı test durduruldu.")

    finally:

        try:

            eval_js("window.v2Live.stop()")

        except Exception:

            pass

        for nesne in [
            live_pipeline.hands,
            live_pipeline.pose,
            live_pipeline.face
        ]:

            nesne.close()

        frame_buffer.clear()
        time_buffer.clear()

        print("\n✅ Canlı kamera testi sonlandırıldı.")
        print("✅ Model ağırlıkları değiştirilmedi.")
        print("✅ Kamera videosu dosyaya kaydedilmedi.")

# %%
# @title 💾 V2 Drive Kayıt Durumu Kontrolü

KAYIT_KONTROL_AKTIF = True  # @param {type:"boolean"}
HASH_KONTROLU = False  # @param {type:"boolean"}

if not KAYIT_KONTROL_AKTIF:
    print("⏭️ Drive kayıt kontrolü atlandı.")

else:
    import os
    import json
    import hashlib
    import numpy as np
    import pandas as pd

    DRIVE_ANA = "/content/drive/MyDrive/AUTSL_Modelim"

    V2_TEMIZ = os.path.join(
        DRIVE_ANA,
        "V2_Temiz_Baslangic"
    )

    BEKLENEN_PAKET = 282
    BEKLENEN_VIDEO = 28142
    FEATURE_DIM = 219
    MAX_FRAMES = 30

    print("=" * 55)
    print("☁️ V2 GOOGLE DRIVE KAYIT KONTROLÜ")
    print("=" * 55)

    if not os.path.isdir(DRIVE_ANA):
        raise FileNotFoundError(
            "❌ Model klasörü bulunamadı. "
            "Önce Google Drive'ı bağla."
        )

    # ==================================================
    # 1. HASH HESAPLAMA
    # ==================================================

    def sha256_hesapla(path):

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for blok in iter(
                lambda: f.read(1024 * 1024),
                b""
            ):
                h.update(blok)

        return h.hexdigest()

    # ==================================================
    # 2. MODEL DOSYALARI
    # ==================================================

    print("\n🧠 V2 MODEL DOSYALARI\n")

    model_dosyalari = {
        "sign_language_transformer_v2_temiz.pth":
            "İlk V2 modeli",

        "sign_language_transformer_v2_temiz_devam_eniyi.pth":
            "En başarılı V2 modeli",

        "sign_language_transformer_v2_temiz_devam_checkpoint.pth":
            "Eğitim checkpoint dosyası"
    }

    mevcut_model = 0

    for dosya, aciklama in model_dosyalari.items():

        yol = os.path.join(DRIVE_ANA, dosya)

        if os.path.isfile(yol):

            boyut = os.path.getsize(yol) / (1024 ** 2)

            print(f"✅ {aciklama}")
            print(f"   Dosya: {dosya}")
            print(f"   Boyut: {boyut:.2f} MB")

            mevcut_model += 1

        else:
            print(f"❌ Eksik: {dosya}")

    # ==================================================
    # 3. DOĞRU AUTSL KELİME SÖZLÜĞÜ
    # ==================================================

    print("\n📖 AUTSL KELİME SÖZLÜĞÜ\n")

    sozluk_yolu = os.path.join(
        DRIVE_ANA,
        "SignList_AUTSL_Dogru.csv"
    )

    sozluk_gecerli = False

    if os.path.isfile(sozluk_yolu):

        try:

            df = pd.read_csv(
                sozluk_yolu,
                encoding="utf-8-sig"
            )

            gerekli_sutunlar = {"ClassId", "TR", "EN"}

            if not gerekli_sutunlar.issubset(df.columns):
                raise ValueError("Gerekli sütunlar eksik.")

            ids = pd.to_numeric(
                df["ClassId"],
                errors="raise"
            )

            if (
                len(df) != 226
                or ids.nunique() != 226
                or set(ids) != set(range(226))
            ):
                raise ValueError("Sınıf ID kontrolü başarısız.")

            if (
                df["TR"].isna().any()
                or df["TR"].astype(str).str.strip().eq("").any()
            ):
                raise ValueError("Boş Türkçe kelime bulundu.")

            kelime_222 = str(
                df.loc[ids == 222, "TR"].iloc[0]
            ).strip().lower()

            if kelime_222 != "yorgun":
                raise ValueError(
                    "222 numaralı sınıf yorgun olmalı."
                )

            sozluk_gecerli = True

            print("✅ Doğru AUTSL sözlüğü mevcut.")
            print("🏷️ Sınıf sayısı:", len(df))
            print("🔍 ClassId 222:", kelime_222)

        except Exception as e:
            print("❌ Sözlük kontrol hatası:", e)

    else:
        print("❌ SignList_AUTSL_Dogru.csv bulunamadı.")

    # ==================================================
    # 4. TEMİZ V2 VERİ PAKETLERİ
    # ==================================================

    print("\n📦 TEMİZ V2 VERİ PAKETLERİ\n")

    toplam_video = 0
    basarili_video = 0
    atlanan_video = 0

    gecerli_paketler = []
    hatali_paketler = []

    paketler = []

    if not os.path.isdir(V2_TEMIZ):

        print("❌ V2_Temiz_Baslangic bulunamadı.")

    else:

        paketler = sorted(
            ad for ad in os.listdir(V2_TEMIZ)
            if ad.startswith("paket_")
            and os.path.isdir(
                os.path.join(V2_TEMIZ, ad)
            )
        )

        print("📦 Beklenen paket:", BEKLENEN_PAKET)
        print("📦 Bulunan paket:", len(paketler))
        print("🔍 SHA256 kontrolü:", HASH_KONTROLU)

        for paket in paketler:

            paket_yolu = os.path.join(
                V2_TEMIZ,
                paket
            )

            x_path = os.path.join(paket_yolu, "X.npy")
            y_path = os.path.join(paket_yolu, "y.npy")
            bilgi_path = os.path.join(paket_yolu, "bilgi.json")
            tamamlandi = os.path.join(paket_yolu, "TAMAMLANDI")

            try:

                # Gerekli dosyaları kontrol et.
                for yol in [
                    x_path,
                    y_path,
                    bilgi_path,
                    tamamlandi
                ]:

                    if not os.path.isfile(yol):
                        raise FileNotFoundError(
                            os.path.basename(yol)
                        )

                # Verileri RAM'e tamamen yüklemeden aç.
                X_paket = np.load(
                    x_path,
                    mmap_mode="r",
                    allow_pickle=False
                )

                y_paket = np.load(
                    y_path,
                    mmap_mode="r",
                    allow_pickle=False
                )

                with open(
                    bilgi_path,
                    "r",
                    encoding="utf-8"
                ) as f:
                    bilgi = json.load(f)

                # Özellik boyutlarını doğrula.
                if (
                    X_paket.ndim != 3
                    or X_paket.shape[1:] != (
                        MAX_FRAMES,
                        FEATURE_DIM
                    )
                ):
                    raise ValueError(
                        f"Yanlış X boyutu: {X_paket.shape}"
                    )

                # Etiket boyutlarını doğrula.
                if (
                    y_paket.ndim != 1
                    or len(X_paket) != len(y_paket)
                ):
                    raise ValueError(
                        "X ve y kayıt sayıları uyuşmuyor."
                    )

                if not np.isin(
                    y_paket,
                    np.arange(226)
                ).all():
                    raise ValueError("Geçersiz sınıf ID bulundu.")

                durumlar = bilgi.get("durumlar", [])

                if len(durumlar) != len(X_paket):
                    raise ValueError(
                        "Video durum kayıtları uyuşmuyor."
                    )

                # İsteğe bağlı dosya bütünlüğü kontrolü.
                if HASH_KONTROLU:

                    if (
                        sha256_hesapla(x_path)
                        != bilgi.get("X_sha256")
                    ):
                        raise ValueError("X.npy hash uyuşmazlığı.")

                    if (
                        sha256_hesapla(y_path)
                        != bilgi.get("y_sha256")
                    ):
                        raise ValueError("y.npy hash uyuşmazlığı.")

                paket_basarili = sum(
                    durum == "basarili"
                    for durum in durumlar
                )

                toplam_video += len(X_paket)
                basarili_video += paket_basarili
                atlanan_video += (
                    len(X_paket) - paket_basarili
                )

                gecerli_paketler.append(paket)

                del X_paket, y_paket

            except Exception as e:

                hatali_paketler.append(
                    (paket, str(e))
                )

    # ==================================================
    # 5. VERİ PAKETİ SONUÇLARI
    # ==================================================

    print("\n📊 VERİ PAKETİ SONUÇLARI\n")

    print("📦 Toplam paket:", len(paketler))
    print("✅ Kontrolden geçen:", len(gecerli_paketler))
    print("❌ Hatalı paket:", len(hatali_paketler))

    print("\n🎬 Beklenen video:", BEKLENEN_VIDEO)
    print("🎬 Kontrol edilen video:", toplam_video)
    print("✅ Başarılı işaretli video:", basarili_video)
    print("⚠️ Atlanan video:", atlanan_video)

    print(
        "📐 Özellik boyutu:",
        (MAX_FRAMES, FEATURE_DIM)
    )

    if hatali_paketler:

        print("\n❌ HATALI PAKETLER")

        for paket, hata in hatali_paketler:
            print(f" - {paket}: {hata}")

    # ==================================================
    # 6. GENEL SONUÇ
    # ==================================================

    print("\n" + "=" * 55)
    print("📋 V2 GENEL KAYIT DURUMU")
    print("=" * 55)

    print(f"🧠 Model dosyaları: {mevcut_model}/3")

    print(
        "📖 Kelime sözlüğü:",
        "GEÇERLİ" if sozluk_gecerli else "HATALI"
    )

    print(
        f"📦 Geçerli paketler: "
        f"{len(gecerli_paketler)}/{BEKLENEN_PAKET}"
    )

    print(
        f"🎬 Video kayıtları: "
        f"{toplam_video}/{BEKLENEN_VIDEO}"
    )

    if (
        mevcut_model == 3
        and sozluk_gecerli
        and len(paketler) == BEKLENEN_PAKET
        and len(gecerli_paketler) == BEKLENEN_PAKET
        and toplam_video == BEKLENEN_VIDEO
        and not hatali_paketler
    ):
        print("\n✅ Yapılan bütün kayıt kontrolleri başarılı.")

    else:
        print("\n⚠️ Bazı kayıtların incelenmesi gerekiyor.")

    if not HASH_KONTROLU:
        print("ℹ️ SHA256 kontrolü yapılmadı.")

    print("🔒 Hiçbir Drive dosyası değiştirilmedi.")
    print("=" * 55)

# %%
# @title 📥 V2 GÜVENLİ DRIVE GERİ YÜKLEME

GERI_YUKLEME_AKTIF = False  # @param {type:"boolean"}

if not GERI_YUKLEME_AKTIF:
    print("⏸️ V2 geri yükleme atlandı.")
    print("✅ Mevcut dosyalar korundu.")

else:
    import os
    import shutil
    import hashlib
    import tempfile

    from google.colab import drive

    # ==================================================
    # 1. DRIVE BAĞLANTISI
    # ==================================================

    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")

    DRIVE_DIR = "/content/drive/MyDrive/AUTSL_Modelim"
    COLAB_DIR = "/content"

    V2_TEMIZ_DIR = os.path.join(
        DRIVE_DIR,
        "V2_Temiz_Baslangic"
    )

    if not os.path.isdir(DRIVE_DIR):
        raise FileNotFoundError(
            f"Drive klasörü bulunamadı: {DRIVE_DIR}"
        )

    print("☁️ Google Drive bağlantısı hazır.")

    # ==================================================
    # 2. SHA256 HASH HESAPLAMA
    # ==================================================

    def sha256_hesapla(path):

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for blok in iter(
                lambda: f.read(1024 * 1024),
                b""
            ):
                h.update(blok)

        return h.hexdigest()

    # ==================================================
    # 3. GÜVENLİ DOSYA KOPYALAMA
    # ==================================================

    def guvenli_kopyala(dosya_adi):

        kaynak = os.path.join(
            DRIVE_DIR,
            dosya_adi
        )

        hedef = os.path.join(
            COLAB_DIR,
            dosya_adi
        )

        if not os.path.isfile(kaynak):
            print(f"⚠️ Drive'da bulunamadı: {dosya_adi}")
            return "eksik"

        kaynak_hash = sha256_hesapla(kaynak)

        # Mevcut dosyanın doğruluğunu kontrol et.
        if os.path.exists(hedef):

            if (
                os.path.isfile(hedef)
                and sha256_hesapla(hedef) == kaynak_hash
            ):
                print(f"✅ Zaten mevcut ve doğrulandı: {dosya_adi}")
                return "mevcut"

            print(f"⚠️ Mevcut dosya Drive ile farklı: {dosya_adi}")
            print("🔒 Üzerine yazılmadı.")

            return "farkli"

        # Geçici dosya oluştur.
        fd, gecici = tempfile.mkstemp(
            prefix=dosya_adi + "_",
            suffix=".tmp",
            dir=COLAB_DIR
        )

        os.close(fd)

        try:

            shutil.copy2(kaynak, gecici)

            # Kopyalanan dosyayı doğrula.
            if sha256_hesapla(gecici) != kaynak_hash:
                raise RuntimeError(
                    f"❌ Hash doğrulaması başarısız: {dosya_adi}"
                )

            # Mevcut hedefin üzerine yazmadan taşı.
            os.link(gecici, hedef)

            print(f"✅ Geri yüklendi: {dosya_adi}")

            return "yuklendi"

        finally:

            if os.path.exists(gecici):
                os.remove(gecici)

    # ==================================================
    # 4. GÜNCEL V2 MODEL DOSYALARI
    # ==================================================

    print("\n🧠 V2 MODEL DOSYALARI\n")

    v2_dosyalar = [

        # İlk 20 epoch sonunda kaydedilen model
        "sign_language_transformer_v2_temiz.pth",

        # En başarılı V2 modeli (%89.05)
        "sign_language_transformer_v2_temiz_devam_eniyi.pth",

        # Eğitime devam etmek için checkpoint
        "sign_language_transformer_v2_temiz_devam_checkpoint.pth",

        # Doğru AUTSL kelime sözlüğü
        "SignList_AUTSL_Dogru.csv"
    ]

    sonuclar = {
        "yuklendi": 0,
        "mevcut": 0,
        "eksik": 0,
        "farkli": 0,
        "hata": 0
    }

    for dosya in v2_dosyalar:

        try:
            durum = guvenli_kopyala(dosya)
            sonuclar[durum] += 1

        except Exception as hata:

            sonuclar["hata"] += 1

            print(f"❌ {dosya}: {hata}")

    # ==================================================
    # 5. TEMİZ V2 VERİ PAKETLERİNİ KONTROL ET
    # ==================================================

    print("\n📦 TEMİZ V2 VERİ PAKETLERİ\n")

    if os.path.isdir(V2_TEMIZ_DIR):

        paketler = sorted(
            ad for ad in os.listdir(V2_TEMIZ_DIR)
            if ad.startswith("paket_")
            and os.path.isdir(
                os.path.join(V2_TEMIZ_DIR, ad)
            )
        )

        gerekli_dosyalar = [
            "X.npy",
            "y.npy",
            "bilgi.json",
            "TAMAMLANDI"
        ]

        tamamlanan = []
        eksik_paketler = []

        for paket in paketler:

            paket_yolu = os.path.join(
                V2_TEMIZ_DIR,
                paket
            )

            if all(
                os.path.isfile(
                    os.path.join(paket_yolu, dosya)
                )
                for dosya in gerekli_dosyalar
            ):
                tamamlanan.append(paket)

            else:
                eksik_paketler.append(paket)

        print("📦 Beklenen paket sayısı: 282")
        print("📦 Bulunan paket:", len(paketler))
        print("✅ Dosyaları tam paket:", len(tamamlanan))
        print("❌ Eksik dosyalı paket:", len(eksik_paketler))

        if len(tamamlanan) == 282 and len(paketler) == 282:
            print("✅ Beklenen paket sayısına ulaşıldı.")
        else:
            print("⚠️ Paket sayısının kontrol edilmesi gerekiyor.")

        if eksik_paketler:

            print("\n⚠️ Eksik paketler:")

            for paket in eksik_paketler:
                print(" -", paket)

        print("\n🔒 Veri paketleri Drive üzerinde korundu.")
        print("ℹ️ Paketler Colab'a topluca kopyalanmadı.")
        print("ℹ️ Paket içerikleri ve hash değerleri ayrıca doğrulanmalıdır.")

    else:

        print("❌ V2_Temiz_Baslangic klasörü bulunamadı.")

    # ==================================================
    # 6. GERİ YÜKLEME SONUÇLARI
    # ==================================================

    print("\n" + "=" * 55)
    print("📊 V2 GERİ YÜKLEME SONUÇLARI")
    print("=" * 55)

    print("✅ Yeni yüklenen:", sonuclar["yuklendi"])
    print("🔒 Zaten mevcut:", sonuclar["mevcut"])
    print("⚠️ Eksik dosya:", sonuclar["eksik"])
    print("⚠️ Farklı dosya:", sonuclar["farkli"])
    print("❌ Hatalı işlem:", sonuclar["hata"])

    print("\n📁 Drive dosyaları değiştirilmedi.")

    if (
        sonuclar["eksik"] == 0
        and sonuclar["farkli"] == 0
        and sonuclar["hata"] == 0
    ):
        print("✅ V2 model dosyaları başarıyla doğrulandı.")
    else:
        print("⚠️ Yukarıdaki uyarıları kontrol et.")

    print("=" * 55)
