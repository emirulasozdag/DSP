# FaceTransform DSP — Kurulum, Çalıştırma ve Kullanım Rehberi

## Proje Nedir?

**Facial Image Warping, Aging & Expression Transformation** — CENG 384 DSP dersi projesi.  
Yüz fotoğrafı yükleyip, yüz ifadesi değiştirme (gülümseme, kaş kaldırma, dudak genişletme, yüz inceltme), yaşlandırma/gençleştirme simülasyonu ve frekans analizi yapan bir web uygulaması.

---

## Modül Durumu

| Modül | Durum | Açıklama |
|-------|-------|----------|
| Image Upload & Preprocessing (FR-3.1) | ✅ Çalışıyor | Yükleme, Haar face detection, crop, resize 512×512 |
| Facial Landmark Detection (FR-3.2) | ✅ Çalışıyor | MediaPipe Tasks API, 478 nokta tespit |
| Expression Manipulation (FR-3.3) | ✅ Çalışıyor | Delaunay triangulation + affine warp + Gaussian blending |
| Aging / De-Aging (FR-3.4) | ✅ Çalışıyor | Per-channel FFT, Butterworth filtre, CLAHE, bilateral |
| Frequency Analysis (FR-3.5) | ✅ Çalışıyor | 2D FFT, magnitude spectrum, enerji dağılımı |
| Metrics (FR-3.6) | ✅ Çalışıyor | MSE, PSNR, SSIM hesaplama + CSV export |
| Web UI (FR-3.7) | ✅ Çalışıyor | Flask + sidebar dark theme, 6 sayfa |

---

## Kullanılan Teknolojiler

| Kütüphane | Versiyon | Ne İşe Yarar |
|-----------|----------|--------------|
| Flask | 3.x | Web sunucusu (backend) |
| OpenCV (opencv-python) | 4.x | Yüz tespiti (Haar Cascade), warp, Delaunay, seamless blend |
| MediaPipe | 0.10.x | 478 nokta yüz landmark tespiti (Tasks API) |
| NumPy | 2.x | Sayısal hesaplamalar, matris işlemleri |
| SciPy | 1.x | 2D FFT, Butterworth filtre (aging/de-aging) |
| scikit-image | 0.x | SSIM metrik hesaplama |
| Pillow | 12.x | Görüntü format desteği |

### Mimari
```
app.py              → Flask backend + tüm DSP işleme fonksiyonları
templates/index.html → Ana HTML sayfası (sidebar layout, 6 sekme)
static/css/style.css → Dark theme CSS
static/js/app.js     → Frontend JavaScript (API çağrıları, UI logic)
face_landmarker.task  → MediaPipe model dosyası (~4MB)
uploads/              → Yüklenen dosyalar (geçici)
results/              → İşlenmiş sonuçlar (geçici)
```

### DSP Pipeline Detayları

**Expression Warping (FR-3.3):**
1. MediaPipe ile 478 landmark tespit
2. İfade tipine göre hedef noktaları hesapla (smile → ağız köşeleri yukarı vb.)
3. `cv2.Subdiv2D` ile Delaunay üçgenleme
4. Her üçgen için `cv2.getAffineTransform` + `cv2.warpAffine`
5. Convex hull mask + Gaussian blur ile seamless blending

**Aging (FR-3.4):**
1. Her renk kanalı için ayrı 2D FFT (`scipy.fft.fft2`)
2. Butterworth high-pass filtre (cutoff = max_radius × 0.15)
3. Yüksek frekansları güçlendir (kırışıklık efekti)
4. Inverse FFT ile geri dönüştür
5. CLAHE kontrast güçlendirme + kenar bazlı wrinkle noise
6. Ton ayarı (mavi azalt, kırmızı artır)

**De-Aging (FR-3.4):**
1. Per-channel 2D FFT
2. Butterworth low-pass filtre
3. Inverse FFT
4. `cv2.bilateralFilter` ile edge-preserving yumuşatma
5. Parlaklık/kontrast ince ayar

---

## Gereksinimler

- **Python 3.10+** (3.12 önerilir)
- **pip** (Python ile birlikte gelir)
- **Web tarayıcı** (Chrome, Firefox, Edge)
- İnternet bağlantısı (ilk kurulum ve font/ikon CDN için)

---

## Kurulum (Adım Adım)

### 1. Python Kontrolü

Terminali (PowerShell veya CMD) aç ve Python'un yüklü olduğunu kontrol et:

```
python --version
```

`Python 3.10.x` veya üstü çıkmalı. Yoksa [python.org](https://www.python.org/downloads/) adresinden indir.

### 2. Proje Klasörüne Git

```
cd C:\Users\user\Documents\GitHub\DSP
```

### 3. Bağımlılıkları Yükle

```
pip install flask opencv-python mediapipe numpy scipy scikit-image Pillow
```

Bu komut şu kütüphaneleri yükler:
| Kütüphane | Ne İşe Yarar |
|-----------|-------------|
| `flask` | Web sunucusu (backend) |
| `opencv-python` | Görüntü işleme |
| `mediapipe` | Yüz landmark tespiti (468 nokta) |
| `numpy` | Sayısal hesaplamalar |
| `scipy` | FFT (Fourier dönüşümü) |
| `scikit-image` | SSIM metrik hesaplama |
| `Pillow` | Görüntü format desteği |

### 4. MediaPipe Model Dosyası

Proje klasöründe `face_landmarker.task` dosyası olmalı. Yoksa indir:

```
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task', 'face_landmarker.task'); print('İndirildi!')"
```

### 5. Sunucuyu Başlat

```
python app.py
```

Şu çıktıyı görmelisin:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### 6. Tarayıcıda Aç

**ÖNEMLİ:** Tarayıcıya şu adresi yaz:

```
http://127.0.0.1:5000
```

> ⚠️ **DİKKAT:** `templates/index.html` dosyasını doğrudan tarayıcıda AÇMA!  
> Dosyayı direkt açarsan CSS ve JavaScript yüklenmez, sayfa düz beyaz görünür.  
> **Her zaman** `http://127.0.0.1:5000` adresinden aç.

---

## Kullanım

### Tab 1: Image Upload
1. **Choose File** butonuna tıkla veya sürükle-bırak ile yüz fotoğrafı yükle
2. Sistem otomatik olarak: yüzü tespit eder, kırpar, 512×512'ye yeniden boyutlandırır
3. Yükleme başarılı olunca **Proceed to Landmarks** butonuna tıkla

### Tab 2: Landmarks
- 478 noktalık yüz mesh'ini görselleştirir
- Toggle ile landmark görünürlüğünü aç/kapat
- **Export Coordinates** ile CSV olarak indir

### Tab 3: Expression
- **Smile Enhancement:** Gülümseme efekti (ağız köşeleri yukarı)
- **Eyebrow Raising:** Kaş kaldırma
- **Lip Widening:** Dudak genişletme
- **Face Slimming:** Yüz inceltme
- Her birinde intensity slider'ı ile şiddeti ayarla

### Tab 4: Aging / De-Aging
- **Aging:** FFT ile yüksek frekans güçlendirme → kırışıklık efekti
- **De-Aging:** Alçak geçiren filtre → cilt yumuşatma
- Frekans domain'inde işlem yapıyor (gerçek DSP!)

### Tab 5: Frequency Analysis
- 2D Fourier Transform spektrumu
- Orijinal vs dönüştürülmüş karşılaştırma
- Toplam enerji, düşük/yüksek frekans oranları

### Tab 6: Metrics
- MSE, PSNR, SSIM değerleri tablo halinde
- CSV veya PDF olarak export et

---

## Proje Dosya Yapısı

```
DSP/
├── app.py                  ← Flask backend (tüm image processing burada)
├── face_landmarker.task    ← MediaPipe model dosyası
├── requirements.txt        ← Bağımlılık listesi
├── templates/
│   └── index.html          ← Ana sayfa HTML
├── static/
│   ├── css/
│   │   └── style.css       ← Arayüz stilleri (dark theme)
│   └── js/
│       └── app.js          ← Frontend JavaScript
├── uploads/                ← Yüklenen dosyalar (otomatik oluşur)
└── results/                ← Kaydedilen sonuçlar (otomatik oluşur)
```

---

## Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| CSS yüklenmiyor, sayfa düz beyaz | `http://127.0.0.1:5000` adresinden aç, dosyayı direkt açma |
| `ModuleNotFoundError` | `pip install flask opencv-python mediapipe numpy scipy scikit-image Pillow` çalıştır |
| `face_landmarker.task` bulunamadı | Adım 4'teki komutu çalıştır |
| Port 5000 kullanımda | `app.py`'da son satırı `app.run(debug=True, port=5001)` yap |
| Yüz tespit edilmiyor | Düz, cepheden çekilmiş bir fotoğraf kullan |
| Sunucu çöktü | Terminal'de `python app.py` ile tekrar başlat |

---

## Sunucuyu Durdurma

Terminal'de `Ctrl + C` bas.

---

## Presentation İçin İpuçları

1. Sunucuyu önceden başlat, tarayıcıyı hazır tut
2. Güzel, net bir portre fotoğrafı önceden hazırla
3. Her tab'ı sırayla göster — upload → landmarks → expression → aging → frequency → metrics
4. Slider'ları canlı olarak ayarla, sonuçları göster
5. CSV export'u canlı göster
6. Frekans spektrumunu açıkla (aging = high-freq boost, de-aging = low-pass)
