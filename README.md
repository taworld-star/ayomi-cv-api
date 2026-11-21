---
title: Ayomi Foto Fokus Anak
sdk: docker
app_port: 7860
---

# Ayomi Foto Fokus Anak

Layanan **computer vision** untuk fitur AI utama Ayomi, **Foto Fokus Anak**. Pengasuh memotret kegiatan di daycare, memilih anak yang dituju, lalu layanan ini membuat **versi fokus**: hanya anak itu yang tajam, sedangkan anak lain, orang dewasa, dan latar diburamkan. Orang Tua hanya menerima versi fokus anaknya sendiri.

Layanan ini **hanya memproses gambar**. Tiga hal lain tetap di aplikasi web Ayomi (Next.js + Supabase):
- memeriksa login dan peran (Pengasuh yang bertugas, Pengelola, atau Admin), jenis dan ukuran berkas, serta batas 30 foto per menit;
- menyimpan foto asli di bucket privat, lalu **menghapusnya setelah versi fokus ditinjau**;
- tinjauan dan persetujuan Pengasuh sebelum foto dikirim ke Orang Tua.

```text
Pengasuh unggah foto ──▶ Aplikasi Ayomi (Vercel + Supabase)
                              │  POST /segment  (Bearer token)
                              ▼
                        Ayomi Foto Fokus  ──▶ daftar orang: kelas anak/dewasa, confidence, siluet
                              │
Pengasuh mengetuk anak yang dituju ──▶ POST /render ──▶ JPEG versi fokus
                              │
Pengasuh meninjau & menyetujui ──▶ versi fokus dikirim ke Orang Tua; foto asli dihapus
```

## Repository terkait

| Repository | Isi |
| --- | --- |
| [Project-Ayomi_Cut_Meutia](https://github.com/taworld-star/Project-Ayomi_Cut_Meutia) | Aplikasi web Ayomi (Next.js + Supabase): halaman, REST API, database, dan dokumentasi. Memanggil layanan ini dari server |
| [ayomi-cv-api](https://github.com/taworld-star/ayomi-cv-api) (repo ini) | Layanan **Foto Fokus Anak** (YOLO11n-seg) |
| [ayomi-ocr-api](https://github.com/taworld-star/ayomi-ocr-api) | Layanan **OCR dokumen legal** (RapidOCR PP-OCRv5) |
| [ayomi-triage-api](https://github.com/taworld-star/ayomi-triage-api) | Layanan **triage urgensi insiden** (mesin aturan rubrik; IndoBERT eksperimen) |

- Website: https://project-ayomi-cut-meutia.vercel.app
- Dokumentasi API aplikasi (Swagger UI): `docs/05-api/swagger.html` di repository aplikasi
- Dokumentasi semua fitur AI: `docs/06-fitur-ai/` di repository aplikasi

## Ringkasan fitur AI

| Poin | Isi |
| --- | --- |
| **Use case** | Computer Vision (segmentasi orang) |
| **Mengapa AI dibutuhkan** | Foto kegiatan hampir selalu berisi beberapa anak. Mengirim foto apa adanya membagikan wajah anak lain tanpa izin orang tuanya, dan memburamkan secara manual tidak mungkin dilakukan pengasuh untuk setiap foto. Segmentasi membuat foto bisa dibagikan tanpa melanggar privasi anak lain |
| **Model** | **YOLO11n-seg** (Ultralytics), pre-trained COCO, di-fine-tune menjadi 2 kelas: `anak` dan `dewasa`. Dijalankan sebagai **ONNX** di CPU |
| **Alasan memilih** | Ukuran nano cukup cepat di CPU server tanpa GPU; menghasilkan **mask segmentasi** (bukan sekadar kotak) sehingga pemburaman mengikuti bentuk tubuh; ekosistem Ultralytics mudah dilatih di Colab dan diekspor ke ONNX |
| **Data** | **Open Images V7** (Google; anotasi CC BY 4.0, gambar CC BY 2.0). Mask berlabel Boy/Girl menjadi `anak`, Man/Woman menjadi `dewasa`. Hanya gambar yang **semua** orangnya punya mask berlabel umur yang dipakai. Train 2.747, validasi 338, uji 300 gambar. Foto suasana daycare: belum ada |
| **Input** | Foto JPG/PNG/WebP ≤ 5 MB (aplikasi membatasi 4 MB); untuk `/render` ditambah siluet anak target |
| **Output** | Daftar orang (kelas, confidence, siluet), anak yang disarankan, penanda kualitas foto; atau JPEG versi fokus |
| **Manusia** | Hasil **wajib ditinjau** Pengasuh (`requires_human_review: true`). Model tidak mengenali identitas siapa pun |

Pelatihan, evaluasi, dan deploy model ada di notebook `docs/06-fitur-ai/notebooks/foto-fokus-anak-yolo11n-seg.ipynb` di repository aplikasi Ayomi.

## Evaluasi

Set uji 300 foto Open Images (1.142 anak) yang tidak dipakai untuk training. Metriknya meniru kejadian di aplikasi, bukan sekadar mAP. Sumbernya `metrics.json` di repo model Hugging Face.

| Metrik | Arti | YOLO11n-seg COCO (tanpa training) | Model Ayomi | Target |
| --- | --- | --- | --- | --- |
| `target_found` | Anak yang siluet prediksinya cocok dengan siluet asli (IoU ≥ 0,5) | 92,7% | 90,8% | ≥ 95% |
| `target_iou` | Rata-rata kerapian siluet anak | 0,80 | 0,78 | ≥ 0,85 |
| `class_acc` | Ketepatan membedakan anak dan dewasa | tidak bisa | **97,5%** | setinggi mungkin |
| `false_people` | Deteksi tanpa pasangan orang asli | 738 | **565** | serendah mungkin |
| `leak_rate` | Foto berisi > 1 orang yang masih menyisakan > 5% tubuh orang lain tajam | 64% | 62% | serendah mungkin |

**Artinya:** fine-tune menambah kemampuan membedakan anak dan dewasa, dan mengurangi deteksi palsu 23%. Kemampuan menemukan dan merapikan siluet anak masih sedikit di bawah model dasar dan belum mencapai target.

## Batasan dan risiko

- **Kebocoran:** orang yang menempel atau bertumpuk dengan anak target bisa ikut tajam. Karena itu `/render` menambah penanda `overlap_other_person`, dan aplikasi mewajibkan tinjauan Pengasuh.
- **Belum ada foto daycare:** model belum dilatih dengan foto suasana daycare berizin (tahap 2 di notebook), sehingga angka di atas belum mewakili kondisi daycare.
- **Bias belum diuji:** warna kulit, cahaya redup, hijab atau penutup kepala, anak tertutup sebagian, dan kerumunan perlu diuji per kelompok setelah ada foto daycare.
- **Usia ditebak dari penampilan:** remaja atau orang dewasa bertubuh kecil bisa salah kelas.
- **Privasi:** foto hanya ada di memori selama satu permintaan. Tidak ada yang disimpan atau dicatat di log. JPEG hasil `/render` tidak membawa metadata EXIF.

## API

Semua endpoint selain `/health` butuh header `Authorization: Bearer <API_TOKEN>`. Error memakai format `{ "detail": { "code": "...", "error": "..." } }`.

### `GET /health`

```json
{ "status": "ok", "model_version": "focus-yolo11n-seg-v1-20261111" }
```

### `POST /segment` (multipart/form-data)

| Field | Isi |
| --- | --- |
| `file` | Foto JPG, PNG, atau WebP, maksimal 5 MB. Orientasi EXIF dirapikan, lalu sisi terpanjang dikecilkan ke 1.600 px |

Contoh respons (polygon dipersingkat):

```json
{
  "image_size": { "width": 1281, "height": 1600 },
  "people": [
    { "id": 0, "class": "anak", "confidence": 0.9138, "polygon": [[1101, 690], [1038, 722], [1001, 772]] }
  ],
  "suggested_target_id": null,
  "quality_flags": ["multiple_children", "blurry"],
  "requires_human_review": true,
  "model_version": "focus-yolo11n-seg-v1-20261111",
  "latency_ms": 222
}
```

- `people` diurutkan dari confidence tertinggi.
- `suggested_target_id` hanya terisi bila ada tepat satu anak dengan confidence ≥ 0,5. Selebihnya Pengasuh mengetuk sendiri anak yang dituju.
- `quality_flags`: `no_person_found`, `low_confidence`, `multiple_children`, `small_image`, `blurry`.

### `POST /render` (multipart/form-data)

| Field | Isi |
| --- | --- |
| `file` | Foto yang sama |
| `target_polygon` | JSON `[[x, y], …]` siluet anak target dalam piksel, 3–2.000 titik |
| `extra_blur_boxes` | Opsional. JSON `[[x, y, w, h], …]`, maksimal 20 kotak tambahan yang ingin diburamkan |

Respons: `image` (JPEG base64), `other_people_blurred` (jumlah orang lain yang diburamkan), dan `quality_flags` (selalu berisi `manual_review_required`; ditambah `overlap_other_person` bila ada orang lain yang menempel).

Cara kerja: latar diburamkan (Gaussian blur), orang lain dipixelate di atas latar yang buram, lalu siluet anak target yang sedikit diperlebar ditempel kembali dengan tepi yang dihaluskan.

### Error

| Status | `code` | Kapan |
| --- | --- | --- |
| 400 | `empty_file`, `invalid_image`, `invalid_target` | Berkas kosong, bukan gambar, atau siluet tidak valid |
| 401 | `unauthorized` | Token API salah |
| 413 | `file_too_large` | Lebih dari 5 MB |
| 415 | `unsupported_format` | Bukan JPG, PNG, atau WebP |

## Environment

| Variabel | Isi |
| --- | --- |
| `MODEL_REPO` | Repo model Hugging Face berisi `model.onnx` dan `config.json` (threshold dan pengaturan render) |
| `HF_READ_TOKEN` | Token baca Hugging Face, bila repo model privat |
| `API_TOKEN` | Token yang wajib dikirim aplikasi web. Nilai yang sama diisi di `CV_API_TOKEN` pada Vercel |

Di aplikasi web: `CV_API_URL` (alamat layanan ini) dan `CV_API_TOKEN`.

## Menjalankan di laptop

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
set MODEL_REPO=<user>/ayomi-focus-segmenter
set HF_READ_TOKEN=<token baca>
set API_TOKEN=<token acak>
.venv/Scripts/uvicorn app:app --port 7860
```

Lalu isi `CV_API_URL=http://127.0.0.1:7860` dan `CV_API_TOKEN` di `.env.local` aplikasi web.

## Deploy

Layanan dikemas sebagai container Docker (`Dockerfile`, port 7860). Header README ini adalah konfigurasi **Hugging Face Space** bertipe Docker (notebook Sel 18–19 membuat Space ini). Container yang sama juga bisa dijalankan di server Docker lain, misalnya Dokploy. Setelah deploy, isi `CV_API_URL` dan `CV_API_TOKEN` di Environment Variables Vercel.

## Struktur

| Berkas | Isi |
| --- | --- |
| `app.py` | API FastAPI: validasi berkas, segmentasi, render versi fokus, penanda kualitas |
| `requirements.txt` | Dependensi (Ultralytics, ONNX Runtime, FastAPI) |
| `Dockerfile` | Image CPU dengan PyTorch CPU |

## Lisensi

YOLO11 dan Ultralytics berlisensi **AGPL-3.0**, sehingga kode layanan ini dibuka bersama repository-nya. Anotasi Open Images berlisensi CC BY 4.0.
