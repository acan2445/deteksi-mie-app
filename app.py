"""
GUI Deteksi Kandungan Gizi Mie (YOLOv8n + EasyOCR)
Jalankan dengan:  streamlit run app.py
"""
import hashlib
import io
import os
import re

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from thefuzz import fuzz
from ultralytics import YOLO

# ==========================================================================
# 1. ATURAN GIZI (disalin dari notebook kamu, bagian 6 & 7)
# ==========================================================================
BATAS_KLAIM_GIZI = {
    "Lemak Jenuh": [("Bebas", 0.1), ("Rendah", 1.5)],
    "Natrium": [("Bebas", 0.005), ("Sangat Rendah", 0.04), ("Rendah", 0.12)],
}

SARAN_KLAIM_GIZI = {
    "Lemak Jenuh": {
        "Bebas": "Aman dikonsumsi dari sisi lemak jenuh.",
        "Rendah": "Kandungan lemak jenuh rendah, masih tergolong wajar.",
        "Tinggi": "Kandungan lemak jenuh tinggi. Sebaiknya batasi konsumsi, terutama bila punya risiko kolesterol atau penyakit jantung.",
    },
    "Natrium": {
        "Bebas": "Aman dikonsumsi dari sisi natrium/garam.",
        "Sangat Rendah": "Kandungan natrium sangat rendah, masih tergolong wajar.",
        "Rendah": "Kandungan natrium rendah, masih tergolong wajar.",
        "Tinggi": "Kandungan natrium tinggi. Sebaiknya batasi konsumsi, terutama bagi penderita hipertensi atau yang sedang jaga asupan garam.",
    },
}

KLAIM_AMAN = {"Bebas", "Sangat Rendah", "Rendah"}

KATA_KUNCI = {
    "Takaran Saji": ["takaran saji"],
    "Lemak Jenuh": ["lemak jenuh"],
    "Natrium": ["natrium", "garam"],
}
JENIS = {"Takaran Saji": "takaran", "Lemak Jenuh": "lemak", "Natrium": "natrium"}


def klasifikasikan_klaim_gizi(nama_komponen, nilai_gram_per_100g):
    aturan = BATAS_KLAIM_GIZI.get(nama_komponen)
    if aturan is None or nilai_gram_per_100g is None:
        return "Tidak dapat diklasifikasi"
    for label_klaim, batas_maks in aturan:
        if nilai_gram_per_100g <= batas_maks:
            return label_klaim
    return "Tinggi"


def hitung_klaim(komponen, nilai_mentah, takaran):
    """Nilai per saji -> per 100 g -> kategori klaim."""
    if nilai_mentah is None or not takaran:
        return None, "Tidak terbaca"
    gram_per_saji = nilai_mentah / 1000.0 if komponen == "Natrium" else nilai_mentah
    per_100g = gram_per_saji / takaran * 100.0
    return per_100g, klasifikasikan_klaim_gizi(komponen, per_100g)


def tentukan_status(klaim_lemak, klaim_natrium):
    if "Tinggi" in (klaim_lemak, klaim_natrium):
        return "Berisiko"
    if klaim_lemak in KLAIM_AMAN and klaim_natrium in KLAIM_AMAN:
        return "Aman"
    return "Tidak terbaca"


# ==========================================================================
# 2. FUNGSI OCR (disalin dari notebook kamu)
# ==========================================================================
def ekstrak_angka_murni(teks_list, jenis="lemak"):
    teks_gabungan = " ".join([str(t) for t in teks_list])

    if jenis == "takaran":
        match = re.search(r"\(([^)]+)\)", teks_gabungan)
        kandidat_str = match.group(1) if match else teks_gabungan
    else:
        kandidat_str = teks_gabungan

    kandidat_str = (
        kandidat_str.replace(",", ".").replace("ô", "6").replace("O", "0").replace("o", "0")
    )

    kandidat_angka = []
    for t in kandidat_str.split():
        t_clean = re.sub(r"[a-zA-Z]", "", t)
        angka_match = re.search(r"\d+[.]?\d*", t_clean)
        if angka_match:
            kandidat_angka.append(angka_match.group(0))

    if kandidat_angka:
        if len(kandidat_angka) > 1 and kandidat_angka[-1] == "9":
            val = kandidat_angka[-2]
        else:
            val = kandidat_angka[-1]

        if val.endswith("9"):
            if "." in val:
                val = val[:-1]
            elif jenis == "takaran" and float(val) > 200:
                val = val[:-1]
            elif jenis == "lemak" and float(val) > 50:
                val = val[:-1]
            elif jenis == "natrium" and float(val) > 5000:
                val = val[:-1]
        return float(val)
    return None


def kelompokkan_baris(hasil_ocr):
    """Gabungkan potongan teks EasyOCR yang sejajar jadi satu baris tabel."""
    if not hasil_ocr:
        return []
    item = []
    for bbox, teks, _conf in hasil_ocr:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        item.append({"y": sum(ys) / len(ys), "x": min(xs), "tinggi": max(ys) - min(ys), "teks": teks})
    item.sort(key=lambda d: d["y"])
    toleransi = 0.6 * float(np.median([d["tinggi"] for d in item]))

    kelompok, sekarang = [], [item[0]]
    for d in item[1:]:
        if abs(d["y"] - np.mean([s["y"] for s in sekarang])) <= toleransi:
            sekarang.append(d)
        else:
            kelompok.append(sekarang)
            sekarang = [d]
    kelompok.append(sekarang)
    return [" ".join(s["teks"] for s in sorted(k, key=lambda d: d["x"])) for k in kelompok]


def cari_nilai_di_baris(daftar_baris, komponen):
    """Cadangan kalau model YOLO cuma punya 1 kelas 'tabel gizi'."""
    baris_terbaik, skor_terbaik = None, 0
    for b in daftar_baris:
        for kunci in KATA_KUNCI[komponen]:
            skor = fuzz.partial_ratio(kunci, b.lower())
            if skor > skor_terbaik:
                baris_terbaik, skor_terbaik = b, skor
    if baris_terbaik is None or skor_terbaik < 80:
        return None
    teks = re.sub(r"[\d.,]+\s*%", " ", baris_terbaik)
    return ekstrak_angka_murni([teks], jenis=JENIS[komponen])


# ==========================================================================
# 3. PIPELINE: YOLO -> crop -> EasyOCR
# ==========================================================================
def nama_ke_komponen(nama_kelas):
    n = nama_kelas.lower()
    if "takaran" in n:
        return "Takaran Saji"
    if "lemak" in n:
        return "Lemak Jenuh"
    if "natrium" in n:
        return "Natrium"
    return None


def proses_gambar(img_bgr, model, reader, conf):
    nilai = {"Takaran Saji": None, "Lemak Jenuh": None, "Natrium": None}
    detail = []

    pred = model.predict(source=img_bgr, conf=conf, verbose=False)[0]
    gambar_box = cv2.cvtColor(pred.plot(), cv2.COLOR_BGR2RGB)
    if len(pred.boxes) == 0:
        return nilai, detail, gambar_box

    H, W = img_bgr.shape[:2]
    # Box dengan confidence tertinggi diproses duluan
    urutan = sorted(range(len(pred.boxes)), key=lambda i: -float(pred.boxes[i].conf[0]))
    for i in urutan:
        box = pred.boxes[i]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        komponen = nama_ke_komponen(pred.names[int(box.cls[0])])
        if komponen and nilai[komponen] is not None:
            continue  # sudah terisi dari box yang lebih yakin

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        zoom = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        baris = kelompokkan_baris(reader.readtext(zoom, mag_ratio=1.5))

        if komponen:
            nilai[komponen] = ekstrak_angka_murni(baris, jenis=JENIS[komponen])
        else:  # kelas tunggal "tabel gizi" -> cari per baris
            for k in nilai:
                if nilai[k] is None:
                    nilai[k] = cari_nilai_di_baris(baris, k)

        detail.append({
            "komponen": komponen or "Tabel gizi",
            "conf": float(box.conf[0]),
            "crop": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            "teks": baris,
        })
    return nilai, detail, gambar_box


# ==========================================================================
# 4. LOAD MODEL (di-cache supaya tidak load ulang tiap klik)
# ==========================================================================
@st.cache_resource(show_spinner="Memuat model YOLO...")
def muat_yolo(path):
    return YOLO(path)


@st.cache_resource(show_spinner="Memuat EasyOCR (pertama kali agak lama)...")
def muat_ocr():
    return easyocr.Reader(["id", "en"], gpu=torch.cuda.is_available())


def baca_gambar(file_bytes, max_sisi=1600):
    """Bytes -> array BGR. exif_transpose supaya foto HP tidak miring.
    Foto yang terlalu besar diperkecil supaya OCR di server (CPU) tidak lambat."""
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")
    if max_sisi and max(pil.size) > max_sisi:
        pil.thumbnail((max_sisi, max_sisi), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ==========================================================================
# 5. TAMPILAN (GUI)
# ==========================================================================
st.set_page_config(page_title="Deteksi Gizi Mie", page_icon="🍜", layout="wide")
st.title("🍜 Deteksi Kandungan Gizi Mie")
st.caption("YOLOv8n mencari lokasi info gizi → EasyOCR membaca angkanya → dihitung per 100 g → diklasifikasi Aman / Berisiko")

with st.sidebar:
    st.header("Pengaturan")
    path_model = st.text_input("Lokasi file model (.pt)", value="best.pt")
    conf = st.slider("Confidence YOLO", 0.1, 0.95, 0.5, 0.05,
                     help="Makin tinggi = makin ketat, kotak salah berkurang tapi tabel yang samar bisa terlewat.")
    max_sisi = st.select_slider(
        "Ukuran maks. gambar (px)", options=[800, 1200, 1600, 2400, 4000], value=1600,
        help="Foto HP biasanya sangat besar. Diperkecil biar proses lebih cepat. Kalau angka sering salah baca, coba naikkan.")
    st.divider()
    st.markdown(
        "**Aturan status**\n"
        "- **Berisiko**: lemak jenuh *atau* natrium kategori *Tinggi*\n"
        "- **Aman**: keduanya tidak Tinggi\n"
        "- **Tidak terbaca**: angka gagal dibaca"
    )

if not os.path.exists(path_model):
    st.error(f"File model tidak ditemukan: `{path_model}`. Taruh `best.pt` di folder yang sama dengan `app.py`, atau ubah path di sidebar.")
    st.stop()

model = muat_yolo(path_model)
reader = muat_ocr()

tab_upload, tab_kamera = st.tabs(["📱 Foto / Upload (disarankan)", "📷 Kamera browser"])
with tab_upload:
    st.caption(
        "💡 **Di HP:** tekan tombol di bawah lalu pilih **Kamera / Take Photo**. "
        "Yang terbuka adalah kamera bawaan HP (bisa tap untuk fokus, zoom, flash, resolusi penuh). "
        "Di laptop: pilih file foto seperti biasa."
    )
    f_upload = st.file_uploader("Pilih / foto kemasan mie", type=["jpg", "jpeg", "png"])
with tab_kamera:
    st.caption("Kamera bawaan browser: resolusi lebih rendah dan fokus tidak bisa diatur. Untuk hasil OCR terbaik pakai upload gambar.")
    f_kamera = st.camera_input("Foto bagian tabel informasi nilai gizi")

berkas = f_upload or f_kamera
if berkas is None:
    st.info("Upload atau foto kemasan mie dulu ya.")
    st.stop()

data = berkas.getvalue()
id_gambar = hashlib.md5(data).hexdigest()[:10]

# Cache hasil per gambar + conf, karena Streamlit menjalankan ulang script tiap ada input berubah
kunci_cache = f"hasil_{id_gambar}_{conf}_{max_sisi}"
if kunci_cache not in st.session_state:
    with st.spinner("Mendeteksi tabel dan membaca teks..."):
        img_bgr = baca_gambar(data, max_sisi)
        st.session_state[kunci_cache] = proses_gambar(img_bgr, model, reader, conf)
nilai, detail, gambar_box = st.session_state[kunci_cache]

kol_kiri, kol_kanan = st.columns([1, 1], gap="large")

with kol_kiri:
    st.subheader("1. Hasil deteksi YOLO")
    st.image(gambar_box, use_container_width=True)
    if not detail:
        st.warning("Tabel gizi tidak terdeteksi. Coba foto lebih dekat / lebih terang, atau turunkan confidence di sidebar.")
    for d in detail:
        with st.expander(f"{d['komponen']}  (conf {d['conf']:.2f})"):
            st.image(d["crop"], use_container_width=True)
            st.code("\n".join(d["teks"]) or "(tidak ada teks)")

with kol_kanan:
    st.subheader("2. Angka hasil OCR (boleh dikoreksi)")
    st.caption("OCR kadang salah baca. Cocokkan dengan foto, ubah kalau perlu, hasil di bawah otomatis menyesuaikan.")

    takaran = st.number_input("Takaran saji (g)", min_value=0.0, value=nilai["Takaran Saji"],
                              step=1.0, key=f"tak_{id_gambar}_{conf}_{max_sisi}", placeholder="tidak terbaca")
    lemak = st.number_input("Lemak jenuh per saji (g)", min_value=0.0, value=nilai["Lemak Jenuh"],
                            step=0.5, key=f"lem_{id_gambar}_{conf}_{max_sisi}", placeholder="tidak terbaca")
    natrium = st.number_input("Natrium per saji (mg)", min_value=0.0, value=nilai["Natrium"],
                              step=10.0, key=f"nat_{id_gambar}_{conf}_{max_sisi}", placeholder="tidak terbaca")

    st.subheader("3. Hasil analisis")
    per100_lemak, klaim_lemak = hitung_klaim("Lemak Jenuh", lemak, takaran)
    per100_natrium, klaim_natrium = hitung_klaim("Natrium", natrium, takaran)
    status = tentukan_status(klaim_lemak, klaim_natrium)

    if status == "Berisiko":
        st.error("## ⚠️ BERISIKO")
    elif status == "Aman":
        st.success("## ✅ AMAN")
    else:
        st.warning("## ❔ TIDAK TERBACA — isi angka yang kosong di atas")

    tabel = pd.DataFrame([
        {
            "Komponen": "Lemak Jenuh",
            "Per saji": f"{lemak:g} g" if lemak is not None else "-",
            "Per 100 g": f"{per100_lemak:.3f} g" if per100_lemak is not None else "-",
            "Kategori": klaim_lemak,
        },
        {
            "Komponen": "Natrium",
            "Per saji": f"{natrium:g} mg" if natrium is not None else "-",
            "Per 100 g": f"{per100_natrium * 1000:.1f} mg" if per100_natrium is not None else "-",
            "Kategori": klaim_natrium,
        },
    ])
    st.dataframe(tabel, hide_index=True, use_container_width=True)

    for nama, klaim in [("Lemak Jenuh", klaim_lemak), ("Natrium", klaim_natrium)]:
        saran = SARAN_KLAIM_GIZI[nama].get(klaim)
        if saran:
            st.write(f"**{nama}:** {saran}")

st.divider()
st.caption("Hasil bersifat informatif berdasarkan ambang batas klaim gizi per 100 g, bukan pengganti saran medis.")
