import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Klasifikasi Buah & Sayur UAS",
    page_icon="🍎",
    layout="centered"
)

# --- FUNGSI LOAD DATA ---
@st.cache_resource
def load_all_files():
    # 1. Load Model (Nama file sesuai file .h5 Anda)
    model = tf.keras.models.load_model('mobilenetv2_fruits360_optimized.h5')
    
    # 2. Load Label (Nama file sesuai file .json Anda)
    with open('klasifikasi class name.json.json', 'r') as f:
        labels = json.load(f)
        
    return model, labels

# Menjalankan pemuatan file
try:
    model, labels = load_all_files()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat file: {e}")
    st.stop()

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("🍎 Klasifikasi Buah & Sayur")
st.write("Aplikasi AI untuk mendeteksi jenis buah/sayur berdasarkan 131 kategori.")
st.divider()

uploaded_file = st.file_uploader("Pilih atau ambil foto gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan Gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Tombol Prediksi
    if st.button('Mulai Analisis Gambar'):
        with st.spinner('Sedang memproses...'):
            # 1. Preprocessing Gambar
            # MobileNetV2 umumnya menggunakan input 224x224
            img = image.convert('RGB')
            img = img.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalisasi

            # 2. Proses Prediksi
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0]) # Mengubah output menjadi probabilitas
            
            class_id = str(np.argmax(predictions[0]))
            confidence = np.max(predictions[0]) * 100
            
            # 3. Ambil Nama Label dari JSON
            # Menggunakan .get untuk menghindari error jika key tidak ada
            nama_produk = labels.get(class_id, "Kategori Tidak Dikenal")

            # 4. Tampilkan Hasil
            st.success(f"### Hasil Prediksi: **{nama_produk}**")
            st.progress(int(confidence))
            st.write(f"Tingkat Keyakinan (Confidence): **{confidence:.2f}%**")

st.divider()
st.caption("Project UAS Klasifikasi Buah - MobileNetV2 Optimized")