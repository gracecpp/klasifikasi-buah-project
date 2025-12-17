import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

st.title("🍎 Klasifikasi Buah UAS")

# Fungsi mengecek keberadaan file
def check_files():
    files = ['mobilenetv2_fruits360_optimized.h5', 'klasifikasi class name.json']
    for f in files:
        if not os.path.exists(f):
            st.error(f"File TIDAK DITEMUKAN: {f}")
            return False
    return True

if check_files():
    try:
        @st.cache_resource
        def load_model():
            return tf.keras.models.load_model('mobilenetv2_fruits360_optimized.h5')

        @st.cache_data
        def load_labels():
            with open('klasifikasi class name.json', 'r') as f:
                return json.load(f)

        model = load_model()
        labels = load_labels()
        st.success("✅ Model & Label Siap!")
    except Exception as e:
        st.error(f"❌ Error saat memuat model: {e}")
else:
    st.info("Pastikan nama file di GitHub sudah benar.")

