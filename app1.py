import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import numpy as np

# --- 1. SET KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Sentimen Vacuum Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS UNTUK UI MODERN ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .custom-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #4F46E5;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E1B4B;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_transform=True)

# --- 3. LOAD MODEL DAN DATASET (CACHED) ---
@st.cache_resource
def load_models():
    # Menyesuaikan dengan nama file asli hasil upload bray
    tfidf_path = "tfidf_model.pkl"
    knn_path = "knn_model.pkl"
    
    # Cek keberadaan file agar tidak blank/crash mendadak
    if not os.path.exists(tfidf_path) or not os.path.exists(knn_path):
        return None, None, f"File model '{tfidf_path}' atau '{knn_path}' tidak ditemukan di direktori project."
        
    try:
        with open(tfidf_path, 'rb') as f_tfidf:
            tfidf = pickle.load(f_tfidf)
        with open(knn_path, 'rb') as f_knn:
            knn = pickle.load(f_knn)
        return tfidf, knn, None
    except Exception as e:
        return None, None, str(e)

# Panggil fungsi load model
tfidf_model, knn_model, error_msg = load_models()

# Tampilkan pesan jika model gagal dimuat
if error_msg:
    st.error(f"❌ **Gagal Memuat Model:** {error_msg}")
    st.info("💡 Pastikan file `.pkl` berada di folder yang sama dengan file `app1.py` ini.")

# --- 4. HEADER APLIKASI ---
st.markdown('<div class="main-title">🧹 Aplikasi Analisis Sentimen Vacuum Cleaner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Klasifikasi Sentimen Ulasan Menggunakan Metode K-Nearest Neighbors (KNN)</div>', unsafe_allow_html=True)

# --- 5. TABS INTERFACE ---
tab1, tab2 = st.tabs(["💬 Analisis Ulasan Tunggal", "📊 Analisis Masal (Upload File CSV)"])

# ==================== TAB 1: ULASAN TUNGGAL ====================
with tab1:
    st.markdown('<div class="custom-card"><h3>Input Ulasan Baru</h3></div>', unsafe_allow_html=True)
    
    teks_input = st.text_area(
        "Masukkan teks ulasan vacuum cleaner di bawah ini:",
        placeholder="Contoh: Barangnya bagus banget, dayanya kuat dan cepat membersihkan debu...",
        height=100
    )
    
    if st.button("🚀 Analisis Sentimen", key="btn_tunggal"):
        if not teks_input.strip():
            st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu.")
        elif tfidf_model is None or knn_model is None:
            st.error("❌ Analisis tidak dapat dilakukan karena model gagal dimuat.")
        else:
            with st.spinner("Sedang memproses..."):
                try:
                    # 1. Transformasi Teks memakai TF-IDF Vectorizer yang di-load
                    vektor_teks = tfidf_model.transform([teks_input.lower()])
                    
                    # 2. Prediksi Kelas menggunakan Model KNN
                    prediksi_kelas = knn_model.predict(vektor_teks)[0]
                    
                    # Mapping label angka ke teks string sentimen
                    # (Sesuaikan mapping 0/1 ini dengan label asli dataset skripsimu)
                    status_sentimen = "Positif" if prediksi_kelas == 1 else "Negatif"
                    warna_box = "green" if status_sentimen == "Positif" else "red"
                    
                    st.markdown(f"""
                        <div style='padding: 20px; background-color: rgba(0,0,0,0.02); border-radius: 8px; border-left: 5px solid {warna_box};'>
                            <h4>Hasil Analisis:</h4>
                            <p style='font-size: 1.3rem; font-weight: bold; color: {warna_box};'>Sentimen {status_sentimen} (Kelas: {prediksi_kelas})</p>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat pemrosesan prediksi: {e}")

# ==================== TAB 2: ULAASAN MASAL (CSV) ====================
with tab2:
    st.markdown('<div class="custom-card"><h3>Upload File Dataset Ulasan</h3></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file CSV hasil export ulasan", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File berhasil diunggah!")
            
            # Tampilkan pilihan kolom teks ulasan secara dinamis
            kolom_pilihan = st.selectbox("Pilih kolom yang berisi Teks Ulasan:", df.columns)
            
            if st.button("📊 Proses Analisis Dataset", key="btn_masal"):
                if tfidf_model is None or knn_model is None:
                    st.error("❌ Analisis tidak dapat dilakukan karena model gagal dimuat.")
                else:
                    with st.spinner("Memproses seluruh dataset..."):
                        # Buat salinan data dan bersihkan teks kosong
                        df_clean = df.dropna(subset=[kolom_pilihan]).copy()
                        
                        # 1. Ekstrasi fitur seluruh baris teks
                        vektor_dataset = tfidf_model.transform(df_clean[kolom_pilihan].astype(str).str.lower())
                        
                        # 2. Prediksi massal dengan KNN
                        hasil_prediksi = knn_model.predict(vektor_dataset)
                        df_clean['Label_Prediksi'] = hasil_prediksi
                        df_clean['Status_Sentimen'] = df_clean['Label_Prediksi'].apply(lambda x: "Positif" if x == 1 else "Negatif")
                        
                        # --- Visualisasi Hasil ---
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("#### 📈 Distribusi Sentimen")
                            fig = px.pie(
                                df_clean, 
                                names='Status_Sentimen', 
                                color='Status_Sentimen',
                                color_discrete_map={'Positif': '#10B981', 'Negatif': '#EF4444'},
                                hole=0.4
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col2:
                            st.markdown("#### 📋 20 Sampel Data Hasil Analisis")
                            # Filter ulasan organik pendek/repetitif (Opsional/Pola Robot)
                            def cek_pola_robot(teks):
                                kata = str(teks).lower().split()
                                if len(kata) == 0:
                                    return True
                                rasio_unik = len(set(kata)) / len(kata)
                                return rasio_unik < 0.65
                                
                            mask_robot = df_clean[kolom_pilihan].apply(cek_pola_robot)
                            df_manusia = df_clean[~mask_robot].copy()
                            
                            df_display = df_manusia if not df_manusia.empty else df_clean
                            df_display = df_display[[kolom_pilihan, 'Status_Sentimen']].head(20)
                            df_display.index = range(1, len(df_display) + 1)
                            st.dataframe(df_display, use_container_width=True, height=290)
                            
                        # Tombol Download File Hasil Analisis Lengkap
                        csv_data = df_clean.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Unduh Seluruh Hasil Analisis (.csv)",
                            data=csv_data,
                            file_name="hasil_analisis_sentimen_vacuum.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Gagal membaca atau memproses file CSV: {e}")
