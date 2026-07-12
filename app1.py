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

# --- 2. CUSTOM CSS UNTUK UI MODERN (DISESUAIKAN DARI APP.PY) ---
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
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        color: #4B5563;
    }
    .stTabs [aria-selected="true"] {
        color: #4F46E5 !important;
        border-bottom-color: #4F46E5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL DAN DATASET (KODE TETAP MENGGUNAKAN MODALITAS APP1.PY) ---
@st.cache_resource
def load_models():
    tfidf_path = "tfidf_model.pkl"
    knn_path = "knn_model.pkl"
    
    if not os.path.exists(tfidf_path) or not os.path.exists(knn_path):
        return None, None, f"File model '{tfidf_path}' atau '{knn_path}' tidak ditemukan."
        
    try:
        with open(tfidf_path, 'rb') as f_tfidf:
            tfidf = pickle.load(f_tfidf)
        with open(knn_path, 'rb') as f_knn:
            knn = pickle.load(f_knn)
        return tfidf, knn, None
    except Exception as e:
        return None, None, str(e)

# Membongkar model secara terpisah agar tidak AttributeError
tfidf_model, knn_model, error_msg = load_models()

if error_msg:
    st.error(f"❌ **Gagal Memuat Model:** {error_msg}")
    st.info("💡 Pastikan file `.pkl` berada di folder yang sama dengan file script ini.")

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
                    # Proses transformasi menggunakan tfidf_model
                    vektor_teks = tfidf_model.transform([teks_input.lower()])
                    
                    # Proses prediksi menggunakan knn_model
                    prediksi_kelas = knn_model.predict(vektor_teks)[0]
                    
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

# ==================== TAB 2: ULASAN MASAL (CSV) ====================
with tab2:
    st.markdown('<div class="custom-card"><h3>Upload File Dataset Ulasan</h3></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file CSV hasil export ulasan", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File berhasil diunggah!")
            
            # Pilihan pemilihan kolom secara dinamis
            nama_kolom = st.selectbox("Pilih kolom yang berisi Teks Ulasan:", df.columns)
            
            if st.button("📊 Proses Analisis Dataset", key="btn_masal"):
                if tfidf_model is None or knn_model is None:
                    st.error("❌ Analisis tidak dapat dilakukan karena model gagal dimuat.")
                else:
                    with st.spinner("Memproses seluruh dataset..."):
                        df_clean = df.dropna(subset=[nama_kolom]).copy()
                        
                        # Transformasi fitur massal dengan tfidf_model
                        vektor_dataset = tfidf_model.transform(df_clean[nama_kolom].astype(str).str.lower())
                        
                        # Prediksi massal dengan knn_model
                        hasil_prediksi = knn_model.predict(vektor_dataset)
                        df_clean['Label_Prediksi'] = hasil_prediksi
                        df_clean['Status_Sentimen'] = df_clean['Label_Prediksi'].apply(lambda x: "Positif" if x == 1 else "Negatif")
                        
                        # Filter ulasan untuk visualisasi murni
                        df_organik = df_clean.copy()
                        
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
                            
                            def cek_pola_robot(teks):
                                kata = str(teks).lower().split()
                                if len(kata) == 0:
                                    return True
                                rasio_unik = len(set(kata)) / len(kata)
                                return rasio_unik < 0.65 

                            if not df_organik.empty:
                                mask_robot = df_organik[nama_kolom].apply(cek_pola_robot)
                                df_manusia = df_organik[~mask_robot].copy()
                            else:
                                df_manusia = df_clean.copy()
                                
                            if len(df_manusia) >= 20:
                                df_display = df_manusia[[nama_kolom, 'Status_Sentimen']].head(20)
                            else:
                                df_display = df_clean[[nama_kolom, 'Status_Sentimen']].head(20)
                                
                            df_display.index = range(1, len(df_display) + 1)
                            st.dataframe(df_display, use_container_width=True, height=290)
                            
                        csv_data = df_clean.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Unduh Seluruh Hasil Analisis (.csv)",
                            data=csv_data,
                            file_name="hasil_analisis_sentimen_vacuum.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Gagal membaca atau memproses file CSV: {e}")
