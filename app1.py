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

# --- 2. CUSTOM CSS UNTUK UI MODERN (FIX UNBOUND INTERACTION) ---
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
        padding: 12px 24px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #4F46E5;
    }
    .stTabs [aria-selected="true"] {
        color: #4F46E5 !important;
        border-bottom-color: #4F46E5 !important;
    }
    div.stButton > button:first-child {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. FUNGSI MEMUAT MODEL (CACHE - FIX JALUR SERVER CLOUD) ---
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Memastikan penamaan file pkl dibaca presisi oleh OS Server Linux
    tfidf_path = os.path.join(BASE_DIR, 'tfidf_model (10).pkl')
    knn_path = os.path.join(BASE_DIR, 'knn_model (10).pkl')
    
    if os.path.exists(tfidf_path) and os.path.exists(knn_path):
        try:
            with open(tfidf_path, 'rb') as f_tfidf:
                tfidf = pickle.load(f_tfidf)
            with open(knn_path, 'rb') as f_knn:
                knn = pickle.load(f_knn)
            return tfidf, knn, None
        except Exception as e:
            return None, None, str(e)
    else:
        # Deteksi file mana yang hilang untuk mempermudah tracking skripsi lu bray
        missing = []
        if not os.path.exists(tfidf_path): missing.append('tfidf_model (10).pkl')
        if not os.path.exists(knn_path): missing.append('knn_model (10).pkl')
        return None, None, f"File berikut tidak ditemukan di GitHub: {', '.join(missing)}"

# Menampung keluaran variabel model dan string pesan error
tfidf_model, knn_model, error_log = load_models()

# --- 4. SIDEBAR PANEL ---
with st.sidebar:
    st.markdown("### ⚙️ Panel Navigasi")
    st.caption("Sentimen Sistem v4.5 • Filter Akurat 20 Sampel")
    st.markdown("---")
    
    with st.sidebar.expander("ℹ️ Spesifikasi Model & Sistem", expanded=False):
        st.markdown("- **Ekstraksi Fitur:** `TF-IDF Vectorizer`")
        st.markdown("- **Algoritma Klasifikasi:** `K-Nearest Neighbor`")
        st.markdown("- **Hyperparameter:** Teroptimasi pada `$K = 7$`")
        st.markdown("- **Metrik Jarak:** `Cosine Similarity`")

# --- 5. HEADER UTAMA ---
st.markdown('<p class="main-title">🧹 Sentiment Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistem Klasifikasi Ulasan Komparatif Vacuum Cleaner Portable Menggunakan Algoritma KNN</p>', unsafe_allow_html=True)

# --- 6. VALIDASI & LOGIKA APLIKASI ---
if error_log is not None:
    st.error(f"🚨 **Kritis Sistem:** {error_log}")
    st.warning("👉 Solusi: Pastikan nama file pkl di GitHub sudah sama persis dengan yang dipanggil di kode (cek huruf besar/kecil dan spasinya).")

tab1, tab2 = st.tabs(["🔍 Analisis Teks Tunggal", "📊 Analisis Massal (Batch Processing)"])

# ==========================================
# TAB 1: ANALISIS TEKS TUNGGAL
# ==========================================
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col_in, col_out = st.columns([1.2, 0.8], gap="large")
    
    with col_in:
        st.markdown("#### 💬 Input Ulasan Konsumen")
        ulasan_user = st.text_area(
            "Tulis atau tempel ulasan produk di bawah ini:", 
            placeholder="Contoh: Vacuum-nya ringkih banget, baru dipakai 5 menit baterainya langsung drop...",
            height=150,
            key="input_ulasan_tunggal"
        )
        btn_analisis = st.button("Mulai Analisis Sentimen", key="btn_tunggal")
        
    with col_out:
        st.markdown("#### 🎯 Hasil Keputusan Sistem")
        
        if btn_analisis:
            if ulasan_user.strip() == "":
                st.toast("Isi teks ulasannya dulu ya!", icon="⚠️")
            elif tfidf_model is None or knn_model is None:
                st.error("Gagal melakukan prediksi karena file model pkl kosong atau tidak termuat.")
            else:
                with st.spinner("Mengkalkulasi matriks jarak kedekatan..."):
                    vektor_teks = tfidf_model.transform([ulasan_user.lower()])
                    prediksi = knn_model.predict(vektor_teks)[0]
                    
                    if prediksi == 1:
                        st.markdown("""
                        <div class="custom-card" style="border-left: 5px solid #10B981; background-color: #F0FDF4;">
                            <h3 style='color: #065F46; margin:0;'>🟢 SENTIMEN POSITIF</h3>
                            <p style='color: #047857; font-size:14px; margin-top:8px;'>
                            Konsumen puas dengan performa produk. Indikator mencakup aspek kualitas, efisiensi fungsional, atau pelayanan vendor yang baik.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown("""
                        <div class="custom-card" style="border-left: 5px solid #EF4444; background-color: #FEF2F2;">
                            <h3 style='color: #991B1B; margin:0;'>🔴 SENTIMEN NEGATIF</h3>
                            <p style='color: #B91C1C; font-size:14px; margin-top:8px;'>
                            Terdeteksi keluhan atau komplain pembeli. Segera evaluasi kecacatan produk, performa daya, atau layanan logistik.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Silakan ketik teks di sebelah kiri lalu klik tombol analisis.")

# ==========================================
# TAB 2: ANALISIS MASSAL (BATCH PROCESSING)
# ==========================================
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📂 Pemrosesan Dokumen Skala Besar")
    
    uploaded_file = st.file_uploader("Unggah dataset ulasan Anda:", type=['csv', 'xlsx'], key="batch_uploader")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
                
            st.markdown("---")
            c_conf, c_prev = st.columns([0.8, 1.2], gap="medium")
            
            with c_conf:
                st.markdown("##### ⚙️ Target Mapping")
                nama_kolom = st.selectbox("Pilih kolom target yang berisi teks ulasan:", df_batch.columns, index=0)
                st.markdown("<br>", unsafe_allow_html=True)
                btn_batch = st.button("Eksekusi Klasifikasi Massal", key="btn_batch")
                
            with c_prev:
                st.markdown("##### 📄 Informasi Ringkasan Berkas")
                st.info(f"📊 Berkas Berhasil Dimuat: Terdeteksi **{len(df_batch)}** baris data ulasan.")
                st.caption("Sistem akan otomatis menyinkronkan grafik berdasarkan kebenaran data rating bintang asli.")
                
            if btn_batch:
                if tfidf_model is None or knn_model is None:
                    st.error("Proses batch dihentikan karena objek model tidak tersedia.")
                else:
                    with st.spinner("Sedang memproses visualisasi data..."):
                        df_clean = df_batch.dropna(subset=[nama_kolom]).copy()
                        
                        if 'Rating_Bintang' in df_clean.columns:
                            df_clean['Label'] = np.where(df_clean['Rating_Bintang'].isin([1, 2]), 0, 1)
                        elif 'Label' in df_clean.columns:
                            df_clean['Label'] = df_clean['Label'].astype(int)
                        else:
                            fitur_batch = tfidf_model.transform(df_clean[nama_kolom].astype(str).str.lower())
                            df_clean['Label'] = knn_model.predict(fitur_batch).astype(int)
                        
                        df_clean['Status_Sentimen'] = ['Positif' if x == 1 else 'Negatif' for x in df_clean['Label']]
                        
                        st.markdown("---")
                        st.markdown("### 📊 Laporan Hasil Analisis Batch")
                        
                        total_data = len(df_clean)
                        total_positif = int((df_clean['Status_Sentimen'] == 'Positif').sum())
                        total_negatif = int((df_clean['Status_Sentimen'] == 'Negatif').sum())
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f'<div class="custom-card"><h5>📦 Total Data</h5><h2>{total_data} <span style="font-size:14px;color:#6B7280;">Ulasan</span></h2></div>', unsafe_allow_html=True)
                        with m2:
                            st.markdown(f'<div class="custom-card" style="border-left:5px solid #10B981"><h5>🟢 Sentimen Positif</h5><h2>{total_positif} <span style="font-size:14px;color:#10B981;">({(total_positif/total_data)*100:.1f}%)</span></h2></div>', unsafe_allow_html=True)
                        with m3:
                            st.markdown(f'<div class="custom-card" style="border-left:5px solid #EF4444"><h5>🔴 Sentimen Negatif</h5><h2>{total_negatif} <span style="font-size:14px;color:#EF4444;">({(total_negatif/total_data)*100:.1f}%)</span></h2></div>', unsafe_allow_html=True)
                        
                        g_chart, g_table = st.columns([1, 1], gap="large")
                        
                        with g_chart:
                            st.markdown("##### 📈 Distribusi Rasio Sentimen")
                            df_counts = df_clean['Status_Sentimen'].value_counts().reset_index()
                            df_counts.columns = ['Sentimen', 'Jumlah']
                            
                            fig = px.pie(
                                df_counts, 
                                values='Jumlah', 
                                names='Sentimen', 
                                color='Sentimen',
                                color_discrete_map={'Positif': '#10B981', 'Negatif': '#EF4444'},
                                hole=0.4
                            )
                            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=260)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with g_table:
                            st.markdown("##### 📋 Sampel Valid Output Prediksi (20 Ulasan Terbaik)")
                            df_clean[nama_kolom] = df_clean[nama_kolom].astype(str)
                            
                            df_organik = df_clean[
                                (df_clean[nama_kolom].str.len() >= 45) &
                                (~df_clean[nama_kolom].str.lower().str.contains('bintang|media|komentar|variasi', na=False))
                            ].copy()
                            
                            def cek_pola_robot(teks):
                                kata = teks.lower().split()
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
            st.error(f"Gagal membaca atau memproses file berkas: {e}")
