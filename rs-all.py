# app.py - Aplikasi Prediksi Belanja Pasien Rumah Sakit
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, date_trunc, when
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ===============================
# SETUP
# ===============================
st.set_page_config(
    page_title="Prediksi Belanja Pasien Rumah Sakit", 
    layout="wide",
    page_icon="🏥"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #10B981;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #F59E0B;
    }
    .info-box {
        background-color: #E0F2FE;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #0EA5E9;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Aplikasi Prediksi Belanja Pasien Rumah Sakit</h1>', unsafe_allow_html=True)

# ===============================
# INISIALISASI SPARK
# ===============================
@st.cache_resource
def init_spark():
    """Inisialisasi Spark session"""
    try:
        spark = SparkSession.builder \
            .appName("PrediksiBelanjaPasien") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception as e:
        st.error(f"❌ Gagal menginisialisasi Spark: {str(e)}")
        return None

spark = init_spark()

if spark is None:
    st.error("Spark tidak dapat diinisialisasi. Aplikasi tidak dapat berjalan.")
    st.stop()

# ===============================
# FUNGSI UTILITY
# ===============================
def detect_column_names(df):
    """Deteksi otomatis nama kolom berdasarkan pola"""
    column_mapping = {}
    
    for col in df.columns:
        col_str = str(col).strip().upper()
        
        # Deteksi berdasarkan pola nama
        if col_str in ['TOTAL', 'BIAYA', 'HARGA', 'COST', 'AMOUNT', 'NILAI', 'JUMLAH', 'BELANJA', 'TRANSAKSI']:
            column_mapping[col] = 'biaya'
        elif col_str in ['DISKON', 'DISCOUNT', 'POTONGAN', 'DISC']:
            column_mapping[col] = 'diskon'
        elif any(x in col_str for x in ['POLI', 'PELAYANAN', 'LAYANAN', 'KLINIK', 'UNIT', 'BAGIAN']):
            column_mapping[col] = 'poli'
        elif any(x in col_str for x in ['PENJAMIN', 'JAMINAN', 'ASURANSI', 'PEMBAYARAN', 'ASURANSI']):
            column_mapping[col] = 'penjamin'
        elif any(x in col_str for x in ['RAWAT', 'STATUS', 'JENIS RAWAT', 'JENIS']):
            column_mapping[col] = 'status_rawat'
        elif any(x in col_str for x in ['WAKTU', 'TANGGAL', 'DATE', 'TIME', 'ADMISI', 'TGL', 'TANGGAL TRANSAKSI']):
            column_mapping[col] = 'waktu'
        elif any(x in col_str for x in ['NAMA', 'NAME', 'PASIEN', 'PATIENT']):
            column_mapping[col] = 'nama_pasien'
        elif any(x in col_str for x in ['DOKTER', 'DOCTOR', 'DR', 'DOK']):
            column_mapping[col] = 'dokter'
        elif any(x in col_str for x in ['RM', 'ID_PASIEN', 'PASIEN_ID', 'ID', 'NO RM']):
            column_mapping[col] = 'id_pasien'
        elif any(x in col_str for x in ['NO', 'ID_TRANSAKSI', 'TRANSAKSI', 'NO TRANSAKSI', 'INVOICE']):
            column_mapping[col] = 'id_transaksi'
    
    return column_mapping

def clean_numeric_value(value):
    """Membersihkan nilai numerik dari berbagai format"""
    if pd.isna(value):
        return 0.0
    
    try:
        str_val = str(value).strip()
        
        # Skip jika berisi kata kunci yang bukan angka
        exclude_keywords = ['TOTAL', 'GRAND', 'SUM', 'SUB', 'KETERANGAN', 'CATATAN', 'NA', 'N/A', 'NULL', 'NONE']
        if any(keyword in str_val.upper() for keyword in exclude_keywords):
            return 0.0
        
        # Hapus karakter non-numerik kecuali titik dan minus
        cleaned = re.sub(r'[^\d.-]', '', str_val)
        
        if cleaned == '' or cleaned == '-':
            return 0.0
        
        # Konversi ke float
        result = float(cleaned)
        return result if result >= 0 else 0.0  # Pastikan non-negatif
    except:
        return 0.0

# ===============================
# LOAD DATA DENGAN PANDAS (UNTUK CACHE)
# ===============================
@st.cache_data
def load_data_pandas():
    """Load data menggunakan pandas untuk caching - BACA SEMUA DATA"""
    try:
        # Cek file
        file1 = "belanja-pasien-asuransi2025.csv"
        file2 = "belanja-jan-nov2025.csv"
        
        if not os.path.exists(file1):
            st.error(f"❌ File {file1} tidak ditemukan")
            return None, None, None, None
        if not os.path.exists(file2):
            st.error(f"❌ File {file2} tidak ditemukan")
            return None, None, None, None
        
        # Baca file dengan pandas - jangan drop baris apapun
        df1 = pd.read_csv(file1, sep=None, engine='python', dtype=str, encoding='utf-8', on_bad_lines='skip')
        df2 = pd.read_csv(file2, sep=None, engine='python', dtype=str, encoding='utf-8', on_bad_lines='skip')
        
        # Simpan data mentah untuk perhitungan total
        total_raw_rows = len(df1) + len(df2)
        
        # Deteksi kolom
        column_mapping1 = detect_column_names(df1)
        column_mapping2 = detect_column_names(df2)
        
        # Gabungkan mapping
        all_mapping = {**column_mapping1, **column_mapping2}
        
        # Rename kolom
        df1 = df1.rename(columns=all_mapping)
        df2 = df2.rename(columns=all_mapping)
        
        # Gabungkan data - pertahankan semua baris
        df_combined = pd.concat([df1, df2], ignore_index=True)
        
        # ================================================
        # PERHITUNGAN TOTAL DARI SEMUA DATA MENTAH
        # ================================================
        total_biaya_mentah = 0
        total_values_count = 0
        zero_values_count = 0
        
        # Cari semua kolom yang berpotensi berisi nilai biaya
        biaya_cols = []
        for col in df_combined.columns:
            col_name = str(col).upper()
            # Deteksi kolom biaya dengan pola lebih luas
            if any(keyword in col_name for keyword in ['BIAYA', 'HARGA', 'COST', 'AMOUNT', 'NILAI', 'JUMLAH', 'TOTAL', 'TRANSAKSI', 'BELANJA', 'RUPIAH', 'RP']):
                # Exclude kolom yang jelas bukan nilai
                if not any(exclude in col_name for exclude in ['DISKON', 'DISCOUNT', 'POTONGAN', 'DISC', 'KETERANGAN', 'CATATAN', 'STATUS', 'TANGGAL', 'WAKTU', 'NAMA', 'PENJAMIN']):
                    biaya_cols.append(col)
        
        # Hitung total dari semua kolom biaya
        for col in biaya_cols:
            if col in df_combined.columns:
                for value in df_combined[col]:
                    cleaned_val = clean_numeric_value(value)
                    total_biaya_mentah += cleaned_val
                    total_values_count += 1
                    if cleaned_val == 0:
                        zero_values_count += 1
        
        # ================================================
        # PROSES SEMUA DATA TANPA FILTER
        # ================================================
        # Bersihkan kolom numerik
        for col in df_combined.columns:
            if col in ['biaya', 'diskon']:
                df_combined[col] = df_combined[col].apply(clean_numeric_value)
        
        # Buat kolom net_belanja
        if 'biaya' in df_combined.columns and 'diskon' in df_combined.columns:
            df_combined['net_belanja'] = df_combined['biaya'] - df_combined['diskon'].abs()
        elif 'biaya' in df_combined.columns:
            df_combined['net_belanja'] = df_combined['biaya']
        else:
            # Cari kolom lain untuk biaya
            for col in df_combined.columns:
                if any(keyword in str(col).upper() for keyword in ['TOTAL', 'BIAYA', 'HARGA', 'COST', 'AMOUNT']):
                    df_combined['net_belanja'] = df_combined[col].apply(clean_numeric_value)
                    break
            else:
                # Jika tidak ditemukan, buat kolom default
                df_combined['net_belanja'] = 0
        
        # Pastikan net_belanja non-negatif
        df_combined['net_belanja'] = df_combined['net_belanja'].apply(lambda x: max(0, x))
        
        # Parse tanggal dengan toleransi tinggi
        def parse_date_flexible(x):
            if pd.isna(x):
                return pd.NaT
            
            try:
                str_val = str(x).strip()
                
                # Coba berbagai format
                formats_to_try = [
                    '%d/%m/%Y %H:%M',
                    '%d/%m/%Y %H:%M:%S',
                    '%d-%m-%Y %H:%M',
                    '%d-%m-%Y %H:%M:%S',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d %H:%M',
                    '%d/%m/%Y',
                    '%d-%m-%Y',
                    '%Y-%m-%d'
                ]
                
                for fmt in formats_to_try:
                    try:
                        return pd.to_datetime(str_val, format=fmt, errors='raise')
                    except:
                        continue
                
                # Jika semua gagal, coba dengan pandas
                return pd.to_datetime(str_val, errors='coerce')
            except:
                return pd.NaT
        
        if 'waktu' in df_combined.columns:
            df_combined['waktu'] = df_combined['waktu'].apply(parse_date_flexible)
        else:
            df_combined['waktu'] = pd.NaT
        
        # Hitung baris dengan berbagai kondisi
        rows_with_positive_biaya = (df_combined['net_belanja'] > 0).sum()
        rows_with_zero_biaya = (df_combined['net_belanja'] == 0).sum()
        rows_with_no_date = df_combined['waktu'].isna().sum()
        
        # Isi tanggal yang kosong dengan tanggal default (tanggal pertama yang valid atau 2025-01-01)
        default_date = df_combined['waktu'].min() if not df_combined['waktu'].isna().all() else pd.Timestamp('2025-01-01')
        df_combined['waktu'] = df_combined['waktu'].fillna(default_date)
        
        # Ekstrak fitur waktu
        df_combined['tahun'] = df_combined['waktu'].dt.year.fillna(2025)
        df_combined['bulan'] = df_combined['waktu'].dt.month.fillna(1)
        df_combined['hari'] = df_combined['waktu'].dt.day.fillna(1)
        df_combined['hari_minggu'] = df_combined['waktu'].dt.dayofweek.fillna(0) + 1  # 1=Minggu, 7=Sabtu
        df_combined['jam'] = df_combined['waktu'].dt.hour.fillna(0)
        
        # Isi kolom kategori yang kosong
        cat_cols = ['poli', 'penjamin', 'status_rawat', 'nama_pasien']
        for col in cat_cols:
            if col in df_combined.columns:
                df_combined[col] = df_combined[col].astype(str).fillna('TIDAK DIKETAHUI').str.strip()
            else:
                df_combined[col] = 'TIDAK DIKETAHUI'
        
        return df_combined, total_raw_rows, total_biaya_mentah, rows_with_zero_biaya
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, 0, 0, 0

# ===============================
# KONVERSI KE SPARK
# ===============================
def pandas_to_spark(pandas_df):
    """Konversi pandas DataFrame ke Spark DataFrame"""
    if pandas_df is None or pandas_df.empty:
        return None
    
    try:
        # Konversi langsung dengan schema inference
        spark_df = spark.createDataFrame(pandas_df)
        return spark_df
    
    except Exception as e:
        st.error(f"❌ Error converting to Spark: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ===============================
# LOAD DATA
# ===============================
st.sidebar.header("📂 **LOAD DATA**")

# Load data dengan pandas (cached)
with st.spinner("Memuat SEMUA data tanpa filter..."):
    df_pandas, total_raw, total_biaya_mentah, zero_biaya_count = load_data_pandas()

if df_pandas is None:
    st.error("⚠️ Tidak ada data yang dapat diproses.")
    st.stop()

# Hitung statistik
total_processed = len(df_pandas)
positive_biaya_count = total_processed - zero_biaya_count
positive_percentage = (positive_biaya_count / total_processed * 100) if total_processed > 0 else 0

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.markdown("---")
st.sidebar.header("**FILTER DATA**")

# 1. Filter Berdasarkan Periode Waktu
st.sidebar.subheader("Filter Periode")

# Ambil min dan max date dari data
min_date = df_pandas['waktu'].min().date()
max_date = df_pandas['waktu'].max().date()

# Date range selector
date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal:",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# 2. Filter Berdasarkan Poli
st.sidebar.subheader("Filter Poli")

if 'poli' in df_pandas.columns:
    semua_poli = ["SEMUA"] + sorted(df_pandas['poli'].unique().tolist())
    selected_poli = st.sidebar.multiselect(
        "Pilih Poli:",
        options=semua_poli,
        default=["SEMUA"]
    )
else:
    selected_poli = ["SEMUA"]

# 3. Filter Berdasarkan Penjamin
st.sidebar.subheader("Filter Penjamin")

if 'penjamin' in df_pandas.columns:
    semua_penjamin = ["SEMUA"] + sorted(df_pandas['penjamin'].unique().tolist())
    selected_penjamin = st.sidebar.multiselect(
        "Pilih Penjamin:",
        options=semua_penjamin,
        default=["SEMUA"]
    )
else:
    selected_penjamin = ["SEMUA"]

# 4. Filter Berdasarkan Range Biaya
st.sidebar.subheader("Filter Biaya")

min_biaya = float(df_pandas['net_belanja'].min())
max_biaya = float(df_pandas['net_belanja'].max())

biaya_range = st.sidebar.slider(
    "Rentang Biaya (Rp):",
    min_value=min_biaya,
    max_value=max_biaya,
    value=(min_biaya, max_biaya),
    step=10000.0,
    format="Rp %d"
)

# 5. Filter Hari dalam Minggu
st.sidebar.subheader("Filter Hari")

days_mapping = {
    "Minggu": 1, "Senin": 2, "Selasa": 3, "Rabu": 4,
    "Kamis": 5, "Jumat": 6, "Sabtu": 7
}

selected_days = st.sidebar.multiselect(
    "Pilih Hari:",
    options=list(days_mapping.keys()),
    default=list(days_mapping.keys())
)

# 6. Filter Jam Operasional
st.sidebar.subheader("Filter Jam")

jam_range = st.sidebar.slider(
    "Rentang Jam (24 jam):",
    min_value=0,
    max_value=23,
    value=(0, 23),
    step=1
)

# 7. Filter Status Rawat (jika ada)
st.sidebar.subheader("Filter Status Rawat")

if 'status_rawat' in df_pandas.columns:
    semua_status = ["SEMUA"] + sorted(df_pandas['status_rawat'].unique().tolist())
    selected_status = st.sidebar.multiselect(
        "Pilih Status Rawat:",
        options=semua_status,
        default=["SEMUA"]
    )
else:
    selected_status = ["SEMUA"]

# Opsi: Sertakan data dengan biaya = 0
st.sidebar.subheader("Opsi Data")
include_zero_biaya = st.sidebar.checkbox("Sertakan data dengan biaya = 0", value=True)

# Tombol Reset Filter
st.sidebar.markdown("---")
if st.sidebar.button("Reset Semua Filter", use_container_width=True):
    st.rerun()

# ===============================
# APLIKASI FILTER KE DATA
# ===============================

# Helper function untuk apply filter
def apply_filters(df):
    # Buat copy dataframe
    df_filtered = df.copy()
    
    # 1. Filter berdasarkan tanggal
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        df_filtered = df_filtered[(df_filtered['waktu'] >= start_date) & (df_filtered['waktu'] <= end_date)]
    
    # 2. Filter berdasarkan poli
    if 'poli' in df_filtered.columns and "SEMUA" not in selected_poli and selected_poli:
        df_filtered = df_filtered[df_filtered['poli'].isin(selected_poli)]
    
    # 3. Filter berdasarkan penjamin
    if 'penjamin' in df_filtered.columns and "SEMUA" not in selected_penjamin and selected_penjamin:
        df_filtered = df_filtered[df_filtered['penjamin'].isin(selected_penjamin)]
    
    # 4. Filter berdasarkan range biaya
    if not include_zero_biaya:
        df_filtered = df_filtered[df_filtered['net_belanja'] > 0]
    
    df_filtered = df_filtered[(df_filtered['net_belanja'] >= biaya_range[0]) & (df_filtered['net_belanja'] <= biaya_range[1])]
    
    # 5. Filter berdasarkan hari
    if selected_days and len(selected_days) < 7:  # Jika tidak semua hari dipilih
        selected_day_numbers = [days_mapping[day] for day in selected_days]
        df_filtered = df_filtered[df_filtered['hari_minggu'].isin(selected_day_numbers)]
    
    # 6. Filter berdasarkan jam
    df_filtered = df_filtered[(df_filtered['jam'] >= jam_range[0]) & (df_filtered['jam'] <= jam_range[1])]
    
    # 7. Filter berdasarkan status rawat
    if 'status_rawat' in df_filtered.columns and "SEMUA" not in selected_status and selected_status:
        df_filtered = df_filtered[df_filtered['status_rawat'].isin(selected_status)]
    
    return df_filtered

# Terapkan filter
df_filtered = apply_filters(df_pandas)

# Hitung statistik setelah filter
filtered_count = len(df_filtered)
filtered_percentage = (filtered_count / total_processed) * 100

# ===============================
# KONVERSI KE SPARK
# ===============================
with st.spinner("⚡ Mengonversi ke Spark DataFrame..."):
    spark_df = pandas_to_spark(df_filtered)

if spark_df is None:
    st.error("⚠️ Gagal mengonversi data ke Spark.")
    st.stop()

# Hitung metrics dasar
total_rows = spark_df.count()

# ===============================
# TAMPILAN UTAMA DENGAN FILTER
# ===============================

# Tampilkan informasi data
st.markdown(f"""
<div class="success-box">
    <h3>DATA SUKSES DIPROSES</h3>
    <p><strong>{total_processed:,} transaksi diproses</strong> (100% dari total data mentah)</p>
    <p><strong>Total Biaya dari SEMUA Data Mentah: Rp {total_biaya_mentah:,.0f}</strong></p>
    <p>Data mentah: {total_raw:,} baris | Semua diproses: {total_processed:,} baris | Data terfilter: {filtered_count:,} baris</p>
    <p>Periode: {df_pandas['waktu'].min().strftime('%d %b %Y')} hingga {df_pandas['waktu'].max().strftime('%d %b %Y')}</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# METRICS DARI SPARK (TERFILTER)
# ===============================
st.header("**DASHBOARD**")

# Hitung metrics menggunakan Spark
with st.spinner("Menghitung metrics..."):
    # Total transaksi
    total_transaksi = total_rows
    
    # Total biaya (hanya yang > 0 jika exclude_zero_biaya=True)
    total_biaya_result = spark_df.agg(sum("net_belanja")).collect()[0][0]
    total_biaya = float(total_biaya_result) if total_biaya_result else 0
    
    # Rata-rata biaya
    avg_biaya_result = spark_df.agg(avg("net_belanja")).collect()[0][0]
    avg_biaya = float(avg_biaya_result) if avg_biaya_result else 0
    
    # Jumlah pasien unik
    unique_patients_result = spark_df.select("nama_pasien").distinct().count() if 'nama_pasien' in spark_df.columns else 0
    
    # Jumlah hari unik
    unique_days_result = spark_df.select(date_trunc("day", col("waktu"))).distinct().count()
    
    # Rata-rata transaksi per hari
    transaksi_per_hari = total_transaksi / unique_days_result if unique_days_result > 0 else 0
    
    # Rata-rata biaya per hari
    biaya_per_hari = total_biaya / unique_days_result if unique_days_result > 0 else 0
    
    # Hitung transaksi dengan biaya = 0
    zero_biaya_count_filtered = spark_df.filter(col("net_belanja") == 0).count()
    positive_biaya_count_filtered = total_transaksi - zero_biaya_count_filtered

# Tampilkan metrics utama
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("**Total Baris Data**", f"{total_raw:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.metric("**Total Semua Data**", f"Rp {total_biaya_mentah:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.metric("**Transaksi (Filter)**", f"{total_transaksi:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.metric("**Biaya (Filter)**", f"Rp {total_biaya:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Metrics tambahan
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("**Rata-rata Biaya**", f"Rp {avg_biaya:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.metric("**Pasien Unik**", f"{unique_patients_result:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.metric("**Hari Aktif**", f"{unique_days_result:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.metric("**Transaksi/Hari**", f"{transaksi_per_hari:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Metrics untuk data dengan biaya = 0
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("✅ **Biaya > 0**", f"{positive_biaya_count_filtered:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.metric("⭕ **Biaya = 0**", f"{zero_biaya_count_filtered:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.metric("💵 **Biaya/Hari**", f"Rp {biaya_per_hari:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    # Hitung poli unik jika ada
    if 'poli' in spark_df.columns:
        unique_poli = spark_df.select("poli").distinct().count()
        st.metric("🏥 **Poli Unik**", f"{unique_poli:,}")
        st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# VISUALISASI DATA DENGAN PANDAS (TERFILTER)
# ===============================
st.header("**VISUALISASI DATA**")

# Gunakan df_filtered untuk visualisasi
if len(df_filtered) > 100000:
    # Sample data untuk visualisasi jika terlalu besar
    df_viz = df_filtered.sample(n=100000, random_state=42)
else:
    df_viz = df_filtered.copy()

# Tabs untuk berbagai visualisasi
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Trend Bulanan", 
    "Analisis Poli", 
    "Analisis Penjamin", 
    "Analisis Waktu", 
    "Data Detail"
])

with tab1:
    # Trend Bulanan
    st.subheader("**Trend Biaya Bulanan**")
    
    if len(df_viz) > 0:
        # Hitung data bulanan
        df_viz['bulan_tahun'] = df_viz['waktu'].dt.to_period('M').astype(str)
        monthly_data = df_viz.groupby('bulan_tahun').agg({
            'net_belanja': ['sum', 'mean', 'count'],
            'nama_pasien': 'nunique'
        }).reset_index()
        
        monthly_data.columns = ['Periode', 'Total Biaya', 'Rata-rata Biaya', 'Jumlah Transaksi', 'Pasien Unik']
        monthly_data['Periode'] = pd.to_datetime(monthly_data['Periode'])
        monthly_data['Growth'] = monthly_data['Total Biaya'].pct_change() * 100
        
        if len(monthly_data) > 1:
            # Buat visualisasi
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Total Biaya per Bulan', 'Rata-rata Biaya per Bulan', 
                               'Jumlah Transaksi per Bulan', 'Growth Rate (%)'),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # Plot 1: Total Biaya
            fig.add_trace(
                go.Bar(
                    x=monthly_data['Periode'],
                    y=monthly_data['Total Biaya'],
                    name='Total Biaya',
                    marker_color='#3B82F6'
                ),
                row=1, col=1
            )
            
            # Plot 2: Rata-rata Biaya
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['Periode'],
                    y=monthly_data['Rata-rata Biaya'],
                    name='Rata-rata',
                    mode='lines+markers',
                    line=dict(color='#10B981', width=2)
                ),
                row=1, col=2
            )
            
            # Plot 3: Jumlah Transaksi
            fig.add_trace(
                go.Bar(
                    x=monthly_data['Periode'],
                    y=monthly_data['Jumlah Transaksi'],
                    name='Jumlah Transaksi',
                    marker_color='#8B5CF6'
                ),
                row=2, col=1
            )
            
            # Plot 4: Growth Rate
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['Periode'],
                    y=monthly_data['Growth'],
                    name='Growth %',
                    mode='lines+markers',
                    line=dict(color='#EF4444', width=2),
                    fill='tozeroy'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=700,
                showlegend=True,
                template='plotly_white',
                title_text=f"Trend Bulanan - {len(monthly_data)} Bulan"
            )
            
            # Update y-axes labels
            fig.update_yaxes(title_text="Rp", row=1, col=1, tickprefix="Rp ")
            fig.update_yaxes(title_text="Rp", row=1, col=2, tickprefix="Rp ")
            fig.update_yaxes(title_text="Jumlah", row=2, col=1)
            fig.update_yaxes(title_text="%", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabel data bulanan
            with st.expander("📋 **Lihat Data Bulanan**"):
                display_monthly = monthly_data.copy()
                display_monthly['Periode'] = display_monthly['Periode'].dt.strftime('%B %Y')
                display_monthly['Total Biaya'] = display_monthly['Total Biaya'].apply(lambda x: f"Rp {x:,.0f}")
                display_monthly['Rata-rata Biaya'] = display_monthly['Rata-rata Biaya'].apply(lambda x: f"Rp {x:,.0f}")
                display_monthly['Growth'] = display_monthly['Growth'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "-")
                st.dataframe(display_monthly[['Periode', 'Total Biaya', 'Rata-rata Biaya', 'Jumlah Transaksi', 'Pasien Unik', 'Growth']], 
                            use_container_width=True)
        else:
            st.info("ℹ️ Tidak cukup data untuk analisis trend bulanan")
    else:
        st.warning("⚠️ Tidak ada data untuk ditampilkan")

with tab2:
    # Analisis Poli
    st.subheader("🏥 **Analisis Berdasarkan Poli**")
    
    if 'poli' in df_viz.columns and len(df_viz) > 0:
        # Hitung statistik poli
        poli_stats = df_viz.groupby('poli').agg({
            'net_belanja': ['sum', 'mean', 'count'],
            'nama_pasien': 'nunique'
        }).reset_index()
        
        poli_stats.columns = ['Poli', 'Total Biaya', 'Rata-rata Biaya', 'Jumlah Transaksi', 'Pasien Unik']
        poli_stats = poli_stats.sort_values('Total Biaya', ascending=False)
        
        # Ambil top 15
        top_poli = poli_stats.head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig1 = px.bar(
                top_poli,
                x='Poli',
                y='Total Biaya',
                title='🔝 Poliklinik (Total Biaya)',
                color='Total Biaya',
                color_continuous_scale='Viridis'
            )
            fig1.update_yaxes(tickprefix="Rp ")
            fig1.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Pie chart (top 10)
            if len(top_poli) > 0:
                top_10 = top_poli.head(10)
                fig2 = px.pie(
                    top_10,
                    names='Poli',
                    values='Total Biaya',
                    title='Distribusi Biaya',
                    hole=0.4
                )
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)
        
        # Tabel detail
        with st.expander("**Detail Analisis Poli**"):
            display_poli = top_poli.copy()
            display_poli['Total Biaya'] = display_poli['Total Biaya'].apply(lambda x: f"Rp {x:,.0f}")
            display_poli['Rata-rata Biaya'] = display_poli['Rata-rata Biaya'].apply(lambda x: f"Rp {x:,.0f}")
            display_poli['% dari Total'] = (display_poli['Jumlah Transaksi'] / total_transaksi * 100).round(1)
            display_poli['% dari Total'] = display_poli['% dari Total'].apply(lambda x: f"{x}%")
            st.dataframe(display_poli, use_container_width=True)
    else:
        st.info("Kolom 'poli' tidak tersedia dalam data")

with tab3:
    # Analisis Penjamin
    st.subheader("**Analisis Berdasarkan Penjamin**")
    
    if 'penjamin' in df_viz.columns and len(df_viz) > 0:
        # Hitung statistik penjamin
        penjamin_stats = df_viz.groupby('penjamin').agg({
            'net_belanja': ['sum', 'mean', 'count'],
            'nama_pasien': 'nunique'
        }).reset_index()
        
        penjamin_stats.columns = ['Penjamin', 'Total Biaya', 'Rata-rata Biaya', 'Jumlah Transaksi', 'Pasien Unik']
        penjamin_stats = penjamin_stats.sort_values('Total Biaya', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig1 = px.bar(
                penjamin_stats,
                x='Penjamin',
                y='Total Biaya',
                title='Total Biaya per Penjamin',
                color='Jumlah Transaksi',
                color_continuous_scale='Blues'
            )
            fig1.update_yaxes(tickprefix="Rp ")
            fig1.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Donut chart
            if len(penjamin_stats) > 0:
                fig2 = px.pie(
                    penjamin_stats,
                    names='Penjamin',
                    values='Jumlah Transaksi',
                    title='Distribusi Transaksi per Penjamin',
                    hole=0.4
                )
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)
        
        # Metrics cards
        st.subheader("**Metrics per Penjamin**")
        
        # Tampilkan top 4 penjamin
        cols = st.columns(min(4, len(penjamin_stats)))
        for idx, row in penjamin_stats.iterrows():
            if idx < 4:
                with cols[idx]:
                    display_name = row['Penjamin'][:20] + '...' if len(row['Penjamin']) > 20 else row['Penjamin']
                    st.metric(
                        label=f"**{display_name}**",
                        value=f"Rp {row['Total Biaya']:,.0f}",
                        delta=f"{row['Jumlah Transaksi']:,} transaksi"
                    )
        
        # Tabel detail
        with st.expander("**Detail Analisis Penjamin**"):
            display_penjamin = penjamin_stats.copy()
            display_penjamin['Total Biaya'] = display_penjamin['Total Biaya'].apply(lambda x: f"Rp {x:,.0f}")
            display_penjamin['Rata-rata Biaya'] = display_penjamin['Rata-rata Biaya'].apply(lambda x: f"Rp {x:,.0f}")
            display_penjamin['% dari Total'] = (display_penjamin['Jumlah Transaksi'] / total_transaksi * 100).round(1)
            display_penjamin['% dari Total'] = display_penjamin['% dari Total'].apply(lambda x: f"{x}%")
            st.dataframe(display_penjamin, use_container_width=True)
    else:
        st.info("Kolom 'penjamin' tidak tersedia dalam data")

with tab4:
    # Analisis Waktu
    st.subheader("**Analisis Berdasarkan Waktu**")
    
    if len(df_viz) > 0:
        # Pilihan analisis waktu
        time_option = st.radio(
            "Pilih analisis:",
            ["Harian dalam Minggu", "Jam dalam Hari", "Heatmap"],
            horizontal=True
        )
        
        if time_option == "Harian dalam Minggu":
            # Mapping hari
            days_map = {
                1: 'Minggu', 2: 'Senin', 3: 'Selasa', 4: 'Rabu',
                5: 'Kamis', 6: 'Jumat', 7: 'Sabtu'
            }
            
            df_viz['hari_nama'] = df_viz['hari_minggu'].map(days_map)
            
            day_stats = df_viz.groupby('hari_nama').agg({
                'net_belanja': ['sum', 'mean', 'count']
            }).reset_index()
            
            day_stats.columns = ['Hari', 'Total Biaya', 'Rata-rata Biaya', 'Jumlah Transaksi']
            
            # Urutkan
            day_order = ['Minggu', 'Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu']
            day_stats['Hari'] = pd.Categorical(day_stats['Hari'], categories=day_order, ordered=True)
            day_stats = day_stats.sort_values('Hari')
            
            # Visualisasi
            fig = px.line(
                day_stats,
                x='Hari',
                y='Total Biaya',
                title='Pola Biaya per Hari dalam Minggu',
                markers=True
            )
            fig.update_yaxes(tickprefix="Rp ")
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart perbandingan
            fig_bar = px.bar(
                day_stats,
                x='Hari',
                y=['Total Biaya', 'Jumlah Transaksi'],
                title='Perbandingan Biaya dan Transaksi per Hari',
                barmode='group'
            )
            fig_bar.update_yaxes(tickprefix="Rp ", secondary_y=False)
            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        elif time_option == "Jam dalam Hari":
            # Analisis per jam
            hour_stats = df_viz.groupby('jam').agg({
                'net_belanja': ['sum', 'mean', 'count']
            }).reset_index()
            
            hour_stats.columns = ['Jam', 'Total Biaya', 'Rata-rata Biaya', 'Jumlah Transaksi']
            
            fig = px.area(
                hour_stats,
                x='Jam',
                y='Total Biaya',
                title='Pola Biaya per Jam dalam Hari',
                markers=True
            )
            fig.update_yaxes(tickprefix="Rp ")
            fig.update_xaxes(tickmode='linear', dtick=1)
            st.plotly_chart(fig, use_container_width=True)
        
        elif time_option == "Heatmap":
            # Heatmap hari vs jam
            if len(df_viz) > 0:
                heatmap_data = df_viz.pivot_table(
                    index='hari_minggu',
                    columns='jam',
                    values='net_belanja',
                    aggfunc='sum',
                    fill_value=0
                )
                
                # Map hari
                day_names = ['Minggu', 'Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu']
                heatmap_data.index = [day_names[i-1] for i in heatmap_data.index]
                
                fig_heatmap = px.imshow(
                    heatmap_data,
                    title='Heatmap: Distribusi Biaya per Hari dan Jam',
                    color_continuous_scale='Viridis',
                    labels=dict(x="Jam", y="Hari", color="Total Biaya"),
                    aspect="auto"
                )
                fig_heatmap.update_layout(height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Tidak ada data untuk ditampilkan")

with tab5:
    # Data Detail
    st.subheader("**Data Detail Transaksi**")
    
    if len(df_viz) > 0:
        # Pilih kolom untuk ditampilkan
        available_cols = df_viz.columns.tolist()
        default_cols = ['nama_pasien', 'poli', 'penjamin', 'waktu', 'net_belanja']
        
        selected_cols = st.multiselect(
            "Pilih kolom:",
            options=available_cols,
            default=[col for col in default_cols if col in available_cols]
        )
        
        if selected_cols:
            # Tampilkan data
            display_df = df_viz[selected_cols].copy()
            
            # Format
            if 'waktu' in display_df.columns:
                display_df['waktu'] = display_df['waktu'].dt.strftime('%Y-%m-%d %H:%M')
            
            if 'net_belanja' in display_df.columns:
                display_df['net_belanja'] = display_df['net_belanja'].apply(lambda x: f"Rp {x:,.0f}")
            
            # Search
            search_term = st.text_input("🔍 Cari:")
            if search_term:
                mask = display_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
                display_df = display_df[mask]
            
            # Tampilkan
            st.dataframe(display_df.head(1000), use_container_width=True, height=400)
            
            # Download
            csv_data = df_viz[selected_cols].head(10000).to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Download Sample Data",
                data=csv_data,
                file_name="sample_transaksi.csv",
                mime="text/csv"
            )
    else:
        st.warning("Tidak ada data untuk ditampilkan")

# ===============================
# DOWNLOAD DATA TERFILTER
# ===============================
st.sidebar.markdown("---")
st.sidebar.header("**EKSPOR DATA**")

if len(df_filtered) > 0:
    # Pilih format download
    download_format = st.sidebar.radio(
        "Format Download:",
        ["CSV", "Excel"]
    )
    
    # Pilih kolom untuk download
    available_cols_download = df_filtered.columns.tolist()
    default_download_cols = ['nama_pasien', 'poli', 'penjamin', 'waktu', 'net_belanja', 'tahun', 'bulan']
    
    selected_download_cols = st.sidebar.multiselect(
        "Pilih Kolom:",
        options=available_cols_download,
        default=[col for col in default_download_cols if col in available_cols_download]
    )
    
    if selected_download_cols and st.sidebar.button("Download Data Terfilter", use_container_width=True):
        # Siapkan data untuk download
        download_data = df_filtered[selected_download_cols].copy()
        
        if download_format == "CSV":
            csv = download_data.to_csv(index=False, encoding='utf-8-sig')
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"data_terfilter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            # Untuk Excel, kita perlu buffer
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                download_data.to_excel(writer, index=False, sheet_name='Data')
            buffer.seek(0)
            
            st.sidebar.download_button(
                label="Download Excel",
                data=buffer,
                file_name=f"data_terfilter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.ms-excel"
            )
        
        st.sidebar.success(f"Data siap diunduh ({len(download_data):,} baris)")

# ===============================
# FOOTER
# ===============================
st.markdown("---")

with st.expander("**INFORMASI SISTEM**"):
    st.markdown(f"""
    ## **INFORMASI SISTEM**
    
    **Statistik Data:**
    - Total Data Mentah: {total_raw:,} baris
    - Total Biaya Semua Data: Rp {total_biaya_mentah:,.0f}
    - Semua Baris Diproses: {total_processed:,} baris (100%)
    - Data Setelah Filter: {filtered_count:,} baris ({filtered_percentage:.1f}%)
    - Total Biaya Data Filter: Rp {total_biaya:,.0f}
    - Persentase Biaya Terhitung: {(total_biaya/total_biaya_mentah*100 if total_biaya_mentah>0 else 0):.1f}%
    
    **Periode:** {df_pandas['waktu'].min().strftime('%d %b %Y')} - {df_pandas['waktu'].max().strftime('%d %b %Y')}
    **Hari Aktif:** {unique_days_result:,} hari
    
    **Metrics:**
    - Total Biaya: Rp {total_biaya:,.0f}
    - Rata-rata Biaya: Rp {avg_biaya:,.0f}
    - Transaksi/Hari: {transaksi_per_hari:,.0f}
    - Biaya/Hari: Rp {biaya_per_hari:,.0f}
    
    **Filter Aktif:**
    - Periode: {date_range[0] if len(date_range) > 0 else min_date} hingga {date_range[1] if len(date_range) > 1 else max_date}
    - Poli: {', '.join(selected_poli) if 'poli' in df_pandas.columns else 'Semua'}
    - Penjamin: {', '.join(selected_penjamin) if 'penjamin' in df_pandas.columns else 'Semua'}
    - Range Biaya: Rp {biaya_range[0]:,.0f} - Rp {biaya_range[1]:,.0f}
    - Jam: {jam_range[0]}:00 - {jam_range[1]}:00
    - Sertakan biaya = 0: {'Ya' if include_zero_biaya else 'Tidak'}
    
    **Teknologi:**
    - PySpark untuk komputasi besar
    - Pandas untuk visualisasi
    - Streamlit untuk dashboard
    - Plotly untuk grafik interaktif
    """)

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <hr>
    <p>© 2026 <strong>Aplikasi Prediksi Belanja Pasien Rumah Sakit</strong> | Tugas Big Data Analitics</p>
    <p>Total Data: {total_raw:,} rows | Total Biaya Semua Data: Rp {total_biaya_mentah:,.0f} | Data Diproses: 100%</p>
</div>
""".format(total_raw=total_raw, total_biaya_mentah=total_biaya_mentah), unsafe_allow_html=True)

# ===============================
# KESIMPULAN
# ===============================
st.sidebar.markdown("---")
st.sidebar.markdown("### **STATUS SISTEM**")
st.sidebar.info(f"""
✅ **Data Loaded:** {total_raw:,} rows  
✅ **Total Semua Data:** Rp {total_biaya_mentah:,.0f}  
✅ **Diproses:** {total_processed:,} rows (100%)  
✅ **Biaya > 0:** {positive_biaya_count:,} rows  
✅ **Biaya = 0:** {zero_biaya_count:,} rows  
✅ **Filter Applied:** {filtered_count:,} rows
""")

# ===============================
# INFORMASI TAMBAHAN
# ===============================
st.sidebar.markdown("---")
st.sidebar.markdown("### **CARA MENGGUNAKAN**")
st.sidebar.info("""
1. **Semua data sudah diproses** tanpa filter
2. **Pilih filter** di sidebar untuk analisis spesifik
3. **Gunakan opsi "Sertakan biaya = 0"** untuk kontrol data
4. **Gunakan tab** untuk analisis berbeda
5. **Download data** jika perlu
""")

# ===============================
# DEBUG INFO (Opsional)
# ===============================
with st.sidebar.expander("**Debug Info**"):
    st.write(f"Total rows from files: {total_raw}")
    st.write(f"DataFrame shape: {df_pandas.shape}")
    st.write(f"Columns: {list(df_pandas.columns)}")
    if 'biaya' in df_pandas.columns:
        st.write(f"'biaya' column sample: {df_pandas['biaya'].head(5).tolist()}")
    if 'net_belanja' in df_pandas.columns:
        st.write(f"net_belanja stats - Min: {df_pandas['net_belanja'].min()}, Max: {df_pandas['net_belanja'].max()}, Mean: {df_pandas['net_belanja'].mean():.2f}")
