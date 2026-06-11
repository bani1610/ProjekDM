import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
from prophet.serialize import model_from_json

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="RailCast - Aplikasi Prediksi Penumpang Kereta",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS stylesheet
css_file = os.path.join(os.path.dirname(__file__), "style.css") if "__file__" in locals() else "style.css"
if os.path.exists(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Judul Aplikasi (Custom Premium Header)
st.markdown("""
<div class="hero-container">
    <div class="hero-badge"><i class="fa-solid fa-graduation-cap" style="margin-right: 8px;"></i>Data Mining Project &middot; Semester 4</div>
    <h1 class="hero-title"><i class="fa-solid fa-train" style="margin-right: 12px; color: #818cf8;"></i>RailCast</h1>
    <p class="hero-subtitle">Platform Analisis & Peramalan Volume Penumpang Kereta Api menggunakan 3 algoritma AI sekaligus: <strong>Prophet</strong>, <strong>ARIMA</strong>, dan <strong>Holt-Winters</strong>.</p>
</div>
""", unsafe_allow_html=True)

kategori_opsi = {
    "Kereta Bandara": "Kereta_Bandara",
    "Non Jabodetabek (Jawa)": "Non_Jabodetabek_Jawa",
    "Non Jawa (Sumatera + Sulawesi)": "Non_Jawa_Sumatera_plus_Sulawesi",
    "Kereta cepat (Whoosh)": "Kereta_cepat_Whoosh",
    "MRT": "MRT",
    "Jawa (Jabodetabek+Non Jabodetabek)": "Jawa_JabodetabekplusNon_Jabodetabek",
    "LRT": "LRT"
}


st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 1rem; margin-bottom: 1.5rem;">
    <span style="font-size: 1.25rem; color: #818cf8;"><i class="fa-solid fa-sliders"></i></span>
    <h3 style="color: #f1f5f9; font-size: 1.25rem; font-weight: 700; margin: 0; font-family: 'Inter', sans-serif;">
        Panel Kontrol
    </h3>
</div>
""", unsafe_allow_html=True)

kategori_terpilih = st.sidebar.selectbox(
    "Pilih Moda Transportasi Kereta:",
    options=list(kategori_opsi.keys())
)

# Pilihan Jumlah Bulan ke Depan
jumlah_bulan = st.sidebar.slider(
    "Periode Prediksi (Bulan ke Depan):",
    min_value=1,
    max_value=12,
    value=3
)

file_suffix = kategori_opsi[kategori_terpilih]
folder_model = "saved_models"

# 5. Fungsi untuk Memuat Model (Caching agar aplikasi cepat saat dipindah-pindah menu)
@st.cache_resource
def load_all_models(suffix):
    prophet_model = None
    arima_model = None
    hw_model = None
    
    # Path file masing-masing model
    path_prophet = os.path.join(folder_model, f"prophet_{suffix}.json")
    path_arima = os.path.join(folder_model, f"arima_{suffix}.pkl")
    path_hw = os.path.join(folder_model, f"hw_{suffix}.pkl")
    
    # Load Prophet
    if os.path.exists(path_prophet):
        with open(path_prophet, 'r') as f:
            prophet_model = model_from_json(json.load(f))
            
    # Load ARIMA
    if os.path.exists(path_arima):
        with open(path_arima, 'rb') as f:
            arima_model = pickle.load(f)
            
    # Load Holt-Winters
    if os.path.exists(path_hw):
        with open(path_hw, 'rb') as f:
            hw_model = pickle.load(f)
            
    return prophet_model, arima_model, hw_model

# Eksekusi pemuatan model
model_prophet, model_arima, model_hw = load_all_models(file_suffix)

# 6. Proses Prediksi / Forecasting
if st.sidebar.button("Jalankan Prediksi", type="primary"):
    
    with st.spinner("Sedang menghitung prediksi dari ketiga model AI..."):
        
        # Tempat menyimpan hasil prediksi masa depan
        hasil_prediksi = {}
        dates_df = None
        
        # Data historis (diambil dari model Prophet)
        hist_dates = None
        hist_values = None
        full_forecast_prophet = None  # Seluruh fitted + future dari Prophet
        
        # --- PROPHET PREDICTION ---
        if model_prophet is not None:
            future = model_prophet.make_future_dataframe(periods=jumlah_bulan, freq='MS')
            forecast = model_prophet.predict(future)
            
            # Ambil full forecast (historis + masa depan) untuk grafik
            full_forecast_prophet = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            
            # Ambil data aktual/historis dari atribut model
            hist_df = model_prophet.history[['ds', 'y']].copy()
            hist_dates = hist_df['ds'].dt.strftime('%Y-%m-%d').values
            hist_values = hist_df['y'].values
            
            # Ambil n-bulan terakhir untuk tabel & metrik (prediksi masa depan)
            future_forecast = forecast.tail(jumlah_bulan)
            hasil_prediksi['Prophet'] = future_forecast['yhat'].values
            dates_df = future_forecast['ds'].dt.strftime('%Y-%m-%d').values
        
        # --- ARIMA PREDICTION ---
        if model_arima is not None:
            try:
                pred_arima = model_arima.forecast(steps=jumlah_bulan)
                hasil_prediksi['ARIMA'] = pred_arima.values if hasattr(pred_arima, 'values') else pred_arima
            except Exception as e:
                hasil_prediksi['ARIMA'] = [np.nan] * jumlah_bulan
        
        # --- HOLT-WINTERS PREDICTION ---
        if model_hw is not None:
            try:
                pred_hw = model_hw.forecast(steps=jumlah_bulan)
                hasil_prediksi['Holt-Winters'] = pred_hw.values if hasattr(pred_hw, 'values') else pred_hw
            except Exception as e:
                hasil_prediksi['Holt-Winters'] = [np.nan] * jumlah_bulan

        # Jika tanggal gagal digenerate dari prophet, buat tanggal manual alternatif
        if dates_df is None:
            dates_df = [f"Bulan ke-{i+1}" for i in range(jumlah_bulan)]
            
        # 7. Tampilkan Output di Layout Utama
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-top: 1.5rem; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem; color: #4f46e5;"><i class="fa-solid fa-chart-line"></i></span>
            <h2 style="font-size: 1.5rem; font-weight: 700; color: #0f172a; margin: 0; font-family: 'Inter', sans-serif;">
                Hasil Analisis & Prediksi untuk: {kategori_terpilih}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Membuat DataFrame Ringkasan Hasil (hanya data prediksi masa depan)
        df_hasil = pd.DataFrame({'Tanggal / Periode': dates_df})
        for nama_model, nilai in hasil_prediksi.items():
            df_hasil[nama_model] = nilai
            df_hasil[nama_model] = df_hasil[nama_model].apply(lambda x: int(round(x)) if not np.isnan(x) else "Gagal")
            
        # Tampilkan Metrik Ringkasan (Mengambil bulan pertama prediksi sebagai sampel)
        col1, col2, col3 = st.columns(3)
        with col1:
            val_p = df_hasil['Prophet'].iloc[0] if 'Prophet' in df_hasil.columns else "N/A"
            st.metric(label="Prediksi Prophet (Bulan Depan)", value=f"{val_p:,}" if isinstance(val_p, int) else val_p)
        with col2:
            val_a = df_hasil['ARIMA'].iloc[0] if 'ARIMA' in df_hasil.columns else "N/A"
            st.metric(label="Prediksi ARIMA (Bulan Depan)", value=f"{val_a:,}" if isinstance(val_a, int) else val_a)
        with col3:
            val_h = df_hasil['Holt-Winters'].iloc[0] if 'Holt-Winters' in df_hasil.columns else "N/A"
            st.metric(label="Prediksi Holt-Winters (Bulan Depan)", value=f"{val_h:,}" if isinstance(val_h, int) else val_h)

        st.write(" ")
        
        # ================================================================
        # Hitung statistik untuk analisis otomatis
        # ================================================================
        last_actual = hist_values[-1] if hist_values is not None and len(hist_values) > 0 else None
        last_actual_date = hist_dates[-1] if hist_dates is not None and len(hist_dates) > 0 else "N/A"
        
        def hitung_tren(nilai_list, base):
            """Hitung arah tren dan % perubahan rata-rata dari nilai dasar."""
            valid = [v for v in nilai_list if not np.isnan(v)]
            if not valid or base is None or base == 0:
                return "N/A", 0.0
            rata2 = np.mean(valid)
            pct = ((rata2 - base) / base) * 100
            tren = "📈 Naik" if pct > 0 else ("📉 Turun" if pct < 0 else "➡️ Stabil")
            return tren, pct
        
        def hitung_tren_internal(nilai_list):
            """Apakah prediksi cenderung naik/turun dari bulan ke bulan."""
            valid = [v for v in nilai_list if not np.isnan(v)]
            if len(valid) < 2:
                return "➡️ Stabil"
            diffs = [valid[i+1] - valid[i] for i in range(len(valid)-1)]
            avg_diff = np.mean(diffs)
            return "📈 Naik" if avg_diff > 0 else ("📉 Turun" if avg_diff < 0 else "➡️ Stabil")
        
        analisis_model = {}
        for nama_m, nilai_m in hasil_prediksi.items():
            y_vals = [float(v) if isinstance(v, (int, float)) and not np.isnan(float(v)) else np.nan for v in nilai_m]
            tren_vs_historis, pct_vs_historis = hitung_tren(y_vals, last_actual)
            tren_internal = hitung_tren_internal(y_vals)
            valid_vals = [v for v in y_vals if not np.isnan(v)]
            analisis_model[nama_m] = {
                'values': y_vals,
                'tren_vs_historis': tren_vs_historis,
                'pct_vs_historis': pct_vs_historis,
                'tren_internal': tren_internal,
                'min_val': int(round(min(valid_vals))) if valid_vals else None,
                'max_val': int(round(max(valid_vals))) if valid_vals else None,
                'mean_val': int(round(np.mean(valid_vals))) if valid_vals else None,
            }

        # Pembagian Tab Tampilan
        tab1, tab2, tab3 = st.tabs(["Tabel Data Lengkap", "Grafik Historis & Prediksi", "Analisis Prediksi"])
        
        with tab1:
            # --- Sub-tabel 1: Data Historis ---
            st.markdown("#### <i class='fa-solid fa-clock-rotate-left' style='color: #4f46e5; margin-right: 8px;'></i>Data Historis Aktual", unsafe_allow_html=True)
            if hist_dates is not None and hist_values is not None:
                df_historis = pd.DataFrame({
                    'Tanggal': hist_dates,
                    'Volume Penumpang (Aktual)': [int(round(v)) for v in hist_values],
                    'Status': ['✅ Data Aktual'] * len(hist_dates)
                })
                # Tambahkan kolom perubahan bulan ke bulan
                perubahan = [None] + [
                    f"{((hist_values[i] - hist_values[i-1]) / hist_values[i-1] * 100):+.1f}%"
                    for i in range(1, len(hist_values))
                ]
                df_historis['Perubahan (MoM)'] = perubahan
                st.dataframe(
                    df_historis.tail(12),  # Tampilkan 12 bulan terakhir
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"*Menampilkan 12 data historis terakhir dari total {len(hist_dates)} data. Data terakhir: **{last_actual_date}** dengan volume **{int(round(last_actual)):,}** penumpang.")
            else:
                st.info("Data historis tidak tersedia (model Prophet tidak termuat).")

            st.markdown("---")
            
            # --- Sub-tabel 2: Data Prediksi + Trend ---
            st.markdown("#### <i class='fa-solid fa-wand-magic-sparkles' style='color: #10b981; margin-right: 8px;'></i>Data Prediksi Masa Depan", unsafe_allow_html=True)
            df_pred_display = pd.DataFrame({'Tanggal / Periode': dates_df})
            for nama_model, nilai in hasil_prediksi.items():
                y_int = [int(round(float(v))) if not np.isnan(float(v)) else None for v in nilai]
                df_pred_display[nama_model] = [f"{v:,}" if v is not None else "Gagal" for v in y_int]
                
                # Tambahkan kolom % perubahan vs data aktual terakhir
                if last_actual is not None:
                    pct_col = []
                    for v in nilai:
                        try:
                            pct = ((float(v) - last_actual) / last_actual) * 100
                            pct_col.append(f"{pct:+.1f}%")
                        except:
                            pct_col.append("N/A")
                    df_pred_display[f"{nama_model} vs Aktual"] = pct_col
            
            st.dataframe(df_pred_display, use_container_width=True, hide_index=True)
            st.caption("*Angka menunjukkan jumlah penumpang (satuan orang). Kolom '% vs Aktual' adalah perubahan terhadap data aktual terakhir.")
            
        with tab2:
            # ================================================================
            # GRAFIK GABUNGAN: DATA HISTORIS + PREDIKSI MASA DEPAN
            # ================================================================
            fig, ax = plt.subplots(figsize=(14, 6.5))
            
            # Set background color of the figure and axes to transparent/clean
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            
            # --- Plot 1: Data Aktual / Historis ---
            if hist_dates is not None and hist_values is not None:
                ax.plot(
                    hist_dates, hist_values,
                    color='#334155', linewidth=2.5, marker='o', markersize=5,
                    markerfacecolor='#334155', markeredgecolor='white', markeredgewidth=1.2,
                    label='Data Aktual (Historis)', zorder=5
                )
                
                # Shaded area confidence interval dari Prophet (opsional)
                if full_forecast_prophet is not None:
                    hist_len = len(hist_dates)
                    prophet_hist_slice = full_forecast_prophet.head(hist_len)
                    ax.fill_between(
                        prophet_hist_slice['ds'].dt.strftime('%Y-%m-%d').values,
                        prophet_hist_slice['yhat_lower'].values,
                        prophet_hist_slice['yhat_upper'].values,
                        color='#3b82f6', alpha=0.08, label='Confidence Interval Prophet'
                    )
            
            # --- Garis Pemisah & Shaded Region untuk Prediksi ---
            if hist_dates is not None and len(hist_dates) > 0:
                last_hist_date = hist_dates[-1]
                
                # Shaded area untuk future prediction
                if dates_df is not None and len(dates_df) > 0:
                    ax.axvspan(
                        last_hist_date, dates_df[-1],
                        color='#f8fafc', alpha=0.7, label='Periode Prediksi (Masa Depan)', zorder=1
                    )
                
                # Vertical line pemisah
                ax.axvline(
                    x=last_hist_date,
                    color='#4f46e5', linestyle=':', linewidth=1.5, alpha=0.6, zorder=2
                )
                
                # Tentukan posisi y untuk label (di bagian atas grafik)
                y_lim = ax.get_ylim()
                y_pos = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.95
                ax.text(
                    last_hist_date, y_pos,
                    '  ← Historis | Prediksi →  ',
                    fontsize=8, color='#4f46e5', va='top',
                    ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0e7ff', edgecolor='#c7d2fe', alpha=0.9, linewidth=1)
                )
            
            # --- Plot 2: Prediksi Masa Depan dari setiap model ---
            warna_model = {
                'Prophet': '#3b82f6',
                'ARIMA': '#f43f5e',
                'Holt-Winters': '#10b981'
            }
            marker_model = {
                'Prophet': 'o',
                'ARIMA': 's',
                'Holt-Winters': '^'
            }
            
            # Sambungkan titik terakhir historis ke titik pertama prediksi (bridging)
            bridge_date = [hist_dates[-1]] if hist_dates is not None and len(hist_dates) > 0 else []
            bridge_val  = [hist_values[-1]] if hist_values is not None and len(hist_values) > 0 else []
            
            for nama_model, nilai in hasil_prediksi.items():
                y_pred = [float(x) if isinstance(x, (int, float)) else np.nan for x in nilai]
                warna = warna_model.get(nama_model, '#64748b')
                marker_style = marker_model.get(nama_model, 'o')
                
                # Sambungkan dari titik terakhir historis
                x_plot = list(bridge_date) + list(dates_df)
                y_plot = list(bridge_val)  + y_pred
                
                ax.plot(
                    x_plot, y_plot,
                    marker=marker_style, markersize=7, linewidth=2.5,
                    markerfacecolor=warna, markeredgecolor='white', markeredgewidth=1.5,
                    linestyle='--',
                    color=warna,
                    label=f'Prediksi {nama_model}',
                    zorder=4
                )
                
                # Tambahkan anotasi nilai pada titik prediksi
                for i, (xp, yp) in enumerate(zip(dates_df, y_pred)):
                    if not np.isnan(yp):
                        ax.annotate(
                            f'{int(round(yp)):,}',
                            xy=(xp, yp),
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=8, ha='center', color=warna,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=warna, alpha=0.8, linewidth=0.5)
                        )
            
            # --- Konfigurasi Grafik ---
            ax.set_title(
                f"Analisis Tren Historis vs Prediksi Volume Penumpang\n{kategori_terpilih} ({jumlah_bulan} Bulan ke Depan)",
                fontweight='bold', fontsize=13, pad=15, color='#0f172a'
            )
            ax.set_ylabel("Volume Penumpang", fontsize=11, fontweight='semibold', color='#334155')
            ax.set_xlabel("Periode Waktu", fontsize=11, fontweight='semibold', color='#334155')
            
            # Format label sumbu Y agar memiliki pemisah ribuan
            import matplotlib.ticker as ticker
            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            # Style Legend
            ax.legend(
                loc='upper left', 
                frameon=True, 
                facecolor='white', 
                edgecolor='#e2e8f0', 
                framealpha=0.95, 
                fontsize=9.5
            )
            
            # Soft Gridlines horizontal saja
            ax.grid(True, axis='y', linestyle=':', alpha=0.6, color='#cbd5e1')
            ax.grid(False, axis='x')
            
            # Spines styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            
            # Format label sumbu X
            plt.xticks(rotation=45, ha='right', fontsize=8.5, color='#475569')
            plt.yticks(fontsize=8.5, color='#475569')
            
            plt.tight_layout()
            st.pyplot(fig)
            st.caption(
                "📌 **Keterangan:** Garis gelap solid = data aktual historis. "
                "Garis putus-putus berwarna = prediksi model AI ke depan. "
                "Wilayah berbayang biru muda di kanan = rentang waktu prediksi."
            )
        
        # ================================================================
        # TAB 3: ANALISIS OTOMATIS
        # ================================================================
        with tab3:
            st.markdown(f"### <i class='fa-solid fa-magnifying-glass-chart' style='color: #4f46e5; margin-right: 8px;'></i>Analisis Prediksi: **{kategori_terpilih}**", unsafe_allow_html=True)
            st.markdown(
                f"Analisis berikut dihasilkan secara otomatis berdasarkan hasil prediksi "
                f"**{jumlah_bulan} bulan ke depan** dari tiga model AI. "
                f"Data aktual terakhir tercatat pada **{last_actual_date}** "
                f"sebesar **{int(round(last_actual)):,}** penumpang."
                if last_actual is not None else
                "Analisis dihasilkan secara otomatis berdasarkan hasil prediksi."
            )
            st.markdown("---")
            
            # --- Ringkasan Komparatif ---
            st.markdown("#### <i class='fa-solid fa-scale-balanced' style='color: #4f46e5; margin-right: 8px;'></i>Ringkasan Komparatif Antar Model", unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns(3)
            model_means = {nm: info['mean_val'] for nm, info in analisis_model.items() if info['mean_val'] is not None}
            
            if model_means:
                model_tertinggi = max(model_means, key=model_means.get)
                model_terendah  = min(model_means, key=model_means.get)
                
                with col_a:
                    st.metric("🏆 Estimasi Tertinggi", model_tertinggi,
                              f"{model_means[model_tertinggi]:,} penumpang")
                with col_b:
                    all_means = list(model_means.values())
                    konsensus = int(round(np.mean(all_means)))
                    st.metric("🤝 Rata-rata Konsensus", f"{konsensus:,}",
                              "rata-rata ketiga model")
                with col_c:
                    st.metric("📉 Estimasi Terendah", model_terendah,
                              f"{model_means[model_terendah]:,} penumpang")
            
            st.markdown("---")
            
            # --- Analisis Per Model ---
            st.markdown("#### <i class='fa-solid fa-microchip' style='color: #4f46e5; margin-right: 8px;'></i>Analisis Per Model", unsafe_allow_html=True)
            
            emoji_model = {'Prophet': '🔵', 'ARIMA': '🔴', 'Holt-Winters': '🟢'}
            deskripsi_model = {
                'Prophet': 'Model berbasis dekomposisi tren musiman dari Meta/Facebook. Cocok menangkap pola musiman tahunan dan efek hari libur.',
                'ARIMA': 'Model statistik klasik yang menganalisis autokorelasi data deret waktu. Cocok untuk tren jangka pendek yang stabil.',
                'Holt-Winters': 'Model Exponential Smoothing dengan komponen tren dan musiman. Baik untuk data dengan pola musiman yang konsisten.',
            }
            
            for nama_m, info in analisis_model.items():
                emoji = emoji_model.get(nama_m, '⚪')
                with st.expander(f"{emoji} Analisis Model **{nama_m}**", expanded=True):
                    st.markdown(f"*{deskripsi_model.get(nama_m, '')}*")
                    st.markdown("")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Tren vs Data Aktual", info['tren_vs_historis'],
                              f"{info['pct_vs_historis']:+.1f}% rata-rata" if isinstance(info['pct_vs_historis'], float) else "")
                    c2.metric("Pola Prediksi", info['tren_internal'])
                    c3.metric("Prediksi Terendah", f"{info['min_val']:,}" if info['min_val'] else "N/A")
                    c4.metric("Prediksi Tertinggi", f"{info['max_val']:,}" if info['max_val'] else "N/A")
                    
                    st.markdown("")
                    
                    # Narasi otomatis
                    pct = info['pct_vs_historis']
                    tren_v = info['tren_vs_historis']
                    tren_i = info['tren_internal']
                    mean_v = info['mean_val']
                    
                    if isinstance(pct, float) and last_actual is not None:
                        arah = "peningkatan" if pct > 0 else "penurunan"
                        mag = "signifikan" if abs(pct) > 10 else ("moderat" if abs(pct) > 3 else "kecil")
                        narasi = (
                            f"Model **{nama_m}** memproyeksikan **{arah} {mag}** volume penumpang "
                            f"dengan rata-rata **{mean_v:,}** penumpang per bulan selama {jumlah_bulan} bulan ke depan. "
                            f"Dibandingkan data aktual terakhir ({int(round(last_actual)):,} penumpang), "
                            f"model ini memprediksi perubahan rata-rata sebesar **{pct:+.1f}%**. "
                        )
                        if tren_i == "📈 Naik":
                            narasi += f"Dari bulan ke bulan, prediksi menunjukkan kecenderungan **terus meningkat**, mengindikasikan pertumbuhan yang berkelanjutan."
                        elif tren_i == "📉 Turun":
                            narasi += f"Dari bulan ke bulan, prediksi menunjukkan kecenderungan **terus menurun**, mengindikasikan potensi perlambatan volume."
                        else:
                            narasi += f"Dari bulan ke bulan, prediksi relatif **stabil** tanpa perubahan signifikan."
                    else:
                        narasi = f"Model **{nama_m}** menghasilkan prediksi rata-rata **{mean_v:,}** penumpang per bulan."
                    
                    st.info(narasi)
            
            st.markdown("---")
            
            # --- Kesimpulan Umum ---
            st.markdown("#### <i class='fa-solid fa-clipboard-list' style='color: #4f46e5; margin-right: 8px;'></i>Kesimpulan &amp; Rekomendasi", unsafe_allow_html=True)
            
            if model_means and last_actual is not None:
                all_pcts = [info['pct_vs_historis'] for info in analisis_model.values() if isinstance(info['pct_vs_historis'], float)]
                avg_pct_all = np.mean(all_pcts) if all_pcts else 0
                
                konsensus_arah = "peningkatan" if avg_pct_all > 0 else "penurunan"
                all_tren = [info['tren_internal'] for info in analisis_model.values()]
                sepakat = len(set(all_tren)) == 1
                
                kesimpulan = (
                    f"Secara keseluruhan, ketiga model AI memprediksi **{konsensus_arah}** "
                    f"volume penumpang **{kategori_terpilih}** dengan rata-rata konsensus **{avg_pct_all:+.1f}%** "
                    f"dibandingkan data aktual terakhir. "
                )
                if sepakat:
                    kesimpulan += f"Ketiga model **bersepakat** bahwa tren akan {all_tren[0].split(' ')[1].lower()}, menunjukkan konsistensi yang tinggi. "
                else:
                    kesimpulan += "Terdapat **perbedaan pandangan** antar model mengenai arah tren, sehingga diperlukan kehati-hatian dalam interpretasi. "
                
                kesimpulan += (
                    f"Disarankan untuk menggunakan nilai **rata-rata konsensus ({konsensus:,} penumpang/bulan)** "
                    "sebagai acuan perencanaan, dengan mempertimbangkan rentang prediksi sebagai batas atas dan bawah."
                )
                st.success(kesimpulan)
            else:
                st.info("Tidak cukup data untuk menghasilkan kesimpulan otomatis.")

else:
    st.markdown("""
    <div class="custom-card" style="text-align: center; padding: 4rem 2rem; border-radius: 16px; margin-top: 1rem;">
        <div style="font-size: 3.5rem; margin-bottom: 1.5rem; color: #4f46e5;"><i class="fa-solid fa-chart-line"></i></div>
        <h3 style="margin-bottom: 0.75rem; color: #0f172a; font-weight: 700; font-family: 'Inter', sans-serif;">Siap Melakukan Peramalan</h3>
        <p style="color: #64748b; max-width: 500px; margin: 0 auto 1.5rem auto; line-height: 1.6; font-family: 'Inter', sans-serif; font-size: 0.95rem;">
            Silakan pilih parameter peramalan di panel kontrol sebelah kiri, lalu klik tombol <strong>Jalankan Prediksi</strong> untuk memulai pengolahan data multi-model AI.
        </p>
    </div>
    """, unsafe_allow_html=True)