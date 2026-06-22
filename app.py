import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import warnings
import matplotlib.pyplot as plt
from prophet.serialize import model_from_json
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings('ignore')

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

# 5. Fungsi untuk Memuat Model
# ARIMA & HW di-fit langsung dari data training (JSON) agar tidak bergantung
# pada versi pickle/scipy — menghindari error '_xp' di deployment.
@st.cache_resource
def load_all_models(suffix):
    prophet_model = None
    arima_model = None
    hw_model = None

    path_prophet  = os.path.join(folder_model, f"prophet_{suffix}.json")
    path_traindata = os.path.join(folder_model, f"traindata_{suffix}.json")

    # Load Prophet (aman — disimpan sebagai JSON murni)
    if os.path.exists(path_prophet):
        with open(path_prophet, 'r') as f:
            prophet_model = model_from_json(json.load(f))

    # Fit ARIMA & Holt-Winters dari data training JSON (tanpa pickle sama sekali)
    if os.path.exists(path_traindata):
        with open(path_traindata, 'r') as f:
            y_train = json.load(f)["y"]

        # Fit ARIMA
        try:
            fit_arima = ARIMA(y_train, order=(1, 1, 1)).fit()
            arima_model = fit_arima
        except Exception as e:
            st.warning(f"⚠️ ARIMA gagal di-fit: {e}")

        # Fit Holt-Winters
        try:
            fit_hw = ExponentialSmoothing(y_train, trend='add', seasonal=None).fit()
            hw_model = fit_hw
        except Exception as e:
            st.warning(f"⚠️ Holt-Winters gagal di-fit: {e}")

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
            # GRAFIK GABUNGAN: DATA HISTORIS + PREDIKSI MASA DEPAN (Plotly)
            # ================================================================
            import plotly.graph_objects as go
            from datetime import datetime as dt

            def to_dt(val):
                if isinstance(val, str):
                    return dt.strptime(val, '%Y-%m-%d')
                try:
                    return pd.Timestamp(val).to_pydatetime()
                except Exception:
                    return val

            hist_dt  = [to_dt(d) for d in hist_dates] if hist_dates is not None else []
            dates_dt = [to_dt(d) for d in dates_df]   if dates_df  is not None else []

            fig_plotly = go.Figure()

            # --- Confidence Interval Prophet (historis saja, sangat tipis) ---
            if full_forecast_prophet is not None and hist_dt:
                hist_len = len(hist_dt)
                ph = full_forecast_prophet.head(hist_len)
                ci_dates = [to_dt(d) for d in ph['ds'].values]
                fig_plotly.add_trace(go.Scatter(
                    x=ci_dates + ci_dates[::-1],
                    y=list(ph['yhat_upper'].values) + list(ph['yhat_lower'].values[::-1]),
                    fill='toself',
                    fillcolor='rgba(59,130,246,0.07)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Confidence Interval Prophet',
                    hoverinfo='skip',
                    showlegend=True
                ))

            # --- Shaded area periode prediksi ---
            if hist_dt and dates_dt:
                last_hist = hist_dt[-1]
                last_pred = dates_dt[-1]
                y_min_shade = min(hist_values) * 0.97 if hist_values is not None else 0
                y_max_shade = max(hist_values) * 1.15 if hist_values is not None else 1
                fig_plotly.add_trace(go.Scatter(
                    x=[last_hist, last_pred, last_pred, last_hist],
                    y=[y_max_shade, y_max_shade, y_min_shade, y_min_shade],
                    fill='toself',
                    fillcolor='rgba(240,244,255,0.6)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Periode Prediksi (Masa Depan)',
                    hoverinfo='skip',
                    showlegend=True
                ))

            # --- Data Aktual / Historis ---
            if hist_dt and hist_values is not None:
                fig_plotly.add_trace(go.Scatter(
                    x=hist_dt,
                    y=list(hist_values),
                    mode='lines+markers',
                    name='Data Aktual (Historis)',
                    line=dict(color='#334155', width=2.5),
                    marker=dict(size=6, color='#334155',
                                line=dict(color='white', width=1.5)),
                    hovertemplate='<b>%{x|%Y-%m}</b><br>Aktual: %{y:,.0f}<extra></extra>'
                ))

            # --- Prediksi tiap model ---
            warna_model = {
                'Prophet':     '#3b82f6',
                'ARIMA':       '#f43f5e',
                'Holt-Winters':'#10b981'
            }
            marker_model = {
                'Prophet':     'circle',
                'ARIMA':       'square',
                'Holt-Winters':'triangle-up'
            }
            bridge_dt  = [hist_dt[-1]] if hist_dt else []
            bridge_val = [float(hist_values[-1])] if hist_values is not None and len(hist_values) > 0 else []

            for nama_model, nilai in hasil_prediksi.items():
                y_pred = [float(x) if isinstance(x, (int, float)) else None for x in nilai]
                warna  = warna_model.get(nama_model, '#64748b')
                mkr    = marker_model.get(nama_model, 'circle')

                # Garis prediksi (termasuk bridging dari titik terakhir historis)
                fig_plotly.add_trace(go.Scatter(
                    x=bridge_dt + dates_dt,
                    y=bridge_val + y_pred,
                    mode='lines+markers',
                    name=f'Prediksi {nama_model}',
                    line=dict(color=warna, width=2.2, dash='dash'),
                    marker=dict(size=8, color=warna, symbol=mkr,
                                line=dict(color='white', width=1.5)),
                    hovertemplate='<b>%{x|%Y-%m}</b><br>' + nama_model + ': %{y:,.0f}<extra></extra>'
                ))

                # Anotasi nilai pada titik prediksi saja — offset pixel tetap agar tidak melayang
                ay_px = {'Prophet': -28, 'ARIMA': 22, 'Holt-Winters': -48}
                ay_val = ay_px.get(nama_model, -20)

                for xp, yp in zip(dates_dt, y_pred):
                    if yp is not None:
                        fig_plotly.add_annotation(
                            x=xp,
                            y=yp,
                            text=f"<b>{int(round(yp)):,}</b>",
                            showarrow=True,
                            arrowhead=0,
                            arrowwidth=1,
                            arrowcolor=warna,
                            ax=0,
                            ay=ay_val,
                            ayref='pixel',
                            axref='pixel',
                            font=dict(size=8.5, color=warna),
                            bgcolor='rgba(255,255,255,0.88)',
                            bordercolor=warna,
                            borderwidth=1,
                            borderpad=2,
                            xanchor='center',
                            yanchor='middle'
                        )

            # --- Garis pemisah vertikal Historis | Prediksi ---
            if hist_dt:
                fig_plotly.add_vline(
                    x=hist_dt[-1].timestamp() * 1000,
                    line_width=1.5,
                    line_dash='dot',
                    line_color='#4f46e5',
                    opacity=0.7
                )
                fig_plotly.add_annotation(
                    x=hist_dt[-1],
                    y=1.0,
                    yref='paper',
                    text='← Historis | Prediksi →',
                    showarrow=False,
                    font=dict(size=9, color='#4f46e5', family='Inter'),
                    bgcolor='#e0e7ff',
                    bordercolor='#c7d2fe',
                    borderwidth=1,
                    borderpad=4,
                    xanchor='center',
                    yanchor='bottom'
                )

            # --- Layout & styling ---
            total_months = len(hist_dt) + len(dates_dt)
            fig_plotly.update_layout(
                title=dict(
                    text=(
                        f'Analisis Tren Historis vs Prediksi Volume Penumpang<br>'
                        f'<span style="font-size:13px;color:#64748b">'
                        f'{kategori_terpilih} \u2014 {jumlah_bulan} Bulan ke Depan</span>'
                    ),
                    font=dict(size=15, color='#0f172a', family='Inter'),
                    x=0.5, xanchor='center', y=0.97
                ),
                xaxis=dict(
                    title='Periode Waktu',
                    tickformat='%Y-%m',
                    dtick='M1' if total_months <= 18 else 'M2',
                    tickangle=-45,
                    showgrid=False,
                    linecolor='#cbd5e1',
                    tickfont=dict(size=10, color='#475569'),
                    title_font=dict(size=11, color='#334155')
                ),
                yaxis=dict(
                    title='Volume Penumpang',
                    tickformat=',',
                    showgrid=True,
                    gridcolor='rgba(203,213,225,0.5)',
                    gridwidth=1,
                    linecolor='#cbd5e1',
                    tickfont=dict(size=10, color='#475569'),
                    title_font=dict(size=11, color='#334155'),
                ),
                legend=dict(
                    orientation='v',
                    x=0.01, y=0.99,
                    xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#e2e8f0',
                    borderwidth=1,
                    font=dict(size=10)
                ),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                margin=dict(l=60, r=40, t=80, b=80),
                height=500,
                hovermode='x unified'
            )
            fig_plotly.update_xaxes(showline=True, mirror=False)
            fig_plotly.update_yaxes(showline=True, mirror=False)

            st.plotly_chart(fig_plotly, use_container_width=True)
            st.caption(
                "📌 **Keterangan:** Garis gelap solid = data aktual historis. "
                "Garis putus-putus berwarna = prediksi model AI ke depan. "
                "Wilayah berbayang biru muda di kanan = rentang waktu prediksi. "
                "Grafik interaktif: bisa di-zoom & di-hover untuk detail nilai."
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