# app_combined.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn 
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfigurasi Halaman dan Sidebar Umum ---
st.set_page_config(page_title="Analisis Audit & AI", layout="wide")

st.sidebar.header("Pilih Analisis:")
analysis_type = st.sidebar.radio(
    "Pilih jenis analisis yang ingin Anda lakukan:",
    ('Prediksi Penggunaan AI (Regresi Logistik)', 'Segmentasi Profil Audit (K-Means)')
)

st.sidebar.markdown("---") # Pemisah
st.sidebar.info(f"Versi Scikit-learn di app: {sklearn.__version__}")
st.sidebar.warning("Pastikan versi Scikit-learn ini SAMA dengan versi di notebook tempat model disimpan untuk menghindari error saat load model.")
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini menyediakan dua jenis analisis:
1.  **Prediksi Penggunaan AI**: Menggunakan model Regresi Logistik untuk memprediksi apakah AI digunakan dalam audit.
2.  **Segmentasi Profil Audit**: Menggunakan K-Means Clustering untuk mengelompokkan data audit.
""")
st.sidebar.header("Dataset & Model")
st.sidebar.info("""
Dataset yang digunakan: `big4_financial_risk_compliance.csv`.
Pastikan semua file model (.joblib) berada di direktori yang sama dengan aplikasi ini dan dihasilkan dari `Main.ipynb` dengan versi pustaka yang sesuai.
""")


# --- Fungsi untuk Memuat Model (Mengurangi Redundansi) ---
@st.cache_resource
def load_logistic_regression_model():
    try:
        pipeline = joblib.load("logistic_regression_pipeline.joblib")
        firms = joblib.load("unique_firms.joblib")
        industries = joblib.load("unique_industries.joblib")
        return pipeline, firms, industries
    except FileNotFoundError:
        st.error("File model Regresi Logistik atau pendukungnya tidak ditemukan. Jalankan 'Main.ipynb'.")
        return None, None, None
    except Exception as e:
        st.error(f"Error memuat model Regresi Logistik: {e}. Pastikan versi scikit-learn sesuai.")
        return None, None, None

@st.cache_resource
def load_kmeans_model_and_data():
    try:
        model = joblib.load("kmeans_model.joblib")
        scaler = joblib.load("kmeans_scaler.joblib")
        df_original_for_plot = pd.read_csv('big4_financial_risk_compliance.csv')
        return model, scaler, df_original_for_plot
    except FileNotFoundError:
        st.error("File model K-Means, scaler, atau dataset asli tidak ditemukan. Pastikan semua file ada dan 'Main.ipynb' sudah dijalankan.")
        return None, None, None
    except Exception as e:
        st.error(f"Error memuat model K-Means/Scaler/Data: {e}. Pastikan versi scikit-learn sesuai.")
        return None, None, None

# --- Logika untuk Analisis Regresi Logistik (Supervised) ---
def supervised_analysis():
    logreg_pipeline, unique_firms, unique_industries = load_logistic_regression_model()

    if not logreg_pipeline or not unique_firms or not unique_industries:
        st.warning("Model Regresi Logistik tidak dapat dimuat. Fungsi prediksi tidak tersedia.")
        return

    st.title("🤖 Prediksi Penggunaan AI dalam Audit (Regresi Logistik)")
    st.markdown("""
    Aplikasi ini memprediksi kemungkinan penggunaan **Artificial Intelligence (AI)** dalam proses audit berdasarkan berbagai faktor.
    Masukkan nilai-nilai di bawah ini untuk mendapatkan prediksi.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Data Karyawan & Audit:")
        employee_workload_sup = st.slider("Beban Kerja Karyawan (Employee_Workload)", min_value=0, max_value=100, value=60, step=1, key="sup_workload")
        audit_effectiveness_score_sup = st.slider("Skor Efektivitas Audit (Audit_Effectiveness_Score)", min_value=0.0, max_value=10.0, value=7.5, step=0.1, key="sup_score")

        st.subheader("Input Data Keuangan & Risiko:")
        total_revenue_impact_sup = st.slider("Dampak Pendapatan Total (Total_Revenue_Impact)", min_value=0.0, max_value=500.0, value=270.0, step=10.0, key="sup_revenue")
        high_risk_cases_sup = st.slider("Kasus Berisiko Tinggi (High_Risk_Cases)", min_value=0, max_value=500, value=280, step=10, key="sup_risk")
        compliance_violations_sup = st.slider("Pelanggaran Kepatuhan (Compliance_Violations)", min_value=0, max_value=200, value=105, step=5, key="sup_compliance")

    with col2:
        st.subheader("Input Data Perusahaan & Industri:")
        firm_name_options_sup = ["Pilih Firma"] + unique_firms
        firm_name_default_index_sup = 0
        firm_name_sup = st.selectbox("Nama Firma Audit (Firm_Name)", options=firm_name_options_sup, index=firm_name_default_index_sup, key="sup_firm")

        industry_affected_options_sup = ["Pilih Industri"] + unique_industries
        industry_affected_default_index_sup = 0
        industry_affected_sup = st.selectbox("Industri yang Terdampak (Industry_Affected)", options=industry_affected_options_sup, index=industry_affected_default_index_sup, key="sup_industry")

    if st.button("🔮 Prediksi Penggunaan AI", key="sup_predict_button"):
        if firm_name_sup == "Pilih Firma" or industry_affected_sup == "Pilih Industri":
            st.warning("Harap pilih Nama Firma dan Industri yang valid.")
        else:
            workload_x_effectiveness_sup = float(employee_workload_sup) * audit_effectiveness_score_sup

            input_data_sup = {
                'Employee_Workload': [float(employee_workload_sup)],
                'Audit_Effectiveness_Score': [audit_effectiveness_score_sup],
                'Total_Revenue_Impact': [total_revenue_impact_sup],
                'High_Risk_Cases': [float(high_risk_cases_sup)],
                'Compliance_Violations': [float(compliance_violations_sup)],
                'Workload_x_Effectiveness': [workload_x_effectiveness_sup],
                'Firm_Name': [firm_name_sup],
                'Industry_Affected': [industry_affected_sup]
            }
            feature_names_ordered_sup = [
                'Employee_Workload', 'Audit_Effectiveness_Score', 'Total_Revenue_Impact',
                'High_Risk_Cases', 'Compliance_Violations', 'Workload_x_Effectiveness',
                'Firm_Name', 'Industry_Affected'
            ]
            input_df_sup = pd.DataFrame(input_data_sup)[feature_names_ordered_sup]

            try:
                prediction_sup = logreg_pipeline.predict(input_df_sup)[0]
                prediction_proba_sup = logreg_pipeline.predict_proba(input_df_sup)[0]

                st.subheader("Hasil Prediksi:")
                if prediction_sup == 1:
                    st.success(f"✅ **AI kemungkinan DIGUNAKAN** dalam proses audit ini.")
                    st.write(f"   Probabilitas AI Digunakan (Yes): {prediction_proba_sup[1]*100:.2f}%")
                else:
                    st.error(f"⚠️ **AI kemungkinan TIDAK DIGUNAKAN** dalam proses audit ini.")
                    st.write(f"   Probabilitas AI Tidak Digunakan (No): {prediction_proba_sup[0]*100:.2f}%")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# --- Logika untuk Analisis K-Means Clustering (Unsupervised) ---
def unsupervised_analysis():
    kmeans_model, kmeans_scaler, df_original_for_plot = load_kmeans_model_and_data()

    if not kmeans_model or not kmeans_scaler or df_original_for_plot is None:
        st.warning("Model K-Means, scaler, atau data asli tidak dapat dimuat. Fungsi segmentasi tidak tersedia.")
        return

    st.title("📊 Segmentasi Profil Audit (K-Means Clustering)")

    features_kmeans = ['Employee_Workload', 'Audit_Effectiveness_Score']
    cluster_descriptions = {
        0: "**Profil Cluster 0:** Cenderung memiliki Beban Kerja Karyawan yang lebih rendah dengan Skor Efektivitas Audit yang sedang hingga tinggi. Kelompok ini mungkin mewakili audit yang efisien atau area dengan risiko yang lebih terkendali.",
        1: "**Profil Cluster 1:** Cenderung memiliki Beban Kerja Karyawan yang paling tinggi dan juga Skor Efektivitas Audit yang paling tinggi. Ini bisa jadi tim/audit berkinerja tinggi yang menangani banyak pekerjaan dengan baik, atau area yang kompleks namun dikelola dengan efektif.",
        2: "**Profil Cluster 2:** Cenderung memiliki Beban Kerja Karyawan yang tinggi namun dengan Skor Efektivitas Audit yang paling rendah. Kelompok ini mungkin memerlukan perhatian lebih, potensi adanya inefisiensi, atau area dengan tantangan khusus yang mempengaruhi efektivitas."
    }

    # Inisialisasi session state jika belum ada
    if 'user_point_coords_unsup' not in st.session_state:
        st.session_state.user_point_coords_unsup = None
    if 'user_point_cluster_unsup' not in st.session_state:
        st.session_state.user_point_cluster_unsup = None
    if 'predicted_cluster_text_unsup' not in st.session_state:
        st.session_state.predicted_cluster_text_unsup = ""
    if 'cluster_description_text_unsup' not in st.session_state:
        st.session_state.cluster_description_text_unsup = ""


    # --- Fungsi untuk membuat dan menampilkan plot ---
    def display_kmeans_plot(user_coords=None, user_cluster_label=None):
        st.subheader("Visualisasi Cluster:")
        X_plot_original = df_original_for_plot[features_kmeans].copy()
        X_plot_scaled = kmeans_scaler.transform(X_plot_original)
        all_cluster_labels = kmeans_model.predict(X_plot_scaled)
        centroids = kmeans_scaler.inverse_transform(kmeans_model.cluster_centers_)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            x=X_plot_original.iloc[:, 0], y=X_plot_original.iloc[:, 1],
            hue=all_cluster_labels, palette='viridis', s=100, alpha=0.7, ax=ax, legend='full'
        )
        ax.scatter(
            centroids[:, 0], centroids[:, 1], marker='X', s=250, color='red',
            label='Centroids', edgecolor='black'
        )
        if user_coords and user_cluster_label is not None:
            ax.scatter(
                user_coords[0], user_coords[1], marker='P', s=300, color='lime',
                label=f'Input Anda (Cluster {user_cluster_label})', edgecolor='black', zorder=5
            )
        ax.set_title(f'K-Means Clustering (k={kmeans_model.n_clusters})')
        ax.set_xlabel(features_kmeans[0])
        ax.set_ylabel(features_kmeans[1])
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # --- Tampilkan plot di bagian atas ---
    # Plot akan menggunakan data dari session_state jika ada, atau None jika belum ada input pengguna
    display_kmeans_plot(st.session_state.user_point_coords_unsup, st.session_state.user_point_cluster_unsup)

    st.markdown("---") # Pemisah sebelum input
    st.markdown("""
    Aplikasi ini membantu Anda mengidentifikasi segmen atau cluster dari suatu kasus audit (atau profil auditor) berdasarkan **Beban Kerja Karyawan** dan **Skor Efektivitas Audit**.
    Masukkan nilai untuk kedua fitur di bawah ini untuk melihat ke cluster mana data tersebut masuk dan bagaimana posisinya di plot.
    """)

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        employee_workload_unsup = st.slider("Beban Kerja Karyawan (Employee_Workload)", min_value=0, max_value=100, value=st.session_state.get('unsup_workload_val', 50), step=1, key="unsup_workload_slider")
    with col_input2:
        audit_effectiveness_score_unsup = st.slider("Skor Efektivitas Audit (Audit_Effectiveness_Score)", min_value=0.0, max_value=10.0, value=st.session_state.get('unsup_score_val', 6.0), step=0.1, key="unsup_score_slider")


    if st.button("🔍 Tentukan Cluster & Perbarui Plot", key="unsup_cluster_button"):
        st.session_state.unsup_workload_val = employee_workload_unsup # Simpan nilai slider untuk persistensi
        st.session_state.unsup_score_val = audit_effectiveness_score_unsup

        current_user_input_coords = [float(employee_workload_unsup), audit_effectiveness_score_unsup]
        st.session_state.user_point_coords_unsup = current_user_input_coords # Simpan koordinat untuk diplot ulang

        input_data_kmeans_df = pd.DataFrame([current_user_input_coords], columns=features_kmeans)
        try:
            input_data_kmeans_scaled = kmeans_scaler.transform(input_data_kmeans_df)
            predicted_cluster = kmeans_model.predict(input_data_kmeans_scaled)[0]
            st.session_state.user_point_cluster_unsup = predicted_cluster # Simpan cluster untuk diplot ulang

            # Simpan hasil teks ke session state untuk ditampilkan setelah rerun
            st.session_state.predicted_cluster_text_unsup = f"Data input masuk ke **Cluster {predicted_cluster}**."
            if predicted_cluster in cluster_descriptions:
                st.session_state.cluster_description_text_unsup = cluster_descriptions[predicted_cluster]
            else:
                st.session_state.cluster_description_text_unsup = f"Deskripsi untuk Cluster {predicted_cluster} belum tersedia."
            
            # Tidak perlu panggil display_kmeans_plot lagi di sini, karena Streamlit akan rerun dan plot di atas akan update
            # Cukup trigger rerun jika belum otomatis (biasanya st.button sudah trigger rerun)
            # st.experimental_rerun() # Biasanya tidak perlu jika button ditekan

        except Exception as e:
            st.error(f"Terjadi kesalahan saat menentukan cluster: {e}")
            st.session_state.predicted_cluster_text_unsup = "" # Reset jika error
            st.session_state.cluster_description_text_unsup = ""


    # Tampilkan hasil teks dari session state (akan muncul setelah button ditekan dan script rerun)
    if st.session_state.predicted_cluster_text_unsup:
        st.subheader("Hasil Segmentasi:")
        st.success(st.session_state.predicted_cluster_text_unsup)
        st.markdown(st.session_state.cluster_description_text_unsup)


# --- Kontrol Utama Aplikasi Berdasarkan Pilihan Pengguna ---
if analysis_type == 'Prediksi Penggunaan AI (Regresi Logistik)':
    # Reset state unsupervised jika beralih mode agar plot kembali ke default
    if 'user_point_coords_unsup' in st.session_state: st.session_state.user_point_coords_unsup = None
    if 'user_point_cluster_unsup' in st.session_state: st.session_state.user_point_cluster_unsup = None
    if 'predicted_cluster_text_unsup' in st.session_state: st.session_state.predicted_cluster_text_unsup = ""
    if 'cluster_description_text_unsup' in st.session_state: st.session_state.cluster_description_text_unsup = ""
    supervised_analysis()
elif analysis_type == 'Segmentasi Profil Audit (K-Means)':
    unsupervised_analysis()
