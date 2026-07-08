import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================

st.set_page_config(
    page_title="Sistem Rekomendasi Harga Mobil Bekas",
    page_icon="🚗",
    layout="wide"
)

# ==================================================
# LOAD DATASET
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("dataset_mobil_bekas.csv")

df = load_data()

# ==================================================
# MENAMPILKAN DATASET
# ==================================================

st.title("🚗 Sistem Rekomendasi Harga Mobil Bekas")

st.write("Metode K-Nearest Neighbor (KNN)")

# ==================================================
# MENU TAB
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "🚗 Prediksi Harga",
    "📊 Dataset",
    "ℹ️ Tentang"
])

with tab1:
# ==================================================
# PREPROCESSING
# ==================================================

    data = df.copy()

    # Membuat Label Encoder
    le_merek = LabelEncoder()
    le_bahan = LabelEncoder()
    le_transmisi = LabelEncoder()
    le_kondisi = LabelEncoder()

    # Encoding data kategori
    data["merek_kendaraan"] = le_merek.fit_transform(data["merek_kendaraan"])
    data["jenis_bahan_bakar"] = le_bahan.fit_transform(data["jenis_bahan_bakar"])
    data["jenis_transmisi"] = le_transmisi.fit_transform(data["jenis_transmisi"])
    data["kondisi_kendaraan"] = le_kondisi.fit_transform(data["kondisi_kendaraan"])

    #st.divider()
    #st.subheader("Dataset Setelah Label Encoding")

    #st.dataframe(data.head())


# ==================================================
# MENENTUKAN FITUR DAN TARGET
# ==================================================

    FEATURES = [
        "merek_kendaraan",
        "tahun_produksi",
        "jarak_tempuh",
        "jenis_bahan_bakar",
        "jenis_transmisi",
        "kondisi_kendaraan"
    ]

    TARGET = "harga_kendaraan"

    X = data[FEATURES]
    y = data[TARGET]


# ==================================================
# MEMBANGUN MODEL KNN
# ==================================================

    knn = KNeighborsRegressor(
        n_neighbors=5,
        metric="euclidean"
    )

    knn.fit(X, y)

    neighbor_model = NearestNeighbors(
        n_neighbors=5,
        metric="euclidean"
    )

    neighbor_model.fit(X)

    #st.success("Model KNN berhasil dibuat menggunakan Euclidean Distance (K = 5)")


# ==================================================
# FORM INPUT
# ==================================================

    st.divider()
    st.header("📝 Input Data Mobil")

    col1, col2 = st.columns(2)

    with col1:

        merek = st.selectbox(
            "Merek Kendaraan",
            sorted(df["merek_kendaraan"].unique())
        )

        tahun = st.number_input(
            "Tahun Produksi",
            min_value=1990,
            max_value=2030,
            value=2020
        )

        jarak = st.number_input(
            "Jarak Tempuh (KM)",
            min_value=0,
            value=50000
        )

    with col2:

        bahan = st.selectbox(
            "Jenis Bahan Bakar",
            sorted(df["jenis_bahan_bakar"].unique())
        )

        transmisi = st.selectbox(
            "Jenis Transmisi",
            sorted(df["jenis_transmisi"].unique())
        )

        kondisi = st.selectbox(
            "Kondisi Kendaraan",
            sorted(df["kondisi_kendaraan"].unique())
        )

    harga_penjual = st.number_input(
        "Harga yang Diinginkan Penjual (Rp)",
        min_value=0,
        value=200000000,
        step=1000000
    )

    prediksi = st.button("🚗 Hitung Rekomendasi Harga")



# ==================================================
# PREDIKSI HARGA
# ==================================================

    if prediksi:

        # Encode input user
        input_data = pd.DataFrame({
            "merek_kendaraan":[le_merek.transform([merek])[0]],
            "tahun_produksi":[tahun],
            "jarak_tempuh":[jarak],
            "jenis_bahan_bakar":[le_bahan.transform([bahan])[0]],
            "jenis_transmisi":[le_transmisi.transform([transmisi])[0]],
            "kondisi_kendaraan":[le_kondisi.transform([kondisi])[0]]
        })

        # Prediksi harga
        harga_prediksi = knn.predict(input_data)[0]

        # Cari tetangga terdekat
        distance, index = neighbor_model.kneighbors(input_data)

        tetangga = df.iloc[index[0]].copy()

        tetangga["Euclidean Distance"] = distance[0]

    # ==========================
    # HASIL
    # ==========================

        st.divider()

        st.header("📊 Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Harga Penjual",
                f"Rp {harga_penjual:,.0f}"
            )

        with col2:

            st.metric(
                "Harga Rekomendasi",
                f"Rp {harga_prediksi:,.0f}"
            )

    # ==========================
    # SELISIH
    # ==========================

        selisih = harga_penjual - harga_prediksi

        persen = abs(selisih) / harga_prediksi * 100

        st.write("---")

        st.subheader("Analisis Harga")

        st.write(f"Selisih Harga : **Rp {selisih:,.0f}**")

        st.write(f"Persentase Selisih : **{persen:.2f}%**")

    # ==========================
    # STATUS
    # ==========================

        if abs(selisih) <= 5000000:

            st.success("🟢 Harga Wajar")

        elif selisih > 5000000:

            st.warning("🟡 Harga Penjual Terlalu Mahal")

        else:

            st.info("🔵 Harga Penjual Lebih Murah")

    # ==========================
    # TABEL TETANGGA
    # ==========================

        st.write("---")

        st.subheader("🚗 5 Mobil Paling Mirip")

        st.dataframe(
            tetangga[
                [
                    "merek_kendaraan",
                    "tahun_produksi",
                    "jarak_tempuh",
                    "harga_kendaraan",
                    "Euclidean Distance"
                ]
            ],
            use_container_width=True
        )

with tab2:

    st.header("📊 Dataset Mobil Bekas")

    st.write(f"Jumlah Dataset : {len(df)} Data")

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

with tab3:

    st.header("ℹ️ Tentang Sistem")

    st.markdown("""
## Sistem Rekomendasi Harga Mobil Bekas

Dibuat untuk membantu pengguna memperoleh kisaran harga mobil bekas berdasarkan spesifikasi kendaraan menggunakan metode **K-Nearest Neighbor (KNN)**.

### Metode

- K-Nearest Neighbor Regression
- Euclidean Distance
- Nilai K = 5

### Dataset

Dataset terdiri dari atribut:

- Merek Kendaraan
- Tahun Produksi
- Jarak Tempuh
- Jenis Bahan Bakar
- Jenis Transmisi
- Kondisi Kendaraan
- Harga Kendaraan

### Tujuan

Memberikan rekomendasi harga mobil bekas sebagai acuan bagi penjual maupun pembeli sehingga harga yang ditentukan lebih sesuai dengan kondisi kendaraan.

---

Universitas Amikom Yogyakarta

Program Studi S1 Sistem Informasi
""")
