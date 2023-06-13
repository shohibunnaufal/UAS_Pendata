import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# Fungsi untuk membaca file CSV
def read_csv_file(file):
    df = pd.read_csv(file)
    return df

# Fungsi untuk melakukan prediksi
def predict(x, y, features):
    model, score = train_model(x, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score

# Fungsi untuk melatih model
@st.cache()
def train_model(x, y):
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion="entropy",
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )
    model.fit(x, y)
    score = model.score(x, y)
    return model, score

# Tampilan Streamlit
def main():
    tab1, tab2, tab3 = st.columns(3)
    with tab1:
        st.title("Aplikasi Prediksi Penyakit Batu Ginjal")

    with tab2:
        st.title("Halaman Prediksi")
        # Definisikan input fitur di sini

    with tab3:
        st.title("Visualisasi Prediksi Batu Ginjal")
        # Tambahkan visualisasi di sini

    # Load dataset
    df = pd.read_csv('winequality-red.csv')
    x = df[["bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"]]
    y = df[['classification']]

    # Tombol prediksi
    if st.button("Prediksi"):
        features = [bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]
        prediction, score = predict(x, y, features)
        score = score * 100
        st.info("Prediksi Sukses...")

        if prediction == 1:
            st.warning("Orang tersebut rentan terkena penyakit ginjal")
        else:
            st.success("Orang tersebut relatif aman dari penyakit ginjal")

        st.write("Model yang digunakan memiliki tingkat akurasi", score, "%")

if __name__ == "__main__":
    main()
