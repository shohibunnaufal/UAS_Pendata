# import library yang dibutuhkan 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

tab1, tab2, tab3 = st.tabs(["Home", "Prediction", "Visualisation"])

with tab1:
    st.title("Aplikasi Prediksi Penyakit Batu Ginjal")
    
with tab2:
    def app(df,x,y):
        st.title("Halaman Prediksi")
        col1,col2, col3 = st.columns(3)
        with col1:
            bp = st.text_input('Input Nilai bp :')
            sg = st.text_input('Input Nilai sg :')
            al = st.text_input('Input Nilai al :')
            su = st.text_input('Input Nilai su :')
            rbc = st.text_input('Input Nilai rbc :')
            pc = st.text_input('Input Nilai pc :')
            pcc = st.text_input('Input Nilai pcc :')
            ba = st.text_input('Input Nilai ba :')

        with col2:
            bgr = st.text_input('Input Nilai bgr :')
            bu = st.text_input('Input Nilai bu :')
            sc = st.text_input('Input Nilai sc :')
            sod = st.text_input('Input Nilai sod :')
            pot = st.text_input('Input Nilai pot :')
            hemo = st.text_input('Input Nilai hemo :')
            pcv = st.text_input('Input Nilai pcv :')
            wc = st.text_input('Input Nilai wc :')

        with col3:
            rc = st.text_input('Input Nilai rc :')
            htn = st.text_input('Input Nilai htn :')
            dm = st.text_input('Input Nilai dm :')
            cad = st.text_input('Input Nilai cad :')
            appet = st.text_input('Input Nilai appet :')
            pe = st.text_input('Input Nilai pe :')
            ane = st.text_input('Input Nilai ane :')

        features = [bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]

        #tombol prediksi
        if st.button("Prediksi"):
            prediction,score = predict(x,y,features)
            score = score
            st.info("Prediksi Sukses...")

            if (prediction ==1):
                st.warning("Orang tersebut rentan terkena penyakit ginjal")
            else:
                st.success("Orang tersebut relatif aman dari penyakit ginjal")

            st.write("Model yang digunakan memiliki tingkat akurasi",(score*100),"%")

with tab3:
    def app(df,x,y):
        warnings.filterwarnings('ignore')
        st.set_option('deprecation.showPyplotGlobaluse', False)

        st.title("Visualisasi Prediksi Batu Ginjal")

        if st.checkbox("Plot Confusion Matrix"):
            model, score = train_model(x, y)
            plt.figure(figsize=(10, 6))
            plot_confusion_matrix(model, x, y, values_format='d')
            st.pyplot()

        if st.checkbox("Plot Decision Tree"):
            model, score = train_model(x, y)
            dot_data = tree.export_graphviz(
                decision_tree=model,
                max_depth=3,
                out_file=None,
                filled=True,
                rounded=True,
                feature_names=x.columns,
                class_names=['nockd', 'ckd']
            )
            st.graphviz_chart(dot_data)


@st.cache()
def load_data():

    #load dataset
    df = read_csv_file('winequality-red.csv')

    x = df[["bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"]]
    y = df[['classification']]

    return df,x,y

@st.cache()
def train_model(x,y):
    model = DecisionTreeClassifier(
            ccp_alpha=0.0,class_weight=None,criterion="entropy",
            max_depth=4,max_features=None,max_leaf_nodes=None,
            min_impurity_decrease=0.0,min_samples_leaf=1,
            min_samples_split=2,min_weight_fraction_leaf=0.0,
            random_state=42,splitter='best'
        )

    model.fit(x,y)
    score = model.score(x,y)
    return model,score

def predict(x,y,freatures):
    model,score = train_model(x,y)

    prediction = model.predict(np.array(freatures).reshape(1,-1))
    return prediction,score

import streamlit as st
from web_functions import load_data

from Tabs import home, predict,visualise

#kondisi call app fuction
if page in ["Prediction","Visualisation"]:
    Tabs[page].app(df,x,y)

else :
    Tabs[page].app()
