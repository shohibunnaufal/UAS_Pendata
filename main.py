# import library yang dibutuhkan 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

Tabs = {
    "Home" : home,
    "Prediction" : predict,
    "Visualisation" : visualise
}

# membuat sidebar
st.sidebar.title("Navigasi")

# membuat radio option
page = st.sidebar.radio("Pages",list(Tabs.keys()))

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
