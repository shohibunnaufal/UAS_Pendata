# import library yang dibutuhkan 

import streamlit as st
from web_functions import load_data

from Tabs import home, predict,visualise

Tabs = {
    "Home" : home,
    "Prediction" : predict,
    "Visualisation" : visualise
}

# membuat sidebar
st.sidebar.title("Navigasi")

# membuat radio option
page = st.sidebar.radio("Pages",list(Tabs.keys()))

#kondisi call app fuction
if page in ["Prediction","Visualisation"]:
    Tabs[page].app(df,x,y)

else :
    Tabs[page].app()
