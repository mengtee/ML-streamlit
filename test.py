import streamlit as st
import pandas as pd
import os 
from pydantic_settings import BaseSettings 

# Import profiling capabilities
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

# Machine learning 
from pycaret.classification import setup, compare_models, pull, save_model
# from pycaret.clustering import *


with st.sidebar:
    st.image("img/logo.png")
    st.title("Aserious Auto Stream Lit ML")
    choice = st.radio("Navigation", ["Data uploading", "Data profiling", "Machine learning", "Download"])
    st.info("This application allow you to build an automated ML pipeline using streamlit, pandas profiling and pyCaret")


if choice == "Data uploading":
    file = st.file_uploader("Please provide and upload your dataset (csv format)")
    if file:
        df = pd.read_csv(file, index_col= None)
        df.to_csv("original.csv", index =None)
        st.dataframe(df)

# Checking condition for the source data
if os.path.exists("original.csv"):
    df =pd.read_csv("original.csv", index_col = None)

if choice == "Data profiling":
    st.title("Exploratory Data analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice =="Machine learning":
    st.title("Machine Learning")
    target = st.selectbox("Select your target", df.columns)
    if st.button("Train model"):
        setup(df, target=target,  remove_outliers = True)
        setup_df = pull()
        st.info("ML experiment settings")
        st.dataframe(setup_df)
        best_model =compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")

if choice == "Download":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the file", f, "best_model.pkl")

