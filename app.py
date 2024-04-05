import streamlit as st
import pandas as pd
import os 
from pydantic_settings import BaseSettings 

# Import profiling capabilities
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

# Machine learning 
from pycaret.classification import setup, compare_models, pull, save_model

def convert_columns_to_numeric(df, column_names):
    """
    Attempts to convert columns in the dataframe to numeric types.
    Non-convertible values will be set to NaN and then imputed or handled accordingly.
    """
    for col in column_names:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def handle_non_numeric_columns(df):
    """
    Identifies non-numeric columns and attempts to convert them to numeric if possible.
    This function is useful for preprocessing before machine learning tasks.
    """
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
    
    # Attempt to convert non-numeric columns to numeric where it makes sense
    df = convert_columns_to_numeric(df, non_numeric_columns)
    
    # Optionally, handle NaN values after conversion
    # For simplicity, this example fills NaNs with column medians
    # You may choose a more appropriate strategy for your data
    df.fillna(df.median(), inplace=True)
    
    return df

# Sidebar navigation
with st.sidebar:
    st.image("img/logo.png")  # Make sure the path to the logo is correct
    st.title("Aserious Auto Stream Lit ML")
    choice = st.radio("Navigation", ["Data uploading", "Data profiling", "Machine learning", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, pandas profiling, and PyCaret")

# Data uploading
if choice == "Data uploading":
    file = st.file_uploader("Please provide and upload your dataset (csv format)")
    if file:
        df = pd.read_csv(file)
        # Save the original dataframe for later use
        df.to_csv("original.csv", index=False)
        st.dataframe(df)

# Load the dataset if it exists
if os.path.exists("original.csv"):
    df = pd.read_csv("original.csv")

# Data profiling
if choice == "Data profiling":
    st.title("Exploratory Data Analysis")
    # Ensure df is defined to avoid NameError
    if 'df' in locals() or 'df' in globals():
        profile_report = ProfileReport(df, explorative=True)
        st_profile_report(profile_report)

# Machine learning
if choice == "Machine learning":
    st.title("Machine Learning")
    if 'df' in locals() or 'df' in globals():
        # Preprocess the dataframe to ensure numeric data where possible
        df_processed = handle_non_numeric_columns(df)
        target = st.selectbox("Select your target", df_processed.columns)
        
        if st.button("Train model"):
            setup_data = setup(df_processed, target=target, remove_outliers=True,  html=False)
            best_model = compare_models()
            compare_df = pull()
            st.info("ML experiment settings")
            st.dataframe(compare_df)
            st.write("Best Model:", best_model)
            save_model(best_model, "best_model.pkl")

# Download
if choice == "Download":
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download the model", f, "best_model.pkl")
    else:
        st.error("Model not found. Please train a model first.")