import streamlit as st
import pandas as pd

def main():
    st.title("🧪 Model Comparison Tool")
    st.write("Interactively compare multiple ML models using ROC, PR curves, and confusion matrices.")

    st.sidebar.header("Settings")
    st.sidebar.write("Model & data options will go here.")

    st.write("⬅️ Use the sidebar to configure models and data (coming next).")

    st.header("📊 Dataset Preview")

    uploaded_path = "data/customer_segments.csv"  # your uploaded file
    df = pd.read_csv(uploaded_path)

    st.write("Loaded dataset:")
    st.dataframe(df)

if __name__ == "__main__":
    main()