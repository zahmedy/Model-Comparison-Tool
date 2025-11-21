import streamlit as st
import pandas as pd
from src.data_loader import load_csv
from src.preprocess import split_data
from src.models import get_models

def main():
    st.title("🧪 Model Comparison Tool")
    st.write("Interactively compare multiple ML models using ROC, PR curves, and confusion matrices.")

    st.sidebar.header("Settings")
    st.sidebar.write("Model & data options will go here.")

    st.write("⬅️ Use the sidebar to configure models and data (coming next).")

    st.header("📊 Dataset Preview")

    data_path = "data/customer_segments.csv"
    df = load_csv(data_path)

    st.write("Loaded dataset:")
    st.dataframe(df)

    # st.write("DEBUG y dtype:", df["cluster_id"].dtype)
    # st.write(df["cluster_id"].unique())

    st.header("🤖 Model Comparison")

    # Sidebar control for test size
    test_size = st.sidebar.slider(
        "Test size (fraction)", 
        min_value=0.1, max_value=0.5, value=0.2, step=0.05
    )

    # Split data
    X_train, X_test, y_train, y_test, feature_cols = split_data(
        df,
        target_col="cluster_id",
        test_size=test_size
    )

    # Load models
    models = get_models()

    results = []

    st.subheader("📋 Model Results")
    for name, model in models.items():
        st.write(f"### {name}")

        # st.write("DEBUG model type:", type(model))

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy (for now — later: ROC, PR, Confusion Matrix, etc.)
        accuracy = (y_pred == y_test).mean()

        st.write(f"**Accuracy:** {accuracy:.3f}")
        st.write("---")


if __name__ == "__main__":
    main()