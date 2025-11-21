import streamlit as st
import pandas as pd
from src.data_loader import load_csv
from src.preprocess import split_data
from src.models import get_models
from sklearn.metrics import confusion_matrix, roc_auc_score
from src.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall



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

    # Choose which class to treat as "positive" for ROC/PR
    all_classes = sorted(df["cluster_id"].unique().tolist())
    positive_class = st.sidebar.selectbox(
        "Positive class for ROC/PR (one-vs-rest)",
        all_classes,
        index=0
    )


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

        # ---- ROC & PR (one-vs-rest for selected class) ----
        clf = model.named_steps["clf"]

        if hasattr(clf, "predict_proba"):
            y_proba_all = clf.predict_proba(X_test)

            # find index of the selected positive class in clf.classes_
            classes_ = list(clf.classes_)
            pos_idx = classes_.index(positive_class)

            # probability of the positive class
            y_prob = y_proba_all[:, pos_idx]

            # binary labels: 1 if this is positive_class, else 0
            y_test_binary = (y_test == positive_class).astype(int)

            st.subheader(f"ROC Curve — {name} (class {positive_class} vs rest)")
            roc_fig = plot_roc_curve(y_test_binary, y_prob, title=f"ROC — {name}")
            st.pyplot(roc_fig)

            st.subheader(f"Precision-Recall Curve — {name} (class {positive_class} vs rest)")
            pr_fig = plot_precision_recall(y_test_binary, y_prob, title=f"PR — {name}")
            st.pyplot(pr_fig)
        else:
            st.write("Probability scores not available for this model.")


        # Accuracy (for now — later: ROC, PR, Confusion Matrix, etc.)
        accuracy = (y_pred == y_test).mean()
        st.write(f"**Accuracy:** {accuracy:.3f}")

         # ---- Metrics for comparison table ----
        auc_value = None
        if hasattr(clf, "predict_proba"):
            auc_value = roc_auc_score(y_test_binary, y_prob)

        results.append({
            "Model": name,
            "Accuracy": float(accuracy),
            "AUC (class {} vs rest)".format(positive_class): auc_value
        })

        # Confusion matrix heatmap
        cm = confusion_matrix(y_test, y_pred)
        class_names = sorted(y_test.unique().tolist())

        fig = plot_confusion_matrix(
            cm,
            class_name=class_names,
            title=f"Confusion Matrix - {name}"
        )
        st.pyplot(fig)

        st.write("---")
    
    # ----- Summary table -----
    if results:
        st.subheader("📊 Model comparison summary")
        results_df = pd.DataFrame(results)

        # Optional: highlight best values
        st.dataframe(
            results_df.style.highlight_max(
                subset=["Accuracy", f"AUC (class {positive_class} vs rest)"],
                color="lightgreen"
            )
        )


if __name__ == "__main__":
    main()