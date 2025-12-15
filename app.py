import streamlit as st
import pandas as pd
from src.data_loader import load_csv
from src.preprocess import split_data
from src.models import get_models
from sklearn.metrics import confusion_matrix, roc_auc_score
from src.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall


st.set_page_config(
    page_title="Model Comparison Tool",
    page_icon="🧪",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dataframe(uploaded_file):
    """Load uploaded CSV or fall back to sample data."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_csv("data/customer_segments.csv")


def render_dataset_overview(df, target_col):
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Features", f"{len(df.columns) - 1}")
    col3.metric("Target", target_col)

    st.subheader("Dataset preview")
    st.dataframe(df.head(25))

    st.subheader("Target distribution")
    st.bar_chart(df[target_col].value_counts().sort_index())


def main():
    st.title("🧪 Model Comparison Tool")
    st.caption("Evaluate multiple ML models with ROC, PR curves, confusion matrices, and summary metrics.")

    models = get_models()

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], help="If empty, a sample customer segmentation dataset is used.")
        df = load_dataframe(uploaded)

        target_col = st.selectbox("Target column", options=df.columns, index=list(df.columns).index("cluster_id") if "cluster_id" in df.columns else len(df.columns) - 1)
        available_classes = sorted(df[target_col].unique().tolist())
        if len(available_classes) < 2:
            st.error("Target column needs at least two unique classes.")
            return
        positive_class = st.selectbox("Positive class for ROC/PR (one-vs-rest)", available_classes, index=0)

        st.header("Experiment")
        test_size = st.slider("Test size (fraction)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random seed", value=42, min_value=0, max_value=10_000, step=1)
        selected_models = st.multiselect(
            "Models to compare",
            options=list(models.keys()),
            default=list(models.keys()),
        )

    if not selected_models:
        st.warning("Select at least one model to run the comparison.")
        return

    st.header("📊 Dataset")
    render_dataset_overview(df, target_col)

    st.header("🤖 Model comparison")
    run = st.button("Run comparison", type="primary")

    if not run:
        st.info("Choose your settings in the sidebar and click **Run comparison**.")
        return

    with st.spinner("Training and evaluating models..."):
        X_train, X_test, y_train, y_test, feature_cols = split_data(
            df,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
        )

        results = []

        for name in selected_models:
            model = models[name]
            st.subheader(f"{name}")
            st.caption(f"Features: {', '.join(feature_cols)}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            clf = model.named_steps["clf"]

            auc_value = None
            y_test_binary = (y_test == positive_class).astype(int)
            if hasattr(clf, "predict_proba"):
                y_proba_all = clf.predict_proba(X_test)
                classes_ = list(clf.classes_)
                pos_idx = classes_.index(positive_class)
                y_prob = y_proba_all[:, pos_idx]

                st.markdown("**ROC curve**")
                roc_fig = plot_roc_curve(y_test_binary, y_prob, title=f"ROC — {name}")
                st.pyplot(roc_fig, use_container_width=True)

                st.markdown("**Precision-Recall curve**")
                pr_fig = plot_precision_recall(y_test_binary, y_prob, title=f"PR — {name}")
                st.pyplot(pr_fig, use_container_width=True)

                auc_value = roc_auc_score(y_test_binary, y_prob)
            else:
                st.info("Probability scores not available for this model.")

            accuracy = float((y_pred == y_test).mean())
            st.write(f"**Accuracy:** {accuracy:.3f}")

            cm = confusion_matrix(y_test, y_pred)
            class_names = sorted(y_test.unique().tolist())
            fig = plot_confusion_matrix(cm, class_name=class_names, title=f"Confusion Matrix - {name}")
            st.pyplot(fig, use_container_width=True)

            results.append(
                {
                    "Model": name,
                    "Accuracy": accuracy,
                    f"AUC (class {positive_class} vs rest)": auc_value,
                }
            )

            st.divider()

    if results:
        st.subheader("Comparison summary")
        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        st.dataframe(
            results_df.style.highlight_max(
                subset=["Accuracy", f"AUC (class {positive_class} vs rest)"],
                color="lightgreen",
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
