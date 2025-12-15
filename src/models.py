from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_models():
    """Return a dictionary of model name -> sklearn Pipeline."""
    models = {}

    # 1. Logistic Regression (linear baseline)
    models["Logistic Regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="multinomial",
                ),
            ),
        ]
    )

    # 2. Random Forest (non-linear, tree-based)
    models["Random Forest"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                ),
            ),
        ]
    )

    # 3. Gradient Boosting (additive trees, good for tabular data)
    models["Gradient Boosting"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(random_state=42),
            ),
        ]
    )

    return models
