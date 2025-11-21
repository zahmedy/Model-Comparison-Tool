from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



def get_models():
    """
    Returns a dictionary of model name -> sklearn Pipeline.
    """

    models = {}

    # 1. Logistic Regression (Linear)
    models["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            multi_class="multinomial"
        ))
    ])

    # 2. Random Fprest (Non-leaner, tree-based)
    models["Random Forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])

    return models