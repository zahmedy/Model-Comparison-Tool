import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV dataset from a file path."""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")
