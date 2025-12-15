# Model Comparison Tool
Streamlit app for comparing classification models on tabular data with ROC/PR curves, confusion matrices, and summary metrics.

## Features
- Upload your own CSV or use the provided `customer_segments.csv` sample dataset.
- Choose target column and positive class; one-vs-rest ROC/AUC and PR curves generated automatically.
- Adjustable train/test split and random seed for reproducibility.
- Compare multiple models side-by-side (Logistic Regression, Random Forest, Gradient Boosting) with accuracy and AUC.
- Confusion matrix visualization and highlighted comparison table to spot the best-performing models quickly.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```
Then open the Streamlit URL in your browser.

## How to use
1) Upload a CSV (or stick with the sample).  
2) Pick the target column and the class to treat as “positive” for ROC/PR.  
3) Choose the test split, random seed, and which models to compare.  
4) Click **Run comparison** to see per-model ROC/PR curves, confusion matrices, and a ranked summary table.

## Project structure
- `app.py` — Streamlit UI and experiment flow.
- `src/data_loader.py` — CSV loading helper (used for the sample data).
- `src/preprocess.py` — Train/test split logic.
- `src/models.py` — Model registry (pipelines with preprocessing).
- `src/plots.py` — Matplotlib plots for confusion, ROC, and PR curves.
- `data/customer_segments.csv` — Sample customer segmentation dataset.

## Extending
- Add models: register another sklearn pipeline in `src/models.py`.
- New metrics/plots: drop a function into `src/plots.py` and render it inside `app.py`.
- Different datasets: upload via the UI or point to a new default path in `load_dataframe` within `app.py`.
>>>>>>> 98e6176 (update readme for clear instruction on how to use)
