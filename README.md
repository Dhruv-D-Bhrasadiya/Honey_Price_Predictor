# Honey Price Predictor

A small project to train a model that predicts honey prices from honey characteristics, and a Streamlit app to make predictions interactively.

## Repository structure

- `app.py` - Streamlit application that loads pre-trained artifacts (`best_model.pkl`, `model_columns.pkl`, `pollen_types.pkl`) and provides a UI to enter honey characteristics and get a price prediction.
- `main.py` - Script that loads `honey_purity_dataset.csv`, trains several regressors, selects the best model, and saves the artifacts required by the app:
  - `best_model.pkl` - trained model
  - `model_columns.pkl` - column order used by the model
  - `pollen_types.pkl` - list of available pollen/honey types for the app dropdown
- `honey_purity_dataset.csv` - dataset used for training and analysis (should be present in the repository root).
- `Regression_price_of_honey_prediction.ipynb` - notebook version of the analysis and modelling.

## Purpose

The project trains regression models (Ridge, RandomForest, GradientBoosting) on honey chemistry and pollen analysis data to predict honey price. The Streamlit app (`app.py`) provides an easy-to-use UI for non-technical users to get price predictions from model artifacts produced by `main.py`.

## Requirements

This project was developed on Windows. The following are recommended:

- Python 3.8+
- Packages: pandas, numpy, scikit-learn, seaborn, matplotlib, streamlit, joblib

Install dependencies with pip:

```powershell
pip install pandas numpy scikit-learn seaborn matplotlib streamlit joblib
```

(If you prefer, create a virtual environment first.)

## Quick start

1. Ensure the dataset is in the project root:
   - `honey_purity_dataset.csv`

2. Train and produce artifacts (if you don't already have `best_model.pkl`, `model_columns.pkl`, and `pollen_types.pkl`):

```powershell
python main.py
```

This will train models, print evaluation metrics, and save the artifacts used by the Streamlit app.

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

Streamlit will open a local URL (usually `http://localhost:8501`) where you can interact with the app.

## How the app works

- `app.py` loads the saved model and other artifacts using `joblib`.
- The sidebar gives controls for the numerical features and a dropdown for `Pollen_analysis` (types loaded from `pollen_types.pkl`).
- When you click `Predict Price`, inputs are one-hot encoded and aligned with `model_columns.pkl` before being passed to the model for prediction.

## Troubleshooting

- If the Streamlit app raises errors about missing files, make sure the following files exist in the project root:
  - `best_model.pkl`
  - `model_columns.pkl`
  - `pollen_types.pkl`
  If they are missing, run `python main.py` to create them.

- If `honey_purity_dataset.csv` is missing, `main.py` will fail to run. Place the CSV in the same directory as `main.py` and `app.py`.

- If plots do not display when running `main.py` from a headless terminal, consider running the analysis in the Jupyter notebook `Regression_price_of_honey_prediction.ipynb` instead.

## Next steps / Enhancements

- Add a `requirements.txt` or `environment.yml` for reproducible installs.
- Add unit tests for the preprocessing functions and small integration tests for `main.py` and `app.py`.
- Provide a simple Dockerfile to containerize the Streamlit app for deployment.

## License

No license specified. Add a LICENSE file if you plan to publish or share this project.

---

If you want, I can also add a `requirements.txt` and a short `Makefile` or `tasks.json` to make running the training and app commands easier on Windows.