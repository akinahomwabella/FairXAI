# A Framework for Bias Detection and Mitigation in Machine Learning

This repository implements a practical pipeline for detecting and mitigating bias in machine learning models. The framework combines:

- **Fairness Evaluation:** Using fairness metrics (e.g., TPR, FNR) computed via repeated splits.
- **Bias Mitigation:** Reweighting techniques to balance training data.
- **Explainability:** Generation of SHAP and LIME explanations with natural language translation via Phi-2.

## Project Structure

- `bias_pipeline.py`: Contains code for data splitting, model training (XGBoost), bias evaluation, and statistical testing.
- `explainability.py`: Implements functions for generating natural language explanations from SHAP/LIME outputs.
- `shap_values_*.csv`: Input/Output files for SHAP value processing.
- `README.md`: This file.

## Key Dependencies

- Python 3.7+
- Pandas, NumPy
- scikit-learn
- XGBoost
- SHAP, LIME
- Transformers (for Phi-2)
- Matplotlib, Seaborn

## Usage

1. **Data Preprocessing & Model Training:**

   The script `bias_pipeline.py` performs:
   - Stratified data splitting (using `StratifiedShuffleSplit`)
   - Training a baseline and a reweighted XGBoost model
   - Calculation of fairness metrics (e.g., True Positive Rates) across sensitive subgroups (e.g., gender, race)
   - Paired t-tests to assess statistical significance of bias mitigation.

   To run:
   ```bash
   python bias_pipeline.py
