# Breast Cancer Classification

A machine learning project that compares classification models on the Wisconsin Breast Cancer dataset and provides an interactive input section for making predictions.

> **Important:** This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, screen, or guide treatment for any person.

## Project Overview

The goal is to train and evaluate multiple machine learning classifiers that distinguish between **benign** and **malignant** tumor samples. The notebook includes model evaluation metrics and visualizations to make model performance easier to compare.

### Target labels

The notebook's target mapping is:

- `0` — Benign
- `1` — Malignant

Confirm that this mapping matches the target transformation in your notebook before interpreting results.

## Features

- Breast cancer dataset loading and preparation
- Stratified train-test split
- Comparison of multiple classification algorithms
- Five-fold stratified cross-validation
- Evaluation using accuracy, error rate, precision, recall, F1-score, and ROC-AUC
- Visual model comparisons, including:
  - Accuracy comparison
  - Error-rate comparison
  - Precision, recall, and F1-score comparison
  - ROC curves
  - Confusion matrices
  - Precision-recall curves
  - Cross-validation score comparison
- Interactive feature input form using `ipywidgets`
- Prediction output with estimated malignant-class probability
- Optional extensions: hyperparameter tuning, feature importance, model persistence, and a Streamlit interface

## Models

The notebook can compare the following models:

| Model | Notes |
|---|---|
| Logistic Regression | Uses feature scaling |
| Random Forest | Tree-based ensemble |
| Extra Trees | Randomized tree ensemble |
| Support Vector Machine (RBF) | Uses feature scaling |
| K-Nearest Neighbors | Uses feature scaling |
| HistGradientBoosting | Gradient-boosted trees |

Actual performance depends on the data split, preprocessing, parameters, and evaluation procedure. The notebook should report measured results rather than assume one model is best.

## Dataset

The example workflow uses scikit-learn's built-in Wisconsin Breast Cancer dataset:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
X = data.data
```

The dataset contains numerical measurements computed from digitized images of breast mass samples. The notebook should use a consistent target mapping throughout training, evaluation, plotting, and interactive prediction.

## Requirements

- Python 3.9 or newer recommended
- Jupyter Notebook or Google Colab
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- ipywidgets (for the interactive input form)

Install dependencies locally:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets
```

## Run in Google Colab

1. Open the notebook in Google Colab.
2. Run the dataset loading and preprocessing cells.
3. Run the model comparison and cross-validation cells.
4. Run the evaluation and graph cells.
5. Run the interactive input cell.
6. Choose a model, enter feature measurements, and click **Predict Tumor Class**.

For reproducible results, use a fixed random seed and the same stratified split when comparing models. Fit preprocessing steps only on training data; using scikit-learn pipelines helps prevent data leakage.

## Evaluation Metrics

- **Accuracy:** Fraction of all predictions that are correct.
- **Error rate:** `1 - accuracy`.
- **Precision:** Of samples predicted malignant, the fraction that are actually malignant.
- **Recall (sensitivity):** Of actual malignant samples, the fraction detected as malignant.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to rank malignant samples above benign samples across classification thresholds.
- **Confusion matrix:** Counts of true negatives, false positives, false negatives, and true positives.

In this notebook, malignant is the positive class (`1`). False negatives—malignant samples predicted as benign—deserve particular attention. A low error rate alone does not establish clinical suitability.

## Recommended Project Structure

```text
Breast-Cancer-Classification/
├── Breast Cancer Classification.ipynb
├── README.md
├── requirements.txt
└── models/                 # Optional: saved model artifacts
```

This is a suggested structure; adjust the filenames to match your repository.

## Future Improvements

- Tune model hyperparameters using cross-validation
- Add feature-importance and explainability analysis
- Compare calibration and decision thresholds
- Save and reload the fitted preprocessing-and-model pipeline
- Create a Streamlit dashboard
- Add automated tests and document reproducible results

## Limitations

- Dataset performance does not guarantee performance on new populations or clinical settings.
- Predictions and probabilities can be wrong or poorly calibrated.
- The interactive form is intended for experimentation with dataset features, not for entering arbitrary patient data.
- Any real clinical use would require appropriate clinical validation, governance, and expert oversight.

## Author

**Varun Patel**

- GitHub: [varunpatelspace](https://github.com/varunpatelspace)

## License

Add a license file (for example, MIT) if you want others to reuse this project. Do not claim a license until you have chosen and added one.
