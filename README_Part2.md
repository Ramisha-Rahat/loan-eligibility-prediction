<div align="center">

# Loan Eligibility Prediction

### Term Project · DSTT · Part 2 (Assignment 3)
**Model Training → Hyperparameter Tuning → Optimization → Evaluation → Complete ML Pipeline**

**Group 14:** Khadija Fatima · Memoona Ghufran · Ramisha Rahat

</div>

## Project Overview

Banks and lenders must decide quickly and consistently whether an applicant is eligible for a loan. This term project builds a machine-learning pipeline that predicts **loan approval (`Loan_Status`: Y/N)** from applicant details such as income, credit history, education, marital status and property area.

**Assignment 2** covered data acquisition, cleaning, preprocessing, feature engineering and EDA. **Assignment 3 (this part)** trains and compares several models, tunes and optimizes them, evaluates them on an untouched test set and delivers one complete, reusable pipeline that takes raw applicant data and returns a loan decision.

## Dataset

| | |
|---|---|
| **Name** | Loan Eligibility Prediction |
| **Source** | [Kaggle competition: loan-eligibility-prediction](https://www.kaggle.com/competitions/loan-eligibility-prediction) |
| **File used** | `Loan Eligibility Prediction.csv` |
| **Size** | 614 rows × 13 columns (12 after dropping `Customer_ID`) |
| **Target** | `Loan_Status` (Y = approved, N = rejected), about 69% approved / 31% rejected |

<details>
<summary><b>Click to view the column descriptions</b></summary>
<br>

| Column | Description |
|---|---|
| `Customer_ID` | Unique applicant identifier *(dropped during cleaning)* |
| `Gender` | Male / Female |
| `Married` | Marital status (Yes / No) |
| `Dependents` | Number of dependents |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Self-employed status (Yes / No) |
| `Applicant_Income` | Applicant's income |
| `Coapplicant_Income` | Co-applicant's income |
| `Loan_Amount` | Requested loan amount (in thousands) |
| `Loan_Amount_Term` | Loan term in months |
| `Credit_History` | 1 = meets credit guidelines, 0 = does not |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | **Target:** loan approved (Y) or not (N) |

</details>

## What the Notebook Does

| Step | Stage | Details |
|:---:|---|---|
| 1️⃣ | **Setup & Loading** | Loads the CSV, drops duplicates and `Customer_ID` (Assignment 2 cleaning) |
| 2️⃣ | **Reusable Preprocessing** | Assignment 2 steps (IQR outlier capping, 6 engineered features, encoding, scaling) wrapped in a scikit-learn pipeline inside the notebook, so everything is learned from training data only (no leakage) |
| 3️⃣ | **Train / Validation / Test Split** | Stratified **60 / 20 / 20** split. The test set is the same 20% as in Assignment 2 |
| 4️⃣ | **Models & Metrics** | Explanation of every model and every metric used |
| 5️⃣ | **Baseline Models** | 8 models with default settings + a majority-class Dummy reference, compared on the validation set |
| 6️⃣ | **Class Imbalance** | `class_weight` None vs balanced, judged with repeated cross-validation |
| 7️⃣ | **Hyperparameter Tuning** | `GridSearchCV` and `RandomizedSearchCV` with 5-fold stratified CV (ROC-AUC) |
| 8️⃣ | **Optimization** | Default vs tuned, Voting and Stacking ensembles, feature selection, decision-threshold tuning |
| 9️⃣ | **Final Model Selection** | Chosen from cross-validation **before** the test set is opened |
| 🔟 | **Test Evaluation** | Metrics table, ROC and Precision-Recall curves, confusion matrices, classification report |
| 1️⃣1️⃣ | **Interpretation** | Permutation importance, learning curve, error analysis |
| 1️⃣2️⃣ | **Saving the Pipeline** | One `.joblib` file that predicts from raw applicant columns, with a demo on new applicants |
| 1️⃣3️⃣ | **Key Observations** | Written conclusions and limitations |

### Models Used

| Model | Role |
|---|---|
| Dummy (majority class) | Reference point every real model must beat |
| Logistic Regression | Simple, interpretable linear baseline |
| K-Nearest Neighbours | Distance-based, non-parametric |
| Support Vector Machine (RBF) | Non-linear margin-based classifier |
| Naive Bayes (Gaussian) | Fast probabilistic baseline |
| Decision Tree | Easy to explain if/else rules |
| Random Forest | Bagged ensemble of trees |
| Gradient Boosting | Boosted ensemble of trees |
| AdaBoost | Alternative boosting method |
| Voting & Stacking ensembles | Built from the tuned Logistic Regression, SVM, Random Forest and Gradient Boosting |

### Evaluation Metrics

Accuracy · Precision · Recall · F1-score · **Macro-F1** · **ROC-AUC** (main tuning metric) · Specificity (share of rejected loans correctly caught) · Confusion matrix · ROC and Precision-Recall curves.
Accuracy alone is not enough because about 69% of the loans are approved.

## Results

### Final model: Gradient Boosting (tuned, threshold 0.48)

Selected as the candidate with the best cross-validated ROC-AUC (0.766). Tuned parameters: `learning_rate=0.109`, `max_depth=2`, `n_estimators=103`, `min_samples_leaf=4`, `subsample=0.613`.

| Metric (untouched test set, 123 applicants) | Final model | Dummy baseline |
|---|:---:|:---:|
| Accuracy | **0.829** | 0.691 |
| Precision (approved) | 0.848 | 0.691 |
| Recall (approved) | 0.918 | 1.000 |
| F1 (approved) | 0.881 | 0.817 |
| Macro-F1 | **0.789** | 0.409 |
| ROC-AUC | **0.841** | 0.500 |

### All models on the test set

| Model | Accuracy | F1 | Macro-F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|
| Voting Ensemble | 0.829 | 0.884 | 0.780 | 0.846 |
| **Gradient Boosting (final, threshold 0.48)** | **0.829** | **0.881** | **0.789** | **0.841** |
| Gradient Boosting (threshold 0.50) | 0.821 | 0.875 | 0.780 | 0.841 |
| Stacking Ensemble | 0.813 | 0.872 | 0.764 | 0.836 |
| SVM (RBF) | 0.813 | 0.874 | 0.755 | 0.822 |
| Random Forest | 0.837 | 0.886 | 0.800 | 0.820 |
| Logistic Regression | 0.805 | 0.870 | 0.741 | 0.800 |
| Decision Tree | 0.821 | 0.878 | 0.772 | 0.786 |
| KNN | 0.715 | 0.828 | 0.507 | 0.735 |
| Dummy (majority) | 0.691 | 0.817 | 0.409 | 0.500 |

> **Note:** the test set is small (123 applicants, one applicant ≈ 0.8 percentage points), so the differences between the top models are within noise. The final model was chosen from cross-validation, before the test set was used, and was not changed afterwards.

## Key Findings

- **Credit history dominates:** Random Forest importance 0.40 (next feature 0.16) and the largest permutation drop in test ROC-AUC (0.19). Logistic Regression with this single feature already reaches about 0.80 accuracy.
- **Incomes and loan amount come next** (permutation drop about 0.10 to 0.11 each), then `Property_Area` (0.03). `Gender` and `Self_Employed` add nothing.
- **Tuning helped weak or high-variance models most:** Decision Tree ROC-AUC +0.08, KNN +0.06, Gradient Boosting +0.01. Logistic Regression and SVM were already near their best.
- **Ensembles did not beat the best single model** on cross-validated ROC-AUC, so the simpler Gradient Boosting was kept.
- **Class imbalance is a trade-off:** `class_weight="balanced"` raised specificity from about 0.40 to 0.52 but lowered accuracy by 1 to 6 points.
- **Threshold tuning gave only a small gain:** out-of-fold Macro-F1 0.745 → 0.751 when moving from 0.50 to 0.48.
- **Where it fails:** 21 of 123 test applicants were misclassified, mostly applicants whose outcome does not follow the credit-history rule. Accuracy is highest for Semiurban (0.90) and lowest for Rural (0.77) applicants.
- **Limitation:** with only 614 rows and no information on employment history or existing debts, large further gains are unlikely.

## Repository Structure

```text
loan-eligibility-prediction
 ┣ LoanEligibilityPrediction_Assignment2_group14.ipynb   ← Assignment 2 notebook
 ┣ LoanEligibilityPrediction_Assignment3_group14.ipynb   ← Assignment 3 main notebook
 ┣ Loan Eligibility Prediction.csv                       ← Raw dataset (from Kaggle)
 ┣ README.md                                             ← This file
 ┗ Generated when the Assignment 3 notebook runs
    ┣ loan_eligibility_pipeline.joblib   ← Final saved pipeline (raw data in, decision out)
    ┣ baseline_validation.csv            ← Baseline models on the validation set
    ┣ tuning_summary.csv                 ← Best parameters of each tuned model
    ┣ model_comparison_cv.csv            ← Cross-validation comparison of all candidates
    ┣ model_comparison_test.csv          ← Test-set comparison of all models
    ┣ test_predictions.csv               ← Final model predictions and probabilities
    ┗ figures/                           ← 12 saved plots (PNG)
```

## Required Libraries

| Library | Purpose |
|---|---|
| ![numpy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white) | Numerical operations |
| ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white) | Data loading and manipulation |
| ![scipy](https://img.shields.io/badge/scipy-8CAAE6?logo=scipy&logoColor=white) | Parameter distributions for randomized search |
| ![matplotlib](https://img.shields.io/badge/matplotlib-11557C) | Plotting |
| ![seaborn](https://img.shields.io/badge/seaborn-4C8CBF) | Statistical visualizations |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Models, tuning, metrics, pipeline |
| ![joblib](https://img.shields.io/badge/joblib-4B8BBE) | Saving and loading the final pipeline |
| ![jupyter](https://img.shields.io/badge/jupyter-F37626?logo=jupyter&logoColor=white) | Running the notebook |

**Python 3.9 or higher** is required. No extra libraries (such as XGBoost) are needed. `warnings` and `os` are part of the standard library.

## Setup and Execution

### Option A: Run Locally

**1. Clone or download the repository**

```bash
git clone https://github.com/<your-username>/loan-eligibility-prediction.git
cd loan-eligibility-prediction
```

**2. (Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install the dependencies**

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn joblib jupyter
```

**4. Get the dataset**
Download `Loan Eligibility Prediction.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/loan-eligibility-prediction) and place it in the same folder as the notebook.

**5. Start Jupyter and open the notebook**

```bash
jupyter notebook
```

**6. Run all cells** from top to bottom: **Kernel → Restart & Run All**.
The notebook finds the CSV in the current folder (or in `/content/` on Colab), so no path change is needed. The cells depend on each other and must run in order. The full run takes about 5 minutes because of the hyperparameter searches and the repeated cross-validation.

### Option B: Run on Google Colab

We use this method: upload the dataset to Colab and run each cell.

1. Upload `LoanEligibilityPrediction_Assignment3_group14.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload `Loan Eligibility Prediction.csv` through the **Files** panel (the notebook reads it from `/content/`).
3. Click **Runtime → Run all**. All required libraries are already preinstalled in Colab.

## Using the Saved Pipeline

The saved file contains the whole pipeline (feature engineering, preprocessing, model) and the chosen threshold, so it accepts **raw applicant columns**. It is saved together with its own code, so it loads in any new script or notebook **without the Assignment 3 notebook or any extra `.py` file**.

```python
import joblib, numpy as np, pandas as pd

bundle = joblib.load("loan_eligibility_pipeline.joblib")

applicant = pd.DataFrame([{
    "Gender": "Male", "Married": "Yes", "Dependents": 1, "Education": "Graduate",
    "Self_Employed": "No", "Applicant_Income": 5000, "Coapplicant_Income": 2000,
    "Loan_Amount": 120, "Loan_Amount_Term": 360, "Credit_History": 1,
    "Property_Area": "Semiurban"}])

proba = bundle["model"].predict_proba(applicant[bundle["raw_features"]])[:, 1]
decision = np.where(proba >= bundle["threshold"], "Y", "N")
print(proba.round(3), decision)   # approval probability + Y/N decision
```

Use a similar scikit-learn version to the one the file was saved with, otherwise loading may show a version warning.

## Outputs

After the last cell finishes, these files appear in the working directory:

| File | Contents |
|---|---|
| `loan_eligibility_pipeline.joblib` | Final complete pipeline + threshold |
| `baseline_validation.csv` | Validation scores of the default models |
| `tuning_summary.csv` | Search type, best CV ROC-AUC and best parameters per model |
| `model_comparison_cv.csv` | Cross-validated scores of all tuned models and ensembles |
| `model_comparison_test.csv` | Test-set scores of all models and the final model |
| `test_predictions.csv` | Test applicants with actual label, prediction and approval probability |
| `figures/*.png` | Baseline, class-weight, tuning, feature-selection, threshold, ROC/PR, confusion-matrix, importance and learning-curve plots |

**Summary of what the notebook produces:**

- A leak-free 60/20/20 split and a single reusable preprocessing pipeline, all inside one notebook
- Nine models compared, six of them tuned, plus Voting and Stacking ensembles
- Optimization by tuning, ensembling, feature selection and threshold tuning
- Full evaluation: metrics, ROC and PR curves, confusion matrices, importance, learning curve and error analysis
- A saved final pipeline that predicts directly from raw applicant data

## Roadmap

| Assignment | Focus | Status |
|:---:|---|:---:|
| **A2** | Dataset, cleaning, preprocessing, feature engineering, EDA | ✅ Done |
| **A3** | Model training, hyperparameter optimization, evaluation, full pipeline | ✅ Done |
| **A4** | Final report and presentation | ⏳ Upcoming |

---

<div align="center">

**Group 14 · DSTT Term Project**
Khadija Fatima · Memoona Ghufran · Ramisha Rahat

</div>
