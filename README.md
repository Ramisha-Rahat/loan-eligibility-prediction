<div align="center">
#Loan Eligibility Prediction
 
### Term Project · DSTT · Part 1 (Assignment 2)
**Data Acquisition → Cleaning → Preprocessing → Feature Engineering → EDA**
 
**Group 14:** Khadija Fatima · Memoona Ghufran · Ramisha Rahat
 
</div>

## Project Overview
 
Banks and lenders must decide quickly and consistently whether an applicant is eligible for a loan. This term project builds a machine-learning pipeline that predicts **loan approval (`Loan_Status`: Y/N)** from applicant details such as income, credit history, education, marital status and property area.
 
 
## Dataset
 
| | |
|---|---|
| **Name** | Loan Eligibility Prediction |
| **Source** | [Kaggle competition: loan-eligibility-prediction](https://www.kaggle.com/competitions/loan-eligibility-prediction) |
| **File used** | `Loan Eligibility Prediction.csv` |
| **Size** | 614 rows × 13 columns |
| **Target** | `Loan_Status` (Y = approved, N = rejected) |
 
<details>
<summary><b> Click to view the column descriptions</b></summary>
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
 
##  What the Notebook Does
 
| Step | Stage | Details |

| 1️⃣ | **Loading & Description** | Shape, data types, summary statistics, unique values of categorical columns |
| 2️⃣ | **Missing Values** | Checks for nulls and hidden missing values (invalid zeros); median/mode imputation code included and safe to run even with no nulls |
| 3️⃣ | **Duplicates & Irrelevant Columns** | Removes duplicate rows; drops `Customer_ID` since it is only an identifier |
| 4️⃣ | **Exploratory Data Analysis** | Target distribution, numerical distributions and skewness, categorical features vs. loan status, approval rate per category, boxplots, correlation heatmaps |
| 5️⃣ | **Outlier Treatment** | Detected with boxplots and the IQR rule, then capped at the IQR bounds (extreme incomes and loan amounts are real values, not errors) |
| 6️⃣ | **Feature Engineering** | Six new features (see below) |
| 7️⃣ | **Encoding** | Binary encoding + one-hot encoding |
| 8️⃣ | **Feature Selection** | Redundant columns dropped; final correlation heatmap reviewed |
| 9️⃣ | **Split & Scaling** | Stratified 80/20 split, `StandardScaler` fitted on the training set only (no data leakage) |
| 🔟 | **Saving Outputs** | Processed datasets exported as CSV for the next assignment |
 
###  Engineered Features
 
| Feature | Meaning |
|---|---|
|  `Total_Income` | Applicant income + co-applicant income |
|  `EMI` | Loan amount ÷ loan term |
|  `Balance_Income` | Income left after paying the EMI |
|  `Loan_to_Income` | Loan amount relative to total income |
|  `Has_Coapplicant` | Flag (1/0) for the presence of a co-applicant |
|  `Total_Income_log` | Log transform to reduce skewness |

###  Encoding Used
 
| Columns | Method |
|---|---|
| `Gender`, `Married`, `Education`, `Self_Employed` | Binary encoding |
| `Loan_Status` (target) | Y = 1, N = 0 |
| `Property_Area` | One-hot encoding (`drop_first=True`) |
 

 
## Key Findings
 
-  **Clean data:** 614 applicants, no missing values and no duplicate rows.
-  **Moderate class imbalance:** about **69%** of loans are approved and **31%** rejected, so a stratified split was used.
-  **Credit history is the strongest predictor:** approval is about **79%** with good credit history vs. only about **11%** without it (correlation with target: **0.52**).
-  **Property area matters:** Semiurban applicants have the highest approval rate (about **77%**), Urban the lowest (about **66%**).
-  **Small positive effects:** married applicants, graduates and applicants with a co-applicant show slightly higher approval rates.
-  **Little effect:** `Gender` and `Self_Employed`.
-  **Skewed incomes and loan amounts** contained outliers, which were capped with the IQR method; a log transform reduced the skewness.

 
##  Repository Structure
 
```text
 loan-eligibility-prediction
 ┣  LoanEligibilityPrediction_Assignment2_group14.ipynb   ← Main notebook
 ┣  Loan Eligibility Prediction.csv                       ← Raw dataset (from Kaggle)
 ┣  README.md                                             ← This file
 ┗  Generated when the notebook runs
    ┣  loan_processed.csv     ← Full processed dataset (final features)
    ┣  train_processed.csv    ← Scaled training set (80%)
    ┗  test_processed.csv     ← Scaled test set (20%)


##  Required Libraries
 
| Library | Purpose |
|---|---|
| ![numpy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white) | Numerical operations |
| ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white) | Data loading and manipulation |
| ![matplotlib](https://img.shields.io/badge/matplotlib-11557C) | Plotting |
| ![seaborn](https://img.shields.io/badge/seaborn-4C8CBF) | Statistical visualizations |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Train/test split and scaling |
| ![jupyter](https://img.shields.io/badge/jupyter-F37626?logo=jupyter&logoColor=white) | Running the notebook |
 
**Python 3.9 or higher** is required. `warnings` is part of the standard library and needs no installation.

 
##  Setup and Execution
 
###  Option A: Run Locally
 
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
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```
 
**4. Get the dataset**
Download `Loan Eligibility Prediction.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/loan-eligibility-prediction) and place it in the same folder as the notebook.
 
**5. Start Jupyter and open the notebook**
 
```bash
jupyter notebook
```
 
**6. Check the dataset path** in the first data-loading cell. The notebook uses the Colab path by default:
 
```python
df = pd.read_csv("/content/Loan Eligibility Prediction.csv")
```
 
When running locally, change it to:
 
```python
df = pd.read_csv("Loan Eligibility Prediction.csv")

```
**7. Run all cells** from top to bottom: **Kernel → Restart & Run All**.
 
 The cells depend on each other and must run in order.
 
###  Option B: Run on Google Colab
## We use this method to uplaoding dataset in colab and run each cell.
 
1. Upload `LoanEligibilityPrediction_Assignment2_group14.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload `Loan Eligibility Prediction.csv` through the **Files** panel (the notebook reads it from `/content/`).
3. Click **Runtime → Run all**. All required libraries are already preinstalled in Colab.
---
 
## Outputs
 
After the last cell finishes, these files appear in the working directory:
 
| File | Contents |
|---|---|
| `loan_processed.csv` | Full processed dataset with final features |
| `train_processed.csv` | Scaled training set (80%) |
| `test_processed.csv` | Scaled test set (20%) |
 
**Summary of what the notebook produces:**
 
- Cleaned dataset with no missing values or duplicate rows, and no identifier column
- Outlier-treated and encoded features, including six engineered features
- Stratified train/test split with scaled numerical features
- EDA visualizations: target distribution, histograms, count plots, boxplots, correlation heatmaps
---
 
##  Roadmap
 
| Assignment | Focus | Status |
|:---:|---|:---:|
| **A2** | Dataset, cleaning, preprocessing, feature engineering, EDA | ✅ Done |
| **A3** | Model training, hyperparameter optimization, evaluation, full pipeline | ⏳ Upcoming |
| **A4** | Final report and presentation | ⏳ Upcoming |
 
---
 
<div align="center">
**Group 14 · DSTT Term Project**
Khadija Fatima · Memoona Ghufran · Ramisha Rahat
 
 
</div>
