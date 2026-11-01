# Loan Eligibility Prediction(Assignment 2)
Term Project- DSTT- Part 1

Dataset Acquisition, Data Cleaning, Preprocessing, Feature Engineering and Exploratory Data Analysis (EDA)

Group 14 Team members: < Khadija Fatima >, < Memoona Ghufran >, < Ramisha Rahat>

1. Project Overview

Banks and lenders must decide quickly and consistently whether an applicant is eligible for a loan. This term project builds a machine-learning pipeline that predicts loan approval (Loan_Status: Y/N) from applicant details such as income, credit history, education, marital status and property area.

This submission covers the first half of the implementation: everything from dataset acquisition up to EDA and the preparation of model-ready data. Model training, optimization and evaluation follow in later assignments.

The notebook performs the following steps:

Dataset loading and description: shape, data types, summary statistics and unique values of categorical columns.
Data cleaning
Missing values: checked for nulls and "hidden" missing values (invalid zeros). Median/mode imputation code is included and runs safely even when no nulls exist.
Duplicates: duplicate rows are detected and removed.
Irrelevant columns: Customer_ID is dropped because it is only an identifier.
Outliers: detected with boxplots and the IQR rule, then treated by capping (clipping) at the IQR bounds, since extreme incomes and loan amounts are real values rather than errors.
Exploratory Data Analysis
Target distribution (class balance of Loan_Status)
Distributions and skewness of numerical features
Categorical features vs. loan status, including approval rate per category
Boxplots for outlier inspection
Correlation heatmaps
Feature engineering
Total_Income: applicant plus co-applicant income
EMI: loan amount divided by loan term
Balance_Income: income left after paying the EMI
Loan_to_Income: loan amount relative to total income
Has_Coapplicant: flag for the presence of a co-applicant
Total_Income_log: log transform to reduce skewness
Preprocessing and transformations
Binary encoding of Gender, Married, Education, Self_Employed and the target Loan_Status (Y=1, N=0)
One-hot encoding of Property_Area (drop_first=True)
Stratified 80/20 train/test split (random_state=42)
StandardScaler fitted on the training set only, then applied to the test set, to avoid data leakage
Feature selection: redundant columns (Applicant_Income, Coapplicant_Income, Total_Income) are dropped after engineering, and the final correlation heatmap and correlation with the target are reviewed.
Saved outputs: processed datasets are exported as CSV files for use in the next assignment.
2. Dataset Source
Name: Loan Eligibility Prediction
Source: Kaggle competition loan-eligibility-prediction (https://www.kaggle.com/competitions/loan-eligibility-prediction)
File used: Loan Eligibility Prediction.csv
Size: 614 rows and 13 columns
Column	Description
Customer_ID	Unique applicant identifier (dropped during cleaning)
Gender	Male / Female
Married	Marital status (Yes / No)
Dependents	Number of dependents
Education	Graduate / Not Graduate
Self_Employed	Self-employed status (Yes / No)
Applicant_Income	Applicant's income
Coapplicant_Income	Co-applicant's income
Loan_Amount	Requested loan amount (in thousands)
Loan_Amount_Term	Loan term in months
Credit_History	1 = meets credit guidelines, 0 = does not
Property_Area	Urban / Semiurban / Rural
Loan_Status	Target: loan approved (Y) or not (N)
3. Repository Contents
.
├── LoanEligibilityPrediction_Assignment2_group14.ipynb   # Main notebook
├── Loan Eligibility Prediction.csv                       # Raw dataset (download from Kaggle)
├── README.md                                             # This file
└── (generated when the notebook runs)
    ├── loan_processed.csv      # Full processed dataset (final features)
    ├── train_processed.csv     # Scaled training set (80%)
    └── test_processed.csv      # Scaled test set (20%)
4. Required Libraries
Python 3.9 or higher
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter (or JupyterLab)

warnings is part of the Python standard library and needs no installation.

5. Setup Instructions
Option A: Run locally
Clone or download this repository and open a terminal in the project folder.
(Optional) Create a virtual environment
bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
Install the dependencies
bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
Get the dataset. Download Loan Eligibility Prediction.csv from the Kaggle competition page above and place it in the same folder as the notebook.
Option B: Run on Google Colab
Upload LoanEligibilityPrediction_Assignment2_group14.ipynb to Google Colab.
Upload Loan Eligibility Prediction.csv through the Files panel (the notebook reads it from /content/).
All required libraries are already preinstalled in Colab.
6. Steps to Execute the Notebook
Start Jupyter:
bash
   jupyter notebook
Open LoanEligibilityPrediction_Assignment2_group14.ipynb.
Check the dataset path. The first data-loading cell uses the Colab path:
python
   df = pd.read_csv("/content/Loan Eligibility Prediction.csv")

When running locally, change it to:

python
   df = pd.read_csv("Loan Eligibility Prediction.csv")
Run all cells from top to bottom (Kernel → Restart & Run All). The cells depend on each other and must run in order.
After the last cell finishes, the processed files loan_processed.csv, train_processed.csv and test_processed.csv appear in the working directory.
7. Output Summary
Cleaned dataset with no missing values or duplicate rows, and no identifier column
Outlier-treated and encoded features, including six engineered features
Stratified train/test split with scaled numerical features
EDA visualizations: target distribution, histograms, count plots, boxplots, correlation heatmaps
8. Next Steps (Upcoming Assignments)
Assignment 3: model training, hyperparameter optimization, evaluation and the full pipeline, using train_processed.csv and test_processed.csv
Assignment 4: final report and presentation
