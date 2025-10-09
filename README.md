# Loan_Prediction

📊 Loan Default Prediction Project
This project demonstrates a complete machine learning pipeline for predicting loan defaults using synthetic financial data. It includes data generation, preprocessing, model training, evaluation, and visualization to analyze borrower risk factors.

🔧 Key Components
Dataset: A realistic synthetic dataset of 2,000 borrowers with features such as:

Age, Income, Loan Amount, Loan Term

Credit Score, Employment Duration

Prior Default History, Marital Status, Loan Purpose

Target Variable: default — binary indicator (0 or 1) of whether a borrower will default.

Modeling: Two classification models trained and evaluated:

Logistic Regression

Random Forest Classifier

🛠️ Methodology
Data Preprocessing:

Categorical variables (marital_status, purpose) encoded using one-hot encoding.

Numerical features standardized using StandardScaler.

Train-test split (70%-30%) with stratification on the target.

Model Training:

Both models trained on the same training set.

Hyperparameters kept simple for clarity and reproducibility.

Evaluation Metrics:

ROC-AUC Score

Classification Report (Precision, Recall, F1-Score)

Confusion Matrix

ROC Curve

Visualizations:

ROC Curve: Compares model performance in terms of true positive vs. false positive rates.

Confusion Matrix: Shows prediction accuracy and error types.

Credit Score Distribution: Histogram comparing credit scores of defaulters vs. non-defaulters.

Loan Default Counts: Bar chart showing class balance.

Boxplot (by Credit Score Bins): Displays how loan amounts vary across credit score ranges.

📈 Key Insight from Boxplot
The final boxplot groups borrowers into credit score bins (e.g., 300–350, 350–400, ..., 850) to avoid overcrowding from treating credit score as continuous. This reveals:

Borrowers with higher credit scores tend to take larger loans, but with less variance.

Lower credit score groups show wider variation in loan amounts, indicating higher risk dispersion.

✅ Fix Applied: Raw credits_score was binned using pd.cut() to make the boxplot interpretable and visually meaningful.

📂 How to Use
bash
python loan_default_prediction.py
Ensure dependencies are installed:

bash
pip install numpy pandas scikit-learn matplotlib
🎯 Purpose
This project is ideal for:

Learning end-to-end ML workflows

Credit risk modeling

Practicing classification and visualization techniques

Demonstrating model evaluation best practices
