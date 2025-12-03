from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Global variables
model = None
scaler = None
train_columns = None
df_global = None
x_test_global = None
y_test_global = None


# ===================== CREATE DATASET =====================

def create_dataset():
    np.random.seed(42)
    n = 2000

    df = pd.DataFrame({
        "age": np.random.randint(21, 70, size=n),
        "income": np.random.normal(60000, 25000, size=n).clip(8000, 300000),
        "loan_amount": np.random.normal(15000, 10000, size=n).clip(1000, 100000),
        "loan_term_month": np.random.choice([12, 24, 36, 48, 60], size=n, p=[0.05, 0.15, 0.4, 0.25, 0.15]),
        "credits_score": np.random.normal(650, 70, size=n).clip(300, 850),
        "employment_year": np.random.exponential(scale=3, size=n).astype(int),
        "prior_default": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "marital_status": np.random.choice(['single','married','divorced','widowed'], size=n),
        "purpose": np.random.choice(['debt_consolidation','home_improvement','education','car','other'], size=n)
    })

    # Generate target
    loan_ratio = df["loan_amount"] / (df["income"] + 1)

    score = (
        -3.0 * (df["credits_score"] - 650) / 100.0
        + 6.0 * loan_ratio
        - 0.2 * df["employment_year"]
        + 2.0 * df["prior_default"]
        + (df["purpose"] == "debt_consolidation") * 0.5
        + (df["marital_status"] == "single") * 0.1
    )

    prob = 1 / (1 + np.exp(-score))
    df["default"] = (np.random.rand(len(df)) < prob).astype(int)

    return df


# ===================== TRAIN MODEL =====================

def train_model(df):
    global model, scaler, train_columns, x_test_global, y_test_global

    x = df.drop("default", axis=1)
    y = df["default"]

    x = pd.get_dummies(x, drop_first=True)
    train_columns = x.columns

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    num_cols = ["age","income","loan_amount","loan_term_month","credits_score","employment_year","prior_default"]

    x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
    x_test[num_cols] = scaler.transform(x_test[num_cols])

    x_test_global = x_test
    y_test_global = y_test

    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model.fit(x_train, y_train)

    print(f"Model trained successfully!")
    print(f"Training Accuracy: {model.score(x_train, y_train):.3f}")
    print(f"Test Accuracy: {model.score(x_test, y_test):.3f}")


# ===================== PREPARE USER INPUT =====================

def prepare_user_input(form):
    data = {
        "age": float(form["age"]),
        "income": float(form["income"]),
        "loan_amount": float(form["loan_amount"]),
        "loan_term_month": int(form["loan_term_month"]),
        "credits_score": float(form["credits_score"]),
        "employment_year": int(form["employment_year"]),
        "prior_default": int(form["prior_default"]),
        "marital_status": form["marital_status"],
        "purpose": form["purpose"]
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=train_columns, fill_value=0)

    num_cols = ["age","income","loan_amount","loan_term_month","credits_score","employment_year","prior_default"]
    df[num_cols] = scaler.transform(df[num_cols])

    return df


# ===================== ROUTES =====================

@app.route("/")
def dashboard():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        user_input = prepare_user_input(request.form)
        prob = model.predict_proba(user_input)[0][1]
        prediction = "High Risk - Likely to Default" if prob > 0.5 else "Low Risk - Likely Approved"

        return render_template(
            "predict.html",
            result=True,
            prediction=prediction,
            probability=round(prob * 100, 2),
        )

    return render_template("predict.html", result=False)


# ===================== MAIN =====================

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Loan Default Prediction Application")
    print("=" * 60)

    df_global = create_dataset()
    print(f"Dataset created with {len(df_global)} records")
    print(f"Default rate: {df_global['default'].mean():.2%}")

    train_model(df_global)

    print("\n" + "=" * 60)
    print("Application ready! Starting Flask server...")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000)
