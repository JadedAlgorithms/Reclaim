import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train():
    df = pd.read_csv("data/ml_dataset.csv")

    X = df.drop(columns=["recovery_success"])
    y = df["recovery_success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- model ---
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- classification report ---
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["failure", "success"]))

    # --- revenue metrics ---
    amounts = X_test["amount"].values
    actual_success = y_test.values

    total_at_risk       = amounts.sum()
    actual_recovered    = amounts[actual_success == 1].sum()

    # Reclaim: predicted successes
    reclaim_recovered   = amounts[y_pred == 1].sum()

    # Baseline: retry_after_delay = action 1 — simulate using actual outcome for those rows
    baseline_mask       = X_test["action"].values == 1
    baseline_recovered  = amounts[baseline_mask & (actual_success == 1)].sum()
    # scale baseline to all transactions (it would try all of them, not just action==1 rows)
    baseline_coverage   = baseline_mask.mean()
    baseline_recovered_scaled = baseline_recovered / baseline_coverage if baseline_coverage > 0 else 0

    print("=== Revenue Recovery ===")
    print(f"Total revenue at risk:        ₹{total_at_risk:>10,.0f}")
    print(f"Actual recoverable:           ₹{actual_recovered:>10,.0f}  ({actual_recovered/total_at_risk*100:.1f}%)")
    print(f"Reclaim recovered:            ₹{reclaim_recovered:>10,.0f}  ({reclaim_recovered/total_at_risk*100:.1f}%)")
    print(f"Baseline (retry after delay): ₹{baseline_recovered_scaled:>10,.0f}  ({baseline_recovered_scaled/total_at_risk*100:.1f}%)")

    # --- feature importance ---
    coefs = model.named_steps["clf"].coef_[0]
    print("\n=== Feature Importance (logistic regression coefficients) ===")
    for name, coef in sorted(zip(X.columns, coefs), key=lambda x: abs(x[1]), reverse=True):
        bar = "+" * int(abs(coef) * 10) if coef > 0 else "-" * int(abs(coef) * 10)
        print(f"  {name:35s} {coef:+.3f}  {bar}")

    # --- save model ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/reclaim_model.pkl")
    print("\nModel saved to models/reclaim_model.pkl")

    return model


if __name__ == "__main__":
    train()
