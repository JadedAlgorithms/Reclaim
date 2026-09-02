import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def revenue_metrics(X_test, y_test, y_pred, label="Model"):
    amounts       = X_test["amount"].values
    actual        = y_test.values
    total         = amounts.sum()
    recovered     = amounts[y_pred == 1].sum()
    baseline_mask = X_test["action"].values == 1
    base_cov      = baseline_mask.mean()
    baseline      = (amounts[baseline_mask & (actual == 1)].sum() / base_cov
                     if base_cov > 0 else 0)
    print(f"\n=== Revenue Recovery — {label} ===")
    print(f"  Total at risk:            ₹{total:>10,.0f}")
    print(f"  Reclaim recovered:        ₹{recovered:>10,.0f}  ({recovered/total*100:.1f}%)")
    print(f"  Baseline (retry later):   ₹{baseline:>10,.0f}  ({baseline/total*100:.1f}%)")
    print(f"  Uplift over baseline:     {(recovered-baseline)/baseline*100:+.1f}%")
    return recovered


def train():
    df = pd.read_csv("data/ml_dataset.csv")
    X  = df.drop(columns=["recovery_success"])
    y  = df["recovery_success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Model 1: Logistic Regression ─────────────────────────────────────────
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    print("=" * 55)
    print("  MODEL 1 — Logistic Regression")
    print("=" * 55)
    print(classification_report(y_test, lr_pred, target_names=["failure", "success"]))
    lr_rev = revenue_metrics(X_test, y_test, lr_pred, "Logistic Regression")

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("\n" + "=" * 55)
    print("  MODEL 2 — Random Forest")
    print("=" * 55)
    print(classification_report(y_test, rf_pred, target_names=["failure", "success"]))
    rf_rev = revenue_metrics(X_test, y_test, rf_pred, "Random Forest")

    # ── Feature importance (Random Forest) ────────────────────────────────────
    print("\n=== Feature Importance (Random Forest — Gini) ===")
    importances = rf.feature_importances_
    for name, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 100)
        print(f"  {name:35s} {imp:.3f}  {bar}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\n=== 5-Fold Cross-Validation (F1 macro) ===")
    lr_cv = cross_val_score(lr, X, y, cv=5, scoring="f1_macro").mean()
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring="f1_macro").mean()
    print(f"  Logistic Regression:  {lr_cv:.3f}")
    print(f"  Random Forest:        {rf_cv:.3f}")

    # ── Pick winner and save ───────────────────────────────────────────────────
    winner     = rf if rf_rev >= lr_rev else lr
    winner_name = "Random Forest" if rf_rev >= lr_rev else "Logistic Regression"
    print(f"\n✅ Winner: {winner_name}  (₹{max(rf_rev, lr_rev):,.0f} recovered)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(winner, "models/reclaim_model.pkl")
    print("   Saved to models/reclaim_model.pkl")

    return winner


if __name__ == "__main__":
    train()
