import pandas as pd

FEATURE_COLS = [
    "amount", "payment_method", "failure_reason", "risk_tier",
    "attempt_number", "hours_since_failure", "customer_prior_success_rate",
    "action", "recovery_success",
]

ENCODINGS = {
    "payment_method": {"upi": 0, "card": 1, "netbanking": 2, "wallet": 3},
    "failure_reason": {"insufficient_funds": 0, "upi_timeout": 1},
    "risk_tier":      {"low": 0, "medium": 1, "high": 2},
    "action":         {"retry_immediately": 0, "retry_after_delay": 1,
                       "use_another_payment_method": 2, "send_payment_reminder": 3},
}


def build_ml_dataset(customers_path, transactions_path, attempts_path):
    # --- load ---
    customers    = pd.read_csv(customers_path)
    transactions = pd.read_csv(transactions_path)
    attempts     = pd.read_csv(attempts_path)

    # --- join: attempts → transactions → customers ---
    df = attempts.merge(transactions, on="transaction_id", suffixes=("_attempt", "_txn"))
    df = df.merge(customers, on="customer_id")

    # --- time-based features ---
    df = df.sort_values("timestamp_attempt")
    df["attempt_number"] = df.groupby("transaction_id").cumcount() + 1
    df["hours_since_failure"] = (
        (pd.to_datetime(df["timestamp_attempt"]) - pd.to_datetime(df["timestamp_txn"]))
        .dt.total_seconds() / 3600.0
    )

    # --- target ---
    df["recovery_success"] = (df["outcome"] == "success").astype(int)

    # --- customer history (no leakage: shift(1) excludes current row) ---
    df = df.sort_values(["customer_id", "timestamp_attempt"])
    df["customer_prior_successes"] = (
        df.groupby("customer_id")["recovery_success"]
        .transform(lambda x: x.shift(1).expanding().sum())
    )
    df["customer_prior_attempts"] = (
        df.groupby("customer_id")["recovery_success"]
        .transform(lambda x: x.shift(1).expanding().count())
    )
    df["customer_prior_success_rate"] = (
        df["customer_prior_successes"] / df["customer_prior_attempts"]
    ).fillna(0.5)

    # --- encode categoricals ---
    for col, mapping in ENCODINGS.items():
        df[col] = df[col].map(mapping)

    return df[FEATURE_COLS]


def main():
    df = build_ml_dataset(
        "data/customers.csv",
        "data/transactions.csv",
        "data/attempts.csv",
    )
    df.to_csv("data/ml_dataset.csv", index=False)
    print(f"ML dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())


if __name__ == "__main__":
    main()
