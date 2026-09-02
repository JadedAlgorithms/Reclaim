"""
Reclaim Agent
-------------
Takes a failed transaction + customer context and recommends
the best recovery action using the trained model.
"""

import joblib
import pandas as pd
from src.features import ENCODINGS

ACTIONS = [
    "retry_immediately",
    "retry_after_delay",
    "use_another_payment_method",
    "send_payment_reminder",
]

ACTION_LABELS = {
    "retry_immediately":          "Retry immediately",
    "retry_after_delay":          "Retry after a delay",
    "use_another_payment_method": "Switch payment method",
    "send_payment_reminder":      "Send payment reminder",
}

MODEL_PATH = "models/reclaim_model.pkl"


class ReclaimAgent:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)

    def recommend(self, amount, payment_method, failure_reason, risk_tier,
                  attempt_number=1, hours_since_failure=1.0,
                  customer_prior_success_rate=0.5):
        """
        Score all 4 recovery actions and return a ranked recommendation.

        Parameters
        ----------
        amount                    : float  — transaction amount in ₹
        payment_method            : str    — "upi" | "card" | "netbanking" | "wallet"
        failure_reason            : str    — "insufficient_funds" | "upi_timeout"
        risk_tier                 : str    — "low" | "medium" | "high"
        attempt_number            : int    — which attempt this is (default 1)
        hours_since_failure       : float  — hours elapsed since the transaction failed
        customer_prior_success_rate : float — fraction of past attempts that succeeded (0–1)

        Returns
        -------
        list of dicts, sorted by confidence descending
        """
        # encode categoricals
        pm   = ENCODINGS["payment_method"][payment_method]
        fr   = ENCODINGS["failure_reason"][failure_reason]
        rt   = ENCODINGS["risk_tier"][risk_tier]

        results = []
        for action in ACTIONS:
            act = ENCODINGS["action"][action]
            row = pd.DataFrame([{
                "amount":                    amount,
                "payment_method":            pm,
                "failure_reason":            fr,
                "risk_tier":                 rt,
                "attempt_number":            attempt_number,
                "hours_since_failure":       hours_since_failure,
                "customer_prior_success_rate": customer_prior_success_rate,
                "action":                    act,
            }])
            prob = self.model.predict_proba(row)[0][1]  # P(success)
            results.append({
                "action":     action,
                "label":      ACTION_LABELS[action],
                "confidence": round(prob, 3),
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def print_recommendation(self, results, amount):
        best = results[0]
        print(f"\n{'='*50}")
        print(f"  RECLAIM — Recovery Recommendation")
        print(f"{'='*50}")
        print(f"  Transaction at risk: ₹{amount:,.0f}\n")
        print(f"  ✅ Best action: {best['label']}")
        print(f"     Confidence: {best['confidence']*100:.1f}%")
        print(f"\n  All options (ranked):")
        for i, r in enumerate(results):
            bar = "█" * int(r["confidence"] * 20)
            print(f"  {i+1}. {r['label']:30s} {r['confidence']*100:5.1f}%  {bar}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    # --- demo: a ₹47,500 UPI transaction that timed out ---
    agent = ReclaimAgent()
    results = agent.recommend(
        amount=47500,
        payment_method="upi",
        failure_reason="upi_timeout",
        risk_tier="medium",
        attempt_number=1,
        hours_since_failure=2.0,
        customer_prior_success_rate=0.5,
    )
    agent.print_recommendation(results, amount=47500)
