import random

def simulate_outcome(action,failure_reason, risk_tier, attempt_number, hours_since_failure):
    if action == "retry_immediately":
        prob =   0.25
    elif action == "retry_after_delay":
        prob =   0.45
    elif action == "use_another_payment_method":
        prob =  0.60
    elif action == "send_payment_reminder":
        prob =   0.35
    # modifier 1 — failure reason
    if failure_reason == "upi_timeout" and action == "use_another_payment_method":
        prob += 0.20
    risk_multiplier = {"low": 1.1, "medium": 1.0, "high": 0.7}
    prob *= risk_multiplier[risk_tier]
    prob *= 0.85 ** (attempt_number - 1)
    prob = max(0.05, min(0.95, prob))
    return "success" if random.random() < prob else "failure"


    
    