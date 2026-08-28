import random 

def simulate_recovery(action):
    probabilities = {
        "retry_immediately": 0.4,
        "retry_after_delay": 0.7,
        "use_another_payment_method": 0.8,
        "send_payment_reminder": 0.5,
    }
    success_probability = probabilities[action]
    return random.random() < success_probability