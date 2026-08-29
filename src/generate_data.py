import random
import uuid
import pandas as pd
from datetime import datetime, timedelta  
from src.entities import Customer, Transaction, RecoveryAttempt
import dataclasses 
from simulator.recovery_simulator import simulate_outcome

def generate_customers(n):
    customers = []
    for i in range(n):
        customer_id = str(uuid.uuid4())
        preferred_payment_choice = random.choice(["card", "netbanking", "wallet","upi"])
        risk_tier = random.choice(["low","medium","high"])
        customer = Customer(customer_id,preferred_payment_choice,risk_tier)
        customers.append(customer)
    return customers

def generate_transaction(n,customer):
    transactions = []
    for i in range(n):
        transaction_id = str(uuid.uuid4())
        customer_id = customer.customer_id
        amount = round(random.uniform(1000,100000),3)
        payment_method = customer.preferred_payment_method
        failure_reason = random.choice(["insufficient_funds", "upi_timeout"])
        timestamp = datetime.now()
        status = "failed"
        transaction = Transaction(transaction_id,customer_id,amount,payment_method,failure_reason,timestamp,status)
        transactions.append(transaction)
    return transactions
        
def generate_recovery_attempt(n, transaction, customer):
    attempts = []
    for i in range(n):
        attempt_id = str(uuid.uuid4())
        transaction_id = transaction.transaction_id
        action = random.choice(["retry_immediately","retry_after_delay","use_another_payment_method","send_payment_reminder"])
        hours_later = random.randint(1,72)
        timestamp = transaction.timestamp + timedelta(hours = hours_later)
        outcome = simulate_outcome(action, transaction.failure_reason, customer.risk_tier, i + 1, hours_later)
        attempt = RecoveryAttempt(attempt_id,transaction_id,action,timestamp,outcome)
        attempts.append(attempt)
    return attempts

def main():
    customers = generate_customers(100)

    all_transactions = []
    all_attempts = []
    for customer in customers:
        txns = generate_transaction(random.randint(1, 5), customer)
        all_transactions.extend(txns)
        for transaction in txns:
            attempts = generate_recovery_attempt(random.randint(1, 3), transaction, customer)
            all_attempts.extend(attempts)

    pd.DataFrame([dataclasses.asdict(c) for c in customers]).to_csv("data/customers.csv",index=False)
    pd.DataFrame([dataclasses.asdict(t) for t in all_transactions]).to_csv("data/transactions.csv",index=False)
    pd.DataFrame([dataclasses.asdict(a) for a in all_attempts]).to_csv("data/attempts.csv",index=False)
    print(f"Generated {len(customers)} customers, {len(all_transactions)} transactions, {len(all_attempts)} attempts")

if __name__ == "__main__":
    main()
