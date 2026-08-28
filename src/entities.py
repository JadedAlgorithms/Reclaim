class Customer:
    def __init__(self, customer_id):
        self.customer_id = customer_id

class Transaction:
    def __init__(self, transaction_id, customer_id, amount):
        self.transaction_id = transaction_id
        self.customer_id = customer_id
        self.amount = amount

class Recovery_Attempt:
    def __init__(self, attempt_id, transaction_id, action, timestamp, outcome, recovered_amount):
        self.attempt_id = attempt_id
        self.transaction_id = transaction_id
        self.action = action
        self.timestamp = timestamp
        self.outcome = outcome
        self.recovered_amount = recovered_amount