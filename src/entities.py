from dataclasses import dataclass
from datetime import datetime
@dataclass
class Customer:
    customer_id: str
    preferred_payment_method: str
    risk_tier: str

@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    amount: float
    payment_method: str
    failure_reason: str
    timestamp: datetime
    status: str
@dataclass
class RecoveryAttempt:
    attempt_id: str
    transaction_id: str
    action: str
    timestamp: datetime
    outcome: str