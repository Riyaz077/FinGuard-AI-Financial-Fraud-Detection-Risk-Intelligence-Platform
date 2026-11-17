from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = [
    "amount", "hour", "day_of_week", "distance_from_home_km",
    "transactions_last_1h", "avg_amount_24h", "merchant_risk",
    "device_trust", "international", "card_present"
]

def generate_transactions(n: int = 18000, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible, explicitly synthetic transaction dataset."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    day = rng.integers(0, 7, n)
    amount = np.clip(rng.lognormal(3.5, 1.0, n), 1, 12000)
    distance = np.clip(rng.gamma(1.7, 22, n), 0, 1500)
    velocity = np.clip(rng.poisson(1.7, n), 0, 18)
    avg_amount = np.clip(rng.lognormal(3.3, .65, n), 1, 5000)
    merchant_risk = rng.beta(2, 5, n)
    device_trust = rng.beta(7, 2, n)
    international = rng.binomial(1, .12, n)
    card_present = rng.binomial(1, .64, n)

    # Synthetic fraud propensity: intentionally imbalanced and non-linear.
    logit = (
        -5.0
        + 1.05 * (amount > 900)
        + 1.15 * (hour <= 4)
        + 0.0018 * distance
        + 0.22 * velocity
        + 2.0 * merchant_risk
        + 1.45 * (1 - device_trust)
        + 1.05 * international
        - 0.95 * card_present
        + 0.85 * ((amount > 1200) & (distance > 100))
        + 0.75 * ((velocity >= 5) & (device_trust < .35))
        + rng.normal(0, .45, n)
    )
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)
    ids = [f"TXN-{i:07d}" for i in range(1, n + 1)]
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 24 * 60, n), unit="m"
    )
    df = pd.DataFrame({
        "transaction_id": ids,
        "timestamp": dates,
        "amount": amount.round(2),
        "hour": hour,
        "day_of_week": day,
        "distance_from_home_km": distance.round(2),
        "transactions_last_1h": velocity,
        "avg_amount_24h": avg_amount.round(2),
        "merchant_risk": merchant_risk.round(4),
        "device_trust": device_trust.round(4),
        "international": international,
        "card_present": card_present,
        "is_fraud": fraud
    })
    return df.sort_values("timestamp").reset_index(drop=True)
