"""Generates a synthetic customer dataset (RFM-style) with 5 hidden segments.
Replace data/customers.csv with a real dataset (e.g. Kaggle Online Retail / Mall Customers)
whenever you like - the app works with any numeric table."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# name, size, (mean, std) for: age, income(k), orders/yr, avg order value, days since last purchase
SEGMENTS = [
    ("loyal",       150, (42, 8),  (85, 12),  (24, 5), (95, 15),  (12, 6)),
    ("bargain",     170, (27, 5),  (35, 8),   (18, 4), (28, 7),   (20, 10)),
    ("big_spender",  90, (48, 9),  (110, 15), (5, 2),  (260, 40), (60, 25)),
    ("at_risk",     120, (38, 10), (60, 15),  (8, 3),  (60, 15),  (170, 40)),
    ("new",          70, (30, 7),  (50, 15),  (2, 1),  (45, 12),  (10, 6)),
]

blocks = [np.column_stack([rng.normal(m, s, n) for m, s in p]) for _, n, *p in SEGMENTS]
raw = np.vstack(blocks)

df = pd.DataFrame(raw, columns=["age", "annual_income_k", "orders_per_year", "aov", "days_since_last_purchase"])
df["age"] = df["age"].clip(18, 75).round().astype(int)
df["annual_income_k"] = df["annual_income_k"].clip(10).round(1)
df["orders_per_year"] = df["orders_per_year"].clip(1).round().astype(int)
df["days_since_last_purchase"] = df["days_since_last_purchase"].clip(1).round().astype(int)
df["annual_spend"] = (df["orders_per_year"] * df["aov"].clip(10)).round(0).astype(int)
df = df.drop(columns="aov").sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "customer_id", [f"C{i:04d}" for i in range(1, len(df) + 1)])

Path("data").mkdir(exist_ok=True)
df.to_csv("data/customers.csv", index=False)
print(df.shape)
