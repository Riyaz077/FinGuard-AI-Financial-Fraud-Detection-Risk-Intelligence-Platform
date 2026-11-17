from pathlib import Path
import sys
import pandas as pd
from src.data_generator import generate_transactions
from src.modeling import train_and_save

ROOT = Path(__file__).resolve().parent
df = generate_transactions()
df.to_csv(ROOT / "data" / "transactions.csv", index=False)
best, metrics, _, _ = train_and_save(df, str(ROOT / "models"))
print(f"Dataset created: {len(df):,} synthetic transactions")
print(metrics.to_string(index=False))
print(f"Best model by ROC-AUC: {best}")
print("Setup complete. Run: streamlit run app.py")
