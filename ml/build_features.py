"""Step 2: trades.csv + questionnaire.csv -> data/features.csv (one row per trader-month window)."""
import os
import time

import pandas as pd

from features import build_feature_table

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

if __name__ == "__main__":
    t0 = time.time()
    trades = pd.read_csv(os.path.join(DATA, "trades.csv"))
    q = pd.read_csv(os.path.join(DATA, "questionnaire.csv"))
    print(f"Loaded {len(trades):,} trades, {q.trader_id.nunique():,} questionnaires ({time.time()-t0:.0f}s)")
    X = build_feature_table(trades, q, workers=os.cpu_count() or 2)
    X.to_csv(os.path.join(DATA, "features.csv"), index=False)
    print(f"Saved {len(X):,} windows x {X.shape[1]} columns in {time.time()-t0:.0f}s")
