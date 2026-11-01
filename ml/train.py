"""
Train all 14 behaviour models from the command line (same pipeline as the notebook).

  python generate_dataset.py     # ~2-4 min  -> data/trades.csv (~1.9M trades)
  python build_features.py       # ~3-6 min  -> data/features.csv (~29k trader-month windows)
  python train.py                # ~10 min   -> models/*.joblib, models/metrics.json
"""
import time

import training_pipeline as tp

if __name__ == "__main__":
    t0 = time.time()
    df = tp.load_dataset()
    train, test = tp.split(df)
    print(f"train {len(train):,} windows ({train.trader_id.nunique():,} traders) | "
          f"test {len(test):,} windows ({test.trader_id.nunique():,} traders)")
    print("\n[1/2] Algorithm comparison (5-fold GroupKFold ROC-AUC)")
    comparison, algo_cols = tp.compare_algorithms(train, folds=5)
    comparison.to_csv(f"{tp.REPORT_DIR}/algorithm_comparison.csv", index=False)
    print("\n[2/2] Tuning best algorithm per tag + held-out test evaluation")
    results, _ = tp.tune_and_evaluate(train, test, comparison, n_iter=10, folds=3)
    res = tp.results_table(results)
    res.to_csv(f"{tp.REPORT_DIR}/test_metrics.csv", index=False)
    print("\n", res.to_string(index=False))
    print("\nAVERAGE:", res[["accuracy", "f1", "roc_auc"]].mean().round(4).to_dict(),
          f"| total {(time.time()-t0)/60:.1f} min")
