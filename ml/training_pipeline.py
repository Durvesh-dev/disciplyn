"""
Disciplyn - model training pipeline (used by train.py and Disciplyn_Model_Training.ipynb)

Stages
  1. Load features (one row per trader-month window) + ground-truth labels
  2. Group-aware split: 80% traders train / 20% traders test  (no trader appears in both)
  3. Algorithm comparison per tag: LogisticRegression, RandomForest, HistGradientBoosting, MLP
     (+ XGBoost / LightGBM automatically if installed) using 5-fold GroupKFold ROC-AUC
  4. Hyper-parameter tuning of the best algorithm per tag (RandomizedSearchCV, GroupKFold)
  5. Decision-threshold tuning on out-of-fold predictions (max F1)
  6. Final fit on train, evaluation on held-out test: Acc, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier
  7. Explainability: permutation feature importance
"""
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import MODELS

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, MODEL_DIR, REPORT_DIR = (os.path.join(HERE, d) for d in ("data", "models", "reports"))
SEED = 42


def target_of(cfg):
    return cfg["negative"] if cfg["type"] == "pair" else cfg["tag"]


def display_name(cfg):
    return f"{cfg['negative']} / {cfg['positive']}" if cfg["type"] == "pair" else cfg["tag"]


# ----------------------------------------------------------------------------- data
def load_dataset():
    X = pd.read_csv(os.path.join(DATA, "features.csv"))
    y = pd.read_csv(os.path.join(DATA, "labels.csv"))
    df = X.merge(y, on="trader_id", how="inner")
    return df


def split(df, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr_idx, te_idx = next(gss.split(df, groups=df["trader_id"]))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


# ----------------------------------------------------------------------------- algorithms
def candidate_algorithms():
    algos = {
        "LogisticRegression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                            LogisticRegression(max_iter=3000, class_weight="balanced")),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=5,
                                                             class_weight="balanced", n_jobs=-1, random_state=SEED)),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                                               early_stopping=True, class_weight="balanced",
                                                               random_state=SEED),
        "MLP (Neural Net)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
                                                        early_stopping=True, random_state=SEED)),
    }
    try:
        from xgboost import XGBClassifier
        algos["XGBoost"] = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                                         colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        algos["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                           class_weight="balanced", random_state=SEED, verbose=-1)
    except ImportError:
        pass
    return algos


SEARCH_SPACES = {
    "LogisticRegression": {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                           "logisticregression__penalty": ["l2"]},
    "RandomForest": {"randomforestclassifier__n_estimators": [150, 250, 400],
                     "randomforestclassifier__max_depth": [8, 12, 16, None],
                     "randomforestclassifier__min_samples_leaf": [1, 3, 5, 10],
                     "randomforestclassifier__max_features": ["sqrt", 0.5, 0.8]},
    "HistGradientBoosting": {"learning_rate": [0.03, 0.05, 0.08, 0.12], "max_leaf_nodes": [15, 31, 63],
                             "max_iter": [200, 400, 600], "l2_regularization": [0, 0.5, 1.0, 3.0],
                             "min_samples_leaf": [10, 20, 40]},
    "MLP (Neural Net)": {"mlpclassifier__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (64, 64, 32)],
                         "mlpclassifier__alpha": [1e-4, 1e-3, 1e-2], "mlpclassifier__learning_rate_init": [1e-3, 3e-3]},
    "XGBoost": {"n_estimators": [200, 400, 700], "max_depth": [3, 4, 6], "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 0.9, 1.0]},
    "LightGBM": {"n_estimators": [200, 400, 700], "num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1],
                 "min_child_samples": [10, 20, 40]},
}


# ----------------------------------------------------------------------------- stage 3
def compare_algorithms(train, folds=5, log=print):
    algos = candidate_algorithms()
    gkf = GroupKFold(n_splits=folds)
    rows = []
    for name, cfg in MODELS.items():
        X, y, g = train[cfg["features"]], train[target_of(cfg)], train["trader_id"]
        r = {"model": name, "tag(s)": display_name(cfg)}
        for aname, est in algos.items():
            t0 = time.time()
            scores = []
            for tr_i, va_i in gkf.split(X, y, g):
                m = clone(est).fit(X.iloc[tr_i], y.iloc[tr_i])
                scores.append(roc_auc_score(y.iloc[va_i], m.predict_proba(X.iloc[va_i])[:, 1]))
            r[aname] = round(float(np.mean(scores)), 4)
            r[aname + " ±"] = round(float(np.std(scores)), 4)
            log(f"  {name:22s} {aname:22s} CV ROC-AUC {np.mean(scores):.4f} ± {np.std(scores):.4f}  ({time.time()-t0:.0f}s)")
        rows.append(r)
    table = pd.DataFrame(rows)
    algo_cols = list(algos)
    table["best_algorithm"] = table[algo_cols].idxmax(axis=1)
    return table, algo_cols


# ----------------------------------------------------------------------------- stage 4-6
def train_one(name, aname, train, test, n_iter=10, folds=3, log=print, save=True):
    """Tune + threshold + evaluate + save ONE model. Returns (result_dict, artefact_dict)."""
    algos = candidate_algorithms()
    cfg = MODELS[name]
    feats, tgt = cfg["features"], target_of(cfg)
    Xtr, ytr, gtr = train[feats], train[tgt], train["trader_id"]
    Xte, yte = test[feats], test[tgt]
    t0 = time.time()
    space = SEARCH_SPACES.get(aname, {})
    if space and n_iter > 0:
        search = RandomizedSearchCV(clone(algos[aname]), space, n_iter=n_iter, scoring="roc_auc",
                                    cv=GroupKFold(n_splits=folds), random_state=SEED, n_jobs=1, refit=True,
                                    error_score=np.nan)
        search.fit(Xtr, ytr, groups=gtr)
        best, best_params, cv_auc = search.best_estimator_, search.best_params_, float(search.best_score_)
    else:   # no tuning (fallback path)
        best = clone(algos[aname]).fit(Xtr, ytr)
        best_params, cv_auc = {}, float("nan")
    # decision threshold from out-of-fold predictions on TRAIN (never touches test)
    oof = cross_val_predict(clone(best), Xtr, ytr, groups=gtr, cv=GroupKFold(n_splits=folds),
                            method="predict_proba")[:, 1]
    if not np.isfinite(cv_auc):
        cv_auc = float(roc_auc_score(ytr, oof))
    grid = np.linspace(0.2, 0.8, 61)
    thr = float(grid[np.argmax([f1_score(ytr, oof >= t) for t in grid])])
    p = best.predict_proba(Xte)[:, 1]
    pred = (p >= thr).astype(int)
    train_auc = float(roc_auc_score(ytr, best.predict_proba(Xtr)[:, 1]))
    test_auc = float(roc_auc_score(yte, p))
    gap = train_auc - test_auc
    fit_status = "overfitting" if gap > 0.05 else ("underfitting" if train_auc < 0.75 else "good fit")
    imp = permutation_importance(best, Xte, yte, n_repeats=5, random_state=SEED, scoring="roc_auc", n_jobs=1)
    importance = sorted(zip(feats, imp.importances_mean.round(4)), key=lambda x: -x[1])
    fpr, tpr, _ = roc_curve(yte, p)
    result = {
        "tag(s)": display_name(cfg), "predicts": tgt, "type": cfg["type"], "opposite": cfg.get("positive"),
        "algorithm": aname, "best_params": {k.split("__")[-1]: (list(v) if isinstance(v, tuple) else v)
                                            for k, v in best_params.items()},
        "cv_roc_auc_tuned": round(cv_auc, 4), "threshold": round(thr, 3),
        "train_roc_auc": round(train_auc, 4), "train_test_gap": round(gap, 4), "fit_status": fit_status,
        "test": {"accuracy": round(accuracy_score(yte, pred), 4), "precision": round(precision_score(yte, pred), 4),
                 "recall": round(recall_score(yte, pred), 4), "f1": round(f1_score(yte, pred), 4),
                 "roc_auc": round(test_auc, 4), "pr_auc": round(average_precision_score(yte, p), 4),
                 "brier": round(brier_score_loss(yte, p), 4)},
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "n_train_windows": int(len(Xtr)), "n_test_windows": int(len(Xte)),
        "positive_rate_train": round(float(ytr.mean()), 3),
        "features": feats, "feature_importance": [(f, float(v)) for f, v in importance],
        "train_seconds": round(time.time() - t0, 1),
    }
    bundle = {"model": best, "features": feats, "config": cfg, "threshold": thr, "algorithm": aname}
    if save:
        save_bundle(name, bundle)
    t = result["test"]
    log(f"  {name:22s} {aname:22s} thr={thr:.2f} acc={t['accuracy']:.3f} f1={t['f1']:.3f} auc={t['roc_auc']:.3f} "
        f"train_auc={train_auc:.3f} gap={gap:+.3f} [{fit_status}] ({time.time()-t0:.0f}s)")
    return result, {"proba": p, "y": yte.values, "fpr": fpr, "tpr": tpr, "bundle": bundle}


def save_bundle(name, bundle):
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = os.path.join(MODEL_DIR, f"{name}.joblib.tmp")
    joblib.dump(bundle, tmp)
    os.replace(tmp, os.path.join(MODEL_DIR, f"{name}.joblib"))     # atomic save: never a half-written model


def train_with_overfit_guard(name, aname, train, test, n_iter=10, folds=3, log=print):
    """Train the chosen model; if it overfits (train-test ROC-AUC gap > 0.05), also train a simple
    regularised Logistic Regression and keep it when its CROSS-VALIDATED AUC is within 0.01.
    (Decision uses CV on train only - the test set is never used for model selection.)"""
    res, art = train_one(name, aname, train, test, n_iter, folds, log, save=False)
    if res["fit_status"] == "overfitting" and aname != "LogisticRegression":
        log(f"  {name}: overfitting detected (gap {res['train_test_gap']:+.3f}) -> trying regularised LogisticRegression")
        res2, art2 = train_one(name, "LogisticRegression", train, test, 6, folds, log, save=False)
        if res2["cv_roc_auc_tuned"] >= res["cv_roc_auc_tuned"] - 0.01:
            log(f"  {name}: switched to LogisticRegression (CV {res2['cv_roc_auc_tuned']:.3f} vs {res['cv_roc_auc_tuned']:.3f})")
            res, art = res2, art2
            res["note"] = "replaced overfitting model by regularised LogisticRegression"
    save_bundle(name, art["bundle"])
    return res, art


def tune_and_evaluate(train, test, comparison, n_iter=12, folds=3, log=print):
    results, artefacts = {}, {}
    for _, row in comparison.iterrows():
        results[row["model"]], artefacts[row["model"]] = train_with_overfit_guard(
            row["model"], row["best_algorithm"], train, test, n_iter, folds, log)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results, artefacts


def results_table(results):
    return pd.DataFrame([{"model": k, "tag(s)": v["tag(s)"], "algorithm": v["algorithm"], "threshold": v["threshold"],
                          **v["test"], "train_roc_auc": v.get("train_roc_auc"), "gap": v.get("train_test_gap"),
                          "fit": v.get("fit_status")} for k, v in results.items()])
