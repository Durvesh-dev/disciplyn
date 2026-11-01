"""
Disciplyn - ONE crash-safe script: dataset -> features -> 14 models -> reports -> zip.

    python run_all.py                     # full run (10,000 traders, ~25-40 min on Kaggle CPU)
    python run_all.py --quick             # smoke test (~3 min) - run this FIRST
    python run_all.py --fresh             # ignore checkpoints, start over

Safety features
  * Checkpoints: every stage and every model is saved as soon as it finishes. If the run dies,
    simply run the same command again - finished work is skipped, it resumes where it stopped.
  * Per-model isolation: a failing model never stops the others. It is retried with a smaller
    search, then falls back to Logistic Regression. Failures are listed in the final summary.
  * Atomic writes (tmp file + rename) -> no half-written CSV / model files after a crash.
  * Everything is logged to <out>/logs/run.log with timestamps.
  * Output folder is zipped at the end (download one file from Kaggle "Output").
"""
import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# --------------------------------------------------------------------------- logging
class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a", buffering=1)
        self.t0 = time.time()

    def __call__(self, *msg):
        line = f"[{datetime.now():%H:%M:%S} +{(time.time()-self.t0)/60:5.1f}m] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.fh.write(line + "\n")


def atomic_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def default_out():
    return "/kaggle/working/disciplyn_out" if os.path.isdir("/kaggle/working") else os.path.join(HERE, "run_output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out())
    ap.add_argument("--traders", type=int, default=10000)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--tune_iter", type=int, default=15)
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--fresh", action="store_true", help="delete previous output and start over")
    ap.add_argument("--data_dir", default=None, help="reuse an existing data folder (trades.csv etc.)")
    a = ap.parse_args()
    if a.quick:
        a.traders, a.days, a.cv_folds, a.tune_iter = 400, 40, 3, 3
        a.out = a.out.rstrip("/") + "_quick"
    if a.fresh and os.path.isdir(a.out):
        shutil.rmtree(a.out)
    for d in ("data", "models", "reports", "logs", "checkpoints"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    log = Logger(os.path.join(a.out, "logs", "run.log"))
    log("=" * 70)
    log(f"START  out={a.out} traders={a.traders} days={a.days} quick={a.quick} cpus={os.cpu_count()}")

    import numpy as np
    import pandas as pd
    import sklearn
    import training_pipeline as tp
    from features import MODELS, build_feature_table

    tp.DATA = a.data_dir or os.path.join(a.out, "data")
    tp.MODEL_DIR = os.path.join(a.out, "models")
    tp.REPORT_DIR = os.path.join(a.out, "reports")
    ck = os.path.join(a.out, "checkpoints")
    json.dump({"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
               "pandas": pd.__version__, "args": vars(a)},
              open(os.path.join(a.out, "models", "versions.json"), "w"), indent=2)
    failures = []

    # ------------------------------------------------------------------ 1. dataset
    trades_csv = os.path.join(tp.DATA, "trades.csv")
    if os.path.exists(trades_csv) and os.path.exists(os.path.join(tp.DATA, "labels.csv")):
        log("[1/5] dataset exists -> skip")
    else:
        log(f"[1/5] generating dataset ({a.traders:,} traders x {a.days} days)")
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(HERE, "generate_dataset.py"), "--traders", str(a.traders),
                            "--days", str(a.days), "--seed", str(a.seed), "--out", tp.DATA],
                           capture_output=True, text=True)
        log(r.stdout.strip().splitlines()[-1] if r.stdout else "")
        if r.returncode != 0:
            log("dataset generation FAILED:\n" + r.stderr[-3000:])
            raise SystemExit(1)

    # ------------------------------------------------------------------ 2. features
    feat_csv = os.path.join(tp.DATA, "features.csv")
    if os.path.exists(feat_csv):
        log("[2/5] features exist -> skip")
    else:
        log("[2/5] building features")
        trades = pd.read_csv(trades_csv)
        q = pd.read_csv(os.path.join(tp.DATA, "questionnaire.csv"))
        log(f"      loaded {len(trades):,} trades")
        try:
            X = build_feature_table(trades, q, workers=max(1, os.cpu_count() or 1))
        except Exception:
            log("      parallel feature build failed, retrying single-process:\n" + traceback.format_exc()[-1500:])
            X = build_feature_table(trades, q, workers=1)
        atomic_csv(X, feat_csv)
        del trades
        log(f"      {len(X):,} windows x {X.shape[1]} cols")

    df = tp.load_dataset()
    train, test = tp.split(df)
    log(f"      train {len(train):,} windows / {train.trader_id.nunique():,} traders | "
        f"test {len(test):,} / {test.trader_id.nunique():,} traders | overlap "
        f"{len(set(train.trader_id) & set(test.trader_id))}")

    # ------------------------------------------------------------------ 3. algorithm comparison (per model checkpoint)
    log(f"[3/5] algorithm comparison ({a.cv_folds}-fold GroupKFold)")
    rows = []
    for name in MODELS:
        cp = os.path.join(ck, f"compare_{name}.json")
        if os.path.exists(cp):
            rows.append(json.load(open(cp)))
            continue
        try:
            sub = train  # compare_algorithms works on all models; run per model for checkpointing
            saved = dict(MODELS)
            tp.MODELS.clear(); tp.MODELS[name] = saved[name]
            table, algo_cols = tp.compare_algorithms(sub, folds=a.cv_folds, log=log)
            tp.MODELS.clear(); tp.MODELS.update(saved)
            row = table.iloc[0].to_dict()
        except Exception:
            tp.MODELS.clear(); tp.MODELS.update(saved)
            log(f"  {name}: comparison FAILED -> default HistGradientBoosting\n" + traceback.format_exc()[-1500:])
            failures.append(f"compare:{name}")
            row = {"model": name, "tag(s)": tp.display_name(MODELS[name]), "best_algorithm": "HistGradientBoosting"}
        json.dump(row, open(cp, "w"), default=float)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    atomic_csv(comparison, os.path.join(tp.REPORT_DIR, "algorithm_comparison.csv"))

    # ------------------------------------------------------------------ 4. tune + evaluate (per model checkpoint)
    log(f"[4/5] tuning ({a.tune_iter} iters) + threshold + test evaluation")
    results, art = {}, {}
    for _, row in comparison.iterrows():
        name, algo = row["model"], row["best_algorithm"]
        cp = os.path.join(ck, f"result_{name}.json")
        if os.path.exists(cp) and os.path.exists(os.path.join(tp.MODEL_DIR, f"{name}.joblib")):
            results[name] = json.load(open(cp))
            log(f"  {name}: done earlier -> skip")
            continue
        attempts = [(algo, a.tune_iter), (algo, max(3, a.tune_iter // 3)), ("LogisticRegression", 0)]
        for k, (alg, it) in enumerate(attempts):
            try:
                res, ar = tp.train_with_overfit_guard(name, alg, train, test, n_iter=it, folds=3, log=log)
                results[name], art[name] = res, ar
                json.dump(res, open(cp, "w"), indent=1, default=float)
                break
            except Exception:
                log(f"  {name}: attempt {k+1} ({alg}, iter={it}) FAILED\n" + traceback.format_exc()[-1500:])
                failures.append(f"train:{name}:attempt{k+1}")
        else:
            log(f"  {name}: ALL attempts failed - model skipped")

    json.dump(results, open(os.path.join(tp.MODEL_DIR, "metrics.json"), "w"), indent=2, default=float)

    # ------------------------------------------------------------------ 5. reports (never fatal)
    log("[5/5] reports")
    try:
        res = tp.results_table(results)
        atomic_csv(res, os.path.join(tp.REPORT_DIR, "test_metrics.csv"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = res.set_index("tag(s)")[["accuracy", "f1", "roc_auc"]]
        ax = r.plot.barh(figsize=(11, 7), color=["#2a6f97", "#61a5c2", "#e09f3e"], width=0.8)
        ax.invert_yaxis(); ax.set_xlim(0.5, 1); ax.set_title("Held-out test performance (unseen traders)")
        plt.tight_layout(); plt.savefig(os.path.join(tp.REPORT_DIR, "test_metrics.png"), dpi=150); plt.close()
        fit = res.set_index("tag(s)")[["train_roc_auc", "roc_auc"]].rename(columns={"roc_auc": "test_roc_auc"})
        ax = fit.plot.barh(figsize=(11, 7), color=["#94a3b8", "#2a6f97"], width=0.8)
        ax.invert_yaxis(); ax.set_xlim(0.5, 1); ax.set_title("Overfitting check: train vs test ROC-AUC (gap < 0.05 = good fit)")
        plt.tight_layout(); plt.savefig(os.path.join(tp.REPORT_DIR, "fit_check.png"), dpi=150); plt.close()
        if art:
            fig, axes = plt.subplots(3, 5, figsize=(20, 12)); axes = axes.ravel()
            for axx, (n, v) in zip(axes, art.items()):
                axx.plot(v["fpr"], v["tpr"], lw=2); axx.plot([0, 1], [0, 1], "--", color="grey")
                axx.set_title(f"{results[n]['tag(s)'][:34]}\nAUC={results[n]['test']['roc_auc']:.3f}", fontsize=9)
            for axx in axes[len(art):]:
                axx.axis("off")
            plt.tight_layout(); plt.savefig(os.path.join(tp.REPORT_DIR, "roc_curves.png"), dpi=120); plt.close()
        log("\n" + res[["model", "algorithm", "accuracy", "f1", "roc_auc", "train_roc_auc", "gap", "fit"]].to_string(index=False))
        log("AVERAGE: " + json.dumps(res[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean().round(4).to_dict()))
    except Exception:
        log("report generation failed (models are still saved):\n" + traceback.format_exc()[-1500:])
        failures.append("reports")

    # ------------------------------------------------------------------ zip models + reports (+ features)
    try:
        bundle = os.path.join(a.out, "_bundle")
        shutil.rmtree(bundle, ignore_errors=True)
        for d in ("models", "reports", "logs"):
            shutil.copytree(os.path.join(a.out, d), os.path.join(bundle, d))
        os.makedirs(os.path.join(bundle, "data"), exist_ok=True)
        for f in ("features.csv", "labels.csv", "questionnaire.csv", "traders.csv"):
            if os.path.exists(os.path.join(tp.DATA, f)):
                shutil.copy(os.path.join(tp.DATA, f), os.path.join(bundle, "data", f))
        z = shutil.make_archive(a.out.rstrip("/") + "_models_and_reports", "zip", bundle)
        shutil.rmtree(bundle, ignore_errors=True)
        log(f"ZIP: {z}")
    except Exception:
        log("zip failed:\n" + traceback.format_exc()[-1000:])

    log(f"FINISHED: {len(results)}/{len(MODELS)} models trained | failures: {failures or 'none'}")
    return 0 if len(results) == len(MODELS) else 2


if __name__ == "__main__":
    sys.exit(main())
