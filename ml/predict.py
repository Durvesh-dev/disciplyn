"""
Inference: a trader's trade book (+ optional questionnaire) -> 20 behaviour tags + Behaviour Score.

  python predict.py samples/persona_impulsive_trades.csv [samples/persona_impulsive_questionnaire.csv]
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from features import ALL_FEATURES, MODELS, trader_features

HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = None
UNCERTAIN_BAND = 0.1   # |p - threshold| below this -> "too close to call"


def load_models(model_dir=os.path.join(HERE, "models")):
    global _MODELS
    if _MODELS is None:
        _MODELS = {name: joblib.load(os.path.join(model_dir, f"{name}.joblib")) for name in MODELS}
    return _MODELS


def display_name(name: str) -> str:
    cfg = MODELS[name]
    return f"{cfg['negative']} / {cfg['positive']}" if cfg["type"] == "pair" else cfg["tag"]


def predict_from_features(features: dict) -> dict:
    """Run the 14 models on ONE already-engineered feature row.

    `features` maps feature name -> number (or None/NaN when unknown, e.g. a user who
    skipped the questionnaire). Missing keys become NaN. This is the single code path
    used by both analyze() (features computed from a trade book) and the /api/predict-features
    endpoint (features typed in by hand), so there is no skew between them.
    """
    row = {f: features.get(f, np.nan) for f in ALL_FEATURES}
    X = pd.DataFrame([row]).astype(float)

    tags, models_out, predictions = [], [], []
    for name, b in load_models().items():
        cfg = b["config"]
        p = float(b["model"].predict_proba(X[b["features"]])[:, 1][0])
        thr = float(b.get("threshold", 0.5))          # tuned decision threshold per model
        uncertain = abs(p - thr) < UNCERTAIN_BAND     # too close to call -> "uncertain"
        p3 = round(p, 3)
        if cfg["type"] == "pair":
            tags.append({"tag": cfg["negative"], "polarity": "negative", "probability": p3,
                         "present": p >= thr and not uncertain, "model": name})
            tags.append({"tag": cfg["positive"], "polarity": "positive", "probability": round(1 - p, 3),
                         "present": p < thr and not uncertain, "model": name})
            predicted_tag = cfg["negative"] if p >= thr else cfg["positive"]
            polarity = "negative" if p >= thr else "positive"
        else:
            tags.append({"tag": cfg["tag"], "polarity": cfg["polarity"], "probability": p3,
                         "present": p >= thr and not uncertain, "model": name})
            predicted_tag = cfg["tag"] if p >= thr else f"Not {cfg['tag']}"
            flip = {"negative": "positive", "positive": "negative"}
            polarity = cfg["polarity"] if p >= thr else flip[cfg["polarity"]]
        models_out.append({"model": name, "probability": p3, "uncertain": uncertain})
        predictions.append({
            "model": name, "display_name": display_name(name), "type": cfg["type"],
            "algorithm": b.get("algorithm", "unknown"), "probability": p3, "threshold": round(thr, 3),
            "predicted_tag": predicted_tag, "polarity": polarity, "uncertain": uncertain,
            "confidence": round(p if p >= thr else 1 - p, 3),
            "features_used": {f: (None if pd.isna(row[f]) else float(row[f])) for f in b["features"]},
        })

    # Behaviour score 0-100: average "goodness" across the 14 dimensions
    good = []
    for m in models_out:
        cfg = MODELS[m["model"]]
        bad = cfg["type"] == "pair" or cfg.get("polarity") == "negative"
        good.append(1 - m["probability"] if bad else m["probability"])
    score = int(round(100 * float(np.mean(good))))

    return {"behaviour_score": score, "tags": tags, "models": models_out, "predictions": predictions}


def analyze(trades: pd.DataFrame, questionnaire: pd.DataFrame | None = None, uncertain_band=(0.4, 0.6)):
    feats = trader_features(trades, questionnaire)
    out = predict_from_features(feats)

    # top evidence for the LLM / dashboard
    stats = {
        "n_trades": int(feats["_n_trades"]), "win_rate": round(feats["_win_rate"], 3),
        "net_pnl_inr": round(feats["_net_pnl"], 2),
        "trades_per_day": round(feats["trades_per_day"], 2),
        "quick_reentry_after_loss_rate": round(feats["quick_reentry_after_loss_rate"], 3),
        "size_after_loss_vs_win": round(feats["size_after_loss_vs_win"], 2),
        "sl_set_rate": round(feats["sl_set_rate"], 3), "sl_modified_rate": round(feats["sl_modified_rate"], 3),
        "chase_entry_rate": round(feats["chase_entry_rate"], 3),
        "loss_hold_vs_win_hold": round(feats["loss_hold_vs_win_hold"], 2),
        "avg_leverage": round(feats["avg_leverage"], 2), "max_drawdown_pct": round(feats["max_drawdown_pct"], 2),
        "target_held_rate": round(feats["target_held_rate"], 3),
        "early_exit_rate": round(feats["early_exit_rate"], 3),
        "off_plan_rate": round(feats["off_plan_rate"], 3), "journal_loss_rate": round(feats["journal_loss_rate"], 3),
        "strategy_count": int(feats["strategy_count"]),
    }
    return {"behaviour_score": out["behaviour_score"], "tags": out["tags"], "models": out["models"], "stats": stats,
            "has_questionnaire": questionnaire is not None and len(questionnaire) > 0}


if __name__ == "__main__":
    tr = pd.read_csv(sys.argv[1])
    q = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else None
    r = analyze(tr, q)
    print("Behaviour score:", r["behaviour_score"])
    for t in sorted(r["tags"], key=lambda x: -x["probability"]):
        if t["present"]:
            print(f"  [{t['polarity'][:3].upper()}] {t['tag']:30s} {t['probability']:.2f}")
    print(json.dumps(r["stats"], indent=1))
