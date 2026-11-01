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

from features import MODELS, trader_features

HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = None


def load_models(model_dir=os.path.join(HERE, "models")):
    global _MODELS
    if _MODELS is None:
        _MODELS = {name: joblib.load(os.path.join(model_dir, f"{name}.joblib")) for name in MODELS}
    return _MODELS


def analyze(trades: pd.DataFrame, questionnaire: pd.DataFrame | None = None, uncertain_band=(0.4, 0.6)):
    feats = trader_features(trades, questionnaire)
    X = pd.DataFrame([feats])
    tags, models_out = [], []
    for name, b in load_models().items():
        cfg = b["config"]
        p = float(b["model"].predict_proba(X[b["features"]])[:, 1][0])
        thr = float(b.get("threshold", 0.5))          # tuned decision threshold per model
        uncertain = abs(p - thr) < 0.1                # too close to call -> "uncertain"
        if cfg["type"] == "pair":
            tags.append({"tag": cfg["negative"], "polarity": "negative", "probability": round(p, 3),
                         "present": p >= thr and not uncertain, "model": name})
            tags.append({"tag": cfg["positive"], "polarity": "positive", "probability": round(1 - p, 3),
                         "present": p < thr and not uncertain, "model": name})
        else:
            tags.append({"tag": cfg["tag"], "polarity": cfg["polarity"], "probability": round(p, 3),
                         "present": p >= thr and not uncertain, "model": name})
        models_out.append({"model": name, "probability": round(p, 3), "uncertain": uncertain})

    # Behaviour score 0-100: average "goodness" across the 14 dimensions
    good = []
    for name, b in load_models().items():
        cfg = b["config"]
        p = next(m["probability"] for m in models_out if m["model"] == name)
        bad = cfg["type"] == "pair" or cfg.get("polarity") == "negative"
        good.append(1 - p if bad else p)
    score = int(round(100 * float(np.mean(good))))

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
    return {"behaviour_score": score, "tags": tags, "models": models_out, "stats": stats,
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
