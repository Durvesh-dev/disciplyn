"""
Parity test: predict_from_features(trader_features(...)) must match analyze(...) exactly.

The web demo posts hand-typed feature values to /api/predict-features, which calls
predict_from_features. The notebooks and ml/demo.py call analyze. Both must be the same
code path, or the demo would show different numbers from the report.

  python test_predict_features.py        (plain asserts)
  pytest test_predict_features.py
"""
import os

import pandas as pd

from features import trader_features
from predict import analyze, predict_from_features

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")

# Behaviour Scores published in the report / demo for the three sample traders.
EXPECTED_SCORES = {"impulsive": 7, "disciplined": 94, "fearful": 80}


def _persona(pid):
    tr = pd.read_csv(os.path.join(SAMPLES, f"persona_{pid}_trades.csv"))
    q = pd.read_csv(os.path.join(SAMPLES, f"persona_{pid}_questionnaire.csv"))
    return tr, q


def check_persona(pid):
    tr, q = _persona(pid)
    ref = analyze(tr, q)
    out = predict_from_features(trader_features(tr, q))

    assert ref["behaviour_score"] == out["behaviour_score"], \
        f"{pid}: score {out['behaviour_score']} != analyze {ref['behaviour_score']}"
    assert ref["behaviour_score"] == EXPECTED_SCORES[pid], \
        f"{pid}: score {ref['behaviour_score']} != expected {EXPECTED_SCORES[pid]}"

    ref_p = {m["model"]: m["probability"] for m in ref["models"]}
    out_p = {m["model"]: m["probability"] for m in out["models"]}
    assert ref_p == out_p, f"{pid}: probabilities differ: {ref_p} vs {out_p}"
    assert len(out["predictions"]) == 14, f"{pid}: {len(out['predictions'])} predictions, expected 14"

    # the rich per-model output must agree with the shared decision rule
    for p in out["predictions"]:
        assert p["probability"] == ref_p[p["model"]]
        assert p["uncertain"] == (abs(p["probability"] - p["threshold"]) < 0.1)
        assert p["polarity"] in ("negative", "positive")
        assert p["features_used"], f"{pid}/{p['model']}: no features recorded"

    return ref["behaviour_score"]


def test_impulsive():
    check_persona("impulsive")


def test_disciplined():
    check_persona("disciplined")


def test_fearful():
    check_persona("fearful")


def test_missing_questionnaire_is_handled():
    """A user who skipped the questionnaire: q_* features absent -> NaN, all 14 models still run."""
    tr, _ = _persona("impulsive")
    feats = {k: v for k, v in trader_features(tr, None).items() if not k.startswith("q_")}
    out = predict_from_features(feats)
    assert len(out["predictions"]) == 14
    assert 0 <= out["behaviour_score"] <= 100


if __name__ == "__main__":
    for pid in EXPECTED_SCORES:
        print(f"  {pid:12s} score {check_persona(pid):3d}  matches analyze() and the expected value")
    test_missing_questionnaire_is_handled()
    print("  no questionnaire: 14 predictions, score in range")
    print("\nAll checks passed.")
