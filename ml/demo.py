"""
Disciplyn - live demo for the mentor.   Run:  python demo.py
Press Enter to move from one section to the next.
"""
import json
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def title(t):
    input(f"\n[Enter] -> {t}")
    print("\n" + "=" * 78 + f"\n  {t}\n" + "=" * 78)


# ---------------------------------------------------------------- 1. dataset
title("1. DATASET")
labels = pd.read_csv("data/labels.csv")
feats = pd.read_csv("data/features.csv")
quest = pd.read_csv("data/questionnaire.csv")
if os.path.exists("data/trades.csv"):
    print("  loading 1.9M trades ...")
    t = pd.read_csv("data/trades.csv", usecols=["trader_id", "symbol", "trade_date"])
    print(f"  Raw trades          : {len(t):,} trades | {t.trader_id.nunique():,} traders | "
          f"{t.symbol.nunique()} NSE symbols | {t.trade_date.min()} -> {t.trade_date.max()}")
print(f"  Questionnaire       : {len(quest):,} answers from {quest.trader_id.nunique():,} traders")
print(f"  Training samples    : {len(feats):,} trader-month windows x 66 engineered features")
print(f"  Labels              : {labels.shape[1]-1} behaviour tags per trader")
print("\n  Tag distribution (share of traders):")
print(labels.drop(columns="trader_id").mean().round(2).sort_values(ascending=False).to_string())

# ---------------------------------------------------------------- 2. models
title("2. 14 MODELS FOR 20 TAGS  (held-out test set: 2,000 unseen traders)")
m = json.load(open("models/metrics.json"))
rows = [{"Model": k, "Tag(s)": v["tag(s)"], "Algorithm": v["algorithm"],
         "Accuracy": v["test"]["accuracy"], "Precision": v["test"]["precision"], "Recall": v["test"]["recall"],
         "F1": v["test"]["f1"], "ROC-AUC": v["test"]["roc_auc"]} for k, v in m.items()]
r = pd.DataFrame(rows)
print(r.drop(columns="Model").to_string(index=False))
print("\n  AVERAGE ->", r[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].mean().round(3).to_dict())

# ---------------------------------------------------------------- 3. fit check
title("3. OVERFITTING / UNDERFITTING CHECK")
fit = pd.DataFrame([{"Tag(s)": v["tag(s)"], "Train AUC": v.get("train_roc_auc"), "CV AUC": v["cv_roc_auc_tuned"],
                     "Test AUC": v["test"]["roc_auc"], "Gap": v.get("train_test_gap"), "Status": v.get("fit_status")}
                    for v in m.values()])
print(fit.to_string(index=False))
print(f"\n  Max train-test gap = {fit['Gap'].max():.3f}  (< 0.05 means no overfitting)")
print(f"  Min train AUC      = {fit['Train AUC'].min():.3f}  (> 0.75 means no underfitting)")

# ---------------------------------------------------------------- 4. live prediction
from predict import analyze  # noqa: E402

truth = labels.set_index("trader_id")
NAMES = {"impulsive": "IMPULSIVE", "disciplined": "DISCIPLINED", "fearful": "MIXED-BEHAVIOUR"}
for p in ["impulsive", "disciplined", "fearful"]:
    title(f"4. LIVE PREDICTION - {NAMES[p]} TRADER (never seen during training)")
    tr = pd.read_csv(f"samples/persona_{p}_trades.csv")
    q = pd.read_csv(f"samples/persona_{p}_questionnaire.csv")
    tid = tr.trader_id.iloc[0]
    res = analyze(tr, q)
    s = res["stats"]
    print(f"  Trader {tid} | {s['n_trades']} trades in last month | win rate {s['win_rate']:.0%} | "
          f"net P&L Rs {s['net_pnl_inr']:,.0f}")
    print(f"\n  BEHAVIOUR SCORE: {res['behaviour_score']}/100\n")
    present = sorted([x for x in res["tags"] if x["present"]], key=lambda x: -x["probability"])
    ok = 0
    for x in present:
        actual = int(truth.loc[tid, x["tag"]]) if tid in truth.index else None
        ok += actual == 1
        mark = "" if actual is None else ("  correct" if actual == 1 else "  WRONG")
        print(f"   {'[-]' if x['polarity'] == 'negative' else '[+]'} {x['tag']:30s} {x['probability']:.0%}{mark}")
    if tid in truth.index:
        print(f"\n  {ok}/{len(present)} detected tags match the ground truth")
    print(f"\n  Evidence: {s['trades_per_day']:.1f} trades/day | stop-loss moved on {s['sl_modified_rate']:.0%} | "
          f"chased {s['chase_entry_rate']:.0%} of entries | holds losers {s['loss_hold_vs_win_hold']:.1f}x longer than winners | "
          f"avg leverage {s['avg_leverage']:.1f}x")

print("\n" + "=" * 78 + "\n  Demo complete. Charts: ml/reports/   Full walkthrough: Disciplyn_Model_Training.ipynb\n" + "=" * 78)
