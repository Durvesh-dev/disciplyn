import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbrun import run  # noqa: E402

ML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml"))

C = []
md = lambda s: C.append(("md", s))
code = lambda s: C.append(("code", s))

md("""# Disciplyn: Trader Behaviour Classification (Model Training)

**Goal:** detect 20 behavioural tags (10 negative, 10 positive) from a trader's trade book, order book, market context
and onboarding questionnaire, so the RAG + LLM coach can give tailored advice.

**20 tags → 14 binary models.** Six tag pairs are mutually exclusive opposites, so one model each:

| Model | Negative (1) | Positive (0) |
|---|---|---|
| stop_loss | Ignoring Stop-Losses | Stop-Loss Adherence |
| frequency | Overtrading | Controlled Trading Frequency |
| discipline | Lack of Discipline | Disciplined Trading |
| profit_exit | Greed | Profit Booking |
| entry_timing | FOMO | Patience |
| leverage | Overleveraging | Risk Management |

The other 8 have their own models: Revenge Trading, Panic Selling, Loss Aversion, Confirmation Bias, Emotional Stability,
Consistent Position Sizing, Learning from Losses, Strategic Consistency.

**Notebook outline:** 1. Dataset → 2. EDA → 3. Feature engineering → 4. Train/test split → 5. Algorithm comparison (5-fold CV)
→ 6. Hyper-parameter tuning + threshold tuning → 7. Test evaluation (ROC, confusion matrices) → 8. Explainability
→ 9. Effect of data size → 10. Cold start (no questionnaire) → 11. Inference on sample traders.""")

code("""import time, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.9)
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 200)
from features import MODELS
import training_pipeline as tp
print("models defined:", len(MODELS))""")

md("""## 1. Dataset

No public dataset labels retail trades with behaviour tags, so we **simulated 10,000 Indian retail traders over
60 trading days** (`generate_dataset.py`). Each simulated trader has hidden behavioural traits, and two latent
factors, *impulsivity* and *fear*, correlate the traits as they would in real people. The traits drive every
decision: trade frequency, re-entering after a loss, chasing moves, moving stop-losses, holding losers, leverage,
switching strategies, journaling.

The labels come from the hidden traits. The models only see what a **broker API (trade book + order book)**,
the **Market Data API** and the **questionnaire** would provide, so the test scores measure real detection ability.""")

code("""trades = pd.read_csv("data/trades.csv")
traders = pd.read_csv("data/traders.csv")
quest = pd.read_csv("data/questionnaire.csv")
labels = pd.read_csv("data/labels.csv")
print(f"trades        : {trades.shape[0]:>10,} rows x {trades.shape[1]} columns")
print(f"traders       : {traders.shape[0]:>10,}")
print(f"questionnaire : {quest.shape[0]:>10,} answers from {quest.trader_id.nunique():,} traders ({quest.trader_id.nunique()/len(traders):.0%})")
print(f"labels        : {labels.shape[0]:>10,} traders x {labels.shape[1]-1} tags")
print(f"symbols       : {trades.symbol.nunique()} NSE instruments, date range {trades.trade_date.min()} -> {trades.trade_date.max()}")
trades.head(5)""")

code("""groups = {
 "Trade book": ["trader_id","trade_id","trade_date","entry_time","exit_time","symbol","side","qty","entry_price","exit_price","pnl","capital_at_entry","leverage","strategy","in_trading_plan","exit_reason","holding_minutes","r_multiple"],
 "Order book": ["stop_loss_set","stop_loss_price","target_set","target_price","sl_modified_count","sl_widened","sl_hit","sl_honored","averaged_down"],
 "Price path": ["mfe_r","mae_r","target_hit_but_held"],
 "Market Data API": ["pre_entry_return_15m_pct","volume_zscore_entry","rsi_14_entry","macd_hist_entry","market_sentiment_entry","india_vix","market_move_during_hold_pct","volume_zscore_exit","post_exit_return_30m_pct"],
 "Journal": ["journal_note"]}
pd.DataFrame([{"source": k, "n_columns": len(v), "columns": ", ".join(v)} for k, v in groups.items()])""")

code("""quest[["set_no","question_no","construct","reverse_scored","question","answer"]].drop_duplicates("question").head(10)""")

md("## 2. Exploratory data analysis")

code("""fig, ax = plt.subplots(1, 3, figsize=(16, 4))
tpt = trades.groupby("trader_id").size()
ax[0].hist(tpt, bins=60, color="#2a6f97"); ax[0].set_title(f"Trades per trader (median {int(tpt.median())})")
trades.exit_reason.value_counts().plot.barh(ax=ax[1], color="#61a5c2"); ax[1].set_title("Exit reason")
ax[2].hist(trades.r_multiple.clip(-6, 6), bins=80, color="#e09f3e"); ax[2].set_title("Trade outcome (R-multiple)")
plt.tight_layout()""")

code("""rate = labels.drop(columns="trader_id").mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
neg = {m["negative"] for m in MODELS.values() if m["type"]=="pair"} | {m["tag"] for m in MODELS.values() if m.get("polarity")=="negative"}
rate.plot.barh(ax=ax, color=["#e11d48" if t in neg else "#10b981" for t in rate.index])
ax.set_title("Share of traders with each tag (red = negative, green = positive)"); ax.set_xlim(0, 1)
for i, v in enumerate(rate): ax.text(v + 0.01, i, f"{v:.0%}", va="center")""")

code("""neg_cols = [t for t in labels.columns if t in neg]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(labels[neg_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar=False)
ax.set_title("Correlation between negative behaviours (traders who revenge-trade also tend to overtrade, FOMO...)")""")

md("""Each behaviour leaves a measurable footprint in the trades. A few examples, using raw trades joined with their trader's label:""")

code("""t = trades.merge(labels[["trader_id","FOMO","Loss Aversion","Ignoring Stop-Losses","Overleveraging"]], on="trader_id")
t = t.sample(200_000, random_state=0)
sgn = np.where(t.side=="LONG", 1, -1)
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
sns.kdeplot(data=t.assign(move=t.pre_entry_return_15m_pct*sgn), x="move", hue="FOMO", common_norm=False, ax=ax[0], fill=True)
ax[0].set_title("FOMO: price move in trade direction 15 min BEFORE entry")
sns.boxplot(data=t[t.pnl<0], x="Loss Aversion", y="holding_minutes", ax=ax[1], showfliers=False)
ax[1].set_title("Loss Aversion: holding time of LOSING trades")
t.groupby("Ignoring Stop-Losses").sl_widened.mean().plot.bar(ax=ax[2], color=["#10b981","#e11d48"])
ax[2].set_title("Ignoring SL: share of trades with stop widened")
sns.kdeplot(data=t, x="leverage", hue="Overleveraging", common_norm=False, ax=ax[3], fill=True, clip=(0, 12))
ax[3].set_title("Overleveraging: leverage per trade"); plt.tight_layout()""")

code("""# Revenge trading: what happens right after a losing trade (same day)
s = trades.sort_values(["trader_id","entry_time"]).copy()
s["prev_loss"] = (s.groupby("trader_id").pnl.shift() < 0) & (s.trade_date == s.groupby("trader_id").trade_date.shift())
s["gap"] = (pd.to_datetime(s.entry_time) - pd.to_datetime(s.groupby("trader_id").exit_time.shift())).dt.total_seconds()/60
s["size_change"] = s.leverage / s.groupby("trader_id").leverage.shift()
s = s[s.prev_loss].merge(labels[["trader_id","Revenge Trading"]], on="trader_id")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.kdeplot(data=s, x=s.gap.clip(0, 150), hue="Revenge Trading", common_norm=False, fill=True, ax=ax[0])
ax[0].set_title("Minutes until next trade after a LOSS")
sns.boxplot(data=s, x="Revenge Trading", y=s.size_change.clip(0, 5), showfliers=False, ax=ax[1])
ax[1].set_title("Position size vs previous trade, after a LOSS"); plt.tight_layout()""")

md("""## 3. Feature engineering

`features.py` turns raw trades into **behavioural features**. One training sample is **one trader-month
(a 20-trading-day window)**, the same unit used in production, where every user is re-scored monthly to track improvement.
Each of the 14 models uses **its own feature subset**, chosen from the evidence for that behaviour.
Missing questionnaire answers (existing users) stay as NaN.""")

code("""df = tp.load_dataset()
feat_cols = sorted({f for m in MODELS.values() for f in m["features"]})
print(f"training samples (trader-month windows): {len(df):,} from {df.trader_id.nunique():,} traders")
print(f"engineered features: {len(feat_cols)}  | windows with questionnaire: {df.q_impulsivity.notna().mean():.0%}")
pd.DataFrame([{"model": k, "target": tp.target_of(v), "n_features": len(v["features"]), "features": ", ".join(v["features"])}
              for k, v in MODELS.items()])""")

code("""df[feat_cols].describe().T[["mean","std","min","50%","max"]].round(3).head(25)""")

md("""## 4. Train / test split (by trader)

All windows from one trader go to the same side of the split (`GroupShuffleSplit`), so the test set contains
**traders the models have never seen**. Cross-validation uses `GroupKFold` for the same reason.""")

code("""train, test = tp.split(df, test_size=0.2)
print(f"train: {len(train):,} windows / {train.trader_id.nunique():,} traders")
print(f"test : {len(test):,} windows / {test.trader_id.nunique():,} traders")
print("trader overlap:", len(set(train.trader_id) & set(test.trader_id)))""")

md("""## 5. Algorithm comparison (5-fold GroupKFold, ROC-AUC)

Each of the 14 tasks is tested with 4 algorithm families. The best one per tag goes forward to tuning.
XGBoost and LightGBM are added automatically if they are installed.""")

code("""t0 = time.time()
comparison, algo_cols = tp.compare_algorithms(train, folds=5)
print(f"\\ncompleted in {(time.time()-t0)/60:.1f} min")
comparison[["model","tag(s)"] + algo_cols + ["best_algorithm"]]""")

code("""fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(comparison.set_index("tag(s)")[algo_cols], annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "CV ROC-AUC"})
ax.set_title("Cross-validated ROC-AUC per tag and algorithm"); ax.set_ylabel("")
print(comparison.best_algorithm.value_counts().to_string())""")

md("""## 6–7. Hyper-parameter tuning, threshold tuning and held-out test evaluation

For the best algorithm per tag:
- `RandomizedSearchCV` (GroupKFold) tunes the hyper-parameters.
- The decision threshold is picked to maximise F1 on **out-of-fold train predictions**, never on the test set.
- The final model is evaluated once on the unseen test traders.""")

code("""t0 = time.time()
results, art = tp.tune_and_evaluate(train, test, comparison, n_iter=10, folds=3)
print(f"\\ncompleted in {(time.time()-t0)/60:.1f} min")
res = tp.results_table(results)
res""")

code("""avg = res[["accuracy","precision","recall","f1","roc_auc","pr_auc"]].mean()
print("AVERAGE over 14 models:"); print(avg.round(4).to_string())
fig, ax = plt.subplots(figsize=(11, 7))
r = res.set_index("tag(s)")[["accuracy","f1","roc_auc"]]
r.plot.barh(ax=ax, color=["#2a6f97","#61a5c2","#e09f3e"], width=0.8); ax.invert_yaxis()
ax.set_xlim(0.5, 1); ax.set_title("Held-out test performance (unseen traders)"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("reports/test_metrics.png", dpi=150)""")

code("""fig, axes = plt.subplots(3, 5, figsize=(20, 12)); axes = axes.ravel()
for ax, (name, a) in zip(axes, art.items()):
    ax.plot(a["fpr"], a["tpr"], color="#2a6f97", lw=2); ax.plot([0,1],[0,1],"--", color="grey", lw=1)
    ax.set_title(f"{results[name]['tag(s)'][:34]}\\nAUC={results[name]['test']['roc_auc']:.3f}", fontsize=9)
    ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
axes[-1].axis("off"); plt.suptitle("ROC curves on test set", y=1.0); plt.tight_layout(); plt.savefig("reports/roc_curves.png", dpi=130)""")

code("""fig, axes = plt.subplots(3, 5, figsize=(20, 11)); axes = axes.ravel()
for ax, (name, v) in zip(axes, results.items()):
    cm = np.array(v["confusion_matrix"])
    lab = ["0", "1"] if v["type"]=="single" else [v["opposite"][:16], v["predicts"][:16]]
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax, xticklabels=lab, yticklabels=lab)
    ax.set_title(v["tag(s)"][:38], fontsize=9); ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("actual", fontsize=8)
    ax.tick_params(labelsize=7)
axes[-1].axis("off"); plt.tight_layout(); plt.savefig("reports/confusion_matrices.png", dpi=130)""")

md("""## 8. Explainability: permutation feature importance

Each bar shows the drop in test ROC-AUC when that feature is shuffled. This is how the coach can say
*why* a tag was assigned (e.g. "you re-entered within 10 minutes after 40% of your losses").""")

code("""fig, axes = plt.subplots(3, 5, figsize=(22, 12)); axes = axes.ravel()
for ax, (name, v) in zip(axes, results.items()):
    fi = pd.DataFrame(v["feature_importance"], columns=["feature","importance"]).head(6)[::-1]
    ax.barh(fi.feature, fi.importance, color="#2a6f97"); ax.set_title(v["tag(s)"][:38], fontsize=9); ax.tick_params(labelsize=7)
axes[-1].axis("off"); plt.tight_layout(); plt.savefig("reports/feature_importance.png", dpi=130)""")

md("""## 9. Does more data help? Learning curves

To check that the dataset is big enough, each model is retrained on growing subsets of training traders and
scored on the same test set. The point where the curve flattens shows that adding more data would not raise accuracy much.""")

code("""import joblib
from sklearn.base import clone
sizes = [250, 500, 1000, 2000, 4000, train.trader_id.nunique()]
ids = train.trader_id.drop_duplicates().sample(frac=1, random_state=0).values
curve = []
for name in ["revenge", "panic", "emotional_stability", "entry_timing", "loss_aversion", "leverage"]:
    b = joblib.load(f"models/{name}.joblib"); f, y = b["features"], tp.target_of(b["config"])
    for n in sizes:
        sub = train[train.trader_id.isin(ids[:n])]
        m = clone(b["model"]).fit(sub[f], sub[y])
        from sklearn.metrics import roc_auc_score
        curve.append({"model": results[name]["tag(s)"], "traders": n, "windows": len(sub),
                      "test_roc_auc": roc_auc_score(test[y], m.predict_proba(test[f])[:, 1])})
curve = pd.DataFrame(curve)
fig, ax = plt.subplots(figsize=(10, 5))
for k, g in curve.groupby("model"): ax.plot(g.traders, g.test_roc_auc, marker="o", label=k)
ax.set_xscale("log"); ax.set_xlabel("training traders (log scale)"); ax.set_ylabel("test ROC-AUC")
ax.set_title("Learning curves: performance vs training data size"); ax.legend(fontsize=8)
plt.savefig("reports/learning_curves.png", dpi=150, bbox_inches="tight")
curve.pivot(index="traders", columns="model", values="test_roc_auc").round(3)""")

md("""## 10. Cold start: new users with only a questionnaire vs existing users with no questionnaire

The models are evaluated separately on test windows **with** and **without** questionnaire answers.
Tags like *Confirmation Bias* and *Emotional Stability* benefit most from self-report answers.""")

code("""rows = []
for name, v in results.items():
    b = joblib.load(f"models/{name}.joblib"); f, y = b["features"], tp.target_of(b["config"])
    has_q = test.q_impulsivity.notna()
    for lbl, mask in [("with questionnaire", has_q), ("no questionnaire", ~has_q)]:
        rows.append({"tag(s)": v["tag(s)"], "subset": lbl, "uses_q_feature": any(c.startswith("q_") for c in f),
                     "roc_auc": roc_auc_score(test.loc[mask, y], b["model"].predict_proba(test.loc[mask, f])[:, 1])})
cold = pd.DataFrame(rows).pivot_table(index=["tag(s)","uses_q_feature"], columns="subset", values="roc_auc").round(3)
cold""")

md("## 11. Inference: full 20-tag report for unseen test traders")

code("""from predict import analyze, _MODELS
import predict; predict._MODELS = None   # reload freshly tuned models
hidden = pd.read_csv("data/hidden_traits.csv")
q_ids = set(quest.trader_id); test_ids = [t for t in test.trader_id.unique() if t in q_ids]
h = hidden[hidden.trader_id.isin(test_ids)].copy()
h["badness"] = h[["revenge","frequency","entry_timing","discipline","leverage","stop_loss","panic","loss_aversion","profit_exit"]].mean(1)
personas = {"impulsive": h.nlargest(5, "badness").trader_id.iloc[-1], "disciplined": h.nsmallest(5, "badness").trader_id.iloc[-1],
            "fearful": h.nlargest(3, "panic").trader_id.iloc[-1]}
import os; os.makedirs("samples", exist_ok=True)
for k, tid in personas.items():
    tr = trades[trades.trader_id == tid]; qq = quest[quest.trader_id == tid]
    last_month = tr[tr.trade_date >= sorted(tr.trade_date.unique())[-20]]
    last_month.to_csv(f"samples/persona_{k}_trades.csv", index=False); qq.to_csv(f"samples/persona_{k}_questionnaire.csv", index=False)
    r = analyze(last_month, qq)
    truth = labels.set_index("trader_id").loc[tid]
    present = [t for t in r["tags"] if t["present"]]
    correct = sum(truth[t["tag"]] == 1 for t in present)
    print(f"\\n=== {k.upper()} trader {tid}: behaviour score {r['behaviour_score']}/100 | {len(last_month)} trades | "
          f"{correct}/{len(present)} detected tags match ground truth")
    for t in sorted(present, key=lambda x: -x["probability"]):
        print(f"   {'NEG' if t['polarity']=='negative' else 'POS'}  {t['tag']:30s} p={t['probability']:.2f}  truth={int(truth[t['tag']])}")""")

md("""## Conclusion

- **Dataset:** about 1.9 million simulated trades from 10,000 traders, with trade book, order book, market context and questionnaire columns, cut into 20-trading-day training windows.
- **Models:** 14 binary models cover all 20 tags. For each tag the best of 4 algorithm families was picked by grouped cross-validation, then tuned, with a decision threshold chosen on out-of-fold predictions.
- **Results:** see the test table in section 7. The hardest tags are the most internal ones (Panic Selling, Emotional Stability), which matches how hard they are to spot from trades alone.
- **Next steps with real users:** keep the same features and pipeline, and fine-tune on labels from real users via questionnaire answers, expert review of trade journals, and coach-chat feedback.""")

if __name__ == "__main__":
    os.makedirs(os.path.join(ML, "reports"), exist_ok=True)
    run(C, os.path.join(ML, "Disciplyn_Model_Training.ipynb"), ML)
