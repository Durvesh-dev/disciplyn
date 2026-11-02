# Disciplyn: ML + RAG backend (demo build)

```
disciplyn/
├── ml/
│   ├── generate_dataset.py   # synthetic dataset generator (ground-truth behaviour)
│   ├── features.py           # feature engineering + the 14 model definitions (own feature set per model)
│   ├── build_features.py     # trades -> trader-month feature windows
│   ├── training_pipeline.py  # algorithm comparison, CV, tuning, thresholds, evaluation
│   ├── train.py              # CLI: runs the whole pipeline
│   ├── Disciplyn_Model_Training.ipynb  # executed notebook with all outputs (show this to the mentor)
│   ├── reports/              # ROC curves, confusion matrices, feature importance, learning curves
│   ├── predict.py            # trade book (+questionnaire) -> 20 tags + Behaviour Score
│   ├── data/                 # trades.csv, questionnaire.csv, labels.csv, traders.csv, features.csv
│   ├── models/               # 14 .joblib models, metrics.json, model_metrics.png
│   └── samples/              # 3 demo traders (impulsive / fearful / disciplined)
├── backend/                  # FastAPI: /api/analyze, /api/chat (RAG), /api/metrics
└── frontend-dropin/app/dashboard/page.tsx   # Next.js page: score, tag chart, chat
```

## Run it (about 5 minutes)

```bash
# 1. backend (models are already trained)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000        # open http://localhost:8000/docs

# 2. frontend: copy frontend-dropin/app/dashboard/page.tsx into the Next.js app
#    .env.local ->  NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev   # open http://localhost:3000/dashboard
```

Optional real LLM (free tiers, takes about 2 minutes to get a key):
`export GROQ_API_KEY=...` (console.groq.com) **or** `export GEMINI_API_KEY=...` (aistudio.google.com).
Without a key, the chat uses a built-in rule-based coach that follows the same RAG pipeline, so the demo works offline.

Retrain from scratch: `cd ml && python generate_dataset.py && python build_features.py && python train.py` (about 15 min total).

## Run the prediction demo (feature -> prediction web page)

One command, no Node/npm, works offline:

```
.venv\Scripts\activate
cd backend
uvicorn main:app --port 8000
```

then open http://localhost:8000

The page shows all 66 model inputs as editable fields, grouped by behaviour area and prefilled
from the **Impulsive trader** preset. Press **Predict** (or click a preset, which predicts
automatically) to run all 14 models and get 14 prediction cards plus the Behaviour Score.
Presets: Impulsive trader (score 7), Disciplined trader (94), Mixed trader (80), Dataset median (67).
`?preset=disciplined` in the URL opens straight on that trader. Questionnaire fields may be left
blank: they are sent as null and become NaN, exactly like a user who never answered it.

Endpoints behind the page: `GET /api/feature-schema` (fields, percentile ranges, presets, model
metadata) and `POST /api/predict-features` (`{"features": {name: number | null}}`). Both run the
same `predict_from_features()` in `ml/predict.py` that `analyze()` uses, so the page and the
notebooks always agree. `python ml/test_predict_features.py` asserts that parity.

## The 14 models (20 tags)

Six mutually exclusive pairs each share **one** model (P(negative), positive = 1 − P):

| Model | Negative tag | Positive tag |
|---|---|---|
| stop_loss | Ignoring Stop-Losses | Stop-Loss Adherence |
| frequency | Overtrading | Controlled Trading Frequency |
| discipline | Lack of Discipline | Disciplined Trading |
| profit_exit | Greed | Profit Booking |
| entry_timing | FOMO | Patience |
| leverage | Overleveraging | Risk Management |

Eight standalone models: Revenge Trading, Panic Selling, Loss Aversion, Confirmation Bias,
Emotional Stability, Consistent Position Sizing, Learning from Losses, Strategic Consistency.

If a prediction falls between 0.4 and 0.6, the tag is reported as **uncertain** rather than forced.

## Dataset (large-scale)

No public dataset labels retail trades with behaviour tags, so we simulate **10,000 Indian retail traders × 60
trading days = 1,944,823 trades** (`generate_dataset.py`, reproducible with seed 42, about 3 min). Each trader has
hidden behavioural traits, correlated through latent *impulsivity* and *fear* factors. The traits drive every
decision, and the **labels come from the hidden traits**. The models only see broker, market and questionnaire
columns. One training sample = one **trader-month (20-trading-day window)** → **28,875 samples × 66 engineered features**
(`build_features.py`).

| File | Rows | Content |
|---|---|---|
| trades.csv | 1,944,823 | trade book + order book (SL set/moved/widened/honoured, averaging down) + price path (MFE/MAE) + Market API context (pre-entry move, volume z, RSI, MACD, sentiment, India VIX) + journal flag |
| questionnaire.csv | 69,620 | 6,962 traders (70%), 5 question sets, 10 Likert items, reverse-scored items |
| labels.csv | 10,000 | ground truth for all 20 tags |
| features.csv | 28,875 | engineered features per trader-month |

## Training methodology (`Disciplyn_Model_Training.ipynb`, `training_pipeline.py`)

1. Group split by trader: 8,000 train / 2,000 test traders (no overlap).
2. Algorithm comparison per tag: Logistic Regression, Random Forest, HistGradientBoosting and an MLP neural net, using 5-fold GroupKFold ROC-AUC (XGBoost/LightGBM are added automatically if installed).
3. RandomizedSearchCV tuning of the best algorithm per tag.
4. Decision threshold tuned for F1 on out-of-fold train predictions.
5. A single final evaluation on unseen test traders: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score.
6. Permutation feature importance, learning curves, and a cold-start (no questionnaire) analysis.

## Results: 2,000 unseen test traders (5,751 windows)

Average over the 14 models: **Accuracy 0.866 · Precision 0.823 · Recall 0.895 · F1 0.857 · ROC-AUC 0.943 · PR-AUC 0.932**.
Best: Overtrading (AUC 0.985), Discipline (0.984), Learning from Losses (0.974). Hardest: Emotional Stability (0.908),
Strategic Consistency (0.906), Panic Selling (0.912), which are the most internal behaviours.
**Learning curves flatten after about 2,000 traders**, so 10,000 traders is more than enough. The remaining error comes from
traders whose behaviour sits near the boundary, not from too little data. Charts are in `ml/reports/`.

## Pipeline (matches the architecture diagram)

OAuth → questionnaire (new users) → fetch trade/order book → **feature engineering (+ market data)**
→ 14 models → tags + Behaviour Score → **context aggregator** (tags + trade stats + market snapshot +
top-4 retrieved book principles) → LLM coach (streamed).

## Likely viva questions

- **"Is the data real?"** It's synthetic, with known ground truth, because no labelled public dataset exists. The same
  `features.py` code runs on real broker CSVs (e.g. a Zerodha tradebook mapped to these columns). The next step is
  fine-tuning on real users, with labels from questionnaires and expert review (weak supervision).
- **"Why only 14 models for 20 tags?"** Six tag pairs are mutually exclusive opposites, so one binary model covers each pair.
- **"Why different features per model?"** Each tag has its own evidence. For example, FOMO uses market context at entry,
  while stop-loss behaviour uses order-book modifications. See `MODELS` in `features.py`.
- **"How is a new user without history handled?"** The questionnaire features act as a prior. HistGradientBoosting handles missing values natively.
- **"Advice compliance?"** The coach gives behavioural coaching only, never buy/sell calls, and includes a disclaimer.
- **Book knowledge base:** in this demo it holds paraphrased principles with source attribution. In production, the
  team's own copies of the books are chunked into a vector DB, and TF-IDF is replaced with embeddings.
