"""
Disciplyn API (FastAPI)

  cd backend && pip install -r requirements.txt && uvicorn main:app --port 8000
  Prediction demo: http://localhost:8000      Docs: http://localhost:8000/docs
"""
import io
import json
import math
import os
import sys

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
sys.path.insert(0, ML_DIR)
from features import ALL_FEATURES, MODELS, trader_features  # noqa: E402
from predict import analyze, display_name, load_models, predict_from_features  # noqa: E402

app = FastAPI(title="Disciplyn API", version="0.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
SAMPLES = os.path.join(ML_DIR, "samples")
PERSONAS = {"impulsive": "Impulsive trader", "fearful": "Fearful trader", "disciplined": "Disciplined trader"}
load_models()


def _stream_answer(*a, **kw):
    """Imported lazily: the RAG chat pulls in httpx and the knowledge base, and a problem
    there must never take down the prediction demo."""
    from rag import stream_answer
    return stream_answer(*a, **kw)


def _persona(pid):
    if pid not in PERSONAS:
        raise HTTPException(404, "unknown persona")
    tr = pd.read_csv(f"{SAMPLES}/persona_{pid}_trades.csv")
    q = pd.read_csv(f"{SAMPLES}/persona_{pid}_questionnaire.csv")
    return tr, q


# --------------------------------------------------------------------------
# Human-readable metadata for the 66 engineered features: label, one-line
# description, form group, unit. Used by /api/feature-schema (the demo form).
# --------------------------------------------------------------------------
G_FREQ = "Trading frequency"
G_ENTRY = "Entry timing (FOMO)"
G_SL = "Stop-loss & orders"
G_EXIT = "Exits & profit taking"
G_LEV = "Leverage & position sizing"
G_LOSS = "Behaviour after losses"
G_PANIC = "Panic & fear"
G_BIAS = "Confirmation bias"
G_STRAT = "Strategy consistency"
G_LEARN = "Learning & journaling"
G_Q = "Questionnaire (optional)"

FEATURE_META = {
    # ---- trading frequency
    "trades_per_day": ("Trades per day", "Average number of trades taken on an active day.", G_FREQ, "count"),
    "max_trades_per_day": ("Busiest day", "Highest number of trades in a single day.", G_FREQ, "count"),
    "median_gap_min": ("Median gap between trades", "Typical minutes between two entries on the same day.", G_FREQ, "minutes"),
    "active_days_ratio": ("Active-days ratio", "Share of available trading days on which the trader traded.", G_FREQ, "ratio"),
    "off_plan_rate": ("Off-plan trades", "Share of trades taken outside the trader's own written plan.", G_FREQ, "ratio"),
    "trades_after_daily_limit_rate": ("Trades after daily loss limit", "Share of trades taken after the 2%-of-capital daily loss limit was already breached.", G_FREQ, "ratio"),
    "low_quality_entry_rate": ("Low-quality entries", "Share of entries that were either chased or off-plan.", G_FREQ, "ratio"),
    # ---- entry timing / FOMO
    "chase_entry_rate": ("Chased entries", "Share of entries made after a sharp favourable move on unusually high volume.", G_ENTRY, "ratio"),
    "avg_pre_entry_move": ("Pre-entry move", "Average price move in the trade's direction in the 15 minutes before entry.", G_ENTRY, "%"),
    "avg_entry_volume_z": ("Entry volume z-score", "How unusual volume was at entry, in standard deviations.", G_ENTRY, "z-score"),
    "extreme_rsi_entry_rate": ("Extreme-RSI entries", "Share of entries made into overbought (long) or oversold (short) RSI.", G_ENTRY, "ratio"),
    "chase_win_rate_gap": ("Chase win-rate penalty", "Win rate on calm entries minus win rate on chased entries.", G_ENTRY, "ratio"),
    # ---- stop-loss & orders
    "sl_set_rate": ("Stop-loss set", "Share of trades that had a stop-loss order placed.", G_SL, "ratio"),
    "target_set_rate": ("Target set", "Share of trades that had a target order placed.", G_SL, "ratio"),
    "sl_modified_rate": ("Stop-loss modified", "Share of trades where the stop-loss was moved after entry.", G_SL, "ratio"),
    "sl_widened_rate": ("Stop-loss widened", "Share of trades where the stop-loss was moved further away from entry.", G_SL, "ratio"),
    "sl_honor_rate": ("Stop-loss honoured", "Of the trades that hit their stop, the share actually exited there.", G_SL, "ratio"),
    "avg_loss_r": ("Average loss", "Mean R-multiple of losing trades; -1 means losing exactly the planned risk.", G_SL, "R"),
    "worst_loss_r": ("Worst loss", "R-multiple of the single worst losing trade.", G_SL, "R"),
    "loss_beyond_1r_rate": ("Losses beyond 1R", "Share of losing trades that lost more than 1.2x the planned risk.", G_SL, "ratio"),
    # ---- exits & profit taking
    "avg_win_r": ("Average win", "Mean R-multiple of winning trades.", G_EXIT, "R"),
    "target_held_rate": ("Held past target", "Share of trades held on after the target was already hit.", G_EXIT, "ratio"),
    "giveback_ratio": ("Profit given back", "Share of the best unrealised profit handed back before exiting.", G_EXIT, "ratio"),
    "win_hold_vs_median": ("Winner holding time", "Average holding time of winners versus the median trade.", G_EXIT, "x"),
    "target_exit_rate": ("Exited at target", "Share of trades closed by the target order.", G_EXIT, "ratio"),
    "winners_turned_losers_rate": ("Winners turned losers", "Share of trades that were up over 1.5R and still finished near or below flat.", G_EXIT, "ratio"),
    # ---- leverage & sizing
    "avg_leverage": ("Average leverage", "Mean leverage used across trades.", G_LEV, "x"),
    "max_leverage": ("Peak leverage", "Highest leverage used on any single trade.", G_LEV, "x"),
    "p90_leverage": ("90th-percentile leverage", "Leverage the trader exceeds on one trade in ten.", G_LEV, "x"),
    "high_leverage_rate": ("High-leverage trades", "Share of trades taken above 2x leverage.", G_LEV, "ratio"),
    "avg_risk_pct_capital": ("Risk per trade", "Average capital at risk per trade, from leverage and stop distance.", G_LEV, "%"),
    "max_drawdown_pct": ("Max drawdown", "Largest peak-to-trough fall in account equity over the period.", G_LEV, "%"),
    "daily_pnl_volatility_pct": ("Daily P&L volatility", "Standard deviation of daily P&L as a share of capital.", G_LEV, "%"),
    "leverage_cv": ("Leverage variability", "Coefficient of variation of leverage; higher means less consistent sizing.", G_LEV, "ratio"),
    "risk_cv": ("Risk variability", "Coefficient of variation of per-trade risk.", G_LEV, "ratio"),
    "max_to_median_size": ("Largest vs typical size", "Biggest position divided by the typical position.", G_LEV, "x"),
    # ---- behaviour after losses
    "gap_after_loss_vs_win": ("Wait after loss vs win", "Time waited before re-entering after a loss, relative to after a win; below 1 means rushing back.", G_LOSS, "x"),
    "quick_reentry_after_loss_rate": ("Re-entry within 10 min of a loss", "Share of post-loss trades re-entered within 10 minutes.", G_LOSS, "ratio"),
    "size_after_loss_vs_win": ("Size-up after a loss", "Position size change after a loss relative to after a win; above 1 means sizing up.", G_LOSS, "x"),
    "loss_streak_trade_rate": ("Trading through losing streaks", "Share of trades taken while already two or more losses deep.", G_LOSS, "ratio"),
    "pnl_after_loss_r": ("Result after a loss", "Average R-multiple of the trade taken right after a loss.", G_LOSS, "R"),
    "loss_hold_vs_win_hold": ("Losers held vs winners", "Average holding time of losers divided by that of winners.", G_LOSS, "x"),
    "averaged_down_rate": ("Averaged down", "Share of trades where more was added to a losing position.", G_LOSS, "ratio"),
    "late_exit_loss_rate": ("Late exits on losers", "Share of trades exited manually, late, at a loss.", G_LOSS, "ratio"),
    # ---- panic & fear
    "early_exit_rate": ("Early manual exits", "Share of trades closed by hand before the stop or target was reached.", G_PANIC, "ratio"),
    "early_exit_in_selloff_rate": ("Early exits during adverse moves", "Share of trades bailed out of while the market moved against them.", G_PANIC, "ratio"),
    "post_exit_recovery_rate": ("Recovered after exit", "Share of losing trades where price moved favourably within 30 minutes of exiting.", G_PANIC, "ratio"),
    "avg_exit_volume_z": ("Exit volume z-score", "How unusual volume was when exiting losers, in standard deviations.", G_PANIC, "z-score"),
    "small_loss_short_hold_rate": ("Small, fast losses", "Share of trades cut quickly for a small loss, well before the stop.", G_PANIC, "ratio"),
    # ---- confirmation bias
    "same_idea_reentry_rate": ("Re-entered the same idea", "Re-entries into the same symbol and side right after a loss, per losing trade.", G_BIAS, "ratio"),
    "against_macd_rate": ("Traded against MACD", "Share of entries taken against the MACD histogram.", G_BIAS, "ratio"),
    "against_sentiment_rate": ("Traded against sentiment", "Share of entries taken against prevailing market sentiment.", G_BIAS, "ratio"),
    "side_concentration": ("One-sided bias", "Share of trades on the trader's preferred side (long or short).", G_BIAS, "ratio"),
    # ---- strategy consistency
    "strategy_count": ("Strategies used", "Number of distinct strategies traded in the period.", G_STRAT, "count"),
    "strategy_switch_rate": ("Strategy switching", "Share of trades that used a different strategy from the previous trade.", G_STRAT, "ratio"),
    "switch_after_loss_rate": ("Switched after a loss", "Strategy switches that followed a losing trade, per losing trade.", G_STRAT, "ratio"),
    "top_strategy_share": ("Main strategy share", "Share of trades belonging to the most-used strategy.", G_STRAT, "ratio"),
    # ---- learning & journaling
    "journal_loss_rate": ("Journalled losses", "Share of losing trades with a journal note written.", G_LEARN, "ratio"),
    "mistake_rate_trend": ("Mistake trend", "Mistake rate in the second half of the period minus the first half; negative is improvement.", G_LEARN, "ratio"),
    "second_half_vs_first_half_pnl": ("P&L trend", "Average R in the second half of the period minus the first half.", G_LEARN, "R"),
    "repeat_mistake_rate": ("Repeated mistakes", "Share of trades that repeated the previous trade's mistake.", G_LEARN, "ratio"),
    # ---- questionnaire (blank = the trader never answered it)
    "q_impulsivity": ("Impulsivity (self-reported)", "Questionnaire impulsivity score; higher means more impulsive. Leave blank if unanswered.", G_Q, "1-5 scale"),
    "q_confirmation_bias": ("Confirmation bias (self-reported)", "Questionnaire confirmation-bias score; higher means more biased. Leave blank if unanswered.", G_Q, "1-5 scale"),
    "q_emotional": ("Emotional reactivity (self-reported)", "Questionnaire emotional-reactivity score; higher means more reactive. Leave blank if unanswered.", G_Q, "1-5 scale"),
    "q_discipline": ("Indiscipline (self-reported)", "Questionnaire discipline score, reverse-scored: higher means less disciplined. Leave blank if unanswered.", G_Q, "1-5 scale"),
    "q_learning": ("Learning habit (self-reported)", "Questionnaire learning score; higher means more reflective. Leave blank if unanswered.", G_Q, "1-5 scale"),
}

GROUP_ORDER = [G_FREQ, G_ENTRY, G_SL, G_EXIT, G_LEV, G_LOSS, G_PANIC, G_BIAS, G_STRAT, G_LEARN, G_Q]
FORM_FEATURES = [f for f in ALL_FEATURES if not f.startswith("_")]
USED_BY = {f: [m for m, cfg in MODELS.items() if f in cfg["features"]] for f in FORM_FEATURES}


def _clean(v):
    """JSON-safe number (NaN / inf -> None)."""
    if v is None:
        return None
    v = float(v)
    return None if not math.isfinite(v) else round(v, 4)


def _step(lo, hi):
    span = abs(hi - lo) if lo is not None and hi is not None else 1.0
    if span <= 2:
        return 0.01
    if span <= 20:
        return 0.1
    return 1.0


_SCHEMA = None


def _feature_schema():
    """Built once: percentile ranges from the real training feature table + persona presets."""
    global _SCHEMA
    if _SCHEMA is not None:
        return _SCHEMA

    df = pd.read_csv(os.path.join(ML_DIR, "data", "features.csv"), usecols=FORM_FEATURES)
    lo, hi, med = df.quantile(0.01), df.quantile(0.99), df.median()

    features = []
    for f in FORM_FEATURES:
        label, desc, group, unit = FEATURE_META[f]
        a, b = _clean(lo[f]), _clean(hi[f])
        features.append({
            "name": f, "label": label, "description": desc, "group": group, "unit": unit,
            "min": a, "max": b, "median": _clean(med[f]), "step": _step(a, b),
            "used_by": USED_BY[f], "optional": f.startswith("q_"),
        })
    order = list(FEATURE_META)       # keep the readable order the metadata is written in
    features.sort(key=lambda x: (GROUP_ORDER.index(x["group"]), order.index(x["name"])))

    presets = {}
    for pid, key in [("impulsive", "impulsive"), ("disciplined", "disciplined"), ("fearful", "mixed")]:
        feats = trader_features(*_persona(pid))
        presets[key] = {f: _clean(feats.get(f)) for f in FORM_FEATURES}
    presets["median"] = {f: _clean(med[f]) for f in FORM_FEATURES}

    metrics = json.load(open(os.path.join(ML_DIR, "models", "metrics.json")))
    models = []
    for name, b in load_models().items():
        cfg = b["config"]
        m = metrics.get(name, {})
        models.append({
            "model": name, "display_name": display_name(name), "type": cfg["type"],
            "negative_tag": cfg.get("negative"), "positive_tag": cfg.get("positive"),
            "tag": cfg.get("tag"), "polarity": cfg.get("polarity"),
            "algorithm": b.get("algorithm", m.get("algorithm", "unknown")),
            "threshold": round(float(b.get("threshold", 0.5)), 3),
            "test_accuracy": m.get("test", {}).get("accuracy"),
            "test_roc_auc": m.get("test", {}).get("roc_auc"),
            "features": cfg["features"],
        })

    _SCHEMA = {
        "features": features,
        "groups": [g for g in GROUP_ORDER if any(x["group"] == g for x in features)],
        "presets": presets,
        "preset_labels": {"impulsive": "Impulsive trader", "disciplined": "Disciplined trader",
                          "mixed": "Mixed trader", "median": "Dataset median"},
        "models": models,
        "n_dataset_rows": int(len(df)),
    }
    return _SCHEMA


@app.get("/api/health")
def health():
    return {"ok": True, "models": 14}


@app.get("/api/personas")
def personas():
    return [{"id": k, "name": v} for k, v in PERSONAS.items()]


@app.get("/api/analyze/{persona_id}")
def analyze_persona(persona_id: str):
    tr, q = _persona(persona_id)
    out = analyze(tr, q)
    daily = tr.groupby("trade_date")["pnl"].sum().cumsum().round(0)
    out["equity_curve"] = [{"date": d, "cum_pnl": float(v)} for d, v in daily.items()]
    return out


@app.post("/api/analyze/upload")
async def analyze_upload(trades: UploadFile = File(...), questionnaire: UploadFile | None = File(None)):
    tr = pd.read_csv(io.BytesIO(await trades.read()))
    q = pd.read_csv(io.BytesIO(await questionnaire.read())) if questionnaire else None
    if len(tr) < 10:
        raise HTTPException(400, "need at least 10 trades")
    return analyze(tr, q)


@app.get("/api/metrics")
def metrics():
    return json.load(open(os.path.join(ML_DIR, "models", "metrics.json")))


@app.get("/api/feature-schema")
def feature_schema():
    return _feature_schema()


class FeatureReq(BaseModel):
    features: dict[str, float | None]


def _validated_row(features: dict) -> dict:
    """Reject unknown names and non-finite values; null/missing stay out of the row (-> NaN)."""
    unknown = sorted(set(features) - set(FORM_FEATURES))
    if unknown:
        raise HTTPException(400, f"unknown feature(s): {', '.join(unknown)}")
    row = {}
    for name, v in features.items():
        if v is None:
            continue
        v = float(v)
        if not math.isfinite(v):
            raise HTTPException(400, f"{name}: expected a finite number")
        row[name] = v
    return row


@app.post("/api/predict-features")
def predict_features(req: FeatureReq):
    known = set(FORM_FEATURES)
    row = _validated_row(req.features)

    out = predict_from_features(row)
    preds = out["predictions"]
    return {
        "behaviour_score": out["behaviour_score"],
        "predictions": preds,
        "summary": {
            "negative": sum(1 for p in preds if p["polarity"] == "negative" and not p["uncertain"]),
            "positive": sum(1 for p in preds if p["polarity"] == "positive" and not p["uncertain"]),
            "uncertain": sum(1 for p in preds if p["uncertain"]),
        },
        "n_features_supplied": len(row),
        "n_features_missing": len(known) - len(row),
    }


class ChatReq(BaseModel):
    persona_id: str | None = None
    analysis: dict | None = None
    features: dict[str, float | None] | None = None      # the demo page's current form values
    messages: list[dict]


# the trade stats the coach quotes; all of them are also form features
COACH_STATS = ["trades_per_day", "quick_reentry_after_loss_rate", "size_after_loss_vs_win", "sl_set_rate",
               "sl_modified_rate", "chase_entry_rate", "loss_hold_vs_win_hold", "avg_leverage", "max_drawdown_pct",
               "target_held_rate", "early_exit_rate", "off_plan_rate", "journal_loss_rate", "strategy_count"]


def _analysis_from_features(features: dict) -> dict:
    row = _validated_row(features)
    out = predict_from_features(row)
    return {"behaviour_score": out["behaviour_score"], "tags": out["tags"], "models": out["models"],
            "stats": {k: row.get(k, float("nan")) for k in COACH_STATS}}


@app.post("/api/chat")
async def chat(req: ChatReq):
    if req.features is not None:
        analysis = _analysis_from_features(req.features)
    else:
        analysis = req.analysis or analyze(*_persona(req.persona_id or "impulsive"))
    return StreamingResponse(_stream_answer(analysis, req.messages), media_type="text/plain; charset=utf-8")


# the demo UI: mounted last so every /api/* route wins
if os.path.isdir(STATIC):
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
