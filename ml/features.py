"""
Feature engineering: trades (+ market context) + questionnaire  ->  one feature row per trader.

The SAME function is used for training and for live inference on a user's
uploaded/broker-fetched trade book, so there is no train/serve skew.
"""
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 14 models. Each model gets its OWN feature subset (as required by the spec).
# Merged models predict the NEGATIVE tag; positive tag = 1 - p.
# --------------------------------------------------------------------------
MODELS = {
    "stop_loss": {
        "type": "pair", "negative": "Ignoring Stop-Losses", "positive": "Stop-Loss Adherence",
        "features": ["sl_set_rate", "sl_modified_rate", "sl_widened_rate", "sl_honor_rate",
                     "avg_loss_r", "loss_beyond_1r_rate", "worst_loss_r"]},
    "frequency": {
        "type": "pair", "negative": "Overtrading", "positive": "Controlled Trading Frequency",
        "features": ["trades_per_day", "max_trades_per_day", "median_gap_min", "off_plan_rate",
                     "trades_after_daily_limit_rate", "low_quality_entry_rate", "active_days_ratio"]},
    "discipline": {
        "type": "pair", "negative": "Lack of Discipline", "positive": "Disciplined Trading",
        "features": ["sl_set_rate", "target_set_rate", "off_plan_rate", "trades_after_daily_limit_rate",
                     "sl_modified_rate", "q_discipline", "strategy_count"]},
    "profit_exit": {
        "type": "pair", "negative": "Greed", "positive": "Profit Booking",
        "features": ["target_held_rate", "giveback_ratio", "win_hold_vs_median", "target_exit_rate",
                     "avg_win_r", "winners_turned_losers_rate"]},
    "entry_timing": {
        "type": "pair", "negative": "FOMO", "positive": "Patience",
        "features": ["chase_entry_rate", "avg_pre_entry_move", "avg_entry_volume_z",
                     "extreme_rsi_entry_rate", "q_impulsivity", "chase_win_rate_gap"]},
    "leverage": {
        "type": "pair", "negative": "Overleveraging", "positive": "Risk Management",
        "features": ["avg_leverage", "max_leverage", "p90_leverage", "avg_risk_pct_capital",
                     "max_drawdown_pct", "high_leverage_rate"]},
    "revenge": {
        "type": "single", "tag": "Revenge Trading", "polarity": "negative",
        "features": ["gap_after_loss_vs_win", "quick_reentry_after_loss_rate", "size_after_loss_vs_win",
                     "loss_streak_trade_rate", "q_impulsivity", "pnl_after_loss_r"]},
    "panic": {
        "type": "single", "tag": "Panic Selling", "polarity": "negative",
        "features": ["early_exit_rate", "early_exit_in_selloff_rate", "post_exit_recovery_rate",
                     "avg_exit_volume_z", "small_loss_short_hold_rate", "q_emotional"]},
    "loss_aversion": {
        "type": "single", "tag": "Loss Aversion", "polarity": "negative",
        "features": ["loss_hold_vs_win_hold", "averaged_down_rate", "avg_loss_r", "loss_beyond_1r_rate",
                     "late_exit_loss_rate", "avg_win_r"]},
    "confirmation_bias": {
        "type": "single", "tag": "Confirmation Bias", "polarity": "negative",
        "features": ["same_idea_reentry_rate", "against_macd_rate", "against_sentiment_rate",
                     "q_confirmation_bias", "side_concentration"]},
    "emotional_stability": {
        "type": "single", "tag": "Emotional Stability", "polarity": "positive",
        "features": ["gap_after_loss_vs_win", "size_after_loss_vs_win", "early_exit_rate",
                     "chase_entry_rate", "daily_pnl_volatility_pct", "q_emotional", "q_impulsivity"]},
    "consistent_sizing": {
        "type": "single", "tag": "Consistent Position Sizing", "polarity": "positive",
        "features": ["leverage_cv", "risk_cv", "size_after_loss_vs_win", "max_to_median_size"]},
    "learning": {
        "type": "single", "tag": "Learning from Losses", "polarity": "positive",
        "features": ["journal_loss_rate", "mistake_rate_trend", "second_half_vs_first_half_pnl",
                     "q_learning", "repeat_mistake_rate"]},
    "strategic_consistency": {
        "type": "single", "tag": "Strategic Consistency", "polarity": "positive",
        "features": ["strategy_count", "strategy_switch_rate", "switch_after_loss_rate",
                     "top_strategy_share", "off_plan_rate"]},
}

ALL_FEATURES = sorted({f for m in MODELS.values() for f in m["features"]})


def _safe_ratio(a, b, default=1.0):
    return float(a / b) if b and np.isfinite(b) and b != 0 else default


def _questionnaire_scores(q: pd.DataFrame) -> dict:
    """Average Likert per construct, reverse-scored items flipped. Higher = more of the construct
    (impulsivity, bias, emotional reactivity, discipline, learning)."""
    if q is None or len(q) == 0:
        return {}
    q = q.copy()
    q["score"] = np.where(q["reverse_scored"] == 1, 6 - q["answer"], q["answer"])
    s = q.groupby("construct")["score"].mean()
    return {"q_impulsivity": s.get("impulsivity", np.nan),
            "q_confirmation_bias": s.get("confirmation_bias", np.nan),
            "q_emotional": s.get("emotional", np.nan),
            # discipline items ask about GOOD discipline -> flip so higher = less disciplined
            "q_discipline": 6 - s["discipline"] if "discipline" in s else np.nan,
            "q_learning": s.get("learning", np.nan)}


def trader_features(tr: pd.DataFrame, q: pd.DataFrame | None = None) -> dict:
    tr = tr.sort_values("entry_time").reset_index(drop=True)
    tr["entry_dt"] = pd.to_datetime(tr["entry_time"])
    tr["exit_dt"] = pd.to_datetime(tr["exit_time"])
    n = len(tr)
    f = {}
    loss = tr["pnl"] < 0
    win = ~loss
    sgn = np.where(tr["side"] == "LONG", 1, -1)
    cap = tr["capital_at_entry"].iloc[0]

    # ---- daily / frequency
    per_day = tr.groupby("trade_date").size()
    f["trades_per_day"] = per_day.mean()
    f["max_trades_per_day"] = per_day.max()
    f["active_days_ratio"] = len(per_day) / max(1, (tr["entry_dt"].max() - tr["entry_dt"].min()).days * 5 / 7 + 1)
    tr["gap_min"] = tr.groupby("trade_date")["entry_dt"].diff().dt.total_seconds() / 60
    tr["prev_exit_gap"] = (tr["entry_dt"] - tr["exit_dt"].shift()).dt.total_seconds() / 60
    same_day = tr["trade_date"] == tr["trade_date"].shift()
    f["median_gap_min"] = tr["gap_min"].median() if tr["gap_min"].notna().any() else 240
    f["off_plan_rate"] = 1 - tr["in_trading_plan"].mean()
    # trades taken after daily loss limit (2% of capital) was already breached
    tr["cum_day_pnl_before"] = tr.groupby("trade_date")["pnl"].cumsum() - tr["pnl"]
    f["trades_after_daily_limit_rate"] = (tr["cum_day_pnl_before"] < -0.02 * tr["capital_at_entry"]).mean()

    # ---- entry quality / FOMO (market context)
    chase = (tr["pre_entry_return_15m_pct"] * sgn > 1.0) & (tr["volume_zscore_entry"] > 1.8)
    f["chase_entry_rate"] = chase.mean()
    f["avg_pre_entry_move"] = (tr["pre_entry_return_15m_pct"] * sgn).mean()
    f["avg_entry_volume_z"] = tr["volume_zscore_entry"].mean()
    f["extreme_rsi_entry_rate"] = (((sgn > 0) & (tr["rsi_14_entry"] > 70)) | ((sgn < 0) & (tr["rsi_14_entry"] < 30))).mean()
    f["chase_win_rate_gap"] = (win[~chase].mean() - win[chase].mean()) if chase.sum() >= 3 and (~chase).sum() >= 3 else 0.0
    f["low_quality_entry_rate"] = (chase | (tr["in_trading_plan"] == 0)).mean()

    # ---- stop-loss (order book)
    f["sl_set_rate"] = tr["stop_loss_set"].mean()
    f["target_set_rate"] = tr["target_set"].mean()
    f["sl_modified_rate"] = (tr["sl_modified_count"] > 0).mean()
    f["sl_widened_rate"] = tr["sl_widened"].mean()
    hit = tr["sl_hit"] == 1
    f["sl_honor_rate"] = tr.loc[hit, "sl_honored"].mean() if hit.any() else 0.5
    lr = tr.loc[loss, "r_multiple"]
    f["avg_loss_r"] = lr.mean() if len(lr) else 0
    f["worst_loss_r"] = lr.min() if len(lr) else 0
    f["loss_beyond_1r_rate"] = (lr < -1.2).mean() if len(lr) else 0

    # ---- profit taking / greed
    wr = tr.loc[win, "r_multiple"]
    f["avg_win_r"] = wr.mean() if len(wr) else 0
    f["target_held_rate"] = tr["target_hit_but_held"].mean()
    f["giveback_ratio"] = (1 - tr["r_multiple"].clip(lower=0) / tr["mfe_r"].clip(lower=0.1))[tr["mfe_r"] > 1].mean()
    med_hold = tr["holding_minutes"].median()
    f["win_hold_vs_median"] = _safe_ratio(tr.loc[win, "holding_minutes"].mean(), med_hold)
    f["target_exit_rate"] = (tr["exit_reason"] == "target").mean()
    f["winners_turned_losers_rate"] = ((tr["mfe_r"] > 1.5) & (tr["r_multiple"] < 0.3)).mean()

    # ---- leverage / risk
    f["avg_leverage"] = tr["leverage"].mean()
    f["max_leverage"] = tr["leverage"].max()
    f["p90_leverage"] = tr["leverage"].quantile(0.9)
    f["high_leverage_rate"] = (tr["leverage"] > 2).mean()
    sl_dist = (tr["entry_price"] - tr["stop_loss_price"]).abs() / tr["entry_price"]
    risk_pct = (tr["leverage"] * sl_dist.fillna(0.01)) * 100
    f["avg_risk_pct_capital"] = risk_pct.mean()
    eq = cap + tr["pnl"].cumsum()
    f["max_drawdown_pct"] = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    daily = tr.groupby("trade_date")["pnl"].sum() / cap * 100
    f["daily_pnl_volatility_pct"] = daily.std() if len(daily) > 1 else 0

    # ---- revenge / emotional (behaviour conditional on previous result)
    prev_loss = loss.shift().fillna(False).astype(bool) & same_day
    prev_win = win.shift().fillna(False).astype(bool) & same_day
    g_l = tr.loc[prev_loss, "prev_exit_gap"].median()
    g_w = tr.loc[prev_win, "prev_exit_gap"].median()
    f["gap_after_loss_vs_win"] = _safe_ratio(g_l, g_w) if pd.notna(g_l) and pd.notna(g_w) else 1.0
    f["quick_reentry_after_loss_rate"] = (tr.loc[prev_loss, "prev_exit_gap"] < 10).mean() if prev_loss.any() else 0
    lev_prev = tr["leverage"].shift()
    s_l = (tr["leverage"] / lev_prev)[prev_loss].median()
    s_w = (tr["leverage"] / lev_prev)[prev_win].median()
    f["size_after_loss_vs_win"] = _safe_ratio(s_l, s_w) if pd.notna(s_l) and pd.notna(s_w) else 1.0
    streak = loss.astype(int).groupby((~loss).cumsum()).cumsum().shift().fillna(0)
    f["loss_streak_trade_rate"] = (streak >= 2).mean()
    f["pnl_after_loss_r"] = tr.loc[prev_loss, "r_multiple"].mean() if prev_loss.any() else 0

    # ---- panic
    early = tr["exit_reason"] == "manual_early"
    f["early_exit_rate"] = early.mean()
    selloff = (tr["market_move_during_hold_pct"] * sgn < -0.8)
    f["early_exit_in_selloff_rate"] = (early & selloff).mean()
    f["post_exit_recovery_rate"] = ((tr["post_exit_return_30m_pct"] * sgn > 0.4) & loss).mean()
    f["avg_exit_volume_z"] = tr.loc[loss, "volume_zscore_exit"].mean() if loss.any() else 0
    f["small_loss_short_hold_rate"] = (loss & (tr["r_multiple"] > -0.6) & (tr["holding_minutes"] < 0.5 * med_hold)).mean()

    # ---- loss aversion
    f["loss_hold_vs_win_hold"] = _safe_ratio(tr.loc[loss, "holding_minutes"].mean(), tr.loc[win, "holding_minutes"].mean())
    f["averaged_down_rate"] = tr["averaged_down"].mean()
    f["late_exit_loss_rate"] = ((tr["exit_reason"] == "manual_late") & loss).mean()

    # ---- confirmation bias
    same_idea = (tr["symbol"] == tr["symbol"].shift()) & (tr["side"] == tr["side"].shift()) & loss.shift().fillna(False).astype(bool)
    f["same_idea_reentry_rate"] = same_idea.sum() / max(1, loss.sum())
    f["against_macd_rate"] = (np.sign(tr["macd_hist_entry"]) == -sgn).mean()
    f["against_sentiment_rate"] = (np.sign(tr["market_sentiment_entry"]) == -sgn).mean()
    f["side_concentration"] = max((sgn > 0).mean(), (sgn < 0).mean())

    # ---- sizing consistency
    f["leverage_cv"] = tr["leverage"].std() / max(tr["leverage"].mean(), 1e-6)
    f["risk_cv"] = risk_pct.std() / max(risk_pct.mean(), 1e-6)
    f["max_to_median_size"] = _safe_ratio(tr["leverage"].max(), tr["leverage"].median())

    # ---- learning
    f["journal_loss_rate"] = tr.loc[loss, "journal_note"].mean() if loss.any() else 0
    mistake = chase | prev_loss & (tr["prev_exit_gap"] < 10) | (tr["sl_widened"] == 1)
    half = n // 2
    f["mistake_rate_trend"] = mistake.iloc[half:].mean() - mistake.iloc[:half].mean() if n >= 10 else 0
    f["second_half_vs_first_half_pnl"] = (tr["r_multiple"].iloc[half:].mean() - tr["r_multiple"].iloc[:half].mean()) if n >= 10 else 0
    f["repeat_mistake_rate"] = (mistake & mistake.shift().fillna(False).astype(bool)).mean()

    # ---- strategy
    sc = tr["strategy"].value_counts(normalize=True)
    f["strategy_count"] = len(sc)
    f["top_strategy_share"] = sc.iloc[0]
    sw = tr["strategy"] != tr["strategy"].shift()
    f["strategy_switch_rate"] = sw.iloc[1:].mean() if n > 1 else 0
    f["switch_after_loss_rate"] = (sw & loss.shift().fillna(False).astype(bool)).sum() / max(1, loss.sum())

    # ---- questionnaire (NaN for users without answers; HistGradientBoosting handles NaN natively)
    qs = _questionnaire_scores(q)
    for k in ["q_impulsivity", "q_confirmation_bias", "q_emotional", "q_discipline", "q_learning"]:
        f[k] = qs.get(k, np.nan)

    # summary stats (not model inputs, used by dashboard/LLM context)
    f["_n_trades"] = n
    f["_win_rate"] = win.mean()
    f["_net_pnl"] = tr["pnl"].sum()
    return {k: (float(v) if v is not None and pd.notna(v) else np.nan) for k, v in f.items()}


WINDOW_DAYS = 20   # one training sample = one trader's 20-trading-day window (~1 month)
MIN_TRADES = 10


def _window_job(args):
    tid, tr, q = args
    out = []
    days = np.sort(tr["trade_date"].unique())
    win_of_day = {d: i // WINDOW_DAYS for i, d in enumerate(days)}
    tr = tr.assign(_w=tr["trade_date"].map(win_of_day))
    for w, g in tr.groupby("_w"):
        if len(g) < MIN_TRADES:
            continue
        out.append({"sample_id": f"{tid}_W{w}", "trader_id": tid, "window": int(w),
                    "window_start": g["trade_date"].min(), **trader_features(g.drop(columns="_w"), q)})
    return out


def build_feature_table(trades: pd.DataFrame, questionnaire: pd.DataFrame, workers: int = 2) -> pd.DataFrame:
    """Sliding monthly windows per trader -> many more training rows, and it mirrors production
    (we re-score each user every month to track improvement)."""
    from multiprocessing import Pool
    qg = dict(tuple(questionnaire.groupby("trader_id"))) if len(questionnaire) else {}
    jobs = ((tid, tr, qg.get(tid)) for tid, tr in trades.groupby("trader_id", sort=False))
    rows = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_window_job, jobs, chunksize=50)):
            rows.extend(r)
            if (i + 1) % 1000 == 0:
                print(f"  features: {i + 1:,} traders, {len(rows):,} windows", flush=True)
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
