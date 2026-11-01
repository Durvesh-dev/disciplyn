"""Tag and model definitions for Disciplyn.

20 behaviour tags -> 14 binary classifiers.
6 conflicting pairs share one model (prob = negative tag, 1 - prob = positive tag).
8 tags have their own model.
"""

MODELS = {
    # ---- merged pairs: label 1 = negative tag, label 0 = positive tag ----
    "stop_loss": {
        "kind": "pair", "negative": "Ignoring Stop-Losses", "positive": "Stop-Loss Adherence",
        "features": ["sl_set_rate", "sl_modified_rate", "sl_widened_rate", "sl_breach_rate",
                     "avg_loss_vs_planned_risk", "max_loss_pct_capital", "loser_hold_ratio"],
    },
    "trade_frequency": {
        "kind": "pair", "negative": "Overtrading", "positive": "Controlled Trading Frequency",
        "features": ["trades_per_day", "max_trades_in_day", "no_setup_rate", "avg_gap_min",
                     "intraday_rate", "win_rate", "q_impulsivity"],
    },
    "discipline": {
        "kind": "pair", "negative": "Lack of Discipline", "positive": "Disciplined Trading",
        "features": ["no_setup_rate", "sl_set_rate", "target_set_rate", "daily_loss_limit_breach_rate",
                     "off_hours_rate", "size_cv", "sl_widened_rate", "q_discipline"],
    },
    "profit_exit": {
        "kind": "pair", "negative": "Greed", "positive": "Profit Booking",
        "features": ["giveback_ratio", "target_exceed_rate", "winner_hold_ratio",
                     "target_set_rate", "winners_turned_losers_rate", "avg_win_pct"],
    },
    "entry_timing": {
        "kind": "pair", "negative": "FOMO", "positive": "Patience",
        "features": ["chase_rate", "avg_pre_entry_move", "high_volume_entry_rate", "extreme_rsi_entry_rate",
                     "no_setup_rate", "avg_gap_min", "q_impulsivity"],
    },
    "leverage_risk": {
        "kind": "pair", "negative": "Overleveraging", "positive": "Risk Management",
        "features": ["avg_leverage", "max_leverage", "high_leverage_rate", "avg_risk_pct_capital",
                     "max_loss_pct_capital", "avg_rr_planned"],
    },
    # ---- standalone negative tags: label 1 = tag present ----
    "revenge_trading": {
        "kind": "single", "tag": "Revenge Trading", "polarity": "negative",
        "features": ["post_loss_gap_ratio", "post_loss_size_ratio", "quick_reentry_after_loss_rate",
                     "post_loss_win_rate", "trades_per_day", "q_impulsivity"],
    },
    "panic_selling": {
        "kind": "single", "tag": "Panic Selling", "polarity": "negative",
        "features": ["panic_exit_rate", "early_exit_in_drop_rate", "avg_market_move_during_hold_on_exit",
                     "exit_before_sl_rate", "q_emotional"],
    },
    "loss_aversion": {
        "kind": "single", "tag": "Loss Aversion", "polarity": "negative",
        "features": ["loser_hold_ratio", "avg_loss_vs_planned_risk", "averaging_down_rate",
                     "win_loss_size_ratio", "sl_modified_rate"],
    },
    "confirmation_bias": {
        "kind": "single", "tag": "Confirmation Bias", "polarity": "negative",
        "features": ["averaging_down_rate", "same_symbol_repeat_loss_rate", "loser_hold_ratio",
                     "q_confirmation", "q_discipline"],
    },
    # ---- standalone positive tags: label 1 = tag present ----
    "emotional_stability": {
        "kind": "single", "tag": "Emotional Stability", "polarity": "positive",
        "features": ["post_loss_gap_ratio", "post_loss_size_ratio", "panic_exit_rate", "chase_rate",
                     "size_cv", "pnl_streak_behaviour_shift", "q_emotional"],
    },
    "position_sizing": {
        "kind": "single", "tag": "Consistent Position Sizing", "polarity": "positive",
        "features": ["size_cv", "post_loss_size_ratio", "post_win_size_ratio", "max_size_jump",
                     "avg_risk_pct_capital"],
    },
    "learning_from_losses": {
        "kind": "single", "tag": "Learning from Losses", "polarity": "positive",
        "features": ["journal_rate", "journal_after_loss_rate", "mistake_repeat_trend",
                     "same_symbol_repeat_loss_rate", "q_learning"],
    },
    "strategic_consistency": {
        "kind": "single", "tag": "Strategic Consistency", "polarity": "positive",
        "features": ["n_strategies", "strategy_switch_rate", "switch_after_loss_rate",
                     "top_strategy_share", "no_setup_rate"],
    },
}

# Trait names used by the simulator (one latent ground-truth trait per model).
TRAITS = list(MODELS.keys())

ALL_FEATURES = sorted({f for m in MODELS.values() for f in m["features"]})
