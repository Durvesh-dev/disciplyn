"""
Disciplyn - Synthetic Trader Behaviour Dataset Generator
========================================================

Why synthetic?
  No public dataset exists where real retail trades are labelled with behavioural
  tags (Revenge Trading, FOMO, ...). We therefore simulate traders whose hidden
  behavioural traits are KNOWN (ground truth), let those traits drive their
  trading decisions, and record only what a broker + market API would show:

    trades.csv          -> one row per round-trip trade (broker trade book + order book
                           events such as stop-loss modifications) + market context at
                           entry/exit (from the Market Data API: MACD, RSI, volume,
                           sentiment, India VIX)
    questionnaire.csv   -> onboarding questionnaire answers (5 random sets, Likert 1-5)
    labels.csv          -> ground-truth 20 behaviour tags per trader (from hidden traits)
    traders.csv         -> trader profile (capital, experience, segment)

The ML models never see the hidden traits; they must recover them from the
observable trades + market context + questionnaire, exactly like production.

Usage:  python generate_dataset.py --traders 1500 --days 30 --seed 42
"""
import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SYMBOLS = {  # NSE symbol -> approx price
    "RELIANCE": 2900, "TCS": 4100, "HDFCBANK": 1650, "INFY": 1850, "ICICIBANK": 1250,
    "SBIN": 820, "TATAMOTORS": 980, "ADANIENT": 3100, "BAJFINANCE": 7200, "ITC": 470,
    "AXISBANK": 1180, "LT": 3600, "MARUTI": 12400, "SUNPHARMA": 1750, "WIPRO": 540,
    "NIFTY_FUT": 24500, "BANKNIFTY_FUT": 52000,
}
STRATEGIES = ["ORB", "VWAP_Reversion", "Breakout", "Pullback", "Momentum",
              "Mean_Reversion", "News_Based", "Gap_Fill"]

# Hidden behavioural dimensions. 14 dims -> 14 models.
# For merged pairs, trait = probability-like intensity of the NEGATIVE tag.
NEG_PAIRS = {  # dim : (negative tag, positive opposite tag)
    "stop_loss": ("Ignoring Stop-Losses", "Stop-Loss Adherence"),
    "frequency": ("Overtrading", "Controlled Trading Frequency"),
    "discipline": ("Lack of Discipline", "Disciplined Trading"),
    "profit_exit": ("Greed", "Profit Booking"),
    "entry_timing": ("FOMO", "Patience"),
    "leverage": ("Overleveraging", "Risk Management"),
}
NEG_SINGLE = {"revenge": "Revenge Trading", "panic": "Panic Selling",
              "loss_aversion": "Loss Aversion", "confirmation_bias": "Confirmation Bias"}
POS_SINGLE = {"emotional_stability": "Emotional Stability",
              "consistent_sizing": "Consistent Position Sizing",
              "learning": "Learning from Losses",
              "strategic_consistency": "Strategic Consistency"}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_traits(rng):
    """Correlated traits: a latent 'impulsivity' factor drives most negatives."""
    imp = rng.normal()
    fear = rng.normal()          # second latent factor: fear / anxiety
    t = {}
    t["revenge"] = sigmoid(1.2 * imp + rng.normal(0, 1.0) - 0.3)
    t["frequency"] = sigmoid(1.0 * imp + rng.normal(0, 1.0) - 0.2)
    t["entry_timing"] = sigmoid(1.0 * imp + rng.normal(0, 1.0) - 0.2)
    t["discipline"] = sigmoid(0.9 * imp + rng.normal(0, 1.0) - 0.3)
    t["leverage"] = sigmoid(0.8 * imp + rng.normal(0, 1.1) - 0.4)
    t["profit_exit"] = sigmoid(0.6 * imp - 0.4 * fear + rng.normal(0, 1.1) - 0.2)
    t["panic"] = sigmoid(1.2 * fear + rng.normal(0, 1.0) - 0.4)
    t["loss_aversion"] = sigmoid(0.6 * fear + 0.4 * imp + rng.normal(0, 1.0) - 0.3)
    t["stop_loss"] = sigmoid(0.5 * imp + 0.5 * fear + rng.normal(0, 1.0) - 0.3)
    t["confirmation_bias"] = sigmoid(0.4 * imp + rng.normal(0, 1.2) - 0.2)
    # positive traits (high = good)
    reactivity = (t["revenge"] + t["panic"] + t["entry_timing"]) / 3
    t["emotional_stability"] = float(np.clip(1 - reactivity + rng.normal(0, 0.12), 0, 1))
    t["consistent_sizing"] = sigmoid(-0.8 * imp + rng.normal(0, 1.0) + 0.2)
    t["learning"] = sigmoid(-0.3 * imp + rng.normal(0, 1.2))
    t["strategic_consistency"] = sigmoid(-0.7 * imp + rng.normal(0, 1.0) + 0.1)
    return t


def make_labels(tid, t):
    row = {"trader_id": tid}
    for dim, (neg, pos) in NEG_PAIRS.items():
        row[neg] = int(t[dim] > 0.5)
        row[pos] = 1 - row[neg]          # mutually exclusive pair
    for dim, tag in NEG_SINGLE.items():
        row[tag] = int(t[dim] > 0.5)
    for dim, tag in POS_SINGLE.items():
        row[tag] = int(t[dim] > 0.5)
    return row


def simulate_trader(tid, rng, days, start_date):
    t = sample_traits(rng)
    capital = float(np.round(rng.lognormal(np.log(5e5), 0.7), -3))
    experience = int(rng.choice([0, 1, 2, 3, 5, 8], p=[.25, .25, .2, .15, .1, .05]))
    profile = {"trader_id": tid, "capital_inr": capital, "experience_years": experience,
               "segment": rng.choice(["Equity Intraday", "F&O", "Equity Intraday + F&O"]),
               "broker": rng.choice(["Zerodha", "Groww", "Dhan", "Upstox"])}

    plan_strats = list(rng.choice(STRATEGIES, size=2, replace=False))
    fav_symbols = list(rng.choice(list(SYMBOLS), size=5, replace=False))
    cur_strat = plan_strats[0]
    # continuous, overlapping leverage habit (not a clean split) -> realistic difficulty
    base_lev = float(np.exp(rng.normal(np.log(0.5 + 3.5 * t["leverage"] ** 1.5), 0.45)))
    size_noise = 0.08 + 0.7 * (1 - t["consistent_sizing"])
    daily_loss_limit = 0.02 * capital

    trades, tid_n = [], 0
    for d in range(days):
        day = start_date + timedelta(days=d + 2 * (d // 5))   # skip weekends
        learn_decay = 1 - 0.6 * t["learning"] * (d / days)    # learners make fewer mistakes over time
        n_planned = rng.poisson(1.2 + 8.0 * t["frequency"] ** 1.6)
        n_planned = max(n_planned, 0)
        minute = 15 + rng.integers(0, 40)           # minutes after 09:00
        day_pnl, prev, k = 0.0, None, 0
        market_bias = rng.normal(0, 1)              # day's market direction
        vix = float(np.clip(rng.normal(14, 3), 9, 30))
        while True:
            # --- decide whether to take another trade
            revenge_now = (prev is not None and prev["pnl"] < 0 and
                           rng.random() < t["revenge"] * 0.85 * learn_decay)
            breached = day_pnl < -daily_loss_limit
            if k >= n_planned and not revenge_now:
                break
            if breached and rng.random() > t["discipline"]:   # disciplined traders stop
                break
            if k > 40:
                break
            gap = rng.integers(1, 9) if revenge_now else int(rng.exponential(35 + 60 * (1 - t["frequency"])) + 3)
            minute += gap
            if minute > 375 - 5:                     # 15:15 cut-off
                break

            # --- strategy choice / switching
            if prev is not None and prev["pnl"] < 0 and rng.random() < 0.6 * (1 - t["strategic_consistency"]):
                cur_strat = rng.choice(STRATEGIES)
            off_plan = (cur_strat not in plan_strats) or (rng.random() < 0.35 * t["discipline"])
            # re-entry on same symbol/same direction after a loss -> confirmation bias
            if prev is not None and prev["pnl"] < 0 and rng.random() < 0.7 * t["confirmation_bias"]:
                symbol, side = prev["symbol"], prev["side"]
            else:
                symbol = rng.choice(fav_symbols) if rng.random() < 0.8 else rng.choice(list(SYMBOLS))
                side = "LONG" if rng.random() < 0.55 + 0.1 * np.sign(market_bias) else "SHORT"
            sgn = 1 if side == "LONG" else -1

            # --- market context at entry (Market Data API)
            fomo_now = rng.random() < t["entry_timing"] * 0.8 * learn_decay
            macd_hist = rng.normal(0.4 * market_bias, 1.0)
            sentiment = float(np.clip(np.tanh(0.6 * market_bias + rng.normal(0, 0.4)), -1, 1))
            if fomo_now:
                pre_ret = sgn * abs(rng.normal(1.6, 0.6))     # chasing a move already made
                vol_z = abs(rng.normal(3.0, 0.8))
                rsi = 72 + abs(rng.normal(6, 4)) if sgn > 0 else 28 - abs(rng.normal(6, 4))
            else:
                pre_ret = rng.normal(-0.2 * sgn, 0.5)          # patient: enter on pullback
                vol_z = rng.normal(0.3, 0.9)
                rsi = float(np.clip(rng.normal(50 - 6 * sgn, 9), 15, 85))
            # confirmation bias: ignores contrary indicator readings
            if rng.random() < t["confirmation_bias"] * 0.6:
                macd_hist = -sgn * abs(macd_hist) - 0.3
                sentiment = -sgn * abs(sentiment)

            # --- sizing / leverage
            size_mult = rng.uniform(1.6, 3.0) if revenge_now else 1.0
            lev = max(0.1, base_lev * size_mult * float(np.exp(rng.normal(0, size_noise))))
            price = SYMBOLS[symbol] * (1 + rng.normal(0, 0.03))
            position_value = capital * lev
            qty = max(1, int(position_value / price))
            position_value = qty * price
            sl_set = rng.random() < (0.97 - 0.45 * t["discipline"] - 0.25 * t["stop_loss"])
            target_set = rng.random() < (0.95 - 0.55 * t["discipline"])
            sl_pct = abs(rng.normal(0.8, 0.25)) + 0.2
            rr = rng.uniform(1.5, 2.5)
            risk_inr = position_value * sl_pct / 100

            # --- outcome
            p_win = 0.50 - 0.10 * fomo_now - 0.07 * revenge_now - 0.05 * off_plan
            win = rng.random() < p_win
            hold = rng.lognormal(np.log(25), 0.6)
            sl_mod, sl_widened, sl_hit, sl_honored = 0, 0, 0, None
            exit_reason, target_hit_held, averaged_down, mkt_move = "target", 0, 0, rng.normal(0, 0.5)
            post_exit = rng.normal(0, 0.4)
            panic_now = False
            if win:
                mfe_R = rr + abs(rng.normal(0.6, 0.5))
                if rng.random() < t["profit_exit"] * 0.85:      # greed: hold past target, give back
                    target_hit_held = 1
                    R = mfe_R * rng.uniform(0.05, 0.55) * (1 if rng.random() < 0.7 else -0.5)
                    hold *= rng.uniform(2, 5)
                    exit_reason = "manual_late"
                else:
                    R = rr if target_set else rr * rng.uniform(0.6, 1.1)
                    exit_reason = "target" if target_set else "manual"
                mae_R = abs(rng.normal(0.3, 0.2))
            else:
                mae_R = 1 + abs(rng.normal(0.8, 0.6))
                mfe_R = abs(rng.normal(0.3, 0.3))
                panic_now = rng.random() < t["panic"] * 0.55 and not fomo_now
                if panic_now:                        # exits early in a sharp market drop, then market recovers
                    mkt_move = -sgn * abs(rng.normal(1.3, 0.4))
                    vol_exit_z = abs(rng.normal(2.8, 0.7))
                    R = -abs(rng.normal(0.35, 0.15))
                    hold *= rng.uniform(0.15, 0.4)
                    post_exit = sgn * abs(rng.normal(0.9, 0.4))
                    exit_reason = "manual_early"
                    mae_R = abs(R) + 0.1
                else:
                    sl_hit = 1 if sl_set else 0
                    if sl_set and rng.random() < t["stop_loss"] * 0.9:   # moves the stop
                        sl_mod = int(rng.integers(1, 4)); sl_widened = 1; sl_honored = 0
                        R = -(1 + abs(rng.normal(1.0, 0.7)))
                        hold *= rng.uniform(1.5, 3)
                        exit_reason = "manual"
                    elif sl_set:
                        sl_honored = 1
                        R = -1.0 - abs(rng.normal(0, 0.05))
                        exit_reason = "stop_loss"
                    else:
                        R = -(0.8 + abs(rng.normal(0.8, 0.8)))
                        exit_reason = "manual"
                    if rng.random() < t["loss_aversion"] * 0.85:   # refuses to close loser
                        hold *= rng.uniform(2.5, 7)
                        R *= rng.uniform(1.3, 2.4)
                        averaged_down = int(rng.random() < 0.6)
                        exit_reason = "manual_late" if exit_reason != "stop_loss" else "stop_loss"
            vol_exit = vol_exit_z if panic_now else rng.normal(0.3, 0.9)
            pnl = R * risk_inr
            entry_time = datetime.combine(day, datetime.min.time()) + timedelta(hours=9, minutes=int(minute))
            exit_time = entry_time + timedelta(minutes=float(min(hold, 375 - minute + 5)))
            exit_price = price * (1 + sgn * R * sl_pct / 100)
            journal = int(pnl < 0 and rng.random() < 0.05 + 0.85 * t["learning"]) or int(pnl >= 0 and rng.random() < 0.3 * t["learning"])
            row = dict(
                trader_id=tid, trade_id=f"{tid}-{tid_n:04d}", trade_date=day.date().isoformat(),
                entry_time=entry_time.isoformat(timespec="minutes"), exit_time=exit_time.isoformat(timespec="minutes"),
                symbol=symbol, side=side, qty=qty, entry_price=round(price, 2), exit_price=round(exit_price, 2),
                pnl=round(pnl, 2), capital_at_entry=round(capital, 2), leverage=round(position_value / capital, 3),
                strategy=cur_strat, in_trading_plan=int(not off_plan),
                stop_loss_set=int(sl_set), stop_loss_price=round(price * (1 - sgn * sl_pct / 100), 2) if sl_set else None,
                target_set=int(target_set), target_price=round(price * (1 + sgn * rr * sl_pct / 100), 2) if target_set else None,
                sl_modified_count=sl_mod, sl_widened=sl_widened, sl_hit=sl_hit, sl_honored=sl_honored,
                target_hit_but_held=target_hit_held, averaged_down=averaged_down, exit_reason=exit_reason,
                mfe_r=round(mfe_R, 3), mae_r=round(mae_R, 3), r_multiple=round(R, 3),
                holding_minutes=round(float((exit_time - entry_time).total_seconds() / 60), 1),
                # market context (Market Data API)
                pre_entry_return_15m_pct=round(pre_ret, 3), volume_zscore_entry=round(vol_z, 3),
                rsi_14_entry=round(float(rsi), 2), macd_hist_entry=round(macd_hist, 3),
                market_sentiment_entry=round(sentiment, 3), india_vix=round(vix, 2),
                market_move_during_hold_pct=round(mkt_move, 3), volume_zscore_exit=round(vol_exit, 3),
                post_exit_return_30m_pct=round(post_exit, 3), journal_note=journal,
            )
            trades.append(row)
            capital = max(capital + pnl, 1e4)
            day_pnl += pnl
            prev, k, tid_n = row, k + 1, tid_n + 1
    return profile, t, trades


QUESTION_BANK = {
    # construct -> list of (question, reverse_scored)
    "impulsivity": [("I often enter a trade quickly so I don't miss a move.", False),
                    ("After a loss, I want to get back into the market immediately.", False)],
    "confirmation_bias": [("Once I have a view on a stock, I mostly look for news that supports it.", False),
                          ("I actively look for reasons my trade idea could be wrong.", True)],
    "emotional": [("My emotions (fear, frustration, excitement) affect my trading decisions.", False),
                  ("I stay calm when a trade goes against me.", True)],
    "discipline": [("I follow a written trading plan with entry, stop-loss and target.", True),
                   ("I break my own rules when I feel confident.", False)],
    "learning": [("I review my losing trades and note what went wrong.", True),
                 ("I keep repeating the same mistakes.", False)],
}
CONSTRUCT_TRAIT = {"impulsivity": ["revenge", "entry_timing", "frequency"],
                   "confirmation_bias": ["confirmation_bias"],
                   "emotional": ["__inv_emotional_stability"],
                   "discipline": ["discipline"], "learning": ["__inv_learning"]}


def simulate_questionnaire(tid, t, rng):
    """5 question sets; each set asks every construct with slightly different wording (set_no
    stored). Self-reports are noisy and biased toward socially desirable answers."""
    set_no = int(rng.integers(1, 6))
    rows = []
    qn = 1
    for construct, qs in QUESTION_BANK.items():
        vals = []
        for tr in CONSTRUCT_TRAIT[construct]:
            vals.append(1 - t[tr[6:]] if tr.startswith("__inv_") else t[tr])
        true = float(np.mean(vals))
        for q, rev in qs:
            latent = true - 0.12 + rng.normal(0, 0.18)   # under-reporting bias + noise
            ans = int(np.clip(np.round(1 + 4 * latent), 1, 5))
            if rev:
                ans = 6 - ans
            rows.append({"trader_id": tid, "set_no": set_no, "question_no": qn,
                         "construct": construct, "reverse_scored": int(rev),
                         "question": q, "answer": ans})
            qn += 1
    return rows


def _simulate_chunk(args):
    """Simulate a block of traders. Each trader has its own seed -> fully reproducible
    regardless of how many worker processes are used."""
    ids, seed, days, q_rate, out, chunk_no = args
    start = datetime(2026, 7, 1)
    profiles, labels, trades, qrows, traits = [], [], [], [], []
    for i in ids:
        rng = np.random.default_rng([seed, i])
        tid = f"TR{i:05d}"
        p, t, tr = simulate_trader(tid, rng, days, start)
        profiles.append(p); trades.extend(tr)
        labels.append(make_labels(tid, t))
        traits.append({"trader_id": tid, **{k: round(v, 4) for k, v in t.items()}})
        if rng.random() < q_rate:
            qrows.extend(simulate_questionnaire(tid, t, rng))
    pd.DataFrame(trades).to_csv(f"{out}/_trades_{chunk_no:04d}.csv", index=False)
    return profiles, labels, qrows, traits, len(trades)


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--traders", type=int, default=10000)
    ap.add_argument("--days", type=int, default=60, help="trading days per trader")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--questionnaire_rate", type=float, default=0.7,
                    help="share of traders that answered the onboarding questionnaire")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    chunk = 250
    jobs = [(list(range(s, min(s + chunk, a.traders))), a.seed, a.days, a.questionnaire_rate, a.out, n)
            for n, s in enumerate(range(0, a.traders, chunk))]
    profiles, labels, qrows, traits, total = [], [], [], [], 0
    with Pool(a.workers) as pool:
        for k, (p, l, q, t, n) in enumerate(pool.imap(_simulate_chunk, jobs)):
            profiles += p; labels += l; qrows += q; traits += t; total += n
            print(f"  simulated {min((k + 1) * chunk, a.traders):,}/{a.traders:,} traders, {total:,} trades", flush=True)
    # merge chunk files into one trades.csv (streamed, low memory)
    with open(f"{a.out}/trades.csv", "w", newline="") as fo:
        for n in range(len(jobs)):
            path = f"{a.out}/_trades_{n:04d}.csv"
            with open(path) as fi:
                header = fi.readline()
                if n == 0:
                    fo.write(header)
                fo.write(fi.read())
            os.remove(path)
    pd.DataFrame(profiles).to_csv(f"{a.out}/traders.csv", index=False)
    pd.DataFrame(qrows).to_csv(f"{a.out}/questionnaire.csv", index=False)
    pd.DataFrame(labels).to_csv(f"{a.out}/labels.csv", index=False)
    pd.DataFrame(traits).to_csv(f"{a.out}/hidden_traits.csv", index=False)
    print(f"Done: {a.traders:,} traders x {a.days} days = {total:,} trades -> {a.out}")


if __name__ == "__main__":
    main()
