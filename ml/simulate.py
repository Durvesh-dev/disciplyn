"""Synthetic trader simulator with ground-truth behaviour.

Each trader gets hidden behaviour traits (0..1). Trades are generated so those
traits leave footprints in the trade/order book (quick re-entries after losses,
widened stop-losses, chasing big candles, ...). The models only see the trades
(+ questionnaire answers) and must recover the hidden traits.

Usage: python -m ml.simulate --traders 1500 --days 22
Outputs: data/trades.csv, data/traders.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from .config import TRAITS

SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "TATAMOTORS",
           "ITC", "LT", "AXISBANK", "BAJFINANCE", "ADANIENT", "NIFTY FUT", "BANKNIFTY FUT"]
STRATS = ["ORB breakout", "VWAP pullback", "EMA crossover", "Support bounce",
          "Gap fill", "Momentum", "Mean reversion", "News play"]


def sig(x):
    return 1 / (1 + np.exp(-x))


def sample_traits(rng):
    z = rng.normal()  # shared "impulsivity" factor -> realistic correlations
    t = {}
    for name in ["stop_loss", "trade_frequency", "discipline", "profit_exit", "entry_timing",
                 "leverage_risk", "revenge_trading", "panic_selling", "loss_aversion",
                 "confirmation_bias"]:
        t[name] = float(sig(1.0 * z + rng.normal(0, 1.1)))          # negative intensity
    for name in ["position_sizing", "learning_from_losses", "strategic_consistency"]:
        t[name] = float(sig(-0.9 * z + rng.normal(0, 1.1)))         # positive intensity
    # emotional stability is the opposite of the emotional behaviours
    emo_neg = np.mean([t["revenge_trading"], t["panic_selling"], t["entry_timing"]])
    t["emotional_stability"] = float(np.clip(1 - emo_neg + rng.normal(0, 0.12), 0, 1))
    return z, t


def questionnaire(rng, z, t, missing_p=0.3):
    """Likert 1-5 self-report answers (noisy). Some users skip -> NaN."""
    def lik(x, noise=0.15):
        return float(np.clip(np.round(1 + 4 * np.clip(x + rng.normal(0, noise), 0, 1)), 1, 5))
    if rng.random() < missing_p:
        return {k: np.nan for k in ["q_impulsivity", "q_discipline", "q_emotional", "q_confirmation", "q_learning"]}
    return {
        "q_impulsivity": lik(sig(z)),
        "q_discipline": lik(1 - t["discipline"]),
        "q_emotional": lik(t["emotional_stability"]),
        "q_confirmation": lik(t["confirmation_bias"]),
        "q_learning": lik(t["learning_from_losses"]),
    }


def simulate_trader(rng, tid, t, days, start="2026-09-01"):
    capital = float(rng.uniform(2e5, 2e6))
    n_strats = 1 + int(round(5 * (1 - t["strategic_consistency"])))
    my_strats = list(rng.choice(STRATS, size=n_strats, replace=False))
    cur_strat = my_strats[0]
    fav_syms = list(rng.choice(SYMBOLS, size=5, replace=False))
    base_lev = 1 + 5 * t["leverage_risk"] * rng.uniform(0.6, 1.4)
    size_sd = 0.08 + 0.6 * (1 - t["position_sizing"])
    rows = []
    trade_no = 0
    bdays = pd.bdate_range(start, periods=days)
    last_loss_sym = None
    for d_i, day in enumerate(bdays):
        progress = d_i / max(days - 1, 1)
        improve = 1 - 0.7 * t["learning_from_losses"] * progress   # learners make fewer mistakes over time
        n = rng.poisson(0.8 + 7 * t["trade_frequency"]) + 1
        clock = 9 * 60 + 15 + rng.exponential(20)
        day_pnl = 0.0
        last_loss = False
        k = 0
        while k < n and clock < 15 * 60 + 25:
            # daily loss-limit rule: disciplined traders stop after -2% day
            if day_pnl < -0.02 * capital and rng.random() > t["discipline"]:
                break
            revenge = last_loss and rng.random() < t["revenge_trading"]
            gap = rng.exponential(4) if revenge else rng.exponential(25 + 70 * (1 - t["trade_frequency"]))
            clock += gap
            if clock > 15 * 60 + 29:
                break
            if revenge:
                n += 1  # revenge trades add extra trades
            # --- strategy ---
            if last_loss and rng.random() < (1 - t["strategic_consistency"]) * 0.6:
                cur_strat = my_strats[rng.integers(len(my_strats))]
            no_setup_p = (0.03 + 0.35 * t["discipline"] + 0.25 * t["trade_frequency"] * (0.5 + revenge)) * improve
            strategy = "none" if rng.random() < no_setup_p else cur_strat
            # --- symbol ---
            if last_loss_sym and rng.random() < 0.7 * t["confirmation_bias"]:
                sym = last_loss_sym
            else:
                sym = fav_syms[rng.integers(5)]
            side = "LONG" if rng.random() < 0.6 else "SHORT"
            # --- market context at entry (FOMO = chasing) ---
            chasing = rng.random() < 0.75 * t["entry_timing"]
            if chasing:
                pre_move = rng.uniform(1.2, 4.0)
                vol_z = rng.uniform(1.8, 5)
                rsi = rng.uniform(72, 90)
            else:
                pre_move = rng.normal(0, 0.5)
                vol_z = rng.normal(0, 1)
                rsi = rng.normal(52, 10)
            rsi_dir = rsi if side == "LONG" else 100 - rsi
            macd = rng.normal(0.3 * pre_move, 0.5)
            # --- size & leverage ---
            lev = max(1.0, base_lev * rng.lognormal(0, 0.15))
            frac = 0.25 * rng.lognormal(0, size_sd)
            if revenge:
                frac *= rng.uniform(1.5, 3.0)
                lev *= rng.uniform(1.0, 1.5)
            price = float(rng.uniform(200, 3500))
            notional = capital * frac * lev / 5
            qty = max(1, int(notional / price))
            # --- plan: stop-loss & target ---
            sl_set = rng.random() > 0.75 * max(t["stop_loss"], 0.6 * t["discipline"]) * improve
            sl_pct = rng.uniform(0.6, 1.5) if sl_set else np.nan
            rr = rng.uniform(1.0, 3.0) * (1.2 - 0.5 * t["leverage_risk"])
            tgt_set = rng.random() > 0.7 * t["discipline"] * improve
            tgt_pct = (sl_pct if sl_set else 1.0) * rr if tgt_set else np.nan
            # --- price path (in % from entry, trade direction) ---
            steps = 75
            drift = (-0.012 if chasing else 0.004)
            path = np.cumsum(rng.normal(drift, 0.17, steps))
            mkt_path = path + rng.normal(0, 0.05, steps)
            exit_idx, reason = steps - 1, "time"
            sl_mod, sl_wid = 0, 0
            averaged = 0
            cur_sl = sl_pct if sl_set else np.inf
            if not sl_set and rng.random() > t["loss_aversion"]:
                cur_sl = 2.0  # mental stop
            target_passed = False
            for i in range(steps):
                p = path[i]
                # panic: sharp early drop -> dump before stop
                if i < 20 and p < -0.45 and mkt_path[i] < -0.4 and rng.random() < 0.25 * t["panic_selling"]:
                    exit_idx, reason = i, "panic"
                    break
                if p <= -cur_sl:
                    if rng.random() < 0.8 * t["stop_loss"] * improve or rng.random() < 0.5 * t["loss_aversion"]:
                        cur_sl *= 1.8
                        sl_mod += 1
                        sl_wid = 1
                        if averaged == 0 and rng.random() < 0.6 * max(t["confirmation_bias"], t["loss_aversion"]):
                            averaged = 1
                        continue
                    exit_idx, reason = i, "stoploss"
                    break
                if not np.isnan(tgt_pct) and p >= tgt_pct and not target_passed:
                    if rng.random() < 0.85 * t["profit_exit"]:
                        target_passed = True  # greed: keep holding
                        continue
                    exit_idx, reason = i, "target"
                    break
                if target_passed and p < tgt_pct * 0.4:
                    exit_idx, reason = i, "giveback"
                    break
                # non-loss-averse traders cut stale losers; loss-averse hold
                if i > 35 and p < -0.3 and rng.random() < 0.08 * (1 - t["loss_aversion"]):
                    exit_idx, reason = i, "cut"
                    break
                # disciplined profit taking without target
                if np.isnan(tgt_pct) and p > 1.0 and rng.random() < 0.15 * (1 - t["profit_exit"]):
                    exit_idx, reason = i, "booked"
                    break
            ret = float(path[exit_idx])
            if averaged:
                qty_eff = qty * 1.5
                ret_eff = ret * 1.3
            else:
                qty_eff, ret_eff = qty, ret
            pnl = qty_eff * price * ret_eff / 100
            hold_min = (exit_idx + 1) * 3
            entry_t = day + pd.Timedelta(minutes=float(clock))
            exit_t = entry_t + pd.Timedelta(minutes=hold_min)
            journal_p = 0.05 + 0.75 * t["learning_from_losses"] * (1.0 if pnl < 0 else 0.5)
            rows.append({
                "trader_id": tid, "trade_id": f"{tid}-{trade_no}", "symbol": sym, "strategy": strategy,
                "side": side, "entry_time": entry_t, "exit_time": exit_t, "qty": int(qty_eff),
                "entry_price": round(price, 2),
                "exit_price": round(price * (1 + (ret if side == "LONG" else -ret) / 100), 2),
                "pnl": round(pnl, 2), "capital": round(capital, 2), "leverage": round(lev, 2),
                "sl_pct": None if np.isnan(sl_pct) else round(sl_pct, 3),
                "sl_modified_count": sl_mod, "sl_widened": sl_wid,
                "target_pct": None if np.isnan(tgt_pct) else round(tgt_pct, 3),
                "exit_reason": reason, "mfe_pct": round(float(path[:exit_idx + 1].max()), 3),
                "pre_entry_move_pct": round(pre_move, 3), "volume_z": round(vol_z, 2),
                "rsi": round(rsi_dir, 1), "macd_hist": round(macd, 3),
                "market_move_during_hold_pct": round(float(mkt_path[exit_idx]), 3),
                "averaged_down": averaged, "journal_note": int(rng.random() < journal_p),
            })
            trade_no += 1
            k += 1
            capital += pnl
            day_pnl += pnl
            last_loss = pnl < 0
            last_loss_sym = sym if pnl < 0 else None
            clock += hold_min
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traders", type=int, default=1500)
    ap.add_argument("--days", type=int, default=22)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    trades, traders = [], []
    for i in range(a.traders):
        tid = f"T{i:05d}"
        z, t = sample_traits(rng)
        q = questionnaire(rng, z, t)
        traders.append({"trader_id": tid, **{f"trait_{k}": round(v, 4) for k, v in t.items()}, **q})
        trades += simulate_trader(rng, tid, t, a.days)
        if (i + 1) % 250 == 0:
            print(f"  simulated {i + 1}/{a.traders} traders")
    pd.DataFrame(trades).to_csv(f"{a.out}/trades.csv", index=False)
    pd.DataFrame(traders).to_csv(f"{a.out}/traders.csv", index=False)
    print(f"trades: {len(trades)}  traders: {len(traders)}")


if __name__ == "__main__":
    main()
