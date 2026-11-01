"""
RAG: Context Aggregator + LLM Advisor.

Context = user behaviour tags + trade stats + market snapshot + retrieved book principles.
Retrieval: TF-IDF over the knowledge base, boosted by the user's detected negative tags.
(Production: swap TF-IDF for embeddings + vector DB over chunked books.)

LLM: uses GROQ_API_KEY or GEMINI_API_KEY if set (both have free tiers),
otherwise a rule-based advisor so the demo always works offline.
"""
import json
import os

try:
    import httpx
except ImportError:  # only needed for real LLM / market API calls
    httpx = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
KB = json.load(open(os.path.join(HERE, "knowledge_base.json"), encoding="utf-8"))
_docs = [" ".join(d["tags"]) + " " + d["text"] for d in KB]
_vec = TfidfVectorizer(stop_words="english").fit(_docs)
_M = _vec.transform(_docs)

SYSTEM = """You are Disciplyn, a trading-behaviour coach for Indian retail traders (NSE/BSE).
You do NOT give buy/sell recommendations or investment advice. You help the trader reduce negative
behaviours and strengthen positive ones. Always:
- ground advice in the trader's own numbers (stats and tags provided),
- cite the technique source in brackets, e.g. [Trading in the Zone],
- give 2-3 concrete rules they can apply tomorrow,
- be warm, short and specific (under 180 words)."""


def retrieve(query: str, neg_tags: list[str], k: int = 4):
    sims = cosine_similarity(_vec.transform([query + " " + " ".join(neg_tags)]), _M)[0]
    for i, d in enumerate(KB):
        sims[i] += 0.15 * len(set(d["tags"]) & set(neg_tags))
    return [KB[i] for i in sims.argsort()[::-1][:k]]


def market_snapshot():
    """Hook for the team's Market Data API. Set MARKET_API_URL to use the real one."""
    url = os.getenv("MARKET_API_URL")
    if url and httpx:
        try:
            return httpx.get(url, timeout=5).json()
        except Exception:
            pass
    return {"index": "NIFTY 50", "sentiment": "mildly bullish", "macd": "bullish crossover on daily",
            "volume": "above 20-day average", "india_vix": 13.8, "note": "mock data - MARKET_API_URL not set"}


def build_context(analysis: dict, query: str):
    neg = sorted([t for t in analysis["tags"] if t["polarity"] == "negative" and t["present"]],
                 key=lambda t: -t["probability"])
    pos = [t["tag"] for t in analysis["tags"] if t["polarity"] == "positive" and t["present"]]
    docs = retrieve(query, [t["tag"] for t in neg[:4]])
    ctx = {
        "behaviour_score": analysis["behaviour_score"],
        "negative_behaviours": [f"{t['tag']} ({t['probability']:.0%})" for t in neg],
        "positive_behaviours": pos,
        "trade_stats": analysis["stats"],
        "market": market_snapshot(),
        "techniques": [{"source": d["source"], "text": d["text"]} for d in docs],
    }
    return ctx, neg, docs


def _llm_endpoint():
    if os.getenv("GROQ_API_KEY"):
        return ("https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_API_KEY"),
                os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    if os.getenv("GEMINI_API_KEY"):
        return ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                os.getenv("GEMINI_API_KEY"), os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    return None


def fallback_answer(ctx, neg, docs, query):
    s = ctx["trade_stats"]
    if not neg:
        lines = [f"Your behaviour score is **{ctx['behaviour_score']}/100** and no strong negative patterns showed up. "
                 f"Strengths: {', '.join(ctx['positive_behaviours'][:4])}.",
                 "Keep journaling and keep risk per trade constant so this stays repeatable."]
        return "\n\n".join(lines)
    top = neg[0]["tag"]
    evidence = {
        "Revenge Trading": f"{s['quick_reentry_after_loss_rate']:.0%} of your trades after a loss were re-entered within 10 minutes, at {s['size_after_loss_vs_win']:.1f}x your usual size change.",
        "Overtrading": f"You average {s['trades_per_day']:.1f} trades a day and {s['off_plan_rate']:.0%} of trades were outside your plan.",
        "FOMO": f"{s['chase_entry_rate']:.0%} of entries came after a big move with a volume spike.",
        "Ignoring Stop-Losses": f"You moved your stop-loss on {s['sl_modified_rate']:.0%} of trades.",
        "Loss Aversion": f"You hold losing trades {s['loss_hold_vs_win_hold']:.1f}x longer than winners.",
        "Overleveraging": f"Average leverage is {s['avg_leverage']:.1f}x with a max drawdown of {s['max_drawdown_pct']:.0f}%.",
        "Greed": f"On {s['target_held_rate']:.0%} of trades you held past your target and gave profit back.",
        "Panic Selling": f"{s['early_exit_rate']:.0%} of trades were exited early in a sharp move, before your stop.",
        "Lack of Discipline": f"Stop-loss set on only {s['sl_set_rate']:.0%} of trades; {s['off_plan_rate']:.0%} were off-plan.",
        "Confirmation Bias": "You often re-enter the same stock in the same direction right after a loss, against the indicators.",
    }.get(top, "")
    out = [f"Your behaviour score is **{ctx['behaviour_score']}/100**. The biggest pattern I see is **{top}** "
           f"({neg[0]['probability']:.0%} confidence). {evidence}"]
    if len(neg) > 1:
        out.append("Also showing up: " + ", ".join(t["tag"] for t in neg[1:4]) + ".")
    out.append("**Try these tomorrow:**")
    for d in docs[:3]:
        out.append(f"- {d['text']} *[{d['source'].split(' - ')[0]}]*")
    m = ctx["market"]
    out.append(f"\nMarket context: {m['index']} sentiment is {m['sentiment']}, VIX {m['india_vix']}. "
               "Calmer markets are a good time to practise the new rules with smaller size.")
    out.append("\n_Behavioural coaching only, not investment advice._")
    return "\n".join(out)


async def stream_answer(analysis: dict, messages: list[dict]):
    query = messages[-1]["content"] if messages else ""
    ctx, neg, docs = build_context(analysis, query)
    ep = _llm_endpoint()
    if ep is None:
        import asyncio
        for word in fallback_answer(ctx, neg, docs, query).split(" "):
            yield word + " "
            await asyncio.sleep(0.015)
        return
    url, key, model = ep
    payload = {"model": model, "stream": True, "temperature": 0.4,
               "messages": [{"role": "system", "content": SYSTEM + "\n\nTRADER CONTEXT (JSON):\n" + json.dumps(ctx, default=str)}]
               + messages[-10:]}
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", url, json=payload, headers={"Authorization": f"Bearer {key}"}) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                try:
                    delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                except Exception:
                    delta = None
                if delta:
                    yield delta
