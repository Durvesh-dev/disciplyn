"use client";
/**
 * Disciplyn demo dashboard - drop into the Next.js App Router project as app/dashboard/page.tsx
 * Needs: recharts (already in the stack) + Tailwind. Set NEXT_PUBLIC_API_URL (default http://localhost:8000).
 */
import { useEffect, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell,
} from "recharts";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Tag = { tag: string; polarity: "negative" | "positive"; probability: number; present: boolean; model: string };
type Analysis = {
  behaviour_score: number;
  tags: Tag[];
  stats: Record<string, number>;
  equity_curve?: { date: string; cum_pnl: number }[];
  has_questionnaire: boolean;
};
type Msg = { role: "user" | "assistant"; content: string };

const PERSONAS = [
  { id: "impulsive", name: "Impulsive trader" },
  { id: "fearful", name: "Fearful trader" },
  { id: "disciplined", name: "Disciplined trader" },
];

export default function Dashboard() {
  const [persona, setPersona] = useState("impulsive");
  const [data, setData] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLoading(true);
    setMsgs([]);
    fetch(`${API}/api/analyze/${persona}`)
      .then((r) => r.json())
      .then(setData)
      .finally(() => setLoading(false));
  }, [persona]);

  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [msgs]);

  async function send(text: string) {
    if (!text.trim() || !data) return;
    const next: Msg[] = [...msgs, { role: "user", content: text }];
    setMsgs([...next, { role: "assistant", content: "" }]);
    setInput("");
    setStreaming(true);
    const res = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ analysis: data, messages: next }),
    });
    const reader = res.body!.getReader();
    const dec = new TextDecoder();
    let acc = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      acc += dec.decode(value, { stream: true });
      setMsgs([...next, { role: "assistant", content: acc }]);
    }
    setStreaming(false);
  }

  const neg = data?.tags.filter((t) => t.polarity === "negative").sort((a, b) => b.probability - a.probability) ?? [];
  const pos = data?.tags.filter((t) => t.polarity === "positive" && t.present) ?? [];
  const chart = neg.map((t) => ({ name: t.tag, value: Math.round(t.probability * 100) }));
  const scoreColor = (s: number) => (s >= 70 ? "text-emerald-600" : s >= 40 ? "text-amber-600" : "text-rose-600");

  return (
    <main className="mx-auto max-w-6xl p-6 space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Your behaviour report</h1>
          <p className="text-sm text-neutral-500">14 ML models · trade book + market context + questionnaire</p>
        </div>
        <div className="flex gap-2">
          {PERSONAS.map((p) => (
            <button key={p.id} onClick={() => setPersona(p.id)}
              className={`rounded-full px-4 py-1.5 text-sm border ${persona === p.id ? "bg-neutral-900 text-white" : "bg-white"}`}>
              {p.name}
            </button>
          ))}
        </div>
      </header>

      {loading || !data ? (
        <div className="h-64 animate-pulse rounded-2xl bg-neutral-100" />
      ) : (
        <>
          <section className="grid gap-4 md:grid-cols-4">
            <div className="rounded-2xl border p-5">
              <p className="text-sm text-neutral-500">Behaviour score</p>
              <p className={`text-5xl font-semibold ${scoreColor(data.behaviour_score)}`}>{data.behaviour_score}</p>
              <p className="text-xs text-neutral-400">out of 100</p>
            </div>
            <Stat label="Trades analysed" value={data.stats.n_trades} />
            <Stat label="Win rate" value={`${Math.round(data.stats.win_rate * 100)}%`} />
            <Stat label="Net P&L" value={`₹${Math.round(data.stats.net_pnl_inr).toLocaleString("en-IN")}`} />
          </section>

          <section className="grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl border p-5">
              <h2 className="mb-3 font-medium">Negative behaviour risk (%)</h2>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={chart} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                    {chart.map((c) => (
                      <Cell key={c.name} fill={c.value > 60 ? "#e11d48" : c.value > 40 ? "#f59e0b" : "#10b981"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="rounded-2xl border p-5 space-y-4">
              <div>
                <h2 className="mb-2 font-medium">Strengths</h2>
                <div className="flex flex-wrap gap-2">
                  {pos.length ? pos.map((t) => (
                    <span key={t.tag} className="rounded-full bg-emerald-50 px-3 py-1 text-sm text-emerald-700">
                      {t.tag} · {Math.round(t.probability * 100)}%
                    </span>
                  )) : <span className="text-sm text-neutral-500">None detected yet</span>}
                </div>
              </div>
              <div>
                <h2 className="mb-2 font-medium">Detected issues</h2>
                <div className="flex flex-wrap gap-2">
                  {neg.filter((t) => t.present).map((t) => (
                    <span key={t.tag} className="rounded-full bg-rose-50 px-3 py-1 text-sm text-rose-700">
                      {t.tag} · {Math.round(t.probability * 100)}%
                    </span>
                  ))}
                </div>
              </div>
              {data.equity_curve && (
                <div>
                  <h2 className="mb-2 font-medium">Cumulative P&L</h2>
                  <ResponsiveContainer width="100%" height={140}>
                    <LineChart data={data.equity_curve}>
                      <XAxis dataKey="date" hide />
                      <YAxis hide />
                      <Tooltip />
                      <Line dataKey="cum_pnl" dot={false} strokeWidth={2} stroke="#0f172a" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </section>

          <section className="rounded-2xl border p-5">
            <h2 className="mb-3 font-medium">Ask your coach</h2>
            <div className="h-80 overflow-y-auto space-y-3 rounded-xl bg-neutral-50 p-4">
              {msgs.length === 0 && (
                <div className="flex flex-wrap gap-2">
                  {["Why do I keep losing after a bad trade?", "What should I fix first?", "Give me rules for tomorrow"].map((s) => (
                    <button key={s} onClick={() => send(s)} className="rounded-full border bg-white px-3 py-1 text-sm">{s}</button>
                  ))}
                </div>
              )}
              {msgs.map((m, i) => (
                <div key={i} className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm ${m.role === "user" ? "ml-auto bg-neutral-900 text-white" : "bg-white border"}`}>
                  {m.content || "…"}
                </div>
              ))}
              <div ref={endRef} />
            </div>
            <form onSubmit={(e) => { e.preventDefault(); send(input); }} className="mt-3 flex gap-2">
              <input value={input} onChange={(e) => setInput(e.target.value)} disabled={streaming}
                placeholder="Ask about your trading behaviour…" className="flex-1 rounded-xl border px-4 py-2 text-sm" />
              <button disabled={streaming} className="rounded-xl bg-neutral-900 px-5 py-2 text-sm text-white disabled:opacity-50">Send</button>
            </form>
          </section>
        </>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-2xl border p-5">
      <p className="text-sm text-neutral-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
    </div>
  );
}
