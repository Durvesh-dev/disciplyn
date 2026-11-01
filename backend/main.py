"""
Disciplyn API (FastAPI)

  cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000
  Docs: http://localhost:8000/docs
"""
import io
import json
import os
import sys

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
sys.path.insert(0, ML_DIR)
from predict import analyze, load_models  # noqa: E402
from rag import stream_answer  # noqa: E402

app = FastAPI(title="Disciplyn API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
SAMPLES = os.path.join(ML_DIR, "samples")
PERSONAS = {"impulsive": "Impulsive trader", "fearful": "Fearful trader", "disciplined": "Disciplined trader"}
load_models()


def _persona(pid):
    if pid not in PERSONAS:
        raise HTTPException(404, "unknown persona")
    tr = pd.read_csv(f"{SAMPLES}/persona_{pid}_trades.csv")
    q = pd.read_csv(f"{SAMPLES}/persona_{pid}_questionnaire.csv")
    return tr, q


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


class ChatReq(BaseModel):
    persona_id: str | None = None
    analysis: dict | None = None
    messages: list[dict]


@app.post("/api/chat")
async def chat(req: ChatReq):
    analysis = req.analysis or analyze(*_persona(req.persona_id or "impulsive"))
    return StreamingResponse(stream_answer(analysis, req.messages), media_type="text/plain; charset=utf-8")
