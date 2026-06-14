# अंतर्यामी · Antaryami

> *The inner knower.* A voice AI agent that speaks to EV battery-swap customers — in Hindi, English, or Hinglish — over a phone call, and actually knows what to do.

---

## What it is

A real-time voice pipeline that lets battery-swap service customers call a number, speak naturally (in any mix of Hindi/English), and get instant help — account info, nearest swap station, battery health, complaint registration — all handled by an AI that sounds like a person and responds in under a second.

**The hard part:** Phone call → 8kHz Twilio audio → STT → LLM with live tool calls → TTS → back to caller, all in a streaming loop with sub-second perceived latency.

---

## How it works

```
Phone call (Twilio)
  │
  ▼
STT  ──  Sarvam AI  (Hindi/English, streaming)
  │
  ▼
LLM  ──  AWS Bedrock  (tool-calling, intent detection)
  │         │
  │         └─ Tools: user profile · battery health · nearest station ·
  │                   swap history · complaint logger · call insights (RAG)
  │
  ▼
Sentence buffer  ──  fires TTS per sentence, not full response
  │
  ▼
TTS  ──  Sarvam bulbul:v3  (connection-pooled, resampled to 8kHz)
  │
  ▼
Back to caller
```

---

## What makes it fast

| Optimization | Effect |
|---|---|
| Sentence-level LLM→TTS streaming | TTS starts before LLM finishes |
| TTS connection pool (3 persistent WSs) | −200ms connection overhead per call |
| Parallel tool execution (`asyncio.gather`) | Multiple DB calls run concurrently |
| Fire-and-forget DB writes | LLM doesn't wait for conversation logging |
| MongoDB 2dsphere indexes | O(1) nearest-station geo lookup |

---

## What it knows

- **User account** — profile, subscription, vehicle
- **Battery** — health status, issue reporting, safety escalation
- **Stations** — nearest swap point via geocoded address (phone callers describe their location verbally)
- **History** — last swap attempt, last service center visit
- **Call insights** — RAG over past calls via Pinecone + Bedrock embeddings; surfaces similar cases and policy snippets
- **Human handoff** — escalates when needed

---

## Stack

| Layer | Tech |
|---|---|
| Voice / Telephony | Twilio Media Streams |
| STT | Sarvam AI (async streaming) |
| LLM | AWS Bedrock (Claude / Titan) |
| TTS | Sarvam bulbul:v3 |
| Vector search | Pinecone + Bedrock embeddings |
| Backend | FastAPI + Python 3.13 |
| Database | MongoDB |
| Frontend | Next.js 16 · React 19 · Three.js |

---

## Languages

Hindi (देवनागरी) · English · Hinglish — detected per utterance, matched in response.

---

## Run it

```bash
# Server
cd server
uv sync
cp .env.example .env  # fill API keys
uvicorn main:app --reload

# Client
cd client
npm install
npm run dev
```

Required keys: `SARVAM_API_KEY`, `TWILIO_*`, `AWS_*` (Bedrock), `PINECONE_*`, `MONGODB_URI`.
