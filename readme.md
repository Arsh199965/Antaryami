# BatterySmart Voice AI Platform

End-to-end voice AI system for BatterySmart support: real-time speech-to-text (STT), LLM response generation, text-to-speech (TTS), Twilio voice streaming, and supporting data pipelines for call analysis and retrieval.

## What’s in this repo

| Area | Description |
| --- | --- |
| `server/` | FastAPI backend for the live voice AI system (STT → LLM → TTS), Twilio voice/media streams, batteries, users, call transcripts, locations, and agent WebSocket connections. |
| `data-normalization/` | Offline pipeline to normalize call recordings from Google Sheets/Drive into structured JSON with diarization, transcription, and Claude analysis. |
| `data-ingestion/` | Offline pipeline to embed normalized call JSON and upsert vectors into Pinecone for retrieval in the server’s `getCallInsights` tool. |
| `client/` | Next.js frontend (create-next-app scaffold). |
| `server/sample_data/` | Excel-based seeding instructions to populate MongoDB collections. |

## High-level architecture

```
Call audio (Twilio / in-app)                Normalized JSON output
          │                                              ▲
          ▼                                              │
  FastAPI server (STT → LLM → TTS)        data-normalization pipeline
          │                                              │
          ▼                                              ▼
   MongoDB + Pinecone retrieval   ←   data-ingestion (embeddings)
```

## Server (FastAPI)

The FastAPI app provides:

- **Real-time voice stack**: STT WebSocket, LLM processing, and TTS WebSocket.
- **Twilio integration**: voice webhook + media stream WebSocket.
- **Core APIs**: batteries, users, call transcripts, locations, agents.
- **MongoDB** persistence and Pinecone-based retrieval for call insights.

### Key endpoints

The root endpoint (`/`) returns a quick map of available routes, including:

- `ws://localhost:8000/stt/ws/audio`
- `GET /stt/health`
- `POST /api/text/process`
- `GET /api/text/health`
- `ws://localhost:8000/tts/ws`
- `GET /tts/health`
- `POST /twilio/voice`
- `ws://localhost:8000/twilio/media`
- `PUT /api/batteries/{battery_id}`
- `GET /api/batteries/{battery_id}`
- `GET /api/calls/transcripts`
- `GET /api/calls/transcripts/{call_id}`
- `GET /api/calls/analytics/summary`
- `ws://localhost:8000/agent/ws/connect`
- `GET /agent/queue/status`
- `GET /agent/health`
- `GET /docs` (Swagger UI)

### Server setup

1. **Create environment file**
   ```bash
   cp server/.env.example server/.env
   ```

2. **Set required values** (examples from `.env.example`):
   - `SONIOX_API_KEY` (STT)
   - `CARTESIA_API_KEY` (TTS)
   - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WEBSOCKET_URL`
   - `MONGODB_URL`
   - `PINECONE_API_KEY` + Pinecone index names
   - AWS credentials if using Bedrock embeddings

3. **Run the server**
   ```bash
   cd server
   uv run main.py
   ```

## Data normalization pipeline

Processes call recordings from public Google Sheets/Drive into structured JSON with:
- Speaker diarization + transcription (Soniox)
- Claude (Bedrock) analysis for summary, problems, solutions, and sentiment

### Quick start

```bash
cd data-normalization
npm install
cp .env.example .env
npm start
```

See `data-normalization/README.md` for all flags (date range, row, dry-run, verbose) and output format.

## Data ingestion pipeline (Pinecone)

Embeds normalized JSON output and upserts into Pinecone indexes used by the server’s `getCallInsights` tool.

### Quick start

```bash
cd data-ingestion
npm install
cp .env.example .env
npm run ingest:dir
```

See `data-ingestion/README.md` for additional flags and index behavior.

## Client (Next.js)

The frontend is a standard Next.js app scaffold. To run locally:

```bash
cd client
npm install
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

## Sample data seeding

The `server/sample_data` folder contains Excel-to-MongoDB seeding instructions and a mapping of Excel sheets to collections.

```bash
cd server
uv run python -m scripts.seed_from_excel
```

See `server/sample_data/README.md` for details and seed order.

## Suggested local workflow

1. Run **data-normalization** to generate `output/*.json` from call recordings.
2. Run **data-ingestion** to embed and upsert JSONs into Pinecone.
3. Start the **server** to power real-time voice AI and retrieval.
4. (Optional) Start the **client** for UI interactions.

## Repository structure

```
client/              # Next.js frontend
server/              # FastAPI backend
  sample_data/       # MongoDB seed mapping
  scripts/           # Seeding tools
  routers/           # API route handlers
  services/          # Voice / LLM / integrations
  db/                # MongoDB connections + indexes
  modules/           # Config & utilities

data-normalization/  # Call recording normalization
  output/            # Generated JSONs

data-ingestion/      # Pinecone ingestion pipeline
```

## Notes

- The backend expects MongoDB and external APIs (Soniox, Cartesia, Twilio, Bedrock) to be configured via environment variables.
- Pinecone indexes are created automatically by the ingestion pipeline if missing.
- For Twilio Media Streams, use an internet-reachable WebSocket URL (e.g., via ngrok).
