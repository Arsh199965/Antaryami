# Server Latency Reduction Guide

This document summarizes **latency sources** in the text/voice pipeline and **concrete ways to reduce latency**.

## Pipeline Overview

The main latency path is:

1. **REST** `/api/text/process` or **WebSocket** (Twilio/STT) → `LLMService.process()` → `ResponsePipeline.process_text()`
2. **Pipeline steps (2 AI calls max):** LLM with broad tools (intent + tool decision in one call) → optional tool-wait audio → Tool execution → Final LLM response (only if tools were called). Intent is **derived from the first LLM call** (from tool names) for logging; no separate intent-detection round-trip.

Each step adds round-trip or I/O time. Below are the main sources and mitigations.

---

## 1. Two-Call Pipeline (Intent Merged into First Call)

**Where:** `modules/response/response.py`  
**Change:** The pipeline no longer runs a separate intent-detection Bedrock call. The first LLM call uses a broad tool set (`general` intent) so the model both understands the user and decides which tools to call in one round-trip. Intent for logging/analytics is derived from the tool names returned (e.g. `getNearestStation` to `station_query`).

**Result:** 1 AI call when no tools are used; 2 AI calls when tools are used (first = intent + tool decision, second = final response with tool results). Saves ~200–600 ms when tools are used.

**Tool-wait audio:** When the first LLM returns tool calls, an optional callback `on_before_tool_execution` runs before running tools (e.g. play a pre-recorded "please wait" clip so the user hears feedback while tools and the second call run). Twilio and STT routers pass this callback; see `services/greeting_audio.py` for `get_tool_wait_mulaw_8k_chunks()` / `get_tool_wait_float32_44100_chunks()`. Set `TOOL_WAIT_AUDIO_PATH` in `.env` to override the default WAV path.

**Mitigations:**

- **Disable when acceptable:** Set `INTENT_DETECTION_ENABLED=false` in `.env`. The pipeline already skips intent for **streaming** when this is false; the same behavior is now applied to **non-streaming** `process_text()`. Intent defaults to `"general"` and all tools remain available (filtering is just less aggressive).
- **Lighter classifier (future):** Use a small local model or keyword/regex for common intents (e.g. “nearest station”, “battery”, “human”) and only call Bedrock for ambiguous cases.

---

## 2. DB Writes Block the Pipeline

**Where:** `modules/response/response.py` — after intent detection, `update_one` (conversations) and `insert_one` (intent_logs) are awaited before proceeding to the LLM.

**Cost:** ~10–50 ms per request (MongoDB round-trip).

**Mitigation:**

- **Fire-and-forget:** Run these writes in a background task (`asyncio.create_task`) so the pipeline continues to the LLM without waiting. Errors are logged but do not block the response. Implemented in this repo.

---

## 3. Tool Execution Is Sequential

**Where:** `modules/response/response.py` — tool calls are executed in a `for` loop, one after another.

**Cost:** If the LLM requests 2–3 tools (e.g. getUserInfo + getNearestStation), total time = sum of each tool’s latency.

**Mitigation:**

- **Parallelize independent tools:** When the LLM returns multiple tool calls, run them with `asyncio.gather()`. They are independent once the LLM has decided. Implemented in this repo.

---

## 4. MongoDB Indexes Disabled

**Where:** `main.py` — `await create_indexes(db)` is commented out.

**Cost:** Without indexes, `find_one({ "user_id": ... })`, `find({ "session_id": ... })`, and geo queries on stations can be slow on larger datasets (full collection scans).

**Mitigation:**

- **Enable indexes:** Uncomment `create_indexes(db)` in the app lifespan. Ensure `db/indexes.py` defines indexes for:
  - `users`: `user_id`, `phone_number` (for Twilio user lookup)
  - `stations`: `station_id`, `location` (2dsphere for nearest-station)
  - `conversations`, `intent_logs`, `subscriptions`, etc. as in `indexes.py`
- **User lookup:** Add an index on `users.phone_number` (or normalized field) to speed up `lookup_user_by_phone()` in Twilio flow.

---

## 5. getNearestStation Loads All Stations

**Where:** `modules/response/tools/service_center.py` — `db.stations.find(query)` then `to_list(length=None)` loads every station; distance is computed in Python.

**Cost:** With many stations, this is slow and memory-heavy.

**Mitigation:**

- **Use geo index:** Rely on MongoDB’s 2dsphere index and `$geoNear` (or `find` with `$near`) so the database returns only nearby stations. This requires `create_indexes` to create the `location` 2dsphere index. Then refactor the tool to use a geo query instead of loading all stations.

---

## 6. getUserInfo Multiple Sequential Reads

**Where:** `modules/response/tools/user_info.py` — fetches user, then subscriptions (cursor), then vehicle, then battery, then issues.

**Cost:** Several round-trips in sequence.

**Mitigation:**

- **Parallelize after user:** Once `user` is loaded, run `subscriptions`, `vehicle` (if vehicle_id), and `battery` (if battery_id) in parallel with `asyncio.gather()`. Keep the final “issues” read after battery if it depends on battery doc.

---

## 7. Call Insights Tool Uses Sync (Blocking) Code

**Where:** `modules/response/tools/call_insights.py` — `_get_embedding()` (Bedrock embeddings) and `_query_index()` (Pinecone) are synchronous and block the event loop.

**Cost:** Can add hundreds of ms and block other requests.

**Mitigation:**

- **Offload to thread pool:** Run embedding and Pinecone queries in `asyncio.to_thread()` so they don’t block the async event loop. Implemented in this repo (optional, can be enabled when needed).

---

## 8. User Lookup by Phone (Twilio)

**Where:** `services/user_lookup.py` — multiple `find_one` attempts (exact, regex, variants) to resolve caller phone to user_id.

**Cost:** Several DB round-trips per Twilio call start.

**Mitigation:**

- **Index:** Add index on `users.phone_number` (and optionally a normalized field) so each lookup is a single indexed find.
- **Cache:** For repeated callers, cache `phone → user_id` in memory (e.g. TTL cache) to avoid DB on every utterance.

---

## 9. Logging and Observability

**Where:** Many `logger.info()` calls along the pipeline (response.py, llm_client.py, tools).

**Cost:** Minor; can add a few ms if logging is synchronous or writes to slow sinks.

**Mitigation:**

- In production, set log level to `WARNING` or `ERROR` for hot paths, or use async logging.
- Add optional timing spans (e.g. one log line with durations for intent, LLM, tools, final LLM) to measure impact of changes.

---

## Summary of Implemented Changes

| Change | File(s) | Effect |
|--------|--------|--------|
| Two-call pipeline (intent merged) | `modules/response/response.py`, `intent_tools_mapping.py` | Intent + tool decision in one LLM call; intent derived from tool names; 2 AI calls max |
| Tool-wait audio hook + loader | `modules/response/response.py`, `services/greeting_audio.py` | `on_before_tool_execution` callback; `get_tool_wait_*` chunks; `TOOL_WAIT_AUDIO_PATH` env |
| Tool-wait in Twilio/STT | `routers/twilio.py`, `routers/stt.py` | Play tool-wait WAV when tools will run |
| Enable MongoDB indexes on startup | `main.py`, `db/indexes.py` | Faster DB queries; add `phone_number` for Twilio lookup |
| DB persistence in background | `modules/response/response.py` | Don’t wait for conversation/intent_log writes before LLM |
| Parallel tool execution | `modules/response/response.py` | Multiple tools run concurrently |
| Call insights in thread pool | `modules/response/tools/call_insights.py` | Avoid blocking event loop on embedding + Pinecone |

---

## Quick Wins Checklist

- [x] Enable `create_indexes(db)` and add `users.phone_number` index
- [x] Fire-and-forget DB persistence for conversation/intent_log
- [x] Parallelize tool execution with `asyncio.gather`
- [x] Two-call pipeline: intent merged into first LLM call; intent derived from tool names
- [x] Run call_insights sync code in `asyncio.to_thread`
- [ ] Refactor getNearestStation to use MongoDB geo query ($geoNear / 2dsphere)
- [ ] Parallelize getUserInfo sub-queries (vehicle, battery, subscriptions)
- [ ] Optional: in-memory cache for phone → user_id in Twilio flow
- [x] Tool-wait audio when tools will run (Twilio + STT); optional `TOOL_WAIT_AUDIO_PATH`
