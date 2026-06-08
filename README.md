# Diagram Analyser — ML-Based Quality Evaluation for Diagrams

Master's Final Year Project. Automatically analyses the visual quality of diagrams using a rule-based metric engine, computer vision, and LLM synthesis, surfaced through a React web app.

---

## Architecture

```
src/              Python backend (FastAPI)
Diagram Analyser App/   React + TypeScript frontend (Vite)
```

**Backend** — `src/`

| File | Role |
|---|---|
| `api.py` | FastAPI REST server, 8 endpoints, asyncio lock for EasyOCR |
| `suggestion_engine.py` | 13-metric engine + LLM synthesis (gpt-4o-mini vision) |
| `threshold_manager.py` | Per-metric thresholds in CSV, EMA online learning (α=0.10) |
| `chat_router.py` | 8-intent deterministic router with LLM fallback |
| `analyze_diagram.py` | CLI entry point |
| `features/` | One module per metric (brevity, color harmony, whitespace, etc.) |
| `detection/` | Shape & label detection via OpenCV + EasyOCR |

**Frontend** — `Diagram Analyser App/src/app/`

| Component | Role |
|---|---|
| `AnalysisTab` | Upload diagram, view composite score + per-metric cards with LLM panels |
| `MetricsDragBoard` | Drag-and-drop metric prioritisation |
| `VersionHistoryTab` | Compare diagram versions over time |
| `ChatTab` | Conversational interface backed by `chat_router.py` |

---

## Metrics

13 rule-based metrics covering:

- **Text** — label readability, label overlap, label contrast, brevity score, font hierarchy
- **Layout** — whitespace distribution, edge margin ratio, orientation consistency, layout structure score
- **Structure** — isolated/disconnected boxes, cognitive chunk density, color harmony, label area ratio

Each metric has a threshold-guided severity level (good / warning / critical). Thresholds adapt over sessions via EMA.

---

## Setup

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
uvicorn src.api:app --reload
```

### Frontend

```bash
cd "Diagram Analyser App"
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`, proxies API to `http://localhost:8000`.

---

## Deployment (public URL via ngrok)

Run each command in a separate terminal.

**Terminal 1 — start the backend:**

```bash
python -m uvicorn src.api:app --port 8001 --host 0.0.0.0
```

**Terminal 2 — expose it publicly:**

```bash
ngrok http 8001 --request-header-add "ngrok-skip-browser-warning:true"
```

Copy the `https://` forwarding URL from the ngrok output and set it as the API base URL in the frontend (e.g. in `.env` or `src/config.ts`).

---

## Stack

- **Backend:** Python 3.11, FastAPI, OpenCV, EasyOCR, PyTorch, scikit-learn, OpenAI SDK
- **Frontend:** React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui
