# CreditMitra — User Manual

**Project ID:** TIHI-506825 · **Status as of:** 12 October 2026

For two readers: the **analyst** who uploads statements and reads reports
(Part A), and the **operator** who installs and runs the system (Part B).

---

## Part A — Analyst guide

### A1. Processing a statement

1. Open the dashboard (http://localhost:5173 on a laptop install,
   http://localhost:8080 with Docker).
2. Drag a bank statement **PDF** onto the upload area, or click to choose one.
   Digital (text) PDFs work; scanned image-only statements are not supported.
3. Press **Process**. A progress indicator shows while the statement is read,
   payees are identified and the score is computed — typically under a minute
   for a two-month statement and about three minutes for six months.
4. When it finishes, three tabs appear.

The uploaded PDF is deleted from the server as soon as processing ends.

### A2. Lifestyle Credit Report tab

- **Score (300–900) and band** — Poor (< 580), Fair (580–669), Good (670–739),
  Very Good (740–799), Excellent (800+).
- **Archetype and narrative** — the dominant financial pattern:

  | Archetype | Meaning |
  |---|---|
  | Salaried Saver | Regular salary, sustained commitments (SIP, insurance, rent), low leverage and risk |
  | Gig Hustler | Income from recurring gig-platform payouts rather than one employer, fully digital |
  | BNPL-Heavy Spender | Several buy-now-pay-later apps carry a meaningful share of spend |
  | Gambler | Significant share of spend on gambling or crypto platforms |
  | Aspirational Overspender | High discretionary spend with a thin balance buffer |
  | Cash-Reliant Informal | Heavy cash withdrawals, little digital spending to assess |
  | Overextended | Payments bouncing on a recurring basis |
  | Balanced | No single pattern dominates |

- **Lifestyle indices (radar, 0–100)** — L1 Essential Stability, L2
  Aspirational spend, L3 Digital Maturity, L4 Commitment, L5 Leverage
  (higher = less leverage), L6 Risk Appetite (higher = less risk).
- **Factors** — up to six signed reasons, e.g. "Regular salary detected
  (+60)", "2 bounced/returned payments (−70)". These are why the score is
  what it is.

If the tab says **"Credit report unavailable"**, the transactions were read
but there was not enough dated, amount-bearing data to score. The other tabs
are still accurate.

### A3. Spend Analytics tab

Monthly inflows and outflows, and spend by category (groceries, food delivery,
shopping, investments, BNPL, …). Payments to people are shown separately from
merchant spend.

### A4. Transaction Ledger tab

Every extracted row with its date, narration, amount, the identified payee,
**person/merchant** label, category, and risk badges (gambling, BNPL, crypto).
Searchable and sortable.

### A5. How to read the score responsibly

- The score is an **explainable summary of behaviour in this statement**, not a
  probability of default. Its weights are expert-set and have not yet been
  validated against loan outcomes. Use it alongside, not instead of, normal
  underwriting.
- **Healthcare spending never affects the score.**
- Merchants the system does not recognise appear as category *other*; they
  neither help nor hurt the score.
- A statement covering more months gives more reliable indices (several
  depend on month-to-month consistency).

---

## Part B — Operator guide

### B1. Requirements

- Windows 10/11, macOS or Linux; 16 GB RAM recommended (the engine uses ~3 GB).
- Python ≥ 3.10, Node.js ≥ 22, [Ollama](https://ollama.com); a GPU is optional
  but speeds up payee extraction.
- The engine and dashboard repositories side by side.

### B2. Install (laptop)

Full steps: `CreditMitra-engine/docs/RUNNING.md`. In short:

```powershell
cd CreditMitra-engine
python -m venv venv
venv\Scripts\pip install -e ".[dev]"
# copy classifier models into data\models (see data\models\README.md), then:
venv\Scripts\pip install onnx
venv\Scripts\python scripts\quantize_classifier.py build      # smaller classifier
powershell -File scripts\build_payee_model.ps1 -MergedDir <payee-merged> -LlamaCpp <llama.cpp>
cd ..\CreditMitra-dashboard
npm install
```

### B3. Start and stop

```powershell
cd CreditMitra-engine
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
```

This starts Ollama if needed, warns about anything missing, and opens the
engine (port 8000) and dashboard (port 5173) in their own windows. Close those
windows to stop. Models keep loading for 1–2 minutes after start.

### B4. Docker

```bash
cd CreditMitra-engine/deploy
cp .env.example .env        # set MERCHANT_API_KEY
docker compose up --build -d
```

Dashboard at http://localhost:8080. Volumes, privacy notes and the optional
containerised Ollama: `CreditMitra-engine/docs/DEPLOY.md`.

### B5. Checking it works

```powershell
python scripts\make_sample_statements.py
venv\Scripts\python scripts\e2e_check.py data\sample_statements\*.pdf
```

Four synthetic statements in HDFC, SBI, ICICI and Axis layouts should complete
as Gig Hustler, Salaried Saver, BNPL-Heavy Spender and Gambler.

### B6. Merchant curation (merchant server)

- `GET /api/v1/merchants/unresolved` — names nobody recognised, with how
  many statements they appeared in.
- `POST /api/v1/merchants/propose` — run the curation job (needs
  `LANGSEARCH_API_KEY` and `GEMINI_API_KEY`; without them nothing is proposed).
  On the Gemini free tier each model has a small daily allowance (about 20
  requests for the larger flash models, ~4 per merchant); the default
  `GEMINI_MODEL=gemini-3.5-flash-lite` goes further. When the allowance runs
  out the job stops and the remaining names wait for the next run.
- `GET /api/v1/merchants/candidates?status=held` — what was held back and why.
- `POST /api/v1/merchants/reverse/{name}` — undo an automatic promotion.

All require the `X-API-Key` header.

### B7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Every payee empty | Ollama not running or `payee-lora` model missing (`ollama list`) |
| People labelled as merchants, log mentions "heuristic fallback" | Classifier files missing from `data/models/` |
| Task fails "No transactions could be extracted… headers seen: …" | Unsupported statement layout; the headers in the message show what was found |
| Task fails "The engine stopped before this statement finished" | The engine restarted mid-task; upload the statement again |
| Dashboard network error | Engine not on port 8000, or the dashboard's origin missing from `CORS_ORIGINS` |
| Engine crashes while loading models | Low memory; build the FP16 classifier and close other heavy applications |
