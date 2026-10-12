# CreditMitra — Feature Specification Document

**Project:** CreditMitra — Privacy-Preserving Agentic AI System for Automated Credit Underwriting from Heterogeneous Bank Statements
**Programme:** TIH-IoT Chanakya Fellowship 2026 · Project ID TIHI-506825
**Team:** Disha Kwatra (lead), Gargi Kalia, Shivansh Katiyar · Mentors: Dr. Sandesh Jain, Dr. Somesh Kumar (ABV-IIITM Gwalior)
**Status as of:** 12 October 2026 · TRL 4 (validated in the lab; not yet deployed in a relevant environment)

This document specifies what the system does, feature by feature, with the
acceptance criteria each feature is held to and its current status. Design
rationale lives in the Design Document; how each criterion is tested lives in
the Test Cases document.

---

## 1. Scope

**In scope.** A lender-side tool that takes a borrower's bank statement (PDF),
extracts every transaction, identifies who each payment went to, separates
people from merchants, works out what kind of business each merchant is, and
turns the result into a 300–900 lifestyle credit score with an archetype and
human-readable reasons — while keeping the statement's contents on the
machine that processes it.

**Out of scope (current phase).** Bureau data (CIBIL) integration; loan
origination workflows; multi-tenant hosting for many lenders (see §4, F11);
scanned/image-only statements (OCR is disabled for memory reasons).

**Users.** A credit analyst or loan officer at a bank/NBFC who uploads a
statement and reads the report; an operator who deploys and maintains the
system; a curator who reviews automatically learned merchants.

---

## 2. Proposal deliverables → features

| Proposal deliverable | Features |
|---|---|
| D1 Robust extraction from diverse bank statements | F1, F2 |
| D2 Payee extraction model for varied narration formats | F3 |
| D3 High-recall merchant / non-merchant classifier | F4 |
| D4 Credit-risk scoring made dynamic with merchant information | F5, F6, F7, F8 |
| D5 Privacy across the pipeline | F9 (and privacy criteria in every feature) |
| D6 Scalable GUI application | F10, F11 |

---

## 3. Features

Status key: **Done** (implemented and tested) · **Partial** (implemented, with a
stated gap) · **Planned**.

### F1 — Statement upload and asynchronous processing — *Done*

- **Input:** one PDF via `POST /api/v1/statements/process` (multipart field `pdf`).
- **Output:** `{task_id, status}` immediately; the pipeline runs in the background.
  Progress via `GET /status/{task_id}` (`pending → processing → completed | failed`);
  the report via `GET /report/{task_id}`.
- **Acceptance criteria**
  - Non-PDF uploads are rejected with HTTP 400.
  - The upload endpoint never blocks on processing.
  - The uploaded file is deleted when processing ends, on success *and* failure.
  - A task interrupted by an engine crash is marked `failed` with a re-upload
    message at the next start, and its upload deleted; it never stays
    `processing` forever.

### F2 — Transaction extraction from heterogeneous layouts — *Partial*

- **Input:** statement PDF. **Output:** rows with `date, particulars, deposits,
  withdrawals, balance` plus parsed `txn_date, amount, direction, balance_val`.
- **Behaviour:** Docling table extraction with a pdfplumber fallback; columns
  are recognised by keyword (e.g. "Withdrawal Amt.", "Transaction Remarks",
  "Balance (INR)"), not by exact header; single-amount layouts with a Dr/Cr
  column are supported; continuation pages and wrapped narrations are joined;
  opening/closing-balance and orphan cheque-reference rows are dropped.
- **Acceptance criteria**
  - Every transaction row of a supported layout is extracted (verified on HDFC,
    SBI, ICICI and Axis layouts — 66/66, 70/70, 68/68, 54/54 rows — and a real
    Canara statement).
  - Indian amount formats (lakh grouping, `Cr/Dr` suffixes, parentheses) and
    day-first date formats parse correctly.
  - An unreadable layout **fails the task** with the headers that were seen,
    rather than completing with zero transactions.
- **Gap:** real statements have been tested from one bank (Canara); the other
  layouts are rendered from synthetic data. A multi-bank real-statement set is
  pending (WP1/WP5).

### F3 — Payee extraction — *Done*

- **Input:** one narration string. **Output:** the payee name.
- **Behaviour:** fine-tuned Qwen2.5-1.5B (QLoRA) served locally by Ollama
  (Q8_0, 1.6 GB). A predicted payee whose words cannot be traced back to the
  narration is treated as invented and replaced by the narration's UPI name
  segment.
- **Acceptance criteria**
  - Exact match ≥ 80% on the held-out validation set (measured: 85.2% on 486
    narrations; Q8_0 build 82.7% on a 150-row subset, identical to FP16).
  - The grounding check rejects no correct payee on the validation set (0/486).
  - The model runs on the processing machine; narrations are never sent to an
    external service.
- **Note:** a differentially private adapter (DP-SGD, ε ≈ 1.99) is trained and
  benchmarked (81.9% exact match). The deployed model is currently the
  standard adapter; switching is a rebuild with the DP checkpoint.

### F4 — Person / merchant classification — *Done*

- **Input:** payee name. **Output:** `person | merchant` with confidence.
- **Behaviour:** Qwen3-Embedding-0.6B (ONNX, FP16) + AttentionMLP; a name made
  entirely of Indian name words is overridden to *person*; dictionary and
  keyword fallback when models are absent; each distinct name classified once
  per statement.
- **Acceptance criteria**
  - Classifier test accuracy ≥ 97% (measured 98.36%, AUC 0.997).
  - Person-name guard: ≥ 95% of held-out person names recognised (measured
    99.3%), ≤ 5% of merchant names flipped (measured 3.4%, nearly all bare
    owner names).
  - FP16 model labels identically to FP32 on a 600-name check.

### F5 — Merchant enrichment — *Done*

- **Input:** merchant names. **Output:** per name `canonical_name, category,
  is_essential, lifestyle_dim, risk_flag, recurring_type` from a fixed
  26-category taxonomy.
- **Tiers (cheapest and most private first):** SQLite cache → curated
  dictionary (239 merchants) → on-device vector index for near misses →
  shared merchant server → optional web-search-grounded LLM.
- **Acceptance criteria**
  - Order-preserving, deduplicated batches; every failure returns the safe
    `other / neutral` default and never blocks the pipeline.
  - `is_essential`, `lifestyle_dim` and `risk_flag` are always derived from
    `category`, never chosen by a model.
  - High-severity categories (gambling, BNPL/lending, crypto, loan EMI) are
    never reached by a prefix that merely starts a longer name, and never by a
    fuzzy match.
  - Vector tier: ≤ 0.5% false matches on local-shop and person names (measured
    0.3% at threshold 0.75, recovering 67% of near-miss brand variants).

### F6 — Self-curating merchant knowledge (merchant server) — *Partial*

- **Behaviour:** names nobody can resolve are logged with frequency and the
  number of *distinct statements* they appeared in. A curation job searches the
  web (LangSearch), asks an LLM twice (with and without the evidence), and adds
  merchants that clear a confidence bar to the dictionary, with provenance.
  Automatic promotions can be listed and reversed.
- **Acceptance criteria**
  - A name is proposed only after appearing in ≥ 3 distinct statements;
    resubmitting one statement does not count again.
  - High-severity categories require ≥ 0.90 confidence **and** agreement of
    both LLM paths.
  - Every promotion records model, confidence, summary and evidence.
  - Fails closed: no API key configured → 503; no LLM key → proposes nothing.
- **Gap:** the LangSearch and Gemini calls have not yet been exercised against
  the live APIs (tested with fakes).

### F7 — Cash-flow features and lifestyle indices — *Done*

- **Output:** `FeatureVector` (income regularity, salary / gig-income
  detection, FOIR, balance-buffer days, essential/discretionary ratios, cash
  withdrawal ratio, bounces, BNPL and gambling shares) and six 0–100 lifestyle
  indices (L1 essential stability, L2 aspirational, L3 digital maturity, L4
  commitment, L5 leverage, L6 risk appetite) plus behavioural texture.
- **Acceptance criteria**
  - **Healthcare is excluded from every ratio and index** (fair lending).
  - Salary is detected for employers that are companies; gig platforms are
    never counted as salary.
  - Only failed debit instructions count as bounces; refunds and reversals never do.

### F8 — Score, band, archetype and reasons — *Done (not predictive-validated)*

- **Output:** `CreditRiskReport` — score 300–900, band (Poor < 580 · Fair · Good
  · Very Good · Excellent ≥ 800), archetype (Salaried Saver, Gig Hustler,
  BNPL-Heavy Spender, Gambler, Aspirational Overspender, Cash-Reliant Informal,
  Overextended, Balanced), up to six signed reasons, one-line narrative.
- **Acceptance criteria**
  - Eight synthetic six-month personas rank and classify as designed.
  - Gig Hustler requires recurring gig-platform payouts; other non-salary
    inflows earn no income points.
  - Scoring failure never fails the task; the report is `null` and the reason is logged.
- **Limitation:** weights are expert-set. The score has **no measured
  predictive validity** against loan outcomes; the incremental-lift
  experiment (WP4) is built and validated on synthetic controls, and awaits
  real outcome data.

### F9 — Privacy controls — *Partial*

- **Criteria met:** statements, narrations, amounts and person names never
  leave the processing machine; only unresolved *merchant* names and an
  opaque statement fingerprint (HMAC under a device-local secret) reach the
  merchant server; external calls (web search, LLM) happen only from the
  server, only for names seen in ≥ 3 distinct statements; uploads deleted
  after processing; crash-interrupted uploads deleted at restart.
- **Gaps:** task results are stored as plain JSON; the engine has no
  authentication (designed as a local tool); the deployed payee model is not
  the DP-trained adapter (see F3).

### F10 — Analyst dashboard — *Done*

- Upload with live progress (polling survives transient failures); report
  tab (score gauge, archetype, radar of the six indices, factor breakdown);
  spend analytics (cash flow, category breakdown); transaction ledger with
  payee type, category and risk badges.
- **Acceptance criteria:** never displays a placeholder score — a missing
  report is shown as "unavailable"; status polling tolerates 5 consecutive
  failed requests and stops immediately when the task no longer exists.

### F11 — Deployment and scalability — *Partial*

- Docker images for engine, merchant server and dashboard; one compose file
  runs the stack with the payee model on host or containerised Ollama.
- **Gap:** not yet deployed to a hosted environment (TRL 5); no multi-tenant
  authentication. Whether the product is a lender-hosted service or a local
  desktop tool is an open decision that determines the authentication design.

---

## 4. Non-functional requirements

| Area | Requirement | Current |
|---|---|---|
| Throughput | A 6-month statement (~200 rows) processed in under 5 minutes on a laptop-class machine | 162 s for 220 rows; 30–42 s for 54–70 rows |
| Memory | Engine fits in 6 GB | ~3 GB with the FP16 classifier |
| Startup | API available immediately; models load in background | `/health` in ~8 s; models warm in 1–2 min |
| Robustness | No stage failure crashes the task; scoring is best-effort | Enforced by tests |
| Explainability | Every score comes with signed reasons and an archetype narrative | Done |
| Auditability | Automatic merchant labels carry provenance and are reversible | Done (merchant server) |
