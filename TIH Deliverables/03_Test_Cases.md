# CreditMitra — Test Cases

**Project ID:** TIHI-506825 · **Status as of:** 12 October 2026

How each feature in the Feature Specification is verified. Almost everything
is an automated test that runs in CI on every pull request; the end-to-end
check and the model evaluations are scripted and repeatable.

---

## 1. Test inventory

| Repository | Suite | Tests | Runs in CI |
|---|---|---:|---|
| Engine | `tests/` (pytest) | 216 | Yes — plus ruff, mypy (strict), Docker build + boot |
| Merchant server | `test_server.py`, `test_proposer.py`, `test_providers.py` | 57 | Yes |
| Dashboard | Vitest | 6 | Yes — plus ESLint, Prettier, tsc, build, Docker build + SPA routing |
| End-to-end | `scripts/e2e_check.py` against a running engine | 5 statements | Manual / scripted |

Engine tests by area:

| File | Tests | Covers |
|---|---:|---|
| `test_statement_layout.py` | 44 | F2 column detection across layouts |
| `test_extraction_fallback.py` | 5 | F2 Docling → pdfplumber fallback, unreadable layouts fail loudly |
| `test_borderless_fallback.py` | 3 | F2 boxed-header, borderless-body statements rebuilt from word positions |
| `test_persona_ranking.py` | 62 | F7/F8 personas: ordering, discrimination, fair lending, salary detection |
| `test_quality_guards.py` | 27 | F3 grounding, F4 person guard, F7 gig income, bounces (incl. UNPAID) and healthcare exclusion, F1 crash recovery |
| `test_dictionary_matching.py` | 19 | F5 dictionary coverage and high-severity safety |
| `test_merchant_vectors.py` | 13 | F5 vector tier |
| `test_enrichment_tiers.py` | 10 | F5 tier order, cache versioning, server as a tier |
| `test_enrichment_websearch_grounding.py` | 4 | F5 web-search tier |
| `test_enrichment_privacy.py` | 3 | F9 outbound payload contents, statement fingerprint |
| `test_incremental_lift.py` | 17 | WP4 measurement harness (null and positive controls) |
| `test_berka_adapter.py` | 3 | WP4 Berka adapter: decision-time cutoff, finished-loans formulation |
| `test_upload_cleanup.py` | 3 | F1 upload deletion, per-payee deduplication |
| `test_health.py` | 3 | API health |

---

## 2. Test cases

Type: **U** unit · **I** integration (API or multi-module) · **E** end-to-end · **M** model evaluation.

### Upload and task lifecycle (F1)

| ID | Case | Expected | Type | Automated by |
|---|---|---|---|---|
| TC-01 | Upload a non-PDF | HTTP 400 | I | `test_health` |
| TC-02 | Task fails during extraction | Status `failed` with message; upload deleted | U | `test_upload_cleanup` |
| TC-03 | Task completes | Status `completed`; upload deleted | U | `test_upload_cleanup` |
| TC-04 | Engine restarts with a task still `processing` | Task marked `failed` with re-upload message; its upload deleted; uploads of finished tasks untouched | U | `test_quality_guards` |

### Extraction (F2)

| ID | Case | Expected | Type | Automated by |
|---|---|---|---|---|
| TC-10 | Headers "Withdrawal Amt.", "Transaction Remarks", "Balance (INR)" | Mapped to canonical columns | U | `test_statement_layout` |
| TC-11 | Single Amount column + Dr/Cr column | Direction and amount correct | U | `test_statement_layout` |
| TC-12 | Continuation page with no header row | Previous layout carried over | U | `test_statement_layout` |
| TC-13 | Narration wrapped onto a second row | Joined into one transaction | U | `test_statement_layout` |
| TC-14 | Opening/closing balance and orphan "Chq:" rows | Dropped | U | `test_statement_layout` |
| TC-15 | Docling crashes or finds nothing | pdfplumber fallback used | U | `test_extraction_fallback` |
| TC-16 | Unrecognisable layout | Task fails naming the headers seen | U | `test_extraction_fallback` |
| TC-17 | HDFC / SBI / ICICI / Axis layouts | Every row extracted | E | `e2e_check.py` (§3) |
| TC-18 | Statement with a boxed header and an unruled body (Canara) | Fallback rebuilds every transaction; matches Docling 220/220 | U/E | `test_borderless_fallback`, real statement |

### Payee extraction (F3)

| ID | Case | Expected | Type | Automated by |
|---|---|---|---|---|
| TC-20 | Payee exact match on validation narrations | ≥ 80% | M | finetune eval (85.2% / 486) |
| TC-21 | Q8_0 build vs F16 | Same exact match | M | 82.7% vs 82.7% on 150 rows |
| TC-22 | Correct payee, incl. expanded truncation ("JIO RECHA" → "Jio Recharge") | Kept | U | `test_quality_guards` |
| TC-23 | Invented payee on a UPI narration ("NYKAA ERE" → "PayViva") | Replaced by the narration's name segment | U | `test_quality_guards` |
| TC-24 | Grounding check vs ground truth | 0 correct payees rejected | M | 0 / 486 |

### Classification (F4)

| ID | Case | Expected | Type | Automated by |
|---|---|---|---|---|
| TC-30 | "Neha", "Rohit Sa", "Ramesh Kumar Sharma" | person | U | `test_quality_guards` |
| TC-31 | "Ganesh Traders", "Sai Medical", "Swiggy", "Axis Bank" | not overridden to person | U | `test_quality_guards` |
| TC-32 | Model says merchant (0.94) for a personal name | Overridden to person | U | `test_quality_guards` |
| TC-33 | Guard on held-out corpora | ≥ 95% persons caught, ≤ 5% merchants flipped | M | 99.3% / 3.4% (`build_person_tokens.py`) |
| TC-34 | FP16 vs FP32 embedding model | No label changes | M | 0 / 600 (`quantize_classifier.py evaluate`) |
| TC-35 | Repeated payees in one statement | Each distinct name classified once | U | `test_upload_cleanup` |

### Enrichment (F5) and merchant server (F6)

| ID | Case | Expected | Type | Automated by |
|---|---|---|---|---|
| TC-40 | Truncated UPI names ("SWIGGY LI", "KREDITBE") | Resolve | U | `test_dictionary_matching` |
| TC-41 | "Navin Kumar", "Simple Mart", "Kuber General Store" | Never BNPL/crypto | U | engine `test_dictionary_matching`, server `test_server` |
| TC-42 | Every dictionary entry | Derived fields match the taxonomy | U | `test_dictionary_matching` |
| TC-43 | Typos ("SWIGY", "NETFLX") and split truncations ("CONNAU GHT PLAZA") | Resolve via vector tier | U | `test_merchant_vectors` |
| TC-44 | Local shops and people | No vector match | U | `test_merchant_vectors` |
| TC-45 | Typo of a high-severity merchant ("DREEM11") | Never fuzzy-matched | U | `test_merchant_vectors` |
| TC-46 | Engine and server resolve the same names | Identical categories | I | 519 probes, 0 differences |
| TC-47 | Dictionary edited | Cached answers from the old version retire | U | `test_enrichment_tiers`, `test_server` |
| TC-48 | One statement resubmitted many times | Counts once toward the proposer's bar | I | `test_proposer` |
| TC-49 | Requests without a statement id | Never count toward the bar | I | `test_proposer` |
| TC-50 | High-severity proposal where the two LLM paths disagree | Held, not promoted | I | `test_proposer` |
| TC-51 | Automatic promotion reversed | Entry removed; hand-curated entries untouched | I | `test_proposer` |
| TC-52 | Server without API key | 503 (fails closed) | I | `test_server` |
| TC-53 | LLM returns 429/5xx, times out, or drops the connection | Retried with backoff | U | `test_providers` |
| TC-54 | LLM daily quota spent mid-run | Run stops; unjudged names stay queued, nothing recorded as an abstention | U/I | `test_providers`, `test_proposer` |
| TC-55 | Live curation on public brand names | Correct categories promoted with evidence; made-up name abstains | E | live run (§3) |

### Scoring (F7, F8)

| ID | Case | Expected | Type | Automated by |
|---|---|---|---|---|
| TC-60 | Eight personas | Each lands on its designed archetype; ordering holds | I | `test_persona_ranking` |
| TC-61 | Heavy healthcare spend added (fair lending) | Score, archetype, every index and feature unchanged | U | `test_quality_guards` |
| TC-62 | Employer classified as a merchant | Salary still detected; Salaried Saver | U | `test_persona_ranking` |
| TC-63 | Non-salary, non-gig inflows (family support) | Not Gig Hustler; no income points | U | `test_quality_guards` |
| TC-64 | Merchant refunds / reversals | Not bounces | U | `test_quality_guards` |
| TC-65 | ECS/NACH return, insufficient funds | Counted as bounces | U | `test_quality_guards` |
| TC-66 | Every index and both scorecard blocks | Can both reward and penalise; no score pinned to a bound | U | `test_persona_ranking` |
| TC-67 | 'NACH UNPAID', 'ECS UNPAID', 'UNPAID CHQ' | Counted as bounces | U | `test_quality_guards` |

### Privacy (F9)

| ID | Case | Expected | Type | Automated by |
|---|---|---|---|---|
| TC-70 | Payload to merchant server | Keys are exactly `merchants` and `statement_id` | U | `test_enrichment_privacy` |
| TC-71 | Statement fingerprint | Stable per file, keyed (not a plain hash) | U | `test_enrichment_privacy` |
| TC-72 | Server ledger | Stores only hashes of statement ids | I | `test_server` |
| TC-73 | Payee model membership inference | AUC ≈ 0.5 | M | research privacy audit (8 MIA variants) |

### Dashboard (F10) and deployment (F11)

| ID | Case | Expected | Type | Automated by |
|---|---|---|---|---|
| TC-80 | Status poll fails twice, then succeeds | No error shown; results displayed | U | `useProcessStatement.test.ts` |
| TC-81 | Five consecutive poll failures | Error shown; polling stops | U | `useProcessStatement.test.ts` |
| TC-82 | Task returns 404 | Polling stops immediately | U | `useProcessStatement.test.ts` |
| TC-83 | Task completes without a report | "Report unavailable" — never a sample score | U/manual | Dashboard |
| TC-84 | Engine image with no models or Ollama | Boots and answers `/health` | I | CI Docker job |
| TC-85 | Dashboard image, client-side route | Serves the app (no 404) | I | CI Docker job |
| TC-86 | Full compose stack | All three services healthy; real statement processed in containers | E | manual (§3) |

---

## 3. End-to-end test

**Procedure** (`CreditMitra-engine/docs/RUNNING.md`):

```powershell
python scripts\make_sample_statements.py
venv\Scripts\python scripts\e2e_check.py data\sample_statements\*.pdf <real statement>.pdf
```

Each statement is uploaded through the real API. The script prints counts only
(never names or narrations), so it is safe on real statements, and exits
non-zero if any statement fails or yields no transactions.

**Latest results**

| Statement | Rows | Merchants resolved | Time | Report |
|---|---:|---:|---:|---|
| HDFC layout (synthetic gig worker) | 70/70 | 58/64 | 40 s | 855 Excellent · Gig Hustler |
| SBI layout (synthetic salaried) | 66/66 | 42/54 | 38 s | 886 Excellent · Salaried Saver |
| ICICI layout (synthetic BNPL user) | 68/68 | 44/50 | 42 s | 513 Poor · BNPL-Heavy Spender |
| Axis layout, Dr/Cr column (synthetic gambler) | 54/54 | 32/42 | 30 s | 391 Poor · Gambler |
| Real Canara statement | 220 | 33/54 | ~185 s | 595 Fair · Balanced |
| Real Canara statement, in Docker | 220 | 34/54 | 115 s | 595 Fair · Balanced |

Re-running the real statement through engine + merchant server confirmed the
privacy gate: 16 unresolved names reached the server, each counted once even
though the statement was submitted twice.

---

## 4. Validation experiments (not pass/fail tests)

- **Persona acceptance suite** — consistency check, not predictive validation:
  the personas were designed by the same team that set the weights.
- **Incremental lift (WP4)** — measures what merchant information adds to a
  cash-flow-only model, with paired bootstrap confidence intervals. Null
  control: no lift manufactured (−0.001 ΔAUC, CI spans 0). Positive control:
  real signal found (+0.276 ΔAUC). PKDD'99 (Berka, real Czech outcomes,
  decision-time cutoff, 620 loans / 63 defaults): cash flow alone AUC 0.897;
  the six transaction-type categories add nothing (−0.026, CI excludes zero).
  Inconclusive for UPI merchants — Berka has none. Pending: a dataset with
  real merchant categories and defaults, then lender data.

## 5. Not yet covered

- Real statements from banks other than Canara.
- Load testing with concurrent uploads.
- Hosted deployment (TRL 5).
