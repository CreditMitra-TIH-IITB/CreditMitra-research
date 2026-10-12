# CreditMitra — Reusable Assets

**Project ID:** TIHI-506825 · **Status as of:** 12 October 2026

Components other projects can take as they are. Each entry gives its location
(all repositories under `github.com/CreditMitra-TIH-IITB`), what it needs, and
what has been measured about it. Supersedes the earlier
`reusable_artifacts_summary.md`, whose paths and statuses had drifted.

---

## 1. Models

| Asset | Location | Reuse notes | Measured |
|---|---|---|---|
| Payee extraction model (Qwen2.5-1.5B, QLoRA) | `CreditMitra-finetune` (training); served via `CreditMitra-engine/scripts/build_payee_model.ps1` | Extracts the counterparty from any Indian bank narration (UPI, IMPS, NEFT, RTGS, POS). Runs locally in Ollama, Q8_0, 1.6 GB | 85.2% exact match (486 val); Q8_0 = F16 on a 150-row check |
| DP payee adapter (DP-SGD, ε ≈ 1.99) | `CreditMitra-finetune` | Same task with a formal privacy guarantee for training data | 81.9% exact match; MIA AUC ≈ 0.51 |
| Person / merchant classifier | `CreditMitra-Merchant-Classification`; FP16 build via `CreditMitra-engine/scripts/quantize_classifier.py` | Names → person or merchant. ONNX, CPU-only, no PyTorch needed | 98.36% test accuracy, AUC 0.997; FP16 identical to FP32 on 600 names |
| Person-name guard word list | `CreditMitra-engine/data/person_name_tokens.txt` (+ `scripts/build_person_tokens.py`) | Drop-in check "is this string just an Indian personal name?" | 99.3% persons caught, 3.4% merchants flipped (held out) |

## 2. Data and knowledge

| Asset | Location | Reuse notes |
|---|---|---|
| Indian merchant dictionary (239 merchants, ~490 aliases incl. UPI truncations) | `CreditMitra-engine/data/india_merchants.json` (synced to the merchant server) | Category, essential/aspirational, risk flag, recurrence per merchant; every entry consistent with the taxonomy (test-enforced) |
| 26-category spending taxonomy | `CreditMitra-engine/docs/taxonomy.md` | Category → index mapping with the fair-lending rule (healthcare excluded) |
| Synthetic Indian narration dataset | `CreditMitra-Synthetic-Transaction-Generator/data` | UPI/IMPS/NEFT/RTGS/POS narrations with payee labels |
| Person vs merchant name corpus (~165k names) | `CreditMitra-Merchant-Classification/data` | Labelled, regionally varied, with truncation and noise augmentation |
| Eight six-month financial personas | `CreditMitra-engine/tests/fixtures/personas` | Synthetic statements that each stress one credit-risk axis |
| Synthetic multi-bank statements | `CreditMitra-engine/scripts/make_sample_statements.py` | Renders HDFC, SBI, ICICI, Axis layouts from personas — layout testing without real customer data |

## 3. Services and libraries

| Asset | Location | Reuse notes |
|---|---|---|
| Statement extraction with layout detection | `CreditMitra-engine/app/services/extraction.py`, `statement_layout.py`, `parsing.py` | Bank-agnostic column detection by keyword; Indian amount/date parsing; fails loudly on unknown layouts |
| Merchant enrichment tiers | `CreditMitra-engine/app/services/merchant_enrichment.py`, `merchant_vectors.py` | Cache → dictionary → on-device vector index → server → optional web search |
| Merchant server | `CreditMitra-merchant-server` | Standalone FastAPI service: shared dictionary, curation ledger counting distinct statements, LLM + web-search proposer with severity tiers and reversible, auditable promotions |
| Lifestyle credit scorer | `CreditMitra-engine/app/services/feature_engineering.py`, `lifestyle_profile.py`, `archetype.py`, `credit_scorer.py` | Explainable 300–900 score, six indices, eight archetypes, signed reasons; weights in one config file |
| Analyst dashboard | `CreditMitra-dashboard` | React 19 UI for upload, report, analytics and ledger; Docker image |

## 4. Tooling and research frameworks

| Asset | Location | Reuse notes |
|---|---|---|
| Synthetic narration generator CLI | `CreditMitra-Synthetic-Transaction-Generator` | Rate-limited LLM generation, resumable, deduplicated, auto-labelled |
| DP-QLoRA training notebook | `CreditMitra-finetune/Payee_Extraction_LoRA_Finetune.ipynb` | Opacus DP-SGD over PEFT adapters with evaluation callbacks |
| LLM privacy audit suite | `CreditMitra-research/benchmark_payee/` (`Outputs_privacy_audit/script1_extraction.py`, `canaray.py`, `mia.py`, `Outputs_privacy_audit/script4_tradeoff.py`, `script5_hyperparam.py`) | Verbatim-extraction, canary, 8-variant membership-inference and privacy/utility trade-off tests for any fine-tuned model |
| Incremental-lift harness | `CreditMitra-engine/experiments/incremental_lift/` | Measures what a feature group adds to a credit model, by input masking and paired bootstrap; validated with null and positive controls; numpy only |
| End-to-end checker | `CreditMitra-engine/scripts/e2e_check.py` | Smoke-tests any running engine; prints counts only, so safe on real data |
| Deployment | `CreditMitra-engine/deploy/`, `docs/DEPLOY.md` | Compose file for engine + merchant server + dashboard (+ optional Ollama) |
