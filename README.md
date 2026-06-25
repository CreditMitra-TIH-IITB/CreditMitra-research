# DP-QLoRA Privacy Audit — Script Suite
## Run order, GPU requirements, outputs

All scripts write to: `benchmark_payee/outputs_privacy_audit/`

---

## Configuration (fixed — do not change)
```
BASE_MODEL   = Qwen/Qwen2.5-1.5B-Instruct
MAX_LENGTH   = 64          # compliance requirement
ε  ≈ 2.0   (outputs_2, ε = 1.9911)
δ  = 2.5×10⁻⁴
C  = 1.5   (clipping norm)
B  = 24    (batch size)
Epochs = 6
```

---

## Script 1 — Training Data Extraction Test
**File:** `script1_extraction.py`
**GPU:** Yes — loads DP model (outputs_2) + non-DP model (outputs_8)
**Runtime:** ~15–25 min on T4/RTX

**What it tests:**
Gives the model the first 50% of a REAL training narration (never the second half)
and checks if it can complete the suffix verbatim. If the DP model completes fewer
suffixes than the non-DP model on TRAINING samples but both are similar on VAL
samples → DP reduced memorization. This is a valid memorization test because the
suffix is HIDDEN from the query.

**Run:**
```bash
python script1_extraction.py --project_root /path/to/benchmark_payee
```

**Outputs:**
- `s1_dp_train.json`, `s1_lora_train.json`, `s1_dp_val.json`, `s1_lora_val.json`
- `script1_extraction_test.pdf`  — 6-panel figure with bar charts, sim distributions,
                                   train vs val matrix, and summary box

---

## Script 2 — Canary Memorization Test (Fixed)
**File:** `script2_canary.py`
**GPU:** Yes — loads DP model (outputs_2) + non-DP model (outputs_8)
**Runtime:** ~20–30 min on T4/RTX

**What it tests (and why the previous canary.py was wrong):**
The original canary.py embedded the canary name IN the query narration
and asked the model to extract it — that is the ordinary task solvable
with zero memorization. THIS script fixes it:

The canary name is HIDDEN from the query. Only the UPI handle is shown:
```
Query: "A UPI transaction was processed using: zephyranth.bvlgari1234@okaxis
        What is the full payee name registered to this handle?"
```
The model can only answer correctly if it memorized the name↔handle mapping
from the training record. DP formally limits this.

Additionally computes the EXPOSURE METRIC (secret-sharer style):
ranks each canary's log-probability against 100 control names.
Rank ≈ 0.5 → not memorized. Rank → 1 → memorized.

**Run:**
```bash
python script2_canary.py --project_root /path/to/benchmark_payee
```

**Outputs:**
- `s2_canary_dp.json`, `s2_canary_nondp.json`, `s2_exposure.json`
- `script2_canary_extraction_rates.pdf`  — exact vs partial recall bars
- `script2_canary_heatmap.pdf`           — per-canary LEAK/SAFE grid
- `script2_canary_exposure.pdf`          — exposure rank per canary
- `script2_canary_summary.pdf`           — one-page combined summary

---

## Script 3 — 8-Variant MIA (targeting outputs_2)
**File:** `script3_mia.py`
**GPU:** Yes — loads DP model (outputs_2) + non-DP (outputs_8) + PT base
**Runtime:** ~30–45 min on T4/RTX (no shadow models)

**What it tests:**
All 8 MIA variants from the Google DP guide and your paper's Fig 6,
ALL targeting outputs_2 (ε = 1.9911) so the attack evidence and the
claimed ε are provably from the same model.

Attack variants:
1. Basic LOSS            — raw loss threshold
2. PT-Ref LOSS           — loss ratio vs pre-trained base
3. Response-Only LOSS    — loss on response tokens only
4. PT-Ref Response       — response-only ratio vs PT base
5. Min-K% Prob           — bottom-K% token probability (Shi et al. 2024)
6. PT-Ref Min-K%         — Min-K% ratio vs PT base
7. Loss Variance         — within-sample token loss variance
8. Zlib Normalised       — loss / compressed byte length

**Run:**
```bash
python script3_mia.py --project_root /path/to/benchmark_payee
```

**Outputs:**
- `s3_mia_dp_metrics.json`, `s3_mia_nondp_metrics.json`
- `script3_mia_bar_comparison.pdf`  — 8 attacks side-by-side bar chart
- `script3_mia_roc_grid.pdf`        — 8 ROC curves in a 2×4 grid
- `script3_loss_distributions.pdf`  — member vs non-member loss histograms
- `script3_mia_summary_table.pdf`   — formatted results table

---

## Script 4 — Privacy-Utility Tradeoff Dashboard
**File:** `script4_tradeoff.py`
**GPU:** None — reads from existing JSON outputs
**Runtime:** < 1 min

**What it produces:**
6-panel dashboard assembling the full story:
- Panel A: F1 vs ε for all models
- Panel B: Privacy-utility knee curve (ε=2 is optimal)
- Panel C: MIA-AUC vs ε (privacy floor — picks up live results from script 3)
- Panel D: ε accumulation across all four training runs
- Panel E: Full benchmark table (all models, all metrics)
- Panel F: Error breakdown (exact→partial, not catastrophic failure)

Run this LAST (after scripts 1-3) so Panel C uses the live MIA numbers.
Or run standalone — it uses the paper's existing numbers as fallback.

**Run:**
```bash
python script4_tradeoff.py --project_root /path/to/benchmark_payee
```

**Outputs:**
- `script4_tradeoff_dashboard.pdf`  — main 6-panel figure
- `script4_knee_standalone.pdf`     — standalone knee (for LaTeX \includegraphics)
- `script4_eps_accumulation.pdf`    — standalone ε curves (for LaTeX)

---

## Recommended run order on Colab/Kaggle GPU

```bash
cd /content/benchmark_payee

# Script 4 first (no GPU) to verify plots work
python script4_tradeoff.py --project_root .

# Then the three GPU scripts
python script1_extraction.py --project_root . --n_samples 100
python script2_canary.py     --project_root . --n_canaries 10
python script3_mia.py        --project_root . --max_samples 486

# Re-run script 4 to pick up live MIA results in Panel C
python script4_tradeoff.py --project_root .
```

Restart runtime between scripts if GPU OOM (each script deletes its models
and calls torch.cuda.empty_cache() before loading the next).

---

## What each output proves (for the benchmark document)

| Script | Evidence type | Claim supported |
|--------|--------------|-----------------|
| 1 | Extraction test | DP suppresses verbatim memorization of training narrations |
| 2 | Canary (fixed) | DP prevents recovery of memorized name↔handle mappings |
| 3 | 8-variant MIA  | No current attack exceeds AUC≈0.51 on ε=2 model |
| 4 | Dashboard      | ε=2 is the utility knee; formal guarantee predicts the attack failure |

Together: formal ceiling (ε=1.9911, Claim 6) + empirical floor (Scripts 1–3).
