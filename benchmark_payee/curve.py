import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\gargi\Downloads\pics\projects\final\Credit_Mitra_IITB_gk\benchmark_payee")

experiments = [
    {"label": "Non-DP (ε=∞)",  "epsilon": float("inf"), "folder": "outputs_8/outputs/eval"},
    {"label": "DP (ε=1)",      "epsilon": 1.0,           "folder": "outputs_1/eval-dp"},
    {"label": "DP (ε=2)",      "epsilon": 2.0,           "folder": "outputs_2/eval-dp"},
    {"label": "DP (ε=4)",      "epsilon": 4.0,           "folder": "outputs_4/eval-dp"},
    {"label": "DP (ε=8)",      "epsilon": 8.0,           "folder": "outputs_8/outputs/eval-dp"},
]



# ── Load metrics ─────────────────────────────────────────────────────────────
results = []
for exp in experiments:
    path = BASE / exp["folder"] / "metrics.json"
    if not path.exists():
        print(f"WARNING: missing {path}")
        continue
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    results.append({
        "label":               exp["label"],
        "epsilon":             exp["epsilon"],
        "avg_char_similarity": m["avg_char_similarity"],
        "avg_token_jaccard":   m["avg_token_jaccard"],
    })
    print(f"Loaded {exp['label']}: CharSim={m['avg_char_similarity']:.3f}  Jaccard={m['avg_token_jaccard']:.3f}")

# ── Separate DP vs Non-DP ────────────────────────────────────────────────────
dp_results  = [r for r in results if r["epsilon"] != float("inf")]
nondp       = next(r for r in results if r["epsilon"] == float("inf"))

epsilons    = [r["epsilon"] for r in dp_results]
cs_vals     = [r["avg_char_similarity"] for r in dp_results]
jac_vals    = [r["avg_token_jaccard"] for r in dp_results]

nondp_cs    = nondp["avg_char_similarity"]
nondp_jac   = nondp["avg_token_jaccard"]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

# Char similarity line
ax.plot(epsilons, cs_vals, "o-", color="#791432", linewidth=2.5,
        markersize=8, label="Char Similarity (DP runs)", zorder=3)

# Token Jaccard line
ax.plot(epsilons, jac_vals, "s-", color="#D79B00", linewidth=2.5,
        markersize=8, label="Token Jaccard (DP runs)", zorder=3)

# Non-DP ceiling lines
ax.axhline(nondp_cs, color="#791432", linestyle="--", linewidth=1.5, alpha=0.5,
           label=f"Non-DP ceiling — CharSim ({nondp_cs:.3f})")
ax.axhline(nondp_jac, color="#D79B00", linestyle="--", linewidth=1.5, alpha=0.5,
           label=f"Non-DP ceiling — Jaccard ({nondp_jac:.3f})")

# Shade the gap between DP and non-DP ceiling
ax.fill_between(epsilons, cs_vals, nondp_cs, alpha=0.07, color="#2980B9")

# Annotate operating point at ε=2
op = next(r for r in dp_results if r["epsilon"] == 2.0)
ax.annotate(
    f"Operating point\nε=2.0",
    xy=(2.0, op["avg_char_similarity"]),
    xytext=(2.8, op["avg_char_similarity"] - 0.055),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=8, color="black",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.6),
)

# Annotate all DP points with their CS value
# Replace the "annotate all DP points" block with this
for eps, cs in zip(epsilons, cs_vals):
    ax.annotate(f"{cs:.3f}", xy=(eps, cs),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8, color="#2980B9")

ax.set_xlabel("Privacy Budget ε  (lower = more private → stronger privacy)", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Privacy–Utility Tradeoff Curve\nStronger privacy (lower ε) → measurable but modest utility cost",
             fontweight="bold", fontsize=12)
ax.set_xticks(epsilons)
ax.set_xticklabels([f"ε={e}" for e in epsilons], fontsize=10)
ax.set_ylim(0.72, 0.98)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_png = BASE / "33_privacy_utility_curve_clean.png"
out_eps = BASE / "33_privacy_utility_curve_clean.eps"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_eps, bbox_inches="tight", format="eps")
plt.show()
print(f"Saved: {out_png}")
print(f"Saved: {out_eps}")