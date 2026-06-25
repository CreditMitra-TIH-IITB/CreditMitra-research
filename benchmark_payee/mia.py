"""
PT-Referenced MIA — LoRA-Leak style attack on your models
==========================================================
Implements the calibrated LOSS attack from LoRA-Leak (Ran et al., 2025):

    Score = Loss(x; Qwen_base) - Loss(x; LoRA_finetuned)

Members (training samples) show a BIGGER loss drop after fine-tuning
because the model memorized them. Non-members show a smaller drop.
DP noise reduces this gap → lower AUC → stronger privacy.

This is the attack your basic MIA missed — and the one
LoRA-Leak shows DP actually defends against (AUC drops to ~0.52).

Run from project root:
    python mia_pt_referenced.py --project_root .

Reads:
    finetune_standalone/data/train.jsonl  → members
    finetune_standalone/data/val.jsonl    → non-members

Saves to: outputs_1/plots/figures/mia_pt_ref/
"""

import argparse
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
from tqdm import tqdm

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 64

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F9F9F9",
    "axes.grid":         True,
    "grid.color":        "#E0E0E0",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
})


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    p.add_argument("--max_samples",  type=int, default=500)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--data_dir",     default="data",
                   help="Path to folder containing train.jsonl and val.jsonl")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────
def load_jsonl(path, n, seed):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    random.Random(seed).shuffle(records)
    return records[:n]


def fmt(rec):
    if "prompt" in rec and "response" in rec:
        return rec["prompt"].strip() + "\n" + rec["response"].strip()
    return " ".join(str(v) for v in rec.values() if isinstance(v, str))


class PayeeDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.encodings = []
        for rec in records:
            enc = tokenizer(
                fmt(rec), max_length=MAX_LENGTH,
                truncation=True, padding="max_length",
                return_tensors="pt"
            )
            self.encodings.append(enc["input_ids"].squeeze(0))

    def __len__(self):           return len(self.encodings)
    def __getitem__(self, i):    return self.encodings[i]


# ── Model loading ─────────────────────────────────────────────────────────────
def load_base_model(tokenizer):
    """Load base Qwen without any adapter."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model


def load_finetuned(adapter_path, tokenizer):
    """Load base + LoRA adapter."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── Per-sample loss ───────────────────────────────────────────────────────────
@torch.no_grad()
def per_sample_loss(model, loader, pad_id, desc="loss") -> np.ndarray:
    losses = []
    for batch in tqdm(loader, desc=f"  {desc}", leave=False):
        ids    = batch.to(next(model.parameters()).device)
        labels = ids.clone()
        labels[labels == pad_id] = -100

        out    = model(input_ids=ids, labels=labels)
        logits = out.logits

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        ce         = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = ce(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        mask     = (shift_labels != -100).float()
        seq_loss = (token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
        losses.extend(seq_loss.cpu().tolist())
    return np.array(losses)


# ── Attack scores ─────────────────────────────────────────────────────────────
def basic_loss_score(ft_loss):
    """
    Basic LOSS attack (your original MIA).
    Score = -loss_finetuned
    Members have lower loss → higher score.
    """
    return -ft_loss


def pt_referenced_score(base_loss, ft_loss):
    """
    PT-Referenced LOSS attack (LoRA-Leak).
    Score = loss_base - loss_finetuned
    Members show bigger loss DROP after fine-tuning → higher score.
    This is what DP actually defends against.
    """
    return base_loss - ft_loss


def evaluate(scores, labels, tag):
    auc_val = roc_auc_score(labels, scores)
    ap      = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    print(f"    {tag:<35} AUC={auc_val:.4f}  AP={ap:.4f}")
    return auc_val, ap, fpr, tpr


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_roc_comparison(results: dict, out: Path):
    """
    4 ROC curves:
    - Basic LOSS: LoRA vs DP-LoRA
    - PT-Referenced: LoRA vs DP-LoRA
    Shows clearly that pt-referenced attack is stronger AND
    that DP reduces it more than basic attack.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left — Basic LOSS attack
    ax = axes[0]
    ax.plot([0,1],[0,1], color="#AAAAAA", lw=1.2, linestyle="--",
            label="Random (AUC=0.500)")
    for name, color, ls in [
        ("lora_basic",  "#82B366", "--"),
        ("dp_basic",    "#D79B00", "-"),
    ]:
        r = results[name]
        ax.plot(r["fpr"], r["tpr"], color=color, lw=2.2, linestyle=ls,
                label=f"{r['label']}  (AUC={r['auc']:.4f})")
    ax.set_title("Basic LOSS Attack\n(your original MIA)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    # Right — PT-Referenced attack
    ax = axes[1]
    ax.plot([0,1],[0,1], color="#AAAAAA", lw=1.2, linestyle="--",
            label="Random (AUC=0.500)")
    for name, color, ls in [
        ("lora_ptref",  "#82B366", "--"),
        ("dp_ptref",    "#D79B00", "-"),
    ]:
        r = results[name]
        ax.plot(r["fpr"], r["tpr"], color=color, lw=2.2, linestyle=ls,
                label=f"{r['label']}  (AUC={r['auc']:.4f})")
    ax.set_title("PT-Referenced LOSS Attack\n(LoRA-Leak style — stronger attack)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    fig.suptitle(
        "MIA Comparison: Basic vs PT-Referenced Attack\n"
        "PT-referenced is stronger — and DP defends against it more",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out / "fig_mia_ptref_roc.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_mia_ptref_roc.pdf")


def plot_auc_summary(results: dict, out: Path):
    """
    Bar chart comparing all 4 AUC values.
    Key message: PT-referenced reveals more risk for LoRA,
    but DP successfully defends against it.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = [
        "Basic LOSS\nQLoRA (non-priv)",
        "Basic LOSS\nDP-QLoRA",
        "PT-Referenced\nQLoRA (non-priv)",
        "PT-Referenced\nDP-QLoRA",
    ]
    keys   = ["lora_basic", "dp_basic", "lora_ptref", "dp_ptref"]
    aucs   = [results[k]["auc"] for k in keys]
    colors = ["#82B366", "#D79B00", "#82B366", "#D79B00"]
    hatch  = ["", "", "///", "///"]

    bars = ax.bar(labels, aucs, color=colors, edgecolor="white",
                  width=0.5, zorder=3)
    for bar, h in zip(bars, hatch):
        bar.set_hatch(h)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(0.5, color="#AAAAAA", lw=1.5, linestyle="--",
               label="Random attacker (0.500)")

    # Annotate: pt-ref reveals more risk for LoRA
    lora_gain = results["lora_ptref"]["auc"] - results["lora_basic"]["auc"]
    if lora_gain > 0:
        ax.annotate("",
                    xy=(2, results["lora_ptref"]["auc"] + 0.003),
                    xytext=(0, results["lora_basic"]["auc"] + 0.003),
                    arrowprops=dict(arrowstyle="<->", color="#82B366", lw=1.3))
        ax.text(1, max(results["lora_ptref"]["auc"],
                       results["lora_basic"]["auc"]) + 0.015,
                f"+{lora_gain:.4f}\npt-ref reveals\nmore risk",
                ha="center", fontsize=8, color="#82B366")

    # Annotate: DP defends against pt-ref
    dp_drop = results["lora_ptref"]["auc"] - results["dp_ptref"]["auc"]
    if dp_drop > 0:
        ax.annotate("",
                    xy=(3, results["dp_ptref"]["auc"] + 0.003),
                    xytext=(2, results["lora_ptref"]["auc"] + 0.003),
                    arrowprops=dict(arrowstyle="<->", color="#AE4132", lw=1.3))
        ax.text(2.5, max(results["lora_ptref"]["auc"],
                         results["dp_ptref"]["auc"]) + 0.015,
                f"-{dp_drop:.4f}\nDP defends",
                ha="center", fontsize=8, color="#AE4132")

    # Legend patches
    import matplotlib.patches as mpatches
    p1 = mpatches.Patch(color="#82B366", label="QLoRA (non-private)")
    p2 = mpatches.Patch(color="#D79B00", label="DP-QLoRA (ε≈0.9938)")
    p3 = mpatches.Patch(facecolor="white", edgecolor="gray",
                         hatch="///", label="PT-Referenced attack")
    ax.legend(handles=[p1, p2, p3], fontsize=9, loc="upper left")

    ax.set_ylabel("MIA-AUC", fontsize=11)
    ax.set_ylim(0.45, max(aucs) + 0.08)
    ax.set_title(
        "MIA-AUC: Basic vs PT-Referenced Attack\n"
        "PT-referenced exposes more risk in LoRA — DP successfully reduces it",
        fontsize=12, fontweight="bold"
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(out / "fig_mia_ptref_summary.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_mia_ptref_summary.pdf")


def plot_loss_drop_distribution(
        base_loss_m, base_loss_nm,
        lora_loss_m, lora_loss_nm,
        dp_loss_m,   dp_loss_nm,
        out: Path):
    """
    Distribution of (base_loss - ft_loss) for members vs non-members.
    This is the pt-referenced score.
    Members should show higher drop (bigger bar to the right).
    DP should compress the member distribution toward zero.
    """
    lora_drop_m  = base_loss_m  - lora_loss_m
    lora_drop_nm = base_loss_nm - lora_loss_nm
    dp_drop_m    = base_loss_m  - dp_loss_m
    dp_drop_nm   = base_loss_nm - dp_loss_nm

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, drop_m, drop_nm, title, color in [
        (axes[0], lora_drop_m,  lora_drop_nm,
         "QLoRA Non-Private\nLoss Drop: Base → Fine-tuned", "#82B366"),
        (axes[1], dp_drop_m,    dp_drop_nm,
         "DP-QLoRA (ε≈0.9938)\nLoss Drop: Base → Fine-tuned", "#D79B00"),
    ]:
        all_vals = np.concatenate([drop_m, drop_nm])
        bins = np.linspace(all_vals.min(), all_vals.max(), 50)

        ax.hist(drop_nm, bins=bins, alpha=0.65, color="#AE4132",
                density=True, edgecolor="white",
                label=f"Non-members  mean={drop_nm.mean():.3f}")
        ax.hist(drop_m,  bins=bins, alpha=0.65, color=color,
                density=True, edgecolor="white",
                label=f"Members  mean={drop_m.mean():.3f}")

        ax.axvline(drop_m.mean(),  color=color,    lw=2, linestyle="--")
        ax.axvline(drop_nm.mean(), color="#AE4132", lw=2, linestyle="--")

        # Annotate separation
        sep = drop_m.mean() - drop_nm.mean()
        ax.set_xlabel("Loss Drop  (base_loss − ft_loss)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{title}\nMean separation = {sep:.4f}", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle(
        "PT-Referenced Score Distribution\n"
        "Larger member/non-member separation = higher MIA-AUC\n"
        "DP should compress member distribution → less separation",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out / "fig_mia_ptref_loss_drop.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_mia_ptref_loss_drop.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    out  = root / "outputs_1/plots/figures/mia_pt_ref"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    print(f"\nProject root: {root}")
    print(f"Output dir  : {out}")

    # ── Load data ──────────────────────────────────────────────────
    members    = load_jsonl(
        root / args.data_dir / "train.jsonl",
        args.max_samples, args.seed)
    nonmembers = load_jsonl(
        root / args.data_dir / "val.jsonl",
        args.max_samples, args.seed)
    n = min(len(members), len(nonmembers))
    members, nonmembers = members[:n], nonmembers[:n]
    print(f"Balanced at n={n} per class")

    tokenizer = get_tokenizer()
    pad_id    = tokenizer.pad_token_id or tokenizer.eos_token_id

    dl_m  = DataLoader(PayeeDataset(members,    tokenizer),
                       batch_size=args.batch_size, shuffle=False)
    dl_nm = DataLoader(PayeeDataset(nonmembers, tokenizer),
                       batch_size=args.batch_size, shuffle=False)

    # ── Step 1: Base model losses (shared for both attacks) ────────
    print("\nLoading BASE model (no adapter)...")
    model_base = load_base_model(tokenizer)

    print("Computing base model losses...")
    base_loss_m  = per_sample_loss(model_base, dl_m,  pad_id, "base members")
    base_loss_nm = per_sample_loss(model_base, dl_nm, pad_id, "base non-members")

    del model_base
    torch.cuda.empty_cache()

    # ── Step 2: Non-private LoRA losses ───────────────────────────
    lora_adapter = root / "outputs_8/outputs/payee-lora"
    print(f"\nLoading non-private LoRA: {lora_adapter}")
    model_lora = load_finetuned(lora_adapter, tokenizer)

    print("Computing LoRA losses...")
    lora_loss_m  = per_sample_loss(model_lora, dl_m,  pad_id, "lora members")
    lora_loss_nm = per_sample_loss(model_lora, dl_nm, pad_id, "lora non-members")

    del model_lora
    torch.cuda.empty_cache()

    # ── Step 3: DP LoRA losses ─────────────────────────────────────
    dp_adapter = root / "outputs_1/payee-lora-dp"
    print(f"\nLoading DP-QLoRA: {dp_adapter}")
    model_dp = load_finetuned(dp_adapter, tokenizer)

    print("Computing DP-LoRA losses...")
    dp_loss_m  = per_sample_loss(model_dp, dl_m,  pad_id, "dp members")
    dp_loss_nm = per_sample_loss(model_dp, dl_nm, pad_id, "dp non-members")

    del model_dp
    torch.cuda.empty_cache()

    # ── Step 4: Compute attack scores ─────────────────────────────
    labels = np.concatenate([np.ones(n), np.zeros(n)])

    print("\n\nRESULTS")
    print("="*60)

    results = {}

    # Basic LOSS attack
    print("\n  Basic LOSS Attack (original MIA):")
    for tag, loss_m, loss_nm, label in [
        ("lora_basic", lora_loss_m, lora_loss_nm, "QLoRA non-private"),
        ("dp_basic",   dp_loss_m,   dp_loss_nm,   "DP-QLoRA (ε≈0.9938)"),
    ]:
        scores = basic_loss_score(
            np.concatenate([loss_m, loss_nm]))
        auc_v, ap, fpr, tpr = evaluate(scores, labels, label)
        results[tag] = {"auc": auc_v, "ap": ap, "fpr": fpr,
                        "tpr": tpr, "label": label}

    # PT-Referenced LOSS attack
    print("\n  PT-Referenced LOSS Attack (LoRA-Leak style):")
    for tag, ft_loss_m, ft_loss_nm, label in [
        ("lora_ptref", lora_loss_m, lora_loss_nm, "QLoRA non-private +ptRef"),
        ("dp_ptref",   dp_loss_m,   dp_loss_nm,   "DP-QLoRA +ptRef"),
    ]:
        scores = pt_referenced_score(
            np.concatenate([base_loss_m,  base_loss_nm]),
            np.concatenate([ft_loss_m,    ft_loss_nm]),
        )
        auc_v, ap, fpr, tpr = evaluate(scores, labels, label)
        results[tag] = {"auc": auc_v, "ap": ap, "fpr": fpr,
                        "tpr": tpr, "label": label}

    print("\n" + "="*60)
    print("  KEY FINDING:")
    gain = results["lora_ptref"]["auc"] - results["lora_basic"]["auc"]
    drop = results["lora_ptref"]["auc"] - results["dp_ptref"]["auc"]
    print(f"  PT-ref reveals +{gain:.4f} more risk in LoRA vs basic attack")
    print(f"  DP reduces pt-ref AUC by {drop:.4f}")
    print("="*60)

    # ── Save metrics ───────────────────────────────────────────────
    summary = {k: {"auc": v["auc"], "ap": v["ap"], "label": v["label"]}
               for k, v in results.items()}
    with open(out / "metrics_ptref.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Generate figures ───────────────────────────────────────────
    print("\nGenerating figures...")
    plot_roc_comparison(results, out)
    plot_auc_summary(results, out)
    plot_loss_drop_distribution(
        base_loss_m, base_loss_nm,
        lora_loss_m, lora_loss_nm,
        dp_loss_m,   dp_loss_nm,
        out
    )

    print(f"\nDone. Outputs: {out}")
    print("  fig_mia_ptref_roc.pdf      — ROC: basic vs pt-referenced")
    print("  fig_mia_ptref_summary.pdf  — AUC bar chart all 4 attacks")
    print("  fig_mia_ptref_loss_drop.pdf — score distributions")
    print("  metrics_ptref.json")


if __name__ == "__main__":
    main()