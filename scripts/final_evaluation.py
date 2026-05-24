"""
Final comprehensive evaluation: base vs DPO.

Produces:
  - data/final_eval.pkl          : full results
  - data/final_ablation.pkl      : model interpolation sweep
  - figures/pareto_front.png     : 2D Pareto front projections
  - figures/distributions.png    : KDE plots per objective
  - figures/ablation_sweep.png   : interpolation alpha sweep
"""

import argparse
import pickle
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from pareto_dpo.config import ParetoDPOConfig
from pareto_dpo.data.dataset import read_smiles_file
from pareto_dpo.model.gpt import ScaffoldGPT
from pareto_dpo.data.tokenizer import load_or_create_tokenizer
from pareto_dpo.optimization.scorer import compute_objectives
from pareto_dpo.evaluation.metrics import (
    compute_validity,
    compute_uniqueness,
    compute_novelty,
    compute_pareto_front_coverage,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def extract_scaffolds(data_path, num_scaffolds):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    smiles_list = read_smiles_file(data_path)
    scaffolds = set()
    target = num_scaffolds * 3
    for smi in tqdm(smiles_list, desc="Extracting scaffolds"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                s = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(s))
                if len(scaffolds) >= target:
                    break
            except Exception:
                pass
    result = list(scaffolds)[:num_scaffolds]
    print(f"Collected {len(scaffolds)} unique scaffolds, using {len(result)}")
    return result


def compute_summary_stats(all_scores, objectives):
    stats = {}
    for obj in objectives:
        vals = [s[obj] for s in all_scores if s[obj] is not None]
        stats[obj] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "valid_frac": len(vals) / len(all_scores),
        }
    return stats


def interpolate_models(base_model, dpo_model, alpha, tokenizer, base_model_path, device):
    interp = ScaffoldGPT.from_pretrained(base_model_path, tokenizer).to(device)
    for interp_param, base_param, dpo_param in zip(
        interp.parameters(), base_model.parameters(), dpo_model.parameters()
    ):
        interp_param.data = (1.0 - alpha) * base_param.data + alpha * dpo_param.data
    return interp


def generate_from_model(model, scaffold, n, temperature, top_k, top_p):
    return model.generate_from_scaffold(
        scaffold,
        num_return_sequences=n,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def evaluate_model(model, scaffolds, objectives, samples_per_scaffold, temperature,
                   top_k, top_p, reference_smiles=None, directions=None):
    all_generated = []
    all_scores = []

    for scaffold in tqdm(scaffolds, desc="Generating"):
        smiles_list = generate_from_model(
            model, scaffold, samples_per_scaffold, temperature, top_k, top_p
        )
        all_generated.extend(smiles_list)
        for smi in smiles_list:
            scores = compute_objectives(smi, objectives)
            all_scores.append(scores)

    valid_smiles = [s for s in all_generated if s.count('.') == 0]
    for smi in valid_smiles:
        scores = compute_objectives(smi, objectives)

    stats = compute_summary_stats(all_scores, objectives)
    validity = compute_validity(all_generated)
    uniqueness = compute_uniqueness(all_generated)

    metrics = {
        "stats": stats,
        "validity": validity,
        "uniqueness": uniqueness,
    }

    if reference_smiles:
        metrics["novelty"] = compute_novelty(all_generated, reference_smiles)

    scores_arr = []
    for s in all_scores:
        row = [s[o] if s[o] is not None else 0.0 for o in objectives]
        scores_arr.append(row)
    scores_arr = np.array(scores_arr)
    if directions is not None:
        metrics["pareto_coverage"] = compute_pareto_front_coverage(
            scores_arr, directions
        )

    metrics["n_total"] = len(all_generated)
    metrics["n_valid"] = int(validity * len(all_generated))
    return metrics, all_scores


def plot_distributions(results, objectives, save_path):
    n_obj = len(objectives)
    fig, axes = plt.subplots(1, n_obj, figsize=(5 * n_obj, 4))

    if n_obj == 1:
        axes = [axes]

    colors = {"base": "#4C72B0", "dpo": "#DD8452"}

    for idx, obj in enumerate(objectives):
        ax = axes[idx]
        for model_name in ["base", "dpo"]:
            vals = [s[obj] for s in results[model_name]["scores"] if s[obj] is not None]
            sns.kdeplot(vals, ax=ax, label=model_name.upper(), color=colors[model_name],
                        fill=True, alpha=0.3)
        ax.set_xlabel(obj.upper())
        ax.set_ylabel("Density")
        ax.set_title(f"{obj.upper()} Distribution")
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_pareto_front(results, objectives, save_path):
    from matplotlib.patches import Rectangle

    pairs = [("qed", "clogp"), ("qed", "sa"), ("clogp", "mw")]
    directions_map = {"qed": "max", "clogp": "max", "sa": "min", "mw": "min"}
    labels_map = {"qed": "QED ↑", "clogp": "clogP ↑", "sa": "SA ↓", "mw": "MW ↓"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"base": "#4C72B0", "dpo": "#DD8452"}

    for idx, (x_obj, y_obj) in enumerate(pairs):
        ax = axes[idx]
        for model_name in ["base", "dpo"]:
            xs = [s[x_obj] for s in results[model_name]["scores"] if s[x_obj] is not None]
            ys = [s[y_obj] for s in results[model_name]["scores"] if s[y_obj] is not None]
            ax.scatter(xs, ys, c=colors[model_name], label=model_name.upper(),
                       alpha=0.3, s=8, rasterized=True)

        ax.set_xlabel(labels_map[x_obj])
        ax.set_ylabel(labels_map[y_obj])
        ax.legend(markerscale=4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_ablation(ablation_results, objectives, save_path):
    fig, axes = plt.subplots(1, len(objectives), figsize=(5 * len(objectives), 4))
    if len(objectives) == 1:
        axes = [axes]

    alphas = sorted([k for k in ablation_results.keys() if k.startswith("interp_")])
    alpha_vals = [float(a.split("_")[1].replace("a", "")) for a in alphas]

    for idx, obj in enumerate(objectives):
        ax = axes[idx]

        base_val = ablation_results["base_t1.0"]["metrics"]["stats"][obj]["median"]
        dpo_val = ablation_results["dpo_t1.0"]["metrics"]["stats"][obj]["median"]

        interp_vals = []
        for a in alphas:
            interp_vals.append(ablation_results[a]["metrics"]["stats"][obj]["median"])

        ax.axhline(y=base_val, color="#4C72B0", linestyle="--", label="Base", alpha=0.7)
        ax.axhline(y=dpo_val, color="#DD8452", linestyle="--", label="DPO", alpha=0.7)
        ax.plot([0] + alpha_vals + [1],
                [base_val] + interp_vals + [dpo_val],
                "ko-", markersize=4, linewidth=1.5)

        ax.set_xlabel("Interpolation α")
        ax.set_ylabel(f"{obj.upper()} (median)")
        ax.set_title(f"{obj.upper()} Interpolation Sweep")
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_radar(results, objectives, save_path):
    labels = [o.upper() for o in objectives]
    n_obj = len(objectives)
    angles = np.linspace(0, 2 * np.pi, n_obj, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = {"base": "#4C72B0", "dpo": "#DD8452"}

    for model_name in ["base", "dpo"]:
        vals = []
        for obj in objectives:
            m = results[model_name]["metrics"]["stats"][obj]["median"]
            vals.append(m)
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=model_name.upper(), color=colors[model_name])
        ax.fill(angles, vals, alpha=0.1, color=colors[model_name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_title("Multi-Objective Performance (median)", pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default="checkpoints/base/epoch_10")
    parser.add_argument("--dpo_model_path", default="checkpoints/dpo_v5")
    parser.add_argument("--data_path", default="chembl_36.smi")
    parser.add_argument("--num_scaffolds", type=int, default=200)
    parser.add_argument("--samples_per_scaffold", type=int, default=64)
    parser.add_argument("--output", default="data/final_eval.pkl")
    parser.add_argument("--ablation", default="data/final_ablation.pkl")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()

    config = ParetoDPOConfig()
    tokenizer = load_or_create_tokenizer(tokenizer_path="data/tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading models...")
    base_model = ScaffoldGPT.from_pretrained(args.base_model_path, tokenizer).to(device)
    base_model.eval()
    dpo_model = ScaffoldGPT.from_pretrained(args.dpo_model_path, tokenizer).to(device)
    dpo_model.eval()

    reference_smiles = read_smiles_file(args.data_path)

    print(f"Extracting {args.num_scaffolds} scaffolds...")
    scaffolds = extract_scaffolds(args.data_path, args.num_scaffolds)
    print(f"Using {len(scaffolds)} scaffolds")

    results = {}

    for model_name, model in [("base", base_model), ("dpo", dpo_model)]:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} model")
        print(f"{'='*60}")
        metrics, all_scores = evaluate_model(
            model, scaffolds, config.objectives,
            args.samples_per_scaffold, 1.0,
            config.gen_top_k, config.gen_top_p,
            reference_smiles, config.objective_directions,
        )
        results[model_name] = {"metrics": metrics, "scores": all_scores}

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<20s} {'Base':>12s} {'DPO':>12s} {'Δ':>12s}")
    print("-" * 56)
    for obj in config.objectives:
        b = results["base"]["metrics"]["stats"][obj]
        d = results["dpo"]["metrics"]["stats"][obj]
        delta = d["median"] - b["median"]
        print(f"{f'{obj} (median)':<20s} {b['median']:>10.4f}  {d['median']:>10.4f}  {delta:>+10.4f}")
    print("-" * 56)
    for metric in ["validity", "uniqueness", "novelty", "pareto_coverage"]:
        bv = results["base"]["metrics"].get(metric, 0)
        dv = results["dpo"]["metrics"].get(metric, 0)
        if isinstance(bv, float):
            print(f"{metric:<20s} {bv:>10.1%}  {dv:>10.1%}  {dv-bv:>+10.1%}")
        else:
            print(f"{metric:<20s} {bv:>12} {dv:>12}")

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {args.output}")

    # --- Ablation study ---
    print(f"\n{'='*60}")
    print("Running ablation study (model interpolation)")
    print(f"{'='*60}")
    ablation = {}
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
    temps_to_check = [0.8, 0.9, 1.0]

    # Baseline
    for temp in temps_to_check:
        metrics, scores = evaluate_model(
            base_model, scaffolds[:50], config.objectives,
            32, temp, config.gen_top_k, config.gen_top_p,
        )
        ablation[f"base_t{temp}"] = {"metrics": metrics, "scores": scores}

    # DPO at temps
    for temp in temps_to_check:
        metrics, scores = evaluate_model(
            dpo_model, scaffolds[:50], config.objectives,
            32, temp, config.gen_top_k, config.gen_top_p,
        )
        ablation[f"dpo_t{temp}"] = {"metrics": metrics, "scores": scores}

    # Interpolation
    for alpha in alphas:
        interp_model = interpolate_models(base_model, dpo_model, alpha, tokenizer,
                                          args.base_model_path, device)
        interp_model.eval()
        metrics, scores = evaluate_model(
            interp_model, scaffolds[:50], config.objectives,
            32, 1.0, config.gen_top_k, config.gen_top_p,
        )
        ablation[f"interp_a{alpha}"] = {"metrics": metrics, "scores": scores}
        del interp_model
        torch.cuda.empty_cache()

    with open(args.ablation, "wb") as f:
        pickle.dump(ablation, f)
    print(f"Ablation saved to {args.ablation}")

    # --- Figures ---
    os.makedirs(args.figures, exist_ok=True)
    plot_distributions(results, config.objectives,
                       os.path.join(args.figures, "distributions.png"))
    plot_pareto_front(results, config.objectives,
                      os.path.join(args.figures, "pareto_front.png"))
    plot_radar(results, config.objectives,
               os.path.join(args.figures, "radar.png"))
    plot_ablation(ablation, config.objectives,
                  os.path.join(args.figures, "ablation_sweep.png"))

    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
