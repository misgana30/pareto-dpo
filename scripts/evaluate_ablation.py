import argparse
import pickle
import copy

import numpy as np
import torch
from tqdm import tqdm

from pareto_dpo.config import ParetoDPOConfig
from pareto_dpo.data.dataset import read_smiles_file
from pareto_dpo.model.gpt import ScaffoldGPT
from pareto_dpo.data.tokenizer import load_or_create_tokenizer
from pareto_dpo.optimization.scorer import compute_objectives


def extract_scaffolds(data_path, num_scaffolds):
    smiles_list = read_smiles_file(data_path)
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
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
            "valid_frac": len(vals) / len(all_scores),
        }
    return stats


def interpolate_models(base_model, dpo_model, alpha, tokenizer, base_model_path):
    alpha = float(alpha)
    interp = ScaffoldGPT.from_pretrained(base_model_path, tokenizer).to(next(base_model.parameters()).device)
    for interp_param, base_param, dpo_param in zip(
        interp.parameters(), base_model.parameters(), dpo_model.parameters()
    ):
        interp_param.data = (1.0 - alpha) * base_param.data + alpha * dpo_param.data
    return interp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default="checkpoints/base/epoch_10")
    parser.add_argument("--dpo_model_path", default="checkpoints/dpo_v4")
    parser.add_argument("--data_path", default="chembl_36.smi")
    parser.add_argument("--num_scaffolds", type=int, default=100)
    parser.add_argument("--samples_per_scaffold", type=int, default=64)
    parser.add_argument("--output", default="data/eval_ablation.pkl")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 0.9, 1.0])
    parser.add_argument("--interpolations", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    args = parser.parse_args()

    config = ParetoDPOConfig()
    tokenizer = load_or_create_tokenizer(tokenizer_path="data/tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading base model...")
    base_model = ScaffoldGPT.from_pretrained(args.base_model_path, tokenizer).to(device)
    base_model.eval()

    print("Loading DPO model...")
    dpo_model = ScaffoldGPT.from_pretrained(args.dpo_model_path, tokenizer).to(device)
    dpo_model.eval()

    print(f"Extracting {args.num_scaffolds} scaffolds...")
    scaffolds = extract_scaffolds(args.data_path, args.num_scaffolds)
    print(f"Using {len(scaffolds)} scaffolds")

    objectives = config.objectives
    all_results = {}

    # --- 1. Baseline: base model at temp=1.0 ---
    all_scores = []
    for scaffold in tqdm(scaffolds, desc="Base model (t=1.0)"):
        smiles_list = base_model.generate_from_scaffold(
            scaffold, num_return_sequences=args.samples_per_scaffold,
            temperature=1.0, top_k=config.gen_top_k, top_p=config.gen_top_p,
        )
        for smi in smiles_list:
            all_scores.append(compute_objectives(smi, objectives))
    all_results["base_t1.0"] = {
        "scores": all_scores,
        "stats": compute_summary_stats(all_scores, objectives),
    }

    # --- 2. DPO model at different temperatures ---
    for temp in args.temperatures:
        all_scores = []
        for scaffold in tqdm(scaffolds, desc=f"DPO model (t={temp})"):
            smiles_list = dpo_model.generate_from_scaffold(
                scaffold, num_return_sequences=args.samples_per_scaffold,
                temperature=temp, top_k=config.gen_top_k, top_p=config.gen_top_p,
            )
            for smi in smiles_list:
                all_scores.append(compute_objectives(smi, objectives))
        all_results[f"dpo_t{temp}"] = {
            "scores": all_scores,
            "stats": compute_summary_stats(all_scores, objectives),
        }

    # --- 3. Interpolated models ---
    for alpha in args.interpolations:
        interp_model = interpolate_models(base_model, dpo_model, alpha, tokenizer, args.base_model_path)
        interp_model.eval()
        all_scores = []
        for scaffold in tqdm(scaffolds, desc=f"Interpolated alpha={alpha}"):
            smiles_list = interp_model.generate_from_scaffold(
                scaffold, num_return_sequences=args.samples_per_scaffold,
                temperature=1.0, top_k=config.gen_top_k, top_p=config.gen_top_p,
            )
            for smi in smiles_list:
                all_scores.append(compute_objectives(smi, objectives))
        all_results[f"interp_a{alpha}"] = {
            "scores": all_scores,
            "stats": compute_summary_stats(all_scores, objectives),
        }
        # Free memory
        del interp_model
        torch.cuda.empty_cache()

    # --- Print results ---
    print("\n" + "=" * 100)
    print(f"{'Model':20s} {'Valid%':>8s} {'QED(med)':>10s} {'clogP(med)':>12s} {'SA(med)':>10s} {'MW(med)':>10s}")
    print("=" * 100)

    for name in sorted(all_results.keys()):
        s = all_results[name]["stats"]
        valid = s["qed"]["valid_frac"]
        qed_med = s["qed"]["median"]
        clogp_med = s["clogp"]["median"]
        sa_med = s["sa"]["median"]
        mw_med = s["mw"]["median"]
        print(f"{name:20s} {valid:>7.1%}  {qed_med:>9.4f}  {clogp_med:>10.4f}  {sa_med:>9.4f}  {mw_med:>9.1f}")

    with open(args.output, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
