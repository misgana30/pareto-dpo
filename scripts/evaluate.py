import argparse
import pickle
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from pareto_dpo.config import ParetoDPOConfig
from pareto_dpo.data.dataset import read_smiles_file
from pareto_dpo.model.gpt import ScaffoldGPT
from pareto_dpo.data.tokenizer import load_or_create_tokenizer
from pareto_dpo.optimization.scorer import compute_objectives


def generate_from_model(model, scaffold, n, temperature, top_k, top_p):
    return model.generate_from_scaffold(
        scaffold,
        num_return_sequences=n,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def extract_scaffolds(data_path, num_scaffolds):
    smiles_list = read_smiles_file(data_path)
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = set()
    # Early-stop once we have enough unique scaffolds (collect buffer for invalid ones)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default="checkpoints/base/epoch_10")
    parser.add_argument("--dpo_model_path", default="checkpoints/dpo_v2")
    parser.add_argument("--data_path", default="chembl_36.smi")
    parser.add_argument("--num_scaffolds", type=int, default=200)
    parser.add_argument("--samples_per_scaffold", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default="data/eval_results.pkl")
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

    results = {"base": {}, "dpo": {}}

    for model_name, model in [("base", base_model), ("dpo", dpo_model)]:
        all_scores = []

        for scaffold in tqdm(scaffolds, desc=f"Generating ({model_name})"):
            smiles_list = generate_from_model(
                model, scaffold, args.samples_per_scaffold,
                args.temperature, config.gen_top_k, config.gen_top_p
            )
            for smi in smiles_list:
                scores = compute_objectives(smi, config.objectives)
                all_scores.append(scores)

        stats = compute_summary_stats(all_scores, config.objectives)
        results[model_name]["scores"] = all_scores
        results[model_name]["stats"] = stats

        print(f"\n{'='*50}")
        print(f"  {model_name.upper()} MODEL ({len(all_scores)} samples)")
        print(f"{'='*50}")
        for obj in config.objectives:
            s = stats[obj]
            print(f"  {obj:6s}: mean={s['mean']:.4f}  median={s['median']:.4f}  "
                  f"std={s['std']:.4f}  [{s['min']:.3f}, {s['max']:.3f}]  "
                  f"valid={s['valid_frac']:.1%}")

    print(f"\n{'='*50}")
    print(f"  DIFF (DPO - Base)")
    print(f"{'='*50}")
    for obj in config.objectives:
        b = results["base"]["stats"][obj]
        d = results["dpo"]["stats"][obj]
        print(f"  {obj:6s}: mean Δ={d['mean']-b['mean']:+.4f}  "
              f"median Δ={d['median']-b['median']:+.4f}")

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
