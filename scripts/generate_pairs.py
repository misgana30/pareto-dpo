import argparse
import pickle
from typing import List

import numpy as np
from tqdm import tqdm
import torch

from pareto_dpo.config import ParetoDPOConfig
from pareto_dpo.data.dataset import read_smiles_file
from pareto_dpo.model.gpt import ScaffoldGPT
from pareto_dpo.data.tokenizer import load_or_create_tokenizer
from pareto_dpo.optimization.scorer import score_molecules
from pareto_dpo.optimization.pareto import build_pareto_preference_pairs


def generate_from_model(
    model, scaffold: str, n: int, temperature: float, top_k: int, top_p: int
) -> List[str]:
    return model.generate_from_scaffold(
        scaffold,
        num_return_sequences=n,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/pareto_pairs.pkl")
    parser.add_argument("--num_scaffolds", type=int, default=1000)
    parser.add_argument("--samples_per_scaffold", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    config = ParetoDPOConfig()
    tokenizer = load_or_create_tokenizer(tokenizer_path="data/tokenizer.json")
    model = ScaffoldGPT.from_pretrained(args.model_path, tokenizer)
    model.eval()

    smiles_list = read_smiles_file(args.data_path)
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = set()
    for smi in tqdm(smiles_list, desc="Extracting scaffolds"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                s = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(s))
            except Exception:
                pass
    scaffolds = list(scaffolds)[: args.num_scaffolds]
    print(f"Using {len(scaffolds)} scaffolds")

    generate_fn = lambda s, n: generate_from_model(
        model, s, n, args.temperature, config.gen_top_k, config.gen_top_p
    )

    pairs = build_pareto_preference_pairs_batched(
        scaffolds=scaffolds,
        generate_fn=generate_fn,
        objectives=config.objectives,
        directions=config.objective_directions,
        num_samples_per_scaffold=args.samples_per_scaffold,
        max_pairs_per_scaffold=config.max_pairs_per_scaffold,
        verbose=True,
    )
    print(f"Built {len(pairs)} preference pairs")

    with open(args.output, "wb") as f:
        pickle.dump(pairs, f)
    print(f"Saved pairs to {args.output}")


if __name__ == "__main__":
    main()
