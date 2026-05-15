import argparse
import json

from pareto_dpo.config import ParetoDPOConfig
from pareto_dpo.data.dataset import read_smiles_file
from pareto_dpo.model.gpt import ScaffoldGPT
from pareto_dpo.data.tokenizer import load_or_create_tokenizer
from pareto_dpo.evaluation.metrics import evaluate_generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--num_scaffolds", type=int, default=200)
    parser.add_argument("--samples_per_scaffold", type=int, default=128)
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
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                s = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(s))
            except Exception:
                pass
    scaffolds = list(scaffolds)[: args.num_scaffolds]
    print(f"Evaluating on {len(scaffolds)} scaffolds")

    metrics = evaluate_generation(
        model=model,
        scaffolds=scaffolds,
        reference_smiles=smiles_list,
        samples_per_scaffold=args.samples_per_scaffold,
        objectives=config.objectives,
        directions=config.objective_directions,
        temperature=args.temperature,
    )

    print(json.dumps(metrics, indent=2))
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
