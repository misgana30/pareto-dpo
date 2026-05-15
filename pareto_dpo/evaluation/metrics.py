from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem

from ..optimization.scorer import compute_objectives


def compute_validity(smiles_list: List[str]) -> float:
    if not smiles_list:
        return 0.0
    valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
    return valid / len(smiles_list)


def compute_uniqueness(smiles_list: List[str]) -> float:
    if not smiles_list:
        return 0.0
    unique = set()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            unique.add(Chem.MolToSmiles(mol))
    return len(unique) / len(smiles_list) if smiles_list else 0.0


def compute_novelty(
    generated: List[str], reference: List[str]
) -> float:
    if not generated:
        return 0.0
    ref_set = set()
    for s in reference:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            ref_set.add(Chem.MolToSmiles(mol))
    novel = 0
    for s in generated:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            if canon not in ref_set:
                novel += 1
    return novel / len(generated)


def compute_pareto_front_coverage(
    scores: np.ndarray, directions: List[str]
) -> float:
    adjusted = scores.copy()
    for i, d in enumerate(directions):
        if d == "min":
            adjusted[:, i] = -adjusted[:, i]

    pareto_mask = np.ones(len(adjusted), dtype=bool)
    for i in range(len(adjusted)):
        for j in range(len(adjusted)):
            if i != j and pareto_mask[j]:
                if np.all(adjusted[j] >= adjusted[i] - 1e-8) and np.any(
                    adjusted[j] > adjusted[i] + 1e-8
                ):
                    pareto_mask[i] = False
                    break

    return pareto_mask.sum() / len(adjusted)


def evaluate_generation(
    model,
    scaffolds: List[str],
    reference_smiles: Optional[List[str]] = None,
    samples_per_scaffold: int = 128,
    objectives: Optional[List[str]] = None,
    directions: Optional[List[str]] = None,
    temperature: float = 1.0,
) -> Dict[str, float]:
    all_generated = []
    per_scaffold_scores = []

    for scaffold in scaffolds:
        generated = model.generate_from_scaffold(
            scaffold,
            num_return_sequences=samples_per_scaffold,
            temperature=temperature,
        )
        all_generated.extend(generated)

        if objectives:
            scores_list = []
            for smi in generated:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    scores = compute_objectives(smi, objectives)
                    scores_list.append([scores[o] for o in objectives])
            if scores_list:
                per_scaffold_scores.extend(scores_list)

    metrics = {
        "validity": compute_validity(all_generated),
        "uniqueness": compute_uniqueness(all_generated),
    }

    if reference_smiles:
        metrics["novelty"] = compute_novelty(all_generated, reference_smiles)

    if per_scaffold_scores and objectives and directions:
        scores_arr = np.array(per_scaffold_scores)
        metrics["pareto_front_coverage"] = compute_pareto_front_coverage(
            scores_arr, directions
        )
        for i, obj in enumerate(objectives):
            metrics[f"mean_{obj}"] = float(np.nanmean(scores_arr[:, i]))
            metrics[f"std_{obj}"] = float(np.nanstd(scores_arr[:, i]))

    metrics["total_generated"] = len(all_generated)
    return metrics
