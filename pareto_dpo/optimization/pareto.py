from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .scorer import score_molecules


def is_pareto_dominant(scores_a: np.ndarray, scores_b: np.ndarray) -> bool:
    return bool(np.all(scores_a >= scores_b - 1e-8) and np.any(scores_a > scores_b + 1e-8))


def build_pareto_preference_pairs(
    smiles_per_scaffold: List[Tuple[str, List[str]]],
    objectives: List[str],
    directions: List[str],
    max_pairs_per_scaffold: int = 256,
    verbose: bool = False,
) -> List[Tuple[str, str, str, np.ndarray, np.ndarray]]:
    all_pairs = []

    iterator = tqdm(smiles_per_scaffold, desc="Building Pareto pairs") if verbose else smiles_per_scaffold

    for scaffold, smiles_list in iterator:
        if len(smiles_list) < 2:
            continue

        scores, valid_mask = score_molecules(smiles_list, objectives, directions)
        n_valid = valid_mask.sum()
        if n_valid < 2 and len(smiles_list) < 2:
            continue

        pairs = []
        n = len(smiles_list)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if is_pareto_dominant(scores[i], scores[j]):
                    pairs.append((
                        scaffold,
                        smiles_list[i],
                        smiles_list[j],
                        scores[i].copy(),
                        scores[j].copy(),
                    ))

        if len(pairs) > max_pairs_per_scaffold:
            idxs = np.random.choice(len(pairs), max_pairs_per_scaffold, replace=False)
            pairs = [pairs[i] for i in idxs]

        all_pairs.extend(pairs)

    return all_pairs


def build_pareto_preference_pairs_batched(
    scaffolds: List[str],
    generate_fn,
    objectives: List[str],
    directions: List[str],
    num_samples_per_scaffold: int = 64,
    max_pairs_per_scaffold: int = 256,
    verbose: bool = False,
) -> List[Tuple[str, str, str, np.ndarray, np.ndarray]]:
    smiles_per_scaffold = []
    for scaffold in (tqdm(scaffolds, desc="Generating pairs") if verbose else scaffolds):
        generated = generate_fn(scaffold, num_samples_per_scaffold)
        smiles_per_scaffold.append((scaffold, generated))

    return build_pareto_preference_pairs(
        smiles_per_scaffold, objectives, directions, max_pairs_per_scaffold, verbose
    )
