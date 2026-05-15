from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, MolFromSmiles
from rdkit.Contrib.SA_Score import sascorer


def compute_qed(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return QED.qed(mol)
    except Exception:
        return None


def compute_clogp(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Crippen.MolLogP(mol)
    except Exception:
        return None


def compute_sa(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return None


def compute_mw(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Descriptors.MolWt(mol)
    except Exception:
        return None


OBJECTIVE_FUNCS = {
    "qed": compute_qed,
    "clogp": compute_clogp,
    "sa": compute_sa,
    "mw": compute_mw,
}


def compute_objectives(
    smiles: str, objectives: List[str]
) -> Dict[str, Optional[float]]:
    scores = {}
    for obj in objectives:
        func = OBJECTIVE_FUNCS.get(obj)
        if func is None:
            raise ValueError(f"Unknown objective: {obj}")
        scores[obj] = func(smiles)
    return scores


def score_molecules(
    smiles_list: List[str],
    objectives: List[str],
    directions: List[str],
    verbose: bool = False,
) -> np.ndarray:
    results = []
    valid_mask = []
    for smi in smiles_list:
        scores = compute_objectives(smi, objectives)
        row = []
        valid = True
        for obj in scores:
            val = scores[obj]
            if val is None:
                valid = False
                row.append(np.nan)
            else:
                row.append(val)
        results.append(np.array(row, dtype=np.float32))
        valid_mask.append(valid)

    results = np.stack(results)
    for i, direction in enumerate(directions):
        if direction == "min":
            results[:, i] = -results[:, i]

    return results, np.array(valid_mask, dtype=bool)
