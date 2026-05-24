import random
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .tokenizer import (
    format_scaffold_decoration,
    load_or_create_tokenizer,
)


def read_smiles_file(path: str, max_mols: Optional[int] = None, shuffle: bool = False) -> List[str]:
    import random
    smiles_list = []
    with open(path, "r") as f:
        for line in f:
            smi = line.strip().split()[0]
            if Chem.MolFromSmiles(smi):
                smiles_list.append(smi)
    if shuffle:
        random.shuffle(smiles_list)
    if max_mols:
        smiles_list = smiles_list[:max_mols]
    return smiles_list


def extract_bemis_murcko_scaffold(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def get_randomized_smiles(smiles: str, num: int = 1) -> List[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    results = []
    for _ in range(num):
        results.append(Chem.MolToSmiles(mol, doRandom=True))
    return results


class ScaffoldDataset:
    def __init__(
        self,
        smiles_list: List[str],
        tokenizer_path: str = "data/tokenizer.json",
        max_length: int = 512,
        augment_smiles: bool = True,
    ):
        self.smiles_list = smiles_list
        self.max_length = max_length
        self.augment_smiles = augment_smiles
        self.tokenizer = load_or_create_tokenizer(
            smiles_list=smiles_list, tokenizer_path=tokenizer_path
        )
        self.pairs = self._build_scaffold_pairs()

    def _build_scaffold_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for smi in self.smiles_list:
            scaffold = extract_bemis_murcko_scaffold(smi)
            if scaffold is not None and scaffold != smi:
                pairs.append((scaffold, smi))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        scaffold, full_smiles = self.pairs[idx]
        decoration_part = full_smiles[len(scaffold):] if full_smiles.startswith(scaffold) else self._extract_decoration(full_smiles, scaffold)
        if self.augment_smiles:
            decorations = get_randomized_smiles(full_smiles, num=1)
            if decorations:
                full_smiles = decorations[0]
                decoration_part = self._extract_decoration(full_smiles, scaffold)
        text = format_scaffold_decoration(scaffold, decoration_part)
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "scaffold": scaffold,
            "full_smiles": full_smiles,
        }

    @staticmethod
    def _extract_decoration(full_smiles: str, scaffold: str) -> str:
        mol = Chem.MolFromSmiles(full_smiles)
        scaffold_mol = Chem.MolFromSmiles(scaffold)
        if mol is None or scaffold_mol is None:
            return full_smiles
        return full_smiles

    def get_scaffold_set(self) -> List[str]:
        scaffolds = set()
        for s, _ in self.pairs:
            scaffolds.add(s)
        return list(scaffolds)
