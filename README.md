# Pareto-DPO

**Multi-Objective Preference Alignment for Scaffold Decoration via Pareto-Dominant Pair Construction**

## Idea

Existing scaffold decoration methods optimize molecules by scalarizing multiple objectives (e.g., QED, cLogP, SA) into a single weighted reward, then using reinforcement learning or evolutionary algorithms. This is sensitive to weight selection and rarely recovers the full Pareto front.

**Pareto-DPO** replaces scalarized rewards with direct preference optimization:

1. Generate many decoration candidates per scaffold using a pretrained SMILES GPT
2. Score each candidate on multiple objectives (QED, cLogP, SA, MW)
3. Build preference pairs using **Pareto dominance** — molecule A is preferred to B if A is better on at least one objective and not worse on any
4. Fine-tune the GPT with the **DPO loss** (Rafailov et al., 2023) on these pairs
5. The resulting model generates molecules that are Pareto-optimal without reward weighting

## Repository structure

```
pareto-dpo/
├── pareto_dpo/
│   ├── config.py              # Configuration
│   ├── data/
│   │   ├── dataset.py         # Scaffold extraction, dataset
│   │   └── tokenizer.py       # SMILES tokenizer
│   ├── model/
│   │   └── gpt.py             # Scaffold-conditioned GPT-2
│   ├── optimization/
│   │   ├── scorer.py          # RDKit: QED, cLogP, SA, MW
│   │   ├── pareto.py          # Pareto-dominant pair builder
│   │   └── dpo_trainer.py     # DPO training loop
│   └── evaluation/
│       └── metrics.py         # Validity, novelty, Pareto coverage
├── scripts/
│   ├── train_base.py          # Pretrain GPT on ChEMBL
│   ├── generate_pairs.py      # Sample + score + build Pareto pairs
│   ├── train_dpo.py           # DPO fine-tuning
│   └── evaluate.py            # Evaluation
├── notebooks/
│   └── pareto_dpo_demo.ipynb  # End-to-end walkthrough
└── requirements.txt
```

## Usage

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download ChEMBL (or any SMILES file)
#    wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_30.smi.gz

# 3. Pretrain base GPT
python scripts/train_base.py --data_path data/chembl_30.smi

# 4. Generate Pareto preference pairs
python scripts/generate_pairs.py --model_path checkpoints/base --data_path data/chembl_30.smi

# 5. DPO fine-tuning
python scripts/train_dpo.py --base_model_path checkpoints/base --pairs_path data/pareto_pairs.pkl

# 6. Evaluate
python scripts/evaluate.py --model_path checkpoints/dpo --data_path data/chembl_30.smi
```


