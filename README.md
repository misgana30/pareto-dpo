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

## Results (best model: dpo_v5)

### Multi-objective improvement

| Metric | Base (GPT-2) | DPO | Δ |
|:---|---:|---:|---:|
| QED ↑ (median) | 0.28 | **0.70** | **+0.41** |
| clogP ↑ (median) | 3.47 | **3.80** | **+0.34** |
| SA ↓ (median) | 3.51 | **2.31** | **−1.21** |
| MW ↓ (median) | 545 | **259** | **−286** |
| Validity | 99.9% | 90.9% | −9.0% |

The DPO-finetuned model generates molecules that are substantially more drug-like (higher QED, lower MW, more synthesizable).

### Figures

![Radar chart](figures/radar.png)
*Multi-objective performance radar — DPO (orange) dominates Base (blue) on all four objectives.*

![Pareto front](figures/pareto_front.png)
*2D Pareto front projections. DPO molecules (orange) concentrate in the high-QED / high-clogP / low-SA / low-MW region.*

![Distributions](figures/distributions.png)
*Distribution shift per objective — DPO pushes the entire distribution toward better values.*

![Ablation](figures/ablation_sweep.png)
*Model interpolation between Base and DPO. Even small α (5–15%) give measurable improvement.*

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
│   ├── generate_pairs_fast.py # Optimized pair generation
│   ├── train_dpo.py           # DPO fine-tuning
│   ├── evaluate.py            # Evaluation
│   ├── evaluate_ablation.py   # Model interpolation study
│   └── final_evaluation.py    # Comprehensive eval + figures
├── figures/                   # Generated evaluation figures
├── data/                      # Tokenizer, pairs, eval results
├── checkpoints/               # Base and DPO model checkpoints
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

# 7. Final evaluation with figures
python scripts/final_evaluation.py --dpo_model_path checkpoints/dpo_v5
```

## Citation

```bibtex
@software{pareto_dpo,
  author = {Misgana},
  title = {Pareto-DPO: Multi-Objective Preference Alignment for Scaffold Decoration},
  url = {https://github.com/misgana30/pareto-dpo},
  year = {2026},
}
```
