from dataclasses import dataclass, field
from typing import List


@dataclass
class ParetoDPOConfig:
    # data
    data_path: str = "data/chembl_30.smi"
    scaffold_cutoff: str = "bemis-murcko"
    max_smiles_len: int = 256
    train_split: float = 0.9

    # model
    model_name: str = "gpt2"
    vocab_size: int = None
    max_length: int = 512
    dropout: float = 0.1

    # scoring / objectives
    objectives: List[str] = field(default_factory=lambda: ["qed", "clogp", "sa", "mw"])
    objective_directions: List[str] = field(
        default_factory=lambda: ["max", "max", "min", "min"]
    )
    sa_model_path: str = None

    # Pareto pair construction
    num_samples_per_scaffold: int = 64
    pareto_batch_size: int = 512
    max_pairs_per_scaffold: int = 256
    temperature: float = 1.0

    # DPO training
    beta: float = 0.1
    learning_rate: float = 5e-6
    batch_size: int = 8
    grad_accum_steps: int = 4
    epochs: int = 3
    max_grad_norm: float = 1.0

    # generation
    gen_temperature: float = 1.0
    gen_top_k: int = 50
    gen_top_p: float = 0.95
    gen_max_length: int = 256

    # logging / saving
    run_name: str = "pareto-dpo-v1"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10
    save_every: int = 1000
    use_wandb: bool = False

    # evaluation
    eval_num_scaffolds: int = 200
    eval_samples_per_scaffold: int = 128
