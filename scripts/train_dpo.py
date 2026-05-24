import argparse
import pickle
import os

from pareto_dpo.config import ParetoDPOConfig
from pareto_dpo.model.gpt import ScaffoldGPT
from pareto_dpo.data.tokenizer import load_or_create_tokenizer
from pareto_dpo.optimization.dpo_trainer import DPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    config = ParetoDPOConfig()
    config.beta = args.beta
    config.learning_rate = args.lr
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.use_wandb = args.use_wandb
    config.checkpoint_dir = args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_or_create_tokenizer(tokenizer_path="data/tokenizer.json")

    model = ScaffoldGPT.from_pretrained(args.base_model_path, tokenizer)
    model.to("cuda")
    ref_model = ScaffoldGPT.from_pretrained(args.base_model_path, tokenizer)
    ref_model.to("cuda")
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    with open(args.pairs_path, "rb") as f:
        pairs = pickle.load(f)
    print(f"Loaded {len(pairs)} preference pairs")

    if args.use_wandb:
        import wandb
        wandb.init(project="pareto-dpo", name=config.run_name, config=vars(config))

    trainer = DPOTrainer(model, ref_model, tokenizer, config)
    trainer.train(pairs)

    model.save_pretrained(args.output_dir)
    # Copy serialized tokenizer file (cannot use save_pretrained due to custom PreTokenizer)
    import shutil
    shutil.copy("data/tokenizer.json", os.path.join(args.output_dir, "tokenizer.json"))
    print(f"DPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
