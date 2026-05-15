import argparse
import os

from pareto_dpo.config import ParetoDPOConfig
from pareto_dpo.data.dataset import ScaffoldDataset, read_smiles_file
from pareto_dpo.model.gpt import ScaffoldGPT
from pareto_dpo.data.tokenizer import load_or_create_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="checkpoints/base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_mols", type=int, default=None)
    args = parser.parse_args()

    config = ParetoDPOConfig()
    config.model_name = args.model_name

    os.makedirs(args.output_dir, exist_ok=True)

    smiles_list = read_smiles_file(args.data_path, max_mols=args.max_mols)
    print(f"Loaded {len(smiles_list)} molecules")

    tokenizer = load_or_create_tokenizer(
        smiles_list=smiles_list, tokenizer_path="data/tokenizer.json"
    )
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    dataset = ScaffoldDataset(
        smiles_list,
        tokenizer_path="data/tokenizer.json",
        max_length=config.max_length,
    )
    print(f"Built {len(dataset)} scaffold-decoration pairs")

    model = ScaffoldGPT.from_pretrained(
        args.model_name, tokenizer, max_length=config.max_length
    )
    model.train()

    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss: {avg_loss:.4f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
