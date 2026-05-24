import argparse
import os

from pareto_dpo.config import ParetoDPOConfig
from pareto_dpo.data.dataset import ScaffoldDataset, read_smiles_file
from pareto_dpo.model.gpt import ScaffoldGPT
from pareto_dpo.data.tokenizer import load_or_create_tokenizer, RegexPreTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="checkpoints/base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_mols", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch number to start from (for resume)")
    args = parser.parse_args()

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    config = ParetoDPOConfig()
    config.model_name = args.model_name

    os.makedirs(args.output_dir, exist_ok=True)

    smiles_list = read_smiles_file(args.data_path, max_mols=args.max_mols, shuffle=True)
    print(f"Loaded {len(smiles_list)} molecules")

    tokenizer = load_or_create_tokenizer(
        smiles_list=smiles_list, tokenizer_path="data/tokenizer.json"
    )
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    dataset = ScaffoldDataset(
        smiles_list,
        tokenizer_path="data/tokenizer.json",
        max_length=config.max_length,
        augment_smiles=False,
    )
    print(f"Built {len(dataset)} scaffold-decoration pairs")

    if args.resume:
        from transformers import PreTrainedTokenizerFast
        model = ScaffoldGPT.from_pretrained(args.resume, tokenizer, max_length=config.max_length)
        print(f"Resumed model from {args.resume}")
    else:
        model = ScaffoldGPT.from_pretrained(
            args.model_name, tokenizer, max_length=config.max_length
        )
    model.to(args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    model.train()

    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_batches = len(loader)
    start_epoch = args.start_epoch
    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % 10000 == 0:
                print(f"Epoch {epoch}/{end_epoch}  Step {step}/{num_batches}  loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}/{end_epoch}  avg_loss: {avg_loss:.4f}", flush=True)

        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.backend_tokenizer.pre_tokenizer = None
        tokenizer.save_pretrained(epoch_dir)
        from tokenizers.pre_tokenizers import PreTokenizer as PT
        from pareto_dpo.data.tokenizer import RegexPreTokenizer
        tokenizer.backend_tokenizer.pre_tokenizer = PT.custom(RegexPreTokenizer())
        print(f"Checkpoint saved to {epoch_dir}", flush=True)

    model.save_pretrained(args.output_dir)
    tokenizer.backend_tokenizer.pre_tokenizer = None
    tokenizer.save_pretrained(args.output_dir)
    from tokenizers.pre_tokenizers import PreTokenizer as PT
    from pareto_dpo.data.tokenizer import RegexPreTokenizer
    tokenizer.backend_tokenizer.pre_tokenizer = PT.custom(RegexPreTokenizer())
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
