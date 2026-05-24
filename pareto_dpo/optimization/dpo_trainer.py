import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from ..data.tokenizer import format_scaffold_decoration


class PreferenceDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.data = []
        for scaffold, chosen_smi, rejected_smi, _, _ in pairs:
            chosen_text = format_scaffold_decoration(scaffold, chosen_smi)
            rejected_text = format_scaffold_decoration(scaffold, rejected_smi)
            self.data.append((chosen_text, rejected_text))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chosen_text, rejected_text = self.data[idx]
        chosen = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
        }


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
):
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    ref_log_ratio = ref_chosen_logps - ref_rejected_logps
    logits = policy_log_ratio - ref_log_ratio
    logits = torch.clamp(logits, min=-50, max=50)
    loss = -F.logsigmoid(beta * logits).mean()
    return loss


def _get_batch_logps(logits, input_ids, attention_mask, label_ignore_id=-100):
    labels = input_ids.clone()
    labels[labels == 0] = label_ignore_id
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()
    per_token_logps = log_softmax_gather(shift_logits, shift_labels)
    return (per_token_logps * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)


def log_softmax_gather(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


class DPOTrainer:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate
        )

    def train(self, pairs, val_pairs=None):
        dataset = PreferenceDataset(pairs, self.tokenizer, self.config.max_length)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.model.train()
        self.ref_model.eval()
        global_step = 0

        for epoch in range(self.config.epochs):
            pbar = tqdm(loader, desc=f"DPO Epoch {epoch + 1}")
            for batch in pbar:
                loss = self._train_step(batch)
                pbar.set_postfix(loss=loss.item())

                if global_step % self.config.log_every == 0:
                    if self.config.use_wandb:
                        import wandb
                        wandb.log({"dpo_loss": loss.item()}, step=global_step)

                if global_step > 0 and global_step % self.config.save_every == 0:
                    self._save_checkpoint(global_step)

                global_step += 1

            self._save_checkpoint(f"epoch-{epoch + 1}")

    def _train_step(self, batch):
        chosen_ids = batch["chosen_input_ids"].to(self.model.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.model.device)
        rejected_ids = batch["rejected_input_ids"].to(self.model.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.model.device)

        with torch.no_grad():
            ref_chosen_out = self.ref_model(
                input_ids=chosen_ids, attention_mask=chosen_mask
            )
            ref_rejected_out = self.ref_model(
                input_ids=rejected_ids, attention_mask=rejected_mask
            )
            ref_chosen_logps = _get_batch_logps(
                ref_chosen_out.logits, chosen_ids, chosen_mask
            )
            ref_rejected_logps = _get_batch_logps(
                ref_rejected_out.logits, rejected_ids, rejected_mask
            )

        policy_chosen_out = self.model(
            input_ids=chosen_ids, attention_mask=chosen_mask
        )
        policy_rejected_out = self.model(
            input_ids=rejected_ids, attention_mask=rejected_mask
        )
        policy_chosen_logps = _get_batch_logps(
            policy_chosen_out.logits, chosen_ids, chosen_mask
        )
        policy_rejected_logps = _get_batch_logps(
            policy_rejected_out.logits, rejected_ids, rejected_mask
        )

        loss = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.config.beta,
        )

        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite loss={loss.item()}, skipping step")
            self.optimizer.zero_grad()
            return torch.tensor(0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()

    def _save_checkpoint(self, tag):
        path = f"{self.config.checkpoint_dir}/{self.config.run_name}-{tag}.pt"
        torch.save(self.model.state_dict(), path)
