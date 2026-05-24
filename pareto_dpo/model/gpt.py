import torch
import torch.nn as nn
from transformers import AutoConfig, GPT2LMHeadModel, PreTrainedTokenizerFast


class ScaffoldGPT(GPT2LMHeadModel):
    def __init__(self, config, tokenizer: PreTrainedTokenizerFast):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.scaffold_token_id = tokenizer.convert_tokens_to_ids("<scaffold>")
        self.decorate_token_id = tokenizer.convert_tokens_to_ids("<decorate>")
        self.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def from_pretrained(
        cls, model_name: str, tokenizer: PreTrainedTokenizerFast, **kwargs
    ):
        config = AutoConfig.from_pretrained(model_name)
        config.vocab_size = len(tokenizer)
        max_length = kwargs.pop("max_length", 512)
        if hasattr(config, "max_length"):
            del config.max_length
        model = super().from_pretrained(model_name, config=config, tokenizer=tokenizer, ignore_mismatched_sizes=True, **kwargs)
        model.resize_token_embeddings(len(tokenizer))
        model.scaffold_token_id = tokenizer.convert_tokens_to_ids("<scaffold>")
        model.decorate_token_id = tokenizer.convert_tokens_to_ids("<decorate>")
        model.pad_token_id = tokenizer.pad_token_id
        model.generation_config.max_length = max_length
        return model

    def generate_from_scaffold(
        self,
        scaffold_smiles: str,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> list:
        prompt = f"<bos><scaffold>{scaffold_smiles}<decorate>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        results = []
        for output in outputs:
            generated = self.tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            results.append(generated.strip())
        return results

    def get_decoration_logps(self, input_ids, attention_mask, labels):
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

    @property
    def device(self):
        return next(self.parameters()).device
