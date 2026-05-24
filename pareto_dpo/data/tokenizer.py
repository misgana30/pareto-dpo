import re
from typing import List, Optional

from transformers import PreTrainedTokenizerFast


SMILES_REGEX = re.compile(
    r"(\[[^\]]+]"
    r"|Br?"
    r"|Cl?"
    r"|N|O|S|P|F|I"
    r"|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def train_tokenizer(
    smiles_list: List[str],
    save_path: str,
    vocab_size: int = 512,
    special_tokens: Optional[List[str]] = None,
):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import PreTokenizer

    if special_tokens is None:
        special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "<scaffold>", "<decorate>"]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = PreTokenizer.custom(RegexPreTokenizer())

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=list("bcnops()=#-+\\/:~@?*>$%.0123456789"),
    )

    def batch_iterator():
        for i in range(0, len(smiles_list), 1000):
            yield smiles_list[i : i + 1000]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.pre_tokenizer = None
    tokenizer.save(save_path)
    return tokenizer


class RegexPreTokenizer:
    def pre_tokenize(self, pretok):
        def splitter(_i, normalized):
            content = normalized.original
            offsets = [(m.start(), m.end()) for m in SMILES_REGEX.finditer(content)]
            return [normalized.slice(slice(s, e)) for s, e in offsets]
        pretok.split(splitter)


def load_or_create_tokenizer(
    smiles_list: Optional[List[str]] = None,
    tokenizer_path: str = "data/tokenizer.json",
    vocab_size: int = 512,
):
    import os

    if os.path.exists(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<scaffold>", "<decorate>"]
        })
        from tokenizers.pre_tokenizers import PreTokenizer as PT
        tokenizer.backend_tokenizer.pre_tokenizer = PT.custom(RegexPreTokenizer())
    else:
        assert smiles_list is not None, "smiles_list required to train tokenizer"
        _ = train_tokenizer(smiles_list, tokenizer_path, vocab_size=vocab_size)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["<scaffold>", "<decorate>"]
        })
        from tokenizers.pre_tokenizers import PreTokenizer as PT
        tokenizer.backend_tokenizer.pre_tokenizer = PT.custom(RegexPreTokenizer())
    return tokenizer


def format_scaffold_prompt(scaffold_smiles: str) -> str:
    return f"<bos><scaffold>{scaffold_smiles}<decorate>"


def format_scaffold_decoration(scaffold_smiles: str, decoration_smiles: str) -> str:
    return f"<bos><scaffold>{scaffold_smiles}<decorate>{decoration_smiles}<eos>"
