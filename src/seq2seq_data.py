"""
Sequence-to-sequence data loading utilities.

Provides data loading for translation and other seq2seq tasks.
spacy and torchtext are optional dependencies.

Features:
- Tokenizer loading (spacy-based)
- Vocabulary building
- Batching with padding
- DataLoader creation for distributed training
"""

import os
from os.path import exists
from typing import Optional, Tuple, Callable, Iterator, List

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader

# Optional imports with graceful fallback
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from torchtext.data.functional import to_map_style_dataset
    from torchtext.vocab import build_vocab_from_iterator
    import torchtext.datasets as datasets
    TORCHTEXT_AVAILABLE = True
except ImportError:
    TORCHTEXT_AVAILABLE = False

try:
    from torch.utils.data.distributed import DistributedSampler
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


def check_dependencies(require_spacy=False, require_torchtext=False):
    """Check if required dependencies are available."""
    if require_spacy and not SPACY_AVAILABLE:
        raise ImportError(
            "spacy is required for tokenization. "
            "Install with: pip install spacy && python -m spacy download de_core_news_sm en_core_web_sm"
        )
    if require_torchtext and not TORCHTEXT_AVAILABLE:
        raise ImportError(
            "torchtext is required for dataset loading. "
            "Install with: pip install torchtext"
        )


def load_tokenizers(src_lang="de", tgt_lang="en"):
    """
    Load spacy tokenizer models for source and target languages.

    Downloads models if they haven't been downloaded already.

    Args:
        src_lang: Source language code (default: "de" for German)
        tgt_lang: Target language code (default: "en" for English)

    Returns:
        tuple: (src_tokenizer, tgt_tokenizer)

    Raises:
        ImportError: If spacy is not available
    """
    check_dependencies(require_spacy=True)

    # Language to spacy model mapping
    lang_to_model = {
        "de": "de_core_news_sm",
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        "it": "it_core_news_sm",
        "pt": "pt_core_news_sm",
        "nl": "nl_core_news_sm",
        "zh": "zh_core_web_sm",
        "ja": "ja_core_news_sm",
    }

    def load_model(lang):
        model_name = lang_to_model.get(lang)
        if model_name is None:
            raise ValueError(f"Unsupported language: {lang}")

        try:
            return spacy.load(model_name)
        except IOError:
            os.system(f"python -m spacy download {model_name}")
            return spacy.load(model_name)

    spacy_src = load_model(src_lang)
    spacy_tgt = load_model(tgt_lang)

    return spacy_src, spacy_tgt


def tokenize(text: str, tokenizer) -> List[str]:
    """
    Tokenize text using a spacy tokenizer.

    Args:
        text: Input text string
        tokenizer: spacy language model

    Returns:
        List of tokens
    """
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter: Iterator, tokenizer: Callable, index: int) -> Iterator[List[str]]:
    """
    Yield tokens from a data iterator.

    Args:
        data_iter: Iterator yielding (src, tgt) tuples
        tokenizer: Tokenization function
        index: 0 for source, 1 for target

    Yields:
        List of tokens for each example
    """
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(
    spacy_src,
    spacy_tgt,
    language_pair=("de", "en"),
    min_freq=2,
    specials=None,
):
    """
    Build vocabularies from the Multi30k dataset.

    Args:
        spacy_src: Source language spacy model
        spacy_tgt: Target language spacy model
        language_pair: Language pair tuple (default: ("de", "en"))
        min_freq: Minimum frequency for vocabulary inclusion
        specials: Special tokens (default: ["<s>", "</s>", "<blank>", "<unk>"])

    Returns:
        tuple: (vocab_src, vocab_tgt)

    Raises:
        ImportError: If torchtext is not available
    """
    check_dependencies(require_torchtext=True)

    if specials is None:
        specials = ["<s>", "</s>", "<blank>", "<unk>"]

    def tokenize_src(text):
        return tokenize(text, spacy_src)

    def tokenize_tgt(text):
        return tokenize(text, spacy_tgt)

    print("Building source vocabulary...")
    train, val, test = datasets.Multi30k(language_pair=language_pair)
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_src, index=0),
        min_freq=min_freq,
        specials=specials,
    )

    print("Building target vocabulary...")
    train, val, test = datasets.Multi30k(language_pair=language_pair)
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_tgt, index=1),
        min_freq=min_freq,
        specials=specials,
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(
    spacy_src,
    spacy_tgt,
    vocab_path="vocab.pt",
    language_pair=("de", "en"),
):
    """
    Load vocabulary from file or build if not exists.

    Args:
        spacy_src: Source language spacy model
        spacy_tgt: Target language spacy model
        vocab_path: Path to save/load vocabulary
        language_pair: Language pair tuple

    Returns:
        tuple: (vocab_src, vocab_tgt)
    """
    if not exists(vocab_path):
        vocab_src, vocab_tgt = build_vocabulary(
            spacy_src, spacy_tgt, language_pair=language_pair
        )
        torch.save((vocab_src, vocab_tgt), vocab_path)
    else:
        vocab_src, vocab_tgt = torch.load(vocab_path)

    print("Vocabulary sizes:")
    print(f"  Source: {len(vocab_src)}")
    print(f"  Target: {len(vocab_tgt)}")

    return vocab_src, vocab_tgt


def collate_batch(
    batch,
    src_pipeline: Callable,
    tgt_pipeline: Callable,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    """
    Collate function for batching translation data.

    Args:
        batch: List of (src_text, tgt_text) tuples
        src_pipeline: Source tokenization function
        tgt_pipeline: Target tokenization function
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to place tensors on
        max_padding: Maximum sequence length
        pad_id: Padding token ID (default: 2 = <blank>)

    Returns:
        tuple: (src_tensor, tgt_tensor)
    """
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id

    src_list, tgt_list = [], []

    for (_src, _tgt) in batch:
        # Process source
        processed_src = torch.cat([
            bs_id,
            torch.tensor(
                src_vocab(src_pipeline(_src)),
                dtype=torch.int64,
                device=device,
            ),
            eos_id,
        ], dim=0)

        # Process target
        processed_tgt = torch.cat([
            bs_id,
            torch.tensor(
                tgt_vocab(tgt_pipeline(_tgt)),
                dtype=torch.int64,
                device=device,
            ),
            eos_id,
        ], dim=0)

        # Pad to max_padding
        src_list.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    return (src, tgt)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_src,
    spacy_tgt,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
    language_pair=("de", "en"),
):
    """
    Create train and validation dataloaders for translation.

    Args:
        device: Device to place tensors on
        vocab_src: Source vocabulary
        vocab_tgt: Target vocabulary
        spacy_src: Source spacy tokenizer
        spacy_tgt: Target spacy tokenizer
        batch_size: Batch size (in tokens for bucketing)
        max_padding: Maximum sequence length
        is_distributed: Whether to use distributed sampling
        language_pair: Language pair tuple

    Returns:
        tuple: (train_dataloader, valid_dataloader)
    """
    check_dependencies(require_torchtext=True)

    def tokenize_src(text):
        return tokenize(text, spacy_src)

    def tokenize_tgt(text):
        return tokenize(text, spacy_tgt)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_src,
            tokenize_tgt,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=language_pair
    )

    train_iter_map = to_map_style_dataset(train_iter)
    valid_iter_map = to_map_style_dataset(valid_iter)

    if is_distributed and DISTRIBUTED_AVAILABLE:
        train_sampler = DistributedSampler(train_iter_map)
        valid_sampler = DistributedSampler(valid_iter_map)
    else:
        train_sampler = None
        valid_sampler = None

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )

    return train_dataloader, valid_dataloader


# =============================================================================
# Synthetic data generation (for testing without dependencies)
# =============================================================================

def data_gen(V, batch_size, nbatches, device=None):
    """
    Generate random data for a src-tgt copy task.

    Useful for testing and debugging the model without any dependencies.

    Args:
        V: Vocabulary size
        batch_size: Batch size
        nbatches: Number of batches to generate
        device: Device to place tensors on

    Yields:
        Batch objects for training
    """
    from src.shared.training import Batch

    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1  # Start token
        src = data.clone().detach()
        tgt = data.clone().detach()

        if device is not None:
            src = src.to(device)
            tgt = tgt.to(device)

        yield Batch(src, tgt, pad=0)


def create_synthetic_dataloaders(
    vocab_size=11,
    batch_size=30,
    train_batches=20,
    valid_batches=5,
    device=None,
):
    """
    Create synthetic dataloaders for testing.

    Args:
        vocab_size: Size of the vocabulary
        batch_size: Batch size
        train_batches: Number of training batches
        valid_batches: Number of validation batches
        device: Device to place tensors on

    Returns:
        tuple: (train_dataloader, valid_dataloader)
    """
    def train_gen():
        return data_gen(vocab_size, batch_size, train_batches, device)

    def valid_gen():
        return data_gen(vocab_size, batch_size, valid_batches, device)

    return train_gen, valid_gen


# =============================================================================
# Simple text file loading (no dependencies required)
# =============================================================================

class SimpleVocab:
    """Simple vocabulary class for basic usage without torchtext."""

    def __init__(self, specials=None):
        self.stoi = {}  # string to index
        self.itos = []  # index to string
        self.specials = specials or ["<pad>", "<unk>", "<s>", "</s>"]

        for special in self.specials:
            self._add_word(special)

        self.default_index = self.stoi.get("<unk>", 0)

    def _add_word(self, word):
        if word not in self.stoi:
            self.stoi[word] = len(self.itos)
            self.itos.append(word)

    def build_from_texts(self, texts, min_freq=1):
        """Build vocabulary from a list of texts."""
        from collections import Counter

        counter = Counter()
        for text in texts:
            tokens = text.strip().split()
            counter.update(tokens)

        for word, count in counter.items():
            if count >= min_freq:
                self._add_word(word)

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens):
        """Convert tokens to indices."""
        return [self.stoi.get(tok, self.default_index) for tok in tokens]

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos

    def set_default_index(self, idx):
        self.default_index = idx


def load_text_file_data(
    src_file: str,
    tgt_file: str,
    vocab_src: Optional[SimpleVocab] = None,
    vocab_tgt: Optional[SimpleVocab] = None,
    min_freq: int = 1,
):
    """
    Load parallel text data from files.

    Simple loading without torchtext/spacy dependencies.

    Args:
        src_file: Path to source language file
        tgt_file: Path to target language file
        vocab_src: Optional pre-built source vocabulary
        vocab_tgt: Optional pre-built target vocabulary
        min_freq: Minimum frequency for vocabulary building

    Returns:
        tuple: (data_pairs, vocab_src, vocab_tgt)
    """
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()

    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()

    assert len(src_lines) == len(tgt_lines), "Source and target files must have same number of lines"

    data_pairs = list(zip(src_lines, tgt_lines))

    # Build vocabularies if not provided
    if vocab_src is None:
        vocab_src = SimpleVocab()
        vocab_src.build_from_texts(src_lines, min_freq=min_freq)

    if vocab_tgt is None:
        vocab_tgt = SimpleVocab()
        vocab_tgt.build_from_texts(tgt_lines, min_freq=min_freq)

    return data_pairs, vocab_src, vocab_tgt
