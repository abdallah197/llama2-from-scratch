from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # nheads for query
    n_kv_heads: Optional[int] = None  # nheads for key and V
    vocab_size: int = -1  # will be set when loading the tokenizer/
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    mode: str = 'train'
    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_length: int = 32

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class TrainArgs(ModelArgs):
    n_epochs: int = 10
    log_interval: int = 12
    eval_iters: int = 200
    lr: float = 3e-4
    warmup_steps: int = 4000


@dataclass
class DataArgs(ModelArgs):
    filepath: str = 'input.txt'
    tokenizer_model_path: str = 'tokenizer.model'
    train_size: float = 0.9


@dataclass
class InferenceArgs(ModelArgs):
    checkpoint_dir: str = 'llama-2-7b'
    tokenizer_path: str = 'tokenizer.model'
    load_model: bool = True
    max_seq_len: int = 10
    temperature: float = 0.6
    top_p: float = 0.9
