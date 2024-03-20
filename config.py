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

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_length: int = 32

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class TrainConfig(ModelArgs):
    n_epocs: int = 10
    eval_interval: int = 10
    eval_iters: int = 200
    lr: float = 3e-4
