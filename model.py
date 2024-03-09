from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


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
    max_seq_length: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_length: int, device: str, theta: float = 10000):
    """
    compute complex MTheta matrix
    Args:
        head_dim: head dimension of the attention head
        seq_length: sequence length
        device: device
        theta: theta number in the paper, defaults to 10,000

    Returns: matrix of complex mtheta R * exp(i m*theta), R=1

    """
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0  # "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_length, device=device)

    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    m_theta = torch.outer(m, theta).float()

    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    m_theta_complex = torch.polar(torch.ones_like(m_theta), m_theta)
    return m_theta_complex


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1  # we must set vocab size
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # we multiply seq_len *2 because the prompt might be long
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_length * 2,
                                                              device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_length)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1  # only one token at a time can be processed.

        # (B, seq_length) -> (B, seq_length, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_length]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.n_layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        return output
