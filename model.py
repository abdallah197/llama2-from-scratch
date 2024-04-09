import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelArgs


def precompute_theta_pos_frequencies(head_dim: int, seq_length: int, theta: float = 10000):
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
    theta = 1.0 / (theta ** (theta_numerator / head_dim))

    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_length)

    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    m_theta = torch.outer(m, theta).float()

    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    m_theta_complex = torch.polar(torch.ones_like(m_theta), m_theta)
    return m_theta_complex


def apply_rotary_embeddings(x: torch.Tensor, m_theta_complex: torch.Tensor):
    """
    Applies four transformation to input embedding and applies rotary embeddings to it.
    Args:
        x: input embeddings B, seq_length, H, Head_dim
        m_theta_complex: Seq_length, Head_dim/2
        device: device str

    Returns: B, seq_length, H, Head_dim

    """
    # step 1, 2 reshaping the input matrix to be in a form of [[x1 x1] [x3 x4]]
    # Separate the last dimension pairs of two values, representing the
    # real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # so that we can apply complex transformation to it -> [x1 + ix2  x3+ ix4]
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape the m_theta complex tensor to match the shape of the x_complex tensor.
    # So we need to add the batch dimension and the num of heads dimension
    # (Seq_Len, Head_Dim/2) -> (1, Seq_Len, 1, Head_Dim/2)
    m_theta_complex = m_theta_complex.unsqueeze(0).unsqueeze(2)

    # step3
    # Multiply each complex number in the x_complex tensor by the corresponding
    # complex number in the mtheta_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * m_theta_complex

    # step4 convert to real values, and flatten
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the key/value tensor along the head dimension a specified number of times.

    This function is useful in self-attention mechanisms where the number of heads for keys/values
    differs from that of queries, requiring an adjustment to match the dimensions for computation.

    Args:
        x (torch.Tensor): The input tensor representing keys or values with shape
                          (batch_size, seq_length, n_kv_heads, head_dim).
        n_rep (int): The number of times each key/value tensor should be repeated along the head dimension.

    Returns:
        torch.Tensor: The expanded tensor with repeated keys/values along the head dimension,
                      resulting in a new shape (batch_size, seq_length, n_kv_heads * n_rep, head_dim).
    """
    # Unpack the shape of the input tensor for clarity
    batch_size, seq_length, n_kv_heads, head_dim = x.shape

    # If no repetition is needed, return the original tensor
    if n_rep == 1:
        return x

    # Expand the tensor along a new dimension to repeat it, followed by a reshape to merge the repeated dimension
    # with the head dimension, effectively increasing the number of heads by n_rep times.
    # Step 1: Introduce a new dimension for repetition with shape (batch_size, seq_length, n_kv_heads, 1, head_dim)
    # Step 2: Expand the tensor along the new dimension to repeat each slice n_rep times, resulting in shape
    #         (batch_size, seq_length, n_kv_heads, n_rep, head_dim)
    # Step 3: Reshape the expanded tensor to merge the new dimension with the head dimension, yielding a new shape
    #         (batch_size, seq_length, n_kv_heads * n_rep, head_dim), effectively increasing the head count.
    return (
        x.unsqueeze(-2)  # Introduce a new dimension for repetition at the second last position
        .expand(batch_size, seq_length, n_kv_heads, n_rep, head_dim)  # Expand along the new dimension
        .reshape(batch_size, seq_length, n_kv_heads * n_rep, head_dim)
        # Merge the new dimension with the head dimension
    )


class RMSNorm(nn.Module):
    """applies RMSNorm paper"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, seq_len, dim) / d, seq, dim -> B, seq_len, dim
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x: torch.Tensor):
        # (dim) * B, Seq_len, Dim - >  B, Seq_len, Dim
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    """Implements a self-attention mechanism with an optional difference in the number of heads for keys/values and
     queries."""

    def __init__(self, args: ModelArgs):
        """Initializes the SelfAttention module with configurable head dimensions and counts."""
        super().__init__()
        # Choose the number of heads for keys and values, defaulting to the overall number of heads if not specified
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # The number of heads for queries is set to the overall number of heads
        self.n_heads_q = args.n_heads
        # Calculate how many times keys and values should be repeated to match the number of query heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Determine the dimension of each head based on the model dimension and the number of heads
        self.head_dim = args.dim // args.n_heads
        # specify the mode inference/run
        self.mode = args.mode
        assert self.mode in ['inference', 'train']
        # Linear transformations for queries, keys, and values
        self.wq = nn.Linear(args.dim, self.head_dim * self.n_heads_q, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        # Linear transformation for the output
        self.wo = nn.Linear(self.head_dim * self.n_heads_q, args.dim, bias=False)

        # Initialize caches for keys and values to zeros
        self.cache_k = torch.zeros(args.batch_size, args.max_seq_length, self.n_kv_heads, self.head_dim)
        self.cache_v = torch.zeros(args.batch_size, args.max_seq_length, self.n_kv_heads, self.head_dim)

        self.device = args.device

    def forward(self, x: torch.Tensor, start_pos: int, m_theta_complex: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SelfAttention module, computes attention scores and applies them to the input."""
        batch_size, seq_length, _ = x.shape  # Input shape: (Batch Size, Sequence Length, Head Dimension)
        # (B, 1, Dim) ->  # (B, 1, HQ * Head dim)
        xq = self.wq(x)
        # (B, 1, Dim) ->  # (B, 1, H_KV * Head dim)
        xk = self.wk(x)
        # (B, 1, Dim) ->  # (B, 1, H_KV * Head dim)
        xv = self.wv(x)

        # (batch_size, 1, H_Q * Head Dim) --> (batch_size, 1, H_Q,  Head Dim)
        xq = xq.view(batch_size, seq_length, self.n_heads_q, self.head_dim)

        # (batch_size, 1, H_KV * Head Dim) --> (batch_size, 1, H_KV,  Head Dim)
        xk = xk.view(batch_size, seq_length, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_length, self.n_kv_heads, self.head_dim)

        # Apply rotary position embeddings to queries and keys (shapes remain unchanged)
        xq = apply_rotary_embeddings(xq, m_theta_complex)
        xk = apply_rotary_embeddings(xk, m_theta_complex)

        # During inference, we apply KV cache.
        if self.mode == 'inference':
            # replace the entry in the cache for this token.
            # we update the KV cache by appending the current attention calculations
            self.cache_k[:batch_size, start_pos:start_pos + seq_length] = xk
            self.cache_v[:batch_size, start_pos:start_pos + seq_length] = xv

            # Retrieve all the cached values and keys up to this token
            # (batch_size, Seq_len_KV, H_KV, Head_Dim)
            keys = self.cache_k[:batch_size, 0:start_pos + seq_length]
            values = self.cache_v[:batch_size, 0:start_pos + seq_length]
        else:
            keys, values = xk, xv

        # Repeat the heads of K, V to reach the number of heads of Q.
        # This is a shortcut and not the optimized solution to implement Grouped Query attention
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose for attention computation to move the head dimension before the sequence length dimension
        # New shapes: (Batch Size, Num Heads for Queries, 1 or Sequence Length, Head Dimension)
        # (B, 1, H, Head_dim) -> (B, H, 1, Head_Dim)
        xq, keys, values = map(lambda t: t.transpose(1, 2), (xq, keys, values))

        # we apply the equation q.kt/sqrt(d)
        # (B, H_Q, 1, Head_dim) @ (B, H_Q, Head_dim, Seq_len_KV) --> (B, HQ,1, Seq_len_KV)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, HQ,1, Seq_len_KV) @ (B, HQ, seq_len_KV, Head_Dim) -> (B, HQ, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, HQ, 1, Head_Dim) -->  (B, 1, H_Q, Head_Dim) --> B, 1, Head_dim
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.wo(output)  # apply the final WO linear transformation


class FeedForward(nn.Module):
    """
    Implements a feed-forward neural network layer as part of a Transformer model architecture.

    This feed-forward network applies a sequence of linear transformations and non-linear activations
    to the input tensor. It optionally allows for the adjustment of the hidden dimension size through
    a multiplier and ensures that the hidden dimension size is a multiple of a specified value.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the FeedForward layer with configurable hidden dimensions.

        Args:
            args (ModelArgs): Configuration arguments including the input dimension (`dim`), an optional
                              feed-forward dimension multiplier (`ffn_dim_multiplier`), and a factor
                              (`multiple_of`) to which the hidden dimension size should be rounded up.
        """
        super().__init__()

        # Calculate the initial hidden dimension size as four times the input dimension size
        hidden_dim = args.dim * 4

        # Optionally reduce the hidden dimension size to two-thirds of its initial size
        hidden_dim = int(2 * hidden_dim / 3)

        # If a feed-forward dimension multiplier is provided, apply it to the hidden dimension size
        if args.ffn_dim_multiplier:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Round the hidden dimension size up to the nearest multiple of a specified factor
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # Define the first linear transformation from input dimension to hidden dimension
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)

        # Define the second linear transformation from hidden dimension back to input dimension
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)

        # Define an additional linear transformation as an alternative path within the layer
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward layer, applying SWiGLU activation function.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, seq_length, input_dimension).

        Returns:
            torch.Tensor: The output tensor after applying the feed-forward transformations,
                           with the same shape as the input tensor (batch_size, seq_length, input_dimension).
        """
        # Apply the first linear transformation followed by the SiLU (Swish) activation function
        # Resulting shape: (batch_size, seq_length, hidden_dimension)
        swish = F.silu(self.w1(x))

        # Apply the additional linear transformation to the input tensor
        # Resulting shape: (batch_size, seq_length, hidden_dimension)
        x_v = self.w3(x)

        # Element-wise multiplication of the SiLU-activated tensor and the additional linearly transformed tensor
        out = swish * x_v

        # Apply the second linear transformation to map back to the input dimension size
        # Resulting shape: (batch_size, seq_length, input_dimension)
        out = self.w2(out)

        return out


class EncoderBlock(nn.Module):
    """
    Implements an Encoder Block for a transformer model following the Llama2 architecture.

    This block consists of a self-attention mechanism followed by a position-wise feed-forward network.
    Each of these components is preceded by layer normalization. Residual connections are also employed
    around both the self-attention and feed-forward networks.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimensionality of the input features.
        head_dim (int): Dimensionality of each attention head.
        attention (SelfAttention): The self-attention mechanism.
        feed_forward (FeedForward): The feed-forward network.
        attention_norm (RMSNorm): Layer normalization before the self-attention.
        ffn_norm (RMSNorm): Layer normalization before the feed-forward network.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the EncoderBlock with the specified parameters.

        Args:
            args (ModelArgs): Configuration arguments for the model. Expected to contain
                              'n_heads', 'dim', 'norm_eps' for initializing various components.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, m_theta_complex: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the EncoderBlock.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, dim).
            start_pos (int): Starting position for processing the input.
            m_theta_complex (torch.Tensor): Additional tensor required for the self-attention mechanism.

        Returns:
            torch.Tensor: The output tensor after processing through the encoder block.
        """
        h = x + self.attention(self.attention_norm(x), start_pos, m_theta_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


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
            self.layers.append(EncoderBlock(self.args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # we multiply seq_len *2 because the prompt might be long
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_length * 2,
                                                              ).to(self.tok_embeddings.weight.device)

    def forward(self, tokens: torch.Tensor, start_pos: int, targets=None):
        # (B, seq_length)
        batch_size, seq_len = tokens.shape

        # (B, seq_length) -> (B, seq_length, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_length]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        if targets is None:
            loss = None
        else:
            B, T, C = output.shape
            output_ = output.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(output_, targets)

        return output, loss
